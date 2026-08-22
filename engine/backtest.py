"""事件驱动型分钟级 A 股回测撮合引擎 BacktestEngine。

信号→成交的时序（严格防未来函数）：
    Bar t 产生的 Signal（基于 t 的 VWAP/close 合成）在 Bar t+1 的 open 价成交，
    即信号与成交之间存在 1 根 Bar 的执行延迟 —— 不会使用信号时点尚未可知的价格。

逐 Bar 推进顺序：
    1) T+1 解冻（roll_to_date）：上一交易日买入的份额转为可卖
    2) 撮合 T+1 顺延减仓目标（每 Bar 按当前价换算；可卖不足保留、跌停暂停）
    3) 撮合上一 Bar 收集的信号（先卖后买释放现金；先到先得，现金不足拒绝）
    4) 收集当前 Bar 产生的新信号，留待下一 Bar 撮合
    5) 收盘 mark-to-market 并记录净值曲线

撮合与风控规则（基于 Target_Weight 差额调仓）：
- 动作 → 目标权重：BUY/ADD → metrics['target_weight']；DECAY_REDUCE →
  metrics['target_weight']（信号层 = simulated × reduce_step_ratio）；
  SELL → 0（清仓）。HOLD 不触发调仓。
- 调仓死区：已持仓且 |Target - Current| < deadzone_th 的微调跳过（避免摩擦）；
  从 0 建仓与强制清仓豁免死区。
- 目标股数 = (目标权重 × 总权益) 按 Bar 开盘价换算，向下取整 100 股整数倍；
  加仓受现金（扣除佣金）约束；减仓/清仓以可卖份额为限（T+1），
  可卖不足时仅卖出最大可卖量，剩余目标权重挂起顺延（每 Bar 再试，跌停暂停）。
- 涨停（Bar high 触及 up_limit）不可买入；跌停（Bar low 触及 down_limit）不可卖出
- 成交价 = open × (1 ± 动态滑点)，成本含佣金/印花税/过户费（见 engine.execution）
- 单股最大仓位上限与总账户杠杆上限（见 engine.risk_control / engine.portfolio）

输出：完整成交日志 TradeLog 与逐 Bar 持仓净值曲线 EquityCurve。
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from data.dataslice import SYMBOL, DataSlice
from engine.execution import ExecutionCost
from engine.portfolio import Account
from engine.risk_control import PositionSizer
from strategy.signals import (
    ACT_ADD, ACT_BUY, ACT_DECAY_REDUCE, ACT_HOLD, ACT_SELL, Signal)

logger = logging.getLogger("engine.backtest")

# 撮合所需的 K 线列
_BAR_COLS = ["open", "high", "low", "close", "amount", "up_limit", "down_limit"]


@dataclass
class TradeLog:
    """完整成交日志：成交单与拒绝单均记录（shares=0 表示被拒，reason 说明原因）。"""

    rows: List[dict] = field(default_factory=list)

    def add(self, **kw) -> None:
        self.rows.append(kw)

    def to_frame(self) -> pd.DataFrame:
        cols = ["ts", "symbol", "side", "price", "shares", "amount",
                "commission", "stamp_duty", "transfer_fee", "slippage_bps",
                "cash_after", "equity_after", "reason"]
        if not self.rows:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(self.rows)
        df["ts"] = pd.to_datetime(df["ts"])
        return df[cols].sort_values(["ts", "symbol"]).reset_index(drop=True)


@dataclass
class EquityCurve:
    """逐 Bar 持仓净值曲线。"""

    rows: List[dict] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        cols = ["ts", "cash", "margin", "position_value",
                "total_equity", "n_positions", "unrealized_pnl"]
        if not self.rows:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(self.rows)
        df["ts"] = pd.to_datetime(df["ts"])
        return df[cols].reset_index(drop=True)


class BacktestEngine:
    """事件驱动型回测引擎：按时间序逐步消费 Signal 并执行撮合。

    :param account:  Account 账户（现金/持仓/T+1 可卖份额）
    :param cost:     ExecutionCost 成本与滑点模型
    :param sizer:    PositionSizer 动态仓位与风控
    :param data:     对齐后的 DataSlice（提供 kline 的 open/high/low/amount/涨跌停价）
    :param signals:  Signal 列表（须按 (timestamp, symbol) 升序，来自信号层状态机）
    :param deadzone_th: 调仓死区（已持仓 |Δ权重| < 该值跳过微调；建仓/清仓豁免）
    """

    def __init__(self, account: Account, cost: ExecutionCost, sizer: PositionSizer,
                 data: DataSlice, signals: Sequence[Signal],
                 deadzone_th: float = 0.05) -> None:
        if not 0.0 <= deadzone_th < 1.0:
            raise ValueError(f"deadzone_th 必须在 [0, 1) 区间，当前: {deadzone_th}")
        self.deadzone_th = deadzone_th
        self.account = account
        self.cost = cost
        self.sizer = sizer
        self.data = data
        self.signals = list(signals)

        # 断言：信号必须按时间升序（先到先得的前提）
        ts_list = [s.timestamp for s in self.signals]
        if any(a > b for a, b in zip(ts_list, ts_list[1:])):
            raise ValueError("signals 必须按 timestamp 升序排列")

        # 预处理：Bar 行情 → {ts: DataFrame(symbol 索引)}；信号 → {ts: [Signal]}
        self._kline_by_ts = self._build_bar_map()
        self._signal_by_ts: Dict[pd.Timestamp, List[Signal]] = defaultdict(list)
        for s in self.signals:
            self._signal_by_ts[s.timestamp].append(s)
        self._axis = self.data.time_axis()

    # ------------------------------------------------------------------
    # 预处理
    # ------------------------------------------------------------------

    def _build_bar_map(self) -> Dict[pd.Timestamp, pd.DataFrame]:
        k = self.data.kline
        need = [c for c in _BAR_COLS if c not in k.columns]
        if need:
            raise ValueError(f"kline 缺少撮合必需列: {need}")
        bar_map: Dict[pd.Timestamp, pd.DataFrame] = {}
        for ts, grp in k.groupby(level=0):
            if SYMBOL not in grp.columns:
                raise ValueError("kline 缺少 symbol 列，无法按标的撮合")
            bar_map[pd.Timestamp(ts)] = grp.set_index(SYMBOL)[_BAR_COLS]
        return bar_map

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """执行回测，返回 (trade_log, equity_curve)。

        - 最后一根 Bar 产生的信号没有后续 Bar 可撮合（next-bar 语义），将被丢弃并告警。
        """
        log = TradeLog()
        curve = EquityCurve()
        pending_signals: List[Signal] = []   # 上一 Bar 收集、本 Bar 撮合
        pending_targets: Dict[str, float] = {}  # T+1 顺延减仓目标（symbol → 目标权重）

        for ts in self._axis:
            date = pd.Timestamp(ts).normalize()
            bar = self._kline_by_ts.get(ts)

            # 1) T+1 解冻：上一交易日买入的份额可卖
            self.account.roll_to_date(date)

            # 2) 撮合 T+1 顺延减仓目标（每 Bar 按当前价换算；可卖不足/跌停保留）
            for sym in list(pending_targets):
                pos = self.account.positions.get(sym)
                if pos is None or pos.shares <= 0:
                    del pending_targets[sym]  # 已无持仓，撤销顺延
                    continue
                if pos.sellable_shares <= 0:
                    continue  # 当日买入仍未解冻（T+1），保留顺延
                if bar is None or sym not in bar.index:
                    continue  # 无报价，保留顺延
                brow = bar.loc[sym]
                if brow["low"] <= brow["down_limit"]:
                    continue  # 跌停暂停，保留顺延
                self._execute_sell_to_target(
                    sym, pending_targets[sym], ts, brow,
                    pending_targets, log, reason="t1_deferred_sell")

            # 3) 撮合上一 Bar 的信号（下一 Bar 开盘价；先到先得）
            for sig in pending_signals:
                self._execute(sig, ts, bar, pending_targets, log)
            pending_signals = []

            # 4) 收集本 Bar 新信号 → 下一 Bar 撮合
            pending_signals = list(self._signal_by_ts.get(ts, []))

            # 5) 收盘 mark-to-market + 净值曲线
            if bar is not None:
                self.account.mark_to_market(
                    {sym: row["close"] for sym, row in bar.iterrows()})
            curve.rows.append({
                "ts": ts,
                "cash": self.account.cash,
                "margin": self.account.margin,
                "position_value": self.account.position_value,
                "total_equity": self.account.total_equity,
                "n_positions": len(self.account.positions),
                "unrealized_pnl": self.account.unrealized_pnl(),
            })

        # 尾部校验
        assert self.account.cash >= -1e-6, "回测结束现金为负"
        if pending_signals:
            logger.warning("%d 个信号在最后一根 Bar 产生，无后续 Bar 可成交，已丢弃",
                           len(pending_signals))
        return log.to_frame(), curve.to_frame()

    # ------------------------------------------------------------------
    # 信号撮合（基于 Target_Weight 差额调仓）
    # ------------------------------------------------------------------

    def _execute(self, sig: Signal, ts: pd.Timestamp, bar: Optional[pd.DataFrame],
                 pending_targets: Dict[str, float], log: TradeLog) -> None:
        """撮合单个信号（成交于 Bar ts 的开盘价）。

        HOLD 不触发调仓；BUY/ADD/DECAY_REDUCE/SELL 映射为目标权重后差额调仓。
        """
        sym = sig.symbol
        if bar is None or sym not in bar.index:
            self._log_reject(log, ts, sym, sig.action, "no_quote")
            return
        if sig.action == ACT_HOLD:
            return  # HOLD：目标权重仅作监控，不调仓
        brow = bar.loc[sym]
        pos = self.account.positions.get(sym)
        current_weight = (pos.shares * float(brow["open"]) / self.account.total_equity
                          if pos is not None and pos.shares > 0 else 0.0)
        target = self._target_for(sig, current_weight)
        self._rebalance(sym, target, sig.action, ts, brow, pending_targets, log)

    def _target_for(self, sig: Signal, current_weight: float) -> float:
        """动作 → 目标权重。

        - SELL → 0（清仓）
        - BUY/ADD/DECAY_REDUCE → metrics['target_weight']
          （信号层已按 ES/PS 与 simulated_weight 计算）
        - 缺失降级：BUY/ADD 用 PositionSizer 公式；DECAY_REDUCE 用
          与信号层一致的减仓比例（reduce_step_ratio，默认 0.8）按当前权重
          折减，保证外部信号链路不中断。
        """
        if sig.action == ACT_SELL:
            return 0.0
        metrics = sig.metrics or {}
        tw = metrics.get("target_weight")
        if tw is not None and not pd.isna(tw):
            return float(tw)
        if sig.action in (ACT_BUY, ACT_ADD):
            return self.sizer.target_ratio(metrics)
        if sig.action == ACT_DECAY_REDUCE:
            ratio = float(metrics.get("reduce_step_ratio", 0.8))
            return max(0.0, current_weight * ratio)
        return current_weight

    def _rebalance(self, sym: str, target: float, action: str, ts: pd.Timestamp,
                   brow: pd.Series, pending_targets: Dict[str, float],
                   log: TradeLog) -> None:
        """把该标的目标权重收敛到 target（死区 / 涨跌停 / T+1 约束）。"""
        equity = self.account.total_equity
        pos = self.account.positions.get(sym)
        open_price = float(brow["open"])
        current_weight = (pos.shares * open_price / equity
                          if pos is not None and pos.shares > 0 else 0.0)
        delta = target - current_weight

        # 调仓死区：已持仓的微调（|Δ| < deadzone_th）跳过，避免交易摩擦；
        # 从 0 建仓（current==0）与强制清仓（target==0）豁免。
        if target > 0.0 and current_weight > 0.0 and abs(delta) < self.deadzone_th:
            return

        if delta > 0.0:  # 建仓 / 加仓
            if brow["high"] >= brow["up_limit"]:
                self._log_reject(log, ts, sym, action, "limit_up")
                return
            self._execute_buy_to_target(sym, target, action, ts, brow, log)
        elif delta < 0.0:  # 减仓 / 清仓
            if brow["low"] <= brow["down_limit"]:
                self._log_reject(log, ts, sym, action, "limit_down")
                return
            reason = "signal_sell" if action == ACT_SELL else "decay_reduce"
            self._execute_sell_to_target(sym, target, ts, brow, pending_targets,
                                         log, reason=reason)

    # ------------------------------------------------------------------
    # 买入 / 加仓
    # ------------------------------------------------------------------

    def _execute_buy_to_target(self, sym: str, target: float, action: str,
                               ts: pd.Timestamp, brow: pd.Series,
                               log: TradeLog) -> None:
        """按目标权重加仓：目标市值 = target × equity，补足差额（100 股整数倍）。

        受单股上限 / 总杠杆 / 当前可用现金（扣除佣金过户费）约束。
        """
        open_price = float(brow["open"])
        bar_amount = float(brow["amount"])
        equity = self.account.total_equity
        target_value = target * equity
        pos = self.account.positions.get(sym)
        current_value = pos.shares * open_price if pos is not None else 0.0
        order_value = max(0.0, target_value - current_value)
        if order_value <= 1e-6:
            return  # 已满目标，无需加仓

        # 单股上限：超过则裁剪至上限
        ok, reason = self.sizer.check_single_position(equity, current_value, order_value)
        if not ok:
            order_value = max(0.0, self.sizer.max_single_position * equity - current_value)
            if order_value <= 1e-6:
                self._log_reject(log, ts, sym, action, reason)
                return

        # 总杠杆上限
        ok, reason = self.sizer.check_leverage(
            equity, self.account.position_value, order_value)
        if not ok:
            self._log_reject(log, ts, sym, action, reason)
            return

        # 动态滑点成交价 + 100 股整数倍
        price0 = self.cost.buy_price(open_price, order_value, bar_amount)
        shares = int(order_value / (price0 * 100.0)) * 100
        if shares <= 0:
            self._log_reject(log, ts, sym, action, "small_order")
            return

        amount = shares * price0
        commission, transfer = self.cost.buy_fees(amount)
        total_cost = amount + commission + transfer

        # 现金检查（先到先得：现金不足直接拒绝）
        if total_cost > self.account.cash + 1e-6:
            self._log_reject(log, ts, sym, action, "insufficient_cash")
            return

        self.account.buy(sym, ts, price0, shares, total_cost)
        log.add(ts=ts, symbol=sym, side=action, price=price0, shares=shares,
                amount=amount, commission=commission, stamp_duty=0.0,
                transfer_fee=transfer,
                slippage_bps=self.cost.slippage_bps(order_value, bar_amount),
                cash_after=self.account.cash,
                equity_after=self.account.total_equity, reason="filled")

    # ------------------------------------------------------------------
    # 卖出
    # ------------------------------------------------------------------

    def _execute_sell_to_target(self, sym: str, target: float, ts: pd.Timestamp,
                                brow: pd.Series, pending_targets: Dict[str, float],
                                log: TradeLog, reason: str = "signal_sell") -> None:
        """按目标权重减仓：目标股数 = target × equity 换算，卖出差额。

        A 股 T+1：卖出以可卖份额为限（当日买入不可卖）；可卖不足时
        仅卖出最大可卖量，剩余目标权重挂起顺延（每 Bar 再试）。
        """
        pos = self.account.positions.get(sym)
        if pos is None or pos.shares <= 0:
            pending_targets.pop(sym, None)
            self._log_reject(log, ts, sym, ACT_SELL, "no_position")
            return

        open_price = float(brow["open"])
        target_shares = int(target * self.account.total_equity
                            / (open_price * 100.0)) * 100  # 100 股整数倍
        sell_shares = pos.shares - target_shares
        if sell_shares <= 0:
            pending_targets.pop(sym, None)  # 已达成目标，撤销顺延
            return

        sell_shares = min(sell_shares, pos.sellable_shares)
        if sell_shares <= 0:
            pending_targets[sym] = target  # T+1 锁定 → 顺延
            self._log_reject(log, ts, sym, ACT_SELL, "t1_lock")
            return

        self._execute_sell(sym, ts, brow, sell_shares, reason=reason, log=log)
        after = self.account.positions.get(sym)
        if after is not None and after.shares > target_shares:
            pending_targets[sym] = target  # 仍有锁定份额未减 → 顺延
        else:
            pending_targets.pop(sym, None)

    def _execute_sell(self, sym: str, ts: pd.Timestamp, brow: pd.Series,
                      shares: int, reason: str, log: TradeLog) -> None:
        """以 Bar 开盘价（扣除动态滑点与费用）卖出指定可卖份额。"""
        pos = self.account.positions.get(sym)
        if pos is None:
            return
        shares = min(int(shares), pos.sellable_shares)
        if shares <= 0:
            return

        open_price = float(brow["open"])
        bar_amount = float(brow["amount"])
        est_amount = shares * open_price
        price0 = self.cost.sell_price(open_price, est_amount, bar_amount)
        gross = shares * price0
        commission, stamp, transfer = self.cost.sell_fees(gross)
        proceeds = gross - commission - stamp - transfer

        self.account.sell(sym, ts, price0, shares, proceeds)
        log.add(ts=ts, symbol=sym, side=ACT_SELL, price=price0, shares=shares,
                amount=gross, commission=commission, stamp_duty=stamp,
                transfer_fee=transfer,
                slippage_bps=self.cost.slippage_bps(est_amount, bar_amount),
                cash_after=self.account.cash,
                equity_after=self.account.total_equity, reason=reason)

    # ------------------------------------------------------------------
    # 拒绝记录
    # ------------------------------------------------------------------

    @staticmethod
    def _log_reject(log: TradeLog, ts: pd.Timestamp, sym: str,
                    side: str, reason: str) -> None:
        log.add(ts=ts, symbol=sym, side=side, price=0.0, shares=0, amount=0.0,
                commission=0.0, stamp_duty=0.0, transfer_fee=0.0,
                slippage_bps=0.0, cash_after=0.0, equity_after=0.0, reason=reason)
