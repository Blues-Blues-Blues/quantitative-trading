"""事件驱动型分钟级 A 股回测撮合引擎 BacktestEngine。

信号→成交的时序（严格防未来函数）：
    Bar t 产生的 Signal（基于 t 的 VWAP/close 合成）在 Bar t+1 的 open 价成交，
    即信号与成交之间存在 1 根 Bar 的执行延迟 —— 不会使用信号时点尚未可知的价格。

逐 Bar 推进顺序：
    1) T+1 解冻（roll_to_date）：上一交易日买入的份额转为可卖
    2) 撮合 T+1 顺延减仓目标（每 Bar 按当前价换算；可卖不足保留、跌停暂停）
    3) 撮合上一 Bar 收集的信号（先卖后买释放现金；先到先得，现金不足拒绝）
    4) 收盘 mark-to-market；按真实账户持仓生成下一 Bar 信号
    5) 记录净值曲线；本 Bar 成交额成为下一 Bar 的流动性输入

撮合与风控规则（基于 Target_Weight 差额调仓）：
- 动作 → 目标权重：BUY/ADD → metrics['target_weight']；DECAY_REDUCE →
  metrics['target_weight']（信号层 = simulated × reduce_step_ratio）；
  SELL → 0（清仓）。HOLD 不触发调仓。
- 调仓死区：已持仓且 |Target - Current| < deadzone_th 的微调跳过（避免摩擦）；
  从 0 建仓与强制清仓豁免死区。
- 目标股数 = (目标权重 × 总权益) 按 Bar 开盘价换算，向下取整 100 股整数倍；
  加仓受现金（扣除佣金）约束；减仓/清仓以可卖份额为限（T+1），
  可卖不足时仅卖出最大可卖量，剩余目标权重挂起顺延（每 Bar 再试，跌停暂停）。
- 开盘价达到涨停价不可买入，达到跌停价不可卖出；盘中高低价不参与开盘撮合
- 成交价 = open × (1 ± 动态滑点)，以上一根已完成 Bar 成交额估计参与率，
  超出涨跌停价时以涨跌停价成交；成本含佣金/印花税/过户费
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
class OrderReport:
    """单次订单尝试的结构化结果；部分成交时分别记录已成交和待顺延数量。"""
    ts: pd.Timestamp
    symbol: str
    side: str
    filled_shares: int
    price: float
    status: str
    reason: str
    remaining_shares: int = 0


@dataclass
class TradeLog:
    """完整成交日志：成交单与拒绝单均记录（shares=0 表示被拒，reason 说明原因）。"""

    rows: List[dict] = field(default_factory=list)
    reports: List[OrderReport] = field(default_factory=list)

    def add(self, **kw) -> None:
        """追加一次撮合尝试，并按成交数量、剩余数量和原因派生结构化状态；原始行与 OrderReport 同步保存。"""
        remaining = int(kw.pop("remaining_shares", 0))
        status = ("pending" if kw["reason"] == "t1_lock" else
                  "rejected" if kw["shares"] == 0 else
                  "partial_pending" if remaining else "filled")
        self.reports.append(OrderReport(
            pd.Timestamp(kw["ts"]), kw["symbol"], kw["side"],
            int(kw["shares"]), float(kw["price"]), status,
            str(kw["reason"]), remaining))
        self.rows.append(kw)

    def to_frame(self) -> pd.DataFrame:
        """把成交及拒绝记录转成稳定列序的表；空日志也返回带完整列名的空表，方便下游无需分支处理。"""
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
        """把逐 Bar 账户快照转成稳定列序的表；时间列统一为 pandas 时间类型。"""
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
                 data: DataSlice, signals: Sequence[Signal] = (),
                 deadzone_th: float = 0.05, state_machine=None,
                 features: Optional[pd.DataFrame] = None) -> None:
        """准备回测依赖、按时间索引的行情与信号，并校验信号顺序；状态机模式会先构建逐 Bar 评估表。实例只允许调用 run 一次，以免复用已变更账户状态。"""
        if not 0.0 <= deadzone_th < 1.0:
            raise ValueError(f"deadzone_th 必须在 [0, 1) 区间，当前: {deadzone_th}")
        self.deadzone_th = deadzone_th
        self.account = account
        self.cost = cost
        self.sizer = sizer
        self.data = data
        self.signals = list(signals)
        self.state_machine = state_machine
        self.features = features
        if state_machine is not None:
            if features is None:
                raise ValueError("state_machine 模式需要 features")
            state_machine.reset()
            evaluation = state_machine._build_eval_table(data, features)
            self._eval_by_ts = {pd.Timestamp(ts): group for ts, group in
                                evaluation.groupby("ts", sort=True)}
        else:
            self._eval_by_ts = {}
        self._has_run = False
        self.trade_log: Optional[pd.DataFrame] = None
        self.equity_curve: Optional[pd.DataFrame] = None
        self.order_reports: List[OrderReport] = []
        self.generated_signals: List[Signal] = []

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
        """将长表行情整理成 timestamp 到 symbol 索引表的映射；撮合循环可直接按时刻取全市场 Bar，且在入口处校验必需列。"""
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
        if self._has_run:
            raise RuntimeError("BacktestEngine.run() 只能执行一次；请新建引擎")
        self._has_run = True
        log = TradeLog()
        curve = EquityCurve()
        pending_signals: List[Signal] = []   # 上一 Bar 收集、本 Bar 撮合
        pending_targets: Dict[str, float] = {}  # T+1 顺延减仓目标（symbol → 目标权重）
        previous_amount: Dict[str, float] = {}

        for ts in self._axis:
            date = pd.Timestamp(ts).normalize()
            bar = self._kline_by_ts.get(ts)

            # 1) T+1 解冻：上一交易日买入的份额可卖
            self.account.roll_to_date(date)
            if bar is not None:
                self.account.mark_to_market(
                    {sym: row["open"] for sym, row in bar.iterrows()
                     if pd.notna(row["open"]) and row["open"] > 0})

            # 2) 先执行上一 Bar 的新决定；同一标的的新决定覆盖旧顺延单。
            active_symbols = {s.symbol for s in pending_signals if s.action != ACT_HOLD}
            for sym in list(pending_targets):
                if sym in active_symbols:
                    del pending_targets[sym]
                    continue
                pos = self.account.positions.get(sym)
                if pos is None or pos.shares <= 0:
                    del pending_targets[sym]  # 已无持仓，撤销顺延
                    continue
                if pos.sellable_shares <= 0:
                    continue  # 当日买入仍未解冻（T+1），保留顺延
                if bar is None or sym not in bar.index:
                    continue  # 无报价，保留顺延
                brow = bar.loc[sym]
                if (pd.isna(brow["open"]) or brow["open"] <= 0 or
                        pd.isna(brow["down_limit"]) or
                        brow["open"] <= brow["down_limit"]):
                    continue  # 跌停暂停，保留顺延
                brow = brow.copy()
                brow["liquidity_amount"] = previous_amount.get(sym, 1e-9)
                self._execute_sell_to_target(
                    sym, pending_targets[sym], ts, brow,
                    pending_targets, log, reason="t1_deferred_sell")

            # 3) 撮合上一 Bar 的信号（下一 Bar 开盘价；先到先得）
            for sig in pending_signals:
                self._execute(sig, ts, bar, pending_targets, log,
                              previous_amount)
            pending_signals = []

            # 5) 收盘 mark-to-market + 净值曲线
            if bar is not None:
                self.account.mark_to_market(
                    {sym: row["close"] for sym, row in bar.iterrows()
                     if pd.notna(row["close"]) and row["close"] > 0})
            # 4) 收盘后决策；此时账户包含本 Bar 已成交及盯市结果。
            if self.state_machine is not None:
                rows = self._eval_by_ts.get(pd.Timestamp(ts))
                if rows is not None:
                    pending_signals = self.state_machine.on_bar(
                        ts, rows, self.account)
            else:
                pending_signals = list(self._signal_by_ts.get(ts, []))
            self.generated_signals.extend(pending_signals)
            curve.rows.append({
                "ts": ts,
                "cash": self.account.cash,
                "margin": self.account.margin,
                "position_value": self.account.position_value,
                "total_equity": self.account.total_equity,
                "n_positions": len(self.account.positions),
                "unrealized_pnl": self.account.unrealized_pnl(),
            })
            if bar is not None:
                previous_amount.update({
                    sym: (float(row["amount"])
                          if pd.notna(row["amount"]) and row["amount"] > 0
                          else 1e-9)
                    for sym, row in bar.iterrows()})

        # 尾部校验
        assert self.account.cash >= -1e-6, "回测结束现金为负"
        if pending_signals:
            logger.warning("%d 个信号在最后一根 Bar 产生，无后续 Bar 可成交，已丢弃",
                           len(pending_signals))
        self.trade_log, self.equity_curve = log.to_frame(), curve.to_frame()
        self.order_reports = log.reports
        return self.trade_log, self.equity_curve

    # ------------------------------------------------------------------
    # 信号撮合（基于 Target_Weight 差额调仓）
    # ------------------------------------------------------------------

    def _execute(self, sig: Signal, ts: pd.Timestamp, bar: Optional[pd.DataFrame],
                 pending_targets: Dict[str, float], log: TradeLog,
                 previous_amount: Optional[Dict[str, float]] = None) -> None:
        """撮合单个信号（成交于 Bar ts 的开盘价）。

        HOLD 不触发调仓；BUY/ADD/DECAY_REDUCE/SELL 映射为目标权重后差额调仓。
        """
        sym = sig.symbol
        if bar is None or sym not in bar.index:
            self._log_reject(log, ts, sym, sig.action, "no_quote")
            return
        if sig.action == ACT_HOLD:
            return  # HOLD：目标权重仅作监控，不调仓
        brow = bar.loc[sym].copy()
        if pd.isna(brow["open"]) or float(brow["open"]) <= 0:
            self._log_reject(log, ts, sym, sig.action, "invalid_open")
            return
        brow["liquidity_amount"] = (previous_amount or {}).get(sym, 1e-9)
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
        # 减仓动作防转增仓：空仓无可减直接返回；目标裁剪不超过当前权重。
        # （信号层 target_weight 基于 simulated_weight，引擎空仓后它仍 >0，
        #   若放行会被 delta>0 分支误判成「买入回补」——正是 SELL 后
        #   DECAY_REDUCE 又买回的 T+1 违规根因。）
        if action == ACT_DECAY_REDUCE and (pos is None or pos.shares <= 0):
            return
        current_weight = (pos.shares * open_price / equity
                          if pos is not None and pos.shares > 0 else 0.0)
        if action == ACT_DECAY_REDUCE:
            target = min(target, current_weight)  # 减仓目标不得超过当前权重
        delta = target - current_weight

        # 调仓死区：已持仓的微调（|Δ| < deadzone_th）跳过，避免交易摩擦；
        # 从 0 建仓（current==0）与强制清仓（target==0）豁免。
        if target > 0.0 and current_weight > 0.0 and abs(delta) < self.deadzone_th:
            return

        if delta > 0.0:  # 建仓 / 加仓
            if pd.isna(brow["up_limit"]):
                self._log_reject(log, ts, sym, action, "missing_limit")
                return
            if brow["open"] >= brow["up_limit"]:
                self._log_reject(log, ts, sym, action, "limit_up")
                return
            self._execute_buy_to_target(sym, target, action, ts, brow,
                                        pending_targets, log)
        elif delta < 0.0:  # 减仓 / 清仓
            if pd.isna(brow["down_limit"]):
                self._log_reject(log, ts, sym, action, "missing_limit")
                return
            if brow["open"] <= brow["down_limit"]:
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
                               pending_targets: Dict[str, float],
                               log: TradeLog) -> None:
        """按目标权重加仓：目标市值 = target × equity，补足差额（100 股整数倍）。

        受单股上限 / 总杠杆 / 当前可用现金（扣除佣金过户费）约束。
        """
        open_price = float(brow["open"])
        bar_amount = float(brow.get("liquidity_amount", 1e-9))
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

        # 动态滑点：先按滑点前开盘价折 100 股整数倍（与卖出侧同基准），
        # 再以"实际股数 × 滑点前价"作为参与率与费用基准。
        shares = int(order_value / (open_price * 100.0)) * 100
        if shares <= 0:
            self._log_reject(log, ts, sym, action, "small_order")
            return

        base_amount = shares * open_price  # 滑点前估值（卖侧 est_amount 同口径）
        price0 = self.cost.buy_price(open_price, base_amount, bar_amount)
        price0 = min(price0, float(brow["up_limit"]))
        amount = shares * price0
        commission, transfer = self.cost.buy_fees(amount)
        total_cost = amount + commission + transfer

        # 现金检查（先到先得：现金不足直接拒绝）
        if total_cost > self.account.cash + 1e-6:
            self._log_reject(log, ts, sym, action, "insufficient_cash")
            return

        self.account.buy(sym, ts, price0, shares, total_cost)
        pending_targets.pop(sym, None)  # 加仓成交 → 旧顺延清仓目标已失效，撤销
        log.add(ts=ts, symbol=sym, side=action, price=price0, shares=shares,
                amount=amount, commission=commission, stamp_duty=0.0,
                transfer_fee=transfer,
                slippage_bps=(price0 / open_price - 1.0) * 1e4,
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
        reject_side = ACT_DECAY_REDUCE if reason == "decay_reduce" else ACT_SELL
        if pos is None or pos.shares <= 0:
            pending_targets.pop(sym, None)
            self._log_reject(log, ts, sym, reject_side, "no_position")
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
            # T+1 锁定 → 顺延：这是设计行为（每 Bar 记一次），并非失败拒绝，
            # 后续 Bar 会继续尝试直至可卖份额解冻。reason 保留 "t1_lock" 以兼容
            # 既有下游判定，展示层负责标注「顺延中」。
            pending_targets[sym] = target
            self._log_reject(log, ts, sym, reject_side, "t1_lock",
                             remaining_shares=pos.shares - target_shares)
            return

        self._execute_sell(sym, ts, brow, sell_shares, reason=reason, log=log)
        after = self.account.positions.get(sym)
        if after is not None and after.shares > target_shares:
            pending_targets[sym] = target  # 仍有锁定份额未减 → 顺延
            log.reports[-1].status = "partial_pending"
            log.reports[-1].remaining_shares = after.shares - target_shares
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
        bar_amount = float(brow.get("liquidity_amount", 1e-9))
        est_amount = shares * open_price
        price0 = self.cost.sell_price(open_price, est_amount, bar_amount)
        price0 = max(price0, float(brow["down_limit"]))
        gross = shares * price0
        commission, stamp, transfer = self.cost.sell_fees(gross)
        proceeds = gross - commission - stamp - transfer

        self.account.sell(sym, ts, price0, shares, proceeds)
        log.add(ts=ts, symbol=sym, side=ACT_SELL, price=price0, shares=shares,
                amount=gross, commission=commission, stamp_duty=stamp,
                transfer_fee=transfer,
                slippage_bps=(1.0 - price0 / open_price) * 1e4,
                cash_after=self.account.cash,
                equity_after=self.account.total_equity, reason=reason)

    # ------------------------------------------------------------------
    # 拒绝记录
    # ------------------------------------------------------------------

    @staticmethod
    def _log_reject(log: TradeLog, ts: pd.Timestamp, sym: str,
                    side: str, reason: str, remaining_shares: int = 0) -> None:
        """记录未成交的订单尝试；统一填零成交和费用字段，使拒绝原因也能进入标准成交日志。"""
        log.add(ts=ts, symbol=sym, side=side, price=0.0, shares=0, amount=0.0,
                commission=0.0, stamp_duty=0.0, transfer_fee=0.0,
                slippage_bps=0.0, cash_after=0.0, equity_after=0.0,
                reason=reason, remaining_shares=remaining_shares)
