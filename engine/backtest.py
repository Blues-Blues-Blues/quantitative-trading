"""事件驱动型分钟级 A 股回测撮合引擎 BacktestEngine。

信号→成交的时序（严格防未来函数）：
    Bar t 产生的 Signal（基于 t 的 VWAP/close 合成）在 Bar t+1 的 open 价成交，
    即信号与成交之间存在 1 根 Bar 的执行延迟 —— 不会使用信号时点尚未可知的价格。

逐 Bar 推进顺序：
    1) T+1 解冻（roll_to_date）：上一交易日买入的份额转为可卖
    2) 撮合 T+1 挂起卖出（次日开盘强制卖出；跌停则推迟到下一 Bar 再试）
    3) 撮合上一 Bar 收集的信号（先卖后买释放现金；先到先得，现金不足拒绝）
    4) 收集当前 Bar 产生的新信号，留待下一 Bar 撮合
    5) 收盘 mark-to-market 并记录净值曲线

撮合与风控规则：
- 涨停（Bar high 触及 up_limit）不可买入；跌停（Bar low 触及 down_limit）不可卖出
- 买入股数按 100 股整数倍向下取整（一手）
- 成交价 = open × (1 ± 动态滑点)，成本含佣金/印花税/过户费（见 engine.execution）
- 单股最大仓位上限与总账户杠杆上限（见 engine.risk_control / engine.portfolio）
- T+1：当日买入份额当日不可卖；SELL 信号遇 T+1 锁定时挂起，次日开盘强制卖出
- 平仓卖出以可卖份额为限（当日加仓的份额仍锁定）

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
from strategy.signals import ACT_ADD, ACT_BUY, ACT_SELL, Signal

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
    """

    def __init__(self, account: Account, cost: ExecutionCost, sizer: PositionSizer,
                 data: DataSlice, signals: Sequence[Signal]) -> None:
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
        pending_sells: Dict[str, pd.Timestamp] = {}  # T+1 挂起卖出（symbol → 首次 SELL 时点）

        for ts in self._axis:
            date = pd.Timestamp(ts).normalize()
            bar = self._kline_by_ts.get(ts)

            # 1) T+1 解冻：上一交易日买入的份额可卖
            self.account.roll_to_date(date)

            # 2) 撮合 T+1 挂起卖出（次日开盘强制卖出；跌停保留到下个 Bar）
            for sym in list(pending_sells):
                pos = self.account.positions.get(sym)
                if pos is None or pos.shares <= 0:
                    del pending_sells[sym]  # 已无持仓，撤销挂起
                    continue
                if pos.sellable_shares <= 0:
                    continue  # 当日买入仍未解冻（T+1），保留挂起
                if bar is None or sym not in bar.index:
                    continue  # 无报价，保留挂起
                brow = bar.loc[sym]
                if brow["low"] <= brow["down_limit"]:
                    log.add(ts=ts, symbol=sym, side=ACT_SELL, price=0.0, shares=0,
                            amount=0.0, commission=0.0, stamp_duty=0.0,
                            transfer_fee=0.0, slippage_bps=0.0,
                            cash_after=self.account.cash,
                            equity_after=self.account.total_equity,
                            reason="limit_down")
                    continue  # 跌停不可卖，继续挂起
                self._execute_sell(sym, ts, brow, pos.sellable_shares,
                                   reason="t1_deferred_sell", log=log)
                del pending_sells[sym]

            # 3) 撮合上一 Bar 的信号（下一 Bar 开盘价；先到先得）
            for sig in pending_signals:
                self._execute(sig, ts, bar, pending_sells, log)
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
    # 信号撮合
    # ------------------------------------------------------------------

    def _execute(self, sig: Signal, ts: pd.Timestamp, bar: Optional[pd.DataFrame],
                 pending_sells: Dict[str, pd.Timestamp], log: TradeLog) -> None:
        """撮合单个信号（成交于 Bar ts 的开盘价）。"""
        sym = sig.symbol
        if bar is None or sym not in bar.index:
            self._log_reject(log, ts, sym, sig.action, "no_quote")
            return
        brow = bar.loc[sym]

        if sig.action in (ACT_BUY, ACT_ADD):
            self._execute_buy(sig, ts, brow, log)
        elif sig.action == ACT_SELL:
            self._execute_sell_signal(sig, ts, brow, pending_sells, log)
        # ACT_HOLD：无操作，不记录

    # ------------------------------------------------------------------
    # 买入 / 加仓
    # ------------------------------------------------------------------

    def _execute_buy(self, sig: Signal, ts: pd.Timestamp, brow: pd.Series,
                     log: TradeLog) -> None:
        sym = sig.symbol
        open_price = float(brow["open"])
        bar_amount = float(brow["amount"])

        # 涨跌停：盘中触及涨停不可买入
        if brow["high"] >= brow["up_limit"]:
            self._log_reject(log, ts, sym, sig.action, "limit_up")
            return

        equity = self.account.total_equity
        target = self.sizer.target_value(equity, sig.metrics or {})

        # 加仓 = 补足目标市值差额；新建仓 = 目标市值
        pos = self.account.positions.get(sym)
        current_value = pos.shares * open_price if pos is not None else 0.0
        order_value = max(0.0, target - current_value)
        if order_value <= 1e-6:
            return  # 已满仓，无需加仓

        # 单股上限：超过则裁剪至上限
        ok, reason = self.sizer.check_single_position(equity, current_value, order_value)
        if not ok:
            order_value = max(0.0, self.sizer.max_single_position * equity - current_value)
            if order_value <= 1e-6:
                self._log_reject(log, ts, sym, sig.action, reason)
                return

        # 总杠杆上限
        ok, reason = self.sizer.check_leverage(
            equity, self.account.position_value, order_value)
        if not ok:
            self._log_reject(log, ts, sym, sig.action, reason)
            return

        # 动态滑点成交价
        price0 = self.cost.buy_price(open_price, order_value, bar_amount)
        shares = int(order_value / (price0 * 100.0)) * 100  # 100 股整数倍
        if shares <= 0:
            self._log_reject(log, ts, sym, sig.action, "small_order")
            return

        amount = shares * price0
        commission, transfer = self.cost.buy_fees(amount)
        total_cost = amount + commission + transfer

        # 现金检查（先到先得：现金不足直接拒绝）
        if total_cost > self.account.cash + 1e-6:
            self._log_reject(log, ts, sym, sig.action, "insufficient_cash")
            return

        self.account.buy(sym, ts, price0, shares, total_cost)
        log.add(ts=ts, symbol=sym, side=sig.action, price=price0, shares=shares,
                amount=amount, commission=commission, stamp_duty=0.0,
                transfer_fee=transfer,
                slippage_bps=self.cost.slippage_bps(order_value, bar_amount),
                cash_after=self.account.cash,
                equity_after=self.account.total_equity, reason="filled")

    # ------------------------------------------------------------------
    # 卖出
    # ------------------------------------------------------------------

    def _execute_sell_signal(self, sig: Signal, ts: pd.Timestamp, brow: pd.Series,
                             pending_sells: Dict[str, pd.Timestamp], log: TradeLog) -> None:
        sym = sig.symbol
        pos = self.account.positions.get(sym)
        if pos is None or pos.shares <= 0:
            self._log_reject(log, ts, sym, ACT_SELL, "no_position")
            return

        # T+1：当日买入份额不可卖 → 挂起，次日开盘强制卖出
        if pos.sellable_shares <= 0:
            pending_sells[sym] = sig.timestamp
            self._log_reject(log, ts, sym, ACT_SELL, "t1_lock")
            return

        # 跌停不可卖出（不挂起：后续若再有 SELL 信号会重新触发）
        if brow["low"] <= brow["down_limit"]:
            self._log_reject(log, ts, sym, ACT_SELL, "limit_down")
            return

        # 平仓：卖出全部可卖份额（当日加仓的锁定份额保留）
        self._execute_sell(sym, ts, brow, pos.sellable_shares, reason="signal_sell", log=log)

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
