"""事件驱动型分钟级 A 股回测撮合引擎 BacktestEngine。

信号→成交的时序（严格防未来函数）：
    Bar t 产生的 Signal（基于 t 的 VWAP/close 合成）在 Bar t+1 的 open 价成交，
    即信号与成交之间存在 1 根 Bar 的执行延迟 —— 不会使用信号时点尚未可知的价格。

逐 Bar 推进顺序：
    1) T+1 解冻（roll_to_date）：上一交易日买入的份额转为可卖
    2) 撮合每个标的尚未完成的目标（风险卖单优先；无报价、跌停及 T+1 顺延）
    3) 买单按有效期处理，午间与隔夜边界可到期
    4) 收盘 mark-to-market；按真实账户持仓生成下一 Bar 信号
    5) 记录净值曲线；本 Bar 成交额成为下一 Bar 的流动性输入

撮合与风控规则（基于 Target_Weight 差额调仓）：
- 动作 → 目标权重：BUY/ADD → metrics['target_weight']；DECAY_REDUCE →
  metrics['target_weight']（信号层 = simulated × reduce_step_ratio）；
  SELL → 0（清仓）。HOLD 不触发调仓。
- 调仓死区：已持仓且 |Target - Current| < deadzone_th 的微调跳过；
  从 0 建仓与强制清仓豁免，风险减仓可单独选择豁免。
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
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import numpy as np

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
    event_seq: int = 0
    fill_id: Optional[str] = None
    signal_id: Optional[str] = None
    decision_ts: Optional[pd.Timestamp] = None
    cancelled_ts: Optional[pd.Timestamp] = None
    superseded_by: Optional[str] = None
    strategy_cause: Optional[str] = None


@dataclass(frozen=True)
class PendingTarget:
    """每标的一份顺延目标；保留动作、策略原因、目标权重与 T+1 历史。"""
    target_weight: float
    signal_id: Optional[str]
    decision_ts: Optional[pd.Timestamp]
    action: str = ACT_SELL
    strategy_cause: Optional[str] = None
    metrics: Optional[dict] = None
    first_attempt: bool = True
    symbol: str = ""
    had_t1_lock: bool = False
    target_locked: bool = False


@dataclass
class TradeLog:
    """完整成交日志：成交单与拒绝单均记录（shares=0 表示被拒，reason 说明原因）。"""

    rows: List[dict] = field(default_factory=list)
    reports: List[OrderReport] = field(default_factory=list)
    signal_id: Optional[str] = None
    decision_ts: Optional[pd.Timestamp] = None
    strategy_cause: Optional[str] = None
    events: List[dict] = field(default_factory=list)
    _fill_count: int = 0

    def add(self, **kw) -> None:
        """追加一次撮合尝试，并按成交数量、剩余数量和原因派生结构化状态；原始行与 OrderReport 同步保存。"""
        remaining = int(kw.pop("remaining_shares", 0))
        if kw["shares"] > 0:
            self._fill_count += 1
        kw.update(event_seq=len(self.rows) + 1,
                  fill_id=f"fill_{self._fill_count:08d}" if kw["shares"] > 0 else None,
                  signal_id=self.signal_id, decision_ts=self.decision_ts)
        status = ("pending" if kw["reason"] in ("t1_lock", "limit_down", "limit_up",
                                                "invalid_open", "no_quote", "missing_limit") else
                  "rejected" if kw["shares"] == 0 else
                  "partial_pending" if remaining else "filled")
        self.reports.append(OrderReport(
            pd.Timestamp(kw["ts"]), kw["symbol"], kw["side"],
            int(kw["shares"]), float(kw["price"]), status,
            str(kw["reason"]), remaining, kw["event_seq"], kw["fill_id"],
            kw["signal_id"], kw["decision_ts"], strategy_cause=self.strategy_cause))
        kw["strategy_cause"] = self.strategy_cause
        self.rows.append(kw)
        self.event(kw["ts"], kw["symbol"], kw["side"], status,
                   kw["reason"])

    def event(self, ts, symbol, action, status, reason, source=None,
              superseded_by=None) -> None:
        self.events.append(dict(ts=pd.Timestamp(ts), symbol=symbol, action=action,
                                status=status, reason=reason,
                                signal_id=source.signal_id if source else self.signal_id,
                                decision_ts=source.decision_ts if source else self.decision_ts,
                                strategy_cause=source.strategy_cause if source else self.strategy_cause,
                                superseded_by=superseded_by))

    def to_frame(self) -> pd.DataFrame:
        """把成交及拒绝记录转成稳定列序的表；空日志也返回带完整列名的空表，方便下游无需分支处理。"""
        cols = ["ts", "symbol", "side", "price", "shares", "amount",
                "commission", "stamp_duty", "transfer_fee", "slippage_bps",
                "cash_after", "equity_after", "reason",
                "event_seq", "fill_id", "signal_id", "decision_ts", "strategy_cause"]
        if not self.rows:
            return pd.DataFrame(columns=cols).astype({"event_seq": "int64", "decision_ts": "datetime64[ns]"})
        df = pd.DataFrame(self.rows)
        df["ts"] = pd.to_datetime(df["ts"])
        df["decision_ts"] = pd.to_datetime(df["decision_ts"])
        return df[cols].sort_values(["ts", "event_seq"], kind="stable").reset_index(drop=True)


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
                 features: Optional[pd.DataFrame] = None, snapshot_sink=None,
                 entry_order_expiry: str = "same_session",
                 risk_reduce_bypass_deadzone: bool = False,
                 risk_reduce_min_shares: int = 100) -> None:
        """准备回测依赖、按时间索引的行情与信号，并校验信号顺序；状态机模式会先构建逐 Bar 评估表。实例只允许调用 run 一次，以免复用已变更账户状态。"""
        if not 0.0 <= deadzone_th < 1.0:
            raise ValueError(f"deadzone_th 必须在 [0, 1) 区间，当前: {deadzone_th}")
        self.deadzone_th = deadzone_th
        if entry_order_expiry not in ("same_session", "next_valid_bar"):
            raise ValueError("entry_order_expiry 必须为 same_session 或 next_valid_bar")
        if (isinstance(risk_reduce_min_shares, bool) or
                not isinstance(risk_reduce_min_shares, int) or
                risk_reduce_min_shares <= 0 or risk_reduce_min_shares % 100):
            raise ValueError("risk_reduce_min_shares 必须为正的 100 股整数倍")
        self.entry_order_expiry = entry_order_expiry
        self.risk_reduce_bypass_deadzone = bool(risk_reduce_bypass_deadzone)
        self.risk_reduce_min_shares = risk_reduce_min_shares
        self.account = account
        self.cost = cost
        self.sizer = sizer
        self.data = data
        self.signals = list(signals)
        self.state_machine = state_machine
        self.features = features
        self.snapshot_sink = snapshot_sink
        supplied_ids = [s.signal_id for s in self.signals if s.signal_id is not None]
        if any(not isinstance(s, str) or not s for s in supplied_ids):
            raise ValueError("signal_id 必须是非空字符串")
        if len(set(supplied_ids)) != len(supplied_ids):
            raise ValueError("重复 signal_id")
        self._reserved_ids = set(supplied_ids)
        self._queued_ids = set()
        self._signal_count = 0
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
        self.order_events: pd.DataFrame = pd.DataFrame()
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
        self._date_ordinals = {day: i for i, day in
                               enumerate(pd.DatetimeIndex(self._axis).normalize().unique())}

    # ------------------------------------------------------------------
    # 预处理
    # ------------------------------------------------------------------

    def _enqueue(self, signals: Sequence[Signal]) -> List[Signal]:
        """按原入队顺序复制信号；预留外部 ID，自动编号不改变下单优先级。"""
        provided = [s.signal_id for s in signals if s.signal_id is not None]
        if any(not isinstance(s, str) or not s for s in provided):
            raise ValueError("signal_id 必须是非空字符串")
        if len(set(provided)) != len(provided) or self._queued_ids.intersection(provided):
            raise ValueError("重复 signal_id")
        self._reserved_ids.update(provided)
        queued = []
        for signal in signals:
            sid = signal.signal_id
            if sid is None:
                while True:
                    self._signal_count += 1
                    sid = f"sig_{self._signal_count:08d}"
                    if sid not in self._reserved_ids and sid not in self._queued_ids:
                        break
            self._queued_ids.add(sid)
            queued.append(replace(signal, signal_id=sid, metrics=dict(signal.metrics or {})))
        return queued

    @staticmethod
    def _cancel_pending(log: TradeLog, source: PendingTarget, ts, superseded_by=None,
                        status="superseded"):
        """在订单回报保留顺延的取消/覆盖信息，不新增虚假的成交日志行。"""
        for report in reversed(log.reports):
            if report.signal_id == source.signal_id and report.status in ("pending", "partial_pending"):
                report.status = status
                report.cancelled_ts = pd.Timestamp(ts)
                report.superseded_by = superseded_by
                break
        log.event(ts, source.symbol, source.action, status, status,
                  source, superseded_by)

    def _send_snapshot(self, ts, signals):
        """仅启用回调时构造当前 Bar 的真实持仓和同源因子表，不保存全期间副本。"""
        factors = ["global_mod", "chain_mod", "agent_ms"]
        columns = ["ts", "symbol", "shares", "mark_price", "market_value", "total_equity"]
        equity = self.account.total_equity
        positions = pd.DataFrame([
            dict(ts=ts, symbol=sym, shares=p.shares, mark_price=p.last_price,
                 market_value=p.shares * p.last_price, total_equity=equity)
            for sym, p in self.account.positions.items() if p.shares > 0
        ], columns=columns)
        if self.state_machine is not None:
            rows = self._eval_by_ts.get(pd.Timestamp(ts), pd.DataFrame())
            values = rows.reindex(columns=["ts", "symbol"] + factors).copy()
        else:
            values = pd.DataFrame([
                dict(ts=ts, symbol=s.symbol, **{f: s.metrics.get(f) for f in factors})
                for s in signals
            ], columns=["ts", "symbol"] + factors)
            # 同股票同一 Bar 可以有多个决策，但快照必须无歧义。
            values = values.drop_duplicates()
        self.snapshot_sink(ts, positions, equity, values)

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

        最后一根 Bar 产生的非 HOLD 决策记入 pending_at_end，不伪造成交。
        """
        if self._has_run:
            raise RuntimeError("BacktestEngine.run() 只能执行一次；请新建引擎")
        self._has_run = True
        log = TradeLog()
        curve = EquityCurve()
        pending_targets: Dict[str, PendingTarget] = {}
        previous_amount: Dict[str, Tuple[pd.Timestamp, float]] = {}

        for ts in self._axis:
            date = pd.Timestamp(ts).normalize()
            bar = self._kline_by_ts.get(ts)

            # 1) T+1 解冻：上一交易日买入的份额可卖
            self.account.roll_to_date(date)
            if bar is not None:
                self.account.mark_to_market(
                    {sym: row["open"] for sym, row in bar.iterrows()
                     if self._valid_price(row["open"])})

            # 所有未完成订单统一按标的存储；风险单先于入场单撮合。
            pending_order = sorted(list(pending_targets), key=lambda s:
                                   0 if pending_targets[s].action == ACT_SELL else
                                   1 if pending_targets[s].action == ACT_DECAY_REDUCE else 2)
            for sym in pending_order:
                source = pending_targets.get(sym)
                if source is None:
                    continue
                entry = source.action in (ACT_BUY, ACT_ADD)
                current_session = self._session(ts)
                decision_session = self._session(source.decision_ts)
                if entry and (self.entry_order_expiry == "same_session" and
                              current_session != decision_session or
                              self.entry_order_expiry == "next_valid_bar" and
                              current_session[0] > decision_session[0] and
                              self._trading_days_between(source.decision_ts, ts) > 1):
                    self._cancel_pending(log, source, ts, status="expired")
                    del pending_targets[sym]
                    continue
                if not entry and (sym not in self.account.positions or
                                  self.account.positions[sym].shares <= 0):
                    self._cancel_pending(log, source, ts, status="filled")
                    del pending_targets[sym]
                    continue
                sig = Signal(sym, source.decision_ts, source.action, "",
                             source.metrics or {}, signal_id=source.signal_id)
                self._execute(sig, ts, bar, pending_targets, log, previous_amount,
                              deferred=source.had_t1_lock)
                if (entry and self.entry_order_expiry == "next_valid_bar" and
                        current_session[0] > decision_session[0] and
                        bar is not None and sym in bar.index and
                        self._valid_price(bar.loc[sym]["open"]) and
                        sym in pending_targets and
                        pending_targets[sym].signal_id == source.signal_id):
                    self._cancel_pending(log, source, ts, status="expired")
                    pending_targets.pop(sym, None)
                elif pending_targets.get(sym) is source:
                    pending_targets[sym] = replace(source, first_attempt=False)

            # 5) 收盘 mark-to-market + 净值曲线
            if bar is not None:
                self.account.mark_to_market(
                    {sym: row["close"] for sym, row in bar.iterrows()
                     if self._valid_price(row["close"])})
            # 4) 收盘后决策；此时账户包含本 Bar 已成交及盯市结果。
            if self.state_machine is not None:
                rows = self._eval_by_ts.get(pd.Timestamp(ts))
                if rows is not None:
                    valid_symbols = {sym for sym, brow in bar.iterrows()
                                     if self._valid_price(brow["close"])} if bar is not None else set()
                    rows = rows.loc[rows[SYMBOL].isin(valid_symbols)]
                    new_signals = self.state_machine.on_bar(ts, rows, self.account)
                else:
                    new_signals = []
            else:
                valid_symbols = {sym for sym, brow in bar.iterrows()
                                 if self._valid_price(brow["close"])} if bar is not None else set()
                new_signals = [s for s in self._signal_by_ts.get(ts, [])
                               if s.symbol in valid_symbols]
            new_signals = self._enqueue(new_signals)
            self.generated_signals.extend(new_signals)
            for sig in new_signals:
                if sig.action == ACT_HOLD:
                    continue
                old = pending_targets.get(sig.symbol)
                if old is not None:
                    old_risk = old.action in (ACT_SELL, ACT_DECAY_REDUCE)
                    new_risk = sig.action in (ACT_SELL, ACT_DECAY_REDUCE)
                    if old_risk and not new_risk:
                        blocked = PendingTarget(0.0, sig.signal_id, sig.timestamp,
                                                sig.action, symbol=sig.symbol)
                        log.event(ts, sig.symbol, sig.action, "rejected",
                                  "risk_order_active", blocked)
                        continue
                    if old.action == ACT_SELL and sig.action == ACT_DECAY_REDUCE:
                        blocked = PendingTarget(0.0, sig.signal_id, sig.timestamp,
                                                sig.action, symbol=sig.symbol)
                        log.event(ts, sig.symbol, sig.action, "rejected",
                                  "risk_order_active", blocked)
                        continue
                    if (old.action == sig.action and
                            old.target_weight == self._target_for(sig, 0) and
                            old.strategy_cause == (sig.metrics or {}).get("exit_cause")):
                        continue
                    self._cancel_pending(log, old, ts, superseded_by=sig.signal_id)
                    if self.state_machine is not None and old.action == ACT_DECAY_REDUCE:
                        self.state_machine.on_reduce_cancelled(sig.symbol)
                pending_targets[sig.symbol] = PendingTarget(
                    self._target_for(sig, 0), sig.signal_id, sig.timestamp, sig.action,
                    (sig.metrics or {}).get("exit_cause"), dict(sig.metrics or {}),
                    True, sig.symbol)
                log.event(ts, sig.symbol, sig.action, "pending", "decision",
                          pending_targets[sig.symbol])
            if self.snapshot_sink is not None:
                self._send_snapshot(ts, new_signals)
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
                previous_amount.update({sym: (ts, float(row["amount"]))
                                        for sym, row in bar.iterrows()
                                        if self._valid_price(row["close"]) and
                                        self._valid_price(row["amount"])})

        # 尾部校验
        assert self.account.cash >= -1e-6, "回测结束现金为负"
        for sym, source in pending_targets.items():
            log.event(self._axis[-1], sym, source.action, "pending_at_end",
                      "no_next_bar", source)
        self.trade_log, self.equity_curve = log.to_frame(), curve.to_frame()
        self.order_reports = log.reports
        self.order_events = pd.DataFrame(log.events).reindex(columns=[
            "ts", "symbol", "action", "status", "reason", "signal_id",
            "decision_ts", "strategy_cause", "superseded_by"])
        return self.trade_log, self.equity_curve

    @staticmethod
    def _valid_price(value) -> bool:
        return bool(pd.notna(value) and np.isfinite(float(value)) and float(value) > 0)

    @staticmethod
    def _session(ts) -> Tuple[pd.Timestamp, str]:
        ts = pd.Timestamp(ts)
        return ts.normalize(), "am" if ts.hour < 12 else "pm"

    def _trading_days_between(self, start, end) -> int:
        return (self._date_ordinals[pd.Timestamp(end).normalize()] -
                self._date_ordinals[pd.Timestamp(start).normalize()])

    # ------------------------------------------------------------------
    # 信号撮合（基于 Target_Weight 差额调仓）
    # ------------------------------------------------------------------

    def _execute(self, sig: Signal, ts: pd.Timestamp, bar: Optional[pd.DataFrame],
                 pending_targets: Dict[str, PendingTarget], log: TradeLog,
                 previous_amount: Optional[Dict[str, Tuple[pd.Timestamp, float]]] = None,
                 deferred: bool = False) -> None:
        """撮合单个信号（成交于 Bar ts 的开盘价）。

        HOLD 不触发调仓；BUY/ADD/DECAY_REDUCE/SELL 映射为目标权重后差额调仓。
        """
        log.signal_id, log.decision_ts = sig.signal_id, sig.timestamp
        log.strategy_cause = (sig.metrics or {}).get("exit_cause")
        sym = sig.symbol
        if bar is None or sym not in bar.index:
            self._log_reject(log, ts, sym, sig.action, "no_quote")
            return
        if sig.action == ACT_HOLD:
            return  # HOLD：目标权重仅作监控，不调仓
        brow = bar.loc[sym].copy()
        if not self._valid_price(brow["open"]):
            self._log_reject(log, ts, sym, sig.action, "invalid_open")
            return
        prior = (previous_amount or {}).get(sym)
        if prior is not None and self._session(prior[0]) == self._session(ts):
            brow["liquidity_amount"] = prior[1]
        else:
            brow["liquidity_amount"] = 1e-9
            log.event(ts, sym, sig.action, "pending", "liquidity_fallback",
                      pending_targets.get(sym))
        pos = self.account.positions.get(sym)
        current_weight = (pos.shares * float(brow["open"]) / self.account.total_equity
                          if pos is not None and pos.shares > 0 else 0.0)
        source = pending_targets.get(sym)
        target = (source.target_weight if source is not None and source.target_locked
                  else self._target_for(sig, current_weight))
        if source is not None and not source.target_locked:
            pending_targets[sym] = replace(source, target_weight=target,
                                           target_locked=True, first_attempt=False)
        self._rebalance(sym, target, sig.action, ts, brow, pending_targets, log,
                        deferred=deferred)

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
                   brow: pd.Series, pending_targets: Dict[str, PendingTarget],
                   log: TradeLog, deferred: bool = False) -> None:
        """把该标的目标权重收敛到 target（死区 / 涨跌停 / T+1 约束）。"""
        equity = self.account.total_equity
        pos = self.account.positions.get(sym)
        open_price = float(brow["open"])
        # 减仓动作防转增仓：空仓无可减直接返回；目标裁剪不超过当前权重。
        # （信号层 target_weight 基于 simulated_weight，引擎空仓后它仍 >0，
        #   若放行会被 delta>0 分支误判成「买入回补」——正是 SELL 后
        #   DECAY_REDUCE 又买回的 T+1 违规根因。）
        if action == ACT_DECAY_REDUCE and (pos is None or pos.shares <= 0):
            pending_targets.pop(sym, None)
            if self.state_machine is not None:
                self.state_machine.on_reduce_cancelled(sym)
            return
        current_weight = (pos.shares * open_price / equity
                          if pos is not None and pos.shares > 0 else 0.0)
        if action == ACT_DECAY_REDUCE:
            target = min(target, current_weight)  # 减仓目标不得超过当前权重
        delta = target - current_weight

        # 调仓死区：已持仓的微调（|Δ| < deadzone_th）跳过，避免交易摩擦；
        # 从 0 建仓（current==0）与强制清仓（target==0）豁免。
        deadzone_applies = not (action == ACT_DECAY_REDUCE and self.risk_reduce_bypass_deadzone)
        if (deadzone_applies and target > 0.0 and current_weight > 0.0 and
                abs(delta) < self.deadzone_th):
            log.event(ts, sym, action, "rejected", "deadzone_skip", pending_targets.get(sym))
            pending_targets.pop(sym, None)
            if action == ACT_DECAY_REDUCE and self.state_machine is not None:
                self.state_machine.on_reduce_cancelled(sym)
            return

        if action == ACT_DECAY_REDUCE and pos is not None:
            target_shares = int(target * equity / (open_price * 100.0)) * 100
            if pos.shares - target_shares < self.risk_reduce_min_shares:
                log.event(ts, sym, action, "rejected", "min_lot_skip", pending_targets.get(sym))
                pending_targets.pop(sym, None)
                if self.state_machine is not None:
                    self.state_machine.on_reduce_cancelled(sym)
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
            if (sym in pending_targets and log.reports and
                    log.reports[-1].signal_id == log.signal_id and
                    log.reports[-1].status == "rejected"):
                pending_targets.pop(sym, None)
        elif delta < 0.0:  # 减仓 / 清仓
            if pd.isna(brow["down_limit"]):
                self._log_reject(log, ts, sym, action, "missing_limit")
                return
            if brow["open"] <= brow["down_limit"]:
                self._log_reject(log, ts, sym, action, "limit_down")
                return
            reason = ("t1_deferred_sell" if deferred else
                      "signal_sell" if action == ACT_SELL else "decay_reduce")
            self._execute_sell_to_target(sym, target, ts, brow, pending_targets,
                                         log, reason=reason)
        else:
            pending_targets.pop(sym, None)

    # ------------------------------------------------------------------
    # 买入 / 加仓
    # ------------------------------------------------------------------

    def _execute_buy_to_target(self, sym: str, target: float, action: str,
                               ts: pd.Timestamp, brow: pd.Series,
                               pending_targets: Dict[str, PendingTarget],
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
        pending_targets.pop(sym, None)
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
                                brow: pd.Series, pending_targets: Dict[str, PendingTarget],
                                log: TradeLog, reason: str = "signal_sell") -> None:
        """按目标权重减仓：目标股数 = target × equity 换算，卖出差额。

        A 股 T+1：卖出以可卖份额为限（当日买入不可卖）；可卖不足时
        仅卖出最大可卖量，剩余目标权重挂起顺延（每 Bar 再试）。
        """
        pos = self.account.positions.get(sym)
        source = pending_targets.get(sym)
        reject_side = source.action if source is not None else (
            ACT_DECAY_REDUCE if reason == "decay_reduce" else ACT_SELL)
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
            if source is not None:
                pending_targets[sym] = replace(source, target_weight=target,
                                               first_attempt=False, had_t1_lock=True)
            log.event(ts, sym, reject_side, "pending", "no_sellable_shares", source)
            self._log_reject(log, ts, sym, reject_side, "t1_lock",
                             remaining_shares=pos.shares - target_shares)
            return

        self._execute_sell(sym, ts, brow, sell_shares, reason=reason, log=log)
        after = self.account.positions.get(sym)
        if after is not None and after.shares > target_shares:
            if source is not None:
                pending_targets[sym] = replace(source, target_weight=target,
                                               first_attempt=False, had_t1_lock=True)
            log.reports[-1].status = "partial_pending"
            log.reports[-1].remaining_shares = after.shares - target_shares
            log.events[-1]["status"] = "partial_pending"
        else:
            pending_targets.pop(sym, None)
        if (source is not None and source.action == ACT_DECAY_REDUCE and
                sym not in pending_targets and self.state_machine is not None):
            self.state_machine.on_confirmed_reduce(sym)

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
        if self.state_machine is not None and sym not in self.account.positions:
            self.state_machine.on_confirmed_exit(sym, ts, log.strategy_cause)

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
