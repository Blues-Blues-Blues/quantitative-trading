"""事件驱动型回测撮合引擎单元测试（集成信号层 + 引擎层，合成 mock 数据）。

覆盖：
- 下一 Bar 开盘价成交（执行延迟，防未来）
- T+1：当日买入不可卖、SELL 挂起次日开盘强制卖出
- 涨跌停拦截（盘中触及不可买/卖）
- 先到先得资金分配（现金不足拒绝）与杠杆/单股上限风控
- 动态仓位公式 Position_Size 与成本/滑点模型
- 100 股整数倍取整、成交日志与净值曲线完整性

时间线约定（与 tests/test_signals.py 一致）：
    D0(01-02) → D1(01-03) → D2(01-04)；
    D2 10:00 首个 BUY 信号（RS/IMS 转正），引擎成交于 D2 10:30 开盘价。
    T+1 相关测试使用 _D4（含 D3=01-05）：D2 买入当日锁定，
    D3 解冻后可验证挂起卖出 / 跌停拦截 / 完整买卖循环。
"""

import numpy as np
import pandas as pd
import pytest

from data.dataslice import DataSlice
from engine.backtest import BacktestEngine
from engine.execution import ExecutionCost
from engine.portfolio import Account
from engine.risk_control import PositionSizer
from strategy.signals import (
    ACT_ADD, ACT_BUY, ACT_DECAY_REDUCE, ACT_SELL, Signal,
    SignalSynthesizer, TradingStateMachine,
)

# ----------------------------------------------------------------------
# Mock 数据构造（与 tests/test_signals.py 一致）
# ----------------------------------------------------------------------

_D3 = ["2024-01-02", "2024-01-03", "2024-01-04"]
_D4 = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
_MAPPING = {"600000": "银行"}


def cn_minutes(dates, freq: str = "30min") -> pd.DatetimeIndex:
    parts = []
    for d in pd.to_datetime(dates):
        parts.append(pd.date_range(f"{d:%Y-%m-%d} 09:30", f"{d:%Y-%m-%d} 11:30", freq=freq))
        parts.append(pd.date_range(f"{d:%Y-%m-%d} 13:00", f"{d:%Y-%m-%d} 15:00", freq=freq))
    return pd.DatetimeIndex(np.concatenate([p.values for p in parts])).sort_values()


def mk_kline(axis, symbols=("600000",), day_ret=0.01, vwap=10.0, amount=1e8):
    days = pd.DatetimeIndex(sorted(set(axis.normalize())))
    day_idx = {d: i for i, d in enumerate(days)}
    rows = []
    for t in axis:
        for s in symbols:
            base = 10.0 * (1 + day_ret * day_idx[t.normalize()])
            rows.append({
                "symbol": s, "open": base * 0.99, "high": base * 1.01,
                "low": base * 0.99, "close": base, "volume": 1e6,
                "amount": amount, "vwap": vwap, "float_market_cap": 1e9,
                "up_limit": base * 1.1, "down_limit": base * 0.9, "is_st": False,
            })
    df = pd.DataFrame(rows)
    df.index = axis.repeat(len(symbols))
    df.index.name = "ts"
    return df


def mk_index(axis, ma20=3050.0, ma60=3000.0, idx_close=3010.0, vwap=3000.0):
    df = pd.DataFrame(index=axis)
    df["index_code"] = "000300.SH"
    df["open"] = idx_close * 0.99
    df["high"] = idx_close * 1.01
    df["low"] = idx_close * 0.99
    df["close"] = idx_close
    df["volume"] = 1e7
    df["vwap"] = vwap
    df["ma20"] = ma20
    df["ma60"] = ma60
    return df


def mk_breadth(axis, adr=2.0):
    df = pd.DataFrame(index=axis)
    df["advancers"] = 3000.0
    df["decliners"] = 1500.0
    df["adr"] = adr
    return df


def mk_industry(axis, flow_start=1e8, flow_growth=0.05):
    days = pd.DatetimeIndex(sorted(set(axis.normalize())))
    day_idx = {d: i for i, d in enumerate(days)}
    df = pd.DataFrame(index=axis)
    df["industry"] = "银行"
    df["open"] = df["high"] = df["low"] = df["close"] = 1000.0
    df["volume"] = 1e6
    df["money_flow"] = [flow_start * (1 + flow_growth * day_idx[t.normalize()])
                        for t in axis]
    return df


_BULL_BASE = dict(
    ofss=0.5, cps=0.3, inst_flow=1e8, north_sync=0.2,
    retail_flow=-1e5, youzi_flow=-5e5, chain_mod=0.05,
    global_mod=0.1, mrs=0.1, irs=0.1, grs=0.2, lock_ratio=0.8,
)


def mk_features(axis, symbols=("600000",), base=None, overrides=None):
    merged = dict(_BULL_BASE)
    merged.update(base or {})
    rows = []
    for t in axis:
        for s in symbols:
            rows.append({"ts": t, "symbol": s, **merged})
    df = pd.DataFrame(rows)
    if overrides:
        for ts, upd in overrides.items():
            ts = pd.Timestamp(ts)
            for c, v in upd.items():
                df.loc[df["ts"] == ts, c] = v
    return df.set_index("ts")


def full_env(dates=_D3, symbols=("600000",), base=None, overrides=None,
             kline_kw=None, mapping=_MAPPING):
    """牛市环境：ds + 特征长表 + 状态机生成的信号。"""
    axis = cn_minutes(dates)
    ds = DataSlice(
        kline=mk_kline(axis, symbols=symbols, **(kline_kw or {})),
        index_min=mk_index(axis),
        breadth=mk_breadth(axis),
        industry=mk_industry(axis),
        meta={"symbols": list(symbols)},
    )
    features = mk_features(axis, symbols=symbols, base=base, overrides=overrides)
    syn = SignalSynthesizer(symbol_to_industry=dict(mapping))
    sm = TradingStateMachine(synthesizer=syn)
    signals = sm.run(ds, features)
    return ds, signals


def run_backtest(ds, signals, account=None, cost=None, sizer=None):
    account = account or Account(initial_cash=1e8)
    cost = cost or ExecutionCost()
    sizer = sizer or PositionSizer()
    eng = BacktestEngine(account, cost, sizer, ds, signals)
    return eng, eng.run()


# ----------------------------------------------------------------------
# 执行延迟：下一 Bar 开盘价成交
# ----------------------------------------------------------------------

class TestNextBarFill:

    def test_buy_filled_at_next_bar_open(self):
        ds, signals = full_env()
        eng, (log, curve) = run_backtest(ds, signals)
        buys = log[log["reason"] == "filled"]
        buys_buy = buys[buys["side"] == ACT_BUY]
        # 首个 BUY 信号在 D0 10:00，成交于下一 Bar 10:30 的 open
        assert not buys_buy.empty
        first = buys_buy.iloc[0]
        assert first["ts"] == pd.Timestamp("2024-01-02 10:30")
        # 成交价 = open × (1 + 滑点)，滑点 > 0
        open_price = 10.0 * 0.99  # D0 的 open
        assert first["price"] > open_price
        assert first["shares"] % 100 == 0  # 一手取整

    def test_shares_round_to_100(self):
        ds, signals = full_env()
        eng, (log, _) = run_backtest(ds, signals)
        filled = log[log["reason"] == "filled"]
        assert (filled["shares"] % 100 == 0).all()

    def test_signal_unsorted_raises(self):
        ds, signals = full_env()
        rev = sorted(signals, key=lambda s: s.timestamp, reverse=True)
        with pytest.raises(ValueError):
            BacktestEngine(Account(1e8), ExecutionCost(), PositionSizer(), ds, rev)


# ----------------------------------------------------------------------
# T+1 机制
# ----------------------------------------------------------------------

class TestTPlusOne:

    def test_t1_lock_then_next_day_sell(self):
        # D0 10:00 BUY（成交 10:30 当日）；D0 10:30 SELL 信号 → 当日锁仓挂起，
        # 次日（D1）开盘强制卖出；D1 起链式因子持续为负 → 不再重新入场。
        # 注：新架构下需纯度转负（retail/youzi 转正 + inst 转负）才能使 XS<=0。
        weak = {"chain_mod": -1.0, "inst_flow": -1e5,
                "retail_flow": 1e8, "youzi_flow": 1e5}
        ov = {}
        # D0 10:30 起至最后一日全天弱化：SELL 后不再重新入场
        for d in _D4:
            for t in cn_minutes([d]):
                t = pd.Timestamp(t)
                if t >= pd.Timestamp("2024-01-02 10:30"):
                    ov[f"{t:%Y-%m-%d %H:%M}"] = weak
        ds, signals = full_env(dates=_D4, overrides=ov)
        eng, (log, curve) = run_backtest(ds, signals)

        t1_rej = log[(log["reason"] == "t1_lock")]
        assert not t1_rej.empty  # 当日 SELL 被拒（T+1 锁定）

        deferred = log[log["reason"] == "t1_deferred_sell"]
        assert not deferred.empty
        # 挂起卖出成交于次日（D1）开盘
        assert deferred.iloc[0]["ts"] == pd.Timestamp("2024-01-03 09:30")
        assert deferred.iloc[0]["shares"] > 0

        # 次日卖出后不再持仓
        end = curve.iloc[-1]
        assert end["n_positions"] == 0

    def test_account_t1_lock(self):
        acct = Account(initial_cash=1e8)
        acct.buy("600000", pd.Timestamp("2024-01-04 10:30"), 10.0, 10000, 100000.0)
        assert acct.positions["600000"].sellable_shares == 0  # 当日不可卖
        with pytest.raises(AssertionError):
            acct.sell("600000", pd.Timestamp("2024-01-04 11:00"), 10.0, 100, 1000.0)
        acct.roll_to_date(pd.Timestamp("2024-01-05 09:30"))
        assert acct.positions["600000"].sellable_shares == 10000  # 次日解冻
        acct.sell("600000", pd.Timestamp("2024-01-05 09:30"), 10.0, 10000, 99000.0)
        assert "600000" not in acct.positions

    def test_account_buy_non_100_lot_raises(self):
        acct = Account(initial_cash=1e8)
        with pytest.raises(AssertionError):
            acct.buy("600000", pd.Timestamp("2024-01-04 10:30"), 10.0, 150, 1500.0)


# ----------------------------------------------------------------------
# 涨跌停拦截（盘中触及）
# ----------------------------------------------------------------------

class TestLimit:

    def test_limit_up_blocks_buy(self):
        ds, signals = full_env()
        # 使 BUY 撮合 Bar（D0 10:30）盘中触及涨停 → 拒绝买入
        ts = pd.Timestamp("2024-01-02 10:30")
        ds.kline.loc[ds.kline.index == ts, "high"] = \
            ds.kline.loc[ds.kline.index == ts, "up_limit"]
        eng, (log, _) = run_backtest(ds, signals)

        rej = log[(log["side"] == ACT_BUY) & (log["reason"] == "limit_up")]
        assert not rej.empty
        assert rej.iloc[0]["ts"] == ts  # 10:30 撮合 Bar 被拒
        # 引擎开板后会在后续正常 Bar 重新买入（13:30 ADD 信号 → 14:00 成交），
        # 故只断言"存在涨停拒绝记录"，不断言"无持仓"
        assert "600000" in eng.account.positions

    def test_limit_down_blocks_sell(self):
        # D1 10:00 SELL 信号（D0 买入已解冻）→ 撮合于 10:30，该 Bar 盘中触及跌停 → 拒绝卖出
        weak = {"chain_mod": -1.0, "inst_flow": -1e5,
                "retail_flow": 1e8, "youzi_flow": 1e5}
        ds, signals = full_env(dates=_D4,
                               overrides={"2024-01-03 10:00": weak})
        ts = pd.Timestamp("2024-01-03 10:30")
        ds.kline.loc[ds.kline.index == ts, "low"] = \
            ds.kline.loc[ds.kline.index == ts, "down_limit"]
        eng, (log, _) = run_backtest(ds, signals)

        rej = log[(log["side"] == ACT_SELL) & (log["reason"] == "limit_down")]
        assert not rej.empty
        # 卖出被拒 → 仍持有
        assert "600000" in eng.account.positions


# ----------------------------------------------------------------------
# 资金分配与风控
# ----------------------------------------------------------------------

class TestRiskControl:

    def test_insufficient_cash_fifo(self):
        # 4 标的同时 BUY（目标权重各 30%），现金只够前 3 只 → 第 4 只拒绝（先到先得）
        syms = ("600000", "000001", "600036", "601988")
        mapping = {s: "银行" for s in syms}
        ds, _ = full_env(symbols=syms, mapping=mapping)
        sizer = PositionSizer(base_position=0.3, max_single_position=0.3,
                              max_leverage=2.0)
        account = Account(initial_cash=1e8, max_leverage=2.0)
        ts = pd.Timestamp("2024-01-02 10:00")
        sigs = [Signal(s, ts, ACT_BUY, "S_push", {"target_weight": 0.3})
                for s in syms]
        eng, (log, _) = run_backtest(ds, sigs, account=account, sizer=sizer)

        rejected = log[(log["reason"] == "insufficient_cash")]
        assert not rejected.empty
        assert rejected["symbol"].tolist() == ["601988"]  # 最后一个被拒

    def test_leverage_cap(self):
        # 4 标的同时 BUY（目标各 30%），无杠杆账户：第 4 只触发总杠杆上限
        syms = ("600000", "000001", "600036", "601988")
        mapping = {s: "银行" for s in syms}
        ds, _ = full_env(symbols=syms, mapping=mapping)
        sizer = PositionSizer(base_position=0.3, max_single_position=0.3,
                              max_leverage=1.0)
        account = Account(initial_cash=1e8, max_leverage=1.0)
        ts = pd.Timestamp("2024-01-02 10:00")
        sigs = [Signal(s, ts, ACT_BUY, "S_push", {"target_weight": 0.3})
                for s in syms]
        eng, (log, _) = run_backtest(ds, sigs, account=account, sizer=sizer)

        rejected = log[(log["reason"] == "leverage_cap")]
        assert not rejected.empty

    def test_single_position_cap(self):
        # 默认参数：base=0.2、mrs=0.1 → 目标比例 0.3465 → 被 clip 到单股上限 0.3
        ds, signals = full_env()
        eng, (log, _) = run_backtest(ds, signals)
        buys = log[(log["reason"] == "filled") & (log["side"] == ACT_BUY)]
        assert not buys.empty
        max_value = 0.3 * eng.account.initial_cash * 1.01  # 单股上限 + 滑点浮差
        assert (buys["amount"] <= max_value).all()


# ----------------------------------------------------------------------
# 动态仓位公式 / 成本与滑点（纯函数单测）
# ----------------------------------------------------------------------

class TestPositionSizer:

    def test_formula(self):
        sizer = PositionSizer(base_position=0.2, max_single_position=0.3,
                              mrs_sensitivity=10.0, mrs_coef_min=0.5, mrs_coef_max=1.5,
                              chain_scale_floor=0.5)
        # mrs=0.05 → coef=1.5；gmod=0.1 → 1.1；chain=0.05 → 1.05
        # 0.2*1.5*1.1*1.05 = 0.3465 → clip 到单股上限 0.3
        ratio = sizer.target_ratio({"mrs": 0.05, "global_mod": 0.1, "chain_mod": 0.05})
        assert ratio == pytest.approx(0.3)

    def test_mrs_coefficient_clip(self):
        sizer = PositionSizer()
        assert sizer.mrs_coefficient(0.1) == pytest.approx(1.5)   # 2.0 → clip 1.5
        assert sizer.mrs_coefficient(-0.05) == pytest.approx(0.5)  # 0.5
        assert sizer.mrs_coefficient(None) == pytest.approx(1.0)   # 缺失中性

    def test_chain_scale_floor(self):
        sizer = PositionSizer(chain_scale_floor=0.5)
        assert sizer.chain_scale(-1.0) == pytest.approx(0.5)  # 触底下限
        assert sizer.chain_scale(0.3) == pytest.approx(1.3)

    def test_target_value(self):
        sizer = PositionSizer(base_position=0.2, max_single_position=0.3)
        v = sizer.target_value(1e8, {"mrs": None, "global_mod": None, "chain_mod": None})
        assert v == pytest.approx(0.2 * 1e8)  # 全中性 → 基准仓位


class TestExecutionCost:

    def test_commission_min(self):
        cost = ExecutionCost(commission_rate=0.0002, min_commission=5.0)
        comm_big, _ = cost.buy_fees(100000.0)
        assert comm_big == pytest.approx(20.0)     # 100000*0.0002
        comm_small, _ = cost.buy_fees(10000.0)
        assert comm_small == pytest.approx(5.0)    # 2 < 最低 5

    def test_stamp_duty_sell_only(self):
        cost = ExecutionCost(stamp_duty_rate=0.0005)
        _, stamp, _ = cost.sell_fees(100000.0)
        assert stamp == pytest.approx(50.0)      # 印花税仅卖出收取
        commission, transfer = cost.buy_fees(100000.0)
        assert commission == pytest.approx(20.0)  # 佣金
        assert transfer == pytest.approx(1.0)     # 过户费万分之一

    def test_dynamic_slippage(self):
        cost = ExecutionCost(fixed_slippage_bps=2.0, slippage_coef_bps=50.0,
                             slippage_cap_bps=60.0)
        assert cost.slippage_bps(1e6, 1e8) == pytest.approx(2.5)   # 参与率 1%
        assert cost.slippage_bps(5e7, 1e8) == pytest.approx(27.0)  # 参与率 50%
        # 参与率 clip 到 1.0 → 52bp（2 固定 + 50 冲击），封顶 60 兜底
        assert cost.slippage_bps(2e8, 1e8) == pytest.approx(52.0)
        assert cost.slippage_bps(5e7, 1e8) <= 60.0
        # 买入加滑点、卖出减滑点
        assert cost.buy_price(10.0, 1e6, 1e8) > 10.0
        assert cost.sell_price(10.0, 1e6, 1e8) < 10.0

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            ExecutionCost(commission_rate=0.05)
        with pytest.raises(ValueError):
            ExecutionCost(slippage_cap_bps=1.0)  # 小于固定滑点


# ----------------------------------------------------------------------
# 输出完整性
# ----------------------------------------------------------------------

class TestOutputs:

    def test_equity_curve_every_bar(self):
        ds, signals = full_env()
        axis = cn_minutes(_D3)
        eng, (log, curve) = run_backtest(ds, signals)
        assert len(curve) == len(axis)                      # 每 Bar 一条
        assert curve["total_equity"].iloc[0] == pytest.approx(1e8)
        # 恒等式：总权益 = 现金 + 持仓市值
        for _, r in curve.iterrows():
            assert r["total_equity"] == pytest.approx(
                r["cash"] + r["position_value"], abs=1e-6)

    def test_trade_log_columns_and_ordering(self):
        ds, signals = full_env()
        eng, (log, _) = run_backtest(ds, signals)
        assert list(log.columns)[:4] == ["ts", "symbol", "side", "price"]
        assert log["ts"].is_monotonic_increasing

    def test_add_fills_to_target(self):
        # min_add_interval=2 → D0 10:00 BUY、11:00 ADD 信号；
        # 11:00 特征增强（mrs/global_mod 抬升）→ target_weight 显著提高，
        # ADD 差额调仓（Δ > 死区）成交于 11:30
        axis = cn_minutes(_D3)
        ds = DataSlice(kline=mk_kline(axis), index_min=mk_index(axis),
                       breadth=mk_breadth(axis), industry=mk_industry(axis),
                       meta={"symbols": ["600000"]})
        features = mk_features(axis, overrides={
            "2024-01-02 11:00": {"mrs": 10.0, "global_mod": 1.0}})
        syn = SignalSynthesizer(symbol_to_industry=_MAPPING)
        sm = TradingStateMachine(synthesizer=syn, min_add_interval=2)
        signals = sm.run(ds, features)
        eng, (log, _) = run_backtest(ds, signals)
        adds = log[(log["side"] == ACT_ADD) & (log["reason"] == "filled")]
        assert not adds.empty
        assert adds.iloc[0]["ts"] == pd.Timestamp("2024-01-02 11:30")

    def test_no_signals_flat_curve(self):
        ds, _ = full_env()
        eng, (log, curve) = run_backtest(ds, [])
        assert log.empty
        assert np.allclose(curve["total_equity"].to_numpy(), 1e8, rtol=1e-9)
        assert (curve["n_positions"] == 0).all()

    def test_full_buy_sell_cycle_reconciles(self):
        # D0 10:00 BUY（成交 10:30）；D1 14:30 SELL 信号（次日已解冻）→ 15:00 卖出清仓；
        # D2/D3 全天弱化 → 不再重新入场。
        # 新架构下需纯度转负（retail/youzi 转正 + inst 转负）才能使 XS<=0。
        weak = {"chain_mod": -1.0, "inst_flow": -1e5,
                "retail_flow": 1e8, "youzi_flow": 1e5}
        ov = {"2024-01-03 14:30": weak}
        for d in ("2024-01-04", "2024-01-05"):
            for t in cn_minutes([d]):
                ov[f"{t:%Y-%m-%d %H:%M}"] = weak
        ds, signals = full_env(dates=_D4, overrides=ov)
        eng, (log, curve) = run_backtest(ds, signals)
        sells = log[(log["side"] == ACT_SELL) & (log["reason"] == "signal_sell")]
        assert not sells.empty
        assert sells.iloc[0]["ts"] == pd.Timestamp("2024-01-03 15:00")
        assert eng.account.positions == {}
        # 最终权益 ≈ 初始权益（成本拖累，误差 < 0.5%）
        assert eng.account.total_equity == pytest.approx(1e8, rel=0.005)


# ----------------------------------------------------------------------
# 调仓死区（Deadzone Filter）
# ----------------------------------------------------------------------

class TestDeadzone:

    def test_skips_small_delta_on_holding(self):
        # 已持仓 ≈0.20，ADD 目标 0.21（|Δ|=0.01 < 死区 0.05）→ 跳过，无成交
        ds, _ = full_env()
        ts = pd.Timestamp("2024-01-02 10:00")
        buy = Signal("600000", ts, ACT_BUY, "S_push", {"target_weight": 0.2})
        add = Signal("600000", pd.Timestamp("2024-01-03 10:00"), ACT_ADD,
                     "S_push", {"target_weight": 0.21})
        eng, (log, _) = run_backtest(ds, [buy, add])
        filled = log[(log["reason"] == "filled")]
        assert (filled["side"] == ACT_ADD).sum() == 0  # 死区跳过 ADD
        assert (filled["side"] == ACT_BUY).sum() == 1  # 建仓正常成交

    def test_exempt_on_new_position(self):
        # 从 0 建仓豁免死区：小目标权重（0.02 < 死区）也执行
        ds, _ = full_env()
        buy = Signal("600000", pd.Timestamp("2024-01-02 10:00"), ACT_BUY,
                     "S_push", {"target_weight": 0.02})
        eng, (log, _) = run_backtest(ds, [buy])
        filled = log[(log["reason"] == "filled")]
        assert not filled.empty
        assert filled.iloc[0]["shares"] > 0

    def test_deadzone_param_validation(self):
        ds, _ = full_env()
        with pytest.raises(ValueError):
            BacktestEngine(Account(1e8), ExecutionCost(), PositionSizer(), ds, [],
                           deadzone_th=-0.1)
        with pytest.raises(ValueError):
            BacktestEngine(Account(1e8), ExecutionCost(), PositionSizer(), ds, [],
                           deadzone_th=1.5)


# ----------------------------------------------------------------------
# 差额调仓：T+1 顺延 / 跌停跳过
# ----------------------------------------------------------------------

class TestTargetRebalanceT1:

    def test_sell_t1_deferral_carries_over(self):
        # D0 10:00 BUY（成交 10:30，当日锁定）；11:00 SELL → T+1 拒绝且顺延；
        # 次日 D1 解冻后按目标 0 卖出全部（reason t1_deferred_sell）
        ds, _ = full_env(dates=_D4)
        buy = Signal("600000", pd.Timestamp("2024-01-02 10:00"), ACT_BUY,
                     "S_push", {"target_weight": 0.3})
        sell = Signal("600000", pd.Timestamp("2024-01-02 11:00"), ACT_SELL,
                      "S_push", {})
        eng, (log, _) = run_backtest(ds, [buy, sell])
        assert not log[(log["reason"] == "t1_lock")].empty
        deferred = log[(log["reason"] == "t1_deferred_sell")]
        assert not deferred.empty
        assert deferred.iloc[0]["ts"] == pd.Timestamp("2024-01-03 09:30")
        assert deferred.iloc[0]["shares"] > 0
        assert eng.account.positions == {}

    def test_reduce_t1_deferral_partial(self):
        # D0 10:00 BUY（成交 10:30）；当日 11:00 DECAY_REDUCE（目标=当前×0.8）
        # → 可卖 0 顺延；次日解冻后减到目标（非清仓）
        ds, _ = full_env(dates=_D4)
        buy = Signal("600000", pd.Timestamp("2024-01-02 10:00"), ACT_BUY,
                     "S_push", {"target_weight": 0.3})
        reduce_ = Signal("600000", pd.Timestamp("2024-01-02 11:00"),
                         ACT_DECAY_REDUCE, "S_push",
                         {"target_weight": 0.24, "reduce_fraction": 0.5})
        eng, (log, _) = run_backtest(ds, [buy, reduce_])
        assert not log[(log["reason"] == "t1_lock")].empty
        deferred = log[(log["reason"] == "t1_deferred_sell")]
        assert not deferred.empty
        assert eng.account.positions["600000"].shares > 0  # 减仓非清仓

    def test_limit_down_skips_no_pending(self):
        # D0 10:00 BUY（成交 10:30）；D1 10:00 SELL 撮合于 10:30（跌停）
        # → 拒绝且不挂起；无顺延卖出记录，持仓保留
        ds, _ = full_env(dates=_D4)
        buy = Signal("600000", pd.Timestamp("2024-01-02 10:00"), ACT_BUY,
                     "S_push", {"target_weight": 0.3})
        sell = Signal("600000", pd.Timestamp("2024-01-03 10:00"), ACT_SELL,
                      "S_push", {})
        ts_dl = pd.Timestamp("2024-01-03 10:30")
        ds.kline.loc[ds.kline.index == ts_dl, "low"] = \
            ds.kline.loc[ds.kline.index == ts_dl, "down_limit"]
        eng, (log, _) = run_backtest(ds, [buy, sell])
        assert not log[(log["side"] == ACT_SELL) &
                       (log["reason"] == "limit_down")].empty
        assert log[(log["reason"] == "t1_deferred_sell")].empty  # 不挂起
        assert "600000" in eng.account.positions
