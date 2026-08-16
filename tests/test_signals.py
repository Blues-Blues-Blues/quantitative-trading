"""信号合成公式与多层闸门交易状态机单元测试（合成 mock 数据，不联网）。

覆盖：
- 合成公式精确值（Agent_MS / Final_MS / Capital_Purity）与权重和==1 断言
- NaN 分量权重重归一化、Retail_Chase 滚动
- RS / Industry_MS 日频 T-1 可见性（防未来）
- 8 层开仓闸门逐项阻断、6 层平仓闸门逐项触发
- 状态机：BUY / ADD / SELL / HOLD 全流程与内部持仓演进
- 时间升序断言与逐 Bar 输出

时间线约定（RS/IMS 为日频 T-1 对齐）：
    D0(01-02)  → RS/IMS NaN（无更早数据）
    D1(01-03)  → RS/IMS == 中性 0（D0 无涨跌 → alpha 闸门不通过）
    D2(01-04)  → RS/IMS > 0（D1 起有正相对收益）→ 首个 BUY 于 D2 10:00
"""

import numpy as np
import pandas as pd
import pytest

from data.dataslice import DataSlice
from strategy.signals import (
    ACT_ADD, ACT_BUY, ACT_HOLD, ACT_SELL,
    S_NOISE, S_PUSH, S_YOUZI_ONLY,
    Position, SignalSynthesizer, TradingStateMachine,
)


# ----------------------------------------------------------------------
# Mock 数据构造
# ----------------------------------------------------------------------

def cn_minutes(dates, freq: str = "30min") -> pd.DatetimeIndex:
    parts = []
    for d in pd.to_datetime(dates):
        parts.append(pd.date_range(f"{d:%Y-%m-%d} 09:30", f"{d:%Y-%m-%d} 11:30", freq=freq))
        parts.append(pd.date_range(f"{d:%Y-%m-%d} 13:00", f"{d:%Y-%m-%d} 15:00", freq=freq))
    return pd.DatetimeIndex(np.concatenate([p.values for p in parts])).sort_values()


def mk_kline(axis, symbols=("600000",), day_ret=0.01, vwap=10.0, amount=1e8,
             close_override=None):
    """K 线：close 逐日抬升（构造 RS>0）；vwap 恒定（便于止损测试）。"""
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
    if close_override:
        for ts, v in close_override.items():
            df.loc[df.index == pd.Timestamp(ts), "close"] = v
    return df


def mk_index(axis, ma20=3050.0, ma60=3000.0, idx_close=3010.0, vwap=3000.0,
             idx_close_override=None):
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
    if idx_close_override:
        for ts, v in idx_close_override.items():
            df.loc[pd.Timestamp(ts), "close"] = v
    return df


def mk_breadth(axis, adr=2.0):
    df = pd.DataFrame(index=axis)
    df["advancers"] = 3000.0
    df["decliners"] = 1500.0
    df["adr"] = adr
    return df


def mk_industry(axis, flow_start=1e8, flow_growth=0.05):
    """行业资金流逐日抬升（构造 Industry_MS>0）。"""
    days = pd.DatetimeIndex(sorted(set(axis.normalize())))
    day_idx = {d: i for i, d in enumerate(days)}
    df = pd.DataFrame(index=axis)
    df["industry"] = "银行"
    df["open"] = df["high"] = df["low"] = df["close"] = 1000.0
    df["volume"] = 1e6
    df["money_flow"] = [flow_start * (1 + flow_growth * day_idx[t.normalize()])
                        for t in axis]
    return df


# 牛市全通过基准特征（S_push 且 8 层闸门全过）
_BULL_BASE = dict(
    ofss=0.5, cps=0.3, inst_flow=1e8, north_sync=0.2,
    retail_flow=-1e5, youzi_flow=-5e5, chain_mod=0.05,
    global_mod=0.1, mrs=0.1, irs=0.1, grs=0.2, lock_ratio=0.8,
)

_MAPPING = {"600000": "银行"}


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


def full_env(dates):
    """牛市环境：ds + 特征长表。"""
    axis = cn_minutes(dates)
    ds = DataSlice(
        kline=mk_kline(axis),
        index_min=mk_index(axis),
        breadth=mk_breadth(axis),
        industry=mk_industry(axis),
        meta={"symbols": ["600000"]},
    )
    return ds, mk_features(axis)


def bull_syn(**kw):
    """牛市默认合成器（带行业映射，可开仓）。"""
    kw.setdefault("symbol_to_industry", _MAPPING)
    return SignalSynthesizer(**kw)


_D3 = ["2024-01-02", "2024-01-03", "2024-01-04"]
_T_BUY = "2024-01-04 10:00"   # 首个 BUY 时点


def bull_eval_row(syn=None, dates=_D3, at=_T_BUY):
    """取牛市中可开仓那一根评估行（D2 10:00）。"""
    syn = syn or bull_syn()
    ds, features = full_env(dates)
    ev = TradingStateMachine(synthesizer=syn)._build_eval_table(ds, features)
    return ev[ev["ts"] == pd.Timestamp(at)].iloc[0]


# ----------------------------------------------------------------------
# 合成公式
# ----------------------------------------------------------------------

class TestSignalSynthesizer:

    def test_weight_sum_must_be_one(self):
        with pytest.raises(ValueError):
            SignalSynthesizer(weights=(0.5, 0.5, 0.5, 0.5))

    def test_agent_ms_formula(self):
        axis = cn_minutes(["2024-01-02"])
        syn = SignalSynthesizer(weights=(0.35, 0.25, 0.25, 0.15))
        f = mk_features(axis)
        agent = syn._agent_ms(f).iloc[0]
        # 0.35*0.5 + 0.25*0.3 + 0.25*1 + 0.15*0.2
        assert agent == pytest.approx(0.53)

    def test_final_ms_formula(self):
        syn = SignalSynthesizer()
        out = syn.final_ms(pd.Series([0.53]), pd.Series([0.05]), pd.Series([0.1]))
        assert out.iloc[0] == pytest.approx((0.53 + 0.05) * 1.1)

    def test_capital_purity_formula(self):
        axis = cn_minutes(["2024-01-02"])
        f = mk_features(axis)  # inst>0, north=0.2, retail<0, youzi<0
        purity = SignalSynthesizer().capital_purity(f).iloc[0]
        assert purity == pytest.approx(0.35 + 0.25 * 0.2 + 0.2 + 0.2)

    def test_agent_ms_nan_renormalize(self):
        axis = cn_minutes(["2024-01-02"])
        syn = SignalSynthesizer(weights=(0.35, 0.25, 0.25, 0.15))
        f = mk_features(axis, overrides={"2024-01-02 09:30": {"ofss": np.nan}})
        agent = syn._agent_ms(f).iloc[0]
        # ofss 缺失 → 剩余权重 (0.25,0.25,0.15) 和 0.65 重归一化
        expect = (0.25 * 0.3 + 0.25 * 1 + 0.15 * 0.2) / 0.65
        assert agent == pytest.approx(expect)

    def test_all_nan_agent_ms_is_nan(self):
        axis = cn_minutes(["2024-01-02"])
        syn = SignalSynthesizer()
        f = mk_features(axis, overrides={"2024-01-02 09:30": {
            "ofss": np.nan, "cps": np.nan, "inst_flow": np.nan, "north_sync": np.nan}})
        assert np.isnan(syn._agent_ms(f).iloc[0])

    def test_retail_chase_clip(self):
        axis = cn_minutes(["2024-01-02"])
        syn = SignalSynthesizer(chase_window=5)
        f = mk_features(axis)  # retail_flow=-1e5 占绝对量很小 → 追涨度 0
        assert syn._retail_chase(f).iloc[0] == pytest.approx(0.0)

    def test_rs_industry_ms_t_minus_1_visibility(self):
        """日频 RS / Industry_MS：D0 不可见(NaN)、D1 中性 0、D2 起为正。"""
        ds, features = full_env(_D3)
        syn = bull_syn()
        ev = TradingStateMachine(synthesizer=syn)._build_eval_table(ds, features)
        d0, d1, d2 = (pd.Timestamp(x) for x in ("2024-01-02", "2024-01-03", "2024-01-04"))
        assert ev.loc[ev["ts"].dt.normalize() == d0, "rs"].isna().all()
        assert ev.loc[ev["ts"].dt.normalize() == d0, "industry_ms"].isna().all()
        assert (ev.loc[ev["ts"].dt.normalize() == d1, "rs"] == 0.0).all()
        assert (ev.loc[ev["ts"].dt.normalize() == d1, "industry_ms"] == 0.0).all()
        assert (ev.loc[ev["ts"].dt.normalize() == d2, "rs"] > 0).all()
        assert (ev.loc[ev["ts"].dt.normalize() == d2, "industry_ms"] > 0).all()


# ----------------------------------------------------------------------
# 闸门
# ----------------------------------------------------------------------

class TestGates:

    def _row(self, **mut):
        syn = bull_syn()
        row = bull_eval_row(syn).copy()
        for k, v in mut.items():
            row[k] = v
        return syn, row

    def test_all_entry_gates_pass_in_bull(self):
        syn, row = self._row()
        gates = syn.entry_gates(row)
        assert all(gates.values()), gates

    @pytest.mark.parametrize("mut,gate", [
        ({"global_mod": -1.0}, "global"),          # ① 全球层
        ({"ma20": 2900.0}, "system"),              # ② MA20 < MA60
        ({"adr": 0.5}, "system"),                  # ② ADR 不足
        ({"mrs": -0.5}, "beta"),                   # ③ 系统层
        ({"irs": -0.5}, "industry"),               # ④ 产业层
        ({"rs": -0.1}, "alpha"),                   # ⑤ 个股 RS
        ({"industry_ms": -0.1}, "alpha"),          # ⑤ 行业情绪
        ({"final_ms": -1.0}, "stock"),             # ⑥ Final_MS
        ({"lock_ratio": 0.1}, "stock"),            # ⑥ 锁仓比
        ({"capital_purity": -0.5}, "stock"),       # ⑥ 资金纯净度
        ({"amount": 1e5}, "liquidity"),            # ⑦ 成交额
        ({"is_st": True}, "liquidity"),            # ⑦ ST
        ({"close": 11.5}, "liquidity"),            # ⑦ 涨停价上沿 (D2 up_limit=11.22)
    ])
    def test_entry_gate_blocked(self, mut, gate):
        syn, row = self._row(**mut)
        gates = syn.entry_gates(row)
        assert gates[gate] is False
        assert syn.entry_all(row) is False

    def test_entry_time_window(self):
        syn = bull_syn()
        ds, features = full_env(_D3)
        ev = TradingStateMachine(synthesizer=syn)._build_eval_table(ds, features)
        early = ev[ev["ts"] == pd.Timestamp("2024-01-04 09:30")].iloc[0]
        late = ev[ev["ts"] == pd.Timestamp("2024-01-04 15:00")].iloc[0]
        assert syn.entry_gates(early)["time"] is False
        assert syn.entry_gates(late)["time"] is False

    def test_exit_triggers(self):
        syn = bull_syn(th_slippage=0.03, win_hold_max=10)
        row = bull_eval_row(syn).copy()
        pos = Position(symbol="600000", entry_time=row["ts"], entry_vwap=10.0)

        # ① 状态转变
        row["state"] = S_NOISE
        assert syn.exit_triggers(row, pos)["state"] is True
        row["state"] = S_PUSH
        # ② Final_MS 破位
        row["final_ms"] = -0.5
        assert syn.exit_triggers(row, pos)["ms"] is True
        row["final_ms"] = 0.5
        # ③ 资金纯净度转负
        row["capital_purity"] = -0.1
        assert syn.exit_triggers(row, pos)["purity"] is True
        row["capital_purity"] = 0.5
        # ④ 跌破入场 VWAP 止损
        row["close"] = 9.6
        assert syn.exit_triggers(row, pos)["stop"] is True
        row["close"] = 10.0
        # ⑤ 持仓超时
        pos.bars_held = 11
        assert syn.exit_triggers(row, pos)["hold_time"] is True
        pos.bars_held = 1
        # ⑥ 指数熔断 / 全球熔断
        row["index_close"] = 2900.0
        assert syn.exit_triggers(row, pos)["circuit"] is True
        row["index_close"] = 3000.0
        row["grs"] = -2.0
        assert syn.exit_triggers(row, pos)["circuit"] is True


# ----------------------------------------------------------------------
# 状态机
# ----------------------------------------------------------------------

class TestStateMachine:

    def _run(self, syn=None, sm_kw=None, dates=_D3, overrides=None, base=None):
        ds, features = full_env(dates)
        features = mk_features(features.index, base=base, overrides=overrides)
        syn = syn or bull_syn()
        sm = TradingStateMachine(synthesizer=syn, **(sm_kw or {}))
        return sm, ds, features

    def _first_buy(self, signals):
        buys = [s for s in signals if s.action == ACT_BUY]
        assert buys, "未产生 BUY 信号"
        return min(buys, key=lambda s: s.timestamp)

    def test_buy_on_push_all_gates(self):
        sm, ds, features = self._run()
        sigs = sm.run(ds, features)
        first = self._first_buy(sigs)
        # D0/D1 的 RS/IMS 未转正、D2 09:30 在时间窗外 → 首个 BUY 于 D2 10:00
        assert first.timestamp == pd.Timestamp(_T_BUY)
        assert first.action == ACT_BUY and first.state == S_PUSH
        # 未触发平仓 → 状态机内部仍持有该标的
        assert "600000" in sm.positions
        assert sm.positions["600000"].entry_time == pd.Timestamp(_T_BUY)

    def test_every_bar_outputs_and_ordered(self):
        sm, ds, features = self._run()
        sigs = sm.run(ds, features)
        axis = cn_minutes(_D3)
        assert len(sigs) == len(axis)  # 每根 Bar 一个信号
        ts = [s.timestamp for s in sigs]
        assert ts == sorted(ts)
        frame = TradingStateMachine.to_frame(sigs)
        assert list(frame.columns)[:4] == ["timestamp", "symbol", "action", "state"]

    def test_youzi_only_blocks_entry(self):
        # 全程游资主导 + 散户盲从 → 状态恒为 S_youzi_only，绝不开仓
        base = {"youzi_flow": 1e5, "inst_flow": -1e5, "retail_flow": 1e8}
        sm, ds, features = self._run(base=base)
        sigs = sm.run(ds, features)
        assert all(s.state == S_YOUZI_ONLY for s in sigs)
        assert all(s.action != ACT_BUY for s in sigs)

    def test_sell_on_state_change_to_youzi(self):
        # chase_window=1：retail_chase 仅反映当前 Bar，单根注入游资主导即可翻转状态
        sm, ds, features = self._run(
            syn=bull_syn(chase_window=1),
            overrides={"2024-01-04 10:30": {
                "youzi_flow": 1e5, "inst_flow": -1e5, "retail_flow": 1e8}})
        sigs = sm.run(ds, features)
        sell = next(s for s in sigs
                    if s.timestamp == pd.Timestamp("2024-01-04 10:30"))
        assert sell.action == ACT_SELL
        assert sell.state == S_YOUZI_ONLY
        assert sell.metrics["exit_triggers"]["state"] is True

    def test_sell_on_final_ms_exit(self):
        # final_ms 为派生列，直接 override 会被 synthesize 覆盖；
        # 改注入 chain_mod 使 Final_MS < 退出阈值
        sm, ds, features = self._run(overrides={
            "2024-01-04 10:30": {"chain_mod": -1.0}})
        sigs = sm.run(ds, features)
        sell = next(s for s in sigs
                    if s.timestamp == pd.Timestamp("2024-01-04 10:30"))
        assert sell.action == ACT_SELL
        assert sell.metrics["exit_triggers"]["ms"] is True

    def test_sell_on_stop_loss(self):
        ds, features = full_env(_D3)
        ds = DataSlice(
            kline=mk_kline(ds.kline.index,
                           close_override={"2024-01-04 10:30": 9.6}),
            index_min=ds.index_min, breadth=ds.breadth,
            industry=ds.industry, meta=ds.meta)
        sm = TradingStateMachine(bull_syn())
        sigs = sm.run(ds, features)
        sell = next(s for s in sigs
                    if s.timestamp == pd.Timestamp("2024-01-04 10:30"))
        assert sell.action == ACT_SELL
        assert sell.metrics["exit_triggers"]["stop"] is True

    def test_sell_on_hold_max(self):
        sm, ds, features = self._run(
            syn=bull_syn(win_hold_max=1), sm_kw={"min_add_interval": 999})
        sigs = sm.run(ds, features)
        # BUY 于 D2 10:00；11:00 时 bars_held=2 > 1 → SELL
        sell = next(s for s in sigs
                    if s.timestamp == pd.Timestamp("2024-01-04 11:00"))
        assert sell.action == ACT_SELL
        assert sell.metrics["exit_triggers"]["hold_time"] is True

    def test_sell_on_circuit_breaker(self):
        ds, features = full_env(_D3)
        ds = DataSlice(
            kline=ds.kline,
            index_min=mk_index(ds.kline.index,
                               idx_close_override={"2024-01-04 10:30": 2900.0}),
            breadth=ds.breadth, industry=ds.industry, meta=ds.meta)
        sm = TradingStateMachine(bull_syn())
        sigs = sm.run(ds, features)
        sell = next(s for s in sigs
                    if s.timestamp == pd.Timestamp("2024-01-04 10:30"))
        assert sell.action == ACT_SELL
        assert sell.metrics["exit_triggers"]["circuit"] is True

    def test_add_after_min_interval(self):
        sm, ds, features = self._run(sm_kw={"min_add_interval": 2})
        sigs = sm.run(ds, features)
        # 10:30 距上次加仓不足 2 → HOLD；11:00 bars_held=2 → ADD
        a = next(s for s in sigs if s.timestamp == pd.Timestamp("2024-01-04 10:30"))
        b = next(s for s in sigs if s.timestamp == pd.Timestamp("2024-01-04 11:00"))
        assert a.action == ACT_HOLD
        assert b.action == ACT_ADD

    def test_sell_after_add_resumes_position(self):
        sm, ds, features = self._run(overrides={
            "2024-01-04 11:30": {"chain_mod": -1.0}})
        sigs = sm.run(ds, features)
        sell = next(s for s in sigs
                    if s.timestamp == pd.Timestamp("2024-01-04 11:30"))
        assert sell.action == ACT_SELL

    def test_no_buy_without_industry_mapping(self):
        ds, features = full_env(_D3)
        sm = TradingStateMachine(SignalSynthesizer(symbol_to_industry={}))
        sigs = sm.run(ds, features)
        assert all(s.action != ACT_BUY for s in sigs)

    def test_buy_with_industry_mapping(self):
        sm, ds, features = self._run(syn=bull_syn())
        sigs = sm.run(ds, features)
        assert self._first_buy(sigs).action == ACT_BUY
