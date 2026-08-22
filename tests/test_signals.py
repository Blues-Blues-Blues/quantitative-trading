"""信号合成、连续评分（ES/PS/XS）与交易状态机单元测试（合成 mock 数据，不联网）。

覆盖：
- 合成公式精确值（Agent_MS / Final_MS / Capital_Purity）与权重和==1 断言
- NaN 分量权重重归一化、Retail_Chase 滚动
- RS / Industry_MS 日频 T-1 可见性（防未来）
- 连续评分：ES 有界化/归一化/游资衰减；PS 时间衰减/动量豁免/资金稳定；
  XS 回撤/一票否决（溃逃、跳水）
- A 股硬过滤层（ST / 涨跌停 / 时间窗 / 成交额）
- 状态机：BUY / ADD / SELL / DECAY_REDUCE / HOLD 全流程
- 历史兼容包装（entry_gates / exit_triggers）新语义

时间线约定（RS/IMS 为日频 T-1 对齐）：
    D0(01-02)  → RS/IMS NaN（无更早数据）
    D1(01-03)  → RS/IMS == 中性 0
    D2(01-04)  → RS/IMS > 0 → 首个 BUY 于 D2 10:00
"""

import numpy as np
import pandas as pd
import pytest

from data.dataslice import DataSlice
from strategy.signals import (
    ACT_ADD, ACT_BUY, ACT_DECAY_REDUCE, ACT_HOLD, ACT_SELL,
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


# 牛市全通过基准特征（ES 高 / XS 高 / S_push）
_BULL_BASE = dict(
    ofss=0.5, cps=0.3, inst_flow=1e8, north_sync=0.2,
    retail_flow=-1e5, youzi_flow=-5e5, chain_mod=0.05,
    global_mod=0.1, mrs=0.1, irs=0.1, grs=0.2, lock_ratio=0.8,
)

# 弱信号基准（ES 远低于开仓门槛）
_WEAK_BASE = dict(
    ofss=-0.5, cps=-0.3, inst_flow=-1e8, north_sync=-0.2,
    retail_flow=1e8, youzi_flow=1e5, chain_mod=-0.5,
    global_mod=-0.1, mrs=-1.0, irs=-0.5, grs=-0.2, lock_ratio=0.1,
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
# 连续评分：ES / PS / XS
# ----------------------------------------------------------------------

def _row(**mut):
    """牛市评估行 + 可选列覆盖（含 state，供评分函数直接使用）。"""
    syn = bull_syn()
    row = bull_eval_row(syn).copy()
    for k, v in mut.items():
        row[k] = v
    return syn, row


def _pos(**mut):
    p = Position(symbol="600000", entry_time=pd.Timestamp(_T_BUY),
                 entry_vwap=10.0, last_price=10.2, bars_held=0,
                 high_price_watermark=10.2, avg_cost=10.0)
    for k, v in mut.items():
        setattr(p, k, v)
    return p


class TestEntryScore:

    def test_bull_es_above_entry_threshold(self):
        """牛市基准：ES ≈ sigmoid(3 * 0.3776) ≈ 0.756 > th_es_entry=0.4。"""
        syn, row = _row()
        # final_ms=0.638 → 0.319；purity=0.8；mrs=0.1 → 0.0333
        expect_raw = 0.4 * 0.319 + 0.3 * 0.8 + 0.3 * (0.1 / 3.0)
        expect = 1.0 / (1.0 + np.exp(-3.0 * expect_raw))
        assert syn.calculate_entry_score(row) == pytest.approx(expect)
        assert 0.0 <= syn.calculate_entry_score(row) <= 1.0

    def test_weak_es_below_entry_threshold(self):
        syn = bull_syn()
        row = bull_eval_row(syn).copy()
        row.update({"final_ms": -2.0, "capital_purity": -1.0, "mrs": -1.0,
                    "state": S_NOISE})
        assert syn.calculate_entry_score(row) < 0.1

    def test_es_bounded_even_with_extreme_inputs(self):
        """有界化：Final_MS 极端 ±10 也只按 ±final_ms_clip 参与，ES 不饱和出界。"""
        syn, row = _row(final_ms=10.0, capital_purity=1.0, mrs=10.0)
        es = syn.calculate_entry_score(row)
        assert 0.0 <= es <= 1.0
        assert es == pytest.approx(
            1.0 / (1.0 + np.exp(-3.0 * (0.4 * 1.0 + 0.3 * 1.0 + 0.3 * 1.0))))

    def test_es_youzi_only_decay(self):
        syn, row = _row(state=S_YOUZI_ONLY)
        base = syn.calculate_entry_score(row)
        assert base == pytest.approx(0.5 * syn.calculate_entry_score(
            row.drop("state").to_dict() | {"state": S_PUSH}))

    def test_es_missing_component_neutral(self):
        """分量缺失视为中性 0：不影响其余分量，ES 不崩溃。"""
        syn, row = _row(final_ms=np.nan, mrs=np.nan)
        es = syn.calculate_entry_score(row)
        expect = 1.0 / (1.0 + np.exp(-3.0 * (0.3 * 0.8)))
        assert es == pytest.approx(expect)

    def test_es_invariant_to_unused_columns(self):
        """ES 公式不依赖 RS / Industry_MS（产业分量废弃）→ 缺失不影响。"""
        syn, row = _row(rs=np.nan, industry_ms=np.nan)
        assert syn.calculate_entry_score(row) == pytest.approx(
            syn.calculate_entry_score(bull_eval_row(syn)))


class TestPositionScore:

    def test_time_decay_formula(self):
        syn = SignalSynthesizer()
        # avg_cost 对齐 close=10.2（无浮盈）→ 不触发动量豁免，纯衰减生效
        pos = _pos(bars_held=10, avg_cost=10.2, high_price_watermark=10.2)
        assert syn.time_decay(bull_eval_row(syn), pos) == pytest.approx(0.95)
        pos.bars_held = 30
        assert syn.time_decay(bull_eval_row(syn), pos) == pytest.approx(0.95 ** 3)

    def test_time_decay_momentum_exempt(self):
        """浮盈拉开安全垫（>1.5%）→ 时间衰减豁免 = 1.0。"""
        syn = SignalSynthesizer()
        row = bull_eval_row(syn).copy()
        row["close"] = 10.3  # (10.3-10.0)/10.0 = 3% > 1.5%
        pos = _pos(bars_held=200)
        assert syn.time_decay(row, pos) == pytest.approx(1.0)

    def test_fund_stability_cancel_ratio(self):
        syn = SignalSynthesizer()
        row = bull_eval_row(syn).copy()
        row["cancel_ratio"] = 0.5
        assert syn.fund_stability(row) == pytest.approx(0.7)
        row["cancel_ratio"] = 0.1
        assert syn.fund_stability(row) == pytest.approx(1.0)

    def test_fund_stability_nan_neutral(self):
        """cancel_ratio 缺失（无逐笔数据）→ 中性 1.0，不惩罚。"""
        syn = SignalSynthesizer()
        row = bull_eval_row(syn).copy()
        row["cancel_ratio"] = np.nan
        assert syn.fund_stability(row) == pytest.approx(1.0)

    def test_fund_stability_book_thin(self):
        """盘口变薄（|OBI| 与 |big_flow| 同时趋零）→ 惩罚。"""
        syn = SignalSynthesizer()
        row = bull_eval_row(syn).copy()
        row["cancel_ratio"] = 0.0
        row["obi"] = 0.01
        row["big_flow"] = 0.01
        assert syn.book_thin(row) is True
        assert syn.fund_stability(row) == pytest.approx(0.7)
        row["obi"] = 0.5  # 盘口失衡恢复 → 不再变薄
        assert syn.book_thin(row) is False
        assert syn.fund_stability(row) == pytest.approx(1.0)

    def test_ps_in_bounds(self):
        syn, row = _row()
        # close=10.2 有浮盈 → 豁免衰减（time_decay=1.0），fs=1.0
        pos = _pos(bars_held=30)
        ps = syn.calculate_position_score(row, pos)
        assert 0.0 <= ps <= 1.0
        assert ps == pytest.approx(syn.calculate_entry_score(row) * 1.0 * 1.0)


class TestExitScore:

    def test_xs_formula(self):
        syn = SignalSynthesizer()
        row = bull_eval_row(syn).copy()   # final_ms=0.638→0.319, purity=0.8
        pos = _pos(high_price_watermark=10.0)
        row["close"] = 9.5                # drawdown = 0.05
        expect = 0.5 * 0.319 + 0.3 * 0.8 - 0.2 * 0.05
        assert syn.calculate_exit_score(row, pos) == pytest.approx(expect)

    def test_xs_drawdown_from_high_watermark(self):
        syn = SignalSynthesizer()
        row = bull_eval_row(syn).copy()
        pos = _pos(high_price_watermark=10.0)
        row["close"] = 9.0
        assert syn.drawdown_from_high(row, pos) == pytest.approx(0.1)
        row["close"] = 10.5  # 创新高 → 回撤 0
        assert syn.drawdown_from_high(row, pos) == pytest.approx(0.0)

    def test_xs_runaway_veto(self):
        """游资溃逃：big_flow<0 且 Retail_Chase>阈值 → XS=-1.0。"""
        syn, row = _row(big_flow=-0.1, retail_chase=0.9)
        assert syn.veto(row) is True
        assert syn.calculate_exit_score(row, _pos()) == -1.0

    def test_xs_index_dive_veto(self):
        """大盘跳水：沪深300 跌破 VWAP*(1-1.5%) → XS=-1.0。"""
        syn, row = _row(index_close=2900.0, index_vwap=3000.0)
        assert syn.veto(row) is True
        assert syn.calculate_exit_score(row, _pos()) == -1.0

    def test_xs_no_veto_when_inputs_missing(self):
        """一票否决输入缺失（无 big_flow/指数）→ 不触发（保守不误杀）。"""
        syn, row = _row(big_flow=np.nan, index_close=np.nan, index_vwap=np.nan)
        assert syn.veto(row) is False


class TestHardFilters:

    def test_st_blocks(self):
        syn, row = _row(is_st=True)
        assert syn.hard_filters(row)["st"] is False
        assert syn.hard_all(row) is False

    def test_limit_up_blocks(self):
        row_up_limit = 11.22  # D2 基准 10.2 × 1.1，close 触及涨停价
        syn, row = _row(close=row_up_limit)
        assert syn.hard_filters(row)["limit"] is False
        assert syn.hard_all(row) is False

    def test_liquidity_blocks(self):
        syn, row = _row(amount=1e5)
        assert syn.hard_filters(row)["liquidity"] is False
        assert syn.hard_all(row) is False

    def test_time_window(self):
        syn = bull_syn()
        ds, features = full_env(_D3)
        ev = TradingStateMachine(synthesizer=syn)._build_eval_table(ds, features)
        early = ev[ev["ts"] == pd.Timestamp("2024-01-04 09:30")].iloc[0]
        late = ev[ev["ts"] == pd.Timestamp("2024-01-04 15:00")].iloc[0]
        assert syn.hard_filters(early)["time"] is False
        assert syn.hard_filters(late)["time"] is False


# ----------------------------------------------------------------------
# 历史接口兼容包装（新语义）
# ----------------------------------------------------------------------

class TestCompatibilityGates:

    def test_entry_gates_pass_in_bull(self):
        syn, row = _row()
        gates = syn.entry_gates(row)
        assert all(gates.values()), gates
        assert syn.entry_all(row) is True

    def test_entry_gates_blocked_on_low_es(self):
        syn, row = _row(final_ms=-2.0, capital_purity=-1.0, mrs=-1.0,
                        state=S_NOISE)
        gates = syn.entry_gates(row)
        assert gates["stock"] is False
        assert syn.entry_all(row) is False

    def test_entry_gates_blocked_on_hard_filter(self):
        syn, row = _row(is_st=True)
        assert syn.entry_gates(row)["liquidity"] is False
        assert syn.entry_all(row) is False

    def test_exit_triggers_on_low_xs(self):
        """XS 破清仓线 → 全部兼容平仓键为真。"""
        syn, row = _row(final_ms=-2.0, capital_purity=-1.0)
        trig = syn.exit_triggers(row, _pos())
        assert all(trig.values())
        assert syn.exit_any(row, _pos()) is True

    def test_exit_triggers_circuit_on_veto(self):
        syn, row = _row(index_close=2900.0, index_vwap=3000.0)
        assert syn.exit_triggers(row, _pos())["circuit"] is True

    def test_exit_triggers_inactive_in_bull(self):
        syn, row = _row()
        trig = syn.exit_triggers(row, _pos())
        assert not any(trig.values())


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
        # 新架构：ES 无日频因子依赖，硬过滤通过即开仓 → 首个 BUY 于 D0 10:00
        #（D0 09:30/09:45 在时间窗外 09:45 之前，不可开仓）
        assert first.timestamp == pd.Timestamp("2024-01-02 10:00")
        assert first.action == ACT_BUY and first.state == S_PUSH
        assert first.metrics["es"] >= 0.4
        # 内部持仓：高水位 = 入场价、加权成本 = 入场 VWAP
        pos = sm.positions["600000"]
        assert pos.high_price_watermark == pytest.approx(pos.last_price)
        assert pos.avg_cost == pytest.approx(pos.entry_vwap)

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
        # 全程游资主导 + 散户盲从 → 状态恒为 S_youzi_only（ES 衰减），绝不开仓
        base = {"youzi_flow": 1e5, "inst_flow": -1e5, "retail_flow": 1e8}
        sm, ds, features = self._run(base=base)
        sigs = sm.run(ds, features)
        assert all(s.state == S_YOUZI_ONLY for s in sigs)
        assert all(s.action != ACT_BUY for s in sigs)

    def test_sell_on_youzi_runaway_veto(self):
        """游资溃逃一票否决：big_flow<0 且 Retail_Chase>阈值 → 强制 SELL。"""
        sm, ds, features = self._run(
            syn=bull_syn(chase_window=1),
            overrides={"2024-01-04 10:30": {
                "youzi_flow": 1e5, "inst_flow": -1e5,
                "retail_flow": 1e8, "big_flow": -0.1}})
        sigs = sm.run(ds, features)
        sell = next(s for s in sigs
                    if s.timestamp == pd.Timestamp("2024-01-04 10:30"))
        assert sell.action == ACT_SELL
        assert sell.state == S_YOUZI_ONLY
        assert sell.metrics["xs"] == -1.0
        # 新架构无冷却期：清仓后信号恢复可重新买入；若已重新买入须晚于清仓时点
        if "600000" in sm.positions:
            assert sm.positions["600000"].entry_time > sell.timestamp

    def test_decay_reduce_on_final_ms_dip(self):
        """XS 落入 (0, th_xs_reduce)：chain_mod=-1 → final_ms 转弱 → DECAY_REDUCE。"""
        sm, ds, features = self._run(overrides={
            "2024-01-04 10:30": {"chain_mod": -1.0}})
        sigs = sm.run(ds, features)
        sig = next(s for s in sigs
                   if s.timestamp == pd.Timestamp("2024-01-04 10:30"))
        assert sig.action == ACT_DECAY_REDUCE
        assert 0.0 < sig.metrics["xs"] < 0.2
        assert sig.metrics["reduce_fraction"] == 0.5
        assert "600000" in sm.positions  # 减仓不清仓

    def test_sell_when_purity_turns_negative(self):
        """资金纯净度转负 + Final_MS 走弱 → XS <= 0 → SELL 清仓。"""
        sm, ds, features = self._run(overrides={
            "2024-01-04 10:30": {"chain_mod": -1.0,
                                 "retail_flow": 1e8, "youzi_flow": 1e5}})
        sigs = sm.run(ds, features)
        sell = next(s for s in sigs
                    if s.timestamp == pd.Timestamp("2024-01-04 10:30"))
        assert sell.action == ACT_SELL
        assert sell.metrics["xs"] <= 0.0
        if "600000" in sm.positions:
            assert sm.positions["600000"].entry_time > sell.timestamp

    def test_sell_on_circuit_breaker(self):
        # 大盘跳水（指数跌破 VWAP-1.5%）→ 一票否决 → SELL
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
        assert sell.metrics["xs"] == -1.0

    def test_add_after_min_interval(self):
        sm, ds, features = self._run(sm_kw={"min_add_interval": 2})
        sigs = sm.run(ds, features)
        # XS 维持高位 → 10:30 距上次加仓不足 2 → HOLD；11:00 bars_held=2 → ADD
        a = next(s for s in sigs if s.timestamp == pd.Timestamp("2024-01-04 10:30"))
        b = next(s for s in sigs if s.timestamp == pd.Timestamp("2024-01-04 11:00"))
        assert a.action == ACT_HOLD
        assert b.action == ACT_ADD

    def test_sell_after_add_resumes_position(self):
        # 加仓后 Final_MS 走弱 + 纯净度转负 → SELL 清仓
        sm, ds, features = self._run(overrides={
            "2024-01-04 11:30": {"chain_mod": -1.0,
                                 "retail_flow": 1e8, "youzi_flow": 1e5}})
        sigs = sm.run(ds, features)
        sell = next(s for s in sigs
                    if s.timestamp == pd.Timestamp("2024-01-04 11:30"))
        assert sell.action == ACT_SELL
        if "600000" in sm.positions:
            assert sm.positions["600000"].entry_time > sell.timestamp

    def test_buy_with_industry_mapping(self):
        sm, ds, features = self._run(syn=bull_syn())
        sigs = sm.run(ds, features)
        assert self._first_buy(sigs).action == ACT_BUY

    def test_buy_without_industry_mapping(self):
        """ES 公式不含产业分量 → 无行业映射也可开仓（行为变更，产业闸门废弃）。"""
        ds, features = full_env(_D3)
        sm = TradingStateMachine(SignalSynthesizer(symbol_to_industry={}))
        sigs = sm.run(ds, features)
        assert self._first_buy(sigs).action == ACT_BUY


# ----------------------------------------------------------------------
# 目标权重 Target_Weight（驱动撮合引擎差额调仓）
# ----------------------------------------------------------------------

class TestTargetWeights:

    @staticmethod
    def _tw(**mut):
        """牛市评估行 + 可选列覆盖。"""
        syn = bull_syn()
        row = bull_eval_row(syn).copy()
        for k, v in mut.items():
            row[k] = v
        return syn, row

    def test_flat_buy_formula(self):
        syn, row = self._tw()
        es = syn.calculate_entry_score(row)
        scale = (np.clip(1.0 + 0.1, *syn.tw_gmod_clip)
                 * np.clip(1.0 + 0.05, *syn.tw_cmod_clip))
        tw = syn.generate_target_weights(row, None)
        assert tw == pytest.approx(syn.base_weight * es * scale)
        assert 0.0 < tw <= syn.max_single_position

    def test_flat_weak_signal_zero(self):
        syn, row = self._tw(final_ms=-2.0, capital_purity=-1.0, mrs=-1.0)
        assert syn.generate_target_weights(row, None) == 0.0

    def test_flat_st_blocked(self):
        syn, row = self._tw(is_st=True)
        assert syn.generate_target_weights(row, None) == 0.0

    def test_flat_outside_time_window(self):
        syn, row = self._tw(ts=pd.Timestamp("2024-01-04 09:30"))
        assert syn.generate_target_weights(row, None) == 0.0

    def test_holding_exit_zero(self):
        syn, row = self._tw(final_ms=-2.0, capital_purity=-1.0)
        assert syn.generate_target_weights(row, _pos()) == 0.0

    def test_holding_reduce_step(self):
        syn, row = self._tw(final_ms=0.2, capital_purity=0.3)
        pos = _pos(simulated_weight=0.2)
        assert 0.0 < syn.calculate_exit_score(row, pos) < syn.th_xs_reduce
        assert syn.generate_target_weights(row, pos) \
            == pytest.approx(0.2 * syn.reduce_step_ratio)

    def test_holding_rebalance_by_ps(self):
        syn, row = self._tw()
        pos = _pos()
        assert syn.calculate_exit_score(row, pos) >= syn.th_xs_reduce
        ps = syn.calculate_position_score(row, pos)
        scale = (np.clip(1.0 + float(row["global_mod"]), *syn.tw_gmod_clip)
                 * np.clip(1.0 + float(row["chain_mod"]), *syn.tw_cmod_clip))
        assert syn.generate_target_weights(row, pos) \
            == pytest.approx(syn.base_weight * ps * scale)

    def test_clip_to_max_single_position(self):
        syn, row = self._tw(final_ms=10.0, capital_purity=1.0, mrs=10.0,
                            global_mod=10.0, chain_mod=10.0)
        assert syn.generate_target_weights(row, None) \
            == pytest.approx(syn.max_single_position)

    def test_mod_missing_neutral_scale(self):
        syn, row = self._tw(global_mod=np.nan, chain_mod=np.nan)
        assert syn._tw_scale(row) == pytest.approx(1.0)

    def test_mod_clip_bounds(self):
        syn, row = self._tw(global_mod=-10.0, chain_mod=-10.0)
        assert syn._tw_scale(row) == pytest.approx(
            syn.tw_gmod_clip[0] * syn.tw_cmod_clip[0])

    def test_buy_signal_writes_target_weight(self):
        sm, ds, features = TestStateMachine()._run()
        sigs = sm.run(ds, features)
        first = min((s for s in sigs if s.action == ACT_BUY),
                    key=lambda s: s.timestamp)
        tw = first.metrics["target_weight"]
        assert 0.0 < tw <= sm.syn.max_single_position
        # 内部模拟权重 = 开仓目标权重（BUY 建仓时初始化）
        assert sm.positions["600000"].simulated_weight == pytest.approx(tw)

    def test_hold_outside_entry_target_zero(self):
        sm, ds, features = TestStateMachine()._run()
        sigs = sm.run(ds, features)
        hold = next(s for s in sigs if s.action == ACT_HOLD)
        assert hold.metrics["target_weight"] == 0.0

    def test_decay_reduce_persists_simulated_weight(self):
        syn, row = self._tw(final_ms=0.2, capital_purity=0.3)
        sm = TradingStateMachine(synthesizer=syn, min_reduce_interval=0)
        pos = _pos(simulated_weight=0.2, last_reduce_bar=0)
        sig = sm._on_holding(row, pos)
        assert sig.action == ACT_DECAY_REDUCE
        assert 0.0 < sig.metrics["xs"] < syn.th_xs_reduce
        assert sig.metrics["target_weight"] == pytest.approx(0.16)
        assert pos.simulated_weight == pytest.approx(0.16)  # 触发后持久化

    def test_reduce_interval_blocks_persist(self):
        """减仓节奏未满足 → HOLD，目标权重仍表达 ×0.8，但模拟权重不更新。"""
        syn, row = self._tw(final_ms=0.2, capital_purity=0.3)
        sm = TradingStateMachine(synthesizer=syn, min_reduce_interval=10)
        pos = _pos(simulated_weight=0.2, last_reduce_bar=0)
        sig = sm._on_holding(row, pos)
        assert sig.action == ACT_HOLD
        assert sig.metrics["target_weight"] == pytest.approx(0.16)
        assert pos.simulated_weight == pytest.approx(0.2)  # 未触发不更新

    def test_add_persists_simulated_weight(self):
        syn, row = self._tw()
        sm = TradingStateMachine(synthesizer=syn, min_add_interval=0)
        pos = _pos(simulated_weight=0.1, last_add_bar=0)
        sig = sm._on_holding(row, pos)
        assert sig.action == ACT_ADD
        assert pos.simulated_weight \
            == pytest.approx(sig.metrics["target_weight"])

    def test_validation_rules(self):
        with pytest.raises(ValueError):
            SignalSynthesizer(base_weight=0.5, max_single_position=0.3)
        with pytest.raises(ValueError):
            SignalSynthesizer(reduce_step_ratio=1.5)
        with pytest.raises(ValueError):
            SignalSynthesizer(tw_gmod_clip=(1.5, 0.2))
        with pytest.raises(ValueError):
            SignalSynthesizer(tw_gmod_clip=(0.0, 1.5))
