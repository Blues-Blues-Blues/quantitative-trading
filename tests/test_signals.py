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
    u = dict(base or {})
    merged = dict(_BULL_BASE)
    merged.update(u)
    rows = []
    for t in axis:
        for s in symbols:
            row = dict(merged)
            # 资金流恒为常量（非 rolling_std 扰动）：常量大额流 → 无状态 tanh
            # _norm_flow 直接输出 ±1 强信号；overrides/base 统一走 merged。
            row["inst_flow"] = merged["inst_flow"]
            row["retail_flow"] = merged["retail_flow"]
            row.update({"ts": t, "symbol": s})
            rows.append(row)
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
        f = mk_features(axis)  # inst_flow 常量 1e8，无滚动扰动
        res = syn._agent_ms(f)
        # 机构流无状态 tanh：inst_flow=1e8 → scale=1e7 → tanh(10)≈1（不衰减、无盲区）。
        inst = np.tanh(f["inst_flow"].astype(float) / 1e7)
        expect = (0.35 * f["ofss"] + 0.25 * f["cps"] + 0.25 * inst
                  + 0.15 * f["north_sync"])
        np.testing.assert_allclose(res.to_numpy(), expect.to_numpy(), rtol=1e-9)
        # 常量大额资金流 → inst≈+1 强信号（无 rolling_std 置 0 / 开盘钝化）
        assert res.iloc[0] == pytest.approx(0.35*0.5 + 0.25*0.3 + 0.25*1.0 + 0.15*0.2)
        # 值域边界 [-1, 1]
        assert (res.dropna() >= -1.0).all()
        assert (res.dropna() <= 1.0).all()

    def test_final_ms_formula(self):
        syn = SignalSynthesizer()
        out = syn.final_ms(pd.Series([0.53]), pd.Series([0.05]), pd.Series([0.1]))
        assert out.iloc[0] == pytest.approx((0.53 + 0.05) * 1.1)

    def test_final_ms_short_side_amplified(self):
        """多空不对称：全球恶化(global_mod<0)时，负向 agent_ms 被放大而非压制。"""
        syn = SignalSynthesizer()
        out = syn.final_ms(pd.Series([-0.5, 0.5]),
                           pd.Series([0.0, 0.0]),
                           pd.Series([-0.3, -0.3]))
        # 空头：base=-0.5<0 → scale=1-(-0.3)=1.3 → -0.65
        np.testing.assert_allclose(out.iloc[0], -0.65, rtol=1e-9)
        # 多头：base=0.5>=0 → scale=1+(-0.3)=0.7 → 0.35
        np.testing.assert_allclose(out.iloc[1], 0.35, rtol=1e-9)
        # 负向信号幅度被放大（|-0.65| > |0.35|）
        assert abs(out.iloc[0]) > abs(out.iloc[1])

    def test_final_ms_bounds(self):
        """量纲有界：多空与 global 极端组合下，输出严格在 [-1, 1]。"""
        syn = SignalSynthesizer()
        out = syn.final_ms(pd.Series([1.0, -1.0, 0.5, -0.6]),
                           pd.Series([0.3, -0.3, 0.2, 0.1]),
                           pd.Series([0.8, -0.8, 0.8, -0.8]))
        assert (out.dropna() >= -1.0).all()
        assert (out.dropna() <= 1.0).all()
        assert out.iloc[0] == pytest.approx(1.0)   # 多头饱和上限
        assert out.iloc[1] == pytest.approx(-1.0)  # 空头饱和下限

    def test_final_ms_nan_preserved(self):
        """agent_ms 缺失 → 输出保持 NaN（fillna(0) 中间计算不污染）。"""
        syn = SignalSynthesizer()
        out = syn.final_ms(pd.Series([np.nan, 0.4]),
                           pd.Series([0.05, 0.05]),
                           pd.Series([0.1, 0.1]))
        assert np.isnan(out.iloc[0])
        # 第二行：base=0.45>=0 → scale=1.1 → 0.495
        np.testing.assert_allclose(out.iloc[1], 0.45 * 1.1, rtol=1e-9)

    def test_capital_purity_formula(self):
        axis = cn_minutes(["2024-01-02"])
        f = mk_features(axis)  # 常量流：inst=1e8, north=0.2, retail=-1e5, youzi=-5e5
        purity = SignalSynthesizer().capital_purity(f).iloc[0]
        # 无状态 tanh 平滑（scale=1e7，常量大流不衰减）+ 固定权重重归一化
        wi, wn, wr, wy = 0.35, 0.25, 0.20, 0.20
        norm_inst = np.tanh(1e8 / 1e7)
        norm_retail = np.tanh(-1e5 / 1e7)
        norm_youzi = np.tanh(-5e5 / 1e7)
        num = wi * norm_inst + wn * 0.2 - wr * norm_retail - wy * norm_youzi
        expect = num / (wi + wn + wr + wy)
        assert purity == pytest.approx(expect, rel=1e-9)
        # 常量大额机构净流入 → 强正纯净度（不再因常量 rolling_std==0 误判为中性 0）
        assert purity > 0.3

    def test_capital_purity_small_flow_neutral(self):
        """资金流极小值不引发突变：常量极小正流 → tanh 近 0 → 纯净度中性。"""
        syn = SignalSynthesizer()
        n = 10
        f = pd.DataFrame({
            "symbol": ["A"] * n, "inst_flow": [1e-6] * n,
            "retail_flow": [0.0] * n, "youzi_flow": [0.0] * n,
            "north_sync": [0.0] * n,
        })
        purity = syn.capital_purity(f)
        assert np.allclose(purity, 0.0)

    def test_capital_purity_missing_renormalize(self):
        """缺失重归一化（与 Agent_MS 一致）：部分缺失按剩余权重归一化；
        全缺失 → w_sum==0 → 输出 NaN。"""
        axis = cn_minutes(["2024-01-02"])
        f = mk_features(axis)  # 常量流：inst=1e8, north=0.2, retail=-1e5, youzi=-5e5
        # 全缺失 → w_sum==0 → NaN
        full_nan = f.assign(inst_flow=np.nan, retail_flow=np.nan,
                            youzi_flow=np.nan, north_sync=np.nan)
        assert SignalSynthesizer().capital_purity(full_nan).isna().all()
        # 仅 youzi 缺失 → 剔除权重 0.20，剩余 (inst,north,retail) 归一化
        part = f.assign(youzi_flow=np.nan)
        p = SignalSynthesizer().capital_purity(part)
        assert not p.isna().any()
        wi, wn, wr = 0.35, 0.25, 0.20
        num = wi * np.tanh(1e8 / 1e7) + wn * 0.2 - wr * np.tanh(-1e5 / 1e7)
        np.testing.assert_allclose(p.iloc[0], num / (wi + wn + wr), rtol=1e-9)

    def test_capital_purity_bounds(self):
        """输出上下界：强多/空流下饱和到 ±1，且恒在 [-1, 1]。"""
        syn = SignalSynthesizer()
        n = 8
        # 多头：机构/北向强流入 + 零售/游资强流出 → 纯净度饱和 +1
        bull = pd.DataFrame({
            "symbol": ["A"] * n, "inst_flow": [1e8] * n,
            "retail_flow": [-1e8] * n, "youzi_flow": [-1e8] * n,
            "north_sync": [1.0] * n,
        })
        pb = syn.capital_purity(bull)
        assert (pb >= -1.0).all() and (pb <= 1.0).all()
        assert pb.iloc[0] == pytest.approx(1.0)  # 0.35+0.25+0.20+0.20
        # 空头：机构/北向强流出 + 零售/游资强涌入 → 纯净度饱和 -1
        bear = pd.DataFrame({
            "symbol": ["A"] * n, "inst_flow": [-1e8] * n,
            "retail_flow": [1e8] * n, "youzi_flow": [1e8] * n,
            "north_sync": [-1.0] * n,
        })
        pe = syn.capital_purity(bear)
        assert pe.iloc[0] == pytest.approx(-1.0)
        assert (pe >= -1.0).all() and (pe <= 1.0).all()

    def test_agent_ms_nan_renormalize(self):
        axis = cn_minutes(["2024-01-02"])
        syn = SignalSynthesizer(weights=(0.35, 0.25, 0.25, 0.15))
        f = mk_features(axis, overrides={"2024-01-02 09:30": {"ofss": np.nan}})
        res = syn._agent_ms(f)
        idx = pd.Timestamp("2024-01-02 09:30")
        # ofss 缺失 → 剔除其权重 0.35，剩余权重 (cps, inst, north)=(0.25,0.25,0.15)，
        # 和 0.65 重归一化；常量大额 inst_flow → inst = tanh(1e8/1e7) ≈ 1。
        expect = (0.25 * f.loc[idx, "cps"] + 0.25 * np.tanh(1e8 / 1e7)
                  + 0.15 * f.loc[idx, "north_sync"]) / 0.65
        np.testing.assert_allclose(res.loc[idx], expect, rtol=1e-9)
        # 仅一分量缺失 → 输出非 NaN（重归一化成功）
        assert not np.isnan(res.loc[idx])
        # 值域边界 [-1, 1]
        assert (res.dropna() >= -1.0).all()
        assert (res.dropna() <= 1.0).all()

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
        """牛市基准：ES 显著高于 th_es_entry=0.4。
        final_ms 已在 [-1,1] 直接使用（不再二次 clip）；mrs 经 ±mrs_clip
        有界缩放后加权，再 sigmoid 归一。"""
        syn, row = _row()
        final_ms = float(row["final_ms"])  # 原值直接参与
        mrs_c = float(np.clip(row["mrs"], -syn.mrs_clip,
                              syn.mrs_clip)) / syn.mrs_clip
        purity = float(row["capital_purity"])
        expect_raw = (syn.w_es_ms * final_ms + syn.w_es_purity * purity
                      + syn.w_es_mrs * mrs_c)
        expect = 1.0 / (1.0 + np.exp(-syn.es_sigmoid_k * expect_raw))
        assert syn.calculate_entry_score(row) == pytest.approx(expect)
        assert 0.0 < syn.calculate_entry_score(row) < 1.0

    def test_weak_es_below_entry_threshold(self):
        syn = bull_syn()
        row = bull_eval_row(syn).copy()
        row.update({"final_ms": -2.0, "capital_purity": -1.0, "mrs": -1.0,
                    "state": S_NOISE})
        assert syn.calculate_entry_score(row) < 0.1

    def test_es_bounded_even_with_extreme_inputs(self):
        """有界化：值域未定的 mrs（±10）仍被 clip 到 ±mrs_clip，ES 恒在 (0,1)。
        final_ms 按契约已在 [-1,1]，直接使用其上界 1.0。"""
        syn, row = _row(final_ms=1.0, capital_purity=1.0, mrs=10.0)
        es = syn.calculate_entry_score(row)
        assert 0.0 < es < 1.0
        # mrs_c = clip(10, ±3)/3 = 1；state 为牛 (S_PUSH) → 无游资衰减
        es_raw = (syn.w_es_ms * 1.0 + syn.w_es_purity * 1.0 + syn.w_es_mrs * 1.0)
        assert es == pytest.approx(1.0 / (1.0 + np.exp(-syn.es_sigmoid_k * es_raw)))

    def test_es_youzi_only_masked_to_zero(self):
        """游资独舞 → 硬掩码否决：ES 直接输出 0.0（非衰减乘积）。"""
        syn, row = _row(state=S_YOUZI_ONLY)
        assert syn.calculate_entry_score(row) == pytest.approx(0.0)
        # 非游资同特征行 → 正常 sigmoid > 0
        push = syn.calculate_entry_score(row.drop("state").to_dict()
                                         | {"state": S_PUSH})
        assert push > 0.0
        assert not (syn.calculate_entry_score(row) == pytest.approx(push))

    def test_es_missing_component_neutral(self):
        """分量缺失视为中性 0：不影响其余分量，ES 不崩溃。"""
        syn, row = _row(final_ms=np.nan, mrs=np.nan)
        es = syn.calculate_entry_score(row)
        expect = 1.0 / (1.0 + np.exp(-3.0 * (syn.w_es_purity
                                             * float(row["capital_purity"]))))
        assert es == pytest.approx(expect)

    def test_es_invariant_to_unused_columns(self):
        """ES 公式不依赖 RS / Industry_MS（产业分量废弃）→ 缺失不影响。"""
        syn, row = _row(rs=np.nan, industry_ms=np.nan)
        assert syn.calculate_entry_score(row) == pytest.approx(
            syn.calculate_entry_score(bull_eval_row(syn)))

    def test_compute_es_vectorized_long(self):
        """批量 _compute_es：全程 pd.Series index 自动对齐、缺失填 0、mrs 有界
        缩放、游资硬掩码为 0，返回值与输入索引对齐（不乱序、不丢失）、严格在
        [0, 1)。"""
        syn = bull_syn()
        idx = cn_minutes(_D3)
        n = len(idx)
        fms = pd.Series(([0.5, -0.3, np.nan] * (n // 3 + 1))[:n], index=idx)
        purity = pd.Series(([0.2, -0.4, 0.9] * (n // 3 + 1))[:n], index=idx)
        mrs = pd.Series(([5.0, -2.0, 1.2] * (n // 3 + 1))[:n], index=idx)
        youzi = pd.Series(([False, True, False] * (n // 3 + 1))[:n], index=idx)
        es = syn._compute_es(fms, purity, mrs, youzi)
        # 索引在计算前后未丢失、未乱序
        assert es.index.equals(idx)
        assert es.index.tolist() == idx.tolist()
        # 游资行才允许 0，其余行严格 (0, 1)
        assert (es[youzi] == 0.0).all()
        assert ((es[~youzi] > 0.0) & (es[~youzi] < 1.0)).all()
        # 逐行对照：fms 缺失→0；mrs clip ±mrs_clip；youzi=True → 硬掩码 0
        # 第 1 行：非游资，fms=0.5、purity=0.2、mrs=5.0→clip=3/3=1
        mrs_c1 = float(np.clip(5.0, -syn.mrs_clip, syn.mrs_clip)) / syn.mrs_clip
        raw1 = syn.w_es_ms * 0.5 + syn.w_es_purity * 0.2 + syn.w_es_mrs * mrs_c1
        assert es.iloc[0] == pytest.approx(
            1.0 / (1.0 + np.exp(-syn.es_sigmoid_k * raw1)))
        # 第 2 行：游资独舞 → 硬掩码 0（不再衰减后 sigmoid）
        assert es.iloc[1] == pytest.approx(0.0)
        # 第 3 行：非游资，fms 缺失→0、purity=0.9、mrs=1.2
        mrs_c3 = float(np.clip(1.2, -syn.mrs_clip, syn.mrs_clip)) / syn.mrs_clip
        raw3 = syn.w_es_ms * 0.0 + syn.w_es_purity * 0.9 + syn.w_es_mrs * mrs_c3
        assert es.iloc[2] == pytest.approx(
            1.0 / (1.0 + np.exp(-syn.es_sigmoid_k * raw3)))
        # 索引对齐测试：打乱 purity 顺序，结果仍按输入 idx 对齐（非位置剥离）
        shuffled = purity.copy().sample(frac=1.0, random_state=0)
        es_align = syn._compute_es(fms, shuffled, mrs, youzi)
        assert es_align.index.equals(idx)


class TestPositionScore:

    def test_time_decay_grace_period(self):
        """衰减保护期：bars_held <= win_decay_grace(30) → Time_Decay = 1.0。
        覆盖：持仓 < 30 分钟。"""
        syn = SignalSynthesizer()  # 默认 win_decay_grace=30
        row = bull_eval_row(syn).copy()
        row["close"] = 9.8  # 即使浮亏也不衰减（保护期内冻结）
        for held in (0, 1, 29, 30):
            pos = _pos(bars_held=held, avg_cost=10.0)
            assert syn.time_decay(row, pos) == pytest.approx(1.0)

    def test_time_decay_profit_asymmetric(self):
        """浮盈态慢衰减：>30 分钟 且 pnl_ratio>0 → factor=0.975。
        覆盖：持仓 > 30 分钟且浮盈。"""
        syn = SignalSynthesizer()
        row = bull_eval_row(syn).copy()
        row["close"] = 10.3  # (10.3-10.0)/10.0 = 3% > 0 → 浮盈
        pos = _pos(bars_held=40, avg_cost=10.0)  # effective=10
        # factor = 1-(1-0.95)*0.5 = 0.975；TD = 0.975^(10/10) = 0.975
        assert syn.time_decay(row, pos) == pytest.approx(0.975)
        pos.bars_held = 50  # effective=20 → 0.975^2
        assert syn.time_decay(row, pos) == pytest.approx(0.975 ** 2)

    def test_time_decay_loss_asymmetric(self):
        """浮亏态快衰减：>30 分钟 且 pnl_ratio<=0 → factor=0.90。
        覆盖：持仓 > 30 分钟且浮亏。"""
        syn = SignalSynthesizer()
        row = bull_eval_row(syn).copy()
        row["close"] = 9.8  # -2% → 浮亏
        pos = _pos(bars_held=40, avg_cost=10.0)  # effective=10
        # factor = 1-(1-0.95)*2.0 = 0.90；TD = 0.90^(10/10) = 0.90
        assert syn.time_decay(row, pos) == pytest.approx(0.90)
        pos.bars_held = 50  # effective=20 → 0.90^2
        assert syn.time_decay(row, pos) == pytest.approx(0.90 ** 2)

    def test_time_decay_floor_clip(self):
        """下限保护：长期浮亏衰减不跌破 0.1。"""
        syn = SignalSynthesizer()
        row = bull_eval_row(syn).copy()
        row["close"] = 9.0  # 深亏
        pos = _pos(bars_held=30 + 30 * 9, avg_cost=10.0)  # effective=270
        assert syn.time_decay(row, pos) == pytest.approx(0.1)

    def test_time_decay_nan_fallback(self):
        """pnl_ratio 无法计算（close 缺失）→ fallback 基准衰减率（中性）。"""
        syn = SignalSynthesizer()
        row = bull_eval_row(syn).copy()
        row["close"] = np.nan
        pos = _pos(bars_held=40)  # effective=10
        assert syn.time_decay(row, pos) == pytest.approx(0.95)

    def test_time_decay_avg_cost_zero_fallback(self):
        """avg_cost = 0（或 <=1e-6）→ pnl 无法计算 → 安全回退 base_decay_rate。"""
        syn = SignalSynthesizer()
        row = bull_eval_row(syn).copy()
        row["close"] = 10.0
        pos = _pos(bars_held=40, avg_cost=0.0)  # effective=10
        assert syn.time_decay(row, pos) == pytest.approx(0.95)
        pos.avg_cost = 1e-9  # 极小均值同样视为非法
        assert syn.time_decay(row, pos) == pytest.approx(0.95)

    def test_time_decay_negative_base_defense(self):
        """纵深防御：setattr 强行放大 loss_mult 产生负底数，不抛复数/domain error。

        绕过构造校验后 raw_factor = 1-(1-0.95)*100 = -4 → factor 钳到 0.01 下限，
        输出恒为合法 [0.1, 1.0] 内的 float。
        """
        syn = SignalSynthesizer()
        syn.pnl_decay_loss_mult = 100.0  # → raw_factor = 1 - 0.05*100 = -4
        row = bull_eval_row(syn).copy()
        row["close"] = 5.0  # 深亏 → 走 loss_mult 分支
        pos = _pos(bars_held=40, avg_cost=10.0)
        td = syn.time_decay(row, pos)  # 不应抛 ComplexWarning / math domain error
        assert isinstance(td, float)
        assert 0.1 <= td <= 1.0
        # factor = max(0.01, -4) = 0.01 → effective=10/interval=10 → 0.01^1，clip≥0.1
        assert td == pytest.approx(max(0.1, 0.01))

    def test_fund_stability_vectorized_differential(self):
        """纯向量化 _compute_fund_stability：多行、NaN、阈值触发/不触发分片正确。

        行0: cancel=NaN(→不撤单)、obi/big 大(不薄) → 1.0
        行1: cancel=0.5>th(→撤单惩罚) → 0.7
        行2: cancel=0.1 不撤单、obi/big=0.01<th(双薄) → 0.7
        行3: cancel=0.1 不撤单、obi 大(不薄) → 1.0
        """
        syn = bull_syn()
        idx = cn_minutes(_D3)
        n = len(idx)
        cancel = pd.Series(([np.nan, 0.5, 0.1, 0.1] * (n // 4 + 1))[:n], index=idx)
        obi = pd.Series(([0.9, 0.3, 0.01, 0.5] * (n // 4 + 1))[:n], index=idx)
        flow = pd.Series(([0.9, 0.05, 0.01, 0.5] * (n // 4 + 1))[:n], index=idx)
        fs = syn._compute_fund_stability(cancel, obi, flow)
        assert fs.index.equals(idx)                      # index 对齐
        assert fs.mode().iloc[0] in (1.0, syn.fund_stability_penalty)
        assert set(fs.unique()) <= {1.0, syn.fund_stability_penalty}
        assert (fs == 1.0).sum() >= 2                    # 至少行0/行3 中性
        # 逐行精确断言（周期性模式，取第 0~3 行校验）
        assert fs.iloc[0::4].eq(1.0).all()
        assert fs.iloc[1::4].eq(syn.fund_stability_penalty).all()
        assert fs.iloc[2::4].eq(syn.fund_stability_penalty).all()
        assert fs.iloc[3::4].eq(1.0).all()

    def test_fund_stability_missing_columns_downgrade(self):
        """缺列（全 NaN 输入）→ 安全降级为全 1.0，不抛 KeyError/异常。"""
        syn = bull_syn()
        idx = cn_minutes(_D3)
        n = len(idx)
        cancel = pd.Series(np.nan, index=idx)
        obi = pd.Series(np.nan, index=idx)
        flow = pd.Series(np.nan, index=idx)
        fs = syn._compute_fund_stability(cancel, obi, flow)
        assert list(fs.unique()) == [1.0]

    def test_ps_in_bounds(self):
        syn, row = _row()
        # close=10.2 有浮盈 → 豁免衰减（time_decay=1.0），fs=1.0
        pos = _pos(bars_held=30)
        ps = syn.calculate_position_score(row, pos)
        assert 0.0 <= ps <= 1.0
        assert ps == pytest.approx(syn.calculate_entry_score(row) * 1.0 * 1.0)

    def test_ps_no_clip_equivalence(self):
        """外层 clip 已移除：结果与裸连乘 es*td*fs 完全一致，且天然有界。"""
        syn, row = _row()
        pos = _pos(bars_held=40, avg_cost=10.0)
        row["close"] = 9.5  # 浮亏 → 真实衰减 factor=0.90
        raw = (syn.calculate_entry_score(row) * syn.time_decay(row, pos)
               * float(row.get("fund_stability", 1.0)))
        ps = syn.calculate_position_score(row, pos)
        assert ps == pytest.approx(raw)
        assert 0.0 <= ps <= 1.0

    def test_ps_es_nan_zeros(self):
        """es 缺失 → 强制 0 兜底：PS = 0.0（标量与 Series 双路）。"""
        syn = bull_syn()
        assert syn._compute_ps(float("nan"), 1.0, 1.0).iloc[0] == pytest.approx(0.0)
        es_s = pd.Series([np.nan, 0.5], index=["a", "b"])
        ps = syn._compute_ps(es_s, 1.0, 1.0)
        assert ps["a"] == pytest.approx(0.0)
        assert ps["b"] == pytest.approx(0.5)

    def test_ps_time_decay_nan_neutral(self):
        """time_decay 缺失 → 回退 1.0（不打折）：PS = es * fund_stability。"""
        syn = bull_syn()
        es_s = pd.Series([0.5])
        fs_s = pd.Series([0.7])
        ps = syn._compute_ps(es_s, float("nan"), fs_s)
        assert ps.iloc[0] == pytest.approx(0.5 * 1.0 * 0.7)
        # 两个乘数同时缺失 → 均不打折 = es
        ps2 = syn._compute_ps(0.5, float("nan"), float("nan"))
        assert ps2.iloc[0] == pytest.approx(0.5)

    def test_compute_ps_vectorized_long(self):
        """批量 _compute_ps：pd.Series index 对齐、es→0 / 乘数→1 兜底、去 clip 天然有界。"""
        syn = bull_syn()
        idx = cn_minutes(_D3)
        n = len(idx)
        es = pd.Series(([0.5, np.nan, 0.2] * (n // 3 + 1))[:n], index=idx)
        td = pd.Series(([0.9, 1.0, np.nan] * (n // 3 + 1))[:n], index=idx)
        fs = pd.Series(([0.7, 1.0, 0.7] * (n // 3 + 1))[:n], index=idx)
        ps = syn._compute_ps(es, td, fs)
        assert ps.index.equals(idx)                                # 索引未丢失/乱序
        assert ps.iloc[0] == pytest.approx(0.5 * 0.9 * 0.7)
        assert ps.iloc[1] == pytest.approx(0.0)                    # es 缺失 → 0
        assert ps.iloc[2] == pytest.approx(0.2 * 1.0 * 0.7)        # td 缺失 → 1.0
        assert ((ps >= 0.0) & (ps <= 1.0)).all()                   # 去 clip 后仍不越界


class TestExitScore:

    def test_xs_formula(self):
        syn = SignalSynthesizer()
        row = bull_eval_row(syn).copy()   # final_ms 已严格在 [-1,1]，直接使用原值
        pos = _pos(high_price_watermark=10.0)
        row["close"] = 9.5                # drawdown = 0.05
        final_ms = float(row["final_ms"])
        expect = (syn.w_xs_ms * final_ms + syn.w_xs_purity
                  * float(row["capital_purity"]) - syn.w_xs_drawdown * 0.05)
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

    def test_xs_zero_watermark_division_defense(self):
        """纯函数零除防御：hwm=0 → dd=0，安全返回基于 ms/purity 的有效分数。"""
        syn = SignalSynthesizer()
        ws = (syn.w_xs_ms, syn.w_xs_purity, syn.w_xs_drawdown)
        xs = syn._compute_xs(0.8, 0.5, 0.0, 9.5, False, ws)  # hwm=0 → dd=0
        assert isinstance(xs, float)
        assert xs == pytest.approx(syn.w_xs_ms * 0.8 + syn.w_xs_purity * 0.5)
        assert -1.0 <= xs <= 1.0

    def test_xs_nan_neutralized(self):
        """纯函数空值中性：final_ms=NaN 与 final_ms=0.0 完全等价。"""
        syn = SignalSynthesizer()
        ws = (syn.w_xs_ms, syn.w_xs_purity, syn.w_xs_drawdown)
        a = syn._compute_xs(np.nan, 0.5, 10.0, 9.5, False, ws)
        b = syn._compute_xs(0.0, 0.5, 10.0, 9.5, False, ws)
        assert a == pytest.approx(b)
        # None 同样中立化
        c = syn._compute_xs(None, 0.5, 10.0, 9.5, False, ws)
        assert c == pytest.approx(b)

    def test_xs_absolute_veto_ignores_score(self):
        """纯函数绝对否决：极度利好因子 + veto_flag=True → 严格 -1.0。"""
        syn = SignalSynthesizer()
        ws = (syn.w_xs_ms, syn.w_xs_purity, syn.w_xs_drawdown)
        xs = syn._compute_xs(1.0, 1.0, 10.0, 5.0, True, ws)  # 高ms/高purity/深回撤
        assert xs == -1.0


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
        #（D0 09:30/09:45 在时间窗 10:00 之前，不可开仓）
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
        # 全程游资主导 + 散户盲从 → 状态恒为 S_youzi_only（ES 硬掩码=0.0），绝不开仓
        base = {"youzi_flow": 1e5, "inst_flow": -1e5, "retail_flow": 1e8}
        sm, ds, features = self._run(base=base)
        sigs = sm.run(ds, features)
        assert all(s.state == S_YOUZI_ONLY for s in sigs)
        assert all(s.metrics["es"] == pytest.approx(0.0) for s in sigs)
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
        """XS 落入 (th_xs_exit, th_xs_reduce) 减仓带：chain_mod=-1 → final_ms 转弱
        → DECAY_REDUCE。final_ms 直接满幅参与（不再 clip 减半），XS 可为负但仍在带内。"""
        sm, ds, features = self._run(overrides={
            "2024-01-04 10:30": {"chain_mod": -1.0}})
        sigs = sm.run(ds, features)
        sig = next(s for s in sigs
                   if s.timestamp == pd.Timestamp("2024-01-04 10:30"))
        assert sig.action == ACT_DECAY_REDUCE
        assert sm.syn.th_xs_exit < sig.metrics["xs"] < sm.syn.th_xs_reduce_high
        assert sig.metrics["reduce_fraction"] == 0.5
        assert "600000" in sm.positions  # 减仓不清仓

    def test_reduce_when_purity_turns_negative(self):
        """资金纯净度转负 + Final_MS 走弱 → XS 落减仓带（th_xs_exit, th_xs_reduce_high）
        → 容错阶梯减仓（新语义：不再一刀切直接清仓）。"""
        sm, ds, features = self._run(overrides={
            "2024-01-04 10:30": {"chain_mod": -1.0,
                                 "retail_flow": 1e8, "youzi_flow": 1e5}})
        sigs = sm.run(ds, features)
        sell = next(s for s in sigs
                    if s.timestamp == pd.Timestamp("2024-01-04 10:30"))
        assert sell.action == ACT_DECAY_REDUCE
        assert sm.syn.th_xs_exit < sell.metrics["xs"] < sm.syn.th_xs_reduce_high

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

    def test_reduce_after_add_resumes_position(self):
        # 加仓后 Final_MS 走弱 + 纯净度转负 → XS 落减仓带 → 容错阶梯减仓（不清仓）
        sm, ds, features = self._run(overrides={
            "2024-01-04 11:30": {"chain_mod": -1.0,
                                 "retail_flow": 1e8, "youzi_flow": 1e5}})
        sigs = sm.run(ds, features)
        sell = next(s for s in sigs
                    if s.timestamp == pd.Timestamp("2024-01-04 11:30"))
        assert sell.action == ACT_DECAY_REDUCE
        assert sm.syn.th_xs_exit < sell.metrics["xs"] < sm.syn.th_xs_reduce_high

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

    def test_test_holding_negative_xs_reduce_not_exit(self):
        """负 XS（如 -0.16，落入 (th_xs_exit, th_xs_reduce_high)）→ 只容错阶梯减仓，不清仓。
        覆盖用户要求：XS=-0.1 时只会触发阶梯减仓而不会被直接清仓。"""
        syn, row = self._tw(final_ms=-0.4, capital_purity=-0.2)
        pos = _pos(simulated_weight=0.2)
        xs = syn.calculate_exit_score(row, pos)
        assert syn.th_xs_exit < xs < syn.th_xs_reduce_high  # 落减仓带
        assert xs < 0.0  # 负 XS 也仅减仓
        assert syn.generate_target_weights(row, pos) \
            == pytest.approx(0.2 * syn.reduce_step_ratio)  # 目标=×0.8，不清仓（≠0）

    def test_holding_crash_zero(self):
        """XS 破极速清仓线（<= th_xs_crash）→ 目标权重 0（极速清仓）。"""
        syn, row = self._tw(final_ms=-2.0, capital_purity=-1.0)
        pos = _pos()
        assert syn.calculate_exit_score(row, pos) <= syn.th_xs_crash
        assert syn.generate_target_weights(row, pos) == 0.0

    def test_holding_exit_zero(self):
        syn, row = self._tw(final_ms=-2.0, capital_purity=-1.0)
        assert syn.generate_target_weights(row, _pos()) == 0.0

    def test_holding_reduce_step(self):
        syn, row = self._tw(final_ms=0.2, capital_purity=0.3)
        pos = _pos(simulated_weight=0.2)
        assert 0.0 < syn.calculate_exit_score(row, pos) < syn.th_xs_reduce_high
        assert syn.generate_target_weights(row, pos) \
            == pytest.approx(0.2 * syn.reduce_step_ratio)

    def test_holding_rebalance_by_ps(self):
        syn, row = self._tw()
        pos = _pos()
        assert syn.calculate_exit_score(row, pos) >= syn.th_xs_reduce_high
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
        # min_add_interval 取极大值以禁用后续 ADD，隔离验证"BUY 建仓时 simulated_weight == target_weight"
        sm, ds, features = TestStateMachine()._run(sm_kw={"min_add_interval": 1000})
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
        assert 0.0 < sig.metrics["xs"] < syn.th_xs_reduce_high
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


class TestReversal:
    """次日低开反包（Reversal / Counter-Attack）机制。"""

    def _row(self, ts, **mut):
        """评估行副本 + 行级 override（直调 _on_holding 用）。
        放宽 start_time=09:30 以激活 09:30~10:00 反包窗口
        （默认 start_time=10:00 时反包受 time 闸门限制不触发）。"""
        syn = bull_syn(start_time="09:30")
        row = bull_eval_row(syn).copy()
        row["ts"] = pd.Timestamp(ts)
        for k, v in mut.items():
            row[k] = v
        return syn, row

    def _pos_next_day(self, **mut):
        base = dict(bars_held=10, entry_time=pd.Timestamp("2024-01-04 10:00"),
                    avg_cost=10.0, high_price_watermark=10.6,
                    simulated_weight=0.2, last_add_bar=0)
        base.update(mut)
        return _pos(**base)

    def test_reversal_add_freeze_exit(self):
        """次日低开 -2% 且 OFSS=0.4、big_flow>0、purity>0：
        ① 不触发止损卖出（豁免 XS 清仓）② 正确反包加仓 ③ 不突破 30% 上限。"""
        syn, row = self._row("2024-01-05 09:45",
                             close=9.7, prev_close=10.0, ofss=0.4,
                             big_flow=5e6, capital_purity=0.2, final_ms=-1.5)
        pos = self._pos_next_day()
        # 前置：XS 已破清仓线（若无反包将 SELL）
        assert syn.calculate_exit_score(row, pos) <= syn.th_xs_exit
        assert syn.reversal_active(row, pos) is True
        assert syn.reversal_overridden(row) is False
        sm = TradingStateMachine(synthesizer=syn, min_add_interval=0)
        sig = sm._on_holding(row, pos)
        assert sig.action == ACT_ADD  # 反包加仓而非清仓
        assert bool(sig.metrics.get("reversal_add"))  # metrics 序列化 bool→1.0
        es = syn.calculate_entry_score(row)
        expect = min(0.2 + syn.base_weight * es * syn.reversal_add_mult,
                     syn.max_single_position)
        assert sig.metrics["target_weight"] == pytest.approx(expect)
        assert sig.metrics["target_weight"] <= syn.max_single_position
        assert pos.simulated_weight == pytest.approx(expect)

    def test_reversal_no_add_beyond_cap(self):
        """加仓后累计目标不得突破 max_single_position（0.3）。"""
        syn, row = self._row("2024-01-05 09:40",
                             close=9.6, prev_close=10.0, ofss=0.4,
                             big_flow=5e6, capital_purity=0.2)
        pos = self._pos_next_day(simulated_weight=0.28,
                                 high_price_watermark=10.5)
        sm = TradingStateMachine(synthesizer=syn, min_add_interval=0)
        sig = sm._on_holding(row, pos)
        assert sig.action == ACT_ADD
        assert sig.metrics["target_weight"] == pytest.approx(syn.max_single_position)
        assert pos.simulated_weight == pytest.approx(syn.max_single_position)

    def test_reversal_freeze_overridden_by_flight(self):
        """反包承接中触发大盘熔断（沪深300 跌破 VWAP-1.5%）→ 冻结失效 → 强制清仓。"""
        syn, row = self._row("2024-01-05 09:45",
                             close=9.7, prev_close=10.0, ofss=0.4,
                             big_flow=5e6, capital_purity=0.2,
                             index_close=2900.0, index_vwap=3000.0)
        pos = self._pos_next_day()
        assert syn.reversal_active(row, pos) is True  # 承接中（big>0）
        assert syn.reversal_overridden(row) is True  # 但大盘跳水解除保护
        sm = TradingStateMachine(synthesizer=syn, min_add_interval=0)
        sm.positions["600000"] = pos  # SELL 分支会 del positions[sym]
        sig = sm._on_holding(row, pos)
        assert sig.action == ACT_SELL  # 冻结解除 → 强制清仓

    def test_reversal_freeze_frozen_within_interval(self):
        """反包冻结期内未达加仓节奏 → HOLD 维持仓位（target=simulated）。"""
        syn, row = self._row("2024-01-05 09:45",
                             close=9.7, prev_close=10.0, ofss=0.4,
                             big_flow=5e6, capital_purity=0.2, final_ms=-1.5)
        pos = self._pos_next_day()
        sm = TradingStateMachine(synthesizer=syn, min_add_interval=10)
        sig = sm._on_holding(row, pos)  # bars_held-last_add=10>=10 → 可加仓
        assert sig.action == ACT_ADD
        pos2 = self._pos_next_day(last_add_bar=9)  # 10-9=1 < 10 → 冻结 HOLD
        sig2 = sm._on_holding(row, pos2)
        assert sig2.action == ACT_HOLD
        assert sig2.metrics["target_weight"] == pytest.approx(pos2.simulated_weight)

    def test_reversal_inactive_outside_window(self):
        """超 reversal_window_end（10:00 后）→ 反包不激活。"""
        syn, row = self._row("2024-01-05 10:05",
                             close=9.7, prev_close=10.0, ofss=0.4,
                             big_flow=5e6, capital_purity=0.2, final_ms=-1.5)
        pos = self._pos_next_day()
        assert syn.reversal_active(row, pos) is False

    def test_reversal_requires_next_day(self):
        """非跨日（同日）→ 不激活。"""
        syn, row = self._row("2024-01-04 09:45",
                             close=9.7, prev_close=10.0, ofss=0.4,
                             big_flow=5e6, capital_purity=0.2)
        pos = self._pos_next_day(entry_time=pd.Timestamp("2024-01-04 09:30"))
        assert syn.reversal_active(row, pos) is False
