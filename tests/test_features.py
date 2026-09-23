"""核心因子计算与主体情绪分层单元测试（合成 mock 数据，不联网）。

覆盖：
- 资金主体分档与净流归一化（retail/inst/youzi）
- 北向共振 North_Sync、两融压力 Margin_Pressure 公式
- OFSS 成分（OBI/AR/Cancel_Ratio/BigFlow）与综合
- CPS 筹码分量、PSS 价格结构范围
- MRS/GRS/IRS 合成与 Global_Mod/Chain_Mod clip
- 防未来：两融/北向 T-1 不可见、龙虎榜 T+1 才可见
- FeatureEngine 端到端组装与缺数据降级
"""

import numpy as np
import pandas as pd
import pytest

from data.aligner import TimeAligner
from data.dataslice import DataSlice, SYMBOL, TRADE_DATE
from indicators.agent_profiling import AgentProfiling
from indicators.environment import Environment
from indicators.feature_engine import FEATURE_COLS, FeatureEngine
from indicators.microstructure import MicroStructure


# ----------------------------------------------------------------------
# Mock 数据构造
# ----------------------------------------------------------------------

def cn_minutes(dates, freq: str = "1min") -> pd.DatetimeIndex:
    parts = []
    for d in pd.to_datetime(dates):
        parts.append(pd.date_range(f"{d:%Y-%m-%d} 09:30", f"{d:%Y-%m-%d} 11:30", freq=freq))
        parts.append(pd.date_range(f"{d:%Y-%m-%d} 13:00", f"{d:%Y-%m-%d} 15:00", freq=freq))
    return pd.DatetimeIndex(np.concatenate([p.values for p in parts])).sort_values()


def mk_kline(axis, symbols=("600000",), base=10.0, vol=1000.0):
    rows = []
    for i, t in enumerate(axis):
        for s in symbols:
            c = base + 0.001 * i
            rows.append({
                "symbol": s, "open": c - 0.1, "high": c + 0.2, "low": c - 0.2,
                "close": c, "volume": vol, "amount": c * vol * 100,
                "vwap": c, "float_market_cap": 1e9, "up_limit": c * 1.1,
                "down_limit": c * 0.9, "is_st": False,
            })
    df = pd.DataFrame(rows)
    df.index = axis.repeat(len(symbols))
    df.index.name = "ts"
    return df


def mk_ticks(rows, symbol="600000"):
    """rows: list of (ts, price, volume, turnover, side, is_cancel)"""
    df = pd.DataFrame(
        [(*r[:1], symbol, *r[1:]) for r in rows],
        columns=["ts", "symbol", "price", "volume", "turnover", "side", "is_cancel"],
    )
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts")


def mk_snapshot(axis, symbol="600000", bid_v=200.0, ask_v=100.0):
    snap = pd.DataFrame(index=axis)
    snap["symbol"] = symbol
    for i in range(1, 6):
        snap[f"bid{i}_p"] = 10.0 - 0.01 * i
        snap[f"bid{i}_v"] = bid_v / i
        snap[f"ask{i}_p"] = 10.0 + 0.01 * i
        snap[f"ask{i}_v"] = ask_v / i
    return snap


def mk_index_min(axis):
    df = pd.DataFrame(index=axis)
    df["index_code"] = "000300.SH"
    for i, t in enumerate(axis):
        c = 3000 + 0.5 * i
        df.loc[t, ["open", "high", "low", "close"]] = [c - 1, c + 2, c - 2, c]
        df.loc[t, "volume"] = 1e7
    df["vwap"] = df["close"]
    df["ma20"] = df["close"].rolling(3, min_periods=1).mean()
    df["ma60"] = df["close"].rolling(3, min_periods=1).mean()
    return df


def mk_breadth(axis):
    df = pd.DataFrame(index=axis)
    df["advancers"] = 3000.0
    df["decliners"] = 1500.0
    df["adr"] = 2.0
    df["north_net"] = 5e7
    return df


def mk_industry(axis, name="银行"):
    df = pd.DataFrame(index=axis)
    df["industry"] = name
    df["open"] = df["high"] = df["low"] = df["close"] = 1000.0
    df["volume"] = 1e6
    df["money_flow"] = np.linspace(0, 1, len(axis)) * 1e8
    return df


def mk_macro(dates, base=100.0):
    rows = []
    for i, d in enumerate(pd.to_datetime(dates)):
        row = {"trade_date": d, "us_spx": base + i, "us_ndx": base + i,
               "us_dow": base + i, "brent": base + i, "gold": base + i,
               "copper": base + i, "us10y": 3.0 + 0.01 * i, "dxy": 100 + i,
               "hsi": base + i, "nky": base + i}
        rows.append(row)
    return pd.DataFrame(rows)


def mk_north_margin(dates):
    df = pd.DataFrame(index=pd.to_datetime(dates))
    df["symbol"] = "600000"
    df["trade_date"] = pd.to_datetime(dates)
    df["north_holding"] = np.linspace(1e7, 1.1e7, len(dates))
    df["north_buy_net"] = [1e6, -5e5, 2e6, -1e6, 5e5][:len(dates)]
    df["margin_fin_balance"] = np.linspace(1e9, 1.12e9, len(dates))
    df["margin_sec_balance"] = np.linspace(5e8, 5.3e8, len(dates))
    return df


def mk_dragon_tiger(dates):
    rows = []
    for i, d in enumerate(pd.to_datetime(dates)):
        rows.append({"symbol": "600000", "trade_date": d,
                     "buy_amount": 1e8, "sell_amount": 5e7,
                     "net_amount": 5e7, "side": 1})
    return pd.DataFrame(rows)


def full_slice(dates):
    axis = cn_minutes(dates, freq="30min")
    return DataSlice(
        kline=mk_kline(axis),
        l2_snapshot=mk_snapshot(axis),
        tick_trades=mk_ticks([
            ("2024-01-02 09:30", 10.0, 1000, 3e4, 1, False),
            ("2024-01-02 09:30", 10.0, 1000, 3e4, 1, False),
            ("2024-01-02 09:30", 10.0, 1000, 3e4, 1, False),
        ]),
        index_min=mk_index_min(axis),
        breadth=mk_breadth(axis),
        industry=mk_industry(axis),
        macro=mk_macro(dates),
        north_margin=mk_north_margin(dates),
        dragon_tiger=mk_dragon_tiger(dates),
        meta={"symbols": ["600000"]},
    )


# ----------------------------------------------------------------------
# 资金主体分层
# ----------------------------------------------------------------------

class TestAgentProfiling:

    def test_bucket_and_normalize(self):
        ticks = mk_ticks([
            ("2024-01-02 09:30", 10.0, 3000, 3e4, 1, False),     # 小单买
            ("2024-01-02 09:30", 10.0, 8000, 8e4, -1, False),    # 中单卖
            ("2024-01-02 09:30", 10.0, 50000, 5e5, 1, False),    # 大单买
            ("2024-01-02 09:30", 10.0, 200000, 2e6, 1, False),   # 超大单买
            ("2024-01-02 09:30", 10.0, 100, 1e3, 1, True),       # 撤单不计
        ])
        norm_base = pd.DataFrame({
            "ts": [pd.Timestamp("2024-01-02 09:30")], "symbol": "600000",
            "norm_base": [1e6]})
        norm_base = norm_base.set_index("ts")

        out = AgentProfiling().net_flows(ticks, norm_base)
        row = out.iloc[0]
        # retail = 3e4 + 0.5*(-8e4) = -1e4 → -1e4/1e6；youzi = 5e5/1e6
        # inst = 2e6 + 0.5*(-8e4) = 1.96e6 → 1.96e6/1e6 = 1.96 → clip 强压到 1.0
        assert row["retail_flow"] == pytest.approx(-1e4 / 1e6)
        assert row["inst_flow"] == pytest.approx(1.0)
        assert row["youzi_flow"] == pytest.approx(5e5 / 1e6)

    def test_custom_thresholds(self):
        ticks = mk_ticks([
            ("2024-01-02 09:30", 10.0, 100, 3e4, 1, False),   # 超过 custom small(2w) → 非小单
        ])
        norm = pd.DataFrame({"ts": [pd.Timestamp("2024-01-02 09:30")],
                             "symbol": "600000", "norm_base": [3e5]})
        norm = norm.set_index("ts")
        out = AgentProfiling(small_th=2e4).net_flows(ticks, norm)
        # 3w ≥ 2w 且 < 20w → 中单 → retail 含 0.5*3e4，基准 3e5 保持未被 clip 饱和
        assert out.iloc[0]["retail_flow"] == pytest.approx(0.5 * 3e4 / 3e5)

    def test_actor_flows_zero_base_no_zero_division(self):
        """纯函数：norm_base=0 且 xl_net=±10000 → inst_flow 被 clip 压到 ±1，
        不抛除零异常。"""
        ap = AgentProfiling()
        idx = pd.RangeIndex(2)
        base = pd.Series([0.0, 0.0], index=idx)  # 基准 0 → safe=1e-6
        r, i, y = ap._compute_actor_flows(
            pd.Series([0.0, 0.0], index=idx), pd.Series([0.0, 0.0], index=idx),
            pd.Series([0.0, 0.0], index=idx),
            pd.Series([1e4, -1e4], index=idx), base)
        assert i.iloc[0] == pytest.approx(1.0)   # 1e4/1e-6 → clip 1.0
        assert i.iloc[1] == pytest.approx(-1.0)
        assert (r == 0.0).all() and (y == 0.0).all()

    def test_actor_flows_mid_split_nan_blocked(self):
        """纯函数：mid 有效、其余档缺失 → NaN 被 fillna(0) 阻断，mid 仍按 0.5 拆分。"""
        ap = AgentProfiling()
        idx = pd.RangeIndex(2)
        r, i, y = ap._compute_actor_flows(
            pd.Series([np.nan, 1.0], index=idx),   # small
            pd.Series([2.0, np.nan], index=idx),   # mid
            pd.Series([np.nan, np.nan], index=idx),  # large
            pd.Series([np.nan, 3.0], index=idx),   # xl
            pd.Series([10.0, 10.0], index=idx))    # base
        # 行0：retail = (0 + 0.5*2)/10 = 0.1；inst = (0 + 0.5*2)/10 = 0.1；youzi = 0
        assert r.iloc[0] == pytest.approx(0.5 * 2 / 10)
        assert i.iloc[0] == pytest.approx(0.5 * 2 / 10)
        assert y.iloc[0] == 0.0
        # 行1：mid=NaN→0；inst = (3 + 0)/10 = 0.3；retail = (1 + 0)/10 = 0.1
        assert i.iloc[1] == pytest.approx(3.0 / 10)
        assert r.iloc[1] == pytest.approx(1.0 / 10)

    def test_north_sync_formula(self):
        dates = pd.date_range("2024-01-02", periods=5, freq="B")
        nm = pd.DataFrame({
            "symbol": "600000", "trade_date": dates,
            "north_holding": [100, 110, 105, 120, 130],
            "north_buy_net": [10, -5, 15, -20, 5],
            "margin_fin_balance": 0.0, "margin_sec_balance": 0.0,
        })
        out = AgentProfiling(holding_days=2).north_sync(nm)
        out = out.set_index(TRADE_DATE)
        # D4: sign(120/105-1)=+1, sign(-20)=-1 → 0.6-0.4=0.2
        assert out["north_sync"].iloc[-2] == pytest.approx(0.2)
        # D5: sign(130/120-1)=+1, sign(5)=+1 → 1.0
        assert out["north_sync"].iloc[-1] == pytest.approx(1.0)

    def test_north_weights_must_sum_to_one(self):
        """fail-fast：north_weights 和 ≠1 → 构造器抛 ValueError。"""
        with pytest.raises(ValueError):
            AgentProfiling(north_weights=[0.6, 0.5])  # 0.6+0.5=1.1 ≠ 1

    def test_north_sync_nan_passthrough(self):
        """NaN 原样传递：非陆股通标的（north_holding/north_buy_net 缺失）
        返回 NaN，不报错、不填空（交由下游动态重归一化）。"""
        nm = pd.DataFrame({
            "symbol": ["600000", "600000", "600000"],
            "trade_date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "north_holding": [np.nan, np.nan, np.nan],
            "north_buy_net": [np.nan, np.nan, np.nan],
            "margin_fin_balance": 0.0, "margin_sec_balance": 0.0,
        })
        out = AgentProfiling(holding_days=2).north_sync(nm)
        assert out["north_sync"].isna().all()

    def test_margin_pressure_formula(self):
        dates = pd.date_range("2024-01-02", periods=5, freq="B")
        nm = pd.DataFrame({
            "symbol": "600000", "trade_date": dates,
            "margin_fin_balance": [100, 102, 105, 108, 112],
            "margin_sec_balance": [50, 50, 51, 52, 53],
            "north_holding": 0.0, "north_buy_net": 0.0,
        })
        out = AgentProfiling(margin_days=2).margin_pressure(nm)
        # D5: (112/105-1) - (53/51-1)
        expect = (112 / 105 - 1) - (53 / 51 - 1)
        assert out["margin_pressure"].iloc[-1] == pytest.approx(expect)


# ----------------------------------------------------------------------
# 微观结构
# ----------------------------------------------------------------------

class TestMicroStructure:

    def _ds_with_orderflow(self):
        axis = pd.DatetimeIndex(["2024-01-02 09:30"])
        ds = DataSlice(
            kline=mk_kline(axis),
            l2_snapshot=mk_snapshot(axis, bid_v=200, ask_v=100),
            tick_trades=mk_ticks([
                ("2024-01-02 09:30", 10.0, 150, 3e5, 1, False),   # 大单买
                ("2024-01-02 09:30", 10.0, 50, 1e4, -1, False),   # 小单卖
                ("2024-01-02 09:30", 10.0, 50, 1e4, 0, True),     # 撤单
            ]),
        )
        return ds

    def test_ofss_components_and_score(self):
        ds = self._ds_with_orderflow()
        comp = MicroStructure().ofss_components(ds)
        row = comp.iloc[0]
        # OBI = (200-100)/300
        assert row["obi"] == pytest.approx(100 / 300)
        # AR = (150-50)/200
        assert row["ar"] == pytest.approx(0.5)
        # Cancel_Ratio = 50/(50+200)
        assert row["cancel_ratio"] == pytest.approx(50 / 250)
        # BigFlow = 3e5/(3e5+1e4)
        assert row["big_flow"] == pytest.approx(3e5 / 3.1e5)

        micro = MicroStructure()
        ofss = micro.ofss(comp).iloc[0]
        # 缺项动态重归一化 + 撤单率对称映射（四分量齐全，权重和=1）。
        # cancel_score = (0.2 - 0.2)/0.2 = 0
        w1, w2, w3, w4 = (micro.w["w1"], micro.w["w2"],
                          micro.w["w3"], micro.w["w4"])
        cancel_score = (micro.cancel_base - 50 / 250) / micro.cancel_base
        num = (w1 * (100 / 300) + w2 * 0.5 + w3 * cancel_score
               + w4 * (3e5 / 3.1e5))
        den = w1 + w2 + w3 + w4  # 全分量有效
        assert ofss == pytest.approx(float(np.clip(num / den, -1.0, 1.0)))

    def test_ofss_single_weight_returns_component(self):
        """仅单权重有效时，重归一化使 OFSS 恒为该分量本身 → 天然有界 [-1,1]。"""
        ds = self._ds_with_orderflow()
        micro = MicroStructure(ofss_weights=(4.0, 0.0, 0.0, 0.0))
        comp = micro.ofss_components(ds)
        ofss = micro.ofss(comp).iloc[0]
        assert ofss == pytest.approx(comp["obi"].iloc[0])  # = 4*obi/4 = obi
        assert -1.0 <= ofss <= 1.0

    def test_ofss_cancel_score_symmetric(self):
        """撤单率对称量纲：cr=0 → 局部 +1.0，cr=2*cancel_base(=0.4) → -1.0。"""
        idx = cn_minutes(["2024-01-02"], freq="30min")[:2]
        micro = MicroStructure(ofss_weights=(0.0, 0.0, 1.0, 0.0))  # 只考察撤单分量
        comp = pd.DataFrame({
            "obi": 0.05, "ar": 0.05, "big_flow": 0.05,
            "cancel_ratio": [0.0, 0.4],
        }, index=idx)
        ofss = micro.ofss(comp)
        assert ofss.iloc[0] == pytest.approx(1.0)   # cr=0 → +1
        assert ofss.iloc[1] == pytest.approx(-1.0)  # cr=0.4 → -1

    def test_pss_range_and_direction(self):
        axis = cn_minutes(["2024-01-02"], freq="30min")
        kline = mk_kline(axis)
        pss = MicroStructure(pss_window=3).pss(kline).dropna()
        assert ((pss >= -1.0) & (pss <= 1.0)).all()

    def test_pss_zero_volatility_flat_neutral(self):
        """区间无波动（一字/停牌，窗口 hi==lo）→ PSS 精确为 0.0，
        杜绝旧公式因微小常数 eps 误判 -1.0。"""
        axis = cn_minutes(["2024-01-02"], freq="30min")
        df = pd.DataFrame({
            "symbol": ["600000"] * len(axis),
            "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
        }, index=axis)
        pss = MicroStructure(pss_body_w=0.5, pss_window=3).pss(df).dropna()
        assert len(pss) > 0
        assert (pss == 0.0).all()

    def test_pss_extremes_isolated(self):
        """极值隔离：pss_body_w=0 屏蔽 body，close==hi → pct=+1.0，
        close==lo → pct=-1.0。"""
        axis = cn_minutes(["2024-01-02"], freq="30min")
        n = len(axis)
        closes = [7.0 if i % 2 == 0 else 11.0 for i in range(n)]
        df = pd.DataFrame({
            "symbol": ["600000"] * n,
            "open": np.full(n, 9.0), "high": np.full(n, 11.0),
            "low": np.full(n, 7.0), "close": closes,
        }, index=axis)
        pss = MicroStructure(pss_body_w=0.0, pss_window=3).pss(df).dropna()
        for ts, val in pss.items():
            c = closes[axis.get_loc(ts)]
            assert val == pytest.approx(2 * (c - 7.0) / 4.0 - 1.0)

    def test_cps_range(self):
        dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
        axis = cn_minutes(dates, freq="30min")
        ds = DataSlice(
            kline=mk_kline(axis),
            tick_trades=mk_ticks([
                ("2024-01-02 09:30", 10.0, 1000, 5e5, 1, False),
                ("2024-01-02 10:00", 10.0, 1000, 5e5, -1, False),
                ("2024-01-03 09:30", 10.0, 1000, 5e5, 1, False),
                ("2024-01-04 09:30", 10.0, 1000, 5e5, 1, False),
            ]),
        )
        comp = MicroStructure(chip_window=2).chip_components(ds)
        cps = MicroStructure(chip_window=2).cps(comp).dropna()  # 前 N 日 drift 无历史 → NaN
        assert ((cps >= -1.0) & (cps <= 1.0)).all()

    def test_cps_panic_100pct_polarity(self):
        """恐慌率 100% → s_panic 必须为 -1.0（越恐慌越空，修复旧公式越恐慌越加分）。"""
        idx = cn_minutes(["2024-01-02"], freq="30min")[:2]
        micro = MicroStructure(cps_weights=(0.0, 0.0, 1.0, 0.0))  # 只看恐慌分量
        comp = pd.DataFrame({
            "lock_ratio": [0.5, 0.5], "accum_delta": [0.0, 0.0],
            "drift": [0.0, 0.0], "panic_ratio": [1.0, 0.0],
        }, index=idx)
        cps = micro.cps(comp)
        assert cps.iloc[0] == pytest.approx(-1.0)   # panic=100% → -1（空头）
        assert cps.iloc[1] == pytest.approx(1.0)    # panic=0%   → +1（多头）

    def test_cps_missing_features_renorm(self):
        """缺项动态重归一化：某特征缺失时，其余特征自适应放大权重，
        与「该特征也不计权重」的重归一化理论值等价。"""
        idx = cn_minutes(["2024-01-02"], freq="30min")[:2]
        micro = MicroStructure()  # 默认 cps 权重 (0.3, 0.3, 0.2, 0.2)
        base = pd.DataFrame({
            "lock_ratio": [0.5, 0.5], "accum_delta": [0.05, 0.05],
            "panic_ratio": [0.25, 0.25], "drift": [0.0, np.nan],  # 第2行 drift 缺失
        }, index=idx)
        cps = micro.cps(base)
        w5, w6, w7, w8 = micro.w["w5"], micro.w["w6"], micro.w["w7"], micro.w["w8"]
        # 第1行：四分量齐全，den = w5+w6+w7+w8
        s1 = (w5 * (2 * 0.5 - 1) + w6 * (0.05 / 0.05)
              + w7 * (1 - 2 * 0.25) + w8 * (0.0 / 0.15))
        den1 = w5 + w6 + w7 + w8
        assert cps.iloc[0] == pytest.approx(float(np.clip(s1 / den1, -1.0, 1.0)))
        # 第2行：drift 缺失，den = w5+w6+w7，只对有效项重归一化
        s2 = (w5 * (2 * 0.5 - 1) + w6 * (0.05 / 0.05) + w7 * (1 - 2 * 0.25))
        den2 = w5 + w6 + w7
        assert cps.iloc[1] == pytest.approx(float(np.clip(s2 / den2, -1.0, 1.0)))


# ----------------------------------------------------------------------
# 环境共振
# ----------------------------------------------------------------------

class TestEnvironment:

    def test_global_mod_clip(self):
        dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
        axis = cn_minutes(dates, freq="30min")
        macro = mk_macro(dates)
        aligned = TimeAligner().align_external(macro, axis,
                                               [c for c in mk_macro(dates) if c != "trade_date"])
        env = Environment(score_window=3, min_periods=1)
        grs = env.grs(aligned)
        gmod = env.global_mod(grs)
        assert gmod.min() >= -0.8 and gmod.max() <= 0.8

    def test_chain_mod_clip(self):
        axis = cn_minutes(["2024-01-02"], freq="30min")
        irs = Environment(score_window=3, min_periods=1).irs(mk_industry(axis))
        cmod = Environment().chain_mod(irs)
        assert cmod.min() >= -0.3 and cmod.max() <= 0.3

    def test_global_mod_3sigma_gradient(self):
        """1.5σ → 0.5*max_gmod：风险梯度线性保留（未饱和）。"""
        env = Environment(max_gmod=0.8)
        out = env.global_mod(pd.DataFrame({"grs": pd.Series([1.5])}))
        assert np.isclose(out.iloc[0], 0.5 * 0.8)

    def test_global_mod_3sigma_saturated_at_max(self):
        """4.0σ → 锁死在 max_gmod：>3σ 饱和。"""
        env = Environment(max_gmod=0.8)
        out = env.global_mod(pd.DataFrame({"grs": pd.Series([4.0])}))
        assert np.isclose(out.iloc[0], 0.8)

    def test_chain_mod_3sigma_gradient_and_nan(self):
        """1.5σ → 0.5*max_cmod；NaN → 中性 0。"""
        env = Environment(max_cmod=0.3)
        out = env.chain_mod(pd.DataFrame({"irs": pd.Series([1.5, np.nan])}))
        assert np.isclose(out.iloc[0], 0.5 * 0.3)
        assert out.iloc[1] == 0.0

    def test_global_mod_nan_neutral(self):
        """缺失 GRS → 中性 0，不向 Final_MS / Target_Weight 传 NaN。"""
        env = Environment(max_gmod=0.8)
        out = env.global_mod(pd.DataFrame({"grs": pd.Series([np.nan])}))
        assert out.iloc[0] == 0.0

    def test_zscore_zero_vol_safe(self):
        """零波动防御：全 1.0 序列 std==0，靠 _EPS 保护安全返回 0.0，
        不抛 ZeroDivisionError、无 NaN"""
        env = Environment(score_window=240, min_periods=20)
        s = pd.Series(1.0, index=pd.RangeIndex(300))
        out = env._zscore(s)
        # 窗口未满(前 min_periods-1 行)为 NaN，窗口满后全 0
        assert out.dropna().shape[0] > 0
        assert (out.fillna(0.0) == 0.0).all()

    def test_mrs_weighted_renormalize_single_comp(self):
        """重归一化契约：单列有效 Z-score(3.0) + 三列全 NaN → 输出 3.0，
        权重让渡给唯一有效分量（下游 ES 负责 /3 压缩）"""
        env = Environment()
        idx = pd.RangeIndex(10)
        comps = {
            "ret": pd.Series([3.0] * 10, index=idx),
            "vol": pd.Series([np.nan] * 10, index=idx),
            "adr": pd.Series([np.nan] * 10, index=idx),
            "north": pd.Series([np.nan] * 10, index=idx),
        }
        out = env._weighted(comps, env.mrs_weights, ["ret", "vol", "adr", "north"])
        assert np.allclose(out.to_numpy(), 3.0)

    def test_grs_rate_polarity_risk_off(self):
        """美债极性：极度 Risk-Off（美股跌、商品跌、美债利率狂飙上行）下，
        GRS 必须为负。若美债未取反，利率上行会给正贡献显著削弱负值。"""
        n = 15
        t = np.arange(n)
        # 加速衰减 → 收益率逐期更负 → 尾部 z 稳定为负
        eq = 100.0 * np.exp(-0.05 * t * t)
        macro = pd.DataFrame(
            {"us_spx": eq, "us_ndx": eq, "us_dow": eq,
             "brent": eq * 0.98, "gold": eq * 1.02, "copper": eq * 0.99,
             # 美债收益率加速上行（Risk-Off 伴生）
             "us10y": 3.0 + 0.05 * t * t, "dxy": 100 + 2.0 * t},
            index=pd.date_range("2024-01-02", periods=n, freq="B"),
        )
        env = Environment(score_window=4, min_periods=2)
        raw_rate = env._zscore(macro["us10y"]).dropna()
        grs = env.grs(macro)["grs"].dropna()
        assert len(raw_rate) > 0 and len(grs) > 0
        # 利率上行 → 原始 z 为正；取反后 rate 分量必须为负
        assert float(raw_rate.iloc[-1]) > 0.0
        # 三路利空叠加 → GRS 为负，美债取反不抵消，反而加强负值
        g = float(grs.iloc[-1])
        assert g < -0.3, f"GRS 应强烈为负，实得 {g}"

    # ------------------------------------------------------------------
    # IRS：产业链共振（资金流 + 海外龙头 + 产业链指数，缺失分量权重让渡）
    # ------------------------------------------------------------------

    def test_irs_full_mapping_uses_all_weights(self):
        """用例1·全映射：资金流/海外/产业链指数三分量全有效，
        输出按完整配置权重线性加权（0.5/0.3/0.2）。"""
        axis = pd.date_range("2024-01-02 09:30", periods=60, freq="30min")
        n = len(axis)
        ind = pd.DataFrame(index=axis)
        ind["industry"] = "银行"
        ind["open"] = ind["high"] = ind["low"] = 1000.0
        ind["close"] = 100.0 + 0.05 * np.arange(n)           # 产业链指数上行
        ind["money_flow"] = np.arange(n) * 1e8               # 资金流上行
        macro = pd.DataFrame({"us_leader": 100.0 + 0.02 * np.arange(n)}, index=axis)
        mapping = {"银行": {"overseas_leader": "us_leader"}}
        env = Environment(score_window=10, min_periods=3,
                          irs_weights=(0.5, 0.3, 0.2))
        out = env.irs(ind, macro, mapping)["irs"]

        zf = env._zscore(ind["money_flow"].astype(float))
        zc = env._zscore(ind["close"].pct_change().fillna(0.0))
        zl = env._zscore(macro["us_leader"].pct_change().fillna(0.0))
        expected = env._weighted({"flow": zf, "leader": zl, "chain": zc},
                                 (0.5, 0.3, 0.2), ["flow", "leader", "chain"])
        mask = out.notna() & expected.notna()
        assert mask.sum() > 0
        assert np.allclose(out[mask].to_numpy(), expected[mask].to_numpy())

    def test_irs_no_overseas_renorm_to_flow_chain(self):
        """用例2·无海外：海外序列全 NaN → 不报错，且行业资金流与产业链指数
        瓜分 100% 权重（0.3 海外权重让渡给二者）。"""
        axis = pd.date_range("2024-01-02 09:30", periods=60, freq="30min")
        n = len(axis)
        ind = pd.DataFrame(index=axis)
        ind["industry"] = "银行"
        ind["open"] = ind["high"] = ind["low"] = 1000.0
        ind["close"] = 100.0 + 0.05 * np.arange(n)
        ind["money_flow"] = np.arange(n) * 1e8
        macro = pd.DataFrame({"us_leader": [np.nan] * n}, index=axis)
        mapping = {"银行": {"overseas_leader": "us_leader"}}
        env = Environment(score_window=10, min_periods=3,
                          irs_weights=(0.5, 0.3, 0.2))
        out = env.irs(ind, macro, mapping)["irs"]
        assert out.notna().sum() > 0  # 海外缺失不中断流程

        zf = env._zscore(ind["money_flow"].astype(float))
        zc = env._zscore(ind["close"].pct_change().fillna(0.0))
        expected = env._weighted({"flow": zf, "chain": zc},
                                 (0.5, 0.3, 0.2), ["flow", "leader", "chain"])
        mask = out.notna() & expected.notna()
        assert mask.sum() > 0
        assert np.allclose(out[mask].to_numpy(), expected[mask].to_numpy())

        # 显式重归一化：海外权重让渡后，资金流占 0.5/0.7、指数占 0.2/0.7
        w_flow, w_chain = 0.5 / 0.7, 0.2 / 0.7
        manual = w_flow * zf + w_chain * zc
        m2 = out.notna() & manual.notna()
        assert np.allclose(out[m2].to_numpy(), manual[m2].to_numpy())


# ----------------------------------------------------------------------
# FeatureEngine 端到端与防未来
# ----------------------------------------------------------------------

class TestFeatureEngine:

    def test_end_to_end_columns(self):
        dates = ["2024-01-02", "2024-01-03"]
        fe = FeatureEngine(symbol_to_industry={"600000": "银行"})
        out = fe.compute(full_slice(dates))
        assert list(out.columns) == FEATURE_COLS
        assert out.index.nunique() == len(cn_minutes(dates, freq="30min"))

    def test_margin_and_north_t_minus_1(self):
        dates = ["2024-01-02", "2024-01-03"]
        ds = full_slice(dates)
        # 北向/两融表带一个「更早的前导交易日」：使其在 01-02 行即可算出一阶值，
        # 从而能验证 T-1 可见性（01-02 行只被 01-03 使用，01-02 当日不可见）
        nm_days = ["2023-12-29", "2024-01-02", "2024-01-03"]
        nm = pd.DataFrame(index=pd.to_datetime(nm_days))
        nm["symbol"] = "600000"
        nm["trade_date"] = pd.to_datetime(nm_days)
        nm["north_holding"] = [1e7, 1.05e7, 1.02e7]
        nm["north_buy_net"] = [0.0, 1e6, -5e5]
        nm["margin_fin_balance"] = [1e9, 1.05e9, 1.1e9]
        nm["margin_sec_balance"] = [5e8, 5.1e8, 5.15e8]
        ds = DataSlice(
            kline=ds.kline, l2_snapshot=ds.l2_snapshot, tick_trades=ds.tick_trades,
            index_min=ds.index_min, breadth=ds.breadth, industry=ds.industry,
            macro=ds.macro, north_margin=nm, dragon_tiger=ds.dragon_tiger,
            meta=ds.meta,
        )
        fe = FeatureEngine(
            symbol_to_industry={"600000": "银行"},
            agent=AgentProfiling(holding_days=1, margin_days=1),
        )
        out = fe.compute(ds)
        d1, d2 = pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")
        # T 日两融/北向数据 T+1 才可见：01-02 当日取 12-29 行（其本身无历史 → NaN），
        # 01-03 起取 01-02 行（一阶值非 NaN）
        assert out.loc[out.index.normalize() == d1, "margin_pressure"].isna().all()
        assert out.loc[out.index.normalize() == d2, "margin_pressure"].notna().any()
        assert out.loc[out.index.normalize() == d1, "north_sync"].isna().all()
        assert out.loc[out.index.normalize() == d2, "north_sync"].notna().any()

    def test_dragon_tiger_t_plus_1(self):
        dates = ["2024-01-02", "2024-01-03"]
        fe = FeatureEngine(symbol_to_industry={"600000": "银行"})
        out = fe.compute(full_slice(dates))
        d1, d2 = pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")
        # T 日榜单 T+1 才可用：D1 无 dt_net，D2 起 = net_amount
        assert out.loc[out.index.normalize() == d1, "dt_net"].isna().all()
        d2_vals = out.loc[out.index.normalize() == d2, "dt_net"]
        assert d2_vals.notna().all() and (d2_vals == 5e7).all()
        assert (out.loc[out.index.normalize() == d2, "dt_avail"] == d2).all()

    def test_missing_data_degrades(self, caplog):
        import logging
        axis = cn_minutes(["2024-01-02"], freq="30min")
        ds = DataSlice(kline=mk_kline(axis))  # 仅 kline
        with caplog.at_level(logging.WARNING):
            out = FeatureEngine().compute(ds)
        assert list(out.columns) == FEATURE_COLS
        assert out["ofss"].isna().all()          # 无 tick/l2
        assert out["north_sync"].isna().all()    # 无 north_margin
        assert any("缺少" in r.message for r in caplog.records)

    def test_global_mod_in_range_end_to_end(self):
        dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        ds = full_slice(dates)
        # 逐笔覆盖全部交易日 + chip_window=2：drift 第 3 日起非 NaN，
        # 经 T-1 对齐后第 4 日 CPS 才有值
        ds = DataSlice(
            kline=ds.kline, l2_snapshot=ds.l2_snapshot,
            tick_trades=mk_ticks([
                ("2024-01-02 09:30", 10.0, 1000, 5e5, 1, False),
                ("2024-01-03 09:30", 10.0, 1000, 5e5, 1, False),
                ("2024-01-04 09:30", 10.0, 1000, 5e5, 1, False),
                ("2024-01-05 09:30", 10.0, 1000, 5e5, 1, False),
            ]),
            index_min=ds.index_min, breadth=ds.breadth, industry=ds.industry,
            macro=ds.macro, north_margin=ds.north_margin,
            dragon_tiger=ds.dragon_tiger, meta=ds.meta,
        )
        fe = FeatureEngine(symbol_to_industry={"600000": "银行"},
                           micro=MicroStructure(chip_window=2))
        out = fe.compute(ds)
        tail = out.loc[out.index.normalize() > "2024-01-02"]
        assert tail["global_mod"].min() >= -0.8
        assert tail["global_mod"].max() <= 0.8
        assert tail["cps"].notna().any() and tail["ofss"].notna().any()
