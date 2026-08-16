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
        # retail = 3e4 + 0.5*(-8e4) = -1e4；inst = 2e6 + 0.5*(-8e4) = 1.96e6；youzi = 5e5
        assert row["retail_flow"] == pytest.approx(-1e4 / 1e6)
        assert row["inst_flow"] == pytest.approx(1.96e6 / 1e6)
        assert row["youzi_flow"] == pytest.approx(5e5 / 1e6)

    def test_custom_thresholds(self):
        ticks = mk_ticks([
            ("2024-01-02 09:30", 10.0, 100, 3e4, 1, False),   # 超过 custom small(2w) → 非小单
        ])
        norm = pd.DataFrame({"ts": [pd.Timestamp("2024-01-02 09:30")],
                             "symbol": "600000", "norm_base": [1.0]})
        norm = norm.set_index("ts")
        out = AgentProfiling(small_th=2e4).net_flows(ticks, norm)
        # 3w ≥ 2w 且 < 20w → 中单 → retail 含 0.5*3e4
        assert out.iloc[0]["retail_flow"] == pytest.approx(0.5 * 3e4)

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
        expect = (0.3 * (100 / 300) + 0.3 * 0.5 + 0.2 * (0.2 - 50 / 250)
                  + 0.2 * (3e5 / 3.1e5))
        assert ofss == pytest.approx(expect)

    def test_ofss_clip_range(self):
        ds = self._ds_with_orderflow()
        # 权重 4*OBI(1/3) ≈ 1.33 > 1 → 触发上界 clip
        micro = MicroStructure(ofss_weights=(4.0, 0, 0, 0))
        ofss = micro.ofss(micro.ofss_components(ds))
        assert ofss.iloc[0] == pytest.approx(1.0)

    def test_pss_range_and_direction(self):
        axis = cn_minutes(["2024-01-02"], freq="30min")
        kline = mk_kline(axis)
        pss = MicroStructure(pss_window=3).pss(kline).dropna()
        assert ((pss >= -1.0) & (pss <= 1.0)).all()

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
