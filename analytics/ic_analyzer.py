"""横截面因子有效性分析（多因子 Rank IC / Normal IC / IC_IR / 分层收益）。

面向真实多标的股票池（横截面评价，非逐股判定），三个模块：

1. UniverseFilter：动态成分股过滤（防幸存者偏差）
   - ST/退市：静态标记全程剔除（当前数据无历史状态表，降级实现）
   - 停牌：当日成交量为 0 → 当日剔除（动态）
   - 次新股：上市未满 60 自然日 → 剔除（动态）
2. 日频截面前向收益：每交易日 14:30 截面
   - 15:00 收盘后无 5min/30min 未来收益，14:30 保证三个窗口都有真实意义：
     * fwd_5m  = 14:35 close / 14:30 close - 1
     * fwd_30m = 15:00 close / 14:30 close - 1
     * fwd_1d  = 次日 14:30 close / 当日 14:30 close - 1（跨交易日 shift，跳过周末）
3. FactorEvaluator / QuantileLayering：
   - 逐日横截面 Rank IC（Spearman）与 Normal IC（Pearson）
   - IC_IR = mean(IC) / std(IC)、t 值、胜率（IC>0 占比）
   - Q1~Q5 分位分层组合净值（等权），Q5-Q1 价差与档位单调性

主入口 run_ic_analysis()：合成列（SignalSynthesizer.synthesize）→ 成分股过滤
→ 面板拼接 → IC 汇总 + 分层净值。
"""

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from data.dataslice import SYMBOL, TRADE_DATE
from strategy.signals import SignalSynthesizer

logger = logging.getLogger("analytics.ic_analyzer")

# 横截面最少有效标的数（少于该值当日不参与 IC）
MIN_CROSS = 5
# 次新股过滤窗口：上市未满 N 自然日剔除
MIN_LIST_DAYS = 60
# 日频截面时点（HHMM）
_SNAP_HM = 1430
# 默认候选因子（均存在于 synthesize 输出或特征表）
DEFAULT_FACTORS: List[str] = [
    "final_ms", "ofss", "cps", "capital_purity", "retail_chase",
    "inst_flow", "lock_ratio", "pss", "mrs", "rs", "industry_ms",
]


class UniverseFilter:
    """动态成分股过滤（防幸存者偏差）。

    :param kline:        标准分钟 K 线长表（DatetimeIndex + symbol + volume）
    :param stock_basic:  load_stock_basic_full() 全表（代码/上市日期/ST标记）；
        缺省/缺列时对应过滤项自动降级跳过
    """

    def __init__(self, kline: pd.DataFrame,
                 stock_basic: Optional[pd.DataFrame] = None) -> None:
        self.kline = kline
        self.stock_basic = stock_basic

    def daily_universe(self) -> pd.DataFrame:
        """每交易日成分股长表 [trade_date, symbol]（当日可交易的股票）。"""
        k = self.kline.copy()
        k["day"] = k.index.normalize()
        uni = (k.groupby(["day", SYMBOL])["volume"].sum()
               .reset_index().rename(columns={"day": TRADE_DATE}))
        uni.columns = [TRADE_DATE, SYMBOL, "vol"]

        # 停牌：全天成交量为 0（无成交）→ 当日剔除
        uni = uni[uni["vol"] > 0].copy()
        n_before = len(uni)

        # 次新股：上市未满 60 自然日 → 剔除
        if self.stock_basic is not None and "上市日期" in self.stock_basic.columns:
            ld = dict(zip(self.stock_basic["代码"],
                          pd.to_datetime(self.stock_basic["上市日期"],
                                         errors="coerce")))
            uni["list_date"] = uni[SYMBOL].map(ld)
            uni = uni[uni["list_date"].isna()
                      | (uni[TRADE_DATE] >= uni["list_date"]
                         + pd.Timedelta(days=MIN_LIST_DAYS))]
            logger.info("成分股过滤：次新股剔除 %d 行（上市未满 %d 天）",
                        n_before - len(uni), MIN_LIST_DAYS)
            n_before = len(uni)

        # ST/退市：静态标记全程剔除（无历史状态表，降级）
        if self.stock_basic is not None and "ST标记" in self.stock_basic.columns:
            st = dict(zip(self.stock_basic["代码"],
                          self.stock_basic["ST标记"].astype(bool)))
            uni["is_st"] = uni[SYMBOL].map(st).fillna(False)
            uni = uni[~uni["is_st"]].copy()
            logger.info("成分股过滤：ST 标记剔除 %d 行（静态快照，降级）",
                        n_before - len(uni))

        logger.info("成分股过滤：%d 个交易日 × 平均 %.1f 只（原始 %d 行 → 保留 %d 行）",
                    uni[TRADE_DATE].nunique(),
                    uni.groupby(TRADE_DATE).size().mean(),
                    int(uni["vol"].notna().sum()) + n_before, len(uni))
        return uni[[TRADE_DATE, SYMBOL]].reset_index(drop=True)


def forward_returns(kline: pd.DataFrame,
                    snap_hm: int = _SNAP_HM) -> pd.DataFrame:
    """日频截面前向收益长表。

    :return: [ts(14:30), symbol, close, fwd_5m, fwd_30m, fwd_1d]
    """
    k = kline[[SYMBOL, "close"]].copy()
    k["day"] = k.index.normalize()
    k["hm"] = k.index.hour * 100 + k.index.minute

    def close_at(hm_limit: int) -> pd.Series:
        """每 (day, symbol) 取 <= hm_limit 的最后一根 close（组内时间升序）。"""
        return (k[k["hm"] <= hm_limit]
                .groupby(["day", SYMBOL])["close"].last())

    c14 = close_at(snap_hm)
    wide = c14.reset_index()
    wide = wide.pivot(index="day", columns=SYMBOL, values="close").sort_index()
    # 次日 14:30 close：按交易日索引 shift（跨周末/节假日自动对齐）
    next_close = wide.shift(-1)

    fwd = pd.DataFrame({"close": wide.stack()})
    fwd["fwd_5m"] = (close_at(snap_hm + 5).reindex(fwd.index) / fwd["close"]) - 1.0
    fwd["fwd_30m"] = (close_at(1500).reindex(fwd.index) / fwd["close"]) - 1.0
    fwd["fwd_1d"] = (next_close.stack().reindex(fwd.index) / fwd["close"]) - 1.0
    fwd["ts"] = fwd.index.get_level_values("day") + pd.Timedelta(
        hours=snap_hm // 100, minutes=snap_hm % 100)
    return fwd.reset_index()[["ts", SYMBOL, "close",
                              "fwd_5m", "fwd_30m", "fwd_1d"]]


def build_panel(syn_features: pd.DataFrame, kline: pd.DataFrame,
                universe: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """合成特征长表 + 前向收益 + 成分股过滤 → 日频截面面板。

    :param syn_features: SignalSynthesizer.synthesize() 输出（含 final_ms 等合成列）
    """
    fwd = forward_returns(kline)
    panel = syn_features.reset_index().rename(columns={"index": "ts"})
    panel["day"] = panel["ts"].dt.normalize()
    panel = panel.merge(fwd, on=["ts", SYMBOL], how="inner")
    if universe is not None and not universe.empty:
        uni = universe.rename(columns={TRADE_DATE: "day"})
        panel = panel.merge(uni.assign(in_universe=True),
                            on=["day", SYMBOL], how="left")
        panel = panel[panel["in_universe"] == True]  # noqa: E712
    return panel


def ic_analysis(panel: pd.DataFrame, factors: Sequence[str],
                horizons: Sequence[str] = ("fwd_5m", "fwd_30m", "fwd_1d"),
                min_cross: int = MIN_CROSS):
    """逐日横截面 Rank IC / Normal IC / IC_IR。

    :return: (ic_summary DataFrame, {factor|horizon: IC 序列 Series})
    """
    p = panel.copy()
    p["day"] = p["ts"].dt.normalize()
    out: List[Dict[str, object]] = []
    ic_series: Dict[str, pd.Series] = {}

    for fac in factors:
        if fac not in p.columns:
            logger.warning("IC 跳过：因子 %s 不在面板中", fac)
            continue
        for h in horizons:
            if h not in p.columns:
                continue
            wf = p.pivot(index="day", columns=SYMBOL, values=fac)
            wr = p.pivot(index="day", columns=SYMBOL, values=h)
            rank_ic, norm_ic, days = [], [], []
            for day in wf.index:
                f, r = wf.loc[day], wr.loc[day]
                m = f.notna() & r.notna()
                if int(m.sum()) < min_cross:
                    continue
                fr, rr = f[m].astype(float), r[m].astype(float)
                if fr.nunique() < 2 or rr.nunique() < 2:
                    continue
                rank_ic.append(stats.spearmanr(fr, rr).statistic)
                norm_ic.append(stats.pearsonr(fr, rr).statistic)
                days.append(day)
            if len(rank_ic) < 2:
                logger.info("IC 样本不足：%s × %s（%d 期 < 2）", fac, h, len(rank_ic))
                continue
            s = pd.Series(rank_ic, index=pd.DatetimeIndex(days), name=f"{fac}|{h}")
            s.index.name = "day"
            ic_series[f"{fac}|{h}"] = s
            mean, std = float(np.mean(rank_ic)), float(np.std(rank_ic, ddof=1))
            out.append({
                "factor": fac, "horizon": h, "n_days": len(s),
                "rank_ic": mean, "ic_ir": mean / std if std > 0 else np.nan,
                "t_stat": mean / (std / np.sqrt(len(s))) if std > 0 else np.nan,
                "win_rate": float(np.mean(np.array(rank_ic) > 0)),
                "normal_ic": float(np.mean(norm_ic)),
            })
    summary = pd.DataFrame(out).sort_values(["factor", "horizon"])
    return summary, ic_series


def _safe_qcut(s: pd.Series, n_q: int) -> pd.Series:
    """按日分层 qcut 的容错封装：样本不足 / 截面常量时整列置 NaN（跳过该日）。

    qcut 在「该日截面样本 < n_q」或「因子值近似常量（如市场级 mrs）」时
    抛 Bin edges 异常，不能让单日异常中断整轮 IC 分层分析。
    """
    if len(s) < 2 or s.nunique() < 2:
        return pd.Series(np.nan, index=s.index)
    try:
        return pd.qcut(s, n_q, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(np.nan, index=s.index)


def quantile_analysis(panel: pd.DataFrame, factors: Sequence[str],
                      horizons: Sequence[str] = ("fwd_1d", "fwd_30m"),
                      n_q: int = 5):
    """Q1~Q5 分位分层组合（每期等权）净值与单调性。

    :return: (quantile_summary DataFrame, {factor|horizon: 档位净值 DataFrame})
    """
    p = panel.copy()
    p["day"] = p["ts"].dt.normalize()
    out: List[Dict[str, object]] = []
    navs: Dict[str, pd.DataFrame] = {}

    for fac in factors:
        if fac not in p.columns:
            continue
        for h in horizons:
            if h not in p.columns:
                continue
            rows = p[["day", SYMBOL, fac, h]].dropna(subset=[fac, h])
            if rows.empty:
                continue
            q = (rows.groupby("day")[fac]
                 .transform(lambda s: _safe_qcut(s, n_q)))
            rows = rows.assign(q=q).dropna(subset=["q"])
            if rows.empty:
                # 因子截面近似常量（如市场级 mrs）→ qcut 无法分档，跳过
                logger.info("分层跳过：%s × %s（qcut 无法分档，因子截面常量）",
                            fac, h)
                continue
            # 各档每日等权收益 → 净值
            g = (rows.groupby(["day", "q"])[h].mean()
                 .unstack("q").sort_index().fillna(0.0))
            nav = (1.0 + g).cumprod()
            q_means = g.mean()
            # 方向判定：档位与平均收益的单调相关（+1 单调递增 / -1 单调递减）
            mono = stats.spearmanr(np.arange(len(q_means)), q_means).statistic
            out.append({
                "factor": fac, "horizon": h,
                **{f"Q{i + 1}": float(q_means.loc[i]) if i in q_means.index else np.nan
                   for i in range(n_q)},
                "spread": float(q_means.iloc[-1] - q_means.iloc[0]),
                "monotonic": float(mono),
                "nav_ratio": float(nav.iloc[-1, -1] / nav.iloc[-1, 0])
                if not nav.empty else np.nan,
            })
            navs[f"{fac}|{h}"] = nav
    summary = pd.DataFrame(out).sort_values(["factor", "horizon"])
    return summary, navs


def run_ic_analysis(ds, features: pd.DataFrame,
                    symbol_to_industry: Dict[str, str],
                    params: Dict[str, object],
                    factors: Optional[Sequence[str]] = None,
                    stock_basic: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    """便捷入口：合成列 → 成分股过滤 → 面板 → IC + 分层。

    :param ds:       对齐后的 DataSlice（提供 kline）
    :param features: FeatureEngine.compute() 输出的特征长表
    :param params:   与回测一致的信号参数（SignalSynthesizer 构造）
    :return: dict(ic_summary, ic_series, quantile_summary, quantile_nav, panel)
    """
    syn = SignalSynthesizer(
        weights=tuple(params["weights"]),
        inst_window=int(params["inst_window"]),
        th_ms_bull=float(params["th_ms_bull"]),
        th_ms_exit=float(params["th_ms_exit"]),
        th_lock=float(params["th_lock"]),
        th_purity=float(params["th_purity"]),
        th_global_min=float(params["th_global_min"]),
        th_adr_min=float(params["th_adr_min"]),
        th_mrs_min=float(params.get("th_mrs_min", 0.0)),
        th_industry_min=float(params.get("th_industry_min", 0.0)),
        win_hold_max=int(params["win_hold_max"]),
        symbol_to_industry=symbol_to_industry,
    )
    syn_features = syn.synthesize(ds, features)
    universe = UniverseFilter(ds.kline, stock_basic).daily_universe()
    panel = build_panel(syn_features, ds.kline, universe)
    fcs = list(factors) if factors else list(DEFAULT_FACTORS)
    ic_summary, ic_series = ic_analysis(panel, fcs)
    q_summary, q_nav = quantile_analysis(panel, fcs)
    return {
        "panel": panel, "universe": universe,
        "ic_summary": ic_summary, "ic_series": ic_series,
        "quantile_summary": q_summary, "quantile_nav": q_nav,
    }
