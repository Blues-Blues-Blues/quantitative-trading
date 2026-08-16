"""统一特征计算与归一化调度器（FeatureEngine）。

输入 DataSlice，输出覆盖全部特征列的个股长表 DataFrame
（DatetimeIndex + symbol 列），列集合见 FEATURE_COLS。

调度流程：
1. TimeAligner 对齐：macro T-1、north_margin/两融 T-1、龙虎榜 T+1（avail_date）
2. 分钟级因子（当日可用）：资金流（retail/inst/youzi）、OFSS 成分与综合、
   PSS、MRS、GRS、IRS
3. 日频因子（T-1 对齐后 ffill 到分钟轴）：North_Sync、Margin_Pressure、
   CPS 成分与综合
4. 龙虎榜 T+1 因子（dt_net），并用 verify_no_lookahead 断言
5. 组装长表：任何数据表缺失时，对应特征列置 NaN 并告警，绝不中断

所有权重与窗口参数由 AgentProfiling / MicroStructure / Environment 构造
入参注入，便于 Optuna 超参数寻优。
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from data.aligner import TimeAligner
from data.dataslice import SYMBOL, TRADE_DATE, DataSlice
from indicators.agent_profiling import AgentProfiling
from indicators.environment import Environment
from indicators.microstructure import MicroStructure

logger = logging.getLogger("indicators.feature_engine")

FEATURE_COLS: List[str] = [
    SYMBOL,
    # 资金主体分层
    "retail_flow", "inst_flow", "youzi_flow",
    "north_sync", "margin_pressure",
    # 订单流与筹码微观结构
    "obi", "ar", "cancel_ratio", "big_flow", "ofss",
    "lock_ratio", "accum_delta", "panic_ratio", "drift", "cps",
    "pss",
    # 宏观与行业环境共振
    "mrs", "grs", "irs", "global_mod", "chain_mod",
    # 龙虎榜（T+1 可用）
    "dt_net",
]

# 日频因子的值列（chip / 北向 / 两融）
_CHIP_COLS = ["lock_ratio", "accum_delta", "panic_ratio", "drift"]


class FeatureEngine:
    """特征计算调度器。

    :param agent:   资金主体分层计算器（缺省 AgentProfiling()）
    :param micro:   微观结构计算器（缺省 MicroStructure()）
    :param env:     环境共振计算器（缺省 Environment()）
    :param aligner: 时间对齐器（缺省 TimeAligner()）
    :param symbol_to_industry: {个股代码: 行业名}，用于把 IRS 映射到个股；
        不提供时 irs / chain_mod 置 NaN
    """

    def __init__(
        self,
        agent: Optional[AgentProfiling] = None,
        micro: Optional[MicroStructure] = None,
        env: Optional[Environment] = None,
        aligner: Optional[TimeAligner] = None,
        symbol_to_industry: Optional[Dict[str, str]] = None,
    ) -> None:
        self.agent = agent or AgentProfiling()
        self.micro = micro or MicroStructure()
        self.env = env or Environment()
        self.aligner = aligner or TimeAligner()
        self.symbol_to_industry = symbol_to_industry or {}

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------

    def compute(self, ds: DataSlice) -> pd.DataFrame:
        """计算全部特征并组装为个股长表。

        :return: DatetimeIndex + 列 = FEATURE_COLS
        """
        aligned = self.aligner.align_slice(ds)
        axis = aligned.time_axis()

        # 基准长表（分钟轴 × 标的）
        base = aligned.kline.reset_index()
        base.columns = ["ts"] + list(base.columns[1:])
        feat = base[["ts", SYMBOL]].copy()

        # ---- 分钟级因子 ----
        feat = self._minute_factors(feat, aligned)

        # ---- 日频因子（T-1 对齐 → ffill 分钟）----
        feat = self._daily_factors(feat, aligned, axis)

        # ---- 环境共振 ----
        feat = self._environment_factors(feat, aligned)

        # ---- 龙虎榜 T+1 因子 + 防未来断言 ----
        feat = self._dragon_tiger(feat, aligned, axis)

        # 固定列顺序输出
        return feat.set_index("ts").reindex(columns=FEATURE_COLS)

    # ------------------------------------------------------------------
    # 分钟级因子
    # ------------------------------------------------------------------

    def _minute_factors(self, feat: pd.DataFrame, ds: DataSlice) -> pd.DataFrame:
        # 资金流（含归一化基准）
        if ds.tick_trades is not None and not ds.tick_trades.empty:
            norm_base = self._norm_base(ds.kline, ds.time_axis())
            flows = self.agent.net_flows(ds.tick_trades, norm_base)
            feat = feat.merge(flows.reset_index().rename(columns={"index": "ts"}),
                              on=["ts", SYMBOL], how="left")
        else:
            logger.warning("缺少 tick_trades，retail/inst/youzi_flow 置 NaN")

        # OFSS 成分与综合
        if (ds.l2_snapshot is not None and not ds.l2_snapshot.empty) \
                or (ds.tick_trades is not None and not ds.tick_trades.empty):
            comp = self.micro.ofss_components(ds)
            comp["ofss"] = self.micro.ofss(comp)
            comp = comp.reset_index().rename(columns={"index": "ts"})
            feat = feat.merge(comp[["ts", SYMBOL, "obi", "ar", "cancel_ratio",
                                    "big_flow", "ofss"]],
                              on=["ts", SYMBOL], how="left")
        else:
            logger.warning("缺少 l2_snapshot 且缺少 tick_trades，OFSS 相关列置 NaN")

        # PSS（与 kline 同 index 的 Series）
        pss = self.micro.pss(ds.kline).rename("pss").reset_index()
        pss = pss.rename(columns={"index": "ts"})
        feat = feat.merge(pss, on=["ts", SYMBOL], how="left")
        return feat

    # ------------------------------------------------------------------
    # 日频因子（T-1 对齐后 ffill 到分钟轴）
    # ------------------------------------------------------------------

    def _daily_factors(self, feat: pd.DataFrame, ds: DataSlice,
                       axis: pd.DatetimeIndex) -> pd.DataFrame:
        # 北向 / 两融
        if ds.north_margin is not None and not ds.north_margin.empty:
            ns = self.agent.north_sync(ds.north_margin)
            mp = self.agent.margin_pressure(ds.north_margin)
            daily = ns.merge(mp, on=[TRADE_DATE, SYMBOL], how="outer")
            aligned_d = self._ffill_daily(daily, axis, ["north_sync", "margin_pressure"])
            feat = feat.merge(aligned_d, on=["ts", SYMBOL], how="left")
        else:
            logger.warning("缺少 north_margin，north_sync / margin_pressure 置 NaN")

        # CPS 筹码分量
        if ds.tick_trades is not None and not ds.tick_trades.empty:
            chip = self.micro.chip_components(ds)
            aligned_c = self._ffill_daily(chip, axis, _CHIP_COLS)
            aligned_c["cps"] = self.micro.cps(aligned_c)
            feat = feat.merge(aligned_c[["ts", SYMBOL] + _CHIP_COLS + ["cps"]],
                              on=["ts", SYMBOL], how="left")
        else:
            logger.warning("缺少 tick_trades，CPS 相关列置 NaN")
        return feat

    def _ffill_daily(self, daily: pd.DataFrame, axis: pd.DatetimeIndex,
                     value_cols: List[str]) -> pd.DataFrame:
        """日频长表按 T-1 可用时点对齐并 ffill 到分钟轴。"""
        rows = []
        for sym, g in daily.groupby(SYMBOL):
            a = self.aligner.align_external(g, axis, value_cols,
                                            date_col=TRADE_DATE)
            a[SYMBOL] = sym
            rows.append(a)
        out = pd.concat(rows)
        return out.reset_index().rename(columns={"index": "ts"})

    def _norm_base(self, kline: pd.DataFrame, axis: pd.DatetimeIndex) -> pd.DataFrame:
        """资金流归一化基准（日频 → T-1 对齐 → 分钟轴长表 [ts, symbol, norm_base]）。"""
        k = kline.copy()
        k["day"] = k.index.normalize()
        if self.agent.norm_by == "amount":
            daily = k.groupby(["day", SYMBOL])["amount"].sum().reset_index()
            daily = daily.rename(columns={"day": TRADE_DATE, "amount": "norm_base"})
            daily["norm_base"] = (
                daily.groupby(SYMBOL)["norm_base"]
                .transform(lambda s: s.rolling(self.agent.norm_days,
                                               min_periods=1).mean()))
        else:  # mcap：流通市值按日末值
            daily = k.groupby(["day", SYMBOL])["float_market_cap"].last().reset_index()
            daily = daily.rename(columns={"day": TRADE_DATE,
                                          "float_market_cap": "norm_base"})
        return self._ffill_daily(daily, axis, ["norm_base"])

    # ------------------------------------------------------------------
    # 环境共振
    # ------------------------------------------------------------------

    def _environment_factors(self, feat: pd.DataFrame, ds: DataSlice) -> pd.DataFrame:
        # MRS（取首个指数代码作为全市场状态）
        mrs = self.env.mrs(ds.index_min, ds.breadth)
        if not mrs.empty:
            code = mrs["index_code"].iloc[0]
            mrs = mrs[mrs["index_code"] == code][["mrs"]].reset_index()
            mrs = mrs.rename(columns={"index": "ts"})
            feat = feat.merge(mrs, on="ts", how="left")
        else:
            logger.warning("MRS 为空，置 NaN")

        # GRS / Global_Mod（macro 已 T-1 对齐）
        grs = self.env.grs(ds.macro)
        if not grs.empty:
            grs["global_mod"] = self.env.global_mod(grs)
            grs = grs.reset_index().rename(columns={"index": "ts"})
            feat = feat.merge(grs, on="ts", how="left")
        else:
            logger.warning("GRS 为空，grs / global_mod 置 NaN")

        # IRS / Chain_Mod（需 symbol→industry 映射）
        if self.symbol_to_industry:
            irs = self.env.irs(ds.industry, ds.macro)
            if not irs.empty:
                irs["chain_mod"] = self.env.chain_mod(irs)
                irs = irs.reset_index().rename(columns={"index": "ts"})
                map_df = pd.DataFrame({
                    SYMBOL: list(self.symbol_to_industry.keys()),
                    "industry": list(self.symbol_to_industry.values()),
                })
                per_symbol = irs.merge(map_df, on="industry", how="inner")
                feat = feat.merge(per_symbol.drop(columns="industry"),
                                  on=["ts", SYMBOL], how="left")
                return feat
            logger.warning("IRS 为空，irs / chain_mod 置 NaN")
        else:
            logger.warning("未提供 symbol_to_industry，irs / chain_mod 置 NaN")
        feat["irs"] = pd.NA
        feat["chain_mod"] = pd.NA
        return feat

    # ------------------------------------------------------------------
    # 龙虎榜 T+1 因子
    # ------------------------------------------------------------------

    def _dragon_tiger(self, feat: pd.DataFrame, ds: DataSlice,
                      axis: pd.DatetimeIndex) -> pd.DataFrame:
        if ds.dragon_tiger is None or ds.dragon_tiger.empty:
            feat["dt_net"] = pd.NA
            return feat

        d = ds.dragon_tiger.dropna(subset=["avail_date"]).copy()
        if d.empty:
            feat["dt_net"] = pd.NA
            return feat
        agg = d.groupby([SYMBOL, "avail_date"])["net_amount"].sum().reset_index()

        # avail_date（披露次日 00:00）起可用：按分钟轴日期 reindex + ffill
        rows = []
        for sym, g in agg.groupby(SYMBOL):
            s = g.set_index("avail_date")["net_amount"]
            mapped = s.reindex(axis.normalize(), method="ffill")
            rows.append(pd.DataFrame(
                {"ts": axis, SYMBOL: sym,
                 "dt_net": mapped.to_numpy(),
                 "dt_avail": mapped.index.to_numpy()}))
        out = pd.concat(rows)

        # 防未来函数断言：每行使用时间(ts)必须晚于其数据可用日
        TimeAligner.verify_no_lookahead(out["ts"], out["dt_avail"],
                                        name="dragon_tiger/dt_net")
        feat = feat.merge(out[["ts", SYMBOL, "dt_net"]], on=["ts", SYMBOL],
                          how="left")
        return feat
