"""资金主体分层因子（Agent Profiling）。

按单笔成交额将逐笔成交划分为 小单(<4w) / 中单(4~20w) / 大单(20~100w) /
超大单(>100w)，聚合出分钟级净流并按个股归一化；另提供北向共振与两融压力
两个日频因子（数据 T+1 披露，必须由调用方先做可用时点对齐，见
FeatureEngine / TimeAligner）。

因子定义：
    Retail_Flow = Norm(小单净流 + 0.5 * 中单净流)
    Inst_Flow   = Norm(超大单净流 + 0.5 * 中单净流)   # 考虑机构算法拆单
    Youzi_Flow  = Norm(大单净流)
    North_Sync  = 0.6*sign(北向20日持仓变化) + 0.4*sign(北向当日净买)
    Margin_Pressure = 融资5日变化率 - 融券5日变化率

归一化口径（norm_by）："amount"= 个股最近 N 日均成交额；"mcap"= 流通市值。
所有阈值与窗口均为初始化入参，便于超参数寻优。
"""

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from data.dataslice import SYMBOL, TRADE_DATE

logger = logging.getLogger("indicators.agent_profiling")


class AgentProfiling:
    """资金主体分层与资金流因子。

    :param small_th:  小单上限（元），默认 4 万
    :param medium_th: 中单上限（元），默认 20 万
    :param large_th:  大单上限（元），默认 100 万
    :param norm_days: 归一化窗口（最近 N 日）
    :param norm_by:   归一化基准："amount"（N 日均成交额）或 "mcap"（流通市值）
    :param holding_days: 北向持仓变化窗口（日），默认 20
    :param margin_days:  两融变化窗口（日），默认 5
    :param north_weights: North_Sync 的 (持仓变化, 当日净买) 权重
    :param mid_split:     中单对机构/散户的归属系数（默认 0.5）
    """

    def __init__(
        self,
        small_th: float = 4e4,
        medium_th: float = 2e5,
        large_th: float = 1e6,
        norm_days: int = 20,
        norm_by: str = "amount",
        holding_days: int = 20,
        margin_days: int = 5,
        north_weights: Sequence[float] = (0.6, 0.4),
        mid_split: float = 0.5,
    ) -> None:
        if norm_by not in ("amount", "mcap"):
            raise ValueError("norm_by 仅支持 'amount' 或 'mcap'")
        self.small_th = small_th
        self.medium_th = medium_th
        self.large_th = large_th
        self.norm_days = norm_days
        self.norm_by = norm_by
        self.holding_days = holding_days
        self.margin_days = margin_days
        self.north_weights = tuple(north_weights)
        self.mid_split = mid_split

    # ------------------------------------------------------------------
    # 分钟级资金流
    # ------------------------------------------------------------------

    def net_flows(
        self,
        ticks: pd.DataFrame,
        norm_base: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """按单笔成交额分档聚合分钟级净流并归一化。

        :param ticks: 逐笔长表（DatetimeIndex + symbol/turnover/side 列）
        :param norm_base: 归一化基准长表（DatetimeIndex + symbol + norm_base 列，
            已按可用时点填充到分钟轴）；为 None 时不做归一化并告警
        :return: 长表 [symbol, retail_flow, inst_flow, youzi_flow]
        """
        t = ticks.copy()
        if "is_cancel" not in t.columns:
            t["is_cancel"] = False
        # 撤单记录不参与资金流统计
        t = t[~t["is_cancel"]]
        if t.empty:
            return pd.DataFrame(columns=[SYMBOL, "retail_flow", "inst_flow", "youzi_flow"])

        t["bucket"] = np.select(
            [t["turnover"] < self.small_th,
             t["turnover"] < self.medium_th,
             t["turnover"] < self.large_th],
            ["small", "med", "large"],
            default="mega",
        )
        t["net"] = t["side"] * t["turnover"]

        # 分钟 × 标的分档净流
        g = (
            t.groupby([t.index, t[SYMBOL], t["bucket"]])["net"]
            .sum()
            .unstack("bucket")
        )
        # 数据中可能缺失某些分档（如自定义阈值后无小单），补齐全部桶
        g = g.reindex(columns=["small", "med", "large", "mega"], fill_value=0.0)
        g = g.reset_index().rename(columns={"level_0": "ts"})

        retail = g["small"] + self.mid_split * g["med"]
        inst = g["mega"] + self.mid_split * g["med"]
        youzi = g["large"]

        out = pd.DataFrame({
            "ts": g["ts"], SYMBOL: g[SYMBOL],
            "retail_flow": retail, "inst_flow": inst, "youzi_flow": youzi,
        })

        if norm_base is not None and not norm_base.empty:
            base = norm_base.rename(columns={"norm_base": "_base"})
            out = out.merge(base, on=["ts", SYMBOL], how="left")
            denom = out["_base"].replace(0, np.nan)
            out["retail_flow"] = out["retail_flow"] / denom
            out["inst_flow"] = out["inst_flow"] / denom
            out["youzi_flow"] = out["youzi_flow"] / denom
            out = out.drop(columns="_base")
        else:
            logger.warning("未提供 norm_base，资金流因子不做归一化（数值为绝对额）")
        return out.set_index("ts")

    # ------------------------------------------------------------------
    # 日频因子（北向 / 两融）—— 调用方需先做 T-1 对齐
    # ------------------------------------------------------------------

    def north_sync(self, north: pd.DataFrame) -> pd.DataFrame:
        """北向共振：0.6*sign(20日持仓变化) + 0.4*sign(当日净买)。

        :param north: 日频长表（DatetimeIndex + symbol/trade_date/
            north_holding/north_buy_net 列），已按可用时点对齐
        :return: [trade_date, symbol, north_sync]
        """
        n = north.sort_values([SYMBOL, TRADE_DATE]).copy()
        chg = n.groupby(SYMBOL, group_keys=False)["north_holding"].pct_change(
            self.holding_days)
        sync = (self.north_weights[0] * np.sign(chg)
                + self.north_weights[1] * np.sign(n["north_buy_net"]))
        out = n[[TRADE_DATE, SYMBOL]].copy()
        out["north_sync"] = sync
        return out

    def margin_pressure(self, nm: pd.DataFrame) -> pd.DataFrame:
        """两融压力：融资5日变化率 - 融券5日变化率。

        :param nm: 日频长表（DatetimeIndex + symbol/trade_date/
            margin_fin_balance/margin_sec_balance 列），已按可用时点对齐
        :return: [trade_date, symbol, margin_pressure]
        """
        n = nm.sort_values([SYMBOL, TRADE_DATE]).copy()
        fin_chg = n.groupby(SYMBOL, group_keys=False)["margin_fin_balance"].pct_change(
            self.margin_days)
        sec_chg = n.groupby(SYMBOL, group_keys=False)["margin_sec_balance"].pct_change(
            self.margin_days)
        out = n[[TRADE_DATE, SYMBOL]].copy()
        out["margin_pressure"] = fin_chg - sec_chg
        return out
