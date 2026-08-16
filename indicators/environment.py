"""宏观与行业环境共振因子（Environment）。

因子定义（各分量先滚动 z-score 标准化再加权合成，权重均为初始化入参）：
    MRS = w1*沪深300分钟收益 + w2*市场量能偏离 + w3*ADR偏离 + w4*北向净流偏离
    GRS = w1*美股隔夜收益 + w2*商品指数偏离 + w3*美债利率波动
    IRS = w1*行业资金流 + w2*海外龙头隔夜涨跌 + w3*产业链指数涨跌
    Global_Mod = clip(GRS, -0.8, 0.8)
    Chain_Mod  = clip(IRS, -0.3, 0.3)

防未来函数约定：
- GRS 基于 macro（已由 TimeAligner 做 T-1 全量对齐），当日外部数据不参与当日
- MRS 基于分钟级实时数据（当前 bar 已收盘），当日可用
- 海外龙头隔夜 / 产业链指数数据缺失时，对应分量权重自动归零并告警，
  绝不中断流程
"""

import logging
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data.dataslice import SYMBOL

logger = logging.getLogger("indicators.environment")

_EPS = 1e-12

_COMMODITY_COLS = ["brent", "gold", "copper"]


class Environment:
    """宏观 / 大盘 / 产业共振因子。

    :param score_window: 滚动 z-score 窗口（分钟），默认 240（一个交易日）
    :param min_periods:  滚动 z-score 最小观测数
    :param mrs_weights:  MRS 分量权重 (收益, 量能, ADR, 北向)
    :param grs_weights:  GRS 分量权重 (美股, 商品, 美债)
    :param irs_weights:  IRS 分量权重 (行业资金流, 海外龙头, 产业链)
    :param grs_clip:     Global_Mod 的 clip 范围
    :param irs_clip:     Chain_Mod 的 clip 范围
    """

    def __init__(
        self,
        score_window: int = 240,
        min_periods: int = 20,
        mrs_weights: Sequence[float] = (0.35, 0.20, 0.25, 0.20),
        grs_weights: Sequence[float] = (0.40, 0.30, 0.30),
        irs_weights: Sequence[float] = (1.00, 0.00, 0.00),
        grs_clip: Tuple[float, float] = (-0.8, 0.8),
        irs_clip: Tuple[float, float] = (-0.3, 0.3),
    ) -> None:
        self.score_window = score_window
        self.min_periods = min_periods
        self.mrs_weights = tuple(mrs_weights)
        self.grs_weights = tuple(grs_weights)
        self.irs_weights = tuple(irs_weights)
        self.grs_clip = grs_clip
        self.irs_clip = irs_clip

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _zscore(self, s: pd.Series) -> pd.Series:
        """滚动 z-score（窗口含当前已收盘 bar，属历史信息）。"""
        mean = s.rolling(self.score_window, min_periods=self.min_periods).mean()
        std = s.rolling(self.score_window, min_periods=self.min_periods).std()
        return (s - mean) / (std + _EPS)

    @staticmethod
    def _weighted(components: Dict[str, pd.Series],
                  weights: Sequence[float],
                  names: Sequence[str]) -> pd.Series:
        """按可用分量加权合成（缺失分量权重自动重归一，不抛错）。"""
        parts, ws = [], []
        for name, w in zip(names, weights):
            s = components.get(name)
            if s is not None and s.notna().any() and w > 0:
                parts.append(s)
                ws.append(w)
        if not parts:
            return pd.Series(np.nan, index=next(iter(components.values())).index)
        ws = np.asarray(ws, dtype=float)
        ws = ws / ws.sum()
        out = ws[0] * parts[0]
        for w, p in zip(ws[1:], parts[1:]):
            out = out + w * p
        return out

    # ------------------------------------------------------------------
    # MRS：大盘状态（分钟级，当日可用）
    # ------------------------------------------------------------------

    def mrs(self, index_min: pd.DataFrame, breadth: Optional[pd.DataFrame]) -> pd.DataFrame:
        """沪深300 收益率 + 量能偏离 + ADR 偏离 + 北向净流偏离 合成。

        :param index_min: 指数 1 分钟长表（index_code/open..close/volume）
        :param breadth:   全市场广度（advancers/decliners/adr[/north_net]）
        :return: 长表 [index_code, mrs]
        """
        if index_min is None or index_min.empty:
            logger.warning("缺少 index_min，MRS 置 NaN")
            return pd.DataFrame(columns=["index_code", "mrs"])

        idx = index_min.copy()
        rows = []
        for code, g in idx.groupby("index_code"):
            g = g.sort_index()
            comps = {
                "ret": self._zscore(g["close"].pct_change().fillna(0.0)),
                "vol": self._zscore(g["volume"].astype(float)),
            }
            names = ["ret", "vol"]
            if breadth is not None and not breadth.empty:
                b = breadth.sort_index()
                if "adr" in b.columns and b["adr"].notna().any():
                    comps["adr"] = self._zscore(b["adr"].fillna(b["adr"].median()))
                    names.append("adr")
                if "north_net" in b.columns and b["north_net"].notna().any():
                    comps["north"] = self._zscore(b["north_net"].fillna(0.0))
                    names.append("north")
            else:
                logger.warning("缺少 breadth，MRS 仅用收益与量能分量")
            s = self._weighted(comps, self.mrs_weights, names)
            rows.append(pd.DataFrame({"index_code": code, "mrs": s}, index=s.index))
        out = pd.concat(rows) if rows else pd.DataFrame(columns=["index_code", "mrs"])
        return out

    # ------------------------------------------------------------------
    # GRS：全球风险（T-1 对齐后的 macro，分钟轴）
    # ------------------------------------------------------------------

    def grs(self, macro: pd.DataFrame) -> pd.DataFrame:
        """美股隔夜收益 + 商品指数偏离 + 美债利率波动 合成。

        :param macro: 已 T-1 对齐的宏观表（DatetimeIndex + MACRO_COLS）
        :return: 长表 [grs]
        """
        if macro is None or macro.empty:
            logger.warning("缺少 macro，GRS / Global_Mod 置 NaN")
            return pd.DataFrame(columns=["grs"])

        m = macro.sort_index()
        us_ret = m["us_spx"].pct_change().fillna(0.0)
        commodity = m[_COMMODITY_COLS].mean(axis=1)
        comps = {
            "us": self._zscore(us_ret),
            "com": self._zscore(commodity),
            "rate": self._zscore(m["us10y"]),
        }
        s = self._weighted(comps, self.grs_weights, ["us", "com", "rate"])
        return pd.DataFrame({"grs": s}, index=m.index)

    def global_mod(self, grs: pd.DataFrame) -> pd.Series:
        """Global_Mod = clip(GRS, -0.8, 0.8)。"""
        return grs["grs"].clip(*self.grs_clip)

    # ------------------------------------------------------------------
    # IRS：产业共振（分钟级，行业资金流 + 可选海外龙头/产业链）
    # ------------------------------------------------------------------

    def irs(
        self,
        industry: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
        mapping: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> pd.DataFrame:
        """行业资金流（+ 海外龙头隔夜涨跌 + 产业链指数涨跌）合成。

        :param industry: 行业长表（DatetimeIndex + industry/open..close/money_flow）
        :param macro:    已对齐宏观表；若 mapping 指定海外龙头列则参与
        :param mapping:  {行业名: {"overseas_leader": macro列名}}，缺省不启用
        :return: 长表 [industry, irs]
        """
        if industry is None or industry.empty:
            logger.warning("缺少 industry，IRS / Chain_Mod 置 NaN")
            return pd.DataFrame(columns=["industry", "irs"])

        ind = industry.copy()
        rows = []
        for name, g in ind.groupby("industry"):
            g = g.sort_index()
            comps = {"flow": self._zscore(g["money_flow"].astype(float))}
            names = ["flow"]
            # 海外龙头隔夜涨跌（需 mapping 提供 macro 列名）
            if mapping and name in mapping:
                col = mapping[name].get("overseas_leader")
                if macro is not None and col in macro.columns and macro[col].notna().any():
                    comps["leader"] = self._zscore(macro[col].pct_change().fillna(0.0))
                    names.append("leader")
                else:
                    logger.warning("行业 %s 的海外龙头数据缺失，分量跳过", name)
            s = self._weighted(comps, self.irs_weights, names)
            rows.append(pd.DataFrame({"industry": name, "irs": s}, index=s.index))
        out = pd.concat(rows) if rows else pd.DataFrame(columns=["industry", "irs"])
        return out

    def chain_mod(self, irs: pd.DataFrame) -> pd.Series:
        """Chain_Mod = clip(IRS, -0.3, 0.3)。"""
        return irs["irs"].clip(*self.irs_clip)
