"""横截面排序闸门：多标的股票池的"相对排序"升级。

8 层开仓硬闸门与 6 层平仓闸门的逐股判定逻辑见 strategy.signals.py
（SignalSynthesizer.entry_gates / exit_triggers）。本模块把第 ⑥ 层
"个股层"从绝对阈值扩展为横截面相对排序：

- add_previous_rank_columns：给分钟长表追加 rank_pct_<factor> 列，
  即因子在"昨日收盘全池截面"的百分位排名（0~1，越大越强）。
  T-1 排名 → 盘中/收盘均无未来函数（收盘后算排名，次日全天复用），
  与日频因子 T-1 对齐的既有设计一致。
- CrossSectionalRankGate：因子 T-1 排名需进入全池前 top_quantile
  才允许开仓；无排名（停牌/数据缺失）→ 保守关闭。

示例（main.py 中按需开启）：
    gate = CrossSectionalRankGate("final_ms", top_quantile=0.2)
    syn = SignalSynthesizer(..., rank_gate=gate)
"""

from typing import List

import numpy as np
import pandas as pd

from data.dataslice import SYMBOL


def add_previous_rank_columns(panel: pd.DataFrame,
                              factor_cols: List[str]) -> pd.DataFrame:
    """给分钟长表追加 T-1 收盘全池百分位排名列。

    对每个因子：取每交易日最后一根 bar（15:00 收盘）的因子值，
    计算当日全池百分位排名（0~1），再按交易日 shift(1) 得到"昨日
    排名"，合并回分钟长表（当日每根 bar 复用同一昨日排名）。

    :return: 新增列 rank_pct_<factor>
    """
    if not factor_cols:
        return panel
    p = panel.copy()
    p["day"] = p["ts"].dt.normalize()
    # 每 (day, symbol) 最后一根 bar（组内时间升序 → last 即 15:00 收盘）
    last = p.groupby(["day", SYMBOL], sort=False)[list(factor_cols)].last()
    # 当日全池横截面百分位（0~1，值越高排名越靠前）
    rank = last.groupby("day")[list(factor_cols)].rank(pct=True)
    # T-1：按交易日 shift(1)。用 level 分组使 symbol 保留在 MultiIndex 中
    #（pandas 3.0 按列 groupby().shift() 会丢失分组键列）
    prev = (rank.reset_index().set_index(["day", SYMBOL]).sort_index()
            .groupby(level=SYMBOL)[list(factor_cols)].shift(1))
    prev.columns = [f"rank_pct_{c}" for c in factor_cols]
    prev = prev.reset_index()
    return p.merge(prev, on=["day", SYMBOL], how="left")


class CrossSectionalRankGate:
    """横截面排序闸门：因子 T-1 全池排名需进入前 top_quantile。

    :param factor:       参与排名的因子列（synthesize 输出列，如 final_ms / ofss）
    :param top_quantile: 排名阈值（0~1）；0.2 = 前 20% 才放行
    """

    def __init__(self, factor: str = "final_ms",
                 top_quantile: float = 0.2) -> None:
        if not 0.0 < top_quantile <= 1.0:
            raise ValueError(f"top_quantile 须在 (0, 1] 内，收到 {top_quantile}")
        self.factor = factor
        self.top_quantile = top_quantile

    @property
    def rank_col(self) -> str:
        return f"rank_pct_{self.factor}"

    def passes(self, row: pd.Series) -> bool:
        """排名进入前 top_quantile 才放行；无排名（停牌/缺失）保守关闭。"""
        p = row.get(self.rank_col, np.nan)
        if pd.isna(p):
            return False
        return bool(p >= 1.0 - self.top_quantile)
