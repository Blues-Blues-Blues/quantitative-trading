"""多源异构数据时钟对齐管道（严格防未来函数）。

职责：
1. 外部数据（美股收盘 / 大宗商品 / 美债 / 美元指数 / 亚太早盘）按
   「T-1 全量对齐」合并至 A 股分钟时间轴——任一 A 股时间点只使用
   严格早于它的外部记录（asof / allow_exact_matches=False），绝无未来函数
2. 龙虎榜按「披露日 + 1 交易日」标记可用日（avail_date），隔离为
   T+1 因子事件表，禁止进入 T 日实时特征
3. 数据缺失容错：外部某日缺失时自动向前填充（ffill）并输出 Warning
   日志，绝不中断流程
4. 提供 verify_no_lookahead 校验工具，可在因子/回测入口做回归断言

对齐主键：A 股交易时间轴（分钟级 DatetimeIndex，取自 DataSlice.time_axis()）。
"""

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from data.dataslice import (
    BREADTH_COLS,
    DRAGON_TIGER_COLS,
    MACRO_COLS,
    SYMBOL,
    TRADE_DATE,
    DataSlice,
)

logger = logging.getLogger("data.aligner")


class LookaheadError(ValueError):
    """检测到未来函数（Look-ahead Bias）时抛出。"""


class TimeAligner:
    """时间对齐管道：外部数据 T-1 对齐 + 龙虎榜 T+1 隔离 + 容错填充。

    :param tz: A 股交易时区（默认 Asia/Shanghai，当前仅用于元数据标注）
    :param logger: 日志器，缺省使用模块级 logger
    """

    def __init__(
        self,
        tz: str = "Asia/Shanghai",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.tz = tz
        self.log = logger or logging.getLogger("data.aligner")

    # ------------------------------------------------------------------
    # 对外对齐入口
    # ------------------------------------------------------------------

    def align_slice(self, ds: DataSlice) -> DataSlice:
        """对齐管道入口：输入原始 DataSlice，输出对齐后的 DataSlice。

        - macro  →  T-1 全量对齐至 A 股时间轴
        - dragon_tiger → 标注 avail_date（披露次日），隔离为事件表
        - 其余表仅做时间规整（升序、去重）；breadth 自动补齐 adr
        """
        axis = ds.time_axis()

        return DataSlice(
            kline=self._sort_dedupe(ds.kline),
            l2_snapshot=self._sort_dedupe(ds.l2_snapshot),
            tick_trades=self._sort_dedupe(ds.tick_trades),
            index_min=self._sort_dedupe(ds.index_min),
            breadth=self._with_adr(self._sort_dedupe(ds.breadth)),
            industry=self._sort_dedupe(ds.industry),
            macro=self.align_external(ds.macro, axis, MACRO_COLS)
            if ds.macro is not None else None,
            north_margin=self._sort_dedupe(ds.north_margin),
            dragon_tiger=self.align_dragon_tiger(ds.dragon_tiger, axis)
            if ds.dragon_tiger is not None else None,
            factors=ds.factors,
            meta={**ds.meta, "aligned_by": "TimeAligner", "t_minus_1_external": True},
        )

    # ------------------------------------------------------------------
    # 外部数据：T-1 全量对齐
    # ------------------------------------------------------------------

    def align_external(
        self,
        external: Optional[pd.DataFrame],
        cn_axis: Union[pd.DatetimeIndex, List],
        value_cols: List[str],
        date_col: str = TRADE_DATE,
    ) -> pd.DataFrame:
        """把「每个外部交易日一行」的表对齐到 A 股分钟时间轴。

        规则：A 股时间点 T 使用的外部值 = 满足 external_date < T 的
        最新一条记录（即 T-1 及更早），当日外部数据绝不参与当日计算。

        :param external: 外部数据表，含 date_col 与 value_cols；
            缺行（如海外休市）时由 asof 自动取前值，表内 NaN 再做 ffill
        :param cn_axis: A 股交易时间轴（DatetimeIndex）
        :param value_cols: 需对齐的数值列
        :return: DataFrame，index=cn_axis（升序、去重），列=value_cols
        """
        index = pd.DatetimeIndex(cn_axis)
        index = index[~index.duplicated()].sort_values()

        if external is None or external.empty:
            self.log.warning("外部数据为空，跳过对齐，返回全 NaN 表")
            return pd.DataFrame(index=index, columns=value_cols)

        ext = external.copy()
        ext[date_col] = pd.to_datetime(ext[date_col])
        # 整行无效（休市/缺数据）的记录不参与 asof，避免把 NaN 当有效值
        ext = ext.dropna(subset=value_cols, how="all")
        if ext.empty:
            self.log.warning("外部数据无任何有效值，返回全 NaN 表")
            return pd.DataFrame(index=index, columns=value_cols)

        ext = (
            ext.sort_values(date_col)
            .drop_duplicates(subset=date_col, keep="last")
        )

        # 交易日级 asof：cn_date 只匹配严格更早的外部日期（T-1）
        cn_dates = pd.DatetimeIndex(index.normalize()).unique().sort_values()
        left = pd.DataFrame({"_cn_date": cn_dates})
        merged = pd.merge_asof(
            left, ext,
            left_on="_cn_date", right_on=date_col,
            direction="backward", allow_exact_matches=False,
        )

        # 映射回分钟轴并做缺失容错（ffill）
        mapping = merged.set_index("_cn_date")[value_cols]
        tmp = pd.DataFrame({"_cn_date": index.normalize()}, index=index)
        aligned = tmp.join(mapping, on="_cn_date")[value_cols]
        aligned = aligned.ffill()

        missing = int(aligned.isna().sum().sum())
        if missing:
            self.log.warning(
                "%d 个外部值在数据起始前即缺失（无更早可用数据），"
                "保持 NaN 不填充，调用方应自行降级处理", missing,
            )
        return aligned

    # ------------------------------------------------------------------
    # 龙虎榜：T+1 隔离
    # ------------------------------------------------------------------

    def align_dragon_tiger(
        self,
        dts: Optional[pd.DataFrame],
        cn_axis: Union[pd.DatetimeIndex, List],
    ) -> pd.DataFrame:
        """标注龙虎榜事件的 T+1 可用日（avail_date），原样返回事件表。

        上榜日 trade_date 的记录在披露次日（下一个 A 股交易日）才可用；
        avail_date 列用于下游校验：任何实时特征时间戳必须晚于 avail_date。
        """
        if dts is None or dts.empty:
            return dts

        out = dts.copy()
        out[TRADE_DATE] = pd.to_datetime(out[TRADE_DATE])
        cn_dates = pd.DatetimeIndex(
            pd.DatetimeIndex(cn_axis).normalize()
        ).unique().sort_values()

        pos = cn_dates.searchsorted(out[TRADE_DATE].to_numpy(), side="left")
        # numpy 2.x 下 np.array([Timestamp, NaT], dtype="datetime64") 会报
        # "'float' object cannot be interpreted as an integer"，改用 pd.to_datetime
        avail = pd.to_datetime(
            [cn_dates[i + 1] if i + 1 < len(cn_dates) else pd.NaT for i in pos]
        )
        out["avail_date"] = avail

        na_count = int(pd.isna(out["avail_date"]).sum())
        if na_count:
            self.log.warning(
                "%d 条龙虎榜记录的上榜日为最后一个交易日，无 T+1 可用日，"
                "avail_date 置 NaT", na_count,
            )
        return out.sort_values(["avail_date", TRADE_DATE, SYMBOL])

    # ------------------------------------------------------------------
    # 防未来函数校验
    # ------------------------------------------------------------------

    @staticmethod
    def verify_no_lookahead(
        ts_index: Union[pd.DatetimeIndex, List],
        available_ts: Union[pd.DatetimeIndex, List],
        name: str = "data",
    ) -> bool:
        """断言「数据被使用的时间」严格晚于「数据可获取时间」。

        :raises LookaheadError: 存在 ts <= available 的行（未来函数）
        :return: True（全部合规）
        """
        ts = pd.DatetimeIndex(ts_index)
        avail = pd.DatetimeIndex(available_ts)
        if len(ts) != len(avail):
            raise ValueError(f"{name}: 时间轴与可用时间长度不一致 "
                             f"({len(ts)} vs {len(avail)})")
        bad = ts <= avail
        if bad.any():
            n = int(bad.sum())
            i = int(np.argmax(bad))
            raise LookaheadError(
                f"{name}: 检测到 {n} 行未来函数（数据可用时间不早于使用时间），"
                f"首例: 使用时间 {ts[i]} <= 可用时间 {avail[i]}"
            )
        return True

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_dedupe(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return df
        return df[~df.index.duplicated(keep="first")].sort_index()

    @staticmethod
    def _with_adr(breadth: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """全市场广度补齐 adr = 上涨家数 / 下跌家数（除零/无值 → NaN）。"""
        if breadth is None or breadth.empty:
            return breadth
        out = breadth.copy()
        if "adr" not in out.columns and {"advancers", "decliners"} <= set(out.columns):
            decl = out["decliners"].where(out["decliners"] > 0)
            out["adr"] = out["advancers"] / decl
        return out
