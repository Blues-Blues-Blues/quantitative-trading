"""数据加载抽象层 DataLoader。

抽象基类定义统一加载接口：每种数据一个 load_* 方法，最终由 load_slice
组装为标准 DataSlice。具体实现负责底层来源（Parquet / CSV / 数据库），
读取时统一规整时间戳（转 DatetimeIndex、升序、去重）。

FileDataLoader 复用 data.storage 的逻辑数据源注册表：构造时传入
「表类型 → 数据源 key」映射，存储位置不明确时由 config.data_sources
统一解析（环境变量 / data_paths.yaml / 运行时注册 / 默认路径）。
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from data import storage
from data.dataslice import DataSlice, SYMBOL, TRADE_DATE

logger = logging.getLogger("data.dataloader")

# 表类型 → 默认逻辑数据源 key（与 config/data_sources.py 内置源对应）
_DEFAULT_SOURCE_MAP: Dict[str, str] = {
    "kline": "daily_cache",     # 基础 K 线（含分钟级文件时改用 l2 或单独源）
    "l2_snapshot": "l2",
    "tick_trades": "l2",
    "index_min": "l2",
    "breadth": "l2",
    "industry": "macro",
    "macro": "macro",
    "dragon_tiger": "macro",
}


def _canonicalize_index(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    """把 index/ts 列规整为升序、去重的 DatetimeIndex（index 方式）。

    同时将 symbol 列统一为字符串（CSV 回读可能推断为 int）。
    日频表处理（契约见 DataSlice.validate）：
    - 无 symbol 的全局日频表（macro 等）以 trade_date 列为 DatetimeIndex；
    - symbol + trade_date 的日频长表（north_margin/dragon_tiger）保留原索引，
      日期过滤由 _slice 按 trade_date 列完成（按 index 去重会误删同日多行）。
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if ts_col in out.columns:
            out[ts_col] = pd.to_datetime(out[ts_col])
            out = out.set_index(ts_col)
        elif TRADE_DATE in out.columns:
            if SYMBOL not in out.columns:
                # 全局日频表：以 trade_date 作 DatetimeIndex
                out[TRADE_DATE] = pd.to_datetime(out[TRADE_DATE])
                out = out.set_index(TRADE_DATE)
            # symbol+trade_date 长表：保留原索引（避免被去重误删）
    out = out[~out.index.duplicated(keep="first")].sort_index()
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str)
    return out


class DataLoader(ABC):
    """数据加载抽象基类。

    子类需实现各 load_* 方法；load_slice 将各表组装为 DataSlice。
    """

    @abstractmethod
    def load_kline(
        self, symbols: List[str], start: str, end: str
    ) -> pd.DataFrame:
        """加载个股分钟/日 K 线，返回标准长表（见 DataSlice.KLINE_COLS）。"""

    @abstractmethod
    def load_l2_snapshot(self, symbols: List[str], start: str, end: str) -> pd.DataFrame:
        """加载 Level-2 五档快照（标准长表）。"""

    @abstractmethod
    def load_tick_trades(self, symbols: List[str], start: str, end: str) -> pd.DataFrame:
        """加载逐笔成交（标准长表）。"""

    @abstractmethod
    def load_index_min(self, start: str, end: str) -> pd.DataFrame:
        """加载沪深300 等指数 1 分钟数据。"""

    @abstractmethod
    def load_breadth(self, start: str, end: str) -> pd.DataFrame:
        """加载全市场涨跌家数。"""

    @abstractmethod
    def load_industry(self, start: str, end: str) -> pd.DataFrame:
        """加载中信一级行业行情与资金流向。"""

    @abstractmethod
    def load_macro(self, start: str, end: str) -> pd.DataFrame:
        """加载全球宏观 / 隔夜数据（含 trade_date 列，T-1 对齐由 aligner 完成）。"""

    @abstractmethod
    def load_dragon_tiger(self, start: str, end: str) -> pd.DataFrame:
        """加载龙虎榜事件表（含 trade_date，aligner 负责标记 T+1 可用日）。"""

    # ---------- 组装 ----------

    def load_slice(
        self,
        symbols: List[str],
        start: str,
        end: str,
        include: Optional[Dict[str, bool]] = None,
    ) -> DataSlice:
        """加载全部/指定数据源并组装为 DataSlice。

        :param include: {表名: 是否加载}，缺省全部加载（None 的表置空）
        """
        flags = include or {
            "l2_snapshot": True, "tick_trades": True, "index_min": True,
            "breadth": True, "industry": True, "macro": True,
            "dragon_tiger": True,
        }
        ds = DataSlice(
            kline=_canonicalize_index(self.load_kline(symbols, start, end)),
            l2_snapshot=_canonicalize_index(self.load_l2_snapshot(symbols, start, end))
            if flags.get("l2_snapshot") else None,
            tick_trades=_canonicalize_index(self.load_tick_trades(symbols, start, end))
            if flags.get("tick_trades") else None,
            index_min=_canonicalize_index(self.load_index_min(start, end))
            if flags.get("index_min") else None,
            breadth=_canonicalize_index(self.load_breadth(start, end))
            if flags.get("breadth") else None,
            industry=_canonicalize_index(self.load_industry(start, end))
            if flags.get("industry") else None,
            macro=_canonicalize_index(self.load_macro(start, end))
            if flags.get("macro") else None,
            dragon_tiger=_canonicalize_index(self.load_dragon_tiger(start, end))
            if flags.get("dragon_tiger") else None,
            meta={"symbols": symbols, "start": start, "end": end},
        )
        return ds


class FileDataLoader(DataLoader):
    """基于 data.storage 的文件数据加载器（Parquet / CSV）。

    :param source_map: {表类型: 逻辑数据源 key}，缺省用 _DEFAULT_SOURCE_MAP
    :param file_pattern: 文件名模板，{kind} 会被替换为表类型，
        如 "minute_{kind}.parquet"；缺省直接使用 "{kind}.{ext}"
    :param fmt: 文件扩展名（由子类设定，如 ".parquet" / ".csv"）
    """

    fmt: str = ".csv"

    def __init__(
        self,
        source_map: Optional[Dict[str, str]] = None,
        file_pattern: str = "{kind}{ext}",
    ) -> None:
        self._source_map = {**_DEFAULT_SOURCE_MAP, **(source_map or {})}
        self._file_pattern = file_pattern

    def _path(self, kind: str) -> Path:
        fname = self._file_pattern.format(kind=kind, ext=self.fmt)
        return storage.resolve(self._source_map[kind], fname)

    def _read(self, kind: str) -> pd.DataFrame:
        path = self._path(kind)
        if not path.exists():
            logger.warning("数据文件不存在，返回空表: %s", path)
            return pd.DataFrame()
        try:
            return storage.read_frame(self._source_map[kind], path.name)
        except storage.DataSourceError as e:
            logger.warning("读取 %s 失败，返回空表: %s", kind, e)
            return pd.DataFrame()

    # ---------- 各表实现 ----------

    def load_kline(self, symbols: List[str], start: str, end: str) -> pd.DataFrame:
        df = self._read("kline")
        return self._slice(df, symbols, start, end, by_symbol=True)

    def load_l2_snapshot(self, symbols: List[str], start: str, end: str) -> pd.DataFrame:
        df = self._read("l2_snapshot")
        return self._slice(df, symbols, start, end, by_symbol=True)

    def load_tick_trades(self, symbols: List[str], start: str, end: str) -> pd.DataFrame:
        df = self._read("tick_trades")
        return self._slice(df, symbols, start, end, by_symbol=True)

    def load_index_min(self, start: str, end: str) -> pd.DataFrame:
        return self._slice(self._read("index_min"), [], start, end)

    def load_breadth(self, start: str, end: str) -> pd.DataFrame:
        return self._slice(self._read("breadth"), [], start, end)

    def load_industry(self, start: str, end: str) -> pd.DataFrame:
        return self._slice(self._read("industry"), [], start, end)

    def load_macro(self, start: str, end: str) -> pd.DataFrame:
        return self._slice(self._read("macro"), [], start, end)

    def load_dragon_tiger(self, start: str, end: str) -> pd.DataFrame:
        return self._slice(self._read("dragon_tiger"), [], start, end)

    # ---------- 工具 ----------

    @staticmethod
    def _slice(
        df: pd.DataFrame,
        symbols: List[str],
        start: str,
        end: str,
        by_symbol: bool = False,
    ) -> pd.DataFrame:
        """按日期范围与标的列表切片；空表/缺列时安全返回。"""
        if df is None or df.empty:
            return df
        out = _canonicalize_index(df)
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        # 按自然日过滤：end 是当天 00:00，需用日期（normalize）比较以保留盘中数据；
        # 日频表以 trade_date 列为主键过滤（索引可为 RangeIndex）
        if TRADE_DATE in out.columns:
            dates = pd.to_datetime(out[TRADE_DATE]).dt.normalize()
        else:
            dates = out.index.normalize()
        out = out[(dates >= s) & (dates <= e)]
        if by_symbol and symbols and "symbol" in out.columns:
            # CSV 回读可能把纯数字代码推断为 int，统一转字符串再比较
            want = {str(s) for s in symbols}
            out = out[out["symbol"].astype(str).isin(want)]
        return out


class ParquetDataLoader(FileDataLoader):
    """Parquet 文件数据加载器（高性能，适合高频与 L2 数据）。"""

    fmt = ".parquet"


class CsvDataLoader(FileDataLoader):
    """CSV 文件数据加载器（轻量）。"""

    fmt = ".csv"
