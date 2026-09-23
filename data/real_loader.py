"""真实数据组装层：data1（万得 L2）+ data2（日频 CSV）→ 标准 DataSlice。

职责分工（三层解耦）：
    data/l2_loader.py    解析 data1 Level-2（行情快照 → K 线/盘口；逐笔成交/委托）
    data/macro_loader.py 清洗 data2 宏观/北向/两融/行业/龙虎榜（日频长表）
    data/aligner.py      严格 T-1 asof 对齐 + 龙虎榜 T+1 隔离 + 防未来函数校验
    本模块               组装 DataSlice + 缺口近似（伪指数/广度/行业资金流）

适配规则（严格防未来函数）：
- 分钟级表（kline / l2_snapshot / tick_trades）直接用当日实时数据（当前 bar 已收盘）
- 日频表（macro / north_margin / industry）由 TimeAligner 做 T-1 全量对齐
- 龙虎榜由 TimeAligner 标注 T+1 可用日（avail_date）
- 独立指数与全市场广度从 data2/index_min、data2/breadth_min 加载；缺失时保持缺失
- 行业 money_flow 用行业指数日间变化经 T-1 对齐，日内恒定
- 个股→行业映射缺失：内置 DEFAULT_SYMBOL_TO_INDUSTRY（按股票名称近似），可覆盖

用法：
    from data.real_loader import RealDataLoader
    ds = RealDataLoader().load_slice(["600237", "600460"], "2023-01-03", "2023-04-30")
    ds = TimeAligner().align_slice(ds)   # load_slice 已内置对齐，可省略
    ds.validate()
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data.aligner import TimeAligner
from data.dataslice import (
    BREADTH_COLS,
    INDEX_MIN_COLS,
    INDUSTRY_COLS,
    KLINE_COLS,
    L2_SNAPSHOT_COLS,
    NORTH_MARGIN_COLS,
    SYMBOL,
    TICK_COLS,
    TRADE_DATE,
    DataSlice,
)
from data.l2_loader import L2DataLoader
from data.macro_loader import MacroDataLoader

logger = logging.getLogger("data.real_loader")

# 项目根目录（data/ 的上级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_ROOT = _PROJECT_ROOT / "data"

# data1 个股→中信行业近似映射（industry_sentiment_history 中存在的行业名；
# 真实行业映射后续可替换，见 config/industry_mapping.yaml）
DEFAULT_SYMBOL_TO_INDUSTRY: Dict[str, str] = {
    "600171": "半导体", "600198": "通信设备", "600237": "元件", "600360": "半导体",
    "600379": "电网设备", "600460": "半导体", "600552": "光学光电子", "600563": "元件",
    "600584": "半导体", "600707": "光学光电子", "600732": "光伏设备", "600877": "半导体",
    "600888": "工业金属", "603005": "半导体", "603019": "计算机设备", "603068": "半导体",
    "603129": "综合", "603290": "半导体", "603297": "光学光电子", "603380": "元件",
}


class RealDataLoader:
    """组装层：L2DataLoader + MacroDataLoader + TimeAligner → 标准 DataSlice。

    :param data_root: 项目 data 目录（默认 data/，内含 data1/、data2/）
    :param symbol_to_industry: 个股→行业映射；缺省用 DEFAULT_SYMBOL_TO_INDUSTRY
    :param min_bar_freq: K 线重采样频率（默认 1min）
    :param l2_depth: 盘口档位深度（默认 5，可 10）
    """

    def __init__(
        self,
        data_root: Path = _DATA_ROOT,
        symbol_to_industry: Optional[Dict[str, str]] = None,
        min_bar_freq: str = "1min",
        l2_depth: int = 5,
    ) -> None:
        self.data1 = Path(data_root) / "data1" / "data"  # data/data1/data/{市场}/{代码}.{市场}/...
        self.data2 = Path(data_root) / "data2"
        self.symbol_to_industry = dict(
            symbol_to_industry or DEFAULT_SYMBOL_TO_INDUSTRY)
        self.freq = min_bar_freq
        self.l2_depth = l2_depth
        self.aligner = TimeAligner()
        self.l2 = L2DataLoader(self.data1)
        self.macro = MacroDataLoader(self.data2)
        self._axis: Optional[pd.DatetimeIndex] = None

    # ------------------------------------------------------------------
    # 组装入口
    # ------------------------------------------------------------------

    def discover_symbols(self) -> List[str]:
        """从 data1 目录发现全部标的（裸 6 位代码）。"""
        syms = set()
        for mkt in ("SH", "SZ", "BJ"):
            d = self.data1 / mkt
            if d.is_dir():
                syms.update(p.name.split(".")[0] for p in d.iterdir()
                            if p.is_dir() and "." in p.name)
        return sorted(syms)

    def load_slice(self, symbols: List[str], start: str, end: str,
                   skip_tick: bool = False) -> DataSlice:
        """组装标准 DataSlice（加载后已统一对齐）。

        :param symbols: 股票代码列表（6 位；可带 .SH/.SZ 后缀）
        :param start/end: 回测区间 "YYYY-MM-DD"（含两端）
        :param skip_tick: 跳过逐笔成交/快照加载（特征缓存命中时用，
            tick/l2_snapshot 置 None；状态机与回测只依赖 kline/特征表）
        """
        self._ensure_parquet_engine()
        symbols = [s.split(".")[0] for s in symbols]
        t0 = pd.Timestamp.now()
        kline = self._load_kline(symbols, start, end)
        logger.info("real_loader: kline %d 行（%.1fs）", len(kline),
                    (pd.Timestamp.now() - t0).total_seconds())
        axis = kline.index[~kline.index.duplicated()]

        if skip_tick:
            logger.info("real_loader: 特征缓存命中，跳过逐笔成交/快照加载")
            tick, snap = None, None
        else:
            t0 = pd.Timestamp.now()
            tick = self.l2.load_tick_trades(symbols, start, end)
            logger.info("real_loader: tick_trades %d 行（%.1fs）", len(tick),
                        (pd.Timestamp.now() - t0).total_seconds())
            snap = self.l2.load_l2_snapshot(symbols, start, end,
                                            depth=self.l2_depth)

        ds = DataSlice(
            kline=kline,
            l2_snapshot=snap,
            tick_trades=tick,
            index_min=self._load_market_table("index_min", INDEX_MIN_COLS,
                                              axis),
            breadth=self._load_market_table("breadth_min", BREADTH_COLS,
                                            axis),
            industry=self._load_industry(symbols, axis),
            macro=self.macro.load_macro(),
            north_margin=self.macro.load_north_margin(symbols),
            dragon_tiger=self.macro.load_dragon_tiger(symbols),
            meta={
                "symbols": symbols, "start": start, "end": end,
                "source": "data1+data2 (real_loader)",
                "market_source": "data2 独立分钟市场文件；缺失则市场因子缺失",
                "index_min": "data2/index_min（独立基准）",
                "breadth": "data2/breadth_min（独立全市场广度）",
                "n_symbols": len(symbols),
                "st_coverage": float(kline["is_st"].notna().mean()),
                "float_cap_coverage": float(kline["float_market_cap"].notna().mean()),
                "industry_money_flow": "行业close日间变化T-1(近似)",
            },
        )
        # 统一对齐：macro T-1、龙虎榜 T+1、各表排序去重
        ds = self.aligner.align_slice(ds)
        return ds

    # ------------------------------------------------------------------
    # 前置能力探测 / K 线（l2_loader 解析 + 衍生列）
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_parquet_engine() -> None:
        """前置探测 parquet 引擎，缺失时立即显式失败（而非淹没在几百条 WARNING 后
        抛出与真实原因无关的 AttributeError）。"""
        try:
            import pyarrow  # noqa: F401
            return
        except ImportError:
            pass
        try:
            import fastparquet  # noqa: F401
            return
        except ImportError as exc:
            raise RuntimeError(
                "缺少 parquet 引擎（pyarrow / fastparquet），"
                "请先执行 pip install pyarrow") from exc

    def _load_kline(self, symbols: List[str], start: str, end: str) -> pd.DataFrame:
        """分钟 K 线 + 衍生列（vwap / 市值 / 涨跌停 / ST）。"""
        k = self.l2.load_kline(symbols, start, end, freq=self.freq)
        if k is None or k.empty:
            # 主 kline 为空属致命情形，不可降级：显式短路，避免后续对
            # RangeIndex 调用 .normalize() 抛出误导性的 AttributeError
            raise ValueError(
                f"data1 无 {symbols} 在 {start}~{end} 的行情数据，"
                f"请检查数据目录与 parquet 引擎（pyarrow/fastparquet）")
        denom = k["volume"].replace(0, np.nan)
        k["vwap"] = k["amount"] / denom
        k["vwap"] = k["vwap"].fillna(k["close"])
        k["float_market_cap"] = np.nan
        # 涨跌停价：A 股规则 = T-1 收盘价 × 幅度，T 日内恒定；
        # 幅度按 ST（±5%）、创业板 300-302 / 科创板 688（±20%）、主板（±10%）区分。
        k["is_st"] = np.nan
        history = self._load_stock_history()
        if history is not None:
            original_index = k.index
            left = k.reset_index().rename(columns={k.index.name or "index": "ts"})
            left["_row"] = np.arange(len(left))
            left = left.sort_values("ts")
            history = history.sort_values("trade_date")
            left = pd.merge_asof(left, history,
                                 left_on="ts", right_on="trade_date",
                                 by=SYMBOL, direction="backward",
                                 suffixes=("", "_history"))
            left = left.sort_values("_row")
            left["is_st"] = left["is_st_history"]
            left["float_market_cap"] = left["close"] * left["float_shares"]
            k = left.drop(columns=["_row", "trade_date", "is_st_history",
                                   "float_shares"]).set_index("ts")
            k.index.name = original_index.name
        # 昨收按 (symbol, 交易日) 取前一日最后收盘（ts 为 index；索引对该 order 独立）
        kd = k[[SYMBOL, "close"]].copy()
        kd["date"] = kd.index.normalize()
        day_last = kd.groupby([SYMBOL, "date"])["close"].last().reset_index()
        day_last["prev_close"] = day_last.groupby(SYMBOL)["close"].shift(1)
        _key = pd.MultiIndex.from_arrays([day_last[SYMBOL], day_last["date"]])
        _pvm = pd.Series(day_last["prev_close"].to_numpy(), index=_key)
        k_idx = pd.MultiIndex.from_arrays([k[SYMBOL], k.index.normalize()])
        prev = _pvm.reindex(k_idx).to_numpy()
        ratio = np.where(
            k["is_st"].isna(), np.nan,
            np.where(k["is_st"].astype(bool), 0.05,
            np.where(k[SYMBOL].str[:3].isin(("300", "301", "302"))
                     | k[SYMBOL].str.startswith("688"), 0.20, 0.10)))
        k["up_limit"] = np.round(prev * (1.0 + ratio), 2)
        k["down_limit"] = np.round(prev * (1.0 - ratio), 2)
        return k[KLINE_COLS]

    def _load_stock_history(self) -> Optional[pd.DataFrame]:
        """Optional effective-date stock history: symbol,trade_date,is_st,float_shares."""
        path = self.data2 / "stock_history.csv"
        if not path.exists():
            logger.warning("缺少 %s；历史 ST/流通股本未知，相关交易过滤保守阻断", path)
            return None
        frame = pd.read_csv(path, dtype={SYMBOL: str})
        required = {SYMBOL, "trade_date", "is_st", "float_shares"}
        if not required <= set(frame.columns):
            raise ValueError(f"{path} 缺少列 {required - set(frame.columns)}")
        frame[SYMBOL] = frame[SYMBOL].str.zfill(6)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["is_st"] = frame["is_st"].astype(str).str.lower().map(
            {"true": True, "1": True, "false": False, "0": False})
        if frame["is_st"].isna().any():
            raise ValueError(f"{path} 的 is_st 列含无法识别的值")
        frame["float_shares"] = pd.to_numeric(frame["float_shares"], errors="coerce")
        return frame.drop_duplicates([SYMBOL, "trade_date"], keep="last")

    def _load_market_table(self, stem: str, columns: List[str],
                           axis: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
        """Load a market-wide minute feed independent of the trading symbol set."""
        path = next((p for p in (self.data2 / f"{stem}.parquet",
                                 self.data2 / f"{stem}.csv") if p.exists()), None)
        if path is None:
            logger.warning("缺少独立市场数据 %s；对应市场因子置缺失", stem)
            return None
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        if "ts" in frame.columns:
            frame = frame.set_index("ts")
        frame.index = pd.to_datetime(frame.index)
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{path} 缺少列 {sorted(missing)}")
        frame = frame.sort_index().loc[lambda x: x.index.isin(axis)]
        return frame

    def _load_industry(self, symbols: List[str], axis: pd.DatetimeIndex) -> pd.DataFrame:
        """行业情绪：行业指数 close → 分钟轴 T-1 填充，money_flow=日间变化。"""
        industries = {self.symbol_to_industry[s] for s in symbols
                      if s in self.symbol_to_industry}
        if not industries:
            return pd.DataFrame()
        di = self.macro.load_industry_sentiment(sorted(industries))
        if di.empty:
            return pd.DataFrame()
        rows = []
        for ind_name, g in di.groupby("行业"):
            a = self.aligner.align_external(
                g, axis, ["close", "money_flow"], date_col=TRADE_DATE)
            a["industry"] = ind_name
            rows.append(a)
        if not rows:
            return pd.DataFrame()
        out = pd.concat(rows)
        out["open"] = out["high"] = out["low"] = out["close"]
        out["volume"] = np.nan
        return out[INDUSTRY_COLS]
