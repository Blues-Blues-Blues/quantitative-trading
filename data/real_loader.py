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
- 缺口近似（无未来函数）：
    * index_min：用 data1 全部标的的分钟等权价构造伪指数（当日可观测）
    * breadth：   用 data1 全部标的每分钟涨跌家数聚合（当日可观测）
    * north_net： 北向大盘日频净流 T-1 填充
    * industry money_flow：用行业指数 close 日间变化（T-1 对齐），日内恒定
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

# 指数代码（伪指数沿用沪深300 的 index_code 约定）
INDEX_CODE = "000300.SH"

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
            index_min=self._load_index_min(kline),
            breadth=self._load_breadth(kline),
            industry=self._load_industry(symbols, axis),
            macro=self.macro.load_macro(),
            north_margin=self.macro.load_north_margin(symbols),
            dragon_tiger=self.macro.load_dragon_tiger(symbols),
            meta={
                "symbols": symbols, "start": start, "end": end,
                "source": "data1+data2 (real_loader)",
                "index_min": "标的分钟等权伪指数(近似)",
                "breadth": "标的涨跌家数聚合(近似)",
                "industry_money_flow": "行业close日间变化T-1(近似)",
            },
        )
        # 统一对齐：macro T-1、龙虎榜 T+1、各表排序去重
        ds = self.aligner.align_slice(ds)
        return ds

    # ------------------------------------------------------------------
    # K 线（l2_loader 解析 + 衍生列）
    # ------------------------------------------------------------------

    def _load_kline(self, symbols: List[str], start: str, end: str) -> pd.DataFrame:
        """分钟 K 线 + 衍生列（vwap / 市值 / 涨跌停 / ST）。"""
        k = self.l2.load_kline(symbols, start, end, freq=self.freq)
        denom = k["volume"].replace(0, np.nan)
        k["vwap"] = k["amount"] / denom
        k["vwap"] = k["vwap"].fillna(k["close"])
        k["float_market_cap"] = k["close"] * 1e9  # 假想 10 亿流通股本（相对值即可）
        # 涨跌停价：A 股规则 = T-1 收盘价 × 幅度，T 日内恒定；
        # 幅度按 ST（±5%）、创业板 300-302 / 科创板 688（±20%）、主板（±10%）区分。
        basic = self.macro.load_stock_basic()
        k["is_st"] = k[SYMBOL].map(lambda s: bool(basic.get(s, False)))
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
            k["is_st"], 0.05,
            np.where(k[SYMBOL].str[:3].isin(("300", "301", "302"))
                     | k[SYMBOL].str.startswith("688"), 0.20, 0.10))
        prev = np.where(np.isfinite(prev), prev, k["close"].to_numpy())  # 首日近似
        k["up_limit"] = np.round(prev * (1.0 + ratio), 2)
        k["down_limit"] = np.round(prev * (1.0 - ratio), 2)
        return k[KLINE_COLS]

    # ------------------------------------------------------------------
    # 近似环境表（全部无未来函数）
    # ------------------------------------------------------------------

    def _load_index_min(self, kline: pd.DataFrame) -> pd.DataFrame:
        """伪指数：data1 全部标的的分钟等权价（当日可观测，无未来函数）。

        ma20/ma60 为分钟滚动均线（240min≈1 日 / 720min≈3 日），供 system 闸门。
        """
        g = kline.groupby(kline.index)
        idx = pd.DataFrame({
            "open": g["open"].mean(), "high": g["high"].mean(),
            "low": g["low"].mean(), "close": g["close"].mean(),
            "volume": g["volume"].sum(),
        })
        idx["vwap"] = idx["close"]
        idx["ma20"] = idx["close"].rolling(240, min_periods=40).mean()
        idx["ma60"] = idx["close"].rolling(720, min_periods=120).mean()
        idx["index_code"] = INDEX_CODE
        return idx[INDEX_MIN_COLS]

    def _load_breadth(self, kline: pd.DataFrame) -> pd.DataFrame:
        """广度近似：data1 全部标的每分钟涨跌家数（基准=当日前收盘近似值）。"""
        k = kline.copy()
        k["day"] = k.index.normalize()
        # 当日第一根 bar 的 close 作为前收盘近似（本 bar 数据，无未来函数）
        base = (k.sort_index().groupby(["day", SYMBOL])["close"]
                .transform("first"))
        k["up"] = k["close"] > base
        k["down"] = k["close"] < base
        g = k.groupby(k.index)
        br = pd.DataFrame({
            "advancers": g["up"].sum(), "decliners": g["down"].sum(),
        })
        br["adr"] = br["advancers"] / br["decliners"].replace(0, np.nan)
        # north_net：北向大盘日频净流 T-1 填充
        flow = self.macro.load_north_daily_flow()
        if flow is not None:
            axis = br.index[~br.index.duplicated()]
            aligned = self.aligner.align_external(
                flow, axis, ["north_net"], date_col=TRADE_DATE)
            br["north_net"] = aligned["north_net"].to_numpy()
        else:
            br["north_net"] = np.nan
        # 消费方 indicators.environment.MRS 按列读取 north_net，必须保留
        return br[[*BREADTH_COLS, "north_net"]]

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
