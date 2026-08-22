"""标准数据切片 DataSlice：多数据帧容器 + 标准列常量。

设计目标：回测 / 因子层只面向统一的 DataSlice，不关心数据来自 Level-2 快照、
逐笔成交、分钟 K 线、宏观还是龙虎榜。各表统一为「长表」结构：
    index = DatetimeIndex（A 股交易时间戳，升序、去重）
    列   = [symbol, 字段...]（宏观表无 symbol，为全市场横截面）

防未来函数约定：
- macro / index_min 等外部来源数据已完成 T-1 全量对齐（见 data.aligner）
- dragon_tiger 为龙虎榜事件表，仅披露次日（T+1）后可用，禁止参与 T 日实时特征
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import pandas as pd

# ---------- 通用列名 ----------
SYMBOL = "symbol"
TRADE_DATE = "trade_date"

# ---------- 个股基础 K 线（1 分钟） ----------
KLINE_COLS: List[str] = [
    SYMBOL, "open", "high", "low", "close", "volume", "amount",
    "vwap", "float_market_cap", "up_limit", "down_limit", "is_st",
]

# ---------- Level-2 快照（五档盘口） ----------
L2_SNAPSHOT_COLS: List[str] = [
    SYMBOL,
    "bid1_p", "bid1_v", "bid2_p", "bid2_v", "bid3_p", "bid3_v",
    "bid4_p", "bid4_v", "bid5_p", "bid5_v",
    "ask1_p", "ask1_v", "ask2_p", "ask2_v", "ask3_p", "ask3_v",
    "ask4_p", "ask4_v", "ask5_p", "ask5_v",
]

# ---------- 逐笔成交（Side: 1=主动买 -1=主动卖 0=中性；is_cancel: 是否撤单记录） ----------
TICK_COLS: List[str] = [SYMBOL, "price", "volume", "turnover", "side", "is_cancel"]

# ---------- 沪深300 指数 1 分钟 ----------
INDEX_MIN_COLS: List[str] = [
    "index_code", "open", "high", "low", "close", "volume", "vwap", "ma20", "ma60",
]

# ---------- 全市场广度（ADR = 上涨家数 / 下跌家数） ----------
BREADTH_COLS: List[str] = ["advancers", "decliners", "adr"]

# ---------- 行业（中信一级指数行情 + 资金流向） ----------
INDUSTRY_COLS: List[str] = [
    "industry", "open", "high", "low", "close", "volume", "money_flow",
]

# ---------- 全球宏观 / 隔夜数据（T-1 全量对齐后进入本表） ----------
MACRO_COLS: List[str] = [
    "us_spx", "us_ndx", "us_dow",   # 美股收盘
    "brent", "gold", "copper",      # 大宗商品
    "us10y", "dxy",                 # 美债 / 美元指数
    "hsi", "nky",                   # 亚太早盘
]

# ---------- 龙虎榜（T+1 因子，avail_date 为披露次日） ----------
DRAGON_TIGER_COLS: List[str] = [
    SYMBOL, TRADE_DATE, "avail_date",
    "buy_amount", "sell_amount", "net_amount", "side",
]

# ---------- 北向 / 两融（日频，T+1 披露后才可见，必须对齐后再使用） ----------
NORTH_MARGIN_COLS: List[str] = [
    SYMBOL, TRADE_DATE,
    "north_holding",      # 北向持股（股）
    "north_buy_net",      # 北向当日净买（元）
    "margin_fin_balance", # 融资余额（元）
    "margin_sec_balance", # 融券余额（元）
]


@dataclass
class DataSlice:
    """多数据帧容器：一次加载/对齐得到的标准数据切片。

    所有 DataFrame 均要求 index 为升序、去重的 DatetimeIndex；
    含 symbol 的表为长表（同一时间戳可有多行，对应不同标的）。
    """

    kline: pd.DataFrame
    l2_snapshot: Optional[pd.DataFrame] = None
    tick_trades: Optional[pd.DataFrame] = None
    index_min: Optional[pd.DataFrame] = None
    breadth: Optional[pd.DataFrame] = None
    industry: Optional[pd.DataFrame] = None
    macro: Optional[pd.DataFrame] = None
    north_margin: Optional[pd.DataFrame] = None
    dragon_tiger: Optional[pd.DataFrame] = None
    factors: Optional[pd.DataFrame] = None
    meta: Dict[str, object] = field(default_factory=dict)

    # ---------- 便捷访问 ----------

    def symbols(self) -> List[str]:
        """去重后的标的列表（按首次出现顺序）。"""
        if SYMBOL in self.kline.columns:
            return list(pd.unique(self.kline[SYMBOL]))
        return []

    def time_axis(self) -> pd.DatetimeIndex:
        """A 股交易时间轴（主键）：取 kline 的去重时间戳作为对齐基准。

        kline 为长表（同一时间戳对应多只标的），故需去重后返回。
        """
        if self.kline.empty:
            raise ValueError("kline 为空，无法确定 A 股时间轴")
        idx = self.kline.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.DatetimeIndex(idx)
        return idx[~idx.duplicated()]

    def subset(self, start: pd.Timestamp, end: pd.Timestamp) -> "DataSlice":
        """按时间窗 [start, end]（含两端）裁剪全部数据表，返回新 DataSlice。

        用于 Walk-Forward 的样本内/样本外切分：各表按各自时间索引
        （kline 为分钟轴，日频表为 trade_date）做范围过滤。
        """
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        # 日级边界（00:00）语义为「含当日全天」：右端取次日的开区间，
        # 否则分钟级表（行时间均晚于 00:00）会被 `ts <= end` 整日截掉
        hi = end + pd.Timedelta(days=1) if end == end.normalize() else end
        fields: Dict[str, Optional[pd.DataFrame]] = {}
        for name in ("kline", "l2_snapshot", "tick_trades", "index_min",
                     "breadth", "industry", "macro", "north_margin",
                     "dragon_tiger", "factors"):
            df: Optional[pd.DataFrame] = getattr(self, name)
            if df is None or df.empty:
                fields[name] = df
                continue
            # 事件表/日频表按自身日期键过滤：
            # 龙虎榜用披露可用日 avail_date，macro/north_margin 用 trade_date，
            # 其余表按索引时间过滤
            if name == "dragon_tiger" and "avail_date" in df.columns:
                ts = df["avail_date"]
            elif "trade_date" in df.columns:
                ts = df["trade_date"]
            else:
                ts = df.index
            fields[name] = df[(ts >= start) & (ts < hi)]
        return DataSlice(**fields, meta=dict(self.meta))

    # ---------- 校验 ----------

    def validate(self, required_cols: Optional[Dict[str, Iterable[str]]] = None) -> None:
        """校验所有表的 schema 与时间索引质量。

        :param required_cols: {表名: 必需列}，缺省时用各表标准列校验
        :raises ValueError: 索引非法 / 必需列缺失
        """
        rules = {
            "kline": KLINE_COLS,
            "l2_snapshot": L2_SNAPSHOT_COLS,
            "tick_trades": TICK_COLS,
            "index_min": INDEX_MIN_COLS,
            "breadth": BREADTH_COLS,
            "industry": INDUSTRY_COLS,
            "macro": MACRO_COLS,
            "north_margin": NORTH_MARGIN_COLS,
            "dragon_tiger": DRAGON_TIGER_COLS,
            "factors": [],
        }
        if required_cols:
            rules.update(required_cols)

        for name in rules:
            df = getattr(self, name)
            if df is None or df.empty:
                continue
            # 事件表与日频外部表跳过索引校验：
            # - 逐笔成交/龙虎榜为事件表（同秒多笔、同日多条属正常形态）
            # - macro/north_margin 以 trade_date 列为主键（aligner 按列排序），
            #   索引可为 RangeIndex；factor 缓存为特征长表（索引即时间）
            if name not in ("dragon_tiger", "tick_trades", "factors",
                            "macro", "north_margin"):
                self._check_index(df, name)
            need = [c for c in rules[name] if c not in df.columns]
            if need:
                raise ValueError(f"{name} 缺少必需列: {need}")

        # 龙虎榜必须携带可用日，防止误入 T 日特征
        if self.dragon_tiger is not None and not self.dragon_tiger.empty:
            if "avail_date" not in self.dragon_tiger.columns:
                raise ValueError("dragon_tiger 缺少 avail_date（披露次日）列，禁止直接使用")

    @staticmethod
    def _check_index(df: pd.DataFrame, name: str) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{name} 的 index 必须为 DatetimeIndex")
        if not df.index.is_monotonic_increasing:
            raise ValueError(f"{name} 的 index 未按时间升序，请先排序")
        if SYMBOL in df.columns:
            # 长表：同一时间戳可有多行（不同标的），但 (时间戳, symbol) 必须唯一
            res = df.reset_index()
            idx_col = res.columns[0]  # 索引列名（命名索引取原名，未命名取 "index"）
            dup = res.duplicated(subset=[idx_col, SYMBOL])
            if dup.any():
                raise ValueError(f"{name} 存在重复的 (时间戳, symbol) 行")
        elif "industry" in df.columns:
            # 行业长表：同一时间戳可有多行（不同行业），按 (时间戳, industry) 排重
            res = df.reset_index()
            idx_col = res.columns[0]
            dup = res.duplicated(subset=[idx_col, "industry"])
            if dup.any():
                raise ValueError(f"{name} 存在重复的 (时间戳, industry) 行")
        elif df.index.has_duplicates:
            raise ValueError(f"{name} 的 index 存在重复时间戳")
