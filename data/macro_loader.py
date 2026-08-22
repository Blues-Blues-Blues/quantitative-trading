"""宏观与日频数据清洗器：data2 CSV → 标准日频长表。

数据来源（data/data2/*.csv）：
    global_macro_2023_2026.csv       全球宏观：美股收盘（SPX/NDX/DOW）、
                                      大宗（BRENT/GOLD/COPPER）、美债 US10Y、
                                      亚太（HSI/NKY）；DXY 缺失 → NaN 告警
    hsgt_north_holdings.csv          北向个股持股（持股数量 / 今日增持资金）
    margin_daily_*.csv               两融余额（融资余额 / 融券余额，可多文件分段）
    industry_sentiment_history.csv   行业指数（close / ma20 / rsi14，90 行业）
    hsgt_north_daily_flow.csv        北向大盘日频净流（当日资金流入）
    lhb_summary_*.csv                龙虎榜（分类净额：其他 / 北向资金 / 机构 / 游资）
    stock_basic_info.csv             股票基础信息（代码 / 名称 / ST 标记）

清洗规则（严格数据类型）：
- 列名 → 标准 schema（data.dataslice 常量）
- symbol 一律 6 位零填充字符串（源 CSV 可能把代码读成 int 导致前导 0 丢失）
- trade_date / avail_date 一律 datetime64
- 价格 / 数量 / 金额列一律 float64；布尔列 bool
- 缺列 / 缺值 → Warning 日志且不中断（如 DXY 缺失、某表不存在）
- 本模块只负责读取与清洗；时间对齐（T-1 asof / ffill）由 data.aligner 完成，
  load_*_aligned 提供「严格 T-1」便捷组合入口

「T 日 09:15 前完成切片与 ffill」语义：
外部日频数据在 A 股 T 日开盘前（09:15）已完整可得。TimeAligner 的严格 T-1 asof
（外部日期必须严格早于 A 股时点，allow_exact_matches=False）即等价于
「09:15 截断快照 + 向前填充」——任一 A 股分钟时点只用 09:15 前已披露的数据。
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from data.aligner import TimeAligner
from data.dataslice import (
    MACRO_COLS,
    NORTH_MARGIN_COLS,
    SYMBOL,
    TRADE_DATE,
)

logger = logging.getLogger("data.macro_loader")

# 标准列 dtype 映射（严格数据类型定义）
_DTYPES: Dict[str, str] = {
    TRADE_DATE: "datetime64[ns]",
    "avail_date": "datetime64[ns]",
    SYMBOL: "string",
    "north_holding": "float64",          # 北向持股（股）
    "north_buy_net": "float64",          # 北向当日净买（元）
    "margin_fin_balance": "float64",     # 融资余额（元）
    "margin_sec_balance": "float64",     # 融券余额（元）
    "north_net": "float64",              # 北向大盘净流（元）
    "close": "float64",
    "ma20": "float64",
    "rsi14": "float64",
    "money_flow": "float64",
    "buy_amount": "float64",
    "sell_amount": "float64",
    "net_amount": "float64",
    "side": "float64",
    "is_st": "bool",
}


class MacroDataLoader:
    """data2 日频 CSV → 标准日频长表（清洗、不做时间对齐）。

    :param data2_root: data/data2 目录
    """

    def __init__(self, data2_root: Path) -> None:
        self.root = Path(data2_root)
        self.aligner = TimeAligner()

    # ------------------------------------------------------------------
    # 全球宏观（T-1 由 TimeAligner 对齐）
    # ------------------------------------------------------------------

    def load_macro(self) -> pd.DataFrame:
        """全球宏观日频表：[trade_date] + MACRO_COLS（9 列）。

        列映射：SPX_close→us_spx、NDX_close→us_ndx、DOW_close→us_dow、
        BRENT_close→brent、COME_GOLD_close→gold、LME_COPPER_close→copper、
        US10Y_close→us10y、HSI_close→hsi、NKY_close→nky。
        源文件缺列（如 DXY）→ 对应列为 NaN + Warning，不中断。
        """
        f = self.root / "global_macro_2023_2026.csv"
        df = self._read_csv(f, "全球宏观")
        if df is None or df.empty:
            logger.warning("全球宏观数据缺失，返回空表（后续对齐为全 NaN）")
            return pd.DataFrame({TRADE_DATE: pd.to_datetime([])}
                                ).reindex(columns=[TRADE_DATE] + MACRO_COLS)
        out = pd.DataFrame({TRADE_DATE: self._to_datetime(df["date"])})
        src_map = {
            "SPX_close": "us_spx", "NDX_close": "us_ndx", "DOW_close": "us_dow",
            "BRENT_close": "brent", "COME_GOLD_close": "gold",
            "LME_COPPER_close": "copper", "US10Y_close": "us10y",
            "HSI_close": "hsi", "NKY_close": "nky",
        }
        for src, dst in src_map.items():
            if src in df.columns:
                out[dst] = pd.to_numeric(df[src], errors="coerce").to_numpy()
            else:
                self._warn_missing(f.name, src, dst)
                out[dst] = np.nan
        if "DXY" in df.columns:
            out["dxy"] = pd.to_numeric(df["DXY"], errors="coerce").to_numpy()
        else:
            self._warn_missing(f.name, "DXY", "dxy")
            out["dxy"] = np.nan
        return self._coerce(out, [TRADE_DATE] + MACRO_COLS)

    # ------------------------------------------------------------------
    # 北向 / 两融
    # ------------------------------------------------------------------

    def load_north_holdings(self, symbols: List[str]) -> pd.DataFrame:
        """北向个股持股日频长表：[symbol, trade_date, north_holding, north_buy_net]。"""
        f = self.root / "hsgt_north_holdings.csv"
        want = set(symbols)
        df = self._read_csv(f, "北向持股")
        if df is None or df.empty:
            return self._empty_north_holdings()
        try:
            h = df[df["代码"].astype(str).isin(want)]
        except KeyError as e:
            logger.warning("北向持股缺列 %s，返回空表", e)
            return self._empty_north_holdings()
        if h.empty:
            return self._empty_north_holdings()
        h = h.rename(columns={
            "代码": SYMBOL, "持股日期": TRADE_DATE,
            "持股数量": "north_holding", "今日增持资金": "north_buy_net"})
        h = h[[SYMBOL, TRADE_DATE, "north_holding", "north_buy_net"]].copy()
        h[SYMBOL] = h[SYMBOL].astype(str).str.zfill(6)
        h[TRADE_DATE] = self._to_datetime(h[TRADE_DATE])
        return self._coerce(h, [SYMBOL, TRADE_DATE, "north_holding", "north_buy_net"])

    def load_margin(self, symbols: List[str]) -> pd.DataFrame:
        """两融余额日频长表（多文件分段 concat）：[symbol, trade_date, 融资/融券余额]。"""
        want = set(symbols)
        parts = []
        for f in sorted(self.root.glob("margin_daily_*.csv")):
            df = self._read_csv(f, "两融")
            if df is None or df.empty:
                continue
            try:
                m = df[df["证券代码"].astype(str).isin(want)]
            except KeyError as e:
                logger.warning("两融文件 %s 缺列 %s，跳过", f.name, e)
                continue
            if m.empty:
                continue
            m = m.rename(columns={
                "证券代码": SYMBOL, "日期": TRADE_DATE,
                "融资余额": "margin_fin_balance",
                "融券余额": "margin_sec_balance"})
            m = m[[SYMBOL, TRADE_DATE, "margin_fin_balance",
                   "margin_sec_balance"]].copy()
            m[SYMBOL] = m[SYMBOL].astype(str).str.zfill(6)
            m[TRADE_DATE] = self._to_datetime(m[TRADE_DATE])
            parts.append(m)
        if not parts:
            logger.warning("未找到任何两融数据（margin_daily_*.csv），返回空表")
            return pd.DataFrame(columns=[SYMBOL, TRADE_DATE,
                                         "margin_fin_balance",
                                         "margin_sec_balance"])
        return self._coerce(pd.concat(parts, ignore_index=True),
                            [SYMBOL, TRADE_DATE, "margin_fin_balance",
                             "margin_sec_balance"])

    def load_north_margin(self, symbols: List[str]) -> pd.DataFrame:
        """北向持股 + 两融余额外连接合并：[symbol, trade_date, ...NORTH_MARGIN_COLS]。"""
        out = self.load_north_holdings(symbols).merge(
            self.load_margin(symbols), on=[SYMBOL, TRADE_DATE], how="outer")
        return self._coerce(out, NORTH_MARGIN_COLS)

    def load_north_daily_flow(self) -> Optional[pd.DataFrame]:
        """北向大盘日频净流：[trade_date, north_net]；文件缺失返回 None。"""
        f = self.root / "hsgt_north_daily_flow.csv"
        df = self._read_csv(f, "北向大盘净流")
        if df is None or df.empty:
            return None
        try:
            out = pd.DataFrame({
                TRADE_DATE: self._to_datetime(df["日期"]),
                "north_net": pd.to_numeric(df["当日资金流入"],
                                           errors="coerce").to_numpy(),
            })
        except KeyError as e:
            logger.warning("北向大盘净流缺列 %s，返回 None", e)
            return None
        return self._coerce(out.dropna(subset=["north_net"]),
                            [TRADE_DATE, "north_net"])

    # ------------------------------------------------------------------
    # 行业情绪
    # ------------------------------------------------------------------

    def load_industry_sentiment(
        self, industries: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """行业指数日频表：[行业, trade_date, close, ma20, rsi14, money_flow]。

        money_flow = 行业指数 close 的日间变化（T-1 可见，无未来函数）。
        :param industries: 只保留指定行业；None 时返回全部
        """
        f = self.root / "industry_sentiment_history.csv"
        df = self._read_csv(f, "行业情绪")
        if df is None or df.empty:
            return pd.DataFrame()
        if industries:
            df = df[df["行业"].isin(industries)]
        if df.empty:
            return pd.DataFrame()
        out = df.rename(columns={"date": TRADE_DATE})[
            ["行业", TRADE_DATE, "close", "ma20", "rsi14"]].copy()
        out[TRADE_DATE] = self._to_datetime(out[TRADE_DATE])
        out["close"] = pd.to_numeric(out["close"], errors="coerce").to_numpy()
        # 行业资金流近似：指数 close 日间变化（当日 T 的 money_flow 在 T-1 已可推断，
        # 经 aligner 严格 T-1 对齐后当日盘内不可见）
        out["money_flow"] = (out.groupby("行业")["close"]
                             .diff().fillna(0.0).to_numpy())
        return self._coerce(out, ["行业", TRADE_DATE, "close", "ma20",
                                  "rsi14", "money_flow"])

    # ------------------------------------------------------------------
    # 龙虎榜 / 股票基础信息
    # ------------------------------------------------------------------

    def load_dragon_tiger(self, symbols: List[str]) -> pd.DataFrame:
        """龙虎榜日频长表：[symbol, trade_date, buy/sell/net_amount, side]。

        分类净额（其他/北向资金/机构/游资）求和为 net_amount；
        无买卖明细 → buy_amount/sell_amount/side 恒 NaN。
        avail_date 由 TimeAligner.align_dragon_tiger 标注 T+1 可用日。
        """
        want = set(symbols)
        parts = []
        for f in sorted(self.root.glob("lhb_summary_*.csv")):
            df = self._read_csv(f, "龙虎榜")
            if df is None or df.empty:
                continue
            try:
                d = df[df["股票代码"].astype(str).isin(want)]
            except KeyError as e:
                logger.warning("龙虎榜文件 %s 缺列 %s，跳过", f.name, e)
                continue
            if d.empty:
                continue
            d = d.rename(columns={"股票代码": SYMBOL, "上榜日": TRADE_DATE})
            d[SYMBOL] = d[SYMBOL].astype(str).str.zfill(6)
            d[TRADE_DATE] = self._to_datetime(d[TRADE_DATE])
            d["net_amount"] = d[["其他", "北向资金", "机构", "游资"]].sum(axis=1)
            d["buy_amount"] = np.nan
            d["sell_amount"] = np.nan
            d["side"] = np.nan
            parts.append(d[[SYMBOL, TRADE_DATE, "buy_amount",
                            "sell_amount", "net_amount", "side"]])
        if not parts:
            logger.info("回测区间内无龙虎榜记录，返回空表")
            return pd.DataFrame(columns=[SYMBOL, TRADE_DATE, "buy_amount",
                                         "sell_amount", "net_amount", "side"])
        out = pd.concat(parts, ignore_index=True)
        out["avail_date"] = pd.NaT
        return self._coerce(out, [SYMBOL, TRADE_DATE, "avail_date",
                                  "buy_amount", "sell_amount",
                                  "net_amount", "side"])

    def load_stock_basic(self) -> Dict[str, bool]:
        """股票基础信息：{代码(6位): 是否 ST}。"""
        f = self.root / "stock_basic_info.csv"
        df = self._read_csv(f, "股票基础信息")
        if df is None or df.empty:
            return {}
        try:
            return {str(r["代码"]).zfill(6): bool(r["ST标记"])
                    for _, r in df[["代码", "ST标记"]].iterrows()}
        except KeyError as e:
            logger.warning("股票基础信息缺列 %s，返回空映射", e)
            return {}

    def load_stock_basic_full(self) -> Optional[pd.DataFrame]:
        """股票基础信息全表（代码/名称/上市日期/ST标记/停牌…）。

        供横截面成分股过滤使用（上市日期 → 次新股剔除、ST 标记 → 剔除）；
        列名保留源文件中文命名，代码列统一为 6 位字符串。
        """
        f = self.root / "stock_basic_info.csv"
        df = self._read_csv(f, "股票基础信息")
        if df is None or df.empty:
            return None
        df = df.copy()
        if "代码" in df.columns:
            df["代码"] = df["代码"].astype(str).str.zfill(6)
        return df

    # ------------------------------------------------------------------
    # 便捷组合入口：严格 T-1 切片 + ffill
    # ------------------------------------------------------------------

    def load_macro_aligned(
        self, cn_axis: Union[pd.DatetimeIndex, List],
        value_cols: List[str] = MACRO_COLS,
    ) -> pd.DataFrame:
        """全球宏观按严格 T-1 对齐到 A 股时间轴（等价 09:15 截断快照 + ffill）。

        :param cn_axis: A 股交易时间轴（DatetimeIndex）
        :return: index=cn_axis、列=value_cols 的 DataFrame
        """
        return self.aligner.align_external(
            self.load_macro(), cn_axis, value_cols, date_col=TRADE_DATE)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _read_csv(f: Path, name: str) -> Optional[pd.DataFrame]:
        """健壮读取：文件缺失/损坏 → Warning + None，不中断。"""
        if not f.exists():
            logger.warning("缺失数据文件 %s（%s），相关表将为空", f.name, name)
            return None
        try:
            return pd.read_csv(f, encoding="utf-8-sig")
        except Exception as e:  # noqa: BLE001 - 文件级容错
            logger.warning("读取 %s 失败（%s），返回空表", f.name, e)
            return None

    @staticmethod
    def _to_datetime(s: pd.Series) -> pd.Series:
        """宽松日期解析：int(YYYYMMDD) / str(YYYY-MM-DD) → datetime64[ns]。

        强制 ns：pandas 2.x 字符串解析可能返回 datetime64[us]，
        与 A 股分钟轴（ns）在 merge_asof 时 dtype 不匹配。
        """
        return pd.to_datetime(s.astype(str), errors="coerce") \
                 .astype("datetime64[ns]")

    @staticmethod
    def _coerce(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """强制标准 dtype；布尔列填充 False 后转 bool。"""
        out = df[cols].copy()
        for c in cols:
            dt = _DTYPES.get(c)
            if dt is None:
                continue
            if dt == "string":
                out[c] = out[c].astype(str)
            elif dt == "bool":
                out[c] = out[c].fillna(False).astype(bool)
            elif dt.startswith("datetime64"):
                # datetime 列保持时间语义（to_numeric 会破坏 datetime64 单位）
                out[c] = pd.to_datetime(out[c], errors="coerce").astype(dt)
            else:
                out[c] = pd.to_numeric(out[c], errors="coerce").astype(dt)
        return out

    @staticmethod
    def _warn_missing(file_name: str, src: str, dst: str) -> None:
        logger.warning("宏观文件 %s 缺列 %s，标准列 %s 置 NaN", file_name, src, dst)

    @staticmethod
    def _empty_north_holdings() -> pd.DataFrame:
        return pd.DataFrame(columns=[SYMBOL, TRADE_DATE,
                                     "north_holding", "north_buy_net"])
