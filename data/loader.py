"""
数据加载器：基于 baostock 获取 A 股日线行情数据。

统一的数据加载接口，返回标准格式的 DataFrame：
    index: DatetimeIndex (按日期升序，已剔除停牌日)
    columns: open, high, low, close, volume, amount, turnover, pctChg, ...

支持本地 CSV 缓存到 data/mock_data/，避免重复联网下载。

使用方式：
    from data.loader import session, load_daily

    with session():
        df = load_daily("600000", "2020-01-01", "2021-12-31", adjustflag=ADJUST_FORWARD)
"""

import os
from contextlib import contextmanager
from typing import Dict, List

import pandas as pd

import baostock as bs

from config.data_sources import get_data_path

# 复权标志
ADJUST_BACKWARD = 1  # 后复权
ADJUST_FORWARD = 2   # 前复权
ADJUST_NONE = 3      # 不复权

# baostock 返回字段与标准列名的映射
_FIELD_MAP = {
    "date": "date",
    "code": "code",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "preclose": "preclose",
    "volume": "volume",
    "amount": "amount",
    "adjustflag": "adjustflag",
    "turn": "turnover",        # 换手率
    "tradestatus": "tradestatus",
    "pctChg": "pctChg",        # 涨跌幅
    "isST": "isST",
}

# 查询字段
_QUERY_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "adjustflag,turn,tradestatus,pctChg,isST"
)

# 数值列
_NUMERIC_COLS = [
    "open", "high", "low", "close", "preclose", "volume",
    "amount", "turnover", "pctChg",
]

# 缓存目录：默认 data/mock_data/，可通过环境变量 QTDATA_DAILY_CACHE_PATH
# 或 config/data_paths.yaml 的 daily_cache 项覆盖（见 config/data_sources.py）
_CACHE_DIR = str(get_data_path("daily_cache"))


def _to_bs_code(symbol: str) -> str:
    """将 '600000' 或 'sh.600000' 统一为 baostock 格式 'sh.600000'。"""
    symbol = symbol.strip().lower()
    if "." in symbol:
        return symbol
    if symbol.startswith(("6", "9")):
        return "sh." + symbol
    if symbol.startswith(("0", "2", "3")):
        return "sz." + symbol
    raise ValueError(f"无法识别的股票代码前缀: {symbol}（baostock 主要覆盖沪深市场）")


def login() -> None:
    """登录 baostock。已登录时先登出，避免重复登录报错。"""
    bs.logout()
    lg = bs.login()
    if lg.error_code != "0":
        raise ConnectionError(f"baostock 登录失败: {lg.error_code} - {lg.error_msg}")


def logout() -> None:
    """登出 baostock。"""
    bs.logout()


@contextmanager
def session():
    """登录上下文管理器，退出时自动登出。"""
    login()
    try:
        yield
    finally:
        logout()


def _cache_path(symbol: str, start_date: str, end_date: str, adjustflag: int) -> str:
    """构造本地缓存文件路径。"""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    fname = f"{symbol.replace('.', '_')}_{start_date}_{end_date}_adj{adjustflag}.csv"
    return os.path.join(_CACHE_DIR, fname)


def _fetch_start_date(start_date: str, warmup_days: int) -> str:
    """计算实际拉取起始日期：按日历日向前扩展预热段。

    交易日约为日历日的 5/7，取 2 倍系数留足节假日缓冲，
    之后在 load_daily 内精确截取 warmup_days 个交易日。
    """
    if warmup_days <= 0:
        return start_date
    start = pd.to_datetime(start_date)
    buffer_days = int(round(warmup_days * 2.0))
    return (start - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")


def _slice_warmup(df: pd.DataFrame, start_date: str, warmup_days: int) -> pd.DataFrame:
    """截取预热区间：保留 start_date 之前最近 warmup_days 个交易日 + 回测段。

    新股/数据不足时保留现有全部数据（从上市日起），不报错。
    """
    if warmup_days <= 0:
        return df
    start_ts = pd.Timestamp(start_date)
    pre = df[df.index < start_ts]
    if len(pre) > warmup_days:
        pre = pre.tail(warmup_days)
    warmup_start = pre.index[0] if len(pre) else df.index[0]
    return df[df.index >= warmup_start]


def load_daily(
    symbol: str,
    start_date: str,
    end_date: str,
    adjustflag: int = ADJUST_NONE,
    use_cache: bool = True,
    warmup_days: int = 0,
) -> pd.DataFrame:
    """
    加载单只股票日线数据。

    :param symbol: 股票代码，如 "600000" 或 "sh.600000"
    :param start_date: 回测起始日期 "YYYY-MM-DD"
    :param end_date: 回测结束日期 "YYYY-MM-DD"
    :param adjustflag: 复权标志，1=后复权 2=前复权 3=不复权
    :param use_cache: 是否使用/写入本地 CSV 缓存
    :param warmup_days: 预热交易日数（默认 0）。>0 时在 start_date 之前
        额外拉取 warmup_days 个交易日的数据，保证 MA60 等长周期指标
        在回测第一天就有效；返回数据 = 预热段 + 回测段，调用方用
        `df[df.index >= start_date]` 切片即可。
    :return: 标准格式 DataFrame，DatetimeIndex 升序，剔除停牌日
    :raises ConnectionError: 未登录或登录失败
    :raises RuntimeError: baostock 查询失败
    :raises ValueError: 无数据
    """
    code = _to_bs_code(symbol)
    fetch_start = _fetch_start_date(start_date, warmup_days)
    cache_file = _cache_path(code, fetch_start, end_date, adjustflag)

    if use_cache and os.path.exists(cache_file):
        df = pd.read_csv(cache_file, parse_dates=["date"])
        df = df.set_index("date").sort_index()
        return _slice_warmup(df, start_date, warmup_days)

    rs = bs.query_history_k_data_plus(
        code,
        _QUERY_FIELDS,
        start_date=fetch_start,
        end_date=end_date,
        frequency="d",
        adjustflag=str(adjustflag),
    )
    if rs.error_code != "0":
        raise RuntimeError(f"查询失败 {code}: {rs.error_code} - {rs.error_msg}")

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        raise ValueError(f"{code} 在 {fetch_start} ~ {end_date} 无数据，请检查代码或日期范围")

    df = pd.DataFrame(rows, columns=rs.fields)
    df = df.rename(columns=_FIELD_MAP)
    df = df[df["tradestatus"] == "1"]  # 剔除停牌日
    df[_NUMERIC_COLS] = df[_NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    if use_cache:
        df.to_csv(cache_file, encoding="utf-8")

    return _slice_warmup(df, start_date, warmup_days)


def load_daily_multi(
    symbols: List[str],
    start_date: str,
    end_date: str,
    adjustflag: int = ADJUST_NONE,
    use_cache: bool = True,
    warmup_days: int = 0,
) -> Dict[str, pd.DataFrame]:
    """
    批量加载多只股票日线数据。

    :return: {symbol: DataFrame}（含预热段，各股数据量可能不同）
    """
    result = {}
    for s in symbols:
        result[s] = load_daily(s, start_date, end_date, adjustflag, use_cache, warmup_days)
    return result
