"""通用基础指标与数据预处理：量能均线、动态阈值、换手率分位数及统一指标入口。

所有指标仅使用当前及历史数据（rolling 只回看过去），无未来函数。
"""

import numpy as np
import pandas as pd

from indicators.shock import atr, bollinger
from indicators.trend import adx, ma, macd


def ma_volume(volume: pd.Series, window: int) -> pd.Series:
    """量能均线：成交量 N 日均值，窗口不足时输出 NaN。"""
    return ma(volume, window)


def rolling_quantile(series: pd.Series, window: int, q: float) -> pd.Series:
    """滚动分位数：窗口不足时用现有数据计算（min_periods=1），自动跳过 NaN。"""
    return series.rolling(window=window, min_periods=1).quantile(q)


def turnover_percentile(turnover: pd.Series, window: int = 60) -> pd.Series:
    """换手率百分位排名（0~1）：当前值在最近 window 日内 <= 当前值的占比（含当日）。"""
    return turnover.rolling(window=window, min_periods=1).apply(
        lambda w: float((w <= w[-1]).mean()), raw=True
    )


def add_dynamic_thresholds(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """动态阈值：基于滚动 window 日（默认 252）分位数，窗口不足时用现有数据。

    前置要求：df 已包含列 adx、bbw、atr、close；
    turnover 缺失时 thresh_turnover_low 填 NaN，不报错。
    """
    df["thresh_adx"] = rolling_quantile(df["adx"], window, 0.25)
    df["thresh_bbw"] = rolling_quantile(df["bbw"], window, 0.30)
    atr_ratio = df["atr"] / df["close"].where(df["close"] != 0)
    df["thresh_atr"] = rolling_quantile(atr_ratio, window, 0.70)
    if "turnover" in df.columns:
        df["thresh_turnover_low"] = rolling_quantile(df["turnover"], window, 0.10)
    else:
        df["thresh_turnover_low"] = float("nan")
    return df


def compute_all(
    df: pd.DataFrame,
    ma_periods=(5, 10, 20),
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    adx_period: int = 14,
    atr_period: int = 14,
    bb_window: int = 20,
    bb_num_std: float = 2.0,
    vol_ma_periods=(5, 10),
    thresh_window: int = 252,
    turn_window: int = 60,
) -> pd.DataFrame:
    """统一指标入口：计算全部指标列并追加到 DataFrame，返回新 DataFrame。

    :param df: loader 返回的标准行情 DataFrame（含 open/high/low/close/volume/turnover）
    :return: 追加全部指标列的 DataFrame（缺失数据以 NaN 填充，不会报错）

    新增列：
        ma5, ma10, ma20            均线
        dif, dea, bar              MACD
        adx                        ADX(14)
        atr                        ATR(14)
        upper, middle, lower, bbw  布林带 + 带宽
        ma5_vol, ma10_vol          量能均线
        turnover_pct               60 日换手率分位数
        thresh_adx, thresh_bbw, thresh_atr, thresh_turnover_low  动态阈值
    """
    out = df.copy()

    # 均线
    for p in ma_periods:
        out[f"ma{p}"] = ma(out["close"], p)

    # MACD
    m = macd(out["close"], macd_fast, macd_slow, macd_signal)
    out["dif"], out["dea"], out["bar"] = m["dif"], m["dea"], m["bar"]

    # 趋势与波动
    out["adx"] = adx(out["high"], out["low"], out["close"], adx_period)
    out["atr"] = atr(out["high"], out["low"], out["close"], atr_period)

    # 布林带
    bb = bollinger(out["close"], bb_window, bb_num_std)
    out["upper"], out["middle"], out["lower"], out["bbw"] = (
        bb["upper"], bb["middle"], bb["lower"], bb["bbw"],
    )

    # 量能均线
    for p in vol_ma_periods:
        out[f"ma{p}_vol"] = ma_volume(out["volume"], p)

    # 换手率分位数（缺 turnover 列时填 NaN）
    if "turnover" in out.columns:
        out["turnover_pct"] = turnover_percentile(out["turnover"], turn_window)
    else:
        out["turnover_pct"] = float("nan")

    # 动态阈值（依赖上方已算出的 adx/bbw/atr）
    add_dynamic_thresholds(out, thresh_window)

    return out
