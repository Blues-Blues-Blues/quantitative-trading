"""通用基础指标与数据预处理：量能均线、动态阈值、换手率分位数及统一指标入口。

所有指标仅使用当前及历史数据（rolling 只回看过去），无未来函数。
"""

import numpy as np
import pandas as pd

from indicators.shock import atr, bollinger, residual_volatility
from indicators.trend import adx, ma, ma_slope, macd


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


def trend_quality(df: pd.DataFrame) -> pd.Series:
    """趋势质量 TQ（0~3）：趋势强度 + 均线多头排列 + 价格站上 MA20。

    T1: ADX >= thresh_adx
    T2: MA5 > MA10 > MA20
    T3: Close > MA20

    任一条件遇 NaN 视为不满足（不加分）。
    """
    t1 = df["adx"] >= df["thresh_adx"]
    t2 = (df["ma5"] > df["ma10"]) & (df["ma10"] > df["ma20"])
    t3 = df["close"] > df["ma20"]
    return t1.astype(int) + t2.astype(int) + t3.astype(int)


def volume_confirm(df: pd.DataFrame) -> pd.Series:
    """量能确认 VC（0~2）：量能放大 + 换手率分位不低。

    V1: MA5_VOL > MA10_VOL
    V2: turnover_percentile >= 0.4

    任一条件遇 NaN 视为不满足（不加分）。
    """
    v1 = df["ma5_vol"] > df["ma10_vol"]
    v2 = df["turnover_pct"] >= 0.4
    return v1.astype(int) + v2.astype(int)


def momentum_decay_cond(df: pd.DataFrame) -> pd.Series:
    """动量衰减条件（bool）：上升趋势中 MACD 柱动能减弱。

    近期均值 = (BAR[t-2] + BAR[t-1]) / 2
    前期均值 = (BAR[t-4] + BAR[t-3]) / 2
    条件 = DIF > 0 且 DIF > DEA 且 BAR > 0 且 (近期均值 < 前期均值)

    注意：条件本身不含"持仓天数"约束；"持仓 >= 5 交易日"的门控由策略层
    结合持仓状态判断（mdm_cond 且持仓天数 >= 5 才视为动量衰减卖出信号）。
    """
    recent = (df["bar"].shift(2) + df["bar"].shift(1)) / 2
    prior = (df["bar"].shift(4) + df["bar"].shift(3)) / 2
    return (
        (df["dif"] > 0)
        & (df["dif"] > df["dea"])
        & (df["bar"] > 0)
        & (recent < prior)
    )


def add_dynamic_thresholds(
    df: pd.DataFrame,
    window: int = 252,
    resid_window: int = 250,
) -> pd.DataFrame:
    """动态阈值：基于滚动分位数，窗口不足时用现有数据计算（min_periods=1）。

    前置要求：df 已包含列 adx、atr、close、resid_vol20；
    turnover 缺失时 thresh_turnover_low 填 NaN，不报错。
    """
    df["thresh_adx"] = rolling_quantile(df["adx"], window, 0.25)
    # thresh_bbw 仅旧的 is_shock 判定使用，已废弃，注释保留备查
    # df["thresh_bbw"] = rolling_quantile(df["bbw"], window, 0.30)
    atr_ratio = df["atr"] / df["close"].where(df["close"] != 0)
    df["thresh_atr"] = rolling_quantile(atr_ratio, window, 0.70)
    df["thresh_resid_vol"] = rolling_quantile(df["resid_vol20"], resid_window, 0.70)
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
    trend_ma: int = 60,
    slope_window: int = 5,
    resid_window: int = 20,
    resid_thresh_window: int = 250,
    slope_threshold: float = 0.0,
) -> pd.DataFrame:
    """统一指标入口：计算全部指标列并追加到 DataFrame，返回新 DataFrame。

    :param df: loader 返回的标准行情 DataFrame（含 open/high/low/close/volume/turnover）
    :param slope_threshold: MA60 5 日斜率阈值（默认 0，主轴向上即可）
    :return: 追加全部指标列的 DataFrame（缺失数据以 NaN 填充，不会报错）

    新增列：
        ma5, ma10, ma20            均线
        dif, dea, bar              MACD
        adx                        ADX(14)
        atr                        ATR(14)
        upper, middle, lower, bbw  布林带 + 带宽
        ma5_vol, ma10_vol          量能均线
        turnover_pct               60 日换手率分位数
        ma60, ma60_slope5          趋势主轴 MA60 及其 5 日斜率
        resid_vol20                20 日残差波动率（偏离 MA60 的滚动标准差）
        thresh_adx, thresh_atr, thresh_resid_vol, thresh_turnover_low  动态阈值
        is_trend                   趋势判定（True=趋势，False=震荡；NaN 视为震荡）
        tq                         趋势质量打分 0~3
        vc                         量能确认打分 0~2
        mdm_cond                   动量衰减条件（bool，不含持仓天数门控）
    """
    out = df.copy()

    # 均线
    for p in ma_periods:
        out[f"ma{p}"] = ma(out["close"], p)

    # MA60 趋势主轴及其斜率、残差波动率
    out["ma60"] = ma(out["close"], trend_ma)
    out["ma60_slope5"] = ma_slope(out["ma60"], slope_window)
    out["resid_vol20"] = residual_volatility(out["close"], out["ma60"], resid_window)

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

    # 动态阈值（依赖上方已算出的 adx/atr/resid_vol20）
    add_dynamic_thresholds(out, thresh_window, resid_thresh_window)

    # 趋势主轴判定 is_trend：MA60 5日斜率 > 阈值 且 残差波动率 < 动态阈值；
    # 任一条件不满足（含 NaN）视为震荡，不交易。
    out["is_trend"] = (out["ma60_slope5"] > slope_threshold) & (
        out["resid_vol20"] < out["thresh_resid_vol"]
    )

    # 旧的震荡市判定（ADX + BBW），已废弃，改用 is_trend，注释保留备查
    # shock = (out["adx"] < out["thresh_adx"]) & (out["bbw"] < out["thresh_bbw"])
    # missing = out[["adx", "bbw", "thresh_adx", "thresh_bbw"]].isna().any(axis=1)
    # out["is_shock"] = shock | missing

    # 因子与打分
    out["tq"] = trend_quality(out)
    out["vc"] = volume_confirm(out)
    out["mdm_cond"] = momentum_decay_cond(out)

    return out
