"""趋势判断指标：MA、MACD、ADX。

所有指标仅使用当前及历史数据（rolling/ewm 均只回看过去），无未来函数。
"""

import numpy as np
import pandas as pd


def ma(close: pd.Series, window: int) -> pd.Series:
    """简单移动平均，窗口不足时输出 NaN。"""
    return close.rolling(window=window).mean()


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD 指标。

    公式：
        DIF  = EMA(close, fast) - EMA(close, slow)
        DEA  = EMA(DIF, signal)
        BAR  = (DIF - DEA) * 2

    :return: DataFrame(dif, dea, bar)，ewm 暖启动（从首日起有值）
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    bar = (dif - dea) * 2
    return pd.DataFrame({"dif": dif, "dea": dea, "bar": bar})


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ADX 平均趋向指标（Wilder 平滑），取值 0~100，衡量趋势强度。

    计算步骤：TR / +DM / -DM -> Wilder 平滑 -> +DI / -DI -> DX -> ADX。
    完全横盘（+DI 与 -DI 均为 0）时 DX 输出 NaN。
    """
    prev_close = close.shift(1)

    # 真实波幅 TR；首行无昨收时 (high-prev) 为 NaN，max 自动跳过 NaN 取 high-low
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    # Wilder 平滑：alpha = 1 / period
    tr_s = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_dm_s = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
    minus_dm_s = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

    # TR 为 0（价格完全无波动）时 DI 输出 NaN，避免除零
    plus_di = 100 * plus_dm_s / tr_s.where(tr_s != 0)
    minus_di = 100 * minus_dm_s / tr_s.where(tr_s != 0)

    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum.where(di_sum != 0)

    return dx.ewm(alpha=1 / period, adjust=False).mean()
