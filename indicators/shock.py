"""震荡/区间判断指标：ATR、布林带（含带宽 BBW）。

所有指标仅使用当前及历史数据（rolling/ewm 均只回看过去），无未来函数。
"""

import pandas as pd


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ATR 平均真实波幅（Wilder 平滑），衡量波动幅度。

    首行无昨收时 TR 取 high-low。
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
    ddof: int = 0,
) -> pd.DataFrame:
    """布林带指标。

    公式：
        MIDDLE = MA(close, window)
        UPPER  = MIDDLE + num_std * STD
        LOWER  = MIDDLE - num_std * STD
        BBW    = (UPPER - LOWER) / MIDDLE    # 带宽

    :param ddof: 标准差自由度，0 = 总体标准差（与国内行情软件一致）
    :return: DataFrame(upper, middle, lower, bbw)；窗口不足时 NaN，MIDDLE 为 0 时 BBW 为 NaN
    """
    middle = close.rolling(window=window).mean()
    std = close.rolling(window=window).std(ddof=ddof)
    upper = middle + num_std * std
    lower = middle - num_std * std
    bbw = (upper - lower) / middle.where(middle != 0)
    return pd.DataFrame(
        {"upper": upper, "middle": middle, "lower": lower, "bbw": bbw}
    )
