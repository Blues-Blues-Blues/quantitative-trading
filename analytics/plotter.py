"""绘图模块：K 线图、买卖点标记、净值曲线图等可视化。

当前已实现：
- plot_trend：收盘价折线 + MA5 + MA20 + 震荡区间（半透明黄色色带）

图片默认保存到 analytics/pictures/ 目录。
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

# 中文字体（Windows），避免图例/标题乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 默认图片保存目录：analytics/pictures/
_PICTURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "analytics", "pictures",
)


def _shock_blocks(is_trend: pd.Series) -> list:
    """找出 is_trend == False 的连续震荡区间 [(起始日, 结束日), ...]。

    输入需为升序日期索引的布尔序列；首尾处于震荡时会从数据首/尾开始铺色带。
    """
    blocks = []
    start = None
    prev = None
    for dt, v in is_trend.items():
        if not v and start is None:
            start = dt
        elif v and start is not None:
            blocks.append((start, prev))
            start = None
        prev = dt
    if start is not None:
        blocks.append((start, prev))
    return blocks


def plot_trend(
    df: pd.DataFrame,
    fname: str = None,
    title: str = None,
    figsize: tuple = (16, 6),
) -> str:
    """绘制收盘价走势 + MA5 + MA20 + MA60（虚线），震荡区间用半透明黄色标注。

    :param df: compute_all 输出的 DataFrame（需含 close/ma5/ma20/is_trend 列；
        含 ma60 列时自动加画 MA60 虚线）。
        若含 code 列，标题会自动标注股票代码。
        注意：若 df 含预热段，请先切片为回测段再传入，本函数按传入数据原样绘制。
    :param fname: 文件名（不含路径时保存到 analytics/pictures/）。
        默认 "price_{起始}_{结束}.png"
    :param title: 图表标题，默认自动生成（含股票代码）
    :return: 保存的图片完整路径
    """
    if df.empty:
        raise ValueError("输入 DataFrame 为空，无法绘图")

    os.makedirs(_PICTURES_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    # 价格与均线
    ax.plot(df.index, df["close"], label="收盘价", color="#1f77b4", linewidth=1.2)
    ax.plot(df.index, df["ma5"], label="MA5", color="#ff7f0e", linewidth=1.0)
    ax.plot(df.index, df["ma20"], label="MA20", color="#2ca02c", linewidth=1.0)
    if "ma60" in df.columns:
        ax.plot(df.index, df["ma60"], label="MA60", color="#9467bd",
                linewidth=1.2, linestyle="--")

    # 震荡区间（is_trend == False 的连续段）半透明黄色
    for s, e in _shock_blocks(df["is_trend"]):
        ax.axvspan(s, e, color="yellow", alpha=0.3)

    # 标题：优先取 DataFrame 的 code 列（如 sh.600000 -> 600000）
    code = ""
    if "code" in df.columns and len(df["code"].dropna()):
        code = str(df["code"].dropna().iloc[0]).split(".")[-1]
    if not title:
        title = f"{code} " if code else ""
        title += f"价格走势（黄色区域=震荡区间）{df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d}"
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate(rotation=30)  # 日期标签斜排，避免重叠

    # 默认文件名
    if fname is None:
        fname = f"price_{df.index[0]:%Y%m%d}_{df.index[-1]:%Y%m%d}.png"
    path = fname if os.path.dirname(fname) else os.path.join(_PICTURES_DIR, fname)

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
