"""演示脚本：可视化回测段的趋势/震荡区间标注。

运行：python demo_plot.py
输出：analytics/pictures/ 下生成价格走势图（黄色区域 = 震荡区间），
     命令行会打印图片保存路径，打开即可查看。
"""

from analytics.plotter import plot_trend
from config import settings
from data.loader import session, load_daily
from indicators.basic import compute_all

# 回测标的时间段（时间段从 config/settings.py 读取）
SYMBOL = "600000"
WARMUP_DAYS = 80

with session():
    df = load_daily(
        SYMBOL,
        settings.START_DATE,
        settings.END_DATE,
        adjustflag=2,          # 前复权
        warmup_days=WARMUP_DAYS,
    )

out = compute_all(df)
backtest = out[out.index >= settings.START_DATE]  # 切片：只画回测段

print(f"回测段 {backtest.index[0].date()} ~ {backtest.index[-1].date()}，"
      f"共 {len(backtest)} 个交易日")
print(f"震荡日 {int((~backtest['is_trend']).sum())} 天，"
      f"趋势日 {int(backtest['is_trend'].sum())} 天")

path = plot_trend(backtest)
print(f"图片已保存: {path}")
