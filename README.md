# 量化交易回测系统（A 股）

基于 [baostock](http://baostock.com/) 行情数据的 A 股量化交易回测框架。项目以「数据层 → 指标层 → 策略层 → 执行引擎层 → 评估与绘图」的分层结构组织代码，当前已实现行情数据加载、技术指标计算、市场状态判定（趋势/震荡）与仓位管理，回测引擎与策略主体仍在开发中。

## 功能特性

- **数据加载**：通过 baostock 获取沪深 A 股日线数据，支持前/后/不复权，自动剔除停牌日，本地 CSV 缓存避免重复联网
- **指标体系**：MA、MACD、ADX、ATR、布林带等常用指标，全部仅使用历史数据（无未来函数）
- **市场状态判定**：基于趋势主轴（默认 MA20）斜率 + 残差波动率的 `is_trend` 判定，区分趋势日与震荡日
- **多因子打分**：趋势质量 TQ（0~3）+ 量能确认 VC（0~2），配合动态阈值（滚动分位数）
- **风险控制**：动量衰减（MDM）预警、波动率降权、震荡市空仓、最大持股数与总仓位上限约束
- **可视化**：价格走势 + 均线 + 震荡区间色带标注，自动输出图片

## 项目结构

```
quantitative trading/
├── main.py                  # 项目入口（组装配置、加载数据、运行回测、展示结果）
├── demo_plot.py             # 演示脚本：绘制回测段价格走势与震荡区间
├── requirements.txt         # 依赖：baostock / pandas / numpy / matplotlib
├── config/
│   └── settings.py          # 全局配置：回测起止日期
├── data/
│   ├── loader.py            # 数据加载器（baostock + CSV 缓存 + 预热段）
│   └── mock_data/           # 本地行情缓存目录
├── indicators/
│   ├── trend.py             # 趋势指标：MA / MACD / ADX
│   ├── shock.py             # 波动指标：ATR / 布林带 / 残差波动率
│   ├── basic.py             # 统一指标入口 compute_all + TQ/VC 打分 + 动态阈值
│   └── __init__.py
├── strategy/
│   ├── base.py              # 策略基类（接口定义）
│   ├── my_strategy.py       # 自定义策略实现
│   ├── position.py          # 仓位管理：权重计算与持仓约束
│   └── __init__.py
├── engine/                  # 执行引擎层（开发中）
│   ├── backtest.py          # 回测循环引擎
│   ├── execution.py         # 订单执行：成交、滑点、手续费
│   ├── portfolio.py         # 账户与持仓管理
│   └── __init__.py
├── analytics/
│   ├── metrics.py           # 绩效指标：夏普、最大回撤、胜率等（开发中）
│   ├── plotter.py           # 绘图模块：走势图 / 震荡区间标注
│   └── pictures/            # 生成的图片输出目录
└── tests/
    ├── test_indicators.py   # 指标层单元测试
    └── test_strategy.py     # 策略层单元测试
```

## 安装

需要 Python 3.9+（建议 3.11+）。

```bash
pip install -r requirements.txt
```

依赖清单：`baostock`、`pandas`、`numpy`、`matplotlib`。

## 快速开始

**1. 运行演示脚本（绘制趋势/震荡可视化图）**

```bash
python demo_plot.py
```

脚本会从 `config/settings.py` 读取回测时间段，加载股票 `600000`（浦发银行）的前复权日线数据（含 80 个交易日预热），计算全部指标后绘制价格走势图（黄色区域 = 震荡区间），图片保存到 `analytics/pictures/`。

**2. 自定义回测**

修改 `config/settings.py` 中的回测起止日期：

```python
START_DATE = "2024-01-01"  # 回测起始日期
END_DATE   = "2026-06-01"  # 回测结束日期
```

## 数据加载

`data/loader.py` 提供统一的数据加载接口：

```python
from data.loader import session, load_daily

with session():  # 自动登录/登出 baostock
    df = load_daily(
        "600000",                 # 股票代码，支持 "600000" 或 "sh.600000"
        "2024-01-01",             # 起始日期
        "2026-06-01",             # 结束日期
        adjustflag=2,             # 1=后复权 2=前复权 3=不复权
        warmup_days=80,           # 预热交易日数（保证长周期指标有效）
    )
```

返回标准格式的 DataFrame：`DatetimeIndex`（升序、剔除停牌日），列为 `open / high / low / close / volume / amount / turnover / pctChg` 等。数据会自动缓存到 `data/mock_data/`，后续加载直接读缓存。

## 指标与市场状态

- `indicators/basic.py` 的 `compute_all()` 是统一指标入口，一次性计算全部指标列
- **趋势判定 `is_trend`**：趋势主轴（默认 MA20）5 日斜率 > 阈值，且 20 日残差波动率低于动态阈值 → 趋势日；否则视为震荡日
- **打分因子**：
  - `tq`（趋势质量 0~3）：ADX 强度 + 均线多头排列 + 价格站上 MA20
  - `vc`（量能确认 0~2）：量能均线放大 + 换手率分位数 ≥ 0.4
  - `mdm_cond`（动量衰减）：上升趋势中 MACD 柱动能减弱，作为卖出预警
- **动态阈值**：基于 252 日滚动分位数（ADX 25% 分位、ATR 70% 分位等）

## 仓位管理

`strategy/position.py` 负责目标权重计算与持仓约束：

- **权重映射**：TQ+VC 总分 5/4/3 分 → 25% / 18% / 10% 基础权重，≤ 2 分不持仓
- **波动率修正**：ATR/Close 超过阈值时权重打 6 折
- **MDM 修正**：动量衰减且持仓 ≥ 5 日时大幅降权
- **震荡市修正**：震荡期不持仓
- **硬约束**：最大持股数（默认 4 只）、总仓位上限（默认 0.95，即至少保留 5% 现金）

## 测试

```bash
python -m pytest tests/ -v
```

## 开发状态

| 模块 | 状态 |
| --- | --- |
| 数据加载 `data/` | ✅ 已完成 |
| 指标计算 `indicators/` | ✅ 已完成 |
| 仓位管理 `strategy/position.py` | ✅ 已完成 |
| 绘图 `analytics/plotter.py` | ✅ 已完成 |
| 回测引擎 `engine/` | 🚧 开发中 |
| 绩效评估 `analytics/metrics.py` | 🚧 开发中 |
| 策略主体 `strategy/my_strategy.py` | 🚧 开发中 |
| 项目入口 `main.py` | 🚧 开发中 |
| 单元测试 `tests/` | 🚧 开发中 |

## 免责声明

本项目仅用于学习与研究，不构成任何投资建议。股市有风险，入市需谨慎。
