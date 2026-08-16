# 量化交易回测系统（A 股）

基于 [baostock](http://baostock.com/) 行情数据的 A 股量化交易回测框架。项目以「数据层 → 指标层 → 策略层 → 执行引擎层 → 评估与绘图」的分层结构组织代码。

当前开发状态：**数据层、指标因子层、信号合成与状态机、回测撮合引擎均已实现并通过单元测试（90 项）**；机器学习寻优、绩效评估与归因、Level-2/宏观真实数据源接入、项目主入口等模块为规划中的占位实现。

## 功能特性

- **数据加载**：通过 baostock 获取沪深 A 股日线数据，支持前/后/不复权，自动剔除停牌日，本地缓存避免重复联网
- **数据访问抽象**：数据源注册表（`config/data_sources.py`）统一管理本地数据位置，支持环境变量 / YAML 配置 / 运行时注册三级覆盖，后续接入真实存储位置无需改动业务代码
- **指标体系**：MA、MACD、ADX、ATR、布林带等常用指标，全部仅使用历史数据（无未来函数）
- **市场状态判定**：基于趋势主轴斜率 + 残差波动率的 `is_trend` 判定，区分趋势日与震荡日
- **多因子打分**：趋势质量 TQ（0~3）+ 量能确认 VC（0~2），配合动态阈值（滚动分位数）
- **时间对齐管道**：`TimeAligner` 多源异构数据时钟对齐（宏观/外盘 T-1 全量对齐、龙虎榜 T+1 隔离），内置 `verify_no_lookahead` 防未来函数校验
- **因子体系**：资金主体分层（小/中/大/超大单净流）、微观结构因子（OFSS 盘口/CPS 筹码/PSS 价格结构）、宏观共振因子（MRS/GRS/IRS 与 Global_Mod/Chain_Mod），统一由 `FeatureEngine` 调度
- **信号合成与状态机**：Agent_MS/Final_MS/Capital_Purity 合成公式、`S_push / S_youzi_only / S_noise` 三态机、8 层开仓硬闸门与 6 层平仓闸门（`strategy/signals.py`）
- **仓位管理**：权重映射、波动率/MDM/震荡市修正、最大持股数与总仓位上限约束
- **回测撮合引擎**：事件驱动型分钟级撮合——T+1 双份额冻结、涨跌停盘中拦截、动态滑点（订单参与率冲击模型）、佣金/印花税/过户费、动态仓位 `Position_Size` 公式、单股与总杠杆上限风控，输出完整成交日志与净值曲线
- **可视化**：价格走势 + 均线 + 震荡区间色带标注，自动输出图片
- **规划中能力**：Optuna 超参寻优与 Walk-Forward 交叉验证、因子 IC/收益归因、实时决策日志、Level-2 快照与宏观数据的真实数据源接入等

## 项目结构

```
quantitative_trading/
├── main.py                     # 项目主入口（支持回测/寻优/实盘模拟模式切换）
├── run_optimization.py         # 机器学习超参数寻优入口脚本
├── demo_plot.py                # 演示脚本：绘制回测段价格走势与震荡区间
├── requirements.txt            # 依赖：baostock / pandas / numpy / scipy / matplotlib / pyarrow / pyyaml / optuna
├── config/                     # ── 全局配置 ──
│   ├── settings.py             # 全局基础配置（回测起止日期等）
│   ├── strategy_params.py      # 策略默认超参数（W_OFSS, TH_MS_BULL 等）
│   ├── industry_mapping.yaml   # 产业链映射：个股→中信行业→海外龙头→核心商品
│   ├── data_sources.py         # 数据源注册表：逻辑名→路径解析（环境变量/YAML/运行时/默认）
│   └── data_paths.yaml         # 本地数据存储位置配置（可选，当前为占位示例）
├── data/                       # ── 数据层 ──
│   ├── loader.py               # 基础数据加载（K线、指数；baostock + CSV 缓存 + 预热段）
│   ├── l2_loader.py            # Level-2 快照与逐笔数据加载器（规划中）
│   ├── macro_loader.py         # 全球宏观、隔夜外盘、商品与汇率加载器（规划中）
│   ├── aligner.py              # 多源异构数据时钟对齐管道（T-1/T+1 防未来函数，已完成）
│   ├── dataslice.py            # 标准数据切片 DataSlice：多数据帧容器 + 标准列常量
│   ├── storage.py              # 统一本地数据访问层（parquet/csv/yaml 读写）
│   └── mock_data/              # 历史数据缓存（可改用 parquet 格式提升性能）
├── indicators/                 # ── 特征工程与因子层 ──
│   ├── basic.py                # 基础技术指标统一入口 compute_all + TQ/VC 打分 + 动态阈值
│   ├── trend.py                # 趋势指标：MA / MACD / ADX
│   ├── shock.py                # 波动指标：ATR / 布林带 / 残差波动率
│   ├── agent_profiling.py      # 资金主体分层（小/中/大/超大单净流、北向、两融）
│   ├── microstructure.py       # 微观结构因子（OFSS 盘口、CPS 筹码、PSS 价格结构）
│   ├── environment.py          # 宏观与共振因子（MRS 大盘、GRS 全球风险、IRS 产业）
│   └── feature_engine.py       # 统一特征计算与归一化调度器 FeatureEngine
├── strategy/                   # ── 策略与状态机 ──
│   ├── base.py                 # 策略抽象基类
│   ├── signals.py              # 信号合成 + 多层闸门状态机（SignalSynthesizer / TradingStateMachine）
│   ├── position.py             # 仓位管理：权重计算与持仓约束
│   ├── sentiment_strategy.py   # Final v2.0 全生态情绪策略实现（规划中）
│   ├── state_machine.py        # 有限状态机外壳（核心逻辑已并入 signals.py）
│   └── gates.py                # 8/6 层开平仓闸门外壳（核心逻辑已并入 signals.py）
├── engine/                     # ── 执行与撮合层 ──
│   ├── backtest.py             # 事件驱动型分钟级回测引擎（TradeLog / EquityCurve）
│   ├── execution.py            # A股交易成本与动态滑点模型（T+1、参与率冲击、涨跌停拦截）
│   ├── portfolio.py            # 账户状态机：现金/持仓/T+1 可卖份额/成本价
│   └── risk_control.py         # 动态仓位 Position_Size 公式 + 单股/总杠杆上限风控
├── optimizer/                  # ── 机器学习优化层 ──
│   ├── search_space.py         # 超参数搜索空间定义（权重、阈值、窗口，规划中）
│   ├── bayesian_opt.py         # 基于 Optuna 的多目标/带约束贝叶斯寻优（规划中）
│   └── walk_forward.py         # 滚动样本外（OOS / Walk-Forward）交叉验证框架（规划中）
├── analytics/                  # ── 评估、归因与监控 ──
│   ├── metrics.py              # 夏普比率、卡玛比率、最大回撤、胜率、盈亏比计算（规划中）
│   ├── attribution.py          # 收益归因（赚宏观、产业还是微观情绪的钱，规划中）
│   ├── ic_analyzer.py          # 因子 IC / Rank IC / IR 分析（规划中）
│   ├── real_time_stream.py     # 结构化实时决策日志生成器（规划中）
│   ├── plotter.py              # 绘图模块：价格走势 / 震荡区间标注（已完成基础版）
│   └── pictures/               # 生成的图片输出目录
└── tests/
    ├── test_data_aligner.py    # 时间对齐、防未来函数与 DataSlice 组装
    ├── test_factors.py         # 微观结构与环境因子计算
    ├── test_features.py        # 主体分层与 FeatureEngine 端到端
    ├── test_state_machine.py   # 状态机转换与闸门过滤
    ├── test_signals.py         # 信号合成公式与状态机全流程
    └── test_backtest_engine.py # A 股撮合规则、T+1、成本滑点与风控
```

## 安装

需要 Python 3.9+（建议 3.11+）。

```bash
pip install -r requirements.txt
```

依赖清单：`baostock`、`pandas`、`numpy`、`scipy`、`matplotlib`、`pyarrow`（或 fastparquet，用于 parquet 高性能存储）、`PyYAML`（配置文件解析）、`optuna`（超参数优化）。

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

## 本地数据访问

在不明确具体存储位置的前提下，通过「逻辑数据源名」访问本地数据：

- `config/data_sources.py`：数据源注册表，路径解析优先级为 环境变量 `QTDATA_<KEY>_PATH` → `config/data_paths.yaml` → 运行时 `set_data_path()` → 项目默认路径
- `data/storage.py`：统一读写层，按「数据源 + 文件名」读写 parquet / csv / yaml

```python
from data import storage

df = storage.read_frame("l2", "order_20240101.parquet")     # 读
storage.write_frame(df, "parquet", "factors_2024.parquet")  # 写
cfg = storage.read_yaml("industry", "industry_mapping.yaml")
```

待数据存储位置确定后，只需在 `config/data_paths.yaml` 或环境变量中登记真实路径，业务代码无需改动。

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

规划方向：结合 MRS（大盘共振）与 Global_Mod（全球风险修正）的动态仓位。

## 信号合成与状态机

`strategy/signals.py` 提供 `SignalSynthesizer`（合成公式）与 `TradingStateMachine`（逐 Bar 状态机）：

- **合成公式**：`Agent_MS = W_OFSS*OFSS + W_CPS*CPS + W_INST*sign(Inst_Flow) + W_NORTH*North_Sync`（权重和强制 == 1），`Final_MS = (Agent_MS + Chain_Mod) * (1 + Global_Mod)`，`Capital_Purity` 刻画资金纯净度
- **状态机**：`S_push`（多证据进攻态，允许开仓/加仓）→ `S_youzi_only`（游资主导禁止开仓）→ `S_noise`（默认态）；已持仓时评估 6 层平仓闸门（状态反转 / 情绪分跌破 / 资金纯净度转负 / 入场 VWAP 止损 / 超时 / 大盘熔断）
- **8 层开仓硬闸门**：全球层 → 系统层（MA20>MA60 + ADR）→ MRS → 产业层 → Alpha（RS/行业情绪）→ 个股层 → 流动性合规 → 交易时间窗
- **防未来**：分钟级合成只用当前及历史 Bar；RS / Industry_MS 为日频因子，经 T-1 asof 对齐后进入分钟轴

## 回测撮合引擎

`engine/` 提供事件驱动型分钟级回测撮合，`Account` 状态机 + `BacktestEngine` 主循环 + `TradeLog` / `EquityCurve` 输出：

- **下一 Bar 成交**：`Bar t` 的信号在 `Bar t+1` 开盘价成交（1 Bar 执行延迟，严格防未来函数）
- **A 股规则**：T+1 双份额跟踪（当日买入锁定、次交易日解冻、SELL 遇锁定挂起次日强制卖出）；盘中触及涨停不可买入、触及跌停不可卖出；100 股整数倍取整
- **成本与滑点**：佣金双边（默认万二，最低 5 元）+ 印花税仅卖出（千 0.5）+ 过户费双边；动态滑点 = 固定 2bp + 参与率 × 50bp（封顶 60bp），参与率 = 订单金额 / Bar 成交额
- **动态仓位**：`Position_Size = Base_Position × MRS_Coefficient × (1 + Global_Mod) × Chain_Mod_Scale`，clip 到单股上限；成交前校验单股最大仓位与总账户杠杆上限
- **资金分配**：信号按时间序先到先得，现金不足拒绝（`insufficient_cash`）；拒绝单同样入日志（`shares=0` + `reason`）

```python
from engine.backtest import BacktestEngine
from engine.execution import ExecutionCost
from engine.portfolio import Account
from engine.risk_control import PositionSizer

ds, signals = ...            # DataSlice（对齐后的行情）+ 状态机产出的 Signal 列表
eng = BacktestEngine(
    Account(initial_cash=1e8, max_leverage=1.0, max_single_position=0.3),
    ExecutionCost(),         # 佣金 / 印花税 / 过户费 / 动态滑点
    PositionSizer(),         # Position_Size 公式 + 单股/杠杆上限
    ds, signals,
)
trade_log, equity_curve = eng.run()   # 完整成交日志 + 逐 Bar 净值曲线
```

## 测试

```bash
python -m pytest tests/ -v
```

当前 90 项单元测试全部通过（合成 mock 数据，不联网）：

| 测试文件 | 覆盖范围 |
| --- | --- |
| `tests/test_data_aligner.py` | 多源时间对齐、T-1/T+1 隔离、防未来函数校验、DataSlice 组装 |
| `tests/test_factors.py` / `tests/test_features.py` | 微观结构/环境/主体分层因子与 FeatureEngine 端到端 |
| `tests/test_state_machine.py` / `tests/test_signals.py` | 合成公式精确值、8/6 层闸门、状态机 BUY/ADD/SELL/HOLD 全流程 |
| `tests/test_backtest_engine.py` | 下一 Bar 成交、T+1 挂起卖出、涨跌停拦截、成本滑点、仓位/杠杆风控、成交日志与净值曲线 |

## 开发状态

| 模块 | 状态 |
| --- | --- |
| 数据加载 `data/loader.py` | ✅ 已完成 |
| 数据访问抽象 `data/storage.py` + `config/data_sources.py` | ✅ 已完成 |
| 时间对齐 `data/aligner.py` + 数据切片 `data/dataslice.py` | ✅ 已完成 |
| 指标计算 `indicators/trend.py` `shock.py` `basic.py` | ✅ 已完成 |
| 因子体系 `indicators/`（agent_profiling / microstructure / environment / feature_engine） | ✅ 已完成 |
| 仓位管理 `strategy/position.py` | ✅ 已完成 |
| 信号合成与状态机 `strategy/signals.py`（含 8/6 层闸门） | ✅ 已完成 |
| 回测撮合引擎 `engine/`（backtest / execution / portfolio / risk_control） | ✅ 已完成 |
| 绘图 `analytics/plotter.py` | ✅ 已完成（基础版） |
| 单元测试 `tests/` | ✅ 90 项通过 |
| 数据扩展 `data/l2_loader.py` `macro_loader.py` | 🚧 规划中 |
| 策略壳 `strategy/`（sentiment / state_machine / gates 外壳） | 🚧 规划中（核心逻辑已并入 signals.py） |
| 机器学习优化 `optimizer/` | 🚧 规划中 |
| 评估归因 `analytics/`（metrics / attribution / ic / stream） | 🚧 规划中 |
| 项目入口 `main.py` / `run_optimization.py` | 🚧 规划中 |

## 免责声明

本项目仅用于学习与研究，不构成任何投资建议。股市有风险，入市需谨慎。
