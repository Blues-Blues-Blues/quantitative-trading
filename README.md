# 量化交易回测系统（A 股）

基于**真实高频数据**（万得 Level-2 逐笔 + 日频 CSV）的 A 股量化交易回测框架。项目以「数据层 → 指标层 → 策略层 → 执行引擎层 → 评估与绘图」的分层结构组织代码。

当前开发状态：数据管道、连续评分与账户驱动的逐 Bar 回测、绩效归因和 Optuna 寻优均已实现；冒烟模式内置 mock 数据可跑通。真实模式现要求独立市场分钟数据及历史 ST 状态，补齐后需重新验证实盘数据结果。

评分模块已完成**纯函数化 / 向量化架构重构**：`ES`（`_compute_es`）、`PS`（`_compute_ps`）、`Fund_Stability`（`_compute_fund_stability`）、`XS`（`_compute_xs`）均为无状态纯算子，长表整列向量化、index 对齐、缺失值中性化兜底、负底数/零除几何防护；`_build_eval_table` 预计算 `fund_stability` 整列供状态机消费，游资独舞走硬掩码否决（ES=0）/一票否决（XS=-1）而非数学衰减。

## 功能特性

- **真实数据接入**：`data/real_loader.py` 将 `data/data1`（万得 Level-2 行情/逐笔成交/逐笔委托 parquet）与 `data/data2`（宏观/北向/两融/行业/龙虎榜 CSV）清洗、解析、对齐为标准 `DataSlice`，缺表自动降级，全部因子严格防未来函数
- **冒烟模式**：`main.py --data smoke` 内置 6 个交易日 mock 数据，秒级跑通「对齐 → 特征 → 连续评分 → 状态机 → 差额调仓回测」全链路回归
- **数据加载抽象**：数据源注册表（`config/data_sources.py`）统一管理本地数据位置，支持环境变量 / YAML 配置 / 运行时注册三级覆盖
- **指标体系**：MA、MACD、ADX、ATR、布林带等常用指标，全部仅使用历史数据（无未来函数）
- **市场状态判定**：基于趋势主轴斜率 + 残差波动率的 `is_trend` 判定，区分趋势日与震荡日
- **多因子打分**：趋势质量 TQ（0~3）+ 量能确认 VC（0~2），配合动态阈值（滚动分位数）
- **时间对齐管道**：`TimeAligner` 多源异构数据时钟对齐（宏观/外盘 T-1 全量对齐、龙虎榜 T+1 隔离），内置 `verify_no_lookahead` 防未来函数校验
- **因子体系**：资金主体分层（小/中/大/超大单净流）、微观结构因子（OFSS 盘口/CPS 筹码/PSS 价格结构）、宏观共振因子（MRS/GRS/IRS 与 Global_Mod/Chain_Mod），统一由 `FeatureEngine` 调度
- **连续评分信号**：全向量化纯函数算子——**ES**（入场分 = `sigmoid` 合成资金主体/纯净度/量价共振，游资独舞硬掩码=0）、**PS**（持仓分 = `ES×Time_Decay×Fund_Stability` 变比分片、缺失兜底）、**XS**（出局分 = 权重线性加权 + 高水位回撤、一票否决硬 -1 的纯函数）；**Fund_Stability** 在评估表预计算整列、缺列/NaN 安全降级 1.0；独立 A 股硬过滤层（ST 禁买、涨跌停禁买卖、时间窗 10:00~14:50、成交额门槛）前置否决，全部评分参数可寻优
- **目标权重 Target_Weight**：连续评分 → 分钟级目标持仓比例（base × ES/PS × 宏观/行业乘子），驱动撮合引擎差额调仓
- **回测撮合引擎**：事件驱动型分钟级撮合——Target_Weight 差额调仓 + 调仓死区（防过度换手）+ T+1 顺延挂起（当日买入锁定、跌停跳过）、动态滑点（订单参与率冲击模型）、佣金/印花税/过户费、单股与总杠杆上限风控，输出完整成交日志与净值曲线
- **超参数优化**：Optuna 贝叶斯寻优（`StrategyOptimizer` + `run_optimization.py`），Dirichlet 式权重归一化（和为 1）、TPE 原生多重硬约束、样本内年化 Sharpe 最大化 + 回撤/换手软惩罚、优化历程收敛图；搜索空间只含决策链实际读取的参数（旧二值化闸门死参数已清理），寻优结果可直接注入 `main.py` 实盘复现；真实数据寻优采用**严格三集隔离**（训练→验证→测试）与 JournalStorage 断点续跑，详见「超参数寻优」一节
- **Walk-Forward 交叉验证**：滚动/扩展训练段的滚动样本外（OOS）验证框架，每折独立寻优并在样本外回测，输出跨折 OOS 评估报告，避免前视偏差与过度拟合
- **绩效指标**：年化收益率、年化夏普、卡玛（Calmar）、Sortino、最大回撤、平均持仓周期、胜率/盈亏比、日收益偏度/峰度（`analytics/metrics.py` + `PerformanceAnalyzer`）
- **实时决策日志流**：`StreamLogger` 每 Bar × 每标的输出标准 JSON（Final_MS / Global_Mod / Chain_Mod / Capital_Purity / Action / State），JSONL 落盘 + 生成器双形态
- **已实现盈亏分摊**：`AttributionEngine` 通过决策/成交 ID 追溯买入批次，按三项入场分数绝对值分配平均成本盈亏，保留未归因原因与覆盖率

- **动态组合暴露**：`ExposureAnalyzer` 在收盘按实际持仓市值/总权益加权三项有符号分数；保留现金影响和逐因子缺失覆盖率
- **因子预测能力检验**：因子 IC / Rank IC / IR 时序分析（横截面 Rank IC 与单标的时序相关双模式，前瞻窗口可配）+ IC 时间桶热力图
- **多子图 Dashboard**：净值回撤图、动态仓位图、归因柱状图、参数敏感度热力图（`plot_report`）
- **20 日人工复盘清单**：自动导出每笔交易触发前后 N 个交易日的日线指标切片 + 逐笔资金流清单（Excel 三 sheet / CSV）
- **可视化**：价格走势 + 均线 + 震荡区间色带标注，自动输出图片
- **规划中能力**：Brinson 基准归因、更多真实数据源（深市/北交所逐笔、期权持仓等）

## 项目结构

```
quantitative_trading/
├── main.py                     # 项目主入口（--data smoke|real 模式切换）
├── run_optimization.py         # 超参数寻优入口（--data smoke|real，严格三集 + Journal 断点续跑）
├── demo_plot.py                # 演示脚本：绘制回测段价格走势与震荡区间
├── requirements.txt            # 依赖：pandas / numpy / scipy / matplotlib / pyarrow / pyyaml / optuna / openpyxl / pytest
├── .gitignore                  # 排除数据目录（data/data1、data/data2）、缓存与临时脚本
├── config/                     # ── 全局配置 ──
│   ├── settings.py             # 全局基础配置（回测起止日期等）
│   ├── strategy_params.py      # 策略默认超参数（评分权重/阈值、DEADZONE_TH 等）
│   ├── industry_mapping.yaml   # 产业链映射：个股→中信行业→海外龙头→核心商品
│   ├── data_sources.py         # 数据源注册表：逻辑名→路径解析（环境变量/YAML/运行时/默认）
│   └── data_paths.yaml         # 本地数据存储位置配置（可选覆盖模板）
├── data/                       # ── 数据层 ──
│   ├── data1/                  # 万得 Level-2 原始 parquet（本地大文件，不入库）
│   │   └── data/{SH,SZ,BJ}/{代码}.{市场}/{日期}.{类型}.parquet   # 行情/逐笔成交/逐笔委托
│   ├── data2/                  # 日频 CSV（宏观/北向/两融/行业/龙虎榜/基础信息，本地不入库）
│   ├── real_loader.py          # 真实数据适配器：data1+data2 → 标准 DataSlice（含降级与近似）
│   ├── loader.py               # 基础数据加载（K线、指数；baostock + CSV 缓存 + 预热段）
│   ├── l2_loader.py            # Level-2 行情/快照/逐笔成交/逐笔委托解析（价格/单位转换、脏数据过滤）
│   ├── macro_loader.py         # 全球宏观、隔夜外盘、商品与汇率加载
│   ├── aligner.py              # 多源异构数据时钟对齐管道（T-1/T+1 防未来函数）
│   ├── dataslice.py            # 标准数据切片 DataSlice：多数据帧容器 + 标准列常量
│   ├── storage.py              # 统一本地数据访问层（parquet/csv/yaml 读写）
│   └── mock_data/              # 冒烟测试 mock 数据（本地，不入库）
├── indicators/                 # ── 特征工程与因子层 ──
│   ├── basic.py                # 基础技术指标统一入口 compute_all + TQ/VC 打分 + 动态阈值
│   ├── trend.py                # 趋势指标：MA / MACD / ADX
│   ├── shock.py                # 波动指标：ATR / 布林带 / 残差波动率
│   ├── agent_profiling.py      # 资金主体分层（小/中/大/超大单净流、北向、两融）
│   ├── microstructure.py       # 微观结构因子（OFSS 盘口、CPS 筹码、PSS 价格结构）
│   ├── environment.py          # 宏观与共振因子（MRS 大盘、GRS 全球风险、IRS 产业）
│   └── feature_engine.py       # 统一特征计算与归一化调度器 FeatureEngine
├── strategy/                   # ── 策略与状态机 ──
│   ├── signals.py              # 连续评分信号（ES/PS/XS）+ A股硬过滤 + 状态机（SignalSynthesizer / TradingStateMachine）
│   └── gates.py                # 开平仓闸门外壳（已被 signals.py 硬过滤层取代，保留兼容）
├── engine/                     # ── 执行与撮合层 ──
│   ├── backtest.py             # 事件驱动型分钟级回测引擎（TradeLog / EquityCurve）
│   ├── execution.py            # A股交易成本与动态滑点模型（T+1、参与率冲击、涨跌停拦截）
│   ├── portfolio.py            # 账户状态机：现金/持仓/T+1 可卖份额/成本价
│   └── risk_control.py         # PositionSizer：单股/总杠杆上限风控与目标权重裁剪
├── optimizer/                  # ── 机器学习优化层 ──
│   ├── search_space.py         # 超参数搜索空间（权重 Dirichlet 归一化、阈值、窗口）
│   ├── bayesian_opt.py         # 基于 Optuna 的带多重硬约束贝叶斯寻优 StrategyOptimizer
│   └── walk_forward.py         # 滚动样本外（OOS / Walk-Forward）交叉验证框架
├── analytics/                  # ── 评估、归因与监控 ──
│   ├── metrics.py              # 绩效指标：年化夏普/Calmar/Sortino、回撤、胜率/盈亏比、
│   │                           #   平均持仓周期、日收益偏度/峰度、FIFO 交易配对
│   ├── performance.py          # PerformanceAnalyzer：指标总表、四子图 Dashboard、
│   │                           #   参数敏感度热力图、20 日人工复盘清单导出（Excel/CSV）
│   ├── attribution.py          # 入场分数绝对值盈亏分摊 + IC 分析
│   ├── exposure.py             # 实际持仓收盘策略分数加权暴露
│   ├── reporting.py            # 报告生命周期、按日 Parquet、质量诊断与图表
│   ├── ic_analyzer.py          # 因子 IC / Rank IC / IR 双模式分析 + Q1~Q5 分位分层净值
│   ├── real_time_stream.py     # StreamLogger：实时/仿真决策日志流（标准 JSONL + 生成器）
│   ├── plotter.py              # 绘图模块：价格走势 / 震荡区间标注
│   └── pictures/               # 生成的图片输出目录（含 optimizer_history.png / dashboard.png）
└── tests/
    ├── test_data_aligner.py    # 时间对齐、防未来函数与 DataSlice 组装
    ├── test_factors.py         # 微观结构与环境因子计算
    ├── test_features.py        # 主体分层与 FeatureEngine 端到端
    ├── test_signals.py         # 信号合成公式与状态机全流程
    ├── test_backtest_engine.py # A 股撮合规则、T+1、成本滑点与风控
    ├── test_optimizer.py       # 贝叶斯寻优、硬约束、Walk-Forward 与收敛图
    └── test_analytics.py       # 绩效/归因/IC/实时流/复盘清单导出
```

## 安装

需要 Python 3.9+（建议 3.11+）。

```bash
pip install -r requirements.txt
```

依赖清单：`pandas`、`numpy`、`scipy`、`matplotlib`、`pyarrow`（parquet 高性能存储）、`PyYAML`（配置文件解析）、`optuna`（超参数优化）、`openpyxl`（Excel 导出）、`pytest`（单元测试）。

## 快速开始

**1. 冒烟模式（秒级回归，内置 mock 数据）**

```bash
python main.py --data smoke
```

内置 6 个交易日 mock 数据，跑通「对齐 → 特征 → 信号 → 状态机 → 回测 → 绩效 → 绘图 → 状态检查」全链路，输出 5 项 PASS 状态检查与 2 张走势图。

**2. 真实数据回测（data1 + data2）**

```bash
python main.py --data real
```

真实数据模式参数（区间、股票子集、评分/过滤阈值）见 `main.py` 顶部 `REAL_START / REAL_END / REAL_SYMBOLS / REAL_PARAMS`：

```python
REAL_START = "2024-01-02"          # 回测起始
REAL_END   = "2024-12-31"          # 回测结束
REAL_SYMBOLS = ["603019"]          # 股票子集（603019=中科曙光）；空列表 = 全部 20 只（全量较慢）
```

- 空 `REAL_SYMBOLS` 时自动发现 `data/data1` 全部标的（当前 20 只沪市）
- **请使用已安装依赖的解释器运行**（如 `.venv\Scripts\python.exe`）。真实数据读取 parquet 依赖 `pyarrow`（或 `fastparquet`），缺失时 `RealDataLoader` 会在入口直接报错提示安装，不再静默降级为空表
- 特征表按数据清单指纹、因子参数、schema 与对齐版本缓存；交易信号依赖实际成交和账户状态，因此逐 Bar 生成，不复用旧的 `signals_*.pkl`。
- 输出到 `analytics/pictures/`（冒烟模式文件带 `smoke_` 前缀），状态检查 5 项（含防未来函数断言）

## 真实数据接入

### 数据布局

```
data/data1/data/{市场}/{代码}.{市场}/{日期}.{类型}.parquet
    - 类型：行情（10 档快照，66 列）/ 逐笔成交 / 逐笔委托
    - 时间：自然日 = YYYYMMDD int；时间 = HHMMSSmmm int（前导 0 省略，如 92500780 → 09:25:00.780）
    - 价格：整数，÷10000 得元（63800 → 6.38）；成交量：股；成交额：元
data/data2/*.csv
    - global_macro_2023_2026.csv       全球宏观（SPX/NDX/DOW/HSI/NKY/BRENT/GOLD/COPPER/US10Y）
    - hsgt_north_holdings.csv          北向个股持股（2017 起）
    - margin_daily_*.csv               两融余额（全市场）
    - industry_sentiment_history.csv   行业指数（close/ma20/rsi14，90 行业）
    - hsgt_north_daily_flow.csv        北向大盘净流 + 沪深300 日频点位
    - lhb_summary_*.csv / lhb_seats_*.csv  龙虎榜净额（分类）+ 席位
    - stock_basic_info.csv             股票基础信息（代码/名称/ST）
    - index_min.csv 或 .parquet          独立指数分钟数据，列 ts + index_code/open/high/low/close/volume/vwap/ma20/ma60
    - breadth_min.csv 或 .parquet        独立全市场分钟广度，列 ts + advancers/decliners/adr（可含 north_net）
    - stock_history.csv                  按日期生效的个股状态，列 symbol/trade_date/is_st/float_shares
```

### 用法

```python
from data.real_loader import RealDataLoader

loader = RealDataLoader()
ds = loader.load_slice(["603019"], "2024-01-02", "2024-12-31")  # 已内置对齐
ds.validate()
```

### 防未来函数设计

- 分钟级表（kline / l2_snapshot / tick_trades）直接用当日实时数据（当前 Bar 已收盘）
- 日频表（macro / north_margin / industry）由 `TimeAligner` 做 T-1 全量对齐
- 龙虎榜由 `TimeAligner` 标注 T+1 可用日（`avail_date`），T+1 前不可见
- 指数和广度只从独立市场文件加载；缺失时对应市场因子保持缺失，不能把交易股票子集冒充全市场。
- ST 和流通股本只从 `stock_history.csv` 的历史生效记录读取；缺失 ST 时开仓过滤保守阻断。缺昨收的首日涨跌停价保持缺失。
- 运行日志列出股票池规模、独立市场数据是否存在、历史 ST 和流通市值覆盖率。

### 已知数据缺口（当前降级处理，不影响运行）

| 缺口 | 处理 |
| --- | --- |
| 600237/600379/603380 无北向持股 | `north_sync` 恒 NaN，对应信号分量保守为中性（不参与评分） |
| 宏观缺 DXY 列 | 源文件无 DXY 列 → `dxy` 恒 NaN（加载器已支持读取，补齐数据列即生效） |
| 9 只股票北向 2023 年 1~3 月起才有数据 | 前段 `north_sync` 为空（T-1 不填充） |
| 股票融券余额全 NaN（仅 ETF 有值） | `margin_pressure` 降级为融资余额变化率 |
| 涨跌停价口径 | 以 **T-1 昨收×幅度** 计算；缺昨收或历史 ST 时保持缺失，不使用本日收盘代替 |
| 行业映射为按名称近似 | 内置 `DEFAULT_SYMBOL_TO_INDUSTRY`，可替换为 `config/industry_mapping.yaml` |

## 指标与市场状态

- `indicators/basic.py` 的 `compute_all()` 是统一指标入口，一次性计算全部指标列
- **趋势判定 `is_trend`**：趋势主轴（默认 MA20）5 日斜率 > 阈值，且 20 日残差波动率低于动态阈值 → 趋势日；否则视为震荡日
- **打分因子**：
  - `tq`（趋势质量 0~3）：ADX 强度 + 均线多头排列 + 价格站上 MA20
  - `vc`（量能确认 0~2）：量能均线放大 + 换手率分位数 ≥ 0.4
  - `mdm_cond`（动量衰减）：上升趋势中 MACD 柱动能减弱，作为卖出预警
- **动态阈值**：基于 252 日滚动分位数（ADX 25% 分位、ATR 70% 分位等）

## 目标权重 Target_Weight 与仓位约束

信号层（`strategy/signals.py`）输出分钟级目标持仓比例，撮合引擎按差额调仓：

- **开仓**：未持仓 + A 股硬过滤全过 + ES ≥ th_es_entry → 目标 = `base_weight × ES × clip(1+Global_Mod) × clip(1+Chain_Mod)`，clip 到单股上限 `max_single_position`
- **持仓 XS 四分链**：正常持仓（XS ≥ th_xs_reduce_high=0.2，目标=`base × PS × 乘子`）→ 容错阶梯减仓（-0.3 < XS < 0.2，目标=`simulated_weight × 0.8`）→ 常规清仓（-0.6 < XS ≤ -0.3，目标=0）→ 极速清仓（XS ≤ -0.6 或一票否决，目标=0）
- **实际权重**：回测逐 Bar 从已成交股数、成本和账户权益同步策略持仓；拒单不会建仓，T+1 顺延卖单不会清除持仓。
- **次日低开反包（Reversal / Counter-Attack）**：持仓跨入次日且深度低开（≤ th_reversal_gap=-1.5%）+ 盘口承接（OFSS > 0.2）+ 大资金逆势净流入（purity>0 且 big_flow>0）→ 豁免 XS 清仓/阶梯减仓，并在窗口内（受开仓 time 闸门共同约束）承接加仓 `target = min(simulated + base×ES×reversal_add_mult, max_single_position)`；大盘熔断（沪深300 跌破 VWAP-1.5%）时保护立即失效强制清仓
- **硬约束**：`engine/risk_control.py` 的 `PositionSizer` 校验单股最大仓位与总账户杠杆上限，目标权重经 `max_single_position` 裁剪；`config/strategy_params.py` 提供全部默认参数（DEADZONE_TH=0.05 等）

## 信号合成与状态机

`strategy/signals.py` 提供 `SignalSynthesizer`（连续评分合成）与 `TradingStateMachine`（逐 Bar 状态机）：

- **ES 入场分**（纯向量化 `_compute_es`，长表整列计算）：`es = sigmoid(k · (w_ms·final_ms + w_purity·capital_purity + w_mrs·mrs_c))`；`final_ms`/`capital_purity` 已定 `[-1,1]` 直接使用（缺失→0，不二次 clip），大盘项 `mrs_c = clip(mrs, ±mrs_clip)/mrs_clip`；**游资独舞硬掩码**：`s_youzi_only=True` 时 `es = 0`；参数可寻优（`w_es_*` / `es_sigmoid_k` / `th_es_entry`），纯函数内 index 对齐、缺失中性化
- **PS 持仓分**（纯向量化 `_compute_ps`）：`PS = ES × Time_Decay × Fund_Stability`，三乘数天然有界故移除外层 `clip`；`es` 缺失→0、`time_decay`/`fund_stability` 缺失→1.0 不做折减；`Time_Decay` 前 `win_decay_grace` 恒 1.0，此后按浮盈亏非对称衰减（浮盈 factor=0.975、浮亏=0.90，`max(0.01, factor)` 防负底数，`clip(factor^(eff_bars/10), 0.1, 1.0)`）
- **Fund_Stability**（纯向量化预计算整列）：`_compute_fund_stability(cancel_ratio, obi, big_flow)` 在 `_build_eval_table` 注入 `fund_stability` 列——撤单率超阈或盘口变薄（`|OBI|` 与 `|big_flow|` 同时趋零）→ `penalty`，缺列/NaN（`fillna` 后不触发）安全降级 1.0；状态机仅消费该列，不重复计算
- **XS 出局分**（纯函数 `_compute_xs`）：权重线性加权 + 高水位回撤 `max(0, (hwm-close)/hwm)`（`hwm≤1e-6 → 0` 防除零）clip 到 [-1,1]；**情绪极值均值反转门控**（`final_ms>0.8` 且当前浮盈率 `>5%` 时情绪分反转为惩罚，支持高处止盈离场）；`veto_flag=True` 一票否决无视权重硬返回 -1.0
- **A 股硬过滤层**（否决前置）：ST 禁买、涨停禁买/跌停禁卖、交易时间窗 10:00~14:50（10:00 整之前禁买，防开盘冲高骗局）、成交额门槛；未持仓时任一不过即不开仓（`hard_filters` 与状态机 BUY 口径一致）
- **一票否决**（纯向量化 `_compute_veto` 预计算整列 `veto_flag`）：游资溃逃（大资金净流出为负且散户追高超阈值）或大盘跳水（沪深300 跌破 VWAP×(1-1.5%)）→ XS=-1，强制出局；`veto_flag` 在评估表 `_build_eval_table` 预计算、缺列/NaN 保守为 False（任一腿缺失不误触发），状态机直接消费
- **状态机动作**：BUY / ADD / SELL / DECAY_REDUCE / HOLD，输出含 `target_weight` 在内的全量评分快照（metrics）
- **防未来**：Bar t 决策 → Bar t+1 开盘价成交；日频因子经 T-1 asof 对齐后进入分钟轴

## 回测撮合引擎

`engine/` 提供事件驱动型分钟级回测撮合，`Account` 状态机 + `BacktestEngine` 主循环 + `TradeLog` / `EquityCurve` 输出：

- **下一 Bar 成交**：`Bar t` 的信号在 `Bar t+1` 开盘价成交（1 Bar 执行延迟，严格防未来函数）
- **差额调仓**：引擎读取信号 `target_weight`，按当前持仓权重与目标的差额下单（BUY 建仓 / ADD 加仓 / DECAY_REDUCE 减仓 / SELL 清仓）；调仓死区（默认 5%）——已持仓且 |Δ权重| 小于死区跳过（建仓/清仓豁免），避免过度换手
- **减仓防转增仓**：DECAY_REDUCE 空仓直接返回、目标权重裁剪不超过当前权重（信号层 `target_weight` 基于模拟权重，引擎空仓后仍 >0，放行会被差额正化误判成「买入回补」——已修复该 T+1 违规根因）
- **T+1 顺延**：当日买入份额锁定不可卖，SELL/减仓遇锁定挂入 `pending_targets` 逐 Bar 顺延，按当前价换算后重试（跌停日跳过但挂起不丢失）；**加仓成交后自动撤销该标的残留顺延目标**；盘中涨停不可买入、跌停不可卖出；100 股整数倍取整
- **成本与滑点**：佣金双边（默认万二，最低 5 元）+ 印花税仅卖出（千 0.5）+ 过户费双边；动态滑点 = 固定 2bp + 参与率 × 50bp（封顶 60bp）；买卖两侧口径统一为「实际成交股数 × 滑点前开盘价」作为参与率与费用基准
- **风控**：`PositionSizer` 校验单股最大仓位与总账户杠杆上限，目标权重经 `max_single_position` 裁剪（拒绝单同样入日志：`shares=0` + `reason`）

```python
from engine.backtest import BacktestEngine
from engine.execution import ExecutionCost
from engine.portfolio import Account
from engine.risk_control import PositionSizer

ds, signals = ...            # DataSlice（对齐后的行情）+ 状态机产出的 Signal 列表
eng = BacktestEngine(
    Account(initial_cash=1e6, max_leverage=1.0, max_single_position=0.3),
    ExecutionCost(),         # 佣金 / 印花税 / 过户费 / 动态滑点
    PositionSizer(),         # 单股/杠杆上限风控与目标权重裁剪
    ds, signals,
)
trade_log, equity_curve = eng.run()   # 完整成交日志 + 逐 Bar 净值曲线
```

## 超参数寻优（run_optimization.py）

`run_optimization.py` 是独立的寻优入口（`--data smoke|real`），复用 `StrategyOptimizer` 做 TPE 贝叶斯寻优：

```bash
python run_optimization.py --data smoke            # 冒烟版：复用 Mock 数据，3 trial 快速回归链路
python run_optimization.py --data real             # 真实数据严格三集寻优（默认 20 trial）
python run_optimization.py --data real --trials 30 --seeds 42,7 --stability-trials 8
```

- **冒烟版**（`--data smoke`）：2~3 次 trial 验证「参数注入 → 权重归一化 → 约束检查 → 收敛图」全链路；Mock 数据仅 6 天，笔数类硬约束物理上无法满足，`best()` 回退按目标值最优试兜底属正常表现
- **严格三集隔离**（真实模式，防前视/防过拟合）：

  | 集合 | 区间 | 用途 |
  | --- | --- | --- |
  | 训练 | 2023-01-03 ~ 2024-06-28 | TPE 采样，目标 = 年化 Sharpe − 回撤/换手软惩罚；hard 约束在内评估 |
  | 验证 | 2024-07-01 ~ 2024-09-30 | 训练段 top-k 候选复评，据验证段 Sharpe 选 best |
  | 测试 | 2024-10-08 ~ 2024-12-31 | 最终评估**一次**，不参与任何选择 |

- **硬约束**（真实模式 `REAL_CONSTRAINT_KWARGS`，训练段度量）：最大回撤 < 35%、胜率 > 30%、盈亏比 > 0.8、有效交易 ≥ 30 笔、年化换手 ≤ 15（宽于 `StrategyOptimizer` 默认 15%/55%/1.5，适配当前现实信号；TPE 原生支持，`constraints_func` 喂给采样器）
- **软惩罚**（合成目标内）：`Score = Sharpe − 1.0×max(0, 回撤−0.25) − 0.03×max(0, 年化换手−6.0)`，给 TPE 平滑引导、不替代硬约束
- **断点续跑**：`JournalStorage` 使用 `data/feature_cache/opt_study.v3.s{seed}.journal`；study 保存配置、数据和代码签名，不一致时拒绝混用旧 trial。
- **多 seed 稳定性**：`--seeds` 首个 seed 执行完整三集，其余 seed 独立寻优并输出 best 参数对比（区间跨度大 = 该维度对随机种子敏感，选参时谨慎）
- **完整三集流程**：训练段寻优 → `top_candidates` 取可行 top-5 → 验证段复评（验证段同时满足参数范围和绩效硬约束者中 Sharpe 最高胜出；无可行解时明确报错）→ 测试段终评一次。
- **窗口参数固定**：真实寻优搜索空间固定 `win_inst=(1,1), win_chip_old=(1,1)`——特征缓存签名含窗口，放开采样会让每种窗口组合触发训练段特征全量重算（约 10 分钟/次），待信号层参数收敛后再做窗口专项寻优
- **参数可行性**：TPE、训练候选和验证选择共用五项绩效指标与四个权重范围的违反量。冒烟及 Walk-Forward 允许显式标记的探索候选。

## 已实现盈亏分摊与收盘暴露报告

常规 `main.py --data smoke` 与 `--data real` 默认启用报告。每次运行写入独立目录：
`analytics/reports/<mode>_<start>_<end>_<UTC时间_随机短串>/`，不覆盖历史运行。

```powershell
& '.\.venv\Scripts\python.exe' main.py --data smoke
& '.\.venv\Scripts\python.exe' main.py --data smoke --report-dir 'analytics/reports'
& '.\.venv\Scripts\python.exe' main.py --data smoke --no-analysis-report
```

**盈亏口径**：买入费用计入平均成本，卖出净额扣除平均成本后得到已实现盈亏 P。
FIFO 仅确定卖出消耗哪些来源批次，各批次先获得 `P × matched_shares / sell_shares`，
再按 `abs(entry_factor) / sum(abs(三项因子))` 分配。三因子固定为
`global_mod / chain_mod / agent_ms`，原始符号保留。任一因子缺失/非法，或全零，
整批次进入 `other`；不对剩余因子重新归一化，不用退出因子，不计算未平仓浮盈。
每笔 SELL 仍算一笔交易，`summary.n_trades` 计主导分类的卖出笔数。
`summary.weight` 是因子 PnL / 总 PnL，总 PnL 为零时留空，不是风险权重。
例如任务书 C3 的两笔 SELL 总盈亏 475 元，三个因子分摊为 220.1 / 154.1 / 100.8 元。

**暴露口径**：每个 Bar 实际成交和收盘盯市后，使用同一 Bar 的已知因子值：
`exposure[f,t] = Σ market_value[i,t] / total_equity[t] × factor_value[f,i,t]`。
保留有符号原值和现金影响，不做标准化、截尾或重新归一化。
因子独立计算市值覆盖率；有缺失时 `exposure` 留空，`observed_exposure` 保存已知贡献之和，
缺失股票的贡献也留空。空仓时暴露为 0、coverage=1、status=flat。
这是策略分数加权暴露，不是统计 beta；盈亏分摊也不代表因果收益贡献、因子收益回归或 Brinson。

**追溯与校验**：全部 Signal（含 HOLD）在引擎入队时复制并分配可复现的 signal_id；
正股数成交有独立 fill_id，拒单没有 fill_id，decision_ts 保留原决策时间。
同时间戳按 event_seq 保留实际执行顺序。外部 ID 保留且必须唯一，顺延成交保留同一来源。
旧日志无来源 ID 时仍可计算平均成本盈亏，但其分摊归入 `unmatched_signal`，不猜测时间关联。
重复 ID、来源矛盾、超卖或负股数直接报错。

| 产物 | 内容 |
| --- | --- |
| `manifest.json` | schema_version=1、模式/区间/股票、参数、初始资金、行业映射摘要、数据来源、代码内容哈希、方法和运行状态 |
| `attribution_trades.csv` | 每 SELL 一行的已实现盈亏和分摊 |
| `attribution_batches.csv` | 来源批次、决策/成交 ID、原始因子、分摊权重及金额 |
| `attribution_summary.csv` | 三因子及 other 的 pnl、weight、n_trades |
| `unattributed_batches.csv` | 未归因批次及固定原因分类 |
| `quality.json` | 买入成交关联率、已卖出股数覆盖率、绝对盈亏覆盖率、other 原因统计、未平仓股数、守恒误差；另含逐因子暴露均值/绝对均值/峰值及缺失 Bar 数 |
| `positions/date=YYYY-MM-DD.parquet` | 当日实际收盘持仓 |
| `exposure_contributions/date=YYYY-MM-DD.parquet` | 当日逐股票、逐因子贡献 |
| `exposure_timeseries/date=YYYY-MM-DD.parquet` | 当日全部 Bar 的暴露、已知贡献、覆盖率和投入权益比例 |
| `attribution.png` / `exposure.png` / `dashboard.png` | 盈亏分摊、日末暴露/覆盖率、绩效面板 |

金额守恒误差上限 1e-6 元，权重校验容差 1e-12；CSV 不提前四舍五入，JSON 非有限值写 null。
覆盖率采用股数和 `Σabs(批次PnL)`，不以可能相互抵消的净盈亏作分母。
无卖出状态为 `no_closed_trades`，图表说明无已实现盈亏；覆盖不全为 `partial` 并告警；
结构错误或守恒失败为 `failed`。未平仓不会为报告强制平仓。
数据来源指纹不可得时明确标记 `unavailable`；文件统计清单缓存键不宣称是已验证的数据内容哈希。

writer 仅缓冲一个交易日，跨日及 finalize 时落盘；图表使用日末摘要，分钟序列保留在分日文件。
默认目录已忽略入库，自定义目录不自动修改 Git 配置。
`StrategyOptimizer.backtest()`、Optuna trial 和 Walk-Forward 默认没有快照回调或详细报告。

本任务的短验证命令（不执行真实回测或寻优）：

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_analytics.py tests/test_backtest_engine.py tests/test_signals.py tests/test_exposure.py tests/test_analysis_reporting.py -q
& '.\.venv\Scripts\python.exe' main.py --data smoke
```

## 离线回测可视化（任务书 1.1）

安装 `requirements.txt` 中的依赖后，正常主回测在原分析运行目录内额外生成
`analytics/reports/<本次运行目录>/backtest.html`，终端打印绝对路径与归因、可视化两个状态。
本次验证 Plotly **6.9.0**；HTML 内嵌一份 Plotly.js 与日频数据，无需联网、服务或相邻文件。
请手动用桌面 Edge / Chrome 打开，也可只复制该 HTML 到中文或带空格的目录。
报告生成不会自动开启浏览器。

日常使用：在 PowerShell 中切到项目根目录，用内置的 6 个交易日、2 只股票先试运行：

```powershell
Set-Location 'D:\quant trade\quantitative trading'
.\.venv\Scripts\python.exe main.py --data smoke
```

结束后复制终端 `HTML 报告：` 后的绝对路径到 Edge 地址栏，或在资源管理器中双击
对应的 `backtest.html`。只查看已经生成的报告不需要 Python，也不需要重跑回测。
代码更新不会改写历史 HTML；请打开新生成的报告。

使用本地真实数据时，先在 `main.py` 调整 `REAL_START`、`REAL_END`、`REAL_SYMBOLS`
（空列表代表全部已发现股票），并确认 `REAL_PARAMS` 是希望使用的固定策略参数，再手动运行：

```powershell
.\.venv\Scripts\python.exe main.py --data real
```

该命令执行真实数据回测，不启动参数寻优；运行时间取决于日期范围、股票数和特征缓存。
当前日期和股票配置在 Python 文件内，不是 `--start` / `--symbols` 命令行参数；
`--ic-start` / `--ic-end` 只用于 IC 分析，不控制这里的回测范围。

`--report-dir` 仍可自定义根目录：绝对路径直接使用，相对路径始终相对项目根目录，
与启动时的工作目录无关。每次沿用 writer 的唯一运行目录，不覆盖历史报告。
`--no-analysis-report` 同时关闭 HTML、归因与暴露 writer；旧 PNG 绘图行为保持不变。
优化器、Walk-Forward、普通引擎调用及模块导入不自动导出 HTML，也不加载 Plotly。
原有 `finalize(log, signals, curve)` 返回 quality 字典并保留旧产物，HTML 状态为 `skipped`；
主入口传 `kline=ds.kline`，不会走此兼容降级路径。

页面顶部包含完整区间的资金、收益、真实成交笔数、已实现盈亏和最大回撤摘要，
下方为净值、当日最深回撤、股票与策略收益对比，以及单列展开的所有股票。
每股提供日 K / 收盘线切换、MA5/20/60、账户成本、四类实际成交、成交量、股数和仓位。
图例可隐藏均线及成本；点击同日成交组查看当天全部实际成交，表格支持股票、动作、日期筛选，
每页 50 条；点击明细用真实成交价定位。筛选不会重算账户摘要，翻页和切图保留选中项。

日期缩放默认联动。关闭后，各股票和账户图独立，同一股票的三个图仍同步；
重新开启采用最近操作的范围。顶部日期输入和“恢复全区间”始终作用于全部图。
框选仅缩放日期轴，拖动可在 Plotly 工具栏选择；表格日期筛选与图形缩放独立。

数据口径：时间统一为上海本地时间，行情先裁至实际首末账户 Bar 再按可用 Bar 聚合。
首尾可能为不完整交易日，不额外下载预热行情或调整复权，未知复权明确标注。
缺行情日保留 null 断线；MA 按有效日收盘滚动计算，缺少共同起点 Bar 的股票不参与收益比较。
股票收益使用准确首 Bar 的 open，账户收益使用初始资金；缩放不改变基准。
成本线按账户平均成本重建，不含手续费；卖出盈亏复用现有 metrics 的含费平均成本算法。
回撤使用全 Bar 累计峰值后取每日最深值，避免遗漏日内回撤。
所有实际成交、逐股日末股数、市值、权益和盈亏 ID 均严格对账；缺快照不会补成空仓。

`manifest.json` 中的 `visualization` 独立记录 `complete / partial / skipped / failed`。
缺行情或共同基准为可展示的 `partial`；重复 ID、超卖、成本股数或市值不一致会失败并抛错。
失败保留已有分析产物，HTML 通过临时文件原子写入；正常无成交或未清仓不视为错误。
报告不改变策略、撮合、T+1、费用或盈亏计算规则。

指定短回归（不运行 `tests/test_optimizer.py`、参数寻优或真实两年回测）：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_visualization_data.py tests/test_html_report.py tests/test_analysis_reporting.py tests/test_exposure.py tests/test_analytics.py tests/test_backtest_engine.py -q
```

可重复生成 4 股 / 20 股各 500 日的合成样例，并在无界面的系统 Edge 中执行真实离线交互验收：

```powershell
# 仅浏览器验收需要此可选工具；日常导出无需安装
.\.venv\Scripts\python.exe -m pip install playwright
.\.venv\Scripts\python.exe tests/visualization_acceptance.py
```

脚本不下载行情、不运行引擎或寻优，输出独立运行目录的 HTML、截图与 `acceptance.json`，
记录生成耗时、文件大小、浏览器版本、首次就绪耗时及错误/网络请求。
若环境禁止启动浏览器，脚本明确失败，不将写文件成功视为交互通过。
手工复核可断网打开单文件，依次切换价格模式和图例、缩放并切换联动、输入日期/恢复、
点击同日成交及分页明细；Chrome 未在本次单独实测。

2026-09-27 验收：Windows 11、Python 3.14.3、Plotly 6.9.0、Edge 154.0.4258.37。
复查后指定六文件 **142 passed**；最终 JS 再经浏览器验收。
本轮另修复了图例隐藏“选中成交”后再点明细不能恢复标记的问题，以及最终 manifest
写入失败前提前打印 HTML 成功路径的问题，并补充对应回归断言。
执行六文件时额外设置 `Study.optimize` 和网络连接失败哨兵，未运行参数寻优、Walk-Forward 或真实两年回测。
复制单个 HTML 后在禁用网络的 Edge 中验证了切图、图例、框选、联动开关、单日/倒置/越界日期、
恢复、重叠汇总点击、超过 50 笔分页、真实价格选中、窗口外定位、懒加载与窗口尺寸变化；
两种规模均无未捕获 JS 异常、无外部资源请求。

| 合成规模 | 生成耗时 | 单文件大小（十进制 MB） | 顶部图表与控件就绪 |
| --- | ---: | ---: | ---: |
| 4 股 × 500 日 | 2.265 秒 | 7.22 MB | 1.798 秒 |
| 20 股 × 500 日 | 8.597 秒 | 15.90 MB | 2.432 秒 |

股票图按滚动延迟初始化，上述为本机实测而非固定性能承诺。最终样例运行目录分别为
`analytics/reports/synthetic_4stocks_500days_eb5dadc2/` 和
`analytics/reports/synthetic_20stocks_500days_d84aedf8/`，各含 `backtest.html` 和 `acceptance.json`。
未验证：Chrome 独立运行、其他操作系统及真实两年数据性能；首尾/中途行情完整性没有新增交易日历检测。

## 测试

```bash
python -m pytest tests/ -v
```

本次因子暴露任务运行下方指定的五个测试文件（合成 mock 数据，不联网）；完整测试目录未在本次重新执行，不把历史数量视为当前全量通过结果：

| 测试文件 | 覆盖范围 |
| --- | --- |
| `tests/test_data_aligner.py` | 多源时间对齐、T-1/T+1 隔离、防未来函数校验、DataSlice 组装 |
| `tests/test_factors.py` / `tests/test_features.py` | 微观结构/环境/主体分层因子与 FeatureEngine 端到端 |
| `tests/test_signals.py` | 连续评分纯函数公式（ES/PS/Fund_Stability/XS）精确值 + 向量化/NaN/零除/否决等健壮性用例、硬过滤层、状态机全流程 |
| `tests/test_backtest_engine.py` | 下一 Bar 成交、T+1 挂起卖出、涨跌停拦截、成本滑点、仓位/杠杆风控、成交日志与净值曲线 |
| `tests/test_optimizer.py` | 绩效指标纯函数、搜索空间归一化、StrategyOptimizer 端到端寻优、Walk-Forward OOS 报告 |
| `tests/test_analytics.py` | 实时流 JSONL、绩效精确值、平均成本批次归因及来源校验、IC/Rank IC/IR、复盘清单、Dashboard |
| `tests/test_exposure.py` | 有符号权益加权、现金影响、缺失覆盖率、空仓和未来因子隔离 |
| `tests/test_analysis_reporting.py` | 分日文件、JSON、报告开关一致性、无交易/未平仓及失败状态 |

## 开发状态

| 模块 | 状态 |
| --- | --- |
| 真实数据接入 `data/real_loader.py`（data1 + data2 → DataSlice） | ✅ 已完成 |
| Level-2 解析 `data/l2_loader.py`（行情/快照/逐笔成交/逐笔委托） | ✅ 已完成 |
| 宏观加载 `data/macro_loader.py` | ✅ 已完成 |
| 时间对齐 `data/aligner.py` + 数据切片 `data/dataslice.py` | ✅ 已完成 |
| 数据访问抽象 `data/storage.py` + `config/data_sources.py` | ✅ 已完成 |
| 指标计算 `indicators/trend.py` `shock.py` `basic.py` | ✅ 已完成 |
| 因子体系 `indicators/`（agent_profiling / microstructure / environment / feature_engine） | ✅ 已完成 |
| 仓位管理（PositionSizer + Target_Weight 差额调仓） | ✅ 已完成 |
| 信号合成与状态机 `strategy/signals.py`（ES/PS/XS + A股硬过滤 + 一票否决） | ✅ 已完成（评分算子已纯函数化/向量化重构） |
| 回测撮合引擎 `engine/`（backtest / execution / portfolio / risk_control） | ✅ 已完成（差额调仓 + 死区 + T+1 顺延） |
| 绩效指标 `analytics/metrics.py` + `performance.py`（PerformanceAnalyzer） | ✅ 已完成（含 Calmar/Sortino/持仓周期/复盘清单/Dashboard） |
| 收益分摊与暴露 `analytics/attribution.py` + `exposure.py`（含 IC/Rank IC/IR） | ✅ 已完成 |
| 实时流 `analytics/real_time_stream.py`（StreamLogger JSONL） | ✅ 已完成 |
| 机器学习优化 `optimizer/`（search_space / bayesian_opt / walk_forward） | ✅ 已完成 |
| 项目主入口 `main.py`（--data smoke / real） | ✅ 已完成 |
| 单元测试 `tests/` | 本次指定五个文件的结果见 `debug/factor_exposure_todo.md` 交付记录 |
| 旧策略壳 `strategy/`（base / position / sentiment / state_machine） | 🗑️ 已删除（空壳死代码，由 signals.py 与 risk_control 取代） |
| Brinson 基准归因 | 🚧 规划中（当前为入场分数盈亏分摊与策略分数暴露，需另行定义基准和模型） |

## Git 仓库注意事项

- 真实数据目录 `data/data1/`、`data/data2/` 与 `data/mock_data/` 已在 `.gitignore` 排除，禁止入库（大体积）
- 提交前钩子（`.git/hooks/pre-commit`）会拦截数据文件、`*.csv/parquet`、`*.pyc`、敏感文件与 >5MB 大文件的暂存
- 推送前请执行 `git status` + `git diff --cached --stat` 核对待推送内容

## 免责声明

本项目仅用于学习与研究，不构成任何投资建议。股市有风险，入市需谨慎。
