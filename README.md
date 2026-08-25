# 量化交易回测系统（A 股）

基于**真实高频数据**（万得 Level-2 逐笔 + 日频 CSV）的 A 股量化交易回测框架。项目以「数据层 → 指标层 → 策略层 → 执行引擎层 → 评估与绘图」的分层结构组织代码。

当前开发状态：**真实数据接入（`data/real_loader.py`，万得 L2 parquet + 日频 CSV）、数据管道与多源时间对齐、核心因子与主体情绪特征工程、连续评分信号与状态机（ES/PS/XS + A股硬过滤）、Target_Weight 差额调仓撮合引擎、绩效评估与收益归因、Optuna 贝叶斯寻优、项目主入口（`main.py --data smoke|real`）均已实现**；冒烟模式内置 mock 数据秒级跑通全链路，真实数据模式全链路验证通过（20 只股票 × 4 个月 / 2 只 × 2 年）。

## 功能特性

- **真实数据接入**：`data/real_loader.py` 将 `data/data1`（万得 Level-2 行情/逐笔成交/逐笔委托 parquet）与 `data/data2`（宏观/北向/两融/行业/龙虎榜 CSV）清洗、解析、对齐为标准 `DataSlice`，缺表自动降级，全部因子严格防未来函数
- **冒烟模式**：`main.py --data smoke` 内置 6 个交易日 mock 数据，秒级跑通「对齐 → 特征 → 连续评分 → 状态机 → 差额调仓回测」全链路回归
- **数据加载抽象**：数据源注册表（`config/data_sources.py`）统一管理本地数据位置，支持环境变量 / YAML 配置 / 运行时注册三级覆盖
- **指标体系**：MA、MACD、ADX、ATR、布林带等常用指标，全部仅使用历史数据（无未来函数）
- **市场状态判定**：基于趋势主轴斜率 + 残差波动率的 `is_trend` 判定，区分趋势日与震荡日
- **多因子打分**：趋势质量 TQ（0~3）+ 量能确认 VC（0~2），配合动态阈值（滚动分位数）
- **时间对齐管道**：`TimeAligner` 多源异构数据时钟对齐（宏观/外盘 T-1 全量对齐、龙虎榜 T+1 隔离），内置 `verify_no_lookahead` 防未来函数校验
- **因子体系**：资金主体分层（小/中/大/超大单净流）、微观结构因子（OFSS 盘口/CPS 筹码/PSS 价格结构）、宏观共振因子（MRS/GRS/IRS 与 Global_Mod/Chain_Mod），统一由 `FeatureEngine` 调度
- **连续评分信号**：ES（入场分，sigmoid 合成资金主体/纯净度/量价共振）+ PS（持仓分 = ES×Time_Decay×Fund_Stability）+ XS（出局分 clip 加权），数值统一有界化；独立 A 股硬过滤层（ST 禁买、涨跌停禁买卖、时间窗 10:00~14:50、成交额门槛）前置否决 + 一票否决（游资溃逃 / 大盘跳水），全部评分参数可寻优
- **目标权重 Target_Weight**：连续评分 → 分钟级目标持仓比例（base × ES/PS × 宏观/行业乘子），驱动撮合引擎差额调仓
- **回测撮合引擎**：事件驱动型分钟级撮合——Target_Weight 差额调仓 + 调仓死区（防过度换手）+ T+1 顺延挂起（当日买入锁定、跌停跳过）、动态滑点（订单参与率冲击模型）、佣金/印花税/过户费、单股与总杠杆上限风控，输出完整成交日志与净值曲线
- **超参数优化**：Optuna 贝叶斯寻优（`StrategyOptimizer`），Dirichlet 式权重归一化（和为 1）、TPE 原生多重硬约束（回撤 < 15%、胜率 > 55%、盈亏比 > 1.5、有效交易 ≥ 30 笔）、样本内年化 Sharpe 最大化、优化历程收敛图；搜索空间只含决策链实际读取的参数（旧二值化闸门死参数已清理），寻优结果可直接注入 `main.py` 实盘复现
- **Walk-Forward 交叉验证**：滚动/扩展训练段的滚动样本外（OOS）验证框架，每折独立寻优并在样本外回测，输出跨折 OOS 评估报告，避免前视偏差与过度拟合
- **绩效指标**：年化收益率、年化夏普、卡玛（Calmar）、Sortino、最大回撤、平均持仓周期、胜率/盈亏比、日收益偏度/峰度（`analytics/metrics.py` + `PerformanceAnalyzer`）
- **实时决策日志流**：`StreamLogger` 每 Bar × 每标的输出标准 JSON（Final_MS / Global_Mod / Chain_Mod / Capital_Purity / Action / State），JSONL 落盘 + 生成器双形态
- **收益归因**：`AttributionEngine` 因子暴露分解——把每笔已实现盈亏按入场时点拆解到宏观共振（Global_Mod）/ 行业共振（Chain_Mod）/ 个股情绪（Agent_MS）
- **因子预测能力检验**：因子 IC / Rank IC / IR 时序分析（横截面 Rank IC 与单标的时序相关双模式，前瞻窗口可配）+ IC 时间桶热力图
- **多子图 Dashboard**：净值回撤图、动态仓位图、归因柱状图、参数敏感度热力图（`plot_report`）
- **20 日人工复盘清单**：自动导出每笔交易触发前后 N 个交易日的日线指标切片 + 逐笔资金流清单（Excel 三 sheet / CSV）
- **可视化**：价格走势 + 均线 + 震荡区间色带标注，自动输出图片
- **规划中能力**：Brinson 基准归因、更多真实数据源（深市/北交所逐笔、期权持仓等）

## 项目结构

```
quantitative_trading/
├── main.py                     # 项目主入口（--data smoke|real 模式切换）
├── run_optimization.py         # 机器学习超参数寻优入口脚本
├── demo_plot.py                # 演示脚本：绘制回测段价格走势与震荡区间
├── requirements.txt            # 依赖：pandas / numpy / scipy / matplotlib / pyarrow / pyyaml / optuna / openpyxl
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
│   ├── attribution.py          # AttributionEngine：因子暴露分解归因（宏观/行业/个股情绪）
│   ├── ic_analyzer.py          # 因子 IC / Rank IC / IR 双模式分析 + Q1~Q5 分位分层净值
│   ├── real_time_stream.py     # StreamLogger：实时/仿真决策日志流（标准 JSONL + 生成器）
│   ├── plotter.py              # 绘图模块：价格走势 / 震荡区间标注
│   └── pictures/               # 生成的图片输出目录（含 optimizer_history.png / dashboard.png）
└── tests/
    ├── test_data_aligner.py    # 时间对齐、防未来函数与 DataSlice 组装
    ├── test_factors.py         # 微观结构与环境因子计算
    ├── test_features.py        # 主体分层与 FeatureEngine 端到端
    ├── test_state_machine.py   # （遗留空壳，状态机语义已迁移至 test_signals.py）
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

依赖清单：`pandas`、`numpy`、`scipy`、`matplotlib`、`pyarrow`（parquet 高性能存储）、`PyYAML`（配置文件解析）、`optuna`（超参数优化）、`openpyxl`（Excel 导出）。

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
REAL_START = "2023-01-03"          # 回测起始
REAL_END   = "2024-12-31"          # 回测结束
REAL_SYMBOLS = ["600171", "600460"]  # 股票子集；空列表 = 全部 20 只（全量较慢）
```

- 空 `REAL_SYMBOLS` 时自动发现 `data/data1` 全部标的（当前 20 只沪市）
- 2 只 × 2 年：首次约 9 分钟（逐笔 6000 万行 + 状态机），**信号缓存命中后约 2.5 分钟**；20 只 × 4 个月约 5~6 分钟
- `main.py` 会自动落盘**信号缓存**（`data/feature_cache/signals_*.pkl`）：回测区间、股票子集、参数任一未变化时跳过状态机重算，直接复用信号结果（寻优每 trial 参数不同因此不命中）
- 输出到 `analytics/pictures/`，状态检查 5 项（含防未来函数断言）

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
```

### 用法

```python
from data.real_loader import RealDataLoader

loader = RealDataLoader()
ds = loader.load_slice(["600171", "600460"], "2023-01-03", "2024-12-31")  # 已内置对齐
ds.validate()
```

### 防未来函数设计

- 分钟级表（kline / l2_snapshot / tick_trades）直接用当日实时数据（当前 Bar 已收盘）
- 日频表（macro / north_margin / industry）由 `TimeAligner` 做 T-1 全量对齐
- 龙虎榜由 `TimeAligner` 标注 T+1 可用日（`avail_date`），T+1 前不可见
- 缺口近似（全部当日可观测，无未来函数）：伪指数（20 标的等权分钟均线）、广度（20 标的涨跌家数）、北向净流（日频 T-1 填充）、行业资金流（指数 close 日间变化 T-1 对齐）

### 已知数据缺口（当前降级处理，不影响运行）

| 缺口 | 处理 |
| --- | --- |
| 600237/600379/603380 无北向持股 | `north_sync` 恒 NaN，对应信号分量保守为中性（不参与评分） |
| 宏观缺 DXY 列 | 源文件无 DXY 列 → `dxy` 恒 NaN（加载器已支持读取，补齐数据列即生效） |
| 9 只股票北向 2023 年 1~3 月起才有数据 | 前段 `north_sync` 为空（T-1 不填充） |
| 股票融券余额全 NaN（仅 ETF 有值） | `margin_pressure` 降级为融资余额变化率 |
| 涨跌停价口径（已修正） | 以 **T-1 昨收×幅度** 计算：ST±5%、创业板（300-302）/科创板（688）±20%、主板±10%，round(2)；数据周期首日无昨收以自身收盘近似 |
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
- **模拟权重**：`Position.simulated_weight` 由状态机动作维护（开仓=目标、减仓 ×0.8、加仓重算），与引擎真实成交仓位相互独立
- **次日低开反包（Reversal / Counter-Attack）**：持仓跨入次日且深度低开（≤ th_reversal_gap=-1.5%）+ 盘口承接（OFSS > 0.2）+ 大资金逆势净流入（purity>0 且 big_flow>0）→ 豁免 XS 清仓/阶梯减仓，并在窗口内（受开仓 time 闸门共同约束）承接加仓 `target = min(simulated + base×ES×reversal_add_mult, max_single_position)`；大盘熔断（沪深300 跌破 VWAP-1.5%）时保护立即失效强制清仓
- **硬约束**：`engine/risk_control.py` 的 `PositionSizer` 校验单股最大仓位与总账户杠杆上限，目标权重经 `max_single_position` 裁剪；`config/strategy_params.py` 提供全部默认参数（DEADZONE_TH=0.05 等）

## 信号合成与状态机

`strategy/signals.py` 提供 `SignalSynthesizer`（连续评分合成）与 `TradingStateMachine`（逐 Bar 状态机）：

- **ES 入场分**：sigmoid 合成资金主体（机构净流带权重偏置）、资金纯净度、量价共振（MRS）等维度，区间 [0,1]；构造参数全部可寻优（w_es_ms / w_es_purity / w_es_mrs / es_sigmoid_k / th_es_entry）
- **PS 持仓分**：`ES × Time_Decay × Fund_Stability`；Time_Decay = 前 30 分钟（win_decay_grace）恒 1.0，此后按浮盈亏非对称衰减：浮盈 factor=0.975（减半）、浮亏 factor=0.90（加倍），`clip(factor^(eff_bars/10), 0.1, 1.0)`
- **XS 出局分**：汇入均线偏离、动量衰减、回撤（高水位）、超时等维度，clip 加权到 [-1,1]，驱动 清仓 / 减仓 / 持有 三分支
- **A 股硬过滤层**（否决前置）：ST 禁买、涨停禁买/跌停禁卖、交易时间窗 10:00~14:50（10:00 整之前禁买，防开盘冲高骗局）、成交额门槛；未持仓时任一不过即不开仓（`hard_filters` 与状态机 BUY 口径一致）
- **一票否决**：游资溃逃（资金流为负且散户追高超阈值）或大盘跳水（沪深300 跌破 VWAP×(1-1.5%)）→ XS=-1，强制出局
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
    Account(initial_cash=1e8, max_leverage=1.0, max_single_position=0.3),
    ExecutionCost(),         # 佣金 / 印花税 / 过户费 / 动态滑点
    PositionSizer(),         # 单股/杠杆上限风控与目标权重裁剪
    ds, signals,
)
trade_log, equity_curve = eng.run()   # 完整成交日志 + 逐 Bar 净值曲线
```

## 测试

```bash
python -m pytest tests/ -v
```

当前 175 项单元测试全部通过（合成 mock 数据，不联网；运行约 40 秒）：

| 测试文件 | 覆盖范围 |
| --- | --- |
| `tests/test_data_aligner.py` | 多源时间对齐、T-1/T+1 隔离、防未来函数校验、DataSlice 组装 |
| `tests/test_factors.py` / `tests/test_features.py` | 微观结构/环境/主体分层因子与 FeatureEngine 端到端 |
| `tests/test_state_machine.py` / `tests/test_signals.py` | 连续评分公式（ES/PS/XS）精确值、硬过滤层、状态机 BUY/ADD/DECAY/SELL/HOLD 全流程 |
| `tests/test_backtest_engine.py` | 下一 Bar 成交、T+1 挂起卖出、涨跌停拦截、成本滑点、仓位/杠杆风控、成交日志与净值曲线 |
| `tests/test_optimizer.py` | 绩效指标纯函数、搜索空间归一化、StrategyOptimizer 端到端寻优、Walk-Forward OOS 报告 |
| `tests/test_analytics.py` | 实时流 JSONL、绩效指标精确值、因子归因盈亏守恒、IC/Rank IC/IR 双模式、复盘清单导出、Dashboard |

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
| 信号合成与状态机 `strategy/signals.py`（ES/PS/XS 连续评分 + A股硬过滤 + 一票否决） | ✅ 已完成 |
| 回测撮合引擎 `engine/`（backtest / execution / portfolio / risk_control） | ✅ 已完成（差额调仓 + 死区 + T+1 顺延） |
| 绩效指标 `analytics/metrics.py` + `performance.py`（PerformanceAnalyzer） | ✅ 已完成（含 Calmar/Sortino/持仓周期/复盘清单/Dashboard） |
| 收益归因 `analytics/attribution.py`（因子暴露分解 + IC/Rank IC/IR） | ✅ 已完成 |
| 实时流 `analytics/real_time_stream.py`（StreamLogger JSONL） | ✅ 已完成 |
| 机器学习优化 `optimizer/`（search_space / bayesian_opt / walk_forward） | ✅ 已完成 |
| 项目主入口 `main.py`（--data smoke / real） | ✅ 已完成 |
| 单元测试 `tests/` | ✅ 175 项通过（信号 73 + 引擎 30 + 优化器 21 + 归因/IC/流 21 + 对齐 15 + 特征 15） |
| 旧策略壳 `strategy/`（base / position / sentiment / state_machine） | 🗑️ 已废弃（空壳死代码，由 signals.py 与 risk_control 取代，待清理） |
| Brinson 基准归因 | 🚧 规划中（当前为因子暴露分解，需基准收益） |

## Git 仓库注意事项

- 真实数据目录 `data/data1/`、`data/data2/` 与 `data/mock_data/` 已在 `.gitignore` 排除，禁止入库（大体积）
- 提交前钩子（`.git/hooks/pre-commit`）会拦截数据文件、`*.csv/parquet`、`*.pyc`、敏感文件与 >5MB 大文件的暂存
- 推送前请执行 `git status` + `git diff --cached --stat` 核对待推送内容

## 免责声明

本项目仅用于学习与研究，不构成任何投资建议。股市有风险，入市需谨慎。
