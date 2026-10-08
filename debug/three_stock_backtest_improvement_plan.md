# 三股回测问题与策略改进方案

> 编写日期：2026-10-03。本文是实施与验证规格；当前只更新此文档，不修改策略、撮合或数据代码。第 10～14 节给出具体文件、函数、参数和测试要求；若与前文概述存在歧义，以具体规格为准。

## 1. 范围与基线

本方案针对 `600171` 上海贝岭、`600732` 爱旭股份、`600888` 新疆众和在 2023-01-03 至 2024-12-31 的连续账户回测。原始结果保存在 `analytics/reports/research_3stocks_2023_2024_nonst_20260929/`，以其中 `summary.json`、`run_config.json`、`trade_log.csv`、`equity_curve.parquet` 为基线。原始报告应保留，后续实验使用独立目录和唯一配置标识。

| 基线指标 | 结果 |
| --- | ---: |
| 期初资金 | 1,000,000 元 |
| 2023 年末权益 / 年收益 | 928,355.69 元 / -7.16% |
| 2024 年末权益 / 年收益 | 1,064,597.19 元 / +14.68% |
| 两年累计收益 / 最大回撤 | +6.46% / 14.78% |
| 成交 / 已平仓交易 | 34 笔 / 17 笔 |

2023 年末三个标的对账户的累计盈亏约为：`600171 -22,773 元`、`600732 -15,646 元`、`600888 -33,225 元`。当年已平仓交易合计盈利约 1.97 万元，主要损失留在未平仓持仓中。2024-02-06 账户曾降至约 86.62 万元。2024 年账户回升主要来自 `600171`，不能仅凭最终两年盈利认定策略稳定有效。

**本次结果的前提**：按用户要求暂时假设三股在整个区间均非 ST；历史 ST 状态未核实。独立指数分钟、市场广度和流通股本缺失；2023 年三个标的的 `MRS` 均缺失。当前参数曾参考另一只股票的 2024 年结果，存在参数选择偏差。后续对比必须写明这些前提是否变化。

## 2. 已确认的问题与待验证的假设

| 类别 | 已确认的事实 | 尚不能直接断定的事 |
| --- | --- | --- |
| 撮合 | 2023 年有 43 条 `DECAY_REDUCE` 因 `invalid_open` 被拒，其中 42 条在 11:30；对应 K 线 `open/close` 为空。现有 `pending_signals` 在本 Bar 处理后清空。 | 让这些订单延续后，账户收益一定改善；需要同参数对照回测。 |
| 减仓 | `deadzone_th=0.0831`，是绝对仓位差 8.31 个百分点。`base_weight × reduce_step_ratio = 0.1913 × 0.601 ≈ 11.50%`。`600888` 的 2023 年仓位大约在 12.7%～15.8%，常规减仓幅度达不到死区。 | 单独缩小死区是最优方案；它可能增加换手与成本。 |
| 清仓 | 当前出局分的价格回撤项权重为 `0.3446`，常规清仓线为 `-0.4482`。其他分量中性时，价格回撤单独无法穿越清仓线。`600888` 两年没有卖出成交。 | 某个固定止损比例就能稳定提高样本外表现。 |
| 重新入场 | `600171`、`600732` 在 2023 年出现卖出后当天重新买入。`_on_flat` 对确认卖出后的重新入场没有冷静期。 | 同日重新入场全部无效；某些情况可能抓住真实反转。 |
| 输入数据 | `MRS` 缺失按中性参与入场评分；中性入场分为 `0.5`，高于本次 `th_es_entry=0.2148`。指数分钟缺失使相关大盘跳水否决支路无法触发。 | 单纯上调入场阈值能解决亏损；可能同时丢掉有效交易。 |

下文按依赖关系安排：先让回测准确表达信号与成交，再改退出、减仓和入场，最后用独立样本评价。

## 3. 第一阶段：修正无效分钟与订单生命周期（最高优先级）

### 3.1 明确“可决策”和“可成交”两种 Bar

- 在 `engine/backtest.py` 中分别检查决策所需的 `close` 等字段、撮合所需的 `open` 与涨跌停价；像本次 11:30 这样开收盘价都为空的占位 Bar 不产生新信号，也不能被当作一次失败后即终止的撮合机会。不能只凭成交额为零就判定 Bar 无效。
- 下一根可成交 Bar 才执行上一根有效 Bar 的信号。保持先形成信号、后按下一可交易开盘价成交的时间顺序，禁止用下一 Bar 的收盘信息决定其开盘订单。
- 对停牌、整日缺报价、涨跌停、午间间隔、最后一根 Bar 分别定义处理方式；不能把它们统一归为 `invalid_open`。

### 3.2 明确不同订单的有效期

建议给待执行订单保存 `signal_id`、`decision_ts`、动作、目标仓位、最晚有效时间和状态。采用以下初始规则，再通过对照回测检验：

1. `BUY/ADD`：仅在产生信号的连续交易时段内寻找下一根可成交 Bar；超过有效期则记为 `expired`，避免午间或隔夜执行已经过时的买入信号。
2. `SELL/DECAY_REDUCE`：在空报价、跌停或 T+1 不可卖时保留待执行状态，到下一可交易 Bar 再尝试；账户实际仓位归零或新的明确风险决策替代它时终止。普通 `HOLD` 不应静默删除未完成的风险订单。
3. 同一标的存在新信号时，明确风险优先级：强制清仓高于减仓，减仓高于新增风险暴露。新目标替代旧目标时，日志同时记录来源和替代原因，不能产生重复成交。
4. 下一次撮合按当时账户权益、实际股数和该 Bar 价格计算可执行数量；继续遵守 100 股整数倍、T+1、涨跌停、现金和费用规则。

`engine/backtest.py::pending_signals`、`pending_targets`、`_execute` 是主要修改点。`TradeLog` / `OrderReport` 应能区分 `filled`、`pending`、`expired`、`superseded`、`rejected`，并能从一条信号追到最终成交或终止。若只修撮合，不应同时改信号阈值。

### 3.3 验收

- 增加最小模拟行情：11:29 有减仓信号、11:30 空价、13:00 有效报价；核对 13:00 执行或明确记录未执行原因。
- 覆盖跨日、停牌、跌停、T+1 锁定、部分卖出、新信号覆盖、末 Bar 信号等情况；验证无未来数据、无重复成交、无负现金、无超卖。
- 在完全相同的股票、区间、参数和费用下运行“仅修撮合”版本，单独报告 2023 年 43 条失败减仓的最终去向及账户结果变化。结果可能更好也可能更差，均保留。

## 4. 第二阶段：让风险退出独立于情绪评分

当前 `strategy/signals.py::_compute_xs` 的回撤项只是综合分的一部分，不能充当明确的价格止损。建议增加独立、可配置的风险触发器：

- **成本止损**：以实际成交后的账户持仓成本为基准，计算 `close / cost_basis - 1`。达到预设亏损边界时发出清仓信号。
- **持仓回撤退出**：以持仓期间可观察到的最高收盘价为基准，计算 `1 - close / high_watermark`。可单独决定是否只在已有浮盈后启用，以免和成本止损重复。
- **最长持仓期限**：现有 `win_hold_max` 参数已被保存，但未在持仓决策里作为实际清仓条件。若启用，先明确单位是有效交易分钟、交易日还是自然日，再独立测试；不要直接把当前默认值当作已验证阈值。
- **账户层风险限额**：可单独测试账户回撤触发的暂停开仓或降仓；以确认的账户权益峰值计算，避免同一分钟先看到未来权益再交易。

决策优先级建议为：一票否决或硬风险退出 → 综合分清仓 → 风险减仓 → 常规持有或加仓。硬风险退出不能被“次日低开反包”保护逻辑冻结，也不能被普通调仓死区挡住。若受跌停或 T+1 约束，保留待执行卖单；**触发止损不等于保证按止损价格成交**。

首次实验应分别打开每一种退出条件。阈值依据事先定义的交易风险预算、股票波动分布与训练样本选择，记录试过的全部候选值；不要围绕 `600888` 的已知跌幅定制一个恰好有效的数字。比较结果除总收益外，还要看单笔最大亏损、持仓天数、错过的反弹、2024 年收益和滑点。

## 5. 第三阶段：区分风险减仓与普通再平衡

目前 `engine/backtest.py::_rebalance` 把普通微调与 `DECAY_REDUCE` 共用一个 8.31 个百分点的死区。建议拆成两类：

1. **普通再平衡**继续使用成本感知的死区，限制无意义的频繁交易。
2. **风险减仓**按事件触发，并使用独立的最小可成交量或仓位变化门槛。若目标低于当前仓位且能卖出至少一个交易单位，就不再被普通死区直接跳过；是否成交仍受交易规则与流动性约束。

`reduce_step_ratio` 目前对应锚定目标（约 11.50%），不是每根 Bar 递归乘 `0.601`。先保持这一语义，避免连续信号使仓位指数式缩小。若希望多级减仓，应另设清楚的风险等级及目标仓位，并做到“进入更差等级才减下一档”；恢复后怎样重置等级也要定义。日志需区分 `deadzone_skip`、`min_lot_skip`、`no_sellable_shares`、`invalid_quote`，否则仅看成交日志无法判断减仓为什么没有执行。

## 6. 第四阶段：控制卖出后的重复买入

在 `strategy/signals.py::TradingStateMachine` 增加标的级“最近一次**确认成交的**清仓时间与原因”。冷静期从实际成交开始计时，不从卖出信号开始计时；被跌停或空价拒绝的卖出不能误触发冷静期。

建议分开试验两种机制：

- **时间冷静期**：风险退出后至少等待至下一个交易日或预设有效交易分钟数。普通获利退出可采用不同规则。
- **再入场确认**：新入场分需高于原阈值一段距离，或连续若干有效 Bar 达标；可配合日线趋势条件。入场条件恢复后再买，而不是在卖出的几分钟后按同一低阈值买回。

记录 `600171` 在 2023-04-13、`600732` 在 2023-02-01 和 2023-09-07 的再入场是否被挡住，随后查看收益、持仓和错失反弹。不要只以这三笔的已知结果选冷静期长度。

## 7. 第五阶段：补足数据并规定缺失时的行为

1. **历史 ST**：取得逐日、按当时可知信息生效的 ST 状态，再重跑合规过滤。当前“全程非 ST”仅是研究情景。
2. **指数分钟及市场广度**：补齐与个股分钟轴一致的历史数据，检查大盘跳水否决和环境因子是否按预期工作。
3. **MRS 与其他因子**：报告每个输入的非空率、缺失持续时间及来源；特别排查本次 `MRS` 全缺时的中性回退。缺失的市场风险输入不应在正式评估中静默当作“风险正常”。
4. **行情质量**：核对 11:30 空占位、停牌缺口、除权分红口径、涨跌停价与成交额单位；继续保留对 2024-02-06/08 时间格式差异的解析测试。

研究模式可以保留明确标记的缺失数据情景；正式样本外评估应设覆盖率门槛，未达到则中止该次评估并报告缺口。数据补齐本身也应成为单独的实验变量，避免把“新数据”和“新策略规则”的效果混在一起。

## 8. 实验顺序与判定口径

| 实验 | 变更 | 重点观察 |
| --- | --- | --- |
| A | 原始基线，只归档与复现 | 与现有 `summary.json`、成交笔数、权益曲线对齐 |
| B | 空 Bar 与订单生命周期修正 | 失败减仓去向、成交延迟、2023 回撤 |
| C1 / C2 / C3 | 各自以 B 为固定基线，分别加入成本止损、回撤退出、持仓期限 | 尾部亏损、2024 盈利是否被过早卖出 |
| D | 以 B 为固定基线，只让风险减仓绕开普通死区 | `600888` 等持仓是否真的减小、换手成本 |
| E | 以 B 为固定基线，分别加入冷静期或再入场确认 | 同日卖买次数、反弹机会成本 |
| F | 补齐历史 ST、指数、广度、MRS 等数据 | 信号与收益变化、原结论是否仍成立 |

每组输出完整配置、代码版本、数据版本、成交/拒单明细、逐日权益、标的盈亏归因。至少比较：

- 2023 与 2024 分年收益、两年收益、最大回撤、日频夏普；期末未平仓盈亏单列，不以已实现盈利替代账户收益。
- 单笔及单股最大亏损、持仓时间、平均与峰值仓位、换手、佣金/税费/滑点、风险信号从决策到实际成交的延迟。
- 与三股同期简单持有、固定仓位策略及合适市场指数的对比；说明基准的复权和资金口径。

验收原则是：先确认撮合正确和约束未被破坏，再评估风险收益权衡。不能只以 2023 年亏损减少为通过条件；如果 2024 年收益、交易成本或样本外表现显著恶化，应如实保留并解释。一次只改变一个机制，再测试组合；完整记录失败实验，避免只展示最佳参数。

## 9. 样本外验证与实施边界

- 预先固定股票池及纳入规则，覆盖不同行业、上涨/下跌/震荡阶段，并纳入退市、停牌及历史 ST 情况，控制幸存者偏差。三只股票、17 笔平仓不足以证明普适性。
- 参数选择与评估按时间顺序分开；使用此前未参与参数选择的后续时期或独立股票作为最终检验。由于本次参数参考过 2024 年另一只股票且现已查看 2023–2024 结果，这两个年份不能再视为完全未触碰的最终样本。
- 若暂无足够新数据，可先完成撮合正确性和机制对照，但结论限于“解释本次样本”，不要宣称已经获得稳健的实盘收益。
- 建议先完成 `tests/test_backtest_engine.py`、`tests/test_signals.py`、`tests/test_real_loader.py` 中对应的小型确定性场景，再运行三股对照；不为每个数值阈值编写与实现同义的测试。

**拟涉及的代码位置**：`engine/backtest.py`（订单生命周期、拒单、风险减仓）、`strategy/signals.py`（独立退出、再入场状态）、`data/real_loader.py` 与 `indicators/feature_engine.py`（数据覆盖与缺失口径）、`scripts/research_backtest_three.py`（可复现实验配置与输出）、相应测试和报告模块。`data/l2_loader.py` 现有时间格式修复先保留，只通过回归测试确认。实际实施时保持原始报告和当前未提交改动，不覆盖既有实验产物。

## 10. 逐文件修改清单

下表以**当前函数名**定位，避免行号随其他工作变化。B～E 属于本次三股问题的直接修复；F 是数据补齐与独立验证，不应混进 B～E 的效果对比。

| 批次 | 文件与函数 | 具体修改 |
| --- | --- | --- |
| B：撮合 | `engine/backtest.py::BacktestEngine.run` | 每个股票分别判断当前 Bar 能否决策、能否撮合；空价 Bar 保留旧权益标记，但不生成新信号。将待执行非 `HOLD` 信号按标的保存，先处理到期/替代，再在下一有效开盘价撮合。 |
| B：撮合 | `engine/backtest.py::BacktestEngine._execute`、`_rebalance` | 无报价/无效开盘价时返回“待执行或过期”结果，不把风险卖单当作终结拒单；维持开盘价、涨跌停、T+1、费用和死区校验顺序。 |
| B：撮合 | `engine/backtest.py::PendingTarget`、`_cancel_pending`、`_execute_sell_to_target` | 保存原动作、策略退出原因和决策来源；兼容已有 T+1 顺延，避免新的空价顺延与旧 `pending_targets` 对同一标的重复执行。 |
| B：记录 | `engine/backtest.py::OrderReport`、`TradeLog.add`、`TradeLog.to_frame` | 记录待执行、到期、被替代及最终成交的状态。保留原成交 `reason` 值；另加策略原因字段或独立事件表。 |
| C：退出 | `strategy/signals.py::SignalSynthesizer.__init__`、新增 `hard_exit_reason` | 接受独立成本止损与持仓回撤阈值，返回明确原因；不改现有 `_compute_xs` 数学公式。 |
| C：退出 | `strategy/signals.py::TradingStateMachine._on_holding` | 在 `reversal_active` 和 XS 决策前检查独立风险退出，设置 `target_weight=0`，记录 `exit_cause`。 |
| C：期限 | `strategy/signals.py::TradingStateMachine.__init__`、`reset`、`on_bar` | 可选地维护交易日序号与实际持仓起始日，达到最长有效交易日数才触发期限退出；不复用语义不清且当前未参与退出判断的 `win_hold_max`。 |
| D：减仓 | `engine/backtest.py::BacktestEngine.__init__`、`_rebalance`、`_execute_sell_to_target` | 为 `DECAY_REDUCE` 增加独立于 `deadzone_th` 的开关与最低减仓股数；没有达到 100 股时记录跳过原因。仍保留 `reduce_step_ratio` 的锚定目标。 |
| E：再入场 | `strategy/signals.py::TradingStateMachine.on_bar`、`_on_flat`、`reset` | 仅在账户确认清仓后记录退出时间与原因；按冷静期及更高的再入场分数门槛决定是否买入。 |
| E：评分一致 | `strategy/signals.py::SignalSynthesizer.entry_gates`、`entry_all`、`generate_target_weights`，以及 `TradingStateMachine._scores` | 增加可选的本次入场阈值透传，使 BUY 决策和 `target_weight` 使用同一个门槛；冷静期挡住买入时目标必须为 0。 |
| F：数据 | `data/real_loader.py::_load_stock_history`、`_load_kline`、`_load_market_table` | 使历史 ST 可独立于流通股本加载；报告股票历史、指数与广度的实际覆盖率。 |
| F：数据 | `indicators/feature_engine.py::_environment_factors` | 保留缺失 `MRS` 为 NaN 的现有含义，额外输出因子覆盖诊断；不要在特征生成阶段伪造市场中性值。 |
| 全部实验 | `scripts/research_backtest_three.py::run_backtest`、`main` | 参数从独立 JSON 文件传入，特征输入目录与结果目录分离；保存解析后的完整参数、策略原因、订单事件和数据覆盖统计。 |
| F：研究脚本 | `scripts/research_backtest_three.py::AssumedNonSTLoader`、`prepare_chunk`、新增 `prepare_market_chunk`、`run_backtest` | 增加显式 `st_mode`：当前假设模式使用原子类，历史模式改用 `RealDataLoader`。每季度按三股的联合时间轴保存一次独立指数/广度；最终 `DataSlice` 也必须接入指数数据，不能只停留在特征阶段。 |
| 接入 | `main.py::REAL_PARAMS`、`run_pipeline`；`optimizer/bayesian_opt.py::StrategyOptimizer.backtest` | 新参数先以关闭值透传，保证普通真实数据入口和研究脚本语义一致；暂不把新阈值加入 `optimizer/search_space.py`。 |
| 验证 | `tests/test_backtest_engine.py`、`tests/test_signals.py`、`tests/test_real_loader.py` | 增加下文列出的确定性场景；原有成交与统计测试必须继续通过。 |

`engine/portfolio.py::Account` 的现有 T+1、成本和现金计算可供上述改动复用，首轮不改账本算法。`analytics/metrics.py::closed_trades` 继续用账户成交日志计算平仓盈亏；不要把策略退出原因覆盖到已有 `TradeLog.reason`，因为既有测试与报告依赖 `signal_sell`、`decay_reduce`、`t1_deferred_sell` 等值。

## 11. 参数、默认值与实验候选

新增的止损、期限、再入场和风险减仓开关**默认关闭**，这样可以独立验证每项机制；`entry_order_expiry` 属于 B 组撮合语义，默认采用同交易时段到期。下表“候选值”只是事先登记的研究网格，不能直接作为实盘配置，也不能在看完三股结果后只保留最优组合。

| 参数与归属 | 类型、默认值 | 实验候选 / 具体含义 |
| --- | --- | --- |
| `entry_order_expiry` → `BacktestEngine.__init__` | `"same_session"` | 新增。`BUY/ADD` 只能在决策发生的上午或下午连续交易时段撮合；过午间或隔夜则到期。`SELL/DECAY_REDUCE` 不套用此规则。 |
| `risk_reduce_bypass_deadzone` → `BacktestEngine.__init__` | `False` | D 组改为 `True`；仅 `DECAY_REDUCE` 不受原 `deadzone_th=0.0831` 阻挡。`SELL` 本来就豁免，普通调仓继续遵守死区。 |
| `risk_reduce_min_shares` → `BacktestEngine.__init__` | `100` 股 | 与现有整手取整一致；目标差不足 100 股时只记跳过，不制造零股成交。 |
| `stop_loss_pct` → `SignalSynthesizer.__init__` | `None`（关闭） | C1 组可分别试 `0.08/0.12/0.16`。若 `close / actual_cost_basis - 1 <= -stop_loss_pct`，发出 `SELL`；成本以账户已成交持仓的 `cost_basis` 为准。 |
| `trailing_stop_pct` → `SignalSynthesizer.__init__` | `None`（关闭） | C2 组可分别试 `0.10/0.15/0.20`。若 `1 - close / high_price_watermark >= trailing_stop_pct`，发出 `SELL`；只用当前及历史有效 Bar 的价格更新水位。 |
| `max_holding_trading_days` → `TradingStateMachine.__init__` | `None`（关闭） | C3 组可分别试 `10/20/40` 个交易日；以全局回测交易日期计数，个股停牌期间仍计入持仓时长。达到期限的当前有效 Bar 收盘产生卖出信号，下一可交易开盘撮合。 |
| `reentry_cooldown_trading_days` → `TradingStateMachine.__init__` | `0`（关闭） | E 组可分别试 `1/2/5`；按实际清仓成交后的交易日序号计算，`1` 表示清仓当日不重买、下一交易日允许。 |
| `th_es_reentry` → `SignalSynthesizer.__init__` | `None`（沿用 `th_es_entry`） | E 组可试 `0.55/0.60`；仅确认清仓后的再入场使用，首次建仓仍用原 `th_es_entry=0.2148`。 |
| `data_quality_mode` → `scripts/research_backtest_three.py::main` | `"scenario"` | F 组使用 `"strict"`；在有效决策 Bar 上检查 `is_st`、指数、广度与 `mrs` 覆盖率，不达预设门槛即停止回测并列出缺失日期。当前非 ST 假设仅允许 `scenario`。 |
| `st_mode` → `scripts/research_backtest_three.py::main` | `"assumed_non_st"` | 现有基线继续使用 `AssumedNonSTLoader`；F 组设为 `"historical"` 后使用 `RealDataLoader` 读取真实有效日期的 `stock_history.csv`。历史模式不得回退到全程非 ST 假设。 |
| 原 `deadzone_th`、`reduce_step_ratio` | 本次基线 `0.0831`、`0.601` | B、C、E 组保持原值；D 组只改变风险减仓是否豁免，不同时优化这两个旧参数。 |

参数校验：百分比必须在 `(0,1)`；`max_holding_trading_days` 为正整数或 `None`；`reentry_cooldown_trading_days >= 0`；`th_es_reentry` 在 `(0,1]` 且不低于 `th_es_entry`；`risk_reduce_min_shares` 为正的 100 股整数倍；`entry_order_expiry` 只接受 `same_session` 或用于敏感性测试的 `next_valid_bar`（只等到下一交易日首根有效 Bar，再没有报价则到期，绝不无限等待）；`data_quality_mode` 只接受 `scenario/strict`；`st_mode` 只接受 `assumed_non_st/historical`，且 `strict` 必须搭配 `historical`。严格模式的覆盖率门槛写入同一实验 JSON，分母只统计具有可决策价格的 Bar；具体门槛在看到新数据的实验结果前固定。实验 JSON 要同时记录旧参数、新参数、是否启用及代码/数据版本。`main.py::run_pipeline` 和 `optimizer/bayesian_opt.py::StrategyOptimizer.backtest` 负责透传，`optimizer/search_space.py` 暂不采样这些新参数，以免把机制验证变成大规模寻优。

## 12. 函数级执行细节

### 12.1 `BacktestEngine.run`：处理午间空 Bar

按标的维护待执行队列，并保持每个时间戳的顺序：①解冻 T+1；②用**当前开盘前已知**的账户与上一有效 Bar 流动性尝试撮合旧订单；③按当前有效收盘价盯市；④仅对价格有效的标的生成新信号；⑤记录权益。可在 `run` 内增加小型辅助函数 `_can_decide(brow)` 和 `_can_execute(brow, action)`，不要只以 `amount == 0` 判断无效。`close` 或 `open` 为 NaN/非正时分别不可决策或不可成交；涨跌停缺失要保留原有拒绝/顺延规则。

具体边界：

1. 11:29 的 `DECAY_REDUCE` 遇到 11:30 空 Bar 时进入待执行状态；11:30 不生成新信号，13:00 首个有效开盘尝试卖出。若 13:00 跌停或当日买入仍受 T+1 锁定，继续顺延。
2. 11:29 的 `BUY/ADD` 遇到 11:30 空 Bar 时按 `entry_order_expiry="same_session"` 到期；13:00 不补买。下午最后时段同理，不隔夜补买。
3. `HOLD` 不进入待执行队列，也不自动撤销尚未完成的卖出。新 `SELL` 覆盖旧 `DECAY_REDUCE`；同等级新风险卖出可更新目标与来源；新的 `BUY/ADD` 不能自动覆盖尚未完成的强制清仓。若将来支持撤销强制清仓，必须由明确的撤单规则和日志记录驱动。
4. 已清仓或目标已达到时清除待卖目标。部分卖出时只顺延剩余目标；相同 `signal_id` 不得生成两份独立卖出义务。回测结束仍未完成的风险订单记 `pending_at_end`，不伪造成交。
5. 当前代码会用空 Bar 的 `amount` 覆盖 `previous_amount` 为 `1e-9`。应改为只从有有效成交额的**已完成** Bar 更新流动性；跨午间或隔夜无同交易时段历史量时使用保守的现有 `1e-9` 兜底并记录 `liquidity_fallback`。不得读取尚未完成的 13:00 Bar 成交额估计 13:00 开盘滑点。
6. `_rebalance` 当前在卖出遇到跌停时只写 `limit_down` 拒单。对风险 `SELL/DECAY_REDUCE` 应改成记录这次不可成交，同时保存待卖目标供后续有效 Bar 再试；`BUY/ADD` 遇涨停仍按其买单有效期处理。`OrderReport.status` 应标为待执行而非终结拒单，原 `TradeLog.reason="limit_down"` 可保持不变。

实现时可给 `PendingTarget` 加 `action`、`exit_cause` 与有效期；复用 T+1 顺延路径，并在 `_cancel_pending` 中统一处理到期/替代。不要另建一条不受 T+1 管理的卖单队列。`_execute_sell_to_target` 在卖出前用**本次开盘价与账户实时权益**重算目标股数；缺报价时保留风险订单。`TradeLog.reason` 保持现有成交分类，新增 `strategy_cause`/`order_status` 到 `OrderReport` 或单独 `order_events.csv`；若增加 `TradeLog.to_frame` 列，应只在现有列尾追加，成交日志仍以 `shares > 0` 判定真实成交。

### 12.2 `SignalSynthesizer` / `TradingStateMachine`：止损与期限

在 `SignalSynthesizer` 增加 `hard_exit_reason(row, pos) -> str | None`，计算两个独立条件，返回 `stop_loss`、`trailing_stop` 或 `None`。`pos.avg_cost` 由 `TradingStateMachine.on_bar` 从 `Account.Position.cost_basis` 同步；`high_price_watermark` 只在有效的持仓 Bar 更新。两条件同时满足时固定优先级，保证同一 Bar 仅发一个 `SELL`。

在 `TradingStateMachine._on_holding` 更新最高价之后、`reversal_active` 分支之前检查硬退出；命中时令 `scores["target_weight"]=0`、`scores["exit_cause"]` 记录原因并返回 `ACT_SELL`。已有 `veto_flag` 可以继续优先处理，或与硬退出合并为固定顺序 `veto → stop_loss → trailing_stop → max_holding_trading_days → XS exit → reduce → add`。关键约束是任何硬退出都不能被反包分支冻结。`_metrics` 当前会把 `_scores` 值统一转为 `float`；因此 `exit_cause` 字符串需在 `_metrics` 单独处理，或在构造 `Signal` 后加入 `metrics`，不能直接送进现有数字循环。

`max_holding_trading_days` 用回测轴上实际出现的**交易日期**计数，不用自然日差，也不把每分钟当成一天。`TradingStateMachine.reset` 重置日期序号；`on_bar` 在首次见到新交易日时递增，并在确认实际建仓时记录该持仓的起始序号。若达到期限但 T+1/跌停无法卖，沿用 12.1 的风险订单顺延。旧 `win_hold_max` 继续只作兼容字段，并在文档中标注不参与此新规则，避免悄悄把默认 240 分钟激活为强制卖出。

### 12.3 `_rebalance`：风险减仓可真正成交

当前流程的 `if target > 0 and current_weight > 0 and abs(delta) < deadzone_th: return` 应变为：

```text
if action == DECAY_REDUCE:
    target = min(target, current_weight)
    if not risk_reduce_bypass_deadzone and abs(target - current_weight) < deadzone_th:
        记录 deadzone_skip；返回
    计算目标股数、可卖股数；若可卖差额不足 risk_reduce_min_shares：记录 min_lot_skip / t1_lock
else:
    保持原有普通调仓死区；SELL 仍不受死区限制
```

实际取整、T+1 和跌停检查继续由 `_execute_sell_to_target` 做；不要为减仓新增账户持仓修改入口。`TradingStateMachine._on_holding` 的 `last_reduce_bar` 最好在**确认减仓成交**后更新，或改成“本次减仓事件已发出/待执行”的独立状态，避免信号虽被死区或空价挡住，状态机却以为已经完成减仓。锚定目标仍为 `min(actual_weight, base_weight × reduce_step_ratio)`；连续 Bar 反复出现同一风险档位时不重复生成相同卖单。

### 12.4 `on_bar` / `_on_flat`：成交确认后的冷静期

在 `TradingStateMachine.reset` 中清空 `last_confirmed_exit`。`BacktestEngine._execute` 从 `sig.metrics["exit_cause"]` 读取策略原因，经 `_rebalance` → `_execute_sell_to_target` → `_execute_sell` 向下传递；T+1 或跌停顺延时同时存入 `PendingTarget`，成交后再写进独立策略原因字段。推荐由 `_execute_sell` 在调用 `Account.sell` 后检查该标的是否真正归零，再调用新增的 `state_machine.on_confirmed_exit(symbol, ts, exit_cause)`；被拒、部分卖出和仅发出信号都不写入该状态。`on_confirmed_exit` 保存交易日序号与原因，`on_bar` 在当前有效 Bar 收盘同步检查冷静期。对已确认的风险减仓可另设 `on_confirmed_reduce` 更新 `last_reduce_bar`，与清仓回调分开。

`_on_flat` 先判断是否还在冷静期，再选择首次入场的 `th_es_entry` 或确认清仓后的 `th_es_reentry`。让 `entry_gates`、`entry_all`、`generate_target_weights` 接受可选 `entry_threshold`，`_scores` 将同一阈值传进去；禁止出现 `BUY` 携带 `target_weight=0`，或 `HOLD` 携带正的可成交目标。若测试不需要区别卖出原因，可先对所有确认清仓统一执行冷静期；后续再区分硬止损与普通 XS 卖出，不要在首轮加入更多自由参数。

### 12.5 `RealDataLoader` 与报告：数据覆盖、兼容日志

`data/real_loader.py::_load_stock_history` 当前强制 `stock_history.csv` 同时存在 `float_shares`。改成必需列 `symbol, trade_date, is_st`，`float_shares` 可选且缺失时填 NaN；`_load_kline` 的 `float_market_cap` 相应保持 NaN。按生效日期向后匹配的规则不变，要求 ST 状态生效日期不晚于被使用的 K 线日期。`_load_market_table` 保留 `index_min`、`breadth_min` 缺失即 `None` 的研究行为，但把文件存在、首尾时间、有效覆盖率写入 `ds.meta` 或研究报告诊断。`indicators/feature_engine.py::_environment_factors` 保留 MRS 缺失为 NaN；正式验证在计算完成后检查覆盖率，达不到预先设定门槛就终止该场实验。

`scripts/research_backtest_three.py` 建议把 `run_backtest(out_dir)` 改为 `run_backtest(chunk_dir, result_dir, params, st_mode)`：B～E 只读原有季度 Parquet 缓存，结果写入新的实验目录；F 因输入数据变化，重新生成季度特征。新增 `--params-json`、`--chunk-dir`、`--result-dir`、`--st-mode`、`--data-quality-mode`，默认行为保持旧命令兼容。各组保存 `resolved_config.json`、`summary.json`、`trade_log.csv`、`order_events.csv` 和权益曲线。`summary.json` 新增实际成交延迟、过期/拒单/死区跳过计数以及明确命名的账户盈亏；保留旧指标键，避免 HTML 报告与旧分析脚本失效。`strategy_cause` 作为新列或订单事件字段，不改旧 `reason` 取值。

这里有两个必须同时处理的接线问题：

1. `prepare_chunk` 目前固定实例化 `AssumedNonSTLoader`。F 组应按 `st_mode` 选加载器；`historical` 模式还需验证每只股票从首日到末日有已生效的 ST 记录，不能在股票历史文件缺失时静默切回假设模式。
2. `prepare_chunk` 当前只存 `kline/features/industry`，`run_backtest` 最终构造的 `DataSlice` 也只有这三类表。F 组新增 `prepare_market_chunk(quarter)`，在该季度三只股票的 `kline` chunk 都写好后，按 `st_mode` 选加载器并调用 `load_slice(SYMBOLS, quarter_start, quarter_end, skip_tick=True)`；该调用的行情轴覆盖三股联合时间轴。核对它与既有 chunk 时间轴的并集一致，再将独立市场表各保存一次为 `index_min`、`breadth`。不能只保存第一只股票的市场表，因为该股票停牌时其他股票的有效分钟可能被漏掉。最终汇总时去重、校验并传入 `DataSlice(index_min=..., breadth=...)`。否则特征工程可能已经有新的 `MRS`，最终 `TradingStateMachine._build_eval_table` 却仍拿不到 `index_close/index_vwap`，大盘跳水否决依旧失效。市场表缺失时研究模式保持 `None` 并明示，严格模式报错。

当前 Q1 预热起点受 `max(START, q_start - 90 天)` 限制，2023Q1 实际没有区间前预热。若能取得 2022 年资料，F 组应让 `prepare_chunk` 从 `q_start - 90 天` 加载，再只写入 2023–2024 正式区间；若取不到，报告各因子的首个有效日期，并在正式评估中明确初段不可交易的规则。此改动属于数据变化实验，不能悄悄并入 B～E。

## 13. 具体测试清单与预期断言

| 测试文件 | 建议用例 | 关键断言 |
| --- | --- | --- |
| `tests/test_backtest_engine.py` | `test_empty_1130_bar_carries_reduce_to_1300` | 11:30 没有成交和新决策，13:00 使用 13:00 开盘价执行 11:29 减仓；`decision_ts` 保持 11:29。 |
| 同上 | `test_entry_expires_across_lunch` | 11:29 买入在 11:30 空价后到期，13:00 不买；订单事件写 `expired`。 |
| 同上 | `test_limit_down_and_t1_keep_one_risk_order` | 跌停或 T+1 锁定时仅保留一份卖出目标；解锁后只卖剩余股数，账户不超卖。 |
| 同上 | `test_reduce_bypasses_normal_deadzone_only_when_enabled` | 同一约 15%→11.5% 的减仓在基线模式被跳过、D 模式真实成交；普通 `ADD` 仍受 8.31 个百分点死区约束。 |
| 同上 | `test_no_lookahead_for_deferred_liquidity` | 13:00 开盘成交只使用此前已完成的量或保守兜底，不能读取 13:00 本 Bar 的 `amount`。 |
| `tests/test_signals.py` | `test_hard_stop_precedes_reversal` | 同一 Bar 反包条件和硬止损同时满足时只发 `SELL`，目标权重为 0，并保留 `exit_cause`。 |
| 同上 | `test_stop_disabled_preserves_original_xs_decision` | 两个新止损参数为 `None` 时，原 XS 路径与既有用例一致。 |
| 同上 | `test_cooldown_starts_after_confirmed_flat` | 卖出仅发信号或被拒时仍视为持仓；真正清仓后当日不再入场，下一允许日才启用再入场门槛。 |
| 同上 | `test_reentry_gate_and_target_weight_agree` | 未达到 `th_es_reentry` 或处于冷静期时返回 `HOLD` 且目标 0；达到门槛且硬过滤通过才有正目标。 |
| `tests/test_real_loader.py` | `test_st_history_without_float_shares` | 只有 ST 生效日期时仍能产生 `is_st` 与涨跌停价，流通市值为 NaN；未来日期的 ST 值不倒灌。 |
| 新增 `tests/test_research_backtest_three.py` | `test_market_tables_survive_chunk_assembly` | 三股季度 chunk 的联合时间轴进入市场表；最终 `DataSlice.index_min/breadth` 存在，评估表能得到 `index_close/index_vwap`，构造的跳水场景能触发否决。 |
| 同上 | `test_historical_st_and_strict_data_mode` | `st_mode=historical` 不调用 `AssumedNonSTLoader`；无历史 ST 或无指数/广度时严格模式明确失败，不静默回退到假设模式。 |

测试先用很小的合成 Bar 覆盖关键路径，随后用三股的季度缓存做 B～E 对照。B 组通过标准是订单生命周期正确、原有账本约束成立、旧成果可追溯；C～E 组通过标准是机制确实按规格触发，并完整报告收益与成本的好坏，不预设必须提高收益率。

## 14. 建议的实施提交顺序

1. 先保存现有 2023–2024 产物及代码/数据指纹，建立小型合成用例。
2. 实施 B：订单生命周期、空 Bar 和诊断日志；运行单元测试及三股“仅撮合修正”回测。
3. 以 B 为固定基线实施 C1/C2/C3：分别加入成本止损、回撤退出、最长持仓期限，每项单独出报告。
4. 以 B 为固定基线实施 D：仅让风险减仓绕开普通死区。
5. 以 B 为固定基线实施 E：分别测试确认成交后的冷静期与再入场阈值；完成单项对照后再测试组合。
6. 获得可信历史数据后实施 F；按事先固定的样本外设计做最终评估。

每一步先核对测试与账户守恒，再查看收益。任何实验若减少 2023 年损失但明显削弱 2024 年收益、增加换手或在独立标的上失效，都应保留完整结果，而不是仅覆盖最初报告。
