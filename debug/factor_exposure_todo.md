# 因子暴露与收益归因：尚未完成的工作

> 基于 2026-09-23 当前工作区代码检查。本文件只记录待实施方案；本次没有修改业务代码，也没有运行回测或测试。此前 `debug/debug.md` 是更广泛的项目问题记录，本文件专门讨论因子暴露与归因。

## 先明确当前实现到了哪里

`analytics/attribution.py` 已有 `AttributionEngine.attribute()`：读取三项入场因子 `global_mod`、`chain_mod`、`agent_ms`，按其绝对值占比分摊**已平仓交易的已实现盈亏**，并生成 `summary` 和柱状图。`analytics/performance.py` 的 Dashboard 能接收外部传入的归因汇总。`AttributionEngine.compute_ic()` 另做因子与前瞻收益的相关性分析；IC 不是持仓暴露或收益归因。

这尚不是完整可用的真实回测因子暴露功能。现有分摊是描述性的入场分数分摊，并未计算组合在每个时点的因子暴露，也不能证明这些因子在统计上产生了相应收益。以下按实现依赖顺序记录缺口。

## 1. 真实 `Signal` 列表中的因子值没有被归因函数读到（先修）

**位置**

- `strategy/signals.py`：`Signal.metrics` 存放 `global_mod`、`chain_mod`、`agent_ms`；`Signal.to_dict()` 保留嵌套的 `metrics`。
- `strategy/signals.py`：`TradingStateMachine.to_frame()` 把信号转成表，但当前只把 `metrics` 作为一列保留，没有把内部键展开成顶层因子列。
- `analytics/attribution.py`：`_as_signals_frame()` 对信号列表调用上述转换；`AttributionEngine.attribute()` 随后用 `e.get("global_mod")` 等读取**顶层列**，取不到时得到 NaN，整笔进入 `other`。
- `tests/test_analytics.py`：当前归因测试手工构造了顶层因子列，未覆盖真实 `Signal` 对象的输入格式。

**应该怎么做**

1. 在 `analytics/attribution.py` 增加统一输入规范化函数。对 `Signal` 列表和含 `metrics` 字典列的 DataFrame，按行展开需要的三个因子；若本来就是顶层列则直接采用。不要原地修改调用方的表。
2. 明确字段名映射：本模块以小写 `global_mod`、`chain_mod`、`agent_ms` 为内部标准；若未来接受 `StreamLogger` 的 `Global_Mod` 等字段，必须显式转换，不能靠猜测列名。
3. 对缺列、非数值、NaN、三项全零分别记录原因。`other` 保留为可审计的未归因分类，不要把缺失静默写成零暴露。

**验收**：用真实 `Signal` 对象列表调用 `attribute()`，在因子有值且成交匹配时，三项 `entry_*` 有值；直接传已展开 DataFrame 得到相同结果；测试不能只构造手工顶层列。

## 2. 决策信号与下一 Bar 的实际成交没有可靠关联（先修）

**位置**

- `engine/backtest.py`：信号在 Bar 收盘产生，放入 `pending_signals`，下一 Bar 开盘才执行；执行还可能被拒绝、缩量或顺延。`engine.generated_signals` 能拿到决策，但成交日志只记录成交时刻 `ts`、股票和交易方向，没有来源信号标识。
- `analytics/metrics.py`：`closed_trades()` 输出的 `entry_ts` 是实际买入成交时间。
- `analytics/attribution.py`：按 `signal.timestamp == trade.entry_ts`、股票相同且动作是 `BUY/ADD` 做精确匹配。

**原因和影响**

正常情况下决策时间早于成交时间，二者不会相等。若成交时刻恰好又有同股票的新 `ADD` 信号，现有查询甚至可能误认这条新信号为本次成交来源。靠“找上一根 Bar 的信号”也不可靠：会遇到拒单、多个信号、T+1 顺延和交易日边界。

**应该怎么做**

1. 在 `strategy/signals.py` 的 `Signal` 或引擎入队层，为每条可执行决策分配回测内唯一且稳定的 `signal_id`，同时保存 `decision_ts`。
2. 在 `engine/backtest.py` 将 `signal_id` 和 `decision_ts` 从 `pending_signals` 传到撮合结果、`TradeLog` 行和 `OrderReport`。每笔真实成交要有独立 `fill_id`；被拒单也记录相同来源 ID，便于解释为何没有建立暴露。顺延单若继续存在，要保留其原始来源及后来覆盖/取消的信息。
3. 在 `analytics/attribution.py` 用成交日志的来源 ID 关联对应信号的因子快照。严格匹配成功的成交才参与该信号的归因；匹配失败的成交归入 `other` 并统计匹配失败数量。不要再用成交时间等于决策时间做主键。
4. 如需兼容旧日志，可提供单独的“尽力匹配”模式，但结果必须标记为低置信度；默认的真实回测报告只接受明确的 ID 关联。

**验收**：信号在 10:00、成交在 10:01 时正确关联 10:00 的因子；10:00 的买单被拒绝时不产生入场暴露；同股票连续发出多个信号时不会误配；跨日或顺延成交能追溯原始决策。

## 3. 加仓、部分卖出与多个买入批次的暴露被压成最早一笔（先修）

**位置**

- `analytics/metrics.py` 的 `closed_trades()`：维护 FIFO 买入批次，但返回的每笔卖出交易只有最早批次的 `entry_ts`；盈亏按整体加权平均成本计算。
- `analytics/attribution.py` 的 `attribute()`：每笔平仓只取一个入场信号快照，把整笔盈亏按该快照分给三个因子。

**原因和影响**

一次卖出可能对应多个不同时间、不同因子暴露的买入/加仓批次。只取最早 `entry_ts` 会把后续加仓的收益和风险都算在最早因子上；部分卖出后剩余买入批次还应保留，供后续卖出使用。

**应该怎么做**

1. 在 `analytics/metrics.py` 把现有的买卖配对逻辑扩展为可复用的**卖出－买入批次匹配表**：每行至少包含 `sell_fill_id`、`buy_fill_id`、`buy_signal_id`、`matched_shares`、买入/卖出时间、归属盈亏。原有 `closed_trades()` 可以在此基础上汇总，避免绩效与归因各维护一套不同的配对规则。
2. 保持项目当前的加权平均成本盈亏口径：先计算每笔卖出的已实现盈亏（含买卖费用），再按此次卖出匹配到各买入批次的股数比例分配给批次；每个批次再根据其自己的入场因子分摊。这样各批次盈亏之和必须精确等于原卖出盈亏。若未来改为纯 FIFO 成本法，要同时更新账户、绩效和归因口径。
3. 买入/加仓被部分成交时，仅已成交股数创建批次；同一买入信号若分多次成交，每次成交单独建批次或共享信号 ID 但保留独立 `fill_id`。卖出只消耗实际可卖和实际成交的股数；未平仓批次不计入**已实现**归因。

**验收**：买入 A、加仓 B、部分卖出、再卖出剩余仓位时，A/B 各使用自己的因子快照；各次卖出及全期间的已归因盈亏与 `closed_trades()` 完全相等；重复买入、部分成交和 T+1 顺延不会重复计算股数。

## 4. 常规回测流程没有生成或保存归因结果（先修）

**位置**

- `main.py` 的 `run_pipeline()`：已有 `trade_log`、`equity_curve` 和 `engine.generated_signals`，但只调用 `analytics.metrics.evaluate()`、绘制价格图和做状态检查；未调用 `AttributionEngine.attribute()`。
- `analytics/performance.py` 的 `PerformanceAnalyzer.plot_report()`：`attribution_summary` 为可选参数，主流程没有传入。
- `analytics/attribution.py` 的 `plot_attribution()`：需外部手动调用。

**应该怎么做**

1. 完成问题 1～3 的数据契约后，在 `main.py` 的 `engine.run()` 后、绩效评估阶段调用 `AttributionEngine.attribute(trade_log, engine.generated_signals)`；如果新接口需要批次匹配表，则传该表与信号映射。此流程只运行一次回测，不另建引擎。
2. 输出逐批次/逐笔归因表、因子汇总表、未匹配明细和覆盖率。例如将可配置的报告目录设为 `analytics/reports/`，文件名含模式、日期范围、参数/数据签名，避免不同实验互相覆盖；生成文件按项目约定加入 `.gitignore`。
3. 在主流程日志显示：已实现盈亏、已归因盈亏、`other` 金额、成功匹配的买入成交比例和未平仓数量。`other` 比例高时给出显式告警；不要仍输出“归因完成”。
4. 将汇总结果传给 `PerformanceAnalyzer.plot_report()` 或调用 `AttributionEngine.plot_attribution()`。报告需显示本次采用的分摊口径和数据覆盖率；无已平仓交易时应输出空结果及原因，不制造零贡献图。

**验收**：`main.py --data smoke` 运行后能得到可追溯的归因表和图；同一批次真实数据回测也能用现有 `engine.generated_signals` 接通。所有文件对应同一个回测实例，归因合计与交易统计的已实现盈亏一致。

## 5. 目前没有“组合随时间变化的因子暴露”（若目标是完整因子暴露，需补）

**位置**：当前 `analytics/attribution.py` 只处理平仓盈亏和入场快照；`engine/backtest.py` 的净值曲线只汇总现金与总持仓市值，没有每个 Bar 的逐股票持仓快照。

**应该怎么做**

1. 新增 `analytics/exposure.py`（建议命名 `ExposureAnalyzer`）。输入为每个 Bar 收盘时的实际股票持仓、账户总权益和同一时点已可见的三项因子值；输出每个时间点、每个因子的组合暴露及股票层贡献。
2. 在 `engine/backtest.py` 的 Bar 收盘盯市后记录或流式输出 `ts, symbol, shares, close, market_value, total_equity`。也可以从成交日志重放仓位，但必须与 `Account` 的 T+1、部分成交和费用口径核对。高频真实数据优先按日期分块输出，避免长期保留全量持仓长表。
3. 在 `analytics/exposure.py` 先定义清晰的**信号加权暴露**口径，例如 `portfolio_exposure[f,t] = Σ_i (position_market_value[i,t] / total_equity[t]) × standardized_factor[f,i,t]`。统一因子方向、归一化区间、缺失值处理及空仓值；宏观因子若对所有股票相同，也要说明它对总仓位的含义。
4. 通过 `engine.generated_signals` 的每 Bar 指标快照或同源评估表，按 `ts + symbol` 关联实际持仓。信号产生于该 Bar 收盘后，暴露时间点必须标为收盘，不能拿收盘因子解释当日开盘的成交决定。
5. 输出暴露时间序列、均值/绝对值/峰值、按股票贡献和图表。若用户要的是标准风险模型中的“因子 beta/因子收益贡献”，还需另取市场因子收益、估计模型与基准；当前三项策略分数的加权和不能直接称为统计因子 beta。

**验收**：空仓时暴露为 0；买入成交后才出现仓位暴露，卖出后消失；部分卖出按实际市值降低暴露；逐股票贡献之和等于组合暴露；不同时间点不会使用未来因子。

## 6. 归因名称、方向与覆盖率需要讲清楚

**位置**：`analytics/attribution.py` 的 `_exposure_weights()` 用 `abs(factor_value)` 归一化，再乘以该笔交易的盈亏；`README.md` 将此功能称为“因子暴露分解归因”。

**应该怎么做**

1. 报告和 API 文档把现有计算明确命名为“按入场分数绝对值分摊已实现盈亏”。负的因子值会得到正的分摊权重；亏损则按负盈亏分摊。这是描述性分摊，不能解释因子的因果贡献或方向收益。
2. 逐笔结果同时保留原始**有符号**因子值、绝对值权重及最终分摊盈亏。`other` 明确分成信号未匹配、因子缺失、因子全零等原因，汇总报告给出金额和股数/交易数覆盖率。
3. 如果业务想回答“哪个因子带来了多少超额收益”，另行定义因子收益估计、暴露与基准的统计归因方法，并在数据满足条件后实现；不要把现有占比分配直接改名为 Brinson 或因子收益贡献。

**验收**：例子 `(-0.5, 0.3, 0.2)` 的原始方向保留，但现有分摊权重解释为 `(0.5, 0.3, 0.2)`；报告展示公式、覆盖率与 `other` 原因；亏损和零/缺失值的处理可复核。

## 建议修改的文件清单

| 文件/目录 | 具体工作 |
| --- | --- |
| `strategy/signals.py` | 为交易决策提供稳定 `signal_id`、`decision_ts`；保持因子快照可供成交关联。 |
| `engine/backtest.py` | 将信号 ID 传入成交日志/订单回报；给每次成交 `fill_id`；按需输出逐 Bar 的逐股票实际持仓。 |
| `analytics/metrics.py` | 在现有已实现盈亏口径下输出卖出与买入批次的匹配明细，并供绩效与归因共用。 |
| `analytics/attribution.py` | 正规化嵌套 `metrics`；通过 ID 关联真实成交；按买入批次分摊；输出未匹配原因、覆盖率与清晰的口径说明。 |
| `analytics/exposure.py`（新增） | 计算并汇总组合逐 Bar 因子暴露与逐股票贡献。 |
| `main.py` | 在常规回测后调用归因/暴露分析，保存表格和图，并将汇总传给 Dashboard。 |
| `analytics/performance.py` | 展示归因覆盖率与暴露时间序列；避免把缺数据的图表当作零贡献。 |
| `tests/test_analytics.py`、`tests/test_backtest_engine.py` | 用真实 `Signal` 对象、下一 Bar 成交、拒单、加仓、部分卖出和空仓情形验证关联及盈亏守恒。 |
| `README.md`、`.gitignore` | 区分“入场分数盈亏分摊”和“动态组合暴露”；说明输出文件及口径，忽略生成的报告目录。 |

## 完成标准

- 常规回测无需手动组装信号表，就能输出可靠的逐笔归因、因子汇总和匹配覆盖率。
- 每笔被归因的盈亏能追溯至**实际成交的买入批次**及其原始决策信号；决策时间与下一 Bar 成交时间不同仍能关联，拒单不产生虚假暴露。
- 所有已平仓批次的归因金额，加上明确列出的 `other`，严格等于同一次回测的已实现盈亏。
- 若宣称已实现“因子暴露”，还需有基于**实际持仓**的逐时点暴露序列；报告清楚标明它是策略分数暴露还是统计风险因子暴露。
