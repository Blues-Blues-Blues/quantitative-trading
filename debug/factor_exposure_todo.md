# 因子暴露与收益归因：AI 实施任务书

> 规格版本：1.0，2026-09-27。问题定位最初记录于 2026-09-23，2026-09-27 已重新核对相关接口。本次仅更新文档，以下功能尚未据此实施或测试。`debug/debug.md` 是历史项目审查，不是本任务的额外实施范围。
>
> **交给执行 AI 的要求**：先核对当前代码，再按本文完成实现、指定测试与文档更新。已有等价实现应复用，不能重复建设或回退用户修改。下文“实施规格 A～H”是确定的接口与验收要求，前六节说明问题位置和原因；实现细节未规定的部分可自行决定，不得自行改变统计口径或扩大范围。环境阻塞须如实报告，不能将未执行的验证写成通过。

## 必须完成的范围与边界

- 必须完成：真实 Signal 输入规范化、信号与成交 ID 关联、含加仓/部分卖出的已实现盈亏分摊、三项现有因子的收盘组合暴露、主回测报告、指定测试及 README 更新。动态暴露是本次必做项。
- 因子固定为 `global_mod`、`chain_mod`、`agent_ms`。已实现盈亏分摊与动态组合暴露是两种不同输出。
- 不实现：统计因子 beta、因子收益回归、Brinson、因子中性化、新策略、新数据源、新的部分成交撮合模型。
- 不改变：交易准入、买卖时点、资金分配、成本滑点、T+1、目标仓位规则、已实现盈亏的平均成本口径。新增元数据和分析不能改变原交易序列。
- 默认不提交 Git、不清理用户已有文件、不运行真实回测或参数寻优。代码实施时允许执行本文列出的短测试和冒烟命令；当前更新文档的任务不执行这些命令。

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

1. 按实施规格 A，在引擎入队层为决策分配回测内唯一且稳定的 `signal_id`，并保存 `decision_ts`；不得通过当前时钟或随机数生成决策 ID。
2. 在 `engine/backtest.py` 将 `signal_id` 和 `decision_ts` 从 `pending_signals` 传到撮合结果、`TradeLog` 行和 `OrderReport`。每笔真实成交要有独立 `fill_id`；被拒单也记录相同来源 ID，便于解释为何没有建立暴露。顺延单若继续存在，要保留其原始来源及后来覆盖/取消的信息。
3. 在 `analytics/attribution.py` 用成交日志的来源 ID 关联对应信号的因子快照。严格匹配成功的成交才参与该信号的归因；匹配失败的成交归入 `other` 并统计匹配失败数量。不要再用成交时间等于决策时间做主键。
4. 旧日志无来源 ID 时统一记为 `other/unmatched_signal`，仍可计算已实现盈亏；本次不实现猜测时间的匹配模式。

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
2. 按实施规格 F 输出逐批次/逐笔归因表、因子汇总表、未匹配明细和覆盖率，默认写入 `analytics/reports/` 的独立运行目录，生成文件加入 `.gitignore`。
3. 在主流程日志显示：已实现盈亏、已归因盈亏、`other` 金额、成功匹配的买入成交比例和未平仓数量。`other` 比例高时给出显式告警；不要仍输出“归因完成”。
4. 将汇总结果传给 `PerformanceAnalyzer.plot_report()` 或调用 `AttributionEngine.plot_attribution()`。报告需显示本次采用的分摊口径和数据覆盖率；无已平仓交易时应输出空结果及原因，不制造零贡献图。

**验收**：`main.py --data smoke` 运行后能得到可追溯的归因表和图；同一批次真实数据回测也能用现有 `engine.generated_signals` 接通。所有文件对应同一个回测实例，归因合计与交易统计的已实现盈亏一致。

## 5. 目前没有“组合随时间变化的因子暴露”（本次必须完成）

**位置**：当前 `analytics/attribution.py` 只处理平仓盈亏和入场快照；`engine/backtest.py` 的净值曲线只汇总现金与总持仓市值，没有每个 Bar 的逐股票持仓快照。

**应该怎么做**

1. 新增 `analytics/exposure.py`，类名固定为 `ExposureAnalyzer`。输入为每个 Bar 收盘时的实际股票持仓、账户总权益和同一时点已可见的三项因子值；输出每个时间点、每个因子的组合暴露及股票层贡献。
2. 按实施规格 E/F，在引擎收盘盯市后通过可选回调发送实际持仓快照，报告层按交易日分块输出；本次不另写一套账户重放实现。
3. **信号加权暴露**固定为 `portfolio_exposure[f,t] = Σ_i (position_market_value[i,t] / total_equity[t]) × factor_value[f,i,t]`。直接使用现有因子的带符号数值，不增加标准化、截尾、取绝对值或对有效股票重新归一化。缺失值和空仓规则见实施规格 E。
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
- 必须提供基于**实际持仓**的逐时点暴露序列，明确命名为“策略分数加权暴露”，不能命名为统计因子 beta。

---

## 实施规格 A：标识、时间与日志字段

### A1. ID 规则

1. 保留现有执行顺序。引擎按实际入队顺序为全部 Signal（含 HOLD）分配 `sig_00000001`、`sig_00000002` 等递增字符串；不得为了 ID 排序而改变多股票下单优先级。相同数据、参数和输入顺序再次运行，应得到相同 ID 和交易结果。
2. `Signal` 末尾新增 `signal_id: Optional[str] = None`，保留现有位置参数调用；入队时复制 Signal 再赋 ID，不改调用方对象。外部已提供 ID 时保留，并检查回测内唯一性；自动生成时跳过已占用 ID。重复外部 ID 抛 `ValueError`。
3. 每个 `shares > 0` 的成交分配 `fill_00000001` 等独立 ID；拒单及零成交记录的 `fill_id` 为空。所有撮合日志行增加递增的 `event_seq`，用于相同时间戳内保持实际执行顺序。
4. `decision_ts` 始终等于原 Signal.timestamp，不随重试改变；`ts` 始终是此次成交/尝试的时刻。ID 唯一性限定在一次引擎运行内，跨运行以报告目录中的 `run_id` 区分。
5. 顺延目标从裸权重改为含 `target_weight, signal_id, decision_ts` 的结构，保留现有覆盖/取消规则。单次订单跨多次部分成交时共享 signal_id，但每次成功成交的 fill_id 不同；不能因补元数据改写撮合规则。

### A2. 字段契约

| 表 | 必须新增或保留的字段 | 类型和约束 |
| --- | --- | --- |
| 信号规范表 | `signal_id, timestamp, decision_ts, symbol, action` | ID 字符串；两个时间相同；时间统一为当前项目无时区的上海本地时间；symbol 字符串 |
| 信号规范表 | `global_mod, chain_mod, agent_ms` | float64，可缺失；另保留输入非法原因 |
| TradeLog | 原有 13 列 + `event_seq, fill_id, signal_id, decision_ts` | 原有列顺序不变，新增列追加；event_seq int64；ID 可空；无 decision_ts 为 NaT |
| OrderReport | 原字段 + 同一组来源字段 | 新字段放 dataclass 末尾并有兼容默认值 |
| 批次匹配表 | `symbol, sell_fill_id, buy_fill_id, buy_signal_id, entry_ts, exit_ts, matched_shares, allocated_pnl` | 每行是一笔卖出匹配一个买入批次；股数正整数，盈亏 float64 |

实现时将 `analytics/metrics.py` 中依靠 `row[0]`、`row[6]` 等位置读取成交字段的逻辑改为按字段名读取。旧日志没有 event_seq 时，先记录输入行序号，再按 `ts + 原行序号` 稳定处理。旧日志没有 fill_id 时可按此顺序生成 `legacy_fill_...` 作为本次分析内部键，但不得伪造来源 signal_id。重复非空 fill_id、超卖或负成交股数属于不合法输入，分析函数须报错，不能静默截断后宣布盈亏守恒。

## 实施规格 B：信号规范化与公开接口

### B1. 因子字段处理

- 接受三种输入：Signal 列表、带 `metrics` 字典列的 DataFrame、已展开的小写因子 DataFrame；本次不直接支持 StreamLogger 的大写字段。
- 顶层和 metrics 内同名字段均为有效有限数且差异大于 `1e-12` 时抛 `ValueError`；否则使用已有有效值。两边均缺失则 NaN，非数值和无穷值记录相应原因。整个转换不得修改调用方的 DataFrame 或 metrics 字典。
- 空输入返回固定 schema 的空表；有非空重复 signal_id 时抛错。缺失 ID 的旧信号可以规范化，但不能参与 ID 匹配。
- 关联成交时额外核对 symbol、决策 action 属于 BUY/ADD、decision_ts 与成交来源一致且早于成交 ts；冲突视为数据错误并报错，不强行取第一条。

### B2. 公开接口保持与新增

```python
# analytics/metrics.py：共用一次匹配算法，不能各自实现成本计算
match_trade_batches(trade_log) -> pd.DataFrame
closed_trades(trade_log) -> list[dict]  # 原有结构及每 SELL 一条的语义保留

# analytics/attribution.py
AttributionEngine.attribute(trade_log, signals) -> (trades_df, summary_df)
AttributionEngine.attribute_batches(trade_log, signals) -> batches_df
AttributionEngine.quality_report(trade_log, signals, batches_df) -> dict

# analytics/exposure.py
ExposureAnalyzer.compute(positions, factor_values, equity_curve) -> (contributions, exposure_ts)
```

内部可抽取私有函数共用结果；`attribute()` 仍返回二元组，第一张表仍为每 SELL 一行，不能静默改成每买入批次一行。已有字段保留，允许追加字段。`attribute_batches()` 返回带批次关联信息、原始因子、分摊权重、`pnl_*`、`pnl_other`、`other_reason` 的详细表。

逐 SELL 汇总时：`pnl_*`、`pnl_other` 对其批次求和；`entry_*` 为匹配股数加权的有符号入场因子（任一相应批次缺失则该字段为 NaN）。主导 `factor` 以该卖出的“股数加权分摊权重”最大者确定，包括 other，平局按 `global_mod, chain_mod, agent_ms, other` 顺序取首个。summary 保留 `factor, pnl, weight, n_trades`：n_trades 计该因子作为主导分类的卖出笔数，不能把批次数当成交易笔数；weight 沿用因子 PnL/总 PnL，总 PnL 为零时 NaN，图表不得据此画成百分比饼图。

## 实施规格 C：成本、盈亏分摊与数值算例

### C1. 确定的计算规则

对每只股票维护持仓股数 Q、含买入费用的成本 C，以及保存 buy_fill_id/buy_signal_id/剩余股数的 FIFO 队列。

- 买入：`Q += shares`；`C += amount + commission + transfer_fee`。
- 卖出 q 股：卖出前平均成本 `c = C / Q`；`P = amount - commission - stamp_duty - transfer_fee - q*c`；之后 `Q -= q`、`C -= q*c`。
- FIFO 只确定此卖出消耗哪些来源批次。匹配批次 b 得到 `P_b = P * matched_shares_b / q`，不按该批次自身原买价重算盈亏。它是平均成本盈亏的来源分配，不是独立 FIFO 成本收益。
- 三项因子均为有限值且绝对值之和 > `1e-12` 时：`a_f = abs(x_f) / sum(abs(x))`，`pnl_f = P_b*a_f`；保留原始因子的正负号供展示。任何一项缺失/非法，或三项全零，该批次的全部 P_b 进入 other，不对剩余有效因子重新归一化。
- 全过程不将金额提前四舍五入到分；最后一条批次/因子分配可承接浮点余差。金额断言使用 `abs(error) <= 1e-6 元`，因子权重断言使用 `1e-12`。显示金额可四舍五入，不覆盖原始表格值。

### C2. 无费用基准例：避免误改为 FIFO 成本

买入 A：100 股 × 10 元；买入 B：100 股 × 20 元；卖出 100 股 × 16 元。平均成本 15 元，已实现盈亏 **100 元**；FIFO 来源批次为 A，所以 A 分配到 100 元。不能返回按 A 原始成本算出的 600 元。

### C3. 含费用、加仓与两次部分卖出的完整例

下列费用直接作为单测日志输入，是固定账本算例，不要求等于当前 ExecutionCost 默认费率。各交易在满足 T+1 的不同日期发生。

| 事件 | 股数 | 价格 | amount | commission | stamp_duty | transfer_fee | 因子 (global, chain, agent) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 买入 A | 200 | 10 | 2000 | 5 | 0 | 1 | (-0.5, 0.3, 0.2) |
| 加仓 B | 100 | 20 | 2000 | 5 | 0 | 1 | (0.2, 0.5, 0.3) |
| 卖出 S1 | 150 | 16 | 2400 | 5 | 0.5 | 0.5 | 不使用退出因子 |
| 卖出 S2 | 150 | 14 | 2100 | 5 | 1 | 1 | 不使用退出因子 |

- 两次买入后：Q=300，C=4012，平均成本=4012/300。
- S1 净回款=2394；扣除成本=2006；P1=**388**。消耗 A 的 150 股，归属 A 的盈亏=388。余下 Q=150、C=2006，来源批次 A 50 股、B 100 股。
- S2 净回款=2093；扣除成本=2006；P2=**87**。A 50 股分配 29，B 100 股分配 58；交易后 Q=0、C=0。
- A 累计分配盈亏=417，B=58。最终因子 PnL：**Global=220.1，Chain=154.1，Agent=100.8，other=0**，总和=**475 元**。
- `closed_trades()` 和 `attribute()` 的 trades_df 都有两笔卖出；批次明细有三行（S1-A、S2-A、S2-B），不能把交易笔数改成 3。

## 实施规格 D：未归因原因和覆盖率

`other_reason` 固定为 `unmatched_signal`、`missing_factor`、`non_numeric_factor`、`non_finite_factor`、`zero_factors`，成功行为空。同一行有多类异常时，按上述顺序选首个；缺少某因子键、None、NaN 均为 missing_factor，正负无穷为 non_finite_factor。重复 ID、矛盾字段等结构错误按规格 B 报错，不归为普通缺失。

quality.json 至少包含：

- `matched_buy_fill_rate`：能通过来源 ID 关联有效 BUY/ADD 信号的买入成交笔数 / 全部正股数 BUY/ADD 成交笔数，无买入时为 null。
- `attributed_sold_share_rate`：具有完整有效因子的已卖出匹配股数 / 全部已卖出匹配股数，无卖出时为 null。
- `attributed_abs_pnl_rate`：有效批次的 Σabs(P_b) / 全部批次的 Σabs(P_b)，分母为零时 null。不能用净总收益作覆盖率分母，因为盈亏会抵消。
- 总已实现盈亏、各因子及 other 的合计、按 other_reason 汇总的行数/股数/金额、剩余未平仓股数、归因守恒误差。
- 有卖出且 attributed_sold_share_rate 小于 1 时状态 `partial` 并告警；无卖出为 `no_closed_trades`；有效归因覆盖完整且守恒时为 `complete`。结构错误或守恒失败直接报错，不能输出 complete。

## 实施规格 E：动态组合暴露

### E1. 时间、输入与计算

在当前 Bar 的实际成交处理完毕、收盘盯市后，使用 Account 的真实持仓和同一 Bar 的因子快照计算。该时点新产生但尚未成交的 BUY/SELL 不能改变暴露。输入因子直接取引擎同源评估表三项列；无评估表的外部信号模式可取当前 Bar 的 Signal.metrics，没有快照则缺失，不能用未来信号填充。

输入 schema：

- positions：`ts, symbol, shares, mark_price, market_value, total_equity`；只记录非零实际持仓，mark_price 使用 Account 当时的盯市价，不补未来报价。
- factor_values：`ts, symbol, global_mod, chain_mod, agent_ms`，每个 `(ts,symbol)` 唯一。
- equity_curve：至少 `ts, total_equity`，含空仓 Bar，提供完整时间轴。

对每个因子 f 和 Bar t：`w_i=market_value_i/total_equity_t`；`contribution_i=w_i*x_i`。使用有符号原值，不新增缩放、截尾或标准化，不把权重重新归一化到 1。总权益非有限或 <=0 时分析报错。

缺失规则按因子独立处理：

- 无持仓：`exposure=observed_exposure=0`、`coverage=1`、`status=flat`。
- 有持仓：`coverage=有效因子的持仓市值/全部持仓市值`；`observed_exposure` 只求有效贡献之和。覆盖率低于 `1-1e-12` 时 `exposure=NaN`、`status=partial`；覆盖完整则 `exposure=observed_exposure`、`status=complete`。
- 持有股票的因子缺失，其 contribution 为 NaN，不是零。因子数值本身为 0 则为有效零暴露。

输出：contributions 为 `ts,symbol,factor,shares,market_value,position_weight,factor_value,contribution` 长表；exposure_ts 为 `ts,factor,exposure,observed_exposure,coverage,status,invested_weight` 长表。空仓 Bar 仍须为三个因子各输出一行。

### E2. 精确数值例

权益 1000 元，A 持仓市值 200、B 为 300、现金 500；A 因子 `(0.5,-0.2,0)`，B 因子 `(-0.1,0.4,0.8)`。三个组合暴露依次为 **0.07、0.08、0.24**，invested_weight=0.5，coverage 均为 1。不能把结果乘 2 去掉现金影响。

若仅 B 的 global_mod 缺失：global 的 observed_exposure=0.1、coverage=200/500=0.4、exposure=NaN、status=partial；chain 与 agent 结果不变。空仓时三个暴露均为 0。

## 实施规格 F：主流程、输出与性能

### F1. 开关与调用

- `main.py --data smoke` 和常规 `--data real` 默认启用归因与暴露报告；增加 `--no-analysis-report` 关闭报告，以及 `--report-dir` 指定根目录（默认 `analytics/reports`）。不影响独立 IC 分析入口。
- `BacktestEngine` 末尾新增可选 `snapshot_sink=None` 回调，入参为当前 ts、持仓表、当时权益及因子表；None 时不创建持仓快照长表、因子分析表或报告文件。ID 记录仍正常进行。
- 新增 `analytics/reporting.py` 管理报告生命周期与按日分块；main 创建 writer、传入回调，在回测结束后 finalize。异常时 manifest 标记 failed；不能重新调用 engine.run() 获取分析数据。
- `StrategyOptimizer.backtest()`、每个 Optuna trial、Walk-Forward 内部回测默认 `snapshot_sink=None`，不生成详细报告。暂不为寻优结果自动增加报告流程。

### F2. 文件和元数据

每次运行独立目录：`<report-dir>/<mode>_<start>_<end>_<run_id>/`。run_id 使用 UTC 时间加随机短串，仅作为报告命名空间；它不参与 signal_id/fill_id 或计算。同输入的数值结果和引擎内 ID 必须可复现，run_id/生成时间不作相等比较。不覆盖已有目录。

固定产物：

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | schema_version=1、运行模式/区间/股票、策略和执行参数、initial_cash、已解析行业映射摘要、数据来源/已有数据指纹、相关代码内容哈希、方法名、状态及生成时间 |
| `attribution_trades.csv` | 每 SELL 一行，保留原统计口径 |
| `attribution_batches.csv` | 每卖出与买入批次匹配一行，含关联 ID 和分摊结果 |
| `attribution_summary.csv` | 三因子及 other 的汇总 |
| `unattributed_batches.csv` | other 批次及原因，可为空表但保留列 |
| `quality.json` | 规格 D 的覆盖率和守恒诊断 |
| `positions/date=YYYY-MM-DD.parquet` | 当天实际持仓快照 |
| `exposure_contributions/date=YYYY-MM-DD.parquet` | 当天逐股票因子贡献 |
| `exposure_timeseries/date=YYYY-MM-DD.parquet` | 当天每 Bar、每因子的组合暴露及覆盖率 |
| `attribution.png`、`exposure.png`、`dashboard.png` | 归因柱状图、每日收盘暴露/覆盖率图、传入归因汇总的现有 Dashboard |

CSV 使用 UTF-8，空值留空；JSON 的 NaN/Inf 变为 null 并保留质量标志，不输出非标准 JSON。方法名分别为 `average_cost_fifo_source_allocation`、`signed_score_equity_weighted_exposure`。manifest 中不可取得的来源指纹应标为 unavailable，不伪称内容哈希已验证。

### F3. 性能与边界

- writer 只缓存一个交易日新增的持仓/贡献/暴露数据；跨日及 finalize 时写 Parquet。不在原有 generated_signals 之外再保留一份全期间持仓表，也不为分析重跑特征工程。
- 绘图只使用每日收盘摘要，完整分钟序列保留在分块文件；关闭报告时没有这些分析开销。
- 空仓日也写三因子的完整 exposure 时间轴；无平仓时 CSV 返回固定 schema、quality 状态 no_closed_trades，图上说明无已实现盈亏。最终持仓不为归因而强制平仓。
- 将默认报告目录加入 `.gitignore`；用户自定义目录不自动修改 Git 配置。不读取、覆盖或删除历史报告。

## 实施规格 G：文件职责与执行顺序

1. `strategy/signals.py`、`engine/backtest.py`：实现 ID 与来源传播，追加列且保持原执行顺序；验证拒单/顺延来源。
2. `analytics/metrics.py`：抽取一次共享匹配算法，保留 closed_trades/evaluate 的合法输入结果。该层不导入归因模块，避免循环依赖。
3. `analytics/attribution.py`：实现规范化、ID join、批次分配、兼容的逐 SELL 输出与质量报告。
4. `analytics/exposure.py`：实现纯分析接口，完成数值/缺失值契约；不要在模块导入时创建目录或产生输出。
5. `analytics/reporting.py`、`main.py`、`analytics/performance.py`：实现按日 writer、开关、CSV/Parquet/JSON/绘图和 Dashboard 接入。
6. `tests/test_analytics.py`、`tests/test_backtest_engine.py`、新增 `tests/test_exposure.py` 与 `tests/test_analysis_reporting.py`：落实以下验收矩阵；README 和本文件底部状态同步。

## 实施规格 H：必做验收与允许执行的命令

| 编号 | 输入/操作 | 必须得到的结果 |
| --- | --- | --- |
| T01 | 同一份嵌套 Signal、metrics DataFrame、展开 DataFrame | 因子与归因结果相同，输入未被修改 |
| T02 | 同名顶层/metrics 因子矛盾；重复 signal_id/fill_id | 明确 ValueError，不静默取第一条 |
| T03 | 10:00 发 BUY、10:01 成交，10:01 另有不同因子的 ADD | 成交关联原 BUY；不使用同时间新 ADD |
| T04 | BUY 被拒；同一信号多次成交；卖出跨日顺延 | 拒单无 fill_id/无新增持仓暴露；成交 ID 不同、来源相同；顺延 decision_ts 不变 |
| T05 | 规格 C2 与 C3 的固定账本 | 盈亏分别 100 与 475；C3 归因为 220.1/154.1/100.8，2 笔 SELL、3 个批次匹配 |
| T06 | 缺失/非法/全零因子，旧日志缺 ID，且正负 PnL 抵消 | 对应 other_reason 正确；不猜信号；守恒成立，覆盖率不以净收益作分母 |
| T07 | 规格 E2、部分卖出、未成交 BUY、最后仍有持仓 | 暴露公式与缺失结果精确匹配；只按实际持仓，未实现盈亏不被归入已实现归因 |
| T08 | 两天的小型真实 Signal→引擎→报告链路 | 分日 Parquet、四张 CSV、两个 JSON 和三张图存在；第二天回调前上一天缓冲已清空；来源可追溯 |
| T09 | 同一小型回测重复执行；报告开/关分别执行 | 不同 run_id 目录；signal_id/fill_id、成交数量/价格/费用、curve 和原 metrics 一致 |
| T10 | 默认引擎 snapshot_sink=None；no-analysis-report | 不调用分析 writer、不写报告目录、不创建全量持仓长表 |
| T11 | 无交易、无卖出、空仓日；读取输出文件 | 固定 schema、明确状态、JSON 合法；暴露时间轴完整，无虚假零收益完成声明 |

现有归因单测应改用真实 Signal 和明确来源 ID，不能继续仅用 `_signals_for()` 人为把成交时间写成决策时间来证明链路正确。原有指标、状态机和撮合测试仍须通过；预期改动仅限新元数据、分析输出与明确标记旧日志缺失，不得放宽断言掩盖交易结果变化。

实施后在项目根目录执行：

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_analytics.py tests/test_backtest_engine.py tests/test_signals.py tests/test_exposure.py tests/test_analysis_reporting.py -q
& '.\.venv\Scripts\python.exe' main.py --data smoke
```

本任务不运行 `run_optimization.py`、真实数据回测、Walk-Forward 寻优或整个 `tests/test_optimizer.py`。需要验证优化器未启用报告时，用模拟依赖/极小固定数据单点调用测试，不执行 optimize。测试失败仅在具体失败修复后重跑相关项；权限或依赖阻塞如实记录，不修改无关权限、成本参数或交易规则以求通过。

## 交付清单与当前状态

- [x] 来源 ID、成交 ID 和规范化输入完成，旧接口保持兼容。
- [x] 批次匹配与两个数值算例通过，原绩效口径不变。
- [x] 动态组合暴露及缺失覆盖率通过固定算例。
- [x] main 默认报告、关闭开关、按日输出和无交易边界完成。
- [x] 指定测试及冒烟通过；README 写明公式、输出、局限和运行方式。
- [x] 最终说明列出修改文件、实际执行的命令、测试数量及报告路径；未完成项保留未勾选，环境阻塞写明原因。

截至 2026-09-27，本任务已完成实现、指定测试及合成数据冒烟。本文前面的问题描述和规格保留为实施前基线；以下记录本次实际交付结果。

### 修改文件

| 文件 | 本次修改 |
| --- | --- |
| `strategy/signals.py` | 兼容地追加 signal_id；序列化携带原始 decision_ts。 |
| `engine/backtest.py` | 按原入队顺序复制/编号，成交与拒单来源传播，event_seq/fill_id，顺延来源及取消/覆盖信息，可选收盘快照回调。 |
| `analytics/metrics.py` | 按字段名读日志，共享平均成本盈亏和 FIFO 来源配对；验证重复 ID、负股数与超卖。 |
| `analytics/attribution.py` | 嵌套/展开输入规范化、严格 ID 关联、批次分摊、兼容每 SELL 汇总、原因与覆盖率、守恒检查。 |
| `analytics/exposure.py`（新增） | 有符号权益加权暴露、逐股贡献、逐因子缺失覆盖率及完整空仓时间轴。 |
| `analytics/reporting.py`（新增） | 独立运行目录、按日缓冲/Parquet、四张 CSV、两个 JSON、三张图、失败生命周期、暴露摘要统计。 |
| `analytics/performance.py` | Dashboard 接入覆盖率和日末暴露，无平仓提示；修正回撤显示范围和日期标签。 |
| `main.py` | 默认报告、关闭开关、自定义目录、参数/来源元数据、同次引擎结果 finalize。 |
| `tests/test_analytics.py` | 真实 Signal 流水线，C2/C3 固定账本，输入格式/冲突/缺失/覆盖率/平均成本守恒等。 |
| `tests/test_backtest_engine.py` | next-bar 来源、拒单、确定性 ID、部分成交和跨日顺延、覆盖取消、关闭快照。 |
| `tests/test_exposure.py`（新增） | E2 精确值、部分卖出、空仓、逐因子缺失、未来因子隔离与非法权益。 |
| `tests/test_analysis_reporting.py`（新增） | 两日真实信号链路、开关及重复运行一致性、文件/schema、空仓/未平仓、失败状态及 JSON。 |
| `README.md` | 两类输出的公式、统计口径、局限、文件、开关及短验证命令。 |
| `.gitignore` | 忽略默认 analytics/reports/，未修改自定义目录配置。 |
| `debug/factor_exposure_todo.md` | 保留用户已有规格，本节登记交付状态。 |

`tests/test_signals.py` 未修改，其原有 97 项测试通过。未修改优化器、交易准入、下单规则、费用滑点或 T+1；未提交 Git。

### 实际执行的验证与结果

以下是实际执行记录。首次完整指定命令出现问题后，仅修复并复跑相关文件；后续为新增边界和图表修改运行对应测试，没有把 collect-only 当成测试通过。

```powershell
# 首轮：186 passed，1 failed，8 errors。
& '.\.venv\Scripts\python.exe' -m pytest tests/test_analytics.py tests/test_backtest_engine.py tests/test_signals.py tests/test_exposure.py tests/test_analysis_reporting.py -q
# 修复 Pandas 空 ID 类型及非整数股数测试构造后：41 passed。
& '.\.venv\Scripts\python.exe' -m pytest tests/test_analytics.py -q
# 新增边界及报告修正后：92 passed。
& '.\.venv\Scripts\python.exe' -m pytest tests/test_analytics.py tests/test_backtest_engine.py tests/test_analysis_reporting.py -q
# 图表、分区 schema 与输入校验修正后：66 passed。
& '.\.venv\Scripts\python.exe' -m pytest tests/test_analytics.py tests/test_exposure.py tests/test_analysis_reporting.py -q
# 有限大因子值的稳定分摊算例：45 passed。
& '.\.venv\Scripts\python.exe' -m pytest tests/test_analytics.py -q
# 最后修正有符号因子均值的中间乘法溢出后：24 passed。
& '.\.venv\Scripts\python.exe' -m pytest tests/test_analytics.py::TestAllocationContract -q
# 仅核对最终用例清单：202 tests collected，不执行测试。
& '.\.venv\Scripts\python.exe' -m pytest tests/test_analytics.py tests/test_backtest_engine.py tests/test_signals.py tests/test_exposure.py tests/test_analysis_reporting.py --collect-only -q
# 两次执行均退出码 0；最后一次使用最终业务代码并写出下述独立目录。
& '.\.venv\Scripts\python.exe' main.py --data smoke
# 无空白错误。
git diff --check
```

最终指定范围共 **202 个不同用例**，均已有通过记录（不是把不同轮次的 passed 相加）：

| 文件 | 用例数 |
| --- | ---: |
| `tests/test_analytics.py` | 45 |
| `tests/test_backtest_engine.py` | 38 |
| `tests/test_signals.py` | 97 |
| `tests/test_exposure.py` | 12 |
| `tests/test_analysis_reporting.py` | 10 |

首轮 8 个初始化错误共享同一原因：Pandas 字符串推断把拒单的 None ID 变为 NaN；已通过 object 类型显式保留空 ID。1 个失败发生在测试向整数列写入 1.5 时，现先构造浮点列，再验证分析函数拒绝非整数股数。没有放宽业务断言。

- C2：平均成本已实现盈亏 100 元，没有误用 FIFO 成本得到 600 元。
- C3：2 笔 SELL、3 条批次匹配；盈亏 388 + 87 = 475 元；因子分摊 220.1 / 154.1 / 100.8 元。
- E2：暴露 0.07 / 0.08 / 0.24，投入权益比例 0.5；缺失例 observed=0.1、coverage=0.4、exposure=NaN。
- T09：重复运行及报告开/关的 ID、日志、净值曲线、原绩效指标精确一致。
- T10：默认引擎不构造快照；main 关闭报告时不创建 writer 或目录。现有 analytics fixture 仅调用固定参数的合成数据 `StrategyOptimizer.backtest()`，确认 snapshot_sink=None，未调用 optimize。

### 最终冒烟产物

目录：`analytics/reports/smoke_2024-01-02_2024-01-09_20260927T100944017552Z_936c64fc/`

已核对四张 CSV、两个合法 JSON、三张 PNG，以及三个数据目录各六个交易日的 Parquet 分区。Dashboard 已做图像检查。

- 60 根 Bar，120 个信号，4 笔成交、2 笔平仓；原五项状态检查全部 PASS。
- 已实现盈亏与分摊合计均为 **16533.128588628228 元**，other=0，守恒误差=0。
- 买入成交关联率、已卖出股数覆盖率、绝对盈亏覆盖率均为 1；status=complete，最终未平仓股数=0。
- 三因子的暴露时间轴均为 60 行，完整保留在分日文件；图表只使用日末点。
- 冒烟数据无来源文件指纹，manifest 明确标记 unavailable；代码内容哈希已写入。

### 未完成项与验证边界

本任务规定的实施项无未完成项，无测试环境阻塞。未运行真实数据回测、`run_optimization.py`、Optuna optimize、Walk-Forward 寻优或整个 `tests/test_optimizer.py`；未宣称完成这些验证。未实现任务明确排除的 beta、因子收益回归、Brinson、新数据源或新的部分成交模型。
