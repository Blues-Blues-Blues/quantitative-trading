# 代码审查问题记录（2026-09-20）

审查范围：全项目静态检查 + `python main.py --data smoke` / `--data real` 实跑 + `pytest tests/`。
结论：单测 217 项全过、真实数据链路可跑通，但存在 **3 个主要问题**（其中 P2 会掩盖 P1、P3 这类问题）与若干次要问题。

| 编号 | 问题 | 严重度 | 状态 |
| --- | --- | --- | --- |
| P1 | 冒烟测试实际 FAIL，被陈旧信号缓存掩盖 | 高 | ✅ 已修复并验证 |
| P2 | 信号缓存无代码版本号，静默复用旧代码的信号 | 高 | ✅ 已修复并验证 |
| P3 | `real_loader` 空表崩溃，报错与真实原因无关 | 中 | ✅ 已修复并验证 |
| P4 | `SearchSpace.is_feasible` 未接入约束机制，越界参数可当选「最优」 | 中 | 📝 已记录，暂不修复 |

修复与验证记录见文末「附：修复与验证记录」。

---

## P1. 冒烟测试实际不通过（被陈旧缓存掩盖）

### 出现部分

- 剧本定义：[main.py](../../main.py) 模块 docstring 第 8~14 行（`剧情设计` 段）
- 决策链：`strategy/signals.py` 的 `TradingStateMachine._on_holding`（清仓判定）+ `SignalSynthesizer._compute_xs` / `_compute_veto`
- 触发点：`python main.py --data smoke`

### 现象

清掉 `data/feature_cache/signals_*.pkl` 后用当前代码重跑，直接非零退出：

```
19:30:20 [INFO] 状态机输出：120 个信号 → {'BUY': 2, 'ADD': 8, 'SELL': 0, 'HOLD': 96}
19:30:21 [INFO] 回测撮合：60 根 Bar 净值点；成交 2 笔 / 拒绝 0 笔
19:30:21 [INFO] 绩效评估：年化Sharpe=2.819 最大回撤=0.97% 总盈亏=+0 元 有效交易=0 笔 胜率=nan%
19:30:22 [FAIL] 存在 ≥1 笔完整平仓交易（实际 0 笔，成交 2 笔）
冒烟测试未通过，请查看上方 FAIL 项
```

对照：缓存命中时同一命令「通过」（`BUY 2 / ADD 8 / SELL 0` + 4 笔平仓）。**即缓存掩盖了回归。**

### 原因

剧本停留在「旧二值闸门」时代，与现行为连续评分决策链不匹配：

1. **剧本假设**：`main.py` docstring 写明「D5~D6 转熊：每 Bar 超大单主动卖（Inst_Flow<0 → 状态机退出 S_push → 触发 SELL）」。
2. **实际行为**：退出由连续 `XS` 决定，`SELL` 的判据是 `xs <= th_xs_exit`（默认 `-0.3`）或一票否决。
3. **熊段算不出足够低的 XS**：
   - mock 只翻了逐笔流方向（`final_ms` 的 4 个合成分量里只动了 1 个），`final_ms` 仍 ≈ +0.28~0.45
   - `capital_purity` ≈ 0（实测约 1e-11，接近精确抵消）
   - 高水位回撤 `dd` 仅 0.004~0.034
   - → `XS = 0.5×0.29 + 0.3×0 - 0.2×0.004 ≈ 0.14`，**远高于 `-0.3`，永不触发 SELL**
4. **一票否决也救不了**：`_compute_veto` 的「游资溃逃」腿要求 `big_flow < 0 且 retail_chase > th_retail_chase(0.65)`；熊段只造了超大单主动卖、没有小单，`retail_chase = 0` → 该腿恒不成立。
5. **附带**：`BUY` 目标 0.1173 → `ADD` 目标 0.1637，Δ = 0.046 < 调仓死区 `DEADZONE_TH = 0.05`，被撮合引擎全部跳过，因此 120 个信号最终只成交 2 笔。

实测熊段关键分量（600000）：

```
ts                   final_ms   capital_purity   big_flow   retail_chase   state
2024-01-08 09:30:00  0.5168     -8.97e-11        -1.0       0.0            S_noise
2024-01-08 15:00:00  0.3060     -8.97e-11        -1.0       0.0            S_noise
```

### 解决方法

三选一（推荐 B）：

- **A. 加重熊段力度**：让 D5~D6 同时满足 `final_ms` 转负与 `retail_chase` 超阈（补小单主动买、抬高 `retail_chase`），并可让 `capital_purity` 转负。改动集中在 `mk_flow_ticks` / `mk_broaden` 等 mock 构造函数。
- **B. 更新剧本描述 + 重设期望值**：承认 `main.py` docstring 的「退出 S_push → SELL」已随二值闸门废除而失效，改写为「XS 跌破清仓线 → SELL」，并同步调整 mock 让熊段能让 `XS <= th_xs_exit`。
- **C. 放宽冒烟断言**：允许「无平仓」通过。**不推荐** —— 会失去「闭环完整性」这一核心回归能力。

同步须修正 README 第 119 行「输出 5 项 PASS 状态检查」的表述（当前代码下不成立）。

---

## P2. 信号缓存无版本号，静默复用旧代码的信号

### 出现部分

- `main.py` 的 `_signals_cache_path`（第 73~82 行）：缓存键 = `label | start | end | symbols | params`
- `main.py` 的 `run_pipeline` 第 390~407 行：命中即 `pickle.load` 直接复用，不做任何有效性校验
- 对照物：`indicators/feature_engine.py` 的特征缓存**有** `_CACHE_SCHEMA_VERSION` / `_ALIGN_VERSION` 参与文件名，信号缓存**没有**任何版本维度

### 现象

实测命中的缓存文件与代码修改时间明显错位：

| 缓存文件 | 生成时间 | 对应源码修改时间 |
| --- | --- | --- |
| `signals_101291e4e6c5fd0d.pkl`（冒烟） | 2026/8/25 | `strategy/signals.py` 2026/9/8 |
| `signals_3641b6d3a8495cbc.pkl`(真实, 603019) | 2026/8/27 | 同上 |

即：**`signals.py` 改完之后，两次「跑通」跑的都是旧代码产出的信号。**

同一真实区间/参数（2024-01-02~2024-12-31, 603019）差异巨大：

| 信号来源 | 状态机输出 | 成交 | 平仓 | Sharpe | 最大回撤 | 总盈亏 |
| --- | --- | --- | --- | --- | --- | --- |
| 旧缓存（8/27） | BUY 9 / ADD 346 / SELL 8 | 17 笔 | 8 笔 | 1.403 | 3.40% | +35940 |
| 当前代码重算 | BUY 3 / ADD 1 / SELL 2 | 6 笔 | 3 笔 | 0.024 | 4.07% | +18235 |

重算一次信号耗时 **299 s**（58070 行），因此该缓存失配的正确性代价很高，且不会报任何错。

### 原因

缓存键只覆盖「回测配置」，不覆盖「产出该结果的代码」。任何对 `strategy/signals.py`、`indicators/*`（影响特征→信号链路）的修改，只要区间/标的/参数不变，就会命中旧 pickle。P1 之所以长期未被发现，正是这个缺陷导致的。

### 解决方法

给缓存键加一个**代码版本或内容指纹**维度，任选：

- **A（推荐，成本最低）**：在 `strategy/signals.py` 顶部声明 `_SIGNAL_CACHE_VERSION = "3.1"`，改动决策逻辑时手动 bump，并把它拼进 `_signals_cache_path` 的 key（与 `feature_engine._CACHE_SCHEMA_VERSION` 的做法保持一致）。
- **B（自动）**：对 `strategy/signals.py` + 参与合成链的 `indicators/*.py` 做 `hashlib.sha256(文件内容)`（或 `os.stat().st_mtime_ns`），取前 16 位作为 key 的一部分，彻底免维护。
- **C（兜底）**：`pickle.load` 后校验缓存内附带的版本字段，不匹配则重算并覆盖。

补充建议：缓存写入目前发生在状态检查之前（`run_pipeline` 第 400~407 行），**冒烟失败时也会落盘缓存**。建议把写缓存挪到 `_status_checks` 全部 PASS 之后，避免把坏结果固化。

---

## P3. `real_loader` 空表崩溃，报错与真实原因无关

### 出现部分

- `data/real_loader.py` 的 `_load_kline`（第 164~191 行），崩溃点在**第 177 行** `kd["date"] = kd.index.normalize()`
- 上游：`data/l2_loader.py` 读取 parquet（依赖 `pyarrow` / `fastparquet`）

### 现象

用未安装 `pyarrow` 的解释器执行 `python main.py --data real`：

```
[WARNING] 行情 文件 20241224.行情.parquet 读取失败（Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'. ...），跳过
...（每个交易日重复数百条）
[WARNING] data1 无 ['603019'] 在 2024-01-02~2024-12-31 的行情数据，返回空表
Traceback (most recent call last):
  File "main.py", line 490, in run_real
  File "data/real_loader.py", line 177, in _load_kline
    kd["date"] = kd.index.normalize()
AttributeError: 'RangeIndex' object has no attribute 'normalize'
```

环境事实：默认解释器 `python`（3.14.3）**无 pyarrow**；`.venv\Scripts\python.exe` 有（pyarrow 25.0.1 / pandas 3.0.5）。用 venv 解释器执行则正常。

### 原因

1. `_load_kline` 假定 `l2.load_kline()` 返回的 `k` 一定带 `DatetimeIndex`，但空表时 pandas 给的是 `RangeIndex` → `.normalize()` 不存在。
2. 上游对空表只打了 WARNING 就放行（`返回空表`），未做「关键表为空 → 显式失败」的短路，与 README 宣称的「缺表自动降级」预期不符：**主 kline 为空是致命情形，不属于可降级范围**。
3. 真实根因（缺 parquet 引擎）被完全淹没在几百条 WARNING 里，最终报错指向一个无关的 `AttributeError`，排查成本高。

### 解决方法

- **A. 前置显式校验（推荐）**：在 `RealDataLoader.load_slice` 入口先做 `import pyarrow`（或 `pyarrow | fastparquet` 至少之一）的能力探测，缺失时直接 `raise RuntimeError("缺少 parquet 引擎，请 pip install pyarrow")`。
- **B. 空表短路**：`_load_kline` 开头加

  ```python
  if k is None or k.empty:
      raise ValueError(
          f"data1 无 {symbols} 在 {start}~{end} 的行情数据，"
          f"请检查数据目录与 parquet 引擎（pyarrow/fastparquet）")
  ```

  避免 `RangeIndex` 下抛 `AttributeError`。
- **C. 文档**：README 明确「运行本项目请使用已安装依赖的解释器（如 `.venv`）」，并把 `pyarrow` 列入快速开始的前置检查。

---

## P4. `SearchSpace.is_feasible` 未接入约束机制，越界参数可当选「最优」

> 状态决策：**仅记录，暂不修复**（2026-09-21 确认）。作为已知问题保留，修复方案已列出供后续处理。

### 出现部分

- 文档声明：[optimizer/search_space.py](../../optimizer/search_space.py) 第 7~9 行——「W_NORTH 越出 `[0, 0.3]` 的 trial 由约束机制判为不可行」
- 可行性判断：[optimizer/search_space.py](../../optimizer/search_space.py) 的 `is_feasible`（第 252 行）——仅被 `run_optimization.py` 日志与单测调用，**不参与任何选优决策**
- 约束链路：[optimizer/bayesian_opt.py](../../optimizer/bayesian_opt.py) 的 `constraints_func`（225 行）只转发 `analytics/metrics.py::constraint_violations`（**5 项回测指标**：回撤/胜率/盈亏比/笔数/年化换手），不含搜索空间可行性
- 选优出口：[optimizer/bayesian_opt.py](../../optimizer/bayesian_opt.py) 的 `best()`（291~293 行）、`top_candidates()`——可行过滤口径 = 约束违反量全 0，同不含 `is_feasible`

### 现象

`run_optimization.py --data real --seeds 43 --trials 3` 实跑中：

- Trial 0 的 `W_NORTH = 0.357`，超出搜索空间声明上限 `0.3`，日志自标 `SearchSpace 可行=False`
- 但该 trial 的 5 项回测指标约束全部满足（违反量 = 0），被 TPE 视为「合规最优」，`best()` 照样选中并进入验证段/测试段终评

### 原因

「搜索空间边界」与「回测指标约束」是两条独立的过滤逻辑，但文档（search_space.py 第 8 行）声称前者由后者兜底——**该声明当前为假**。权重和 = 1 由归一化推导保证（不会崩溃），但单个权重的取值范围没有任何环节强制，`constraint_violations` 也完全不感知参数本身。

### 影响评估（中，不致命）

1. 不破坏回测正确性：净值、指标、`权重和 == 1` 断言均正确，越界参数只是超出文档声明的语义范围。
2. 损害寻优可信度：TPE 持续向 W_NORTH > 0.3 的非法区域采样且视作「通过约束」；越界 W_NORTH 通常意味着更高风险敞口（高换手/高回撤/高收益），「最优」会被系统性偏向激进参数。
3. 影响随 trial 数放大：3 trial 时仅挤占 1/3 采样预算；20+ trial 全量寻优时非法区域会持续吃掉预算，收敛结果被带偏。

### 解决方法（未执行，供后续）

二选一即可，改动几行：

- **A. `constraints_func` 追加第 6 项约束**：按 `is_feasible` 计算越界量（越界 → 非 0），TPE 采样直接惩罚非法区域。
- **B. 选优前置过滤**：`best()` / `top_candidates()` 的候选集先按 `is_feasible` 过滤，再套用现有 5 项硬约束。

注意：`constraints_func` 按 trial 调用、`is_feasible` 需 `params["weights"]`，两者配合时避免重复构造参数（可复用 `_params_and_metrics` 的产物）。

### 对当前寻优结论的标注

本次真实寻优仅 3 trial，且最优解落在声明搜索空间之外（见上），**其参数与终评结果（Sharpe 0.9494 / +5391 元）均「仅供参考」**，不等同于已通过合法性筛检的可用参数。另叠加验证段候选退化、训练/测试活跃度差异大等可信度问题，全量寻优结论须在修复本问题后重新产出。

---

## 次要问题

1. **真实模式出图文件名仍带 `smoke_` 前缀**：`main.py` 的 `_plot_smoke_charts`（第 563~580 行）硬编码 `f"smoke_{sym}.png"`，两种模式共用该函数。建议按 `label` 传前缀。
2. **冒烟日志「2 只股票（?）」**：`run_pipeline` 取 `ds.meta.get("source", "?")`，而 `build_smoke_slice` 的 `meta` 未设 `source`。补一个 `"source": "smoke_mock"` 即可。
3. **死代码仍保留**（全项目零引用，README 已标注「待清理」）：
   - `strategy/base.py`（仅 1 行 docstring）
   - `strategy/sentiment_strategy.py`（仅 1 行注释）
   - `strategy/state_machine.py`（仅 1 行注释）
   - `strategy/position.py`（5.2 KB，无任何 import）
   - `tests/test_state_machine.py`（仅 1 行注释，pytest 收集 0 项）
4. **Git 暂存区与工作区不一致**：`debug/IMP_1-3_improvements.md`、`debug/P0.1_test_analytics.md`、`debug/factor_formulas_refactored.md` 为 `AD` 状态（已 `git add` 但工作区已删除）；`debug/factor_formulas_refactored.docx` 已 add 且仍在盘上；`README.md` / `main.py` 的改动尚未 add。提交前需统一。
5. **`requirements.txt` 缺 `pytest`**：`python -m pytest tests/` 在默认解释器下会 `No module named pytest`。
6. **拒绝日志噪音**：真实回测打出 74 笔 `t1_lock` 拒绝（`engine/backtest.py` 第 379~383 行）。这是 T+1 顺延的设计行为（每 Bar 记录一次），但与「成交 6 笔」并列展示容易误读为大量失败，建议聚合计数或在日志中标注「顺延中」。

---

## 附：本次实跑证据汇总

| 检查项 | 命令 / 解释器 | 结果 |
| --- | --- | --- |
| 单元测试 | `.venv\Scripts\python.exe -m pytest tests/ -q` | **217 passed**（49.9 s；末尾 exit 1 为沙箱写文件限制，与测试无关） |
| 冒烟模式（缓存命中） | `python main.py --data smoke` | 表面通过（4 笔平仓），**数据来自 8/25 旧缓存** |
| 冒烟模式（清缓存） | `python main.py --data smoke` | **FAIL：0 笔平仓，非零退出** |
| 真实模式（缓存命中） | `.venv\Scripts\python.exe main.py --data real` | 5 项 PASS；8 笔平仓 / Sharpe 1.403 / +35940（**8/27 旧缓存**） |
| 真实模式（清缓存） | `.venv\Scripts\python.exe main.py --data real` | 5 项 PASS；3 笔平仓 / Sharpe 0.024 / +18235 / 最大回撤 4.07%；信号重算 299 s |
| 真实模式（无 pyarrow 解释器） | `python main.py --data real` | **崩溃**：`real_loader.py:177 AttributeError` |

## 修复优先级建议（已执行）

1. ~~**P2 先修**~~ —— ✅ 已修（`_SIGNAL_CACHE_VERSION` 进缓存键）
2. ~~**P1 次修**~~ —— ✅ 已修（mock 熊段改为指数跳水触发「大盘跳水」一票否决）
3. ~~**P3 顺手修**~~ —— ✅ 已修（`_ensure_parquet_engine` 前置探测 + 空表短路）
4. ~~次要问题按 4 → 3 → 1 → 2 → 5 → 6 的顺序清理~~ —— 除第 4 项（Git 暂存区）外均已完成

---

## 附：修复与验证记录

修复日期：2026-09-21

### P1 修复要点

`main.py` 的 Mock 构造函数由「只翻逐笔流方向」改为能真正压低连续评分的组合：

| 构造函数 | 改动 |
| --- | --- |
| `mk_index_min` | 熊段指数每 Bar 跳水 -4%、缩量（`vol_f *= 0.85`）；`vwap` 改取近 5 根 close 滚动均值（下跌时滞后于 close）→ 满足 `close < vwap*(1-circuit_index_drop)` |
| `mk_snapshot` | 熊段卖盘量 ×5、买盘量 ×0.2 → OBI/OFSS 转负 |
| `mk_breadth` | 熊段 ADR 2.0 → 0.5、北向净流 +5e7 → -5e8 → MRS 转负 |
| `mk_industry` | 熊段行业资金流自峰值回落至 0 → IRS z-score 转负 |

效果：熊段触发「大盘跳水」一票否决 → `XS` 强制 `-1.0 <= th_xs_exit` → 状态机 SELL。
`main.py` docstring 的剧本描述同步改写为连续 XS 语义，并删除已废除的「退出 S_push → SELL」表述。

### P2 修复要点

- `strategy/signals.py`：新增 `_SIGNAL_CACHE_VERSION = "3.1"`（决策逻辑变化须手动递增）
- `main.py::_signals_cache_path`：`_SIGNAL_CACHE_VERSION` 参与 md5 key
- `main.py::run_pipeline`：冒烟数据（`meta.smoke`）不走信号缓存；**缓存落盘从回测前推迟到状态检查全部 PASS 之后**，避免把坏结果固化
- `README.md`：补充「改动决策逻辑后须手动递增版本号」与「失败不落盘」说明

### P3 修复要点

- `data/real_loader.py`：新增 `_ensure_parquet_engine()`，在 `load_slice` 入口前置探测 `pyarrow` / `fastparquet`，缺失时 `RuntimeError` 明确提示
- `data/real_loader.py::_load_kline`：`k` 为 None/空表时显式 `ValueError` 短路，不再对 `RangeIndex` 调用 `.normalize()`

### 验证结果

| 检查项 | 命令 / 解释器 | 修复前 | 修复后 |
| --- | --- | --- | --- |
| 单元测试 | `.venv\Scripts\python.exe -m pytest tests/ -q` | 217 passed | **217 passed**（55.1 s） |
| 冒烟模式 | `python main.py --data smoke` | **FAIL：SELL 0 / 0 笔平仓，非零退出** | **全链路全部通过 ✅ 5/5 PASS**（3.5 s）；`BUY 2 / ADD 8 / SELL 2`，成交 4 笔 / 2 笔平仓 / Sharpe 10.998 / +16533 元 / 胜率 100% |
| 真实模式（首跑，无信号缓存） | `.venv\Scripts\python.exe main.py --data real` | 旧 key 缓存 8 笔平仓 / Sharpe 1.403 | **5/5 PASS**（403 s）；`BUY 3 / ADD 1 / SELL 2`，成交 6 笔 / 3 笔平仓 / Sharpe 0.024 / 最大回撤 4.07% / +18235 元 |
| 真实模式（缓存命中） | 同上 | — | **5/5 PASS**（73.1 s，提速 5.5×），指标与首跑完全一致 |
| 真实模式（无 pyarrow 解释器） | `python main.py --data real` | **崩溃**：`real_loader.py:177 AttributeError` | 入口即 `RuntimeError: 缺少 parquet 引擎（pyarrow / fastparquet），请先执行 pip install pyarrow` |

### 修复后的缓存状态

- 当前有效信号缓存：`data/feature_cache/signals_17189c324d3db0d7.pkl`（含版本号的新 key）
- 已清理 8 个孤儿信号缓存（旧 key，约 328 MB）
- 保留：11 个特征 parquet（含 10 个旧 `p16` 参数版本，经确认保留以支持新旧参数对比）+ 4 个 Optuna journal

### 后续待办

1. **`debug/` 下 3 个 .md 仍为 `AD` 状态**（已 `git add` 但工作区已删除），提交前需 `git rm --cached` 或恢复文件
2. **Optuna Journal 与代码版本无关联**（与 P2 同源问题）：`opt_study.s42.journal` 内已有 20 个旧代码产出的 COMPLETE trial，`n_trials` 语义为「本次新增数」→ 续跑会与旧 trial 混合排序，建议换 seed 或归档旧 journal
3. **P4 未修复**（`is_feasible` 未接入约束机制）：修复方案见上文 P4 章节，全量寻优须在修复后重新产出——当前 3-trial 寻优结论「仅供参考」