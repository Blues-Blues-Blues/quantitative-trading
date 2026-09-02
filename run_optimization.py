"""机器学习超参数寻优入口（冒烟版）：Optuna 贝叶斯寻优全链路验证。

复用 main.py 的内置 Mock 数据（6 日 × 2 票牛转熊剧情），仅执行 2~3 次
Optuna Trial，验证：
1. 参数注入：SearchSpace.suggest → SignalSynthesizer/MicroStructure/引擎
   （含 Dirichlet 式权重归一化：W_OFSS+W_CPS+W_INST+W_NORTH == 1）
2. 约束检查：TPESampler(constraints_func) 的 4 项硬约束违反量
   （回撤 < 15% / 胜率 > 55% / 盈亏比 > 1.5 / 有效交易 ≥ 30 笔）
3. 收敛图落盘：optimizer_history.png → analytics/pictures/

说明：冒烟数据仅 6 天，"有效交易 ≥ 30 笔"等硬约束在物理上无法满足，
因此所有 trial 都会判为不可行 —— 这正是约束检查机制在正常工作的表现，
best() 会回退为按目标值（年化 Sharpe）最优的 trial 兜底返回。

用法：
    python run_optimization.py
"""

import logging
import os
from typing import Dict, List, Tuple

import optuna

from analytics.metrics import constraint_violations
from data.dataslice import DataSlice
from data.real_loader import RealDataLoader
from main import (
    INITIAL_CASH,
    REAL_SYMBOLS,
    SMOKE_PARAMS,
    SYMBOL_TO_INDUSTRY,
    build_smoke_slice,
)
from optimizer.bayesian_opt import StrategyOptimizer
from optimizer.search_space import SearchSpace

logger = logging.getLogger("run_optimization")

N_TRIALS = 3  # 冒烟版仅跑 2~3 次 Trial
REAL_TRIALS = 20
STABILITY_TRIALS = 8  # 附加稳定性 seed 各自的 trial 数
# 宽松实测硬约束：真实数据当前策略现实（回撤~28%、胜率~30%），
# 默认 15%/55% 会让 TPE 几乎全部采样不可行 → 寻优空转；
# 年化换手硬上限 15（双边成交额/平均权益×244 日），显式压住高频换手成本
REAL_CONSTRAINT_KWARGS = {"max_drawdown": 0.35, "win_rate": 0.30,
                          "pl_ratio": 0.8, "min_trades": 30,
                          "max_turnover_annual": 15.0}
# 合成目标软惩罚（硬约束线内生效，给 TPE 平滑引导，不替代约束）：
#   score = Sharpe - 1.0*max(0, 回撤-0.25) - 0.03*max(0, 年化换手-6.0)
REAL_PENALTY_KWARGS = {"dd_soft": 0.25, "dd_pen": 1.0,
                       "to_soft": 6.0, "to_pen": 0.03}
# 真实数据严格三集：训练（TPE 采样）→ 验证（选 best）→ 测试（终评一次）
TRAIN_START, TRAIN_END = "2023-01-03", "2024-06-28"
VALID_START, VALID_END = "2024-07-01", "2024-09-30"
TEST_START, TEST_END = "2024-10-08", "2024-12-31"


def _build_search_space() -> SearchSpace:
    """窄搜索空间：适配 6 天冒烟数据（chip_window 小 → CPS 更快可用）。"""
    return SearchSpace(
        w_ofss=(0.2, 0.6), w_cps=(0.1, 0.5),
        w_inst=(0.0, 0.4), w_north=(0.0, 0.3),
        win_inst=(1, 3), win_chip_old=(1, 3),
    )


def _print_trial_detail(study: optuna.Study, opt: StrategyOptimizer,
                        search_space: SearchSpace) -> None:
    """逐 Trial 打印：参数注入结果 + 权重归一化校验 + 约束违反量。"""
    logger.info("-" * 78)
    for trial in study.trials:
        if trial.number not in opt._cache:
            # Journal 续跑：该 trial 由先前进程评估，参数已落盘、指标未落盘，
            # 仅打印目标值（指标重算见 opt.best() 的断点恢复路径）
            logger.info("Trial %d（Journal 加载，无进程内缓存）：目标值=%s",
                        trial.number,
                        "NaN" if trial.value is None else f"{trial.value:.4f}")
            continue
        params, metrics = opt._cache[trial.number]  # objective/constraints 同源缓存
        weights = tuple(float(w) for w in params["weights"])
        w_north_derived = params["weights"][-1]
        feasible = search_space.is_feasible(params)
        violations = constraint_violations(metrics, **opt.constraint_kwargs)

        logger.info("Trial %d：目标值(合成=Sharpe−惩罚)=%s",
                    trial.number, "NaN" if trial.value is None else f"{trial.value:.4f}")
        logger.info("    权重注入 (W_OFSS,W_CPS,W_INST,W_NORTH)=%s，和=%s，"
                    "W_NORTH 推导值=%s，SearchSpace 可行=%s",
                    tuple(round(w, 4) for w in weights),
                    f"{sum(weights):.6f}",
                    f"{w_north_derived:.4f}", feasible)
        logger.info("    窗口注入：inst_window=%d chip_window=%d",
                    params["inst_window"], params["chip_window"])
        logger.info("    回测评估：有效交易=%d 笔 胜率=%.0f%% 盈亏比=%s 回撤=%.2f%% "
                    "年化换手=%s 总盈亏=%+.0f 元",
                    metrics["n_trades"],
                    metrics["win_rate"] * 100 if metrics["win_rate"] == metrics["win_rate"] else 0,
                    ("inf" if metrics["profit_loss_ratio"] == float("inf")
                     else f"{metrics['profit_loss_ratio']:.2f}"),
                    metrics["max_drawdown"] * 100, metrics["turnover_annual"],
                    metrics["total_pnl"])
        logger.info("    约束违反量 [回撤, 胜率, 盈亏比, 交易笔数, 年化换手]=%s %s",
                    [f"{v:.1f}" for v in violations],
                    "（全部为 0 = 满足硬约束）" if all(v <= 1e-6 for v in violations)
                    else "（存在违反 → 该 Trial 不可行）")


def run_optimization_smoke() -> None:
    """执行冒烟寻优：2~3 次 Trial + 参数注入/约束打印 + 收敛图。"""
    logger.info("=" * 78)
    logger.info("冒烟寻优启动：%d 次 Optuna Trial（复用 main.py Mock 数据）", N_TRIALS)
    logger.info("=" * 78)

    ds = build_smoke_slice()
    search_space = _build_search_space()

    opt = StrategyOptimizer(
        data=ds,
        search_space=search_space,
        symbol_to_industry=SYMBOL_TO_INDUSTRY,
        account_kwargs={"initial_cash": INITIAL_CASH},
        n_trials=N_TRIALS,
        seed=42,
    )

    # 对照：冒烟显式参数在寻优前先跑一次，确认流水线本身可成交
    metrics, engine = opt.backtest(ds, dict(SMOKE_PARAMS))
    trade_log, _ = engine.run()
    logger.info("基线（SMOKE_PARAMS）回测：成交 %d 笔 / 有效交易 %d 笔，"
                "Sharpe=%s", len(trade_log[trade_log["shares"] > 0]),
                metrics["n_trades"],
                "NaN" if metrics["sharpe"] != metrics["sharpe"] else f"{metrics['sharpe']:.4f}")

    study = opt.optimize(n_trials=N_TRIALS)
    logger.info("Optimize 完成：共 %d 个 Trial", len(study.trials))
    _print_trial_detail(study, opt, search_space)

    # 最优参数与指标（硬约束无可行解时 best() 兜底为按目标值最优）
    best_params, best_metrics = opt.best(study)
    logger.info("-" * 78)
    logger.info("最优参数：weights=%s 阈值=%s 窗口=%s",
                tuple(round(w, 4) for w in best_params["weights"]),
                {k: round(v, 4) for k, v in best_params.items()
                 if k.startswith("th_")},
                {k: v for k, v in best_params.items() if k.endswith("_window")
                 or k == "inst_window"})
    logger.info("最优指标：Sharpe=%s 有效交易=%d 笔 胜率=%.0f%%",
                "NaN" if best_metrics["sharpe"] != best_metrics["sharpe"]
                else f"{best_metrics['sharpe']:.4f}",
                best_metrics["n_trades"],
                best_metrics["win_rate"] * 100 if best_metrics["win_rate"] == best_metrics["win_rate"] else 0)

    # 收敛图落盘（analytics/pictures/optimizer_history.png）
    path = opt.plot_history(study)
    logger.info("收敛图已保存：%s", path)
    logger.info("=" * 78)
    logger.info("冒烟寻优全部通过 ✅（参数注入 / 权重归一化 / 约束检查 / 收敛图）")


def run_optimization_real(n_trials: int = REAL_TRIALS,
                          seeds: Tuple[int, ...] = (42,),
                          stability_trials: int = STABILITY_TRIALS) -> None:
    """真实数据寻优（严格三集隔离 + 合成目标 + top-k 验证复评 + 多 seed 稳定性）。

    训练段 2023-01~2024-06（TPE 采样，目标 = 年化 Sharpe − 回撤/换手软惩罚）→
    验证段 2024-07~09（训练段 top-k 候选复评，据验证段 Sharpe 选 best）→
    测试段 2024-10~12（最终评估一次，不参与任何选择）。

    多 seed 稳定性：seeds[0] 执行完整三集流程；其余 seed 各自独立寻优并
    输出 best 参数对比，检验参数选择是否对随机种子敏感（防单次运气）。
    每个 seed 独立 Journal 文件（opt_study.s{seed}.journal），可断点续跑。
    """
    logger.info("=" * 78)
    logger.info("真实数据寻优启动：主 seed=%s（主 trial 数=%d），"
                "稳定性 seeds=%s（各 %d 个 trial）",
                seeds[0], n_trials, seeds[1:], stability_trials)
    logger.info("严格三集：训练 %s~%s / 验证 %s~%s / 测试 %s~%s",
                TRAIN_START, TRAIN_END, VALID_START, VALID_END,
                TEST_START, TEST_END)
    logger.info("硬约束=%s", REAL_CONSTRAINT_KWARGS)
    logger.info("软惩罚=%s", REAL_PENALTY_KWARGS)
    logger.info("=" * 78)

    loader = RealDataLoader()
    ds_train = loader.load_slice(REAL_SYMBOLS, TRAIN_START, TRAIN_END)
    ds_valid = loader.load_slice(REAL_SYMBOLS, VALID_START, VALID_END)
    ds_test = loader.load_slice(REAL_SYMBOLS, TEST_START, TEST_END)
    ds_train.validate()
    logger.info("三段数据就绪：训练 %d 根 K 线 / 验证 %d / 测试 %d",
                len(ds_train.kline), len(ds_valid.kline), len(ds_test.kline))

    results: Dict[int, Dict[str, object]] = {}
    for idx, seed in enumerate(seeds):
        seg_trials = n_trials if idx == 0 else stability_trials
        logger.info("=" * 78)
        logger.info("开始 seed=%d（%s）：%d 次 Optuna Trial",
                    seed, "主 seed（完整三集）" if idx == 0 else "稳定性 seed",
                    seg_trials)

        opt = StrategyOptimizer(
            data=ds_train,
            # 窗口参数固定为基线(1,1)：特征缓存签名(_params_signature)含窗口，
            # 放开采样会让每种窗口组合触发一次训练段特征全量重算(~10 分钟/次)；
            # 首轮寻优聚焦信号层参数，窗口留待后续专项寻优
            search_space=SearchSpace(win_inst=(1, 1), win_chip_old=(1, 1)),
            # 行业映射必须与实盘回测一致（loader 真实全映射）；用 main.SYMBOL_TO_INDUSTRY
            # 冒烟映射会导致寻优期行业情绪因子失真（600171/600460 不在映射内 → 行业因子退化）
            symbol_to_industry=loader.symbol_to_industry,
            account_kwargs={"initial_cash": INITIAL_CASH},
            n_trials=seg_trials,
            seed=seed,
            use_feature_cache=True,
            constraint_kwargs=dict(REAL_CONSTRAINT_KWARGS),
            penalty_kwargs=dict(REAL_PENALTY_KWARGS),
        )

        if idx == 0:
            # 基线：当前主入口默认参数在训练段的表现（对照组）
            metrics, engine = opt.backtest(ds_train, dict(SMOKE_PARAMS))
            trade_log, _ = engine.run()
            logger.info("训练段基线（REAL 默认参数）：成交 %d 笔 / 有效交易 %d 笔 / "
                        "Sharpe=%s / 回撤=%.2f%% / 年化换手=%s / 总盈亏=%+.0f 元",
                        len(trade_log[trade_log["shares"] > 0]), metrics["n_trades"],
                        f"{metrics['sharpe']:.4f}", metrics["max_drawdown"] * 100,
                        f"{metrics['turnover_annual']:.2f}", metrics["total_pnl"])

        study = opt.optimize(
            n_trials=seg_trials,
            study_name=f"opt_real_s{seed}",
            storage_path=os.path.join(
                "data", "feature_cache", f"opt_study.s{seed}.journal"))
        logger.info("Optimize 完成：seed=%d 累计 %d 个 Trial", seed, len(study.trials))
        _print_trial_detail(study, opt, opt.search_space)

        # 训练段最优（约束无可行解时 best() 兜底为按目标值最优）
        best_params, ts_metrics = opt.best(study)
        results[seed] = best_params
        logger.info("seed=%d 训练段最优：Sharpe=%s 有效交易=%d 笔 胜率=%.0f%% "
                    "回撤=%.2f%% 年化换手=%s",
                    seed, f"{ts_metrics['sharpe']:.4f}", ts_metrics["n_trades"],
                    ts_metrics["win_rate"] * 100,
                    ts_metrics["max_drawdown"] * 100,
                    f"{ts_metrics['turnover_annual']:.2f}")
        opt.plot_history(study)

        if idx == 0:
            chosen, chosen_ts = _select_via_validation(opt, study, ds_valid)
            _final_test_eval(opt, ds_test, chosen)
            results["__chosen__"] = chosen
            logger.info("验证/测试评估完成（测试段仅评估一次，未参与寻优选择）")
        else:
            logger.info("seed=%d（稳定性）仅打印 best，不做完整三集评估", seed)

    _print_stability(results, seeds)
    logger.info("如需落地：将主 seed 的 chosen 参数（或稳定性 seed 的共识区间）"
                "注入 main.py REAL_PARAMS 后重跑真实回测")


def _select_via_validation(opt: StrategyOptimizer, study: optuna.Study,
                           ds_valid: DataSlice, k: int = 5
                           ) -> Tuple[Dict[str, object], Dict[str, float]]:
    """训练段 top-k 候选 → 验证段复评 → 据验证段绩效选 best。

    验证段同时满足硬约束（constraint_kwargs 口径，含换手）为优先条件：
    - 存在可行候选：其中验证段 Sharpe 最高者胜出
    - 无可行候选：回退训练段 top-1 并告警
    返回 (chosen_params, 其训练段 metrics)；测试段评估由调用方完成。
    """
    cands = opt.top_candidates(study, k=k)
    rows: List[Tuple[Dict[str, object], Dict[str, float], Dict[str, float], bool]] = []
    for params, ts_m in cands:
        v_m, _ = opt.backtest(ds_valid, params)
        v_ok = all(float(vv) <= 1e-9 for vv in
                   constraint_violations(v_m, **opt.constraint_kwargs))
        rows.append((params, ts_m, v_m, v_ok))
        logger.info("[top-k] 训练(Sharpe=%.4f) → 验证段 Sharpe=%s 回撤=%.2f%% "
                    "年化换手=%s 有效交易=%d 约束%s",
                    float(ts_m["sharpe"]),
                    f"{v_m['sharpe']:.4f}", float(v_m["max_drawdown"]) * 100,
                    f"{v_m['turnover_annual']:.2f}", int(v_m["n_trades"]),
                    "满足" if v_ok else "违反")
    feasible = [r for r in rows if r[3]]
    if feasible:
        chosen = max(feasible, key=lambda r: float(r[2]["sharpe"]))
        logger.info("验证段选 best：%d 个可行候选中按验证段 Sharpe 最高者 "
                    "(Sharpe=%.4f)", len(feasible), chosen[2]["sharpe"])
    else:
        chosen = rows[0]
        logger.warning("验证段无满足硬约束的候选，回退训练段 top-1")
    return chosen[0], chosen[1]


def _final_test_eval(opt: StrategyOptimizer, ds_test: DataSlice,
                     params: Dict[str, object]) -> None:
    """测试段终评（仅评估一次，不参与任何选择）。"""
    t_metrics, t_engine = opt.backtest(ds_test, params)
    t_log, _ = t_engine.run()
    logger.info("测试段（%s~%s）终评：成交 %d 笔 / 有效交易 %d / "
                "Sharpe=%s / 回撤=%.2f%% / 年化换手=%s / 总盈亏=%+.0f 元",
                TEST_START, TEST_END,
                len(t_log[t_log["shares"] > 0]), t_metrics["n_trades"],
                f"{t_metrics['sharpe']:.4f}",
                t_metrics["max_drawdown"] * 100,
                f"{t_metrics['turnover_annual']:.2f}",
                t_metrics["total_pnl"])


def _print_stability(results: Dict[int, Dict[str, object]],
                     seeds: Tuple[int, ...]) -> None:
    """跨 seed 稳定性：每个 seed 的 best 关键参数 + 参数取值区间。

    区间跨度大 = 该维度对随机种子敏感（辨识弱），选择/解释参数时应谨慎。
    """
    logger.info("=" * 78)
    available = [s for s in seeds if s in results]
    if len(available) < 2:
        logger.info("稳定性检验：仅 %d 个 seed，跳过区间对比", len(available))
        return
    keys = ["weights", "th_es_entry", "th_xs_exit", "th_xs_reduce_high",
            "w_xs_ms", "w_xs_purity", "base_weight", "deadzone_th",
            "win_decay_grace", "reversal_add_mult", "base_decay_rate"]
    logger.info("跨 seed best 参数对比（%s）：", ",".join(map(str, available)))
    for seed in available:
        p = results[seed]
        w = tuple(float(x) for x in p["weights"])
        vals = [f"({w[0]:.3f},{w[1]:.3f},{w[2]:.3f},{w[3]:.3f})"]
        vals += [f"{float(p[k]):.3f}" for k in keys[1:]]
        logger.info("    seed=%-4d %s", seed, "  ".join(vals))
    logger.info("区间对比（min~max）：")
    for k in keys[1:]:
        vs = [float(results[s][k]) for s in available]
        logger.info("    %-22s %12.4f ~ %-12.4f", k, min(vs), max(vs))
    logger.info("=" * 78)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="超参数寻优入口")
    ap.add_argument("--data", choices=("smoke", "real"), default="smoke",
                    help="smoke=Mock 快速回归；real=真实数据严格三集寻优")
    ap.add_argument("--trials", type=int, default=None,
                    help="主 seed 的 Optuna trial 数（默认 smoke=3 / real=20）")
    ap.add_argument("--seeds", default="42",
                    help="逗号分隔的随机种子列表；首个 seed 执行完整三集流程，"
                         "其余为稳定性 seed（默认 42）")
    ap.add_argument("--stability-trials", type=int, default=STABILITY_TRIALS,
                    help="每个稳定性 seed 的 trial 数（默认 %d）" % STABILITY_TRIALS)
    args = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # 统一日志目录（data/logs 已在 .gitignore 排除）：控制台 + 追加写入 run.log
    log_dir = os.path.join("data", "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(os.path.join(log_dir, "run.log"),
                                      encoding="utf-8")],
    )
    # 允许"稳定性 seed 数"通过参数传入（而非常量覆盖 hack）
    if args.data == "real":
        seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
        # None → 默认 REAL_TRIALS；显式 0 = 仅恢复已有 Journal、不再新增 trial
        run_optimization_real(
            n_trials=None if args.trials is None else args.trials,
            seeds=seeds,
            stability_trials=args.stability_trials)
    else:
        run_optimization_smoke()
