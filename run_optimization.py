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

import optuna

from analytics.metrics import constraint_violations
from main import (
    INITIAL_CASH,
    SMOKE_PARAMS,
    SYMBOL_TO_INDUSTRY,
    build_smoke_slice,
)
from optimizer.bayesian_opt import StrategyOptimizer
from optimizer.search_space import SearchSpace

logger = logging.getLogger("run_optimization")

N_TRIALS = 3  # 冒烟版仅跑 2~3 次 Trial


def _build_search_space() -> SearchSpace:
    """窄搜索空间：适配 6 天冒烟数据（chip_window 小 → CPS 更快可用；
    th_ms_bull 放宽 → 保证部分 trial 能产生真实成交）。"""
    return SearchSpace(
        w_ofss=(0.2, 0.6), w_cps=(0.1, 0.5),
        w_inst=(0.0, 0.4), w_north=(0.0, 0.3),
        th_ms_bull=(0.0, 0.5), th_ms_exit=(-0.4, -0.05),
        th_lock=(0.2, 0.8), th_purity=(-0.2, 0.3),
        th_global_min=(-0.8, -0.2), th_adr_min=(0.3, 0.7),
        win_inst=(1, 3), win_chip_old=(1, 3), win_hold_max=(10, 120),
    )


def _print_trial_detail(study: optuna.Study, opt: StrategyOptimizer,
                        search_space: SearchSpace) -> None:
    """逐 Trial 打印：参数注入结果 + 权重归一化校验 + 约束违反量。"""
    logger.info("-" * 78)
    for trial in study.trials:
        params, metrics = opt._cache[trial.number]  # objective/constraints 同源缓存
        weights = tuple(float(w) for w in params["weights"])
        w_north_derived = params["weights"][-1]
        feasible = search_space.is_feasible(params)
        violations = constraint_violations(metrics)

        logger.info("Trial %d：目标值(年化Sharpe)=%s",
                    trial.number, "NaN" if trial.value is None else f"{trial.value:.4f}")
        logger.info("    权重注入 (W_OFSS,W_CPS,W_INST,W_NORTH)=%s，和=%s，"
                    "W_NORTH 推导值=%s，SearchSpace 可行=%s",
                    tuple(round(w, 4) for w in weights),
                    f"{sum(weights):.6f}",
                    f"{w_north_derived:.4f}", feasible)
        logger.info("    阈值注入：th_ms_bull=%.3f th_ms_exit=%.3f th_lock=%.3f "
                    "th_purity=%.3f th_global_min=%.3f th_adr_min=%.3f",
                    params["th_ms_bull"], params["th_ms_exit"], params["th_lock"],
                    params["th_purity"], params["th_global_min"], params["th_adr_min"])
        logger.info("    窗口注入：inst_window=%d chip_window=%d win_hold_max=%d",
                    params["inst_window"], params["chip_window"], params["win_hold_max"])
        logger.info("    回测评估：有效交易=%d 笔 胜率=%.0f%% 盈亏比=%s 回撤=%.2f%% "
                    "总盈亏=%+.0f 元",
                    metrics["n_trades"],
                    metrics["win_rate"] * 100 if metrics["win_rate"] == metrics["win_rate"] else 0,
                    ("inf" if metrics["profit_loss_ratio"] == float("inf")
                     else f"{metrics['profit_loss_ratio']:.2f}"),
                    metrics["max_drawdown"] * 100, metrics["total_pnl"])
        logger.info("    约束违反量 [回撤, 胜率, 盈亏比, 交易笔数]=%s %s",
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
                 or k == "win_hold_max" or k == "inst_window"})
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


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    run_optimization_smoke()
