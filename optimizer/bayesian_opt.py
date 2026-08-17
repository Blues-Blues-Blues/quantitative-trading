"""基于 Optuna 的带约束贝叶斯超参数寻优（StrategyOptimizer）。

- 搜索空间：optimizer.search_space.SearchSpace
  （权重 Dirichlet 式归一化和为 1、阈值、窗口，含因子层 WIN_CHIP_OLD）
- 目标：最大化样本内年化夏普比率（日频重采样，见 analytics.metrics）
- 硬约束：TPESampler(constraints_func=...) 原生约束 —— 采样阶段优先探索满足
  约束的 trial：
      最大回撤 < 15%、胜率 > 55%、盈亏比 > 1.5、有效交易 >= 30 笔
- 流水线（每次 trial）：DataSlice → FeatureEngine（含 chip_window）→
  TradingStateMachine → BacktestEngine → metrics
- 输出：Optuna Study（最优参数 + 全部 trial 记录）、优化历程收敛图、
  最优参数在任意数据切片上的回测评估（供 Walk-Forward 样本外复用）

优化运行的显式步骤（用户也可直接调用各方法）：
    optimizer = StrategyOptimizer(data=train_ds, ...)
    study = optimizer.optimize()
    params, metrics = optimizer.best(study)      # 样本内最优参数与指标
    oos_log, oos_curve = optimizer.backtest(oos_ds, params)  # 样本外回测
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.samplers import TPESampler

from analytics.metrics import constraint_violations, evaluate
from data.dataslice import DataSlice
from engine.backtest import BacktestEngine
from engine.execution import ExecutionCost
from engine.portfolio import Account
from engine.risk_control import PositionSizer
from indicators.feature_engine import FeatureEngine
from indicators.microstructure import MicroStructure
from optimizer.search_space import SearchSpace
from strategy.signals import SignalSynthesizer, TradingStateMachine

logger = logging.getLogger("optimizer.bayesian_opt")

# 无有效收益（未交易/零波动）时目标函数的兜底值
_NAN_OBJECTIVE = -1e6
# 约束函数中 NaN 指标的违反量
_NAN_VIOLATION = 1e9


class StrategyOptimizer:
    """带多重非线性约束的贝叶斯超参数优化器。"""

    def __init__(
        self,
        data: DataSlice,
        search_space: Optional[SearchSpace] = None,
        symbol_to_industry: Optional[Dict[str, str]] = None,
        account_kwargs: Optional[Dict[str, object]] = None,
        cost: Optional[ExecutionCost] = None,
        sizer: Optional[PositionSizer] = None,
        n_trials: int = 200,
        seed: int = 42,
    ) -> None:
        self.data = data
        self.search_space = search_space or SearchSpace()
        self.symbol_to_industry = symbol_to_industry or {}
        self.account_kwargs = account_kwargs or {}
        self.cost = cost or ExecutionCost()
        self.sizer = sizer or PositionSizer()
        self.n_trials = n_trials
        self.seed = seed

        # trial.number → (params, metrics)，避免 objective/constraints_func
        # 在同一 trial 内被 Optuna 多次调用时重复回测
        self._cache: Dict[int, Tuple[Dict[str, object], Dict[str, float]]] = {}

    # ------------------------------------------------------------------
    # 回测流水线（供寻优与 Walk-Forward 样本外复用）
    # ------------------------------------------------------------------

    def backtest(self, ds: DataSlice, params: Dict[str, object]
                 ) -> Tuple[Dict[str, float], BacktestEngine]:
        """在任意数据切片上以给定参数跑完整流水线，返回 (metrics, engine)。"""
        micro = MicroStructure(chip_window=int(params["chip_window"]))
        fe = FeatureEngine(micro=micro, symbol_to_industry=self.symbol_to_industry)
        features = fe.compute(ds)

        syn = SignalSynthesizer(
            weights=tuple(float(w) for w in params["weights"]),
            inst_window=int(params["inst_window"]),
            th_ms_bull=float(params["th_ms_bull"]),
            th_ms_exit=float(params["th_ms_exit"]),
            th_lock=float(params["th_lock"]),
            th_purity=float(params["th_purity"]),
            th_global_min=float(params["th_global_min"]),
            th_adr_min=float(params["th_adr_min"]),
            win_hold_max=int(params["win_hold_max"]),
            symbol_to_industry=self.symbol_to_industry,
        )
        sm = TradingStateMachine(synthesizer=syn)
        signals = sm.run(ds, features)

        engine = BacktestEngine(
            Account(**self.account_kwargs), self.cost, self.sizer, ds, signals)
        trade_log, equity_curve = engine.run()
        return evaluate(equity_curve, trade_log), engine

    def _evaluate(self, params: Dict[str, object]) -> Dict[str, float]:
        """单次回测评估：指标字典（含 sharpe / 约束相关指标）。"""
        try:
            metrics, _ = self.backtest(self.data, params)
        except Exception as exc:  # 参数组合触发引擎异常 → 视为最差评估
            logger.debug("trial 回测失败: %s", exc)
            return {"sharpe": float("nan"), "max_drawdown": float("nan"),
                    "n_trades": 0.0, "win_rate": float("nan"),
                    "profit_loss_ratio": float("nan"), "total_pnl": 0.0}
        return metrics

    def _params_and_metrics(self, trial: optuna.Trial
                            ) -> Tuple[Dict[str, object], Dict[str, float]]:
        """同一 trial 内缓存参数与评估结果（objective / constraints 复用）。"""
        num = trial.number
        if num not in self._cache:
            params = self.search_space.suggest(trial)
            self._cache[num] = (params, self._evaluate(params))
        return self._cache[num]

    # ------------------------------------------------------------------
    # Optuna 回调
    # ------------------------------------------------------------------

    def objective(self, trial: optuna.Trial) -> float:
        """优化目标：样本内年化夏普比率（最大化）。"""
        _, metrics = self._params_and_metrics(trial)
        sharpe = float(metrics.get("sharpe", float("nan")))
        return sharpe if np.isfinite(sharpe) else _NAN_OBJECTIVE

    def constraints_func(self, trial: optuna.Trial) -> List[float]:
        """硬约束违反量（4 个，>= 0；0 = 满足，NaN → 最大违反）。"""
        _, metrics = self._params_and_metrics(trial)
        violations = constraint_violations(metrics)
        return [float(v) if np.isfinite(v) else _NAN_VIOLATION for v in violations]

    # ------------------------------------------------------------------
    # 寻优入口与结果
    # ------------------------------------------------------------------

    def optimize(self, n_trials: Optional[int] = None) -> optuna.Study:
        """运行贝叶斯寻优，返回 Optuna Study。

        TPE sampler 以 constraints_func 为硬约束：采样优先落在可行域。
        """
        sampler = TPESampler(seed=self.seed, constraints_func=self.constraints_func)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(self.objective, n_trials=n_trials or self.n_trials)
        return study

    def best(self, study: optuna.Study) -> Tuple[Dict[str, object], Dict[str, float]]:
        """返回 (最优参数, 其样本内指标)。

        约束寻优下若无任何满足硬约束的 trial，Optuna 的 best_trial 会抛
        ValueError：此时回退为按目标值（Sharpe）最优的 trial 兜底返回。
        """
        try:
            best_trial = study.best_trial
        except ValueError:
            candidates = [t for t in study.trials if t.value is not None]
            if not candidates:
                raise ValueError("study 无任何有效 trial") from None
            best_trial = max(candidates, key=lambda t: float(t.value))
            logger.warning("无满足硬约束的 trial，回退为按目标值最优（trial %d）",
                           best_trial.number)
        params, metrics = self._cache[best_trial.number]
        return params, metrics

    # ------------------------------------------------------------------
    # 收敛图
    # ------------------------------------------------------------------

    def plot_history(self, study: optuna.Study,
                     path: str = "analytics/pictures/optimizer_history.png") -> str:
        """优化历程收敛图：全部 trial 目标值 + 可行域内最优值轨迹。

        :return: 图片保存路径
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        trials = study.trials
        x = [t.number for t in trials]
        y = [t.value for t in trials if t.value is not None]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.axhline(_NAN_OBJECTIVE, color="gray", linestyle="--", linewidth=0.8,
                   label="NaN objective floor")
        ax.scatter(x, y, s=18, alpha=0.6, label="trial Sharpe")
        # 可行域内最优值轨迹（cummax；含 NaN 的 trial 视为不达标跳过）
        best = -np.inf
        xs, ys = [], []
        for t, v in zip(trials, y):
            if np.isfinite(v) and v > best:
                best = v
            xs.append(t.number)
            ys.append(best if np.isfinite(best) else float("nan"))
        ax.plot(xs, ys, color="crimson", linewidth=1.6, label="best-so-far (feasible)")
        ax.set_xlabel("trial")
        ax.set_ylabel("annualized Sharpe (in-sample)")
        ax.set_title("Optimizer history (annualized Sharpe, in-sample)")
        ax.legend()
        fig.tight_layout()

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("收敛图已保存: %s", path)
        return path
