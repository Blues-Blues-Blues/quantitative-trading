"""基于 Optuna 的带约束贝叶斯超参数寻优（StrategyOptimizer）。

- 搜索空间：optimizer.search_space.SearchSpace
  （权重 Dirichlet 式归一化和为 1、阈值、窗口，含因子层 WIN_CHIP_OLD）
- 目标：最大化合成口径 = 样本内年化夏普比率 − 软惩罚
  （回撤超线 / 年化换手超线的平滑惩罚，硬约束线之内即开始生效，给 TPE
  更光滑的引导；见 objective()）
- 硬约束：TPESampler(constraints_func=...) 原生约束 —— 采样阶段优先探索满足
  约束的 trial：
      最大回撤、胜率、盈亏比、有效交易笔数、年化换手（参数化，见 constraint_kwargs）
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


def _window_end_str(params):  # 定义于 import 之前，避免类型注解前向求值问题
    """reversal_window_end 透传：直接给定则原样；给定 reversal_window_span
    （09:30 起分钟数，寻优用）则换算为 HH:MM。"""
    if "reversal_window_end" in params:
        return str(params["reversal_window_end"])
    span = int(params.get("reversal_window_span", 30))
    total = 9 * 60 + 30 + span
    return f"{total // 60:02d}:{total % 60:02d}"

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
# 断点续跑恢复时，按目标值降序最多重算候选的个数（补足未落盘的指标）
_MAX_RE_EVAL = 5


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
        use_feature_cache: bool = False,
        constraint_kwargs: Optional[Dict[str, object]] = None,
        penalty_kwargs: Optional[Dict[str, object]] = None,
    ) -> None:
        self.data = data
        self.search_space = search_space or SearchSpace()
        self.symbol_to_industry = symbol_to_industry or {}
        self.account_kwargs = account_kwargs or {}
        self.cost = cost or ExecutionCost()
        self.sizer = sizer or PositionSizer()
        self.n_trials = n_trials
        self.seed = seed
        self.use_feature_cache = use_feature_cache
        # 硬约束阈值（真实数据寻优用宽松实测口径；None = 默认 回撤15%/胜率55%/盈亏比1.5/交易30）
        self.constraint_kwargs = constraint_kwargs or {}

        # 合成目标软惩罚（硬约束线内即生效，用于给 TPE 平滑引导）：
        #   score = Sharpe - dd_pen*max(0,回撤-dd_soft) - to_pen*max(0,年化换手-to_soft)
        pk = penalty_kwargs or {}
        self.penalty_kwargs = {
            "dd_soft": float(pk.get("dd_soft", 0.25)),
            "dd_pen": float(pk.get("dd_pen", 1.0)),
            "to_soft": float(pk.get("to_soft", 6.0)),
            "to_pen": float(pk.get("to_pen", 0.03)),
        }

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
        # 真实数据寻优（use_feature_cache=True）：特征按 (区间,标的) 落盘缓存，
        # 同一切片首 trial 计算一次、后续命中，避免每 trial 重算特征（关键性能优化）
        if self.use_feature_cache:
            features = fe.compute_cached(ds)
        else:
            features = fe.compute(ds)

        syn = SignalSynthesizer(
            weights=tuple(float(w) for w in params["weights"]),
            inst_window=int(params["inst_window"]),
            # 兼容参数（旧二值化闸门，决策链不再读取；SearchSpace 已停止采样）
            th_ms_bull=float(params.get("th_ms_bull", 0.0)),
            th_ms_exit=float(params.get("th_ms_exit", -0.1)),
            th_lock=float(params.get("th_lock", 0.5)),
            th_purity=float(params.get("th_purity", 0.0)),
            th_global_min=float(params.get("th_global_min", 0.0)),
            th_adr_min=float(params.get("th_adr_min", 1.0)),
            win_hold_max=int(params.get("win_hold_max", 240)),
            # ---- 连续评分 ES（SearchSpace 采样，missing 时用默认值降级）----
            w_es_ms=float(params.get("w_es_ms", 0.4)),
            w_es_purity=float(params.get("w_es_purity", 0.3)),
            w_es_mrs=float(params.get("w_es_mrs", 0.3)),
            es_sigmoid_k=float(params.get("es_sigmoid_k", 3.0)),
            th_es_entry=float(params.get("th_es_entry", 0.4)),
            # ---- 连续评分 XS ----
            th_xs_exit=float(params.get("th_xs_exit", -0.3)),
            th_xs_reduce_high=float(params.get("th_xs_reduce_high", 0.2)),
            th_xs_crash=float(params.get("th_xs_crash", -0.6)),
            w_xs_ms=float(params.get("w_xs_ms", 0.5)),
            w_xs_purity=float(params.get("w_xs_purity", 0.3)),
            w_xs_drawdown=float(params.get("w_xs_drawdown", 0.2)),
            # ---- 次日低开反包 ----
            th_reversal_gap=float(params.get("th_reversal_gap", -0.015)),
            th_reversal_ofss=float(params.get("th_reversal_ofss", 0.2)),
            reversal_add_mult=float(params.get("reversal_add_mult", 0.5)),
            reversal_window_end=_window_end_str(params),
            # ---- 连续评分 PS ----
            base_decay_rate=float(params.get("base_decay_rate", 0.95)),
            win_decay_grace=int(params.get("win_decay_grace", 30)),
            pnl_decay_profit_mult=float(params.get("pnl_decay_profit_mult", 0.5)),
            pnl_decay_loss_mult=float(params.get("pnl_decay_loss_mult", 2.0)),
            cancel_ratio_th=float(params.get("cancel_ratio_th", 0.25)),
            fund_stability_penalty=float(params.get("fund_stability_penalty", 0.7)),
            # ---- 状态 / 一票否决 ----
            th_retail_chase=float(params.get("th_retail_chase", 0.65)),
            # ---- 目标权重 Target_Weight ----
            base_weight=float(params.get("base_weight", 0.20)),
            reduce_step_ratio=float(params.get("reduce_step_ratio", 0.8)),
            tw_gmod_clip=tuple(
                float(x) for x in params.get("tw_gmod_clip", (0.2, 1.5))),
            tw_cmod_clip=tuple(
                float(x) for x in params.get("tw_cmod_clip", (0.5, 1.5))),
            symbol_to_industry=self.symbol_to_industry,
        )
        sm = TradingStateMachine(synthesizer=syn)
        signals = sm.run(ds, features)

        engine = BacktestEngine(
            Account(**self.account_kwargs), self.cost, self.sizer, ds, signals,
            deadzone_th=float(params.get("deadzone_th", 0.05)))
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
        """优化目标（最大化）：训练段年化 Sharpe − 软惩罚。

        软惩罚在硬约束线内即生效（用于给 TPE 更光滑的引导，不替代约束）：
            dd_pen × max(0, 最大回撤 − dd_soft)
            to_pen × max(0, 年化换手 − to_soft)
        """
        _, metrics = self._params_and_metrics(trial)
        sharpe = float(metrics.get("sharpe", float("nan")))
        if not np.isfinite(sharpe):
            return _NAN_OBJECTIVE
        pk = self.penalty_kwargs
        score = sharpe
        dd = float(metrics.get("max_drawdown", float("nan")))
        if np.isfinite(dd):
            score -= pk["dd_pen"] * max(0.0, dd - pk["dd_soft"])
        to = float(metrics.get("turnover_annual", float("nan")))
        if np.isfinite(to):
            score -= pk["to_pen"] * max(0.0, to - pk["to_soft"])
        return float(score)

    def constraints_func(self, trial: optuna.Trial) -> List[float]:
        """硬约束违反量（4 个，>= 0；0 = 满足，NaN → 最大违反）。"""
        _, metrics = self._params_and_metrics(trial)
        violations = constraint_violations(metrics, **self.constraint_kwargs)
        return [float(v) if np.isfinite(v) else _NAN_VIOLATION for v in violations]

    # ------------------------------------------------------------------
    # 寻优入口与结果
    # ------------------------------------------------------------------

    def optimize(self, n_trials: Optional[int] = None,
                 study_name: str = "optuna_study",
                 storage_path: Optional[str] = None) -> optuna.Study:
        """运行贝叶斯寻优，返回 Optuna Study。

        storage_path 非空时使用持久化存储（JournalStorage 单文件）：
        支持中途停止 / 断点续跑——已跑 trial 全部保留，重复调用从断点
        继续新增 trial（n_trials 为"本次新增数"）。显式传 n_trials=0
        表示「仅恢复已有结果、不再新增」，配合续跑把流程收尾
        （best 选择 / 样本外评估）。恢复时显式重绑 TPESampler
        （load_if_exists 会忽略传入 sampler），保证后续 trial 仍走
        含硬约束的 TPE 采样。
        """
        sampler = TPESampler(seed=self.seed, constraints_func=self.constraints_func)
        if storage_path is None:
            storage, load_if = None, False
        else:
            os.makedirs(os.path.dirname(os.path.abspath(storage_path)),
                        exist_ok=True)
            storage = optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(
                    storage_path,
                    # Windows 无符号链接特权：改用"创建锁文件"式锁
                    lock_obj=optuna.storages.journal.JournalFileOpenLock(
                        storage_path)))
            load_if = True
        study = optuna.create_study(
            study_name=study_name, storage=storage,
            direction="maximize", sampler=sampler, load_if_exists=load_if)
        if load_if:
            study.sampler = sampler  # 续跑场景重绑含硬约束的采样器
        study.optimize(self.objective,
                       n_trials=None if n_trials is None
                       else max(0, int(n_trials)))
        return study

    def _trial_violations(self, trial: optuna.Trial) -> List[float]:
        """该 trial 的硬约束违反量（需先在 _cache 中有评估结果）。"""
        _, metrics = self._cache[trial.number]
        return constraint_violations(metrics, **self.constraint_kwargs)

    def best(self, study: optuna.Study) -> Tuple[Dict[str, object], Dict[str, float]]:
        """返回 (最优参数, 其样本内指标)。

        本进程已评估（_cache 中）的 trial 内：
        - 优先返回「满足全部硬约束（self.constraint_kwargs 口径）且目标值
          最优」的 trial；
        - 无可行 trial 时，回退为按目标值（Sharpe）最优的已评估 trial 兜底。

        断点续跑场景（本次进程未新增任何 trial、全部从 Journal 加载，
        _cache 为空）：按目标值降序逐一对候选 trial 重建参数并回测重算，
        返回首个满足硬约束者（最多 _MAX_RE_EVAL 个）；仍无可行解则回退为
        目标值最优。
        """
        cached = [t for t in study.trials
                  if t.value is not None and t.number in self._cache]
        if cached:
            feasible = [t for t in cached
                        if all(float(v) <= 1e-9
                               for v in self._trial_violations(t))]
            pool = feasible or cached
            best_trial = max(pool, key=lambda t: float(t.value))
            if not feasible:
                logger.warning(
                    "无满足硬约束的 trial，回退为按目标值最优（trial %d）",
                    best_trial.number)
            else:
                logger.info("满足硬约束的最优 trial: %d（Sharpe=%.4f）",
                            best_trial.number, float(best_trial.value))
            return self._cache[best_trial.number]

        ranked = [t for t in study.trials if t.value is not None]
        if not ranked:
            raise ValueError("study 无任何有效 trial") from None
        ranked.sort(key=lambda t: float(t.value), reverse=True)
        logger.info("断点续跑：%d 个 Journal trial 无进程内缓存，"
                    "按目标值降序重算候选指标（上限 %d 个）",
                    len(ranked), _MAX_RE_EVAL)
        for trial in ranked[:_MAX_RE_EVAL]:
            params = SearchSpace.params_from_trial(trial)
            metrics = self._evaluate(params)  # 训练段单点重算，补足未落盘指标
            self._cache[trial.number] = (params, metrics)
            if all(float(v) <= 1e-9 for v in
                   constraint_violations(metrics, **self.constraint_kwargs)):
                logger.info("续跑恢复：trial %d 满足硬约束（Sharpe=%.4f）",
                            trial.number, float(trial.value))
                return params, metrics
        best_trial = ranked[0]
        params, metrics = self._cache[best_trial.number]
        logger.warning("续跑恢复：前 %d 个候选均不满足硬约束，"
                       "回退为按目标值最优（trial %d）",
                       min(_MAX_RE_EVAL, len(ranked)), best_trial.number)
        return params, metrics

    def top_candidates(self, study: optuna.Study, k: int = 5,
                       max_re_eval: int = _MAX_RE_EVAL
                       ) -> List[Tuple[Dict[str, object], Dict[str, float]]]:
        """约束可行候选（按目标值降序，至多 k 个）。

        供「训练段 top-k → 验证段复评」使用（严格三集：验证段参与选参、
        测试段仅终评）。进程内已有缓存（本进程刚跑完）直接取；
        Journal 断点续跑缺缓存的 trial 按目标值降序重算指标（上限
        max_re_eval 个）。无可行候选时降级为按目标值最优的 k 个。

        :return: [(params, 训练段metrics), ...]，目标值降序
        """
        def _metrics_of(t: optuna.Trial) -> Dict[str, float]:
            if t.number in self._cache:
                return self._cache[t.number][1]
            params = SearchSpace.params_from_trial(t)
            m = self._evaluate(params)
            self._cache[t.number] = (params, m)
            return m

        # 1) 补足 Journal 无缓存候选的指标（按目标值降序优先，限个）
        missing = [t for t in study.trials
                   if t.value is not None and t.number not in self._cache]
        missing.sort(key=lambda t: float(t.value), reverse=True)
        for t in missing[:max_re_eval]:
            _metrics_of(t)

        ranked = [t for t in study.trials
                  if t.value is not None and t.number in self._cache]
        ranked.sort(key=lambda t: float(t.value), reverse=True)
        feasible = [t for t in ranked
                    if all(float(v) <= 1e-9 for v in
                           constraint_violations(
                               self._cache[t.number][1],
                               **self.constraint_kwargs))]
        pool = feasible or ranked
        if not feasible:
            logger.warning("top_candidates：无满足硬约束的 trial，"
                           "降级按目标值取前 %d", k)
        return [(self._cache[t.number][0], dict(self._cache[t.number][1]))
                for t in pool[:k]]

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
        ax.scatter(x, y, s=18, alpha=0.6, label="trial score")
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
        ax.set_ylabel("synthetic objective (Sharpe - penalties)")
        ax.set_title("Optimizer history (synthetic objective, in-sample)")
        ax.legend()
        fig.tight_layout()

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("收敛图已保存: %s", path)
        return path
