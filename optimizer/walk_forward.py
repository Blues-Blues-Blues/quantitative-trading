"""滚动样本外（OOS / Walk-Forward）交叉验证框架。

划分方案（默认 4 折滚动 OOS）：
- 把交易日按时间均分为 n_seg = train_folds + n_folds 个连续段
- 折 k（k = 0..n_folds-1）：
      OOS 段   = 第 train_folds + k 段
      训练段   = 段 0..(train_folds+k-1)（expanding，向前滚动时不断累积）
                ；expanding=False 时仅用最近 train_folds 段（rolling 固定窗）
- 每折在训练段独立寻优（StrategyOptimizer），把最优参数在 OOS 段回测，
  汇总各折 OOS 绩效 → 有效避免前视偏差与过度拟合。

防未来约定：
- 每个折的训练/OOS 特征都只基于该折切片内的数据独立重算
  （FeatureEngine 在 fold 切片上执行），绝不使用折外数据。
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.dataslice import DataSlice
from optimizer.bayesian_opt import StrategyOptimizer

logger = logging.getLogger("optimizer.walk_forward")


@dataclass
class FoldResult:
    """单折结果：训练/样本外时间窗、最优参数与两段绩效。"""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    best_params: Dict[str, object] = field(default_factory=dict)
    train_metrics: Dict[str, float] = field(default_factory=dict)
    oos_metrics: Dict[str, float] = field(default_factory=dict)


class WalkForward:
    """滚动样本外交叉验证：`run()` 对每折寻优并输出 OOS 评估报告。"""

    def __init__(
        self,
        data: DataSlice,
        n_folds: int = 4,
        train_folds: int = 4,
        expanding: bool = True,
        seed: int = 42,
    ) -> None:
        if n_folds < 1:
            raise ValueError(f"n_folds 必须 >= 1，当前: {n_folds}")
        if train_folds < 1:
            raise ValueError(f"train_folds 必须 >= 1，当前: {train_folds}")
        self.data = data
        self.n_folds = n_folds
        self.train_folds = train_folds
        self.expanding = expanding
        self.seed = seed

    # ------------------------------------------------------------------
    # 折划分
    # ------------------------------------------------------------------

    def _dates(self) -> List[pd.Timestamp]:
        """升序去重的交易日列表。"""
        return sorted(set(pd.Timestamp(t).normalize()
                          for t in self.data.time_axis()))

    def _segments(self) -> List[List[pd.Timestamp]]:
        """把交易日近均分为 n_seg 个连续段。"""
        dates = self._dates()
        n_seg = self.train_folds + self.n_folds
        idx = np.array_split(np.arange(len(dates)), n_seg)
        return [[dates[i] for i in part if len(part)] for part in idx]

    def fold_ranges(self) -> List[Tuple[pd.Timestamp, pd.Timestamp,
                                        pd.Timestamp, pd.Timestamp]]:
        """返回每折的 (train_start, train_end, oos_start, oos_end)。

        OOS 段紧接训练段（无间隔）；训练段 expanding 或固定最近 train_folds 段。
        """
        segments = self._segments()
        ranges = []
        for k in range(self.n_folds):
            oos = segments[self.train_folds + k]
            if self.expanding:
                train = [d for seg in segments[:self.train_folds + k] for d in seg]
            else:
                train = [d for seg in segments[k:self.train_folds + k] for d in seg]
            ranges.append((train[0], train[-1], oos[0], oos[-1]))
        return ranges

    def fold_datasets(self, k: int) -> Tuple[DataSlice, DataSlice]:
        """第 k 折的训练与样本外 DataSlice（各自独立切片）。"""
        ts, te, os_, oe = self.fold_ranges()[k]
        return (self.data.subset(ts, te), self.data.subset(os_, oe))

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run(
        self,
        n_trials: int = 200,
        symbol_to_industry: Optional[Dict[str, str]] = None,
        account_kwargs: Optional[Dict[str, object]] = None,
        optimizer_kwargs: Optional[Dict[str, object]] = None,
        plot_path: Optional[str] = None,
    ) -> List[FoldResult]:
        """对每折：训练段寻优 → 最优参数在 OOS 段回测。

        :param plot_path: 提供时保存每折的优化历程收敛图
            （默认 None 不画图；传 "analytics/pictures/wf_fold_{k}.png" 风格路径）
        :return: 每折一个 FoldResult
        """
        results: List[FoldResult] = []
        for k in range(self.n_folds):
            train_ds, oos_ds = self.fold_datasets(k)
            ts, te, os_, oe = self.fold_ranges()[k]
            logger.info("折 %d/%d: 训练 [%s, %s] → OOS [%s, %s]",
                        k + 1, self.n_folds, ts.date(), te.date(),
                        os_.date(), oe.date())

            opt = StrategyOptimizer(
                data=train_ds,
                symbol_to_industry=symbol_to_industry or {},
                account_kwargs=account_kwargs or {},
                n_trials=n_trials,
                seed=self.seed + k,
                **(optimizer_kwargs or {}),
            )
            study = opt.optimize()
            params, train_metrics = opt.best(study)
            if plot_path is not None:
                opt.plot_history(study, path=plot_path.format(k=k))

            oos_metrics, _ = opt.backtest(oos_ds, params)
            logger.info("  折 %d OOS: sharpe=%.3f mdd=%.1f%% trades=%d",
                        k + 1, oos_metrics.get("sharpe", float("nan")),
                        100 * oos_metrics.get("max_drawdown", 0.0),
                        int(oos_metrics.get("n_trades", 0)))
            results.append(FoldResult(
                fold=k, train_start=ts, train_end=te,
                oos_start=os_, oos_end=oe,
                best_params=params, train_metrics=train_metrics,
                oos_metrics=oos_metrics,
            ))
        return results

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------

    @staticmethod
    def to_frame(results: List[FoldResult]) -> pd.DataFrame:
        """折叠结果 → 表格（每行一折的样本外指标与最优参数要点）。"""
        rows = []
        for r in results:
            rows.append({
                "fold": r.fold + 1,
                "oos_start": r.oos_start,
                "oos_end": r.oos_end,
                "oos_sharpe": r.oos_metrics.get("sharpe", float("nan")),
                "oos_max_drawdown": r.oos_metrics.get("max_drawdown", float("nan")),
                "oos_win_rate": r.oos_metrics.get("win_rate", float("nan")),
                "oos_pl_ratio": r.oos_metrics.get("profit_loss_ratio", float("nan")),
                "oos_n_trades": r.oos_metrics.get("n_trades", float("nan")),
                "w_ofss": r.best_params.get("weights", (None,))[0],
                "w_cps": r.best_params.get("weights", (None, None))[1],
                "th_ms_bull": r.best_params.get("th_ms_bull"),
                "win_hold_max": r.best_params.get("win_hold_max"),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def summary(results: List[FoldResult]) -> Dict[str, float]:
        """各折 OOS 指标的均值（跨折汇总报告）。"""
        keys = ["sharpe", "max_drawdown", "win_rate",
                "profit_loss_ratio", "n_trades"]
        agg: Dict[str, float] = {}
        for key in keys:
            vals = [r.oos_metrics.get(key) for r in results]
            valid = [v for v in vals if v is not None and np.isfinite(v)]
            agg[f"oos_{key}_mean"] = float(np.mean(valid)) if valid else float("nan")
        agg["n_folds"] = float(len(results))
        return agg
