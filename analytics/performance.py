"""绩效评估与可视化：PerformanceAnalyzer。

覆盖任务要求：
- 绩效指标总表：年化收益率、夏普、卡玛（Calmar）、Sortino、最大回撤、
  平均持仓周期、胜率、盈亏比、日度收益偏度/峰度
- 多子图可视化（Matplotlib）：净值回撤图、动态仓位图、归因柱状图、
  参数敏感度热力图（四子图 Dashboard）
- 20 日人工复盘清单导出（Excel/CSV）：触发买入/卖出前后 N 个交易日的
  日线指标切片 + 逐笔资金流清单
"""

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analytics.metrics import (
    annual_return, calmar_ratio, closed_trades, daily_sharpe, holding_period,
    max_drawdown, return_skew_kurtosis, sortino_ratio, trade_stats,
)


class PerformanceAnalyzer:
    """绩效评估、可视化与复盘清单导出。"""

    # ------------------------------------------------------------------
    # 绩效指标
    # ------------------------------------------------------------------

    @staticmethod
    def analyze(equity_curve: pd.DataFrame,
                trade_log: pd.DataFrame) -> Dict[str, float]:
        """完整绩效指标总表。"""
        st = trade_stats(trade_log)
        skew, kurt = return_skew_kurtosis(equity_curve)
        return {
            "annual_return": annual_return(equity_curve),
            "sharpe": daily_sharpe(equity_curve),
            "calmar": calmar_ratio(equity_curve),
            "sortino": sortino_ratio(equity_curve),
            "max_drawdown": max_drawdown(equity_curve),
            "avg_holding_minutes": holding_period(trade_log),
            "n_trades": float(st["n_trades"]),
            "win_rate": float(st["win_rate"]),
            "profit_loss_ratio": float(st["profit_loss_ratio"]),
            "total_pnl": float(st["total_pnl"]),
            "daily_skew": skew,
            "daily_kurtosis": kurt,
        }

    @staticmethod
    def to_frame(metrics: Dict[str, float]) -> pd.DataFrame:
        """指标 dict → 单行 DataFrame（便于落盘 / 报告）。"""
        return pd.DataFrame([metrics])

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------

    @staticmethod
    def plot_report(equity_curve: pd.DataFrame, trade_log: pd.DataFrame,
                    attribution_summary: Optional[pd.DataFrame] = None,
                    study: Optional[object] = None,
                    path: str = "analytics/pictures/dashboard.png") -> str:
        """四子图 Dashboard：净值回撤 / 动态仓位 / 归因柱状图 / 参数敏感度热力图。"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        # 1) 净值 + 回撤
        ax = axes[0, 0]
        s = _equity_series(equity_curve)
        ax.plot(s.index, s.values, color="#1f77b4", linewidth=1.0)
        ax.set_ylabel("total equity")
        ax.set_title("Equity curve & drawdown")
        peak = s.cummax()
        dd = (peak - s) / peak * 100
        ax2 = ax.twinx()
        ax2.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.35)
        ax2.set_ylabel("drawdown (%)")
        ax2.set_ylim(bottom=dd.max() * 1.1 if len(dd) else 1)

        # 2) 动态仓位
        ax = axes[0, 1]
        _plot_position(ax, equity_curve)
        ax.set_title("Position ratio & #positions")

        # 3) 归因柱状图
        ax = axes[1, 0]
        if attribution_summary is not None and not attribution_summary.empty:
            s2 = attribution_summary.set_index("factor")["pnl"]
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#999999"]
            bars = ax.bar(s2.index, s2.values, color=colors[:len(s2)],
                          edgecolor="black", linewidth=0.5)
            ax.axhline(0, color="black", linewidth=0.8)
            for b, v in zip(bars, s2.values):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}",
                        ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=8)
            ax.grid(axis="y", alpha=0.3)
        else:
            ax.text(0.5, 0.5, "no attribution data", ha="center",
                    va="center", transform=ax.transAxes)
        ax.set_ylabel("attributed PnL (CNY)")
        ax.set_title("Attribution by factor exposure")

        # 4) 参数敏感度热力图
        ax = axes[1, 1]
        if study is not None:
            PerformanceAnalyzer.sensitivity_heatmap(study, ax=ax)
        else:
            ax.text(0.5, 0.5, "no optimization history", ha="center",
                    va="center", transform=ax.transAxes)
            ax.set_title("Parameter sensitivity")

        fig.suptitle("Strategy performance dashboard", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return str(out)

    @staticmethod
    def sensitivity_heatmap(study_or_frame: object, x: str = "w_ofss",
                            y: str = "w_cps", bins: int = 5,
                            ax: Optional[plt.Axes] = None) -> plt.Axes:
        """参数敏感度热力图：x × y 两参数分桶后各桶目标值均值。

        输入为 Optuna Study（取 trials.params + value）或 DataFrame
        （列含 x/y 参数与 value）。
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5.5))
        frame = _trials_frame(study_or_frame)
        if frame is None or len(frame) == 0:
            ax.text(0.5, 0.5, "no trial data", ha="center", va="center",
                    transform=ax.transAxes)
            return ax
        if x not in frame.columns or y not in frame.columns:
            ax.text(0.5, 0.5, f"missing params {x}/{y}", ha="center",
                    va="center", transform=ax.transAxes)
            return ax
        df = frame[[x, y, "value"]].dropna()
        if len(df) < 4 or df[x].nunique() < 2 or df[y].nunique() < 2:
            ax.text(0.5, 0.5, "insufficient trials", ha="center", va="center",
                    transform=ax.transAxes)
            return ax
        def _bins(series: pd.Series) -> list:
            try:
                return pd.qcut(series, min(bins, series.nunique()),
                               duplicates="drop")
            except ValueError:
                return pd.cut(series, min(bins, series.nunique()))
        df["xb"] = _bins(df[x])
        df["yb"] = _bins(df[y])
        piv = df.pivot_table(index="yb", columns="xb", values="value",
                             aggfunc="mean")
        piv = piv.astype(float)
        vmax = max(abs(np.nanmax(piv.values)),
                   abs(np.nanmin(piv.values)), 1e-9)
        im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([_mid(c) for c in piv.columns], rotation=30,
                           ha="right", fontsize=8)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([_mid(c) for c in piv.index], fontsize=8)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title("Parameter sensitivity (objective mean)")
        fig = ax.figure
        fig.colorbar(im, ax=ax, shrink=0.85, label="objective")
        return ax

    # ------------------------------------------------------------------
    # 20 日人工复盘清单
    # ------------------------------------------------------------------

    @staticmethod
    def export_review_slices(ds: object, trade_log: pd.DataFrame,
                             days: int = 20,
                             path: str = "analytics/pictures/review.xlsx",
                             fmt: str = "xlsx") -> str:
        """导出每笔平仓触发前后 N 个交易日的指标切片 + 逐笔资金流清单。

        Excel：三个 sheet —— summary（逐笔交易）、daily_slices（日线切片，
        标注 entry/exit 触发日）、tick_flows（窗口内逐笔资金流）。
        CSV：同名目录下三个 .csv 文件。
        """
        trades = closed_trades(trade_log)
        kline = ds.kline
        ticks = ds.tick_trades
        if fmt == "csv":
            out = Path(path)
            out.mkdir(parents=True, exist_ok=True)
        else:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)

        daily_all, tick_all, summary_rows = [], [], []
        for i, t in enumerate(trades):
            sym = t["symbol"]
            daily = _daily_kline(kline, sym)
            if daily.empty:
                continue
            dates = daily.index
            entry_day, exit_day = t["entry_ts"].normalize(), t["exit_ts"].normalize()
            i0 = dates.searchsorted(entry_day)
            lo, hi = max(0, i0 - days), min(len(dates), i0 + days + 1)
            win = daily.iloc[lo:hi].copy()
            win.insert(0, "trade_id", i)
            win["trigger"] = np.where(
                win.index == entry_day, "entry",
                np.where(win.index == exit_day, "exit", ""))
            daily_all.append(win.reset_index())
            # 窗口内逐笔资金流
            if ticks is not None and not ticks.empty:
                t_lo, t_hi = win.index[0], win.index[-1] + pd.Timedelta(days=1)
                sub = ticks[(ticks.index >= t_lo) & (ticks.index < t_hi)
                            & (ticks["symbol"] == sym)].copy()
                if not sub.empty:
                    sub.insert(0, "trade_id", i)
                    tick_all.append(sub.reset_index())
            summary_rows.append({
                "trade_id": i, "symbol": sym,
                "entry_ts": t["entry_ts"], "exit_ts": t["exit_ts"],
                "entry_day": entry_day, "exit_day": exit_day,
                "shares": t["shares"], "entry_price": t["entry_price"],
                "pnl": t["pnl"],
            })

        summary = pd.DataFrame(summary_rows)
        daily_df = (pd.concat(daily_all, ignore_index=True)
                    if daily_all else pd.DataFrame())
        tick_df = (pd.concat(tick_all, ignore_index=True)
                   if tick_all else pd.DataFrame())

        if fmt == "csv":
            summary.to_csv(out / "summary.csv", index=False)
            daily_df.to_csv(out / "daily_slices.csv", index=False)
            tick_df.to_csv(out / "tick_flows.csv", index=False)
            return str(out)
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="summary", index=False)
            daily_df.to_excel(writer, sheet_name="daily_slices", index=False)
            tick_df.to_excel(writer, sheet_name="tick_flows", index=False)
        return str(out)


# ----------------------------------------------------------------------
# 内部工具
# ----------------------------------------------------------------------


def _equity_series(curve: pd.DataFrame) -> pd.Series:
    if not isinstance(curve, pd.DataFrame) or curve.empty:
        return pd.Series(dtype=float)
    if "total_equity" not in curve.columns:
        return pd.Series(dtype=float)
    s = curve.set_index("ts")["total_equity"].astype(float)
    return s[~s.index.duplicated(keep="last")].sort_index()


def _plot_position(ax: plt.Axes, curve: pd.DataFrame) -> None:
    if curve is None or curve.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        return
    c = curve.set_index("ts")
    if "total_equity" in c.columns and "position_value" in c.columns:
        ratio = c["position_value"] / c["total_equity"].replace(0, np.nan)
        ax.fill_between(ratio.index, ratio.values, 0, color="#2ca02c",
                        alpha=0.4, label="position ratio")
    if "n_positions" in c.columns:
        ax.plot(c.index, c["n_positions"], color="#9467bd", linewidth=1.0,
                label="#positions")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylabel("ratio / count")


def _daily_kline(kline: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """单标的分钟 K 线 → 日线（open/high/low/close/volume/amount）。"""
    sub = kline[kline["symbol"] == symbol]
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby(sub.index.normalize())
    daily = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
        "amount": g["amount"].sum(),
        "vwap": g["vwap"].last(),
    })
    return daily.sort_index()


def _trials_frame(study_or_frame: object) -> Optional[pd.DataFrame]:
    """Optuna Study 或 DataFrame → {参数列..., value}。"""
    if isinstance(study_or_frame, pd.DataFrame):
        return study_or_frame if "value" in study_or_frame.columns else None
    trials = getattr(study_or_frame, "trials", None)
    if not trials:
        return None
    rows = []
    for t in trials:
        if t.value is None:
            continue
        row = dict(t.params)
        row["value"] = float(t.value)
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows)


def _mid(interval: object) -> str:
    try:
        if hasattr(interval, "mid"):
            return f"{float(interval.mid):.2f}"
    except (TypeError, ValueError):
        pass
    return str(interval)
