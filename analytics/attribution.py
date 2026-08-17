"""收益归因分析：因子暴露分解 + 因子 IC / Rank IC / IR 时序。

因子暴露分解（归因）：
    把每笔已实现盈亏按「入场时点」的因子绝对暴露占比拆解到三大来源：
        Global_Mod（宏观共振）、Chain_Mod（行业共振）、Agent_MS（个股情绪/盘口）
    例如：一笔 +1000 元的交易，入场时 |Global|=0.73、|Chain|=0.3、|Agent|=0.58，
    则宏观贡献 ≈ 1000×0.73/(0.73+0.3+0.58)。无法取得入场快照的交易归入 other。

因子 IC（预测能力检验）：
    支持横截面 Rank IC（多标的，每个时间戳对全市场做 Spearman 相关）与
    时序相关（单标的，因子与未来收益的滚动相关），前瞻收益窗口可配置
    （默认 60 分钟，或 "1D" 下一个交易日收盘）。IC IR = mean/std。
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analytics.metrics import closed_trades

# 归因因子：metrics 快照键 → 展示名
ATTRIBUTION_FACTORS: List[str] = ["global_mod", "chain_mod", "agent_ms"]
ATTRIBUTION_NAMES: Dict[str, str] = {
    "global_mod": "Global_Mod", "chain_mod": "Chain_Mod", "agent_ms": "Agent_MS",
}

# IC 默认因子清单（须存在于传入的 features 表）
DEFAULT_IC_FACTORS: List[str] = [
    "final_ms", "agent_ms", "global_mod", "chain_mod",
    "ofss", "cps", "inst_flow", "north_sync", "capital_purity",
]


def _as_signals_frame(signals: object) -> pd.DataFrame:
    """Signal 列表或 DataFrame → 标准信号表（timestamp/symbol/action/state + 指标列）。"""
    if isinstance(signals, pd.DataFrame):
        return signals
    from strategy.signals import TradingStateMachine
    return TradingStateMachine.to_frame(signals)


class AttributionEngine:
    """收益归因与因子预测能力分析引擎。"""

    # ------------------------------------------------------------------
    # 归因：因子暴露分解
    # ------------------------------------------------------------------

    @staticmethod
    def attribute(trade_log: pd.DataFrame,
                  signals: object) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """每笔平仓按入场因子暴露拆解已实现盈亏。

        :return: (trades_df, summary_df)
            trades_df 列：symbol, entry_ts, exit_ts, pnl, 入场三因子值,
                          pnl_global_mod, pnl_chain_mod, pnl_agent_ms, factor
            summary_df 行：Global_Mod / Chain_Mod / Agent_MS / other,
                           列为 pnl、weight 与 n_trades
        """
        trades = closed_trades(trade_log)
        sig = _as_signals_frame(signals)
        if not sig.empty:
            sig["timestamp"] = pd.to_datetime(sig["timestamp"])

        rows: List[Dict[str, object]] = []
        for t in trades:
            row: Dict[str, object] = {
                "symbol": t["symbol"], "entry_ts": t["entry_ts"],
                "exit_ts": t["exit_ts"], "pnl": t["pnl"],
                "factor": "other",
                "entry_global_mod": float("nan"),
                "entry_chain_mod": float("nan"),
                "entry_agent_ms": float("nan"),
            }
            for f in ATTRIBUTION_FACTORS:
                row[f"pnl_{f}"] = 0.0
            if not sig.empty:
                entry = sig[(sig["timestamp"] == t["entry_ts"])
                            & (sig["symbol"] == t["symbol"])
                            & (sig["action"].isin(("BUY", "ADD")))]
                if not entry.empty:
                    e = entry.iloc[0]
                    row["entry_global_mod"] = e.get("global_mod", np.nan)
                    row["entry_chain_mod"] = e.get("chain_mod", np.nan)
                    row["entry_agent_ms"] = e.get("agent_ms", np.nan)
            # 绝对暴露占比分解
            exps = {f: float(row[f"entry_{f}"]) for f in ATTRIBUTION_FACTORS}
            weights = _exposure_weights(exps)
            if weights is None:
                rows.append(row)
                continue
            for factor, w in zip(ATTRIBUTION_FACTORS, weights):
                row[f"pnl_{factor}"] = t["pnl"] * w
            row["factor"] = _dominant_factor(exps)
            rows.append(row)

        trades_df = pd.DataFrame(rows)
        summary = _summarize_attribution(trades_df)
        return trades_df, summary

    @staticmethod
    def plot_attribution(summary: pd.DataFrame, path: str) -> str:
        """归因柱状图：各因子贡献盈亏（元）。"""
        fig, ax = plt.subplots(figsize=(8, 5))
        s = summary.set_index("factor")["pnl"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#999999"]
        bars = ax.bar(s.index, s.values,
                      color=colors[:len(s)], edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        for b, v in zip(bars, s.values):
            ax.text(b.get_x() + b.get_width() / 2, v,
                    f"{v:,.0f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9)
        ax.set_ylabel("attributed PnL (CNY)")
        ax.set_title("Attribution: profit decomposed by factor exposure")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 因子 IC / Rank IC / IR
    # ------------------------------------------------------------------

    @staticmethod
    def compute_ic(features: pd.DataFrame, kline: pd.DataFrame,
                   forward: Union[int, str] = 60,
                   factors: Optional[Sequence[str]] = None,
                   mode: str = "auto",
                   window: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """因子 IC / Rank IC / IR 分析（双模式：横截面 / 时序）。

        :param features: 因子长表（index=DatetimeIndex，含 symbol 列 + 因子列）
        :param kline:    分钟 K 线（index=DatetimeIndex，symbol + close 列）
        :param forward:  前瞻收益窗口：分钟数（int）或 "1D"（下一交易日收盘）
        :param factors:  因子列清单（默认 DEFAULT_IC_FACTORS，仅取存在的列）
        :param mode:     "auto"（多标的→横截面）/ "cross" / "ts"（单标的时序）
        :param window:   时序模式下滚动相关窗口
        :return: (summary_df, ic_ts_df)
            summary 列：ic_mean, ic_std, ic_ir, rank_ic_mean, rank_ic_ir, n
            ic_ts     ：index=时间，列=因子，值为（Rank）IC 时序
        """
        factors = [f for f in (factors or DEFAULT_IC_FACTORS)
                   if f in features.columns]
        if not factors:
            raise ValueError("features 中无可用的 IC 因子列")
        if not {"symbol", "close"}.issubset(kline.columns):
            raise ValueError("kline 需包含 symbol 与 close 列")

        df = features.copy()
        df["_fwd_ret"] = _forward_returns(df, kline, forward)
        df = df.dropna(subset=["_fwd_ret"])
        if df.empty:
            raise ValueError("无有效的因子-前瞻收益配对数据")

        n_sym = df.groupby(df.index).nunique()["symbol"].max()
        mode = _resolve_mode(mode, n_sym)
        if mode == "cross":
            return _cross_sectional_ic(df, factors)
        return _time_series_ic(df, factors, window)

    @staticmethod
    def plot_ic_heatmap(ic_ts: pd.DataFrame, path: str,
                        buckets: int = 8) -> str:
        """IC 时序热力图：因子 × 时间桶（每桶 IC 均值）。"""
        if ic_ts.empty:
            raise ValueError("ic_ts 为空，无法绘制热力图")
        n = len(ic_ts)
        k = min(buckets, n)
        idx = np.array_split(np.arange(n), k)
        rows = []
        labels = []
        for i, ids in enumerate(idx):
            rows.append(ic_ts.iloc[ids].mean())
            labels.append(f"B{i + 1}\n{ic_ts.index[ids[0]]:%m-%d}\n{ic_ts.index[ids[-1]]:%m-%d}")
        heat = pd.DataFrame(rows, index=labels).T

        fig, ax = plt.subplots(figsize=(max(6, 1.6 * k), max(4, 0.6 * len(heat))))
        im = ax.imshow(heat.values, aspect="auto", cmap="RdBu_r",
                       vmin=-max(abs(heat.values.max()), abs(heat.values.min()), 1e-9),
                       vmax=max(abs(heat.values.max()), abs(heat.values.min()), 1e-9))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels(heat.index, fontsize=9)
        ax.set_title("Factor IC heatmap over time buckets")
        fig.colorbar(im, ax=ax, shrink=0.8, label="IC")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path


# ----------------------------------------------------------------------
# 内部实现
# ----------------------------------------------------------------------


def _exposure_weights(exps: Dict[str, float]) -> Optional[Tuple[float, ...]]:
    """绝对暴露归一化权重；不可用（NaN / 全零）返回 None。"""
    vals = [float(exps.get(f, float("nan"))) for f in ATTRIBUTION_FACTORS]
    if not all(np.isfinite(v) for v in vals):
        return None
    denom = sum(abs(v) for v in vals)
    if denom <= 1e-12:
        return None
    return tuple(abs(v) / denom for v in vals)


def _dominant_factor(exps: Dict[str, float]) -> str:
    """入场暴露绝对值最大的因子（用于逐笔标注）。"""
    return max(exps, key=lambda f: abs(float(exps[f])))


def _summarize_attribution(trades_df: pd.DataFrame) -> pd.DataFrame:
    total = float(trades_df["pnl"].sum()) if len(trades_df) else 0.0
    rows = []
    for factor in ATTRIBUTION_FACTORS + ["other"]:
        name = ATTRIBUTION_NAMES.get(factor, "other")
        if factor == "other":
            mask = trades_df["factor"] == "other"
            pnl = float(trades_df.loc[mask, "pnl"].sum()) if len(trades_df) else 0.0
            n = int(mask.sum())
        else:
            pnl = float(trades_df[f"pnl_{factor}"].sum()) if len(trades_df) else 0.0
            n = int((trades_df["factor"] == factor).sum())
        rows.append({"factor": name, "pnl": pnl,
                     "weight": (pnl / total if total != 0.0 else float("nan")),
                     "n_trades": n})
    return pd.DataFrame(rows)


def _forward_returns(df: pd.DataFrame, kline: pd.DataFrame,
                     forward: Union[int, str]) -> np.ndarray:
    """因子表每行的前瞻收益（同一标的，无前视）。

    - forward 为分钟数：ts + N 分钟后的第一根 bar 收盘价 / ts 收盘价 - 1
    - forward == "1D"：下一交易日收盘价 / ts 收盘价 - 1
    """
    pivot = kline.pivot_table(index=kline.index, columns="symbol",
                              values="close", aggfunc="last")
    pivot = pivot.loc[~pivot.index.duplicated(keep="last")].sort_index()
    sym_codes = pivot.columns.get_indexer(df["symbol"].values)
    close_now = pivot.reindex(df.index).values[np.arange(len(df)), sym_codes]

    if forward == "1D":
        daily = pivot.resample("D").last().dropna(how="all")
        days = df.index.normalize()
        t_pos = np.clip(daily.index.searchsorted(days), 0, len(daily) - 1)
        today = daily.values[t_pos, sym_codes]
        valid_today = daily.index.searchsorted(days) < len(daily)
        j = daily.index.searchsorted(days + pd.Timedelta(days=1), side="left")
        fwd = daily.values[np.clip(j, 0, len(daily) - 1), sym_codes]
        fwd = np.where((j >= len(daily)) | ~valid_today, np.nan, fwd)
    else:
        anchor = df.index + pd.Timedelta(minutes=int(forward))
        j = pivot.index.searchsorted(anchor, side="left")
        fwd = pivot.values[np.clip(j, 0, len(pivot) - 1), sym_codes]
        fwd = np.where(j >= len(pivot), np.nan, fwd)
    return np.divide(fwd, close_now, out=np.full_like(close_now, np.nan,
                                                      dtype=float),
                     where=close_now != 0) - 1.0


def _resolve_mode(mode: str, n_sym: int) -> str:
    if mode in ("cross", "ts"):
        return mode
    return "cross" if n_sym > 1 else "ts"


def _ic_summary(ic_ts: pd.DataFrame,
                rank_ts: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for f in ic_ts.columns:
        ic = ic_ts[f].dropna()
        rank = rank_ts[f].dropna() if rank_ts is not None else None
        rows.append({
            "factor": f,
            "ic_mean": float(ic.mean()) if len(ic) else float("nan"),
            "ic_std": float(ic.std(ddof=1)) if len(ic) > 1 else float("nan"),
            "ic_ir": (float(ic.mean() / ic.std(ddof=1)) if len(ic) > 1
                      and ic.std(ddof=1) > 1e-12 else float("nan")),
            "rank_ic_mean": (float(rank.mean()) if rank is not None
                             and len(rank) else float("nan")),
            "rank_ic_ir": (float(rank.mean() / rank.std(ddof=1))
                           if rank is not None and len(rank) > 1
                           and rank.std(ddof=1) > 1e-12 else float("nan")),
            "n": len(ic),
        })
    return pd.DataFrame(rows)


def _cross_sectional_ic(df: pd.DataFrame,
                        factors: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """横截面 Rank IC：每个时间戳对多标的做 Spearman/Pearson 相关。"""
    ic_rows, rank_rows = [], []
    for ts, g in df.groupby(level=0):
        if g["symbol"].nunique() < 2:
            continue
        ic, rank = {}, {}
        for f in factors:
            valid = g[[f, "_fwd_ret"]].dropna()
            if len(valid) < 2:
                ic[f] = rank[f] = np.nan
                continue
            ic[f] = valid[f].corr(valid["_fwd_ret"], method="pearson")
            rank[f] = valid[f].corr(valid["_fwd_ret"], method="spearman")
        ic_rows.append((ts, ic))
        rank_rows.append((ts, rank))
    ic_ts = pd.DataFrame([r[1] for r in ic_rows], index=[r[0] for r in ic_rows])
    rank_ts = pd.DataFrame([r[1] for r in rank_rows],
                           index=[r[0] for r in rank_rows])
    ic_ts.index.name = rank_ts.index.name = "ts"
    return _ic_summary(ic_ts, rank_ts), rank_ts


def _time_series_ic(df: pd.DataFrame, factors: Sequence[str],
                    window: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """时序相关：单标的因子与前瞻收益的滚动相关（单值相关为全样本 IC）。"""
    min_p = max(5, window // 2)
    cols = {}
    for f in factors:
        valid = df[[f, "_fwd_ret"]].dropna().sort_index()
        # Spearman = 秩变换后的 Pearson（Rolling.corr 无 method 参数，全版本兼容）
        ranks = valid.rank()
        cols[f] = ranks[f].rolling(window, min_periods=min_p).corr(
            ranks["_fwd_ret"])
    ic_ts = pd.DataFrame(cols)
    ic_ts.index.name = "ts"
    full = pd.DataFrame({
        f: df[[f, "_fwd_ret"]].dropna().sort_index()[f].corr(
            df[[f, "_fwd_ret"]].dropna().sort_index()["_fwd_ret"],
            method="spearman")
        for f in factors}, index=["rank_ic_mean"])
    rows = []
    for f in factors:
        full_ic = df[[f, "_fwd_ret"]].dropna().sort_index()
        if len(full_ic) < 2:
            rows.append({"factor": f, "ic_mean": np.nan, "ic_std": np.nan,
                         "ic_ir": np.nan, "rank_ic_mean": np.nan,
                         "rank_ic_ir": np.nan, "n": 0})
            continue
        rank_ic = full_ic[f].corr(full_ic["_fwd_ret"], method="spearman")
        roll = cols[f].dropna()
        rows.append({
            "factor": f,
            "ic_mean": float(full_ic[f].corr(full_ic["_fwd_ret"])),
            "ic_std": float(roll.std(ddof=1)) if len(roll) > 1 else float("nan"),
            "ic_ir": (float(roll.mean() / roll.std(ddof=1))
                      if len(roll) > 1 and roll.std(ddof=1) > 1e-12
                      else float("nan")),
            "rank_ic_mean": float(rank_ic),
            "rank_ic_ir": float(roll.mean() / roll.std(ddof=1))
            if len(roll) > 1 and roll.std(ddof=1) > 1e-12 else float("nan"),
            "n": len(full_ic),
        })
    return pd.DataFrame(rows), ic_ts
