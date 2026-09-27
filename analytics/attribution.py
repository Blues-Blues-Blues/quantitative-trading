"""按入场分数绝对值分摊已实现盈亏，以及因子 IC / Rank IC / IR。

描述性盈亏分摊（不代表因果贡献或统计因子收益）：
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

from analytics.metrics import (_local_time, _prepare_trade_log, _match_trade_batches, BATCH_COLUMNS)

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


OTHER_REASONS = ["unmatched_signal", "missing_factor", "non_numeric_factor",
                 "non_finite_factor", "zero_factors"]
SIGNAL_COLUMNS = ["signal_id", "timestamp", "decision_ts", "symbol", "action"] + ATTRIBUTION_FACTORS
ENTRY_COLUMNS = [f"entry_{f}" for f in ATTRIBUTION_FACTORS]
WEIGHT_COLUMNS = [f"weight_{f}" for f in ATTRIBUTION_FACTORS + ["other"]]
PNL_COLUMNS = [f"pnl_{f}" for f in ATTRIBUTION_FACTORS + ["other"]]
ATTR_BATCH_COLUMNS = BATCH_COLUMNS + ENTRY_COLUMNS + WEIGHT_COLUMNS + PNL_COLUMNS + ["other_reason"]
TRADE_COLUMNS = ["symbol", "entry_ts", "exit_ts", "shares", "entry_price", "proceeds",
                 "pnl", "sell_fill_id", "factor"] + ENTRY_COLUMNS + PNL_COLUMNS


def _factor_value(value):
    """区分缺失、非法字符串和非有限值，保留审计原因。"""
    if value is None or value is pd.NA:
        return np.nan, "missing_factor"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan, "non_numeric_factor"
    if np.isnan(numeric):
        return np.nan, "missing_factor"
    if not np.isfinite(numeric):
        return np.nan, "non_finite_factor"
    return numeric, ""


def _as_signals_frame(signals: object) -> pd.DataFrame:
    """规范化嵌套/展开信号，不修改调用方；有效同名因子矛盾和重复 ID 均报错。"""
    frame = signals.copy(deep=True) if isinstance(signals, pd.DataFrame) else pd.DataFrame(
        [s.to_dict() for s in (signals if signals is not None else [])])
    rows = []
    for row in frame.to_dict("records"):
        sid = row.get("signal_id")
        if sid is not None and pd.isna(sid):
            sid = None
        if sid == "":
            sid = None
        if sid is not None and not isinstance(sid, str):
            raise ValueError("signal_id 必须为字符串")
        ts = _local_time(row.get("timestamp"))
        decision = _local_time(row.get("decision_ts", ts))
        if pd.isna(ts) or pd.isna(decision) or ts != decision:
            raise ValueError("timestamp 与 decision_ts 必须相同且非空")
        out = dict(signal_id=sid, timestamp=ts, decision_ts=decision,
                   symbol=str(row["symbol"]), action=row["action"])
        nested = row.get("metrics")
        nested = nested if isinstance(nested, dict) else {}
        reasons = []
        for factor in ATTRIBUTION_FACTORS:
            candidates = [_factor_value(d[factor]) for d in (row, nested) if factor in d]
            valid = [v for v, reason in candidates if not reason]
            if len(valid) == 2 and abs(valid[0] - valid[1]) > 1e-12:
                raise ValueError(f"顶层与 metrics 因子冲突: {factor}")
            reason = "" if valid else next((r for r in OTHER_REASONS
                if any(error == r for _, error in candidates)), "missing_factor")
            out[factor] = valid[0] if valid else np.nan
            out[f"reason_{factor}"] = reason
            reasons.append(reason)
        out["factor_reason"] = next((r for r in OTHER_REASONS if r in reasons), "")
        if not out["factor_reason"] and sum(abs(out[f]) for f in ATTRIBUTION_FACTORS) <= 1e-12:
            out["factor_reason"] = "zero_factors"
        rows.append(out)
    result = pd.DataFrame(rows, columns=SIGNAL_COLUMNS + [f"reason_{f}" for f in ATTRIBUTION_FACTORS] + ["factor_reason"])
    result["signal_id"] = pd.Series([row["signal_id"] for row in rows], dtype=object)
    for f in ATTRIBUTION_FACTORS:
        result[f] = result[f].astype("float64")
    if result.signal_id.dropna().duplicated().any():
        raise ValueError("重复 signal_id")
    return result


def _buy_sources(frame, signals):
    """只按 ID 关联实际 BUY/ADD 成交；匹配成功后再核对股票、动作与决策时刻。"""
    lookup = {r["signal_id"]: r for r in signals.to_dict("records") if r["signal_id"] is not None}
    sources = {}
    buys = frame[frame.shares.gt(0) & frame.side.isin(["BUY", "ADD"])]
    for buy in buys.to_dict("records"):
        source = lookup.get(buy["signal_id"])
        if source is not None:
            if (source["symbol"] != buy["symbol"] or source["action"] not in ("BUY", "ADD")
                    or pd.isna(buy["decision_ts"]) or source["decision_ts"] != buy["decision_ts"]
                    or source["decision_ts"] >= buy["ts"]):
                raise ValueError(f"成交与来源信号矛盾: {buy['fill_id']}")
        sources[buy["fill_id"]] = source
    return sources


def _analyze(trade_log, signals):
    """一次账本配对后生成逐批次、逐 SELL 和质量结果，供报告层复用。"""
    trades, matched, remaining = _match_trade_batches(trade_log)
    frame = _prepare_trade_log(trade_log)
    sources = _buy_sources(frame, _as_signals_frame(signals))
    batches = []
    for row in matched.to_dict("records"):
        source = sources.get(row["buy_fill_id"])
        reason = "unmatched_signal" if source is None else source["factor_reason"]
        values = [source[f] if source is not None else np.nan for f in ATTRIBUTION_FACTORS]
        weights = [0.0, 0.0, 0.0, 1.0]
        pnls = [0.0, 0.0, 0.0, row["allocated_pnl"]]
        if not reason:
            weights = list(_exposure_weights(dict(zip(ATTRIBUTION_FACTORS, values)))) + [0.0]
            pnls = [row["allocated_pnl"] * w for w in weights]
            pnls[2] = row["allocated_pnl"] - pnls[0] - pnls[1]
        row.update(zip(ENTRY_COLUMNS, values))
        row.update(zip(WEIGHT_COLUMNS, weights))
        row.update(zip(PNL_COLUMNS, pnls))
        row["other_reason"] = reason
        batches.append(row)
    batches = pd.DataFrame(batches, columns=ATTR_BATCH_COLUMNS)
    rows = []
    by_sell = {key: group for key, group in batches.groupby("sell_fill_id", sort=False)}
    for trade in trades:
        group = by_sell[trade["sell_fill_id"]]
        row = dict(trade)
        for f in ENTRY_COLUMNS:
            row[f] = (float((group[f] * (group.matched_shares / trade["shares"])).sum())
                      if group[f].notna().all() else np.nan)
        for f in PNL_COLUMNS:
            row[f] = float(group[f].sum())
        weighted = [(group[f] * group.matched_shares).sum() for f in WEIGHT_COLUMNS]
        row["factor"] = (ATTRIBUTION_FACTORS + ["other"])[int(np.argmax(weighted))]
        if abs(sum(row[f] for f in PNL_COLUMNS) - row["pnl"]) > 1e-6:
            raise ValueError("逐 SELL 归因不守恒")
        rows.append(row)
    trades_df = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    summary = _summarize_attribution(trades_df)
    total = float(trades_df.pnl.sum())
    error = float(summary.pnl.sum() - total)
    if abs(error) > 1e-6:
        raise ValueError("总归因不守恒")
    good = batches.other_reason.eq("")
    shares = int(batches.matched_shares.sum())
    abs_pnl = float(batches.allocated_pnl.abs().sum())
    reasons = {reason: dict(rows=int((batches.other_reason == reason).sum()),
                           shares=int(batches.loc[batches.other_reason == reason, "matched_shares"].sum()),
                           pnl=float(batches.loc[batches.other_reason == reason, "allocated_pnl"].sum()))
               for reason in OTHER_REASONS}
    quality = dict(
        matched_buy_fill_rate=sum(v is not None for v in sources.values()) / len(sources) if sources else None,
        attributed_sold_share_rate=int(batches.loc[good, "matched_shares"].sum()) / shares if shares else None,
        attributed_abs_pnl_rate=float(batches.loc[good, "allocated_pnl"].abs().sum()) / abs_pnl if abs_pnl else None,
        total_realized_pnl=total, factor_pnl={f: float(batches[f"pnl_{f}"].sum()) for f in ATTRIBUTION_FACTORS},
        other_pnl=float(batches.pnl_other.sum()), other_reasons=reasons,
        remaining_open_shares=remaining, conservation_error=error,
        status="no_closed_trades" if not trades else "complete" if good.all() else "partial")
    return trades_df, summary, batches, quality


class AttributionEngine:
    """收益归因与因子预测能力分析引擎。"""

    # ------------------------------------------------------------------
    # 归因：因子暴露分解
    # ------------------------------------------------------------------

    @staticmethod
    def attribute(trade_log: pd.DataFrame, signals: object) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """按入场分数绝对值分摊已实现盈亏；返回每 SELL 一行及四分类汇总。"""
        trades, summary, _, _ = _analyze(trade_log, signals)
        return trades, summary

    @staticmethod
    def attribute_batches(trade_log: pd.DataFrame, signals: object) -> pd.DataFrame:
        """返回保留来源 ID、带符号原始因子、权重与 other 原因的批次表。"""
        return _analyze(trade_log, signals)[2]

    @staticmethod
    def quality_report(trade_log: pd.DataFrame, signals: object, batches_df: pd.DataFrame) -> dict:
        """重新核对传入批次表属于该账本及信号，返回覆盖率、剩余股数和守恒诊断。"""
        _, _, expected, quality = _analyze(trade_log, signals)
        try:
            pd.testing.assert_frame_equal(expected, batches_df, check_dtype=False, atol=1e-6, rtol=0)
        except AssertionError as exc:
            raise ValueError("批次表与账本/信号不一致") from exc
        return quality

    @staticmethod
    def plot_attribution(summary: pd.DataFrame, path: str, quality: Optional[dict] = None) -> str:
        """归因柱状图：各因子贡献盈亏（元）。"""
        fig, ax = plt.subplots(figsize=(8, 5))
        if ((quality and quality["status"] == "no_closed_trades")
                or ("n_trades" in summary and summary.n_trades.sum() == 0)):
            ax.text(0.5, 0.5, "No closed trades / no realized PnL", ha="center", transform=ax.transAxes)
            fig.savefig(path, dpi=120)
            plt.close(fig)
            return path
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
        title = "Realized PnL * abs(entry score) / sum(abs(entry scores))"
        if quality:
            rate = quality["attributed_sold_share_rate"]
            title += f"\n{quality['status']} | sold-share coverage: {rate:.1%}"
        ax.set_title(title)
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
    # 等价数值计算，防止有限的大数相加溢出；不改变分摊的绝对值占比口径。
    scale = max(abs(v) for v in vals)
    scaled = [abs(v) / scale for v in vals]
    return tuple(v / sum(scaled) for v in scaled)


def _dominant_factor(exps: Dict[str, float]) -> str:
    """入场暴露绝对值最大的因子（用于逐笔标注）。"""
    return max(exps, key=lambda f: abs(float(exps[f])))


def _summarize_attribution(trades_df: pd.DataFrame) -> pd.DataFrame:
    """把因子归因结果汇总为可报告的统计量；对有效样本不足或全常数的输入保留显式缺失值，避免把无定义结果误报为零。"""
    total = float(trades_df["pnl"].sum()) if len(trades_df) else 0.0
    rows = []
    for factor in ATTRIBUTION_FACTORS + ["other"]:
        name = ATTRIBUTION_NAMES.get(factor, "other")
        pnl = float(trades_df[f"pnl_{factor}"].sum()) if len(trades_df) else 0.0
        n = int((trades_df["factor"] == factor).sum()) if len(trades_df) else 0
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
    """解析归因模式并统一别名；集中处理可选值，避免不同调用方采用不同默认口径。"""
    if mode in ("cross", "ts"):
        return mode
    return "cross" if n_sym > 1 else "ts"


def _ic_summary(ic_ts: pd.DataFrame,
                rank_ts: Optional[pd.DataFrame]) -> pd.DataFrame:
    """计算各因子的 IC 汇总统计；只使用成对有效样本，并按模块约定返回样本数、均值和稳定性指标。"""
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
