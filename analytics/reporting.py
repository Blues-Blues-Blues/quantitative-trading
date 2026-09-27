"""单次回测报告生命周期；只缓冲一个交易日的持仓及暴露，跨日写 Parquet。"""

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analytics.attribution import AttributionEngine, _analyze
from analytics.exposure import (ExposureAnalyzer, FACTORS, POSITION_COLUMNS,
                                CONTRIBUTION_COLUMNS, EXPOSURE_COLUMNS)
from analytics.performance import PerformanceAnalyzer

logger = logging.getLogger(__name__)


def _json_value(value):
    """JSON 禁止 NaN/Inf；不覆盖表格原始精度。"""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, Path)):
        return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__, "parameters": _json_value(vars(value))} if hasattr(value, "__dict__") else str(value)


class AnalysisReportWriter:
    """实例对应一次引擎运行；构造才创建目录，模块导入无副作用。"""

    def __init__(self, report_dir="analytics/reports", *, mode="smoke", start=None,
                 end=None, symbols=(), metadata=None):
        now = datetime.now(timezone.utc)
        run_id = now.strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
        name = re.sub(r"[^a-zA-Z0-9_.-]", "-", f"{mode}_{start}_{end}_{run_id}")
        root = Path(__file__).resolve().parents[1]
        self.path = (root / Path(report_dir) / name).resolve()
        self.path.mkdir(parents=True, exist_ok=False)
        root = Path(__file__).resolve().parents[1]
        files = ["strategy/signals.py", "engine/backtest.py", "engine/portfolio.py",
                 "engine/execution.py", "engine/risk_control.py", "analytics/metrics.py",
                 "analytics/attribution.py", "analytics/exposure.py", "analytics/reporting.py",
                 "analytics/performance.py", "main.py", "data/aligner.py", "data/real_loader.py",
                 "indicators/feature_engine.py", "indicators/agent_profiling.py",
                 "indicators/environment.py", "indicators/microstructure.py",
                 "analytics/visualization_data.py", "analytics/html_report.py",
                 "analytics/templates/backtest_report.html", "analytics/templates/backtest_report.js"]
        self.manifest = dict(metadata or {})
        self.manifest.setdefault("strategy_parameters", {})
        self.manifest.setdefault("execution_parameters", {})
        self.manifest.setdefault("initial_cash", None)
        self.manifest.setdefault("industry_mapping_summary", "unavailable")
        self.manifest.setdefault("data_source", "unavailable")
        self.manifest.setdefault("data_fingerprint", "unavailable")
        self.manifest.update(schema_version=1, run_id=run_id, generated_at=now.isoformat(),
                             mode=mode, start=start, end=end, symbols=list(symbols), status="running",
                             methods=["average_cost_fifo_source_allocation", "signed_score_equity_weighted_exposure"],
                             visualization=dict(schema_version=1, status="pending", html_file=None, warnings=[], error=None),
                             code_content_hashes={f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files if (root / f).exists()})
        self._day = None
        self._last_ts = None
        self._positions, self._contributions, self._exposures = [], [], []
        self._daily_close = []
        self._position_days = []
        self._last_equity = None
        self._stats = {f: dict(n=0, sum=0.0, absolute_sum=0.0, peak_abs=0.0, partial_bars=0) for f in FACTORS}
        self._finished = False
        self._write_json("manifest.json", self.manifest)

    def _write_json(self, name, value):
        (self.path / name).write_text(json.dumps(_json_value(value), ensure_ascii=False,
                                               indent=2, allow_nan=False), encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc is not None:
            self.fail(exc)
        elif not self._finished:
            self.fail(RuntimeError("报告未 finalize"))

    def fail(self, exc):
        self.manifest.update(status="failed", error=str(exc))
        if self.manifest["visualization"]["status"] == "pending":
            self.manifest["visualization"].update(status="failed", error=str(exc))
        self._write_json("manifest.json", self.manifest)

    def snapshot(self, ts, positions, total_equity, factor_values):
        """先清空上一交易日缓冲，再接收新一天快照；错误立即使 manifest 失败。"""
        try:
            if self._finished:
                raise RuntimeError("报告已经 finalize")
            if self.manifest["status"] == "failed":
                raise RuntimeError("报告已失败，不能继续接收快照")
            from analytics.metrics import _local_time
            ts = _local_time(ts)
            if self._last_ts is not None and pd.Timestamp(ts) <= self._last_ts:
                raise ValueError("快照时间必须严格递增")
            day = _local_time(ts).normalize()
            if self._day is not None and day < self._day:
                raise ValueError("快照日期必须递增")
            if self._day is not None and day != self._day:
                self._flush_day()
            self._day = day
            equity = pd.DataFrame([dict(ts=ts, total_equity=total_equity)])
            contributions, exposure = ExposureAnalyzer.compute(positions, factor_values, equity)
            position_copy = positions.copy(deep=True)
            position_copy["ts"] = pd.to_datetime(position_copy.ts.map(_local_time))
            self._positions.append(position_copy)
            self._contributions.append(contributions)
            self._exposures.append(exposure)
            self._last_ts = pd.Timestamp(ts)
            self._last_equity = total_equity
            for row in exposure.itertuples(index=False):
                stats = self._stats[row.factor]
                if np.isfinite(row.exposure):
                    stats["n"] += 1
                    stats["sum"] += row.exposure
                    stats["absolute_sum"] += abs(row.exposure)
                    stats["peak_abs"] = max(stats["peak_abs"], abs(row.exposure))
                if row.status == "partial":
                    stats["partial_bars"] += 1
        except Exception as exc:
            self.fail(exc)
            raise

    def _flush_day(self):
        if self._day is None or not self._exposures:
            return
        for folder, buffer, columns in (
            ("positions", self._positions, POSITION_COLUMNS),
            ("exposure_contributions", self._contributions, CONTRIBUTION_COLUMNS),
            ("exposure_timeseries", self._exposures, EXPOSURE_COLUMNS),
        ):
            target = self.path / folder
            target.mkdir(exist_ok=True)
            frame = pd.concat(buffer, ignore_index=True).reindex(columns=columns)
            # 空仓日与有仓日使用相同 Parquet 类型，避免首日 null schema 导致分区读取失败。
            types = {c: "float64" for c in columns if c not in ("ts", "symbol", "factor", "status", "shares")}
            types.update({c: "string" for c in ("symbol", "factor", "status") if c in columns})
            types["ts"] = "datetime64[ns]"
            if "shares" in columns:
                types["shares"] = "int64"
            frame = frame.astype(types)
            frame.to_parquet(target / f"date={self._day:%Y-%m-%d}.parquet", index=False)
        # 长期只保留每日最后一根 Bar 的三个因子，用于绘图。
        self._daily_close.append(self._exposures[-1].copy())
        # 最后快照即使为空也是真实完整状态，不能回找之前的非空持仓。
        self._position_days.append((self._last_ts, self._last_equity, self._positions[-1].copy()))
        self._positions.clear()
        self._contributions.clear()
        self._exposures.clear()

    def finalize(self, trade_log, signals, equity_curve, *, kline=None):
        """使用同一次回测的日志完成归因及图表，绝不重新运行引擎。"""
        try:
            if self._finished:
                raise RuntimeError("报告已经 finalize")
            if self.manifest["status"] == "failed":
                raise RuntimeError("报告已失败，不能 finalize 为成功")
            self._flush_day()
            trades, summary, batches, quality = _analyze(trade_log, signals)
            for name, frame in (("attribution_trades", trades), ("attribution_batches", batches),
                                ("attribution_summary", summary),
                                ("unattributed_batches", batches[batches.other_reason.ne("")])):
                frame.to_csv(self.path / f"{name}.csv", index=False, encoding="utf-8", na_rep="")
            quality["exposure_summary"] = {
                f: dict(mean=s["sum"] / s["n"] if s["n"] else None,
                        mean_abs=s["absolute_sum"] / s["n"] if s["n"] else None,
                        peak_abs=s["peak_abs"] if s["n"] else None,
                        valid_bars=s["n"], partial_bars=s["partial_bars"])
                for f, s in self._stats.items()}
            self._write_json("quality.json", quality)
            AttributionEngine.plot_attribution(summary, str(self.path / "attribution.png"), quality)
            daily = pd.concat(self._daily_close, ignore_index=True) if self._daily_close else pd.DataFrame(columns=EXPOSURE_COLUMNS)
            self._plot_exposure(daily)
            # Dashboard 同样仅绘制日末点；完整分钟数据已在分块文件及引擎曲线中。
            curve = equity_curve.sort_values("ts").groupby(pd.to_datetime(equity_curve.ts).dt.normalize(), sort=True).tail(1)
            PerformanceAnalyzer.plot_report(curve, trade_log, summary, path=str(self.path / "dashboard.png"),
                                            attribution_quality=quality, exposure_daily=daily)
            self.manifest.update(status=quality["status"], quality_file="quality.json")
            if kline is None:
                self.manifest["visualization"].update(status="skipped", error=None,
                                                     reason="未提供行情（兼容三参数调用）")
            else:
                from analytics.visualization_data import build_visualization_data
                from analytics.html_report import write_backtest_html
                rows = []
                symbols = self.manifest["symbols"]
                for ts, equity, frame in self._position_days:
                    if frame.symbol.duplicated().any() or not frame.symbol.isin(symbols).all():
                        raise ValueError("快照含重复或集合外股票")
                    records = {r["symbol"]: r for r in frame.to_dict("records")}
                    for symbol in symbols:
                        rows.append(records.get(symbol, dict(ts=ts, symbol=symbol, shares=0,
                                    mark_price=None, market_value=0, total_equity=equity)))
                data = build_visualization_data(kline=kline, trade_log=trade_log, equity_curve=equity_curve,
                    positions_daily=pd.DataFrame(rows, columns=POSITION_COLUMNS), symbols=symbols,
                    initial_cash=self.manifest["initial_cash"], metadata=self.manifest)
                path = write_backtest_html(data, self.path / "backtest.html")
                self.manifest["visualization"].update(status="partial" if data["warnings"] else "complete",
                    html_file="backtest.html", warnings=data["warnings"], error=None)
            self._write_json("manifest.json", self.manifest)
            self._finished = True
            if kline is not None:
                logger.info("HTML 报告：%s | attribution=%s visualization=%s", path,
                            quality["status"], self.manifest["visualization"]["status"])
                print(f"HTML 报告：{path} | attribution={quality['status']} visualization={self.manifest['visualization']['status']}")
            log = logger.warning if quality["status"] == "partial" else logger.info
            log("分析报告 %s: status=%s realized=%s attributed=%s other=%s buy_fill_coverage=%s open_shares=%s",
                self.path, quality["status"], quality["total_realized_pnl"],
                sum(quality["factor_pnl"].values()), quality["other_pnl"],
                quality["matched_buy_fill_rate"], quality["remaining_open_shares"])
            return quality
        except Exception as exc:
            if kline is not None and not self._finished:
                self.manifest["visualization"].update(status="failed", error=str(exc))
            self.fail(exc)
            raise

    def _plot_exposure(self, daily):
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for factor in FACTORS:
            group = daily[daily.factor.eq(factor)]
            axes[0].plot(group.ts, group.exposure, marker=".", label=factor)
            axes[1].plot(group.ts, group.coverage, marker=".", label=factor)
        axes[0].set_title("Close exposure = sum(market_value / equity * signed score)")
        axes[0].set_ylabel("signed score exposure")
        axes[0].legend()
        axes[1].set_ylabel("market-value coverage")
        axes[1].set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(self.path / "exposure.png", dpi=120)
        plt.close(fig)
