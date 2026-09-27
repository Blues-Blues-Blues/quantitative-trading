"""T08–T11：小型真实 Signal→引擎→报告链路，不运行真实数据或参数寻优。"""
import json

import numpy as np
import pandas as pd
import pytest

from analytics.attribution import AttributionEngine
from analytics.metrics import evaluate
from analytics.reporting import AnalysisReportWriter
from data.dataslice import DataSlice
from engine.backtest import BacktestEngine
from engine.execution import ExecutionCost
from engine.portfolio import Account
from engine.risk_control import PositionSizer
from strategy.signals import Signal


def small_engine(sink=None, scenario="cycle"):
    from test_backtest_engine import mk_kline
    axis = pd.to_datetime(["2024-01-02 10:00", "2024-01-02 10:01", "2024-01-02 10:02",
                           "2024-01-03 10:00", "2024-01-03 10:01"])
    ds = DataSlice(kline=mk_kline(axis), meta={"symbols": ["600000"]})
    actions = ["BUY", "HOLD", "HOLD", "SELL", "HOLD"]
    if scenario == "open":
        actions[3] = "HOLD"
    if scenario == "flat":
        actions = ["HOLD"] * len(axis)
    signals = [Signal("600000", ts, action, "S_push",
                      dict(global_mod=-.5, chain_mod=.3, agent_ms=.2, target_weight=.2))
               for ts, action in zip(axis, actions)]
    return BacktestEngine(Account(1e6), ExecutionCost(), PositionSizer(), ds,
                          signals, snapshot_sink=sink)


def writer_at(path):
    return AnalysisReportWriter(path, mode="smoke", start="2024-01-02", end="2024-01-03",
                                symbols=["600000"], metadata={"initial_cash": 1e6})


def strict_json(path):
    def reject_constant(value):
        raise AssertionError(f"Non-standard JSON constant: {value}")
    return json.loads(path.read_text(encoding="utf8"), parse_constant=reject_constant)


def test_two_day_reports_flush_and_traceability(tmp_path, monkeypatch):
    writer = writer_at(tmp_path)
    flushed = []
    original = writer._flush_day

    def flush():
        original()
        assert not writer._positions and not writer._contributions and not writer._exposures
        flushed.append(writer._day)
    monkeypatch.setattr(writer, "_flush_day", flush)
    engine = small_engine(writer.snapshot)
    log, curve = engine.run()
    assert len(flushed) == 1  # 第二天第一次回调已清空前一天
    quality = writer.finalize(log, engine.generated_signals, curve)
    assert len(flushed) == 2
    assert quality["status"] == "complete"
    for name in ["attribution_trades.csv", "attribution_batches.csv", "attribution_summary.csv",
                 "unattributed_batches.csv", "quality.json", "manifest.json",
                 "attribution.png", "exposure.png", "dashboard.png"]:
        assert (writer.path / name).stat().st_size > 0
    for folder in ["positions", "exposure_contributions", "exposure_timeseries"]:
        assert len(list((writer.path / folder).glob("*.parquet"))) == 2
    exposure = pd.concat([pd.read_parquet(p) for p in sorted((writer.path / "exposure_timeseries").glob("*.parquet"))])
    assert len(exposure) == len(curve) * 3
    assert exposure.iloc[:3].status.tolist() == ["flat"] * 3
    assert exposure.iloc[-3:].status.tolist() == ["flat"] * 3
    batch = pd.read_csv(writer.path / "attribution_batches.csv").iloc[0]
    buy = log[log.fill_id.eq(batch.buy_fill_id)].iloc[0]
    source = next(s for s in engine.generated_signals if s.signal_id == batch.buy_signal_id)
    assert buy.decision_ts == source.timestamp < buy.ts
    manifest = strict_json(writer.path / "manifest.json")
    assert manifest["schema_version"] == 1 and manifest["status"] == "complete"
    assert manifest["data_fingerprint"] == "unavailable"
    assert len(manifest["code_content_hashes"]["engine/backtest.py"]) == 64
    assert strict_json(writer.path / "quality.json")["matched_buy_fill_rate"] == 1


def test_reproducible_on_off_results_and_distinct_directories(tmp_path):
    runs = []
    paths = []
    for enabled in (False, True, True):
        writer = writer_at(tmp_path) if enabled else None
        engine = small_engine(writer.snapshot if writer else None)
        log, curve = engine.run()
        runs.append((log, curve, pd.DataFrame([evaluate(curve, log)]),
                     [s.signal_id for s in engine.generated_signals]))
        if writer:
            writer.finalize(log, engine.generated_signals, curve)
            paths.append(writer.path)
    assert paths[0] != paths[1]
    for run in runs[1:]:
        for original, current in zip(runs[0][:3], run[:3]):
            pd.testing.assert_frame_equal(original, current, check_exact=True)
        assert run[3] == runs[0][3]


@pytest.mark.parametrize("scenario", ["flat", "open"])
def test_no_closed_trades_and_remaining_holdings(tmp_path, scenario):
    writer = writer_at(tmp_path)
    engine = small_engine(writer.snapshot, scenario)
    log, curve = engine.run()
    quality = writer.finalize(log, engine.generated_signals, curve)
    assert quality["status"] == "no_closed_trades"
    assert quality["attributed_sold_share_rate"] is None
    assert quality["attributed_abs_pnl_rate"] is None
    trades = pd.read_csv(writer.path / "attribution_trades.csv")
    batches = pd.read_csv(writer.path / "attribution_batches.csv")
    assert trades.empty and "pnl" in trades
    assert batches.empty and "other_reason" in batches
    exposure = pd.concat([pd.read_parquet(p) for p in (writer.path / "exposure_timeseries").glob("*.parquet")])
    assert len(exposure) == len(curve) * 3
    assert quality["total_realized_pnl"] == 0
    if scenario == "flat":
        assert quality["matched_buy_fill_rate"] is None
        assert (exposure.exposure == 0).all()
    else:
        assert quality["remaining_open_shares"] == sum(p.shares for p in engine.account.positions.values()) > 0
        assert exposure.exposure.ne(0).any()
        assert engine.account.unrealized_pnl() != 0
    strict_json(writer.path / "quality.json")
    assert strict_json(writer.path / "manifest.json")["status"] == "no_closed_trades"


def test_report_failure_marks_manifest(tmp_path):
    writer = writer_at(tmp_path)
    with pytest.raises(RuntimeError, match="deliberate"):
        with writer:
            raise RuntimeError("deliberate")
    assert strict_json(writer.path / "manifest.json")["status"] == "failed"


def test_partial_report_warns_and_json_handles_nonfinite(tmp_path, caplog):
    writer = writer_at(tmp_path)
    writer.manifest["optional_metric"] = np.inf
    engine = small_engine(writer.snapshot)
    engine.signals[0].metrics.pop("agent_ms")
    log, curve = engine.run()
    quality = writer.finalize(log, engine.generated_signals, curve)
    assert quality["status"] == "partial"
    assert "status=partial" in caplog.text
    assert strict_json(writer.path / "manifest.json")["optional_metric"] is None


def test_main_no_report_skips_writer(tmp_path, monkeypatch):
    import main
    import analytics.reporting
    monkeypatch.setattr(analytics.reporting, "AnalysisReportWriter", lambda *a, **k: pytest.fail("writer disabled"))
    monkeypatch.setattr(main, "_plot_smoke_charts", lambda ds: [])
    root = tmp_path / "disabled-reports"
    main.run_smoke(analysis_report=False, report_dir=str(root))
    assert not root.exists()


def test_rejected_buy_never_creates_exposure(tmp_path):
    writer = writer_at(tmp_path)
    engine = small_engine(writer.snapshot)
    ts = pd.Timestamp("2024-01-02 10:01")
    # 修改已构建的撮合表，制造开盘涨停；其余信号只有 HOLD/SELL。
    bar = engine._kline_by_ts[ts]
    bar.loc["600000", "open"] = bar.loc["600000", "up_limit"]
    log, curve = engine.run()
    quality = writer.finalize(log, engine.generated_signals, curve)
    assert log.iloc[0].reason == "limit_up" and pd.isna(log.iloc[0].fill_id)
    assert log.shares.sum() == 0
    assert quality["matched_buy_fill_rate"] is None
    assert quality["status"] == "no_closed_trades"
    for file in (writer.path / "exposure_timeseries").glob("*.parquet"):
        table = pd.read_parquet(file)
        assert table.status.eq("flat").all() and table.exposure.eq(0).all()


def test_snapshot_failure_and_conservation_failure_are_not_complete(tmp_path, monkeypatch):
    writer = writer_at(tmp_path)
    from test_exposure import exposure_inputs
    positions, factors, equity = exposure_inputs()
    with pytest.raises(ValueError, match="权益"):
        writer.snapshot(equity.ts.iloc[0], positions, 0, factors)
    assert strict_json(writer.path / "manifest.json")["status"] == "failed"
    with pytest.raises(RuntimeError, match="失败"):
        writer.finalize(pd.DataFrame(), [], equity)

    from test_analytics import allocation_ledger
    import analytics.attribution as attribution
    original = attribution._summarize_attribution
    def corrupt(trades):
        summary = original(trades)
        summary.loc[0, "pnl"] += 1
        return summary
    monkeypatch.setattr(attribution, "_summarize_attribution", corrupt)
    log, signals = allocation_ledger()
    with pytest.raises(ValueError, match="守恒"):
        AttributionEngine.attribute(log, signals)


def test_empty_and_held_partitions_have_same_arrow_schema(tmp_path):
    import pyarrow.parquet as pq
    writer = writer_at(tmp_path)
    engine = small_engine(writer.snapshot, "open")
    # 第一天全空仓，第二天买入并保留；验证按首分区推断类型也能读后续非空分区。
    engine._signal_by_ts[engine._axis[0]][0].action = "HOLD"
    engine._signal_by_ts[engine._axis[3]][0].action = "BUY"
    log, curve = engine.run()
    writer.finalize(log, engine.generated_signals, curve)
    for folder in ("positions", "exposure_contributions", "exposure_timeseries"):
        paths = sorted((writer.path / folder).glob("*.parquet"))
        assert pq.read_schema(paths[0]) == pq.read_schema(paths[1])
