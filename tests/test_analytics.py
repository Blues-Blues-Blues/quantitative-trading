"""绩效评估、收益归因与实时流输出模块单元测试（合成 mock 数据，不联网）。

覆盖：
- StreamLogger：JSONL 落盘 / 内存生成器 / 标准字段结构
- PerformanceAnalyzer.analyze：年化收益率、夏普、Calmar、Sortino、回撤、
  平均持仓周期、胜率/盈亏比、日收益偏度/峰度（精确值与端到端）
- 参数敏感度热力图（Optuna study / DataFrame 两种输入）
- AttributionEngine：因子暴露分解归因（盈亏守恒）、IC/Rank IC/IR 双模式
- 20 日人工复盘清单导出（Excel 三 sheet / CSV）
- 四子图 Dashboard 落盘
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_optimizer import TRADE_PARAMS, bull_slice  # noqa: E402

from analytics.attribution import AttributionEngine  # noqa: E402
from analytics.metrics import closed_trades  # noqa: E402
from analytics.metrics import match_trade_batches
from analytics.attribution import _as_signals_frame
from analytics.performance import PerformanceAnalyzer  # noqa: E402
from analytics.real_time_stream import (  # noqa: E402
    REQUIRED_FIELDS, StreamLogger, to_stream_frame,
)
from engine.backtest import BacktestEngine  # noqa: E402
from engine.execution import ExecutionCost  # noqa: E402
from engine.portfolio import Account  # noqa: E402
from engine.risk_control import PositionSizer  # noqa: E402
from optimizer.bayesian_opt import StrategyOptimizer  # noqa: E402
from strategy.signals import Signal  # noqa: E402

_MAPPING = {"600000": "银行"}


@pytest.fixture(scope="module")
def pipeline():
    """跑通端到端回测：返回 (equity_curve, trade_log)。

    opt.backtest() 返回同一次运行的曲线与日志。
    """
    ds = bull_slice()
    opt = StrategyOptimizer(data=ds, symbol_to_industry=_MAPPING,
                            account_kwargs={"initial_cash": 1e8})
    _, engine = opt.backtest(ds, TRADE_PARAMS)
    assert engine.snapshot_sink is None  # 固定参数单点回测不启用报告，不调用 optimize。
    return engine.equity_curve, engine.trade_log, engine.generated_signals


# ----------------------------------------------------------------------
# 实时流（JSONL + 生成器）
# ----------------------------------------------------------------------

def _mk_signals():
    return [
        Signal(symbol="600000", timestamp=pd.Timestamp("2024-01-02 10:00"),
               action="HOLD", state="S_noise",
               metrics={"final_ms": -0.2, "global_mod": 0.1, "chain_mod": 0.0,
                        "capital_purity": 0.2, "agent_ms": -0.1}),
        Signal(symbol="600000", timestamp=pd.Timestamp("2024-01-02 10:30"),
               action="BUY", state="S_push",
               metrics={"final_ms": 1.5, "global_mod": 0.7, "chain_mod": 0.3,
                        "capital_purity": 0.55, "agent_ms": 0.5}),
    ]


class TestStreamLogger:

    def test_jsonl_roundtrip(self, tmp_path):
        path = tmp_path / "stream.jsonl"
        with StreamLogger(path) as logger:
            n = logger.log_frame(_mk_signals())
            assert n == 2
        df = StreamLogger.read(path)
        assert len(df) == 2
        for field in REQUIRED_FIELDS:
            assert field in df.columns
        row = df.iloc[1]
        assert row["Action"] == "BUY"
        assert row["State"] == "S_push"
        assert float(row["Final_MS"]) == pytest.approx(1.5)
        assert float(row["Global_Mod"]) == pytest.approx(0.7)
        assert str(row["timestamp"]).startswith("2024-01-02T10:30")

    def test_generator(self):
        rows = list(StreamLogger.generator(_mk_signals()))
        assert len(rows) == 2
        assert rows[0]["symbol"] == "600000"
        assert set(REQUIRED_FIELDS) <= set(rows[0].keys())

    def test_to_stream_frame(self):
        df = to_stream_frame(_mk_signals())
        assert list(df.columns[:len(REQUIRED_FIELDS)]) == REQUIRED_FIELDS
        assert len(df) == 2

    def test_log_without_path_returns_dict(self):
        logger = StreamLogger()
        row = logger.log(_mk_signals()[0])
        assert row["Action"] == "HOLD"
        assert logger.count == 0  # 未落盘


# ----------------------------------------------------------------------
# 绩效指标（精确值 + 端到端）
# ----------------------------------------------------------------------

def _curve(equity: list, days=("2024-01-02", "2024-01-03", "2024-01-04")):
    """每天 4 根 bar；equity 每个元素为当日净值（广播到当日全部 bar）。"""
    idx = pd.DatetimeIndex([])
    for d in days:
        idx = idx.append(pd.date_range(f"{d} 10:00", periods=4, freq="30min"))
    vals = np.repeat(equity, 4)
    assert len(idx) == len(vals)
    return pd.DataFrame({"ts": idx, "total_equity": vals})


class TestPerformanceAnalyzer:

    def test_analyze_end_to_end(self, pipeline):
        curve, log, sig = pipeline
        m = PerformanceAnalyzer.analyze(curve, log)
        assert m["n_trades"] >= 1
        assert 0.0 <= m["win_rate"] <= 1.0
        for key in ("annual_return", "sharpe", "calmar", "sortino",
                    "max_drawdown", "avg_holding_minutes", "profit_loss_ratio",
                    "total_pnl", "daily_skew", "daily_kurtosis"):
            assert key in m

    def test_annual_return_precise(self):
        # 3 个日末值 100 → 110，年化 = (1.1)^(244/3) - 1
        curve = _curve([100.0, 105.0, 110.0])
        assert PerformanceAnalyzer.analyze(curve, pd.DataFrame())["annual_return"] \
            == pytest.approx(1.1 ** (244 / 3) - 1)

    def test_calmar_zero_drawdown_inf(self):
        # 单调上涨 → 回撤 0 → Calmar = +inf（盈利）
        curve = _curve([100.0, 101.0, 103.0])
        assert np.isposinf(PerformanceAnalyzer.analyze(curve, pd.DataFrame())["calmar"])

    def test_sortino_no_downside_inf(self):
        curve = _curve([100.0, 101.0, 103.0])
        assert np.isposinf(PerformanceAnalyzer.analyze(curve, pd.DataFrame())["sortino"])

    def test_avg_holding_period(self, pipeline):
        curve, log, sig = pipeline
        m = PerformanceAnalyzer.analyze(curve, log)
        assert np.isfinite(m["avg_holding_minutes"]) and m["avg_holding_minutes"] >= 0

    def test_analyze_empty_log(self):
        curve = _curve([100.0, 100.0, 100.0])
        m = PerformanceAnalyzer.analyze(curve, pd.DataFrame())
        assert m["n_trades"] == 0
        assert np.isnan(m["win_rate"])

    def test_sensitivity_heatmap_dataframe(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "w_ofss": rng.uniform(0.2, 0.6, 60),
            "w_cps": rng.uniform(0.1, 0.5, 60),
            "value": rng.normal(1.0, 0.5, 60),
        })
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        PerformanceAnalyzer.sensitivity_heatmap(df, ax=ax)
        assert ax.get_title() == "Parameter sensitivity (objective mean)"
        plt.close(fig)

    def test_sensitivity_heatmap_insufficient(self):
        import matplotlib.pyplot as plt
        df = pd.DataFrame({"w_ofss": [0.3] * 6, "w_cps": [0.2] * 6,
                           "value": [1.0] * 6})
        fig, ax = plt.subplots()
        PerformanceAnalyzer.sensitivity_heatmap(df, ax=ax)
        assert "insufficient" in ax.texts[0].get_text()
        plt.close(fig)


# ----------------------------------------------------------------------
# 归因：因子暴露分解 + IC
# ----------------------------------------------------------------------

class TestAttributionEngine:

    def test_attribute_conserves_pnl(self, pipeline):
        curve, log, sig = pipeline
        trades = closed_trades(log)
        trades_df, summary = AttributionEngine.attribute(log, sig)
        assert len(trades_df) == len(trades)
        # 盈亏守恒：逐笔拆解之和 = 归因合计 = 总盈亏
        assert float(trades_df["pnl"].sum()) == pytest.approx(
            float(summary["pnl"].sum()))
        assert float(summary["pnl"].sum()) == pytest.approx(
            sum(t["pnl"] for t in trades))
        # 入场快照可得 → 应有明确主导因子（非 other）
        assert (trades_df["factor"] != "other").all()
        # 每笔 pnl_global + pnl_chain + pnl_agent == pnl
        p = trades_df
        assert np.allclose(p["pnl_global_mod"] + p["pnl_chain_mod"]
                           + p["pnl_agent_ms"], p["pnl"])

    def test_attribute_no_signals_other(self, pipeline):
        curve, log, sig = pipeline
        _, summary = AttributionEngine.attribute(log, pd.DataFrame())
        other = summary[summary["factor"] == "other"].iloc[0]
        assert other["n_trades"] == len(closed_trades(log))

    def test_ic_time_series_mode(self):
        kline, features = _mk_ic_data(n_symbols=1)
        summary, ic_ts = AttributionEngine.compute_ic(
            features, kline, forward=60, window=8)
        assert {"final_ms", "inst_flow"} <= set(summary["factor"])
        row = summary[summary["factor"] == "final_ms"].iloc[0]
        assert np.isfinite(row["rank_ic_mean"])
        assert not ic_ts.empty

    def test_ic_cross_sectional_mode(self):
        kline, features = _mk_ic_data(n_symbols=3)
        summary, ic_ts = AttributionEngine.compute_ic(
            features, kline, forward="1D")
        assert not ic_ts.empty
        row = summary[summary["factor"] == "final_ms"].iloc[0]
        assert np.isfinite(row["rank_ic_mean"])
        assert np.isfinite(row["ic_mean"]) or np.isfinite(row["ic_ir"])

    def test_plot_attribution(self, pipeline, tmp_path):
        curve, log, sig = pipeline
        _, summary = AttributionEngine.attribute(
            log, sig)
        path = AttributionEngine.plot_attribution(
            summary, str(tmp_path / "attr.png"))
        assert os.path.exists(path)

    def test_plot_ic_heatmap(self, tmp_path):
        kline, features = _mk_ic_data(n_symbols=3)
        _, ic_ts = AttributionEngine.compute_ic(features, kline, forward="1D")
        path = AttributionEngine.plot_ic_heatmap(
            ic_ts, str(tmp_path / "ic.png"), buckets=4)
        assert os.path.exists(path)


# ----------------------------------------------------------------------
# 20 日复盘清单 + Dashboard
# ----------------------------------------------------------------------

class TestReviewExport:

    def test_export_xlsx(self, pipeline, tmp_path):
        curve, log, sig = pipeline
        path = PerformanceAnalyzer.export_review_slices(
            bull_slice(), log, days=20, path=str(tmp_path / "review.xlsx"))
        assert os.path.exists(path)
        import openpyxl
        wb = openpyxl.load_workbook(path)
        assert set(wb.sheetnames) == {"summary", "daily_slices", "tick_flows"}
        ws = wb["summary"]
        assert ws.max_row >= 2  # 表头 + 至少一笔交易
        assert ws.cell(1, 2).value == "symbol"

    def test_export_csv(self, pipeline, tmp_path):
        curve, log, sig = pipeline
        out = PerformanceAnalyzer.export_review_slices(
            bull_slice(), log, fmt="csv", path=str(tmp_path / "review_csv"))
        for name in ("summary", "daily_slices", "tick_flows"):
            assert os.path.exists(os.path.join(out, f"{name}.csv"))

    def test_plot_report(self, pipeline, tmp_path):
        curve, log, sig = pipeline
        trades = closed_trades(log)
        _, summary = AttributionEngine.attribute(
            log, sig)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        PerformanceAnalyzer.sensitivity_heatmap(
            pd.DataFrame({"w_ofss": np.linspace(0.2, 0.6, 20),
                          "w_cps": np.linspace(0.1, 0.5, 20),
                          "value": np.sin(np.linspace(0, 6, 20))}),
            ax=ax)
        plt.close(fig)
        path = PerformanceAnalyzer.plot_report(
            curve, log, attribution_summary=summary,
            study=pd.DataFrame({"w_ofss": np.linspace(0.2, 0.6, 12),
                                "w_cps": np.linspace(0.1, 0.5, 12),
                                "value": np.linspace(0, 2, 12)}),
            path=str(tmp_path / "dashboard.png"))
        assert os.path.exists(path)


# ----------------------------------------------------------------------
# IC mock 数据
# ----------------------------------------------------------------------

def _mk_ic_data(n_symbols=1, days=4):
    """多/单标的 30 分钟 K 线 + 因子长表（因子与未来收益正相关）。"""
    dates = pd.date_range("2024-01-02", periods=days, freq="B")
    parts = []
    for d in dates:
        parts.append(pd.date_range(f"{d:%Y-%m-%d} 09:30", f"{d:%Y-%m-%d} 11:30", freq="30min"))
        parts.append(pd.date_range(f"{d:%Y-%m-%d} 13:00", f"{d:%Y-%m-%d} 15:00", freq="30min"))
    axis = pd.DatetimeIndex(np.concatenate([p.values for p in parts])).sort_values()

    krows, frows = [], []
    for i, t in enumerate(axis):
        for s in range(n_symbols):
            sym = f"60000{s}"
            c = 10.0 + 0.01 * i + 0.001 * s * i
            krows.append({"ts": t, "symbol": sym, "close": c})
            factor = 0.05 * i + 0.5 * s      # 单调上升 → 与未来收益正相关
            frows.append({"ts": t, "symbol": sym, "final_ms": factor,
                          "inst_flow": factor * 0.5})
    kline = pd.DataFrame(krows).set_index("ts")
    features = pd.DataFrame(frows).set_index("ts")
    return kline, features


# 固定账本对应任务书 C2/C3；真实 Signal 与下一分钟成交通过 ID 关联。
def allocation_ledger(fees=True):
    dates = pd.to_datetime(["2024-01-02 10:01", "2024-01-03 10:01",
                            "2024-01-04 10:01", "2024-01-05 10:01"])
    quantities = [200, 100, 150, 150] if fees else [100, 100, 100, 100]
    prices = [10, 20, 16, 14]
    rows = []
    for i, (ts, qty, price) in enumerate(zip(dates, quantities, prices)):
        rows.append(dict(ts=ts, symbol="A", side=["BUY", "ADD", "SELL", "SELL"][i],
                         price=price, shares=qty, amount=price * qty,
                         commission=5 if fees else 0,
                         stamp_duty=([0, 0, .5, 1][i] if fees else 0),
                         transfer_fee=([1, 1, .5, 1][i] if fees else 0),
                         fill_id=f"f{i}", signal_id=f"s{i}", event_seq=i+1,
                         decision_ts=ts - pd.Timedelta(minutes=1)))
    signals = [Signal("A", dates[i] - pd.Timedelta(minutes=1), ["BUY", "ADD"][i],
                      "S_push", dict(zip(["global_mod", "chain_mod", "agent_ms"], values)),
                      signal_id=f"s{i}")
               for i, values in enumerate([(-.5, .3, .2), (.2, .5, .3)])]
    return pd.DataFrame(rows), signals


class TestAllocationContract:
    def test_c2_average_cost_not_fifo_cost(self):
        log, signals = allocation_ledger(False)
        log = log.iloc[:3]
        trades, summary = AttributionEngine.attribute(log, signals)
        batches = match_trade_batches(log)
        assert trades.pnl.tolist() == pytest.approx([100], abs=1e-6)
        assert batches.buy_fill_id.tolist() == ["f0"]
        assert batches.allocated_pnl.tolist() == pytest.approx([100], abs=1e-6)
        assert summary.pnl.sum() == pytest.approx(100, abs=1e-6)

    def test_c3_fees_partial_sells_and_source_weights(self):
        log, signals = allocation_ledger()
        trades, summary = AttributionEngine.attribute(log, signals)
        batches = AttributionEngine.attribute_batches(log, signals)
        assert len(trades) == len(closed_trades(log)) == 2
        assert len(batches) == 3
        assert trades.pnl.tolist() == pytest.approx([388, 87], abs=1e-6)
        assert batches.allocated_pnl.tolist() == pytest.approx([388, 29, 58], abs=1e-6)
        assert batches.matched_shares.tolist() == [150, 50, 100]
        assert summary.pnl.tolist() == pytest.approx([220.1, 154.1, 100.8, 0], abs=1e-6)
        assert summary.n_trades.sum() == 2
        assert trades.factor.tolist() == ["global_mod", "chain_mod"]
        assert batches.iloc[0].entry_global_mod == -.5
        assert batches.iloc[0].weight_global_mod == pytest.approx(.5, abs=1e-12)
        quality = AttributionEngine.quality_report(log, signals, batches)
        assert quality["status"] == "complete"
        assert quality["matched_buy_fill_rate"] == quality["attributed_sold_share_rate"] == 1
        assert quality["remaining_open_shares"] == 0
        assert abs(quality["conservation_error"]) <= 1e-6
        # 字段位置无关，不能再以 row[0]/row[6] 提取。
        assert closed_trades(log[log.columns[::-1]]) == closed_trades(log)

    def test_t01_normalization_equivalence_and_no_mutation(self):
        import copy
        log, signals = allocation_ledger()
        original = copy.deepcopy(signals)
        nested = pd.DataFrame([s.to_dict() for s in signals])
        original_nested = nested.copy(deep=True)
        flat = nested.drop(columns="metrics").join(pd.DataFrame([s.metrics for s in signals]))
        expected = AttributionEngine.attribute_batches(log, signals)
        for value in [nested, flat]:
            pd.testing.assert_frame_equal(expected, AttributionEngine.attribute_batches(log, value))
        pd.testing.assert_frame_equal(nested, original_nested)
        assert signals == original
        assert all(_as_signals_frame(nested)[f].dtype == "float64" for f in ["global_mod", "chain_mod", "agent_ms"])

    def test_conflicting_columns_and_duplicate_ids(self):
        log, signals = allocation_ledger()
        nested = pd.DataFrame([s.to_dict() for s in signals])
        nested["global_mod"] = 100.0
        with pytest.raises(ValueError, match="冲突"):
            AttributionEngine.attribute(log, nested)
        with pytest.raises(ValueError, match="signal_id"):
            AttributionEngine.attribute(log, signals + [signals[0]])
        log.loc[1, "fill_id"] = log.loc[0, "fill_id"]
        with pytest.raises(ValueError, match="fill_id"):
            AttributionEngine.attribute(log, signals)

    @pytest.mark.parametrize("field,value", [("symbol", "B"), ("action", "SELL"),
                                             ("timestamp", pd.Timestamp("2024-01-02 10:01"))])
    def test_source_conflicts_raise(self, field, value):
        log, signals = allocation_ledger()
        setattr(signals[0], field, value)
        with pytest.raises(ValueError, match="矛盾"):
            AttributionEngine.attribute(log, signals)

    @pytest.mark.parametrize("value,reason", [
        (None, "missing_factor"), (np.nan, "missing_factor"),
        ("bad", "non_numeric_factor"), (np.inf, "non_finite_factor"),
        (-np.inf, "non_finite_factor")])
    def test_bad_factors_are_auditable(self, value, reason):
        log, signals = allocation_ledger()
        signals[0].metrics["chain_mod"] = value
        batches = AttributionEngine.attribute_batches(log, signals)
        assert batches.iloc[0].other_reason == reason
        assert batches.iloc[0].pnl_other == pytest.approx(388)
        assert batches.iloc[0].weight_other == 1
        assert batches.iloc[0].pnl_global_mod == 0
        assert AttributionEngine.quality_report(log, signals, batches)["status"] == "partial"

    def test_reason_priority_zero_and_legacy(self):
        log, signals = allocation_ledger()
        signals[0].metrics = {"global_mod": "bad", "chain_mod": np.inf}
        signals[1].metrics = dict(global_mod=0, chain_mod=0, agent_ms=0)
        batches = AttributionEngine.attribute_batches(log, signals)
        assert batches.other_reason.tolist() == ["missing_factor", "missing_factor", "zero_factors"]
        legacy = log.drop(columns=["signal_id", "fill_id", "decision_ts", "event_seq"])
        old = AttributionEngine.attribute_batches(legacy, signals)
        assert set(old.other_reason) == {"unmatched_signal"}
        assert old.buy_fill_id.str.startswith("legacy_fill_").all()
        assert old.pnl_other.sum() == pytest.approx(475)

    def test_abs_pnl_coverage_when_profits_cancel(self):
        log, signals = allocation_ledger(False)
        signals[1].metrics = {}
        batches = AttributionEngine.attribute_batches(log, signals)
        quality = AttributionEngine.quality_report(log, signals, batches)
        assert quality["total_realized_pnl"] == 0
        assert quality["attributed_abs_pnl_rate"] == .5
        assert quality["attributed_sold_share_rate"] == .5
        assert AttributionEngine.attribute(log, signals)[1].weight.isna().all()

    @pytest.mark.parametrize("shares", [-1, 301, 1.5])
    def test_invalid_quantities_raise(self, shares):
        log, signals = allocation_ledger()
        log["shares"] = log.shares.astype(float)
        log.loc[2, "shares"] = shares
        with pytest.raises(ValueError):
            match_trade_batches(log)

    def test_same_timestamp_legacy_preserves_input_order(self):
        log, _ = allocation_ledger(False)
        log = log.iloc[:3].drop(columns=["event_seq", "fill_id", "signal_id", "decision_ts"])
        log["ts"] = log.iloc[0].ts
        assert closed_trades(log)[0]["pnl"] == 100

    def test_multiple_buy_fills_share_signal_but_not_fill(self):
        log, signals = allocation_ledger()
        log.loc[1, ["signal_id", "decision_ts"]] = ["s0", signals[0].timestamp]
        batches = AttributionEngine.attribute_batches(log, signals)
        assert set(batches.buy_signal_id) == {"s0"}
        assert set(batches.buy_fill_id) == {"f0", "f1"}
        assert batches.pnl_global_mod.sum() == pytest.approx(475 * .5)

    def test_valid_nested_value_and_shanghai_timezone(self):
        log, signals = allocation_ledger()
        nested = pd.DataFrame([s.to_dict() for s in signals])
        nested["global_mod"] = np.nan
        nested["timestamp"] = nested.timestamp.dt.tz_localize("Asia/Shanghai").dt.tz_convert("UTC")
        pd.testing.assert_frame_equal(AttributionEngine.attribute_batches(log, signals),
                                      AttributionEngine.attribute_batches(log, nested))

    def test_zero_realized_pnl_has_null_absolute_coverage(self):
        log, signals = allocation_ledger(False)
        log = log.iloc[:3].copy()
        log.loc[2, "amount"] = 1500
        batches = AttributionEngine.attribute_batches(log, signals)
        quality = AttributionEngine.quality_report(log, signals, batches)
        assert quality["total_realized_pnl"] == 0
        assert quality["attributed_abs_pnl_rate"] is None
        assert quality["attributed_sold_share_rate"] == 1
        assert quality["status"] == "complete"

    def test_mixed_valid_and_other_batches_preserve_sell_count(self):
        log, signals = allocation_ledger()
        signals[1].metrics = {}
        trades, summary = AttributionEngine.attribute(log, signals)
        assert len(trades) == 2
        assert trades.factor.tolist() == ["global_mod", "other"]
        assert np.isnan(trades.iloc[1].entry_global_mod)
        other = summary[summary.factor.eq("other")].iloc[0]
        assert other.pnl == pytest.approx(58)
        assert other.n_trades == 1
        assert summary.pnl.sum() == pytest.approx(475)

    def test_empty_input_schema(self):
        from analytics.attribution import SIGNAL_COLUMNS, ATTR_BATCH_COLUMNS, TRADE_COLUMNS
        assert set(SIGNAL_COLUMNS) <= set(_as_signals_frame([]))
        trades, summary = AttributionEngine.attribute(pd.DataFrame(), [])
        batches = AttributionEngine.attribute_batches(pd.DataFrame(), [])
        assert list(trades) == TRADE_COLUMNS and list(batches) == ATTR_BATCH_COLUMNS
        assert summary.factor.tolist() == ["Global_Mod", "Chain_Mod", "Agent_MS", "other"]

    def test_finite_large_scores_do_not_overflow_weights(self):
        log, signals = allocation_ledger()
        signals[0].metrics = dict(global_mod=-1e308, chain_mod=1e308, agent_ms=0)
        batches = AttributionEngine.attribute_batches(log, signals)
        row = batches.iloc[0]
        assert row.weight_global_mod == pytest.approx(.5, abs=1e-12)
        assert row.weight_chain_mod == pytest.approx(.5, abs=1e-12)
        assert row.pnl_global_mod == pytest.approx(194, abs=1e-6)
        trades, _ = AttributionEngine.attribute(log, signals)
        assert trades.iloc[0].entry_global_mod == -1e308
