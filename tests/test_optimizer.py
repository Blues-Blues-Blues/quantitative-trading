"""带多重非线性约束的贝叶斯超参数优化引擎单元测试（合成 mock 数据，不联网）。

覆盖：
- analytics.metrics 纯函数：年化 Sharpe / 最大回撤 / FIFO 交易配对 / 约束违反量
- SearchSpace：权重归一化和为 1、各参数边界、可行性判定
- StrategyOptimizer：端到端回测产生真实成交、TPE 约束寻优、收敛图落盘
- WalkForward：折划分、train/OOS 切片、样本外评估报告
- SignalSynthesizer.inst_window：Inst_Flow 平滑窗口

mock 数据为「12 个交易日、前 6 日强牛后 6 日转熊」：
- 牛段：每 bar 1 笔超大单买入 + 1 笔小单卖出（Inst_Flow>0、Retail<0、CPS>0）
- 熊段：每 bar 1 笔超大单卖出（Inst_Flow<0 → 状态机退出 S_push → 触发 SELL）
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_features import (  # noqa: E402
    cn_minutes, mk_breadth, mk_dragon_tiger, mk_industry, mk_macro,
    mk_snapshot, mk_ticks,
)

import optuna  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

from analytics.metrics import (  # noqa: E402
    constraint_violations, daily_sharpe, evaluate, max_drawdown, trade_stats,
)
from data.dataslice import DataSlice  # noqa: E402
from indicators.feature_engine import FeatureEngine  # noqa: E402
from indicators.microstructure import MicroStructure  # noqa: E402
from optimizer.bayesian_opt import StrategyOptimizer  # noqa: E402
from optimizer.search_space import SearchSpace  # noqa: E402
from optimizer.walk_forward import FoldResult, WalkForward  # noqa: E402
from strategy.signals import SignalSynthesizer  # noqa: E402

# ----------------------------------------------------------------------
# Mock 数据：12 个交易日，前 6 日强牛、后 6 日转熊
# ----------------------------------------------------------------------

_BULL_DATES = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
               "2024-01-08", "2024-01-09"]
_BEAR_DATES = ["2024-01-10", "2024-01-11", "2024-01-12",
               "2024-01-15", "2024-01-16", "2024-01-17"]
_DATES = _BULL_DATES + _BEAR_DATES
_MAPPING = {"600000": "银行"}

# 可直接成交的显式参数（权重和为 1，全部落在 SearchSpace 范围，chip_window=1
# 保证第 3 个交易日起 CPS 有值 → 牛段开仓 → 熊段强制卖出）
TRADE_PARAMS = {
    "weights": (0.35, 0.25, 0.25, 0.15),
    "th_ms_bull": 0.3, "th_ms_exit": -0.1, "th_lock": 0.4, "th_purity": 0.1,
    "th_global_min": -0.8, "th_adr_min": 0.3,
    "win_hold_max": 120, "inst_window": 1, "chip_window": 1,
}


def mk_bull_kline(axis, bull_dates, base=10.0, vol=1e5):
    """个股：牛段每 bar 复利 +0.2%、熊段 -0.2%；放量 → amount > 1e7。"""
    bull = {pd.Timestamp(d).normalize() for d in bull_dates}
    closes = [base]
    for t in axis[1:]:
        step = 0.002 if t.normalize() in bull else -0.002
        closes.append(closes[-1] * (1.0 + step))
    rows = []
    for t, c in zip(axis, closes):
        rows.append({
            "symbol": "600000", "open": c * 0.995, "high": c * 1.01,
            "low": c * 0.99, "close": c, "volume": vol,
            "amount": c * vol * 100, "vwap": c, "float_market_cap": 1e9,
            "up_limit": c * 1.1, "down_limit": c * 0.9, "is_st": False,
        })
    df = pd.DataFrame(rows)
    df.index = axis
    df.index.name = "ts"
    return df


def mk_bull_index(axis, base=3000.0):
    """指数：加速上涨（ma20 > ma60，个股跑赢 → RS>0）+ 量能放大 → MRS>0。"""
    n = len(axis)
    closes = base + 0.05 * np.arange(n) + 0.002 * np.arange(n) ** 2
    rows = []
    for i, t in enumerate(axis):
        c = closes[i]
        rows.append({"index_code": "000300.SH", "open": c, "high": c * 1.002,
                     "low": c * 0.998, "close": c,
                     "volume": 1e7 * (1.0 + 0.02 * i), "vwap": c,
                     "ma20": c, "ma60": c * 0.99})
    df = pd.DataFrame(rows, index=axis)
    df.index.name = "ts"
    return df


def mk_flow_ticks(axis, bull_dates, symbol="600000"):
    """牛段：每 bar 超大单买 2e6 + 小单卖 2e4（Inst>0、Retail<0）；
    熊段：每 bar 超大单卖 2e6（Inst<0 → 触发 SELL）。"""
    bull = {pd.Timestamp(d).normalize() for d in bull_dates}
    rows = []
    for t in axis:
        if t.normalize() in bull:
            rows.append((t, 10.0, 0, 2e6, 1, False))
            rows.append((t, 10.0, 0, 2e4, -1, False))
        else:
            rows.append((t, 10.0, 0, 2e6, -1, False))
    return mk_ticks(rows, symbol=symbol)


def mk_north_margin_days(dates):
    """12 天北向/两融日频表（T+1 披露，列与 NORTH_MARGIN_COLS 一致）。"""
    idx = pd.to_datetime(dates)
    return pd.DataFrame({
        "symbol": "600000", "trade_date": idx,
        "north_holding": np.linspace(1e7, 1.2e7, len(dates)),
        "north_buy_net": np.where(np.arange(len(dates)) % 3 == 0, 1e6, -5e5),
        "margin_fin_balance": np.linspace(1e9, 1.2e9, len(dates)),
        "margin_sec_balance": np.linspace(5e8, 5.3e8, len(dates)),
    })


def bull_slice(dates=_DATES, bull_dates=_BULL_DATES) -> DataSlice:
    """完整强牛→转熊 DataSlice（含龙虎榜 T+1 披露日 avail_date）。"""
    axis = cn_minutes(dates, freq="30min")
    ds = DataSlice(
        kline=mk_bull_kline(axis, bull_dates),
        l2_snapshot=mk_snapshot(axis),
        tick_trades=mk_flow_ticks(axis, bull_dates),
        index_min=mk_bull_index(axis),
        breadth=mk_breadth(axis),
        industry=mk_industry(axis),
        macro=mk_macro(dates),
        north_margin=mk_north_margin_days(dates),
        dragon_tiger=mk_dragon_tiger(dates),
        meta={"symbols": ["600000"]},
    )
    # 合规化：龙虎榜须携带披露次日 avail_date（T+1 隔离）
    dt = ds.dragon_tiger
    avail = pd.to_datetime(dt["trade_date"]) + pd.Timedelta(days=1)
    ds.dragon_tiger = dt.assign(avail_date=avail)
    return ds


def _make_optimizer(data=None):
    return StrategyOptimizer(
        data=data or bull_slice(),
        symbol_to_industry=_MAPPING,
        account_kwargs={"initial_cash": 1e8},
    )


# ----------------------------------------------------------------------
# 绩效指标纯函数
# ----------------------------------------------------------------------

class TestMetrics:

    def test_daily_sharpe_rising(self):
        idx = pd.DatetimeIndex([])
        for d in ("2024-01-02", "2024-01-03", "2024-01-04"):
            idx = idx.append(pd.date_range(f"{d} 10:00", periods=4, freq="30min"))
        curve = pd.DataFrame({"ts": idx, "total_equity": [100, 101, 103, 106,
                                                          110, 115, 121, 128,
                                                          136, 145, 155, 166]})
        s = daily_sharpe(curve)
        assert np.isfinite(s) and s > 0

    def test_daily_sharpe_flat_is_nan(self):
        idx = pd.date_range("2024-01-02 10:00", periods=8, freq="30min")
        curve = pd.DataFrame({"ts": idx, "total_equity": [100.0] * 8})
        assert np.isnan(daily_sharpe(curve))

    def test_max_drawdown(self):
        idx = pd.date_range("2024-01-02 10:00", periods=4, freq="30min")
        curve = pd.DataFrame({"ts": idx, "total_equity": [100.0, 120.0, 90.0, 110.0]})
        assert max_drawdown(curve) == pytest.approx(0.25)

    def test_max_drawdown_flat(self):
        idx = pd.date_range("2024-01-02 10:00", periods=4, freq="30min")
        curve = pd.DataFrame({"ts": idx, "total_equity": [100.0] * 4})
        assert max_drawdown(curve) == 0.0

    @staticmethod
    def _log(*rows):
        cols = ["ts", "symbol", "side", "price", "shares", "amount",
                "commission", "stamp_duty", "transfer_fee", "slippage_bps",
                "cash_after", "equity_after", "reason"]
        return pd.DataFrame(rows, columns=cols)

    def test_trade_stats_fifo_pairing(self):
        log = self._log(
            ("2024-01-02 10:30", "600000", "BUY", 10.0, 100, 1000.0,
             5.0, 0.0, 0.1, 0.0, 0.0, 0.0, "filled"),
            ("2024-01-03 10:30", "600000", "SELL", 12.0, 100, 1200.0,
             6.0, 1.2, 0.12, 0.0, 0.0, 0.0, "signal_sell"),
            ("2024-01-04 10:30", "600001", "BUY", 20.0, 100, 2000.0,
             10.0, 0.0, 0.2, 0.0, 0.0, 0.0, "filled"),
            ("2024-01-05 10:30", "600001", "SELL", 15.0, 100, 1500.0,
             7.5, 1.5, 0.15, 0.0, 0.0, 0.0, "t1_deferred_sell"),
        )
        st = trade_stats(log)
        win_pnl = 1200.0 - 6.0 - 1.2 - 0.12 - (1000.0 + 5.0 + 0.1)
        loss_pnl = 1500.0 - 7.5 - 1.5 - 0.15 - (2000.0 + 10.0 + 0.2)
        assert st["n_trades"] == 2
        assert st["win_rate"] == pytest.approx(0.5)
        assert st["profit_loss_ratio"] == pytest.approx(win_pnl / abs(loss_pnl))
        assert st["total_pnl"] == pytest.approx(win_pnl + loss_pnl)

    def test_trade_stats_empty(self):
        st = trade_stats(pd.DataFrame())
        assert st["n_trades"] == 0
        assert np.isnan(st["win_rate"])

    def test_evaluate_and_constraints(self):
        idx = pd.date_range("2024-01-02 10:00", periods=8, freq="30min")
        curve = pd.DataFrame({"ts": idx, "total_equity": np.linspace(100, 130, 8)})
        log = self._log(
            ("2024-01-02 10:30", "600000", "BUY", 10.0, 100, 1000.0,
             5.0, 0.0, 0.1, 0.0, 0.0, 0.0, "filled"),
            ("2024-01-03 10:30", "600000", "SELL", 12.0, 100, 1200.0,
             6.0, 1.2, 0.12, 0.0, 0.0, 0.0, "signal_sell"),
        )
        m = evaluate(curve, log)
        assert set(m) == {"sharpe", "max_drawdown", "n_trades", "win_rate",
                          "profit_loss_ratio", "total_pnl"}
        v = constraint_violations(m)
        assert len(v) == 4
        assert all(x >= 0 for x in v)
        # 只有 1 笔交易 → 交易笔数违反（30 - 1 = 29）
        assert v[3] == pytest.approx(29.0)

    def test_constraint_violations_nan(self):
        v = constraint_violations({"max_drawdown": float("nan"),
                                   "win_rate": float("nan"),
                                   "profit_loss_ratio": float("nan"),
                                   "n_trades": float("nan")})
        assert v == [1e9, 1e9, 1e9, 1e9]


# ----------------------------------------------------------------------
# 搜索空间
# ----------------------------------------------------------------------

class TestSearchSpace:

    @staticmethod
    def _trial():
        return optuna.create_study().ask()

    def test_suggest_weights_sum_to_one(self):
        ss = SearchSpace()
        for _ in range(20):
            p = ss.suggest(self._trial())
            w = p["weights"]
            assert sum(w) == pytest.approx(1.0, abs=1e-9)

    def test_suggest_weights_in_ranges(self):
        ss = SearchSpace()
        for _ in range(20):
            w = ss.suggest(self._trial())["weights"]
            # 前三个权重独立 uniform 采样必须落在范围内；w_north 由归一化推导，
            # 越界 trial 交由约束机制判为不可行（见 test_is_feasible_*）
            for value, (lo, hi) in zip(w[:3], ss.weight_ranges()[:3]):
                assert lo <= value <= hi

    def test_suggest_int_params_bounds(self):
        ss = SearchSpace()
        for _ in range(20):
            p = ss.suggest(self._trial())
            assert 1 <= p["inst_window"] <= 20
            assert 5 <= p["chip_window"] <= 20
            assert 10 <= p["win_hold_max"] <= 120
            assert -0.8 <= p["th_global_min"] <= -0.2
            assert 0.3 <= p["th_adr_min"] <= 0.7

    def test_is_feasible_rejects_out_of_range(self):
        ss = SearchSpace()
        bad = dict(TRADE_PARAMS)
        bad["weights"] = (0.6, 0.5, 0.0, -0.1)  # 和=1 但 w_north 越界
        assert not ss.is_feasible(bad)
        bad2 = dict(TRADE_PARAMS)
        bad2["weights"] = (0.4, 0.4, 0.4, 0.4)  # 和 != 1
        assert not ss.is_feasible(bad2)
        assert ss.is_feasible(TRADE_PARAMS)


# ----------------------------------------------------------------------
# StrategyOptimizer：端到端 + 寻优
# ----------------------------------------------------------------------

class TestStrategyOptimizer:

    def test_backtest_produces_trades(self):
        opt = _make_optimizer()
        metrics, engine = opt.backtest(bull_slice(), TRADE_PARAMS)
        log, curve = engine.run()
        assert len(log) > 0
        assert metrics["n_trades"] >= 1
        assert set(metrics) >= {"sharpe", "max_drawdown", "win_rate",
                                "profit_loss_ratio", "total_pnl"}
        # 有完整交易时 Sharpe 应为有限值（日频样本足够）
        assert np.isfinite(metrics["sharpe"])

    def test_optimize_returns_study_and_best(self):
        opt = _make_optimizer()
        study = opt.optimize(n_trials=6)
        assert len(study.trials) == 6
        params, metrics = opt.best(study)
        assert sum(params["weights"]) == pytest.approx(1.0, abs=1e-9)
        assert "sharpe" in metrics
        # best() 返回的必须是某 trial 的缓存结果（objective/constraints 同源）
        cached = {id(m): n for n, (_, m) in opt._cache.items()}
        assert id(metrics) in cached

    def test_constraints_violation_values(self):
        opt = _make_optimizer()
        study = opt.optimize(n_trials=4)
        v = opt.constraints_func(study.trials[0])
        assert len(v) == 4 and all(x >= 0 for x in v)

    def test_plot_history(self, tmp_path):
        opt = _make_optimizer()
        study = opt.optimize(n_trials=4)
        path = opt.plot_history(study, path=str(tmp_path / "history.png"))
        assert os.path.exists(path)

    def test_inst_window_smoothing(self):
        ds = bull_slice()
        fe = FeatureEngine(micro=MicroStructure(chip_window=1),
                           symbol_to_industry=_MAPPING)
        features = fe.compute(ds)
        syn = SignalSynthesizer(weights=TRADE_PARAMS["weights"], inst_window=5)
        out = syn.synthesize(ds, features)
        expected = features.groupby("symbol")["inst_flow"].transform(
            lambda s: s.rolling(5, min_periods=1).mean())
        pd.testing.assert_series_equal(
            out["inst_flow"].astype(float).reset_index(drop=True),
            expected.astype(float).reset_index(drop=True))


# ----------------------------------------------------------------------
# Walk-Forward：滚动样本外
# ----------------------------------------------------------------------

class TestWalkForward:

    def test_fold_ranges_expanding(self):
        wf = WalkForward(bull_slice(), n_folds=2, train_folds=2, expanding=True)
        ranges = wf.fold_ranges()
        assert len(ranges) == 2
        # 折 0：训练 [d1, d6]，OOS [d7, d9]（紧接训练段）
        ts, te, os_, oe = ranges[0]
        assert ts == pd.Timestamp("2024-01-02")
        assert te == pd.Timestamp("2024-01-09")
        assert os_ == pd.Timestamp("2024-01-10")
        assert oe == pd.Timestamp("2024-01-12")
        # 折 1 训练段 expanding：扩到 d1..d9
        assert ranges[1][0] == pd.Timestamp("2024-01-02")
        assert ranges[1][1] == pd.Timestamp("2024-01-12")
        assert ranges[1][2] == pd.Timestamp("2024-01-15")

    def test_fold_ranges_rolling(self):
        wf = WalkForward(bull_slice(), n_folds=2, train_folds=2, expanding=False)
        ranges = wf.fold_ranges()
        # 12 日近均分 4 段（3 日/段）；rolling 固定最近 train_folds 段：
        # 折 0 训练 = 段 0-1（01-02..01-09），折 1 训练 = 段 1-2（01-05..01-12）
        assert ranges[0][0] == pd.Timestamp("2024-01-02")
        assert ranges[0][1] == pd.Timestamp("2024-01-09")
        assert ranges[1][0] == pd.Timestamp("2024-01-05")
        assert ranges[1][1] == pd.Timestamp("2024-01-12")

    def test_fold_datasets_contain_oos_day(self):
        wf = WalkForward(bull_slice(), n_folds=2, train_folds=2)
        train_ds, oos_ds = wf.fold_datasets(0)
        days = {t.normalize() for t in oos_ds.kline.index}
        assert pd.Timestamp("2024-01-10") in days  # OOS 首日完整保留

    def test_run_produces_report(self):
        wf = WalkForward(bull_slice(), n_folds=2, train_folds=2, seed=7)
        results = wf.run(n_trials=4, symbol_to_industry=_MAPPING,
                         account_kwargs={"initial_cash": 1e8})
        assert len(results) == 2
        assert all(isinstance(r, FoldResult) for r in results)
        assert all(r.oos_metrics["max_drawdown"] >= 0 for r in results)
        frame = WalkForward.to_frame(results)
        for col in ("fold", "oos_start", "oos_end", "oos_sharpe",
                    "oos_max_drawdown", "oos_win_rate", "oos_pl_ratio",
                    "oos_n_trades", "w_ofss", "th_ms_bull"):
            assert col in frame.columns
        summary = WalkForward.summary(results)
        assert summary["n_folds"] == 2
        assert "oos_sharpe_mean" in summary
