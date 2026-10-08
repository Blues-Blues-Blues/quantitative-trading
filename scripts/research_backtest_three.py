"""2023-2024 three-stock research backtest with an explicit non-ST assumption.

Run from the repository root with the project's Python interpreter. Feature
calculation is split into quarter-sized subprocesses to limit peak memory;
the final execution uses one continuous account over the full two years.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analytics.metrics import closed_trades, daily_sharpe, evaluate, max_drawdown
from data.dataslice import DataSlice
from data.real_loader import RealDataLoader
from engine.backtest import BacktestEngine
from engine.execution import ExecutionCost
from engine.portfolio import Account
from engine.risk_control import PositionSizer
from indicators.feature_engine import FeatureEngine
from indicators.microstructure import MicroStructure
from main import INITIAL_CASH, REAL_PARAMS, _window_end_str
from strategy.signals import SignalSynthesizer, TradingStateMachine


SYMBOLS = ("600171", "600732", "600888")
START = pd.Timestamp("2023-01-03")
END = pd.Timestamp("2024-12-31")
QUARTERS = tuple(
    (str(period), max(START, period.start_time), min(END, period.end_time.normalize()))
    for period in pd.period_range("2023Q1", "2024Q4", freq="Q")
)


class AssumedNonSTLoader(RealDataLoader):
    """Use a documented scenario assumption only for this research run."""

    def _load_stock_history(self) -> pd.DataFrame:
        history = pd.DataFrame(
            {
                "symbol": list(SYMBOLS),
                "trade_date": [pd.Timestamp("2022-12-30")] * len(SYMBOLS),
                "is_st": [False] * len(SYMBOLS),
                "float_shares": [np.nan] * len(SYMBOLS),
            }
        )
        history["trade_date"] = history["trade_date"].astype("datetime64[ns]")
        return history


def _within(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = frame.index.normalize()
    return frame.loc[(dates >= start) & (dates <= end)].copy()


def _loader(st_mode: str) -> RealDataLoader:
    if st_mode == "assumed_non_st":
        return AssumedNonSTLoader()
    if st_mode == "historical":
        return RealDataLoader()
    raise ValueError("st_mode 必须为 assumed_non_st 或 historical")


def _check_chunk_modes(chunk_dir: Path, st_mode: str,
                       pairs=None) -> None:
    """历史模式只消费明确标记为历史 ST 的季度缓存，防止混用场景假设。"""
    pairs = pairs or ((symbol, quarter) for symbol in SYMBOLS
                      for quarter, _, _ in QUARTERS)
    for symbol, quarter in pairs:
        path = chunk_dir / f"{symbol}_{quarter}_meta.json"
        if not path.exists():
            if st_mode == "historical":
                raise ValueError(f"历史 ST 模式缺少季度缓存来源标记: {path}")
            continue  # 兼容旧的非 ST 假设缓存
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("st_mode") != st_mode:
            raise ValueError(f"季度缓存 ST 模式不匹配: {path}")


def _validate_historical_st(ds: DataSlice, symbols=SYMBOLS) -> None:
    if not set(symbols) <= set(ds.kline["symbol"].unique()):
        raise ValueError("历史 ST 模式缺少股票行情")
    close = pd.to_numeric(ds.kline["close"], errors="coerce")
    for symbol in symbols:
        subset = ds.kline.loc[(ds.kline["symbol"] == symbol) & close.gt(0) & np.isfinite(close)]
        if subset.empty or subset["is_st"].isna().any():
            raise ValueError(f"{symbol} 的有效决策 Bar 缺少已生效历史 ST 状态")


def prepare_chunk(out_dir: Path, symbol: str, quarter: str,
                  st_mode: str = "assumed_non_st", params=None) -> None:
    q_start, q_end = next((start, end) for name, start, end in QUARTERS if name == quarter)
    # 历史数据验证若有 2022 年资料则纳入 Q1 预热；既有假设场景缓存口径不变。
    warm_start = (max(START, q_start - pd.Timedelta(days=90))
                  if st_mode == "assumed_non_st" else q_start - pd.Timedelta(days=90))
    logging.info("preparing %s %s, load %s..%s", symbol, quarter, warm_start.date(), q_end.date())
    loader = _loader(st_mode)
    ds = loader.load_slice([symbol], str(warm_start.date()), str(q_end.date()))
    if st_mode == "historical":
        _validate_historical_st(ds, (symbol,))
    resolved = dict(REAL_PARAMS, **(params or {}))
    fe = FeatureEngine(
        micro=MicroStructure(chip_window=int(resolved["chip_window"])),
        symbol_to_industry=loader.symbol_to_industry,
    )
    features = fe.compute(ds)
    frames = {
        "kline": _within(ds.kline, q_start, q_end),
        "features": _within(features, q_start, q_end),
        "industry": _within(ds.industry, q_start, q_end),
    }
    paths = {name: out_dir / f"{symbol}_{quarter}_{name}.parquet" for name in frames}
    meta_path = out_dir / f"{symbol}_{quarter}_meta.json"
    existing = [str(path) for path in (*paths.values(), meta_path) if path.exists()]
    if existing:
        raise FileExistsError(f"季度缓存已存在，拒绝覆盖: {existing}")
    for name, frame in frames.items():
        frame.attrs.clear()
        frame.to_parquet(paths[name])
    meta_path.write_text(json.dumps({"symbol": symbol, "quarter": quarter,
                                     "st_mode": st_mode,
                                     "warm_start": str(warm_start.date())},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("prepared %s %s: %d bars, %d features", symbol, quarter,
                 len(frames["kline"]), len(frames["features"]))


def prepare_market_chunk(chunk_dir: Path, quarter: str,
                         st_mode: str = "historical") -> None:
    """用三股联合行情轴加载市场分钟表，避免某股停牌造成轴缺口。"""
    targets = {name: chunk_dir / f"{quarter}_{name}.parquet"
               for name in ("index_min", "breadth")}
    _check_chunk_modes(chunk_dir, st_mode, ((s, quarter) for s in SYMBOLS))
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(f"市场季度缓存已存在，拒绝覆盖: {existing}")
    q_start, q_end = next((start, end) for name, start, end in QUARTERS if name == quarter)
    cached = [pd.read_parquet(chunk_dir / f"{s}_{quarter}_kline.parquet") for s in SYMBOLS]
    axis = pd.DatetimeIndex(pd.concat(cached).index.unique()).sort_values()
    loader = _loader(st_mode)
    ds = loader.load_slice(list(SYMBOLS), str(q_start.date()),
                           str(q_end.date()), skip_tick=True)
    if st_mode == "historical":
        _validate_historical_st(ds)
    actual_axis = pd.DatetimeIndex(ds.kline.index.unique()).sort_values()
    if not actual_axis.equals(axis):
        raise ValueError(f"{quarter} 联合行情轴与季度缓存不一致")
    for name, table in (("index_min", ds.index_min), ("breadth", ds.breadth)):
        if table is not None:
            table.attrs.clear()
            table.to_parquet(targets[name])


def _safe_json(value):
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _experiment_fingerprints(chunk_dir: Path) -> dict:
    code_files = ("engine/backtest.py", "strategy/signals.py", "data/real_loader.py",
                  "indicators/feature_engine.py", "main.py", "scripts/research_backtest_three.py")
    suffixes = ("_kline.parquet", "_features.parquet", "_industry.parquet",
                "_index_min.parquet", "_breadth.parquet")
    chunk_files = sorted([p for p in chunk_dir.glob("*.parquet")
                          if p.name.endswith(suffixes)] +
                         list(chunk_dir.glob("*_meta.json")))
    if not chunk_files:
        raise ValueError(f"无季度 parquet 缓存: {chunk_dir}")
    chunk_manifest = {p.name: _file_sha256(p) for p in chunk_files}
    return {"code_sha256": {name: _file_sha256(ROOT / name) for name in code_files},
            "chunk_sha256": hashlib.sha256(json.dumps(chunk_manifest, sort_keys=True)
                                           .encode("utf-8")).hexdigest(),
            "chunk_files": len(chunk_files)}


def assemble_chunks(chunk_dir: Path, st_mode: str = "assumed_non_st") -> tuple[DataSlice, pd.DataFrame]:
    """只读特征缓存；实验结果始终写入独立目录。"""
    _check_chunk_modes(chunk_dir, st_mode)
    chunks = {name: [] for name in ("kline", "features", "industry")}
    for symbol in SYMBOLS:
        for quarter, _, _ in QUARTERS:
            for name in chunks:
                chunks[name].append(pd.read_parquet(chunk_dir / f"{symbol}_{quarter}_{name}.parquet"))
    kline = pd.concat(chunks["kline"]).sort_index(kind="stable")
    features = pd.concat(chunks["features"]).sort_index(kind="stable")
    industry = pd.concat(chunks["industry"]).sort_index(kind="stable")
    market = {}
    for name in ("index_min", "breadth"):
        parts = [pd.read_parquet(chunk_dir / f"{quarter}_{name}.parquet")
                 for quarter, _, _ in QUARTERS
                 if (chunk_dir / f"{quarter}_{name}.parquet").exists()]
        market[name] = (pd.concat(parts).sort_index(kind="stable")
                        .loc[lambda x: ~x.index.duplicated(keep="last")]
                        if parts else None)
    ds = DataSlice(
        kline=kline, industry=industry, index_min=market["index_min"],
        breadth=market["breadth"],
        meta={
            "symbols": list(SYMBOLS), "start": str(START.date()), "end": str(END.date()),
            "source": "data1+data2; quarter feature calculation",
            "st_mode": st_mode,
            "st_assumption": ("All three stocks treated as non-ST throughout 2023-2024; unverified"
                              if st_mode == "assumed_non_st" else None),
            "market_index": "present" if market["index_min"] is not None else "missing",
            "market_breadth": "present" if market["breadth"] is not None else "missing",
            "float_shares": "missing",
        },
    )
    ds.validate()
    assert len(kline) == len(features), "bar/feature row counts differ"
    assert kline.index.equals(features.index), "bar/feature timestamps differ"
    assert kline["symbol"].reset_index(drop=True).equals(
        features["symbol"].reset_index(drop=True)
    ), "bar/feature symbols differ"
    if st_mode == "historical":
        _validate_historical_st(ds)
    return ds, features


def data_coverage(ds: DataSlice, features: pd.DataFrame) -> dict:
    closes = pd.to_numeric(ds.kline["close"], errors="coerce")
    decision = (closes.gt(0) & np.isfinite(closes)).to_numpy()
    valid_k = ds.kline.iloc[np.flatnonzero(decision)]
    axis = valid_k.index
    result = {"decision_bars": int(decision.sum()), "missing_dates": {}}
    for name, values in (("is_st", ds.kline["is_st"].notna().to_numpy()),
                         ("mrs", features["mrs"].notna().to_numpy())):
        covered = values[decision]
        result[name] = float(covered.mean()) if len(covered) else 0.0
        result["missing_dates"][name] = sorted({str(d.date()) for d in valid_k.index[~covered].normalize()})
    for name, table, col in (("index_min", ds.index_min, "close"),
                             ("breadth", ds.breadth, "adr")):
        covered_axis = (table.index[pd.to_numeric(table[col], errors="coerce").notna()].unique()
                        if table is not None else pd.DatetimeIndex([]))
        covered = axis.isin(covered_axis)
        result[name] = float(covered.mean()) if len(covered) else 0.0
        result["missing_dates"][name] = sorted({str(d.date()) for d in axis[~covered].normalize()})
        result[f"{name}_file_exists"] = table is not None
        result[f"{name}_first_ts"] = str(table.index.min()) if table is not None and len(table) else None
        result[f"{name}_last_ts"] = str(table.index.max()) if table is not None and len(table) else None
    result["factor_first_valid"] = {
        col: str(features.index[features[col].notna()].min()) if features[col].notna().any() else None
        for col in ("mrs", "grs", "irs", "global_mod", "chain_mod") if col in features
    }
    return result


def validate_quality(mode: str, st_mode: str, coverage: dict,
                     thresholds: dict | None) -> None:
    if mode not in ("scenario", "strict"):
        raise ValueError("data_quality_mode 必须为 scenario 或 strict")
    if mode == "strict":
        if st_mode != "historical":
            raise ValueError("strict 数据质量模式必须搭配 historical ST 模式")
        required = {"is_st", "index_min", "breadth", "mrs"}
        if thresholds is None or set(thresholds) != required:
            raise ValueError("strict 模式须在实验 JSON 预先指定 is_st/index_min/breadth/mrs 覆盖门槛")
        for name in sorted(required):
            threshold = float(thresholds[name])
            if not 0 <= threshold <= 1:
                raise ValueError(f"{name} 覆盖门槛必须在 [0, 1]")
            if coverage[name] < threshold:
                raise ValueError(f"{name} 覆盖率 {coverage[name]:.4f} < {threshold:.4f}; "
                                 f"缺失日期: {coverage['missing_dates'][name]}")


def run_backtest(chunk_dir: Path, result_dir: Path, params: dict | None = None,
                 st_mode: str = "assumed_non_st", data_quality_mode: str = "scenario",
                 coverage_thresholds: dict | None = None) -> dict:
    result_dir.mkdir(parents=True, exist_ok=True)
    if any((result_dir / name).exists() for name in
           ("resolved_config.json", "summary.json", "trade_log.csv",
            "equity_curve.parquet", "order_events.csv")):
        raise FileExistsError(f"结果目录已有回测产物，拒绝覆盖: {result_dir}")
    resolved = dict(REAL_PARAMS, **(params or {}))
    ds, features = assemble_chunks(chunk_dir, st_mode)
    coverage = data_coverage(ds, features)
    validate_quality(data_quality_mode, st_mode, coverage, coverage_thresholds)
    sig = inspect.signature(SignalSynthesizer)
    kwargs = {name: resolved[name] for name in sig.parameters if name in resolved}
    kwargs.update(reversal_window_end=_window_end_str(resolved),
                  symbol_to_industry=_loader(st_mode).symbol_to_industry)
    synth = SignalSynthesizer(**kwargs)
    fingerprints = _experiment_fingerprints(chunk_dir)
    config = {"symbols": SYMBOLS, "start": str(START.date()), "end": str(END.date()),
              "parameters": resolved, "st_mode": st_mode, "data_quality_mode": data_quality_mode,
              "coverage_thresholds": coverage_thresholds, "coverage": coverage,
              "fingerprints": fingerprints, "chunk_dir": str(chunk_dir.resolve())}
    (result_dir / "resolved_config.json").write_text(
        json.dumps(_safe_json(config), ensure_ascii=False, indent=2), encoding="utf-8")
    engine = BacktestEngine(
        Account(initial_cash=INITIAL_CASH), ExecutionCost(), PositionSizer(), ds,
        deadzone_th=float(resolved.get("deadzone_th", 0.05)),
        state_machine=TradingStateMachine(
            synthesizer=synth,
            max_holding_trading_days=resolved.get("max_holding_trading_days"),
            reentry_cooldown_trading_days=resolved.get("reentry_cooldown_trading_days", 0)),
        features=features,
        entry_order_expiry=resolved.get("entry_order_expiry", "same_session"),
        risk_reduce_bypass_deadzone=resolved.get("risk_reduce_bypass_deadzone", False),
        risk_reduce_min_shares=resolved.get("risk_reduce_min_shares", 100),
    )
    logging.info("starting continuous backtest: %d bars, %d timestamps", len(ds.kline), len(ds.time_axis()))
    trade_log, equity = engine.run()
    trade_log.to_csv(result_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
    engine.order_events.to_csv(result_dir / "order_events.csv", index=False, encoding="utf-8-sig")
    equity.to_parquet(result_dir / "equity_curve.parquet")
    filled = trade_log.loc[trade_log["shares"] > 0]
    closed = closed_trades(trade_log)
    summary = {
        "assumption": ds.meta,
        "data_coverage": coverage,
        "initial_cash": INITIAL_CASH,
        "final_equity": float(equity["total_equity"].iloc[-1]),
        "account_pnl": float(equity["total_equity"].iloc[-1] - INITIAL_CASH),
        "account_return": float(equity["total_equity"].iloc[-1] / INITIAL_CASH - 1),
        "total_return": float(equity["total_equity"].iloc[-1] / INITIAL_CASH - 1),
        "metrics": evaluate(equity, trade_log),
        "filled_orders": len(filled),
        "gross_traded_value": float(filled["amount"].sum()),
        "fees_paid": float(filled[["commission", "stamp_duty", "transfer_fee"]].sum().sum()),
        "turnover_ratio": float(filled["amount"].sum() / equity["total_equity"].mean()),
        "closed_trades": len(closed),
        "order_status_counts": engine.order_events["status"].value_counts().to_dict()
        if not engine.order_events.empty else {},
        "order_reason_counts": engine.order_events["reason"].value_counts().to_dict()
        if not engine.order_events.empty else {},
        "filled_delay_minutes": [float((row.ts - row.decision_ts).total_seconds() / 60)
                                 for row in filled.itertuples() if pd.notna(row.decision_ts)],
        "by_year": {},
    }
    prior_equity = INITIAL_CASH
    for year in (2023, 2024):
        year_equity = equity.loc[equity["ts"].dt.year == year]
        year_fills = filled.loc[filled["ts"].dt.year == year]
        last_equity = float(year_equity["total_equity"].iloc[-1])
        summary["by_year"][str(year)] = {
            "return": last_equity / prior_equity - 1,
            "end_equity": last_equity,
            "sharpe": daily_sharpe(year_equity),
            "max_drawdown": max_drawdown(year_equity),
            "filled_orders": len(year_fills),
            "closed_trades": sum(pd.Timestamp(t["exit_ts"]).year == year for t in closed),
        }
        prior_equity = last_equity
    (result_dir / "summary.json").write_text(
        json.dumps(_safe_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("complete: %s", result_dir / "summary.json")
    print(json.dumps(_safe_json(summary), ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", nargs=2, metavar=("SYMBOL", "QUARTER"))
    ap.add_argument("--market-chunk", metavar="QUARTER")
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--chunk-dir", type=Path)
    ap.add_argument("--result-dir", type=Path)
    ap.add_argument("--params-json", type=Path)
    ap.add_argument("--st-mode", choices=("assumed_non_st", "historical"))
    ap.add_argument("--data-quality-mode", choices=("scenario", "strict"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.params_json is not None:
        args.params_json = args.params_json.resolve()
    config = json.loads(args.params_json.read_text(encoding="utf-8")) if args.params_json else {}
    params = config.get("parameters", config.get("params", {}))
    st_mode = args.st_mode or config.get("st_mode", "assumed_non_st")
    quality_mode = args.data_quality_mode or config.get("data_quality_mode", "scenario")
    thresholds = config.get("coverage_thresholds")
    validate_quality(quality_mode, st_mode,
                     {name: 0 for name in ("is_st", "index_min", "breadth", "mrs")}
                     if quality_mode == "scenario" else
                     {name: 1 for name in ("is_st", "index_min", "breadth", "mrs")},
                     thresholds)
    result_dir = args.result_dir or (
        ROOT / "analytics" / "reports" /
        f"research_3stocks_2023_2024_{st_mode}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    result_dir = result_dir.resolve()
    chunk_dir = (args.chunk_dir or args.out_dir or
                 result_dir.with_name(result_dir.name + "_chunks")).resolve()
    if args.chunk:
        chunk_dir.mkdir(parents=True, exist_ok=True)
        prepare_chunk(chunk_dir, *args.chunk, st_mode=st_mode, params=params)
        return
    if args.market_chunk:
        prepare_market_chunk(chunk_dir, args.market_chunk, st_mode=st_mode)
        return
    if args.assemble_only:
        run_backtest(chunk_dir, result_dir, params, st_mode, quality_mode, thresholds)
        return
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for symbol in SYMBOLS:
        for quarter, _, _ in QUARTERS:
            command = [sys.executable, str(Path(__file__).resolve()), "--chunk", symbol, quarter,
                       "--chunk-dir", str(chunk_dir), "--st-mode", st_mode]
            if args.params_json:
                command += ["--params-json", str(args.params_json)]
            subprocess.run(command, cwd=ROOT, check=True)
    if st_mode == "historical":
        for quarter, _, _ in QUARTERS:
            prepare_market_chunk(chunk_dir, quarter, st_mode=st_mode)
    run_backtest(chunk_dir, result_dir, params, st_mode, quality_mode, thresholds)


if __name__ == "__main__":
    main()
