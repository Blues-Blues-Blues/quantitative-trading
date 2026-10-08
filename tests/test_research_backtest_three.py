"""季度缓存、联合市场时间轴与严格质量门槛的确定性测试。"""

import pandas as pd
import pytest

from data.dataslice import BREADTH_COLS, INDEX_MIN_COLS, KLINE_COLS, DataSlice
from scripts import research_backtest_three as research
from strategy.signals import SignalSynthesizer, TradingStateMachine


def _frame(axis, symbol, columns):
    data = {col: [0.0] * len(axis) for col in columns}
    if "symbol" in data:
        data["symbol"] = [symbol] * len(axis)
    if "index_code" in data:
        data["index_code"] = ["000300.SH"] * len(axis)
    if "is_st" in data:
        data["is_st"] = [False] * len(axis)
    for col in ("open", "high", "low", "close", "vwap"):
        if col in data:
            data[col] = [10.0] * len(axis)
    if "amount" in data:
        data["amount"] = [1e8] * len(axis)
    if "up_limit" in data:
        data["up_limit"] = [11.0] * len(axis)
    if "down_limit" in data:
        data["down_limit"] = [9.0] * len(axis)
    if "ma20" in data:
        data["ma20"] = [10.0] * len(axis)
    if "ma60" in data:
        data["ma60"] = [10.0] * len(axis)
    return pd.DataFrame(data, index=pd.DatetimeIndex(axis, name="ts"))


def test_market_tables_survive_chunk_assembly(tmp_path, monkeypatch):
    quarter = "2024Q1"
    symbols = ("600171", "600732", "600888")
    times = pd.to_datetime(["2024-01-02 10:00", "2024-01-02 10:01",
                            "2024-01-02 10:02"])
    monkeypatch.setattr(research, "QUARTERS", ((quarter, times[0].normalize(),
                                                times[-1].normalize()),))
    monkeypatch.setattr(research, "SYMBOLS", symbols)
    for i, symbol in enumerate(symbols):
        own = pd.DatetimeIndex([times[i]])
        _frame(own, symbol, KLINE_COLS).to_parquet(tmp_path / f"{symbol}_{quarter}_kline.parquet")
        pd.DataFrame({"symbol": [symbol], "mrs": [.5]}, index=own).to_parquet(
            tmp_path / f"{symbol}_{quarter}_features.parquet")
        pd.DataFrame(index=own).to_parquet(tmp_path / f"{symbol}_{quarter}_industry.parquet")

    class Loader:
        def load_slice(self, *args, **kwargs):
            kline = pd.concat([_frame([times[i]], symbol, KLINE_COLS)
                               for i, symbol in enumerate(symbols)]).sort_index()
            index = _frame(times, None, INDEX_MIN_COLS)
            index["close"] = 9.7
            index["vwap"] = 10.0
            return DataSlice(kline=kline, index_min=index,
                             breadth=_frame(times, None, BREADTH_COLS))

    monkeypatch.setattr(research, "_loader", lambda mode: Loader())
    research.prepare_market_chunk(tmp_path, quarter, st_mode="assumed_non_st")
    ds, features = research.assemble_chunks(tmp_path)
    assert ds.index_min.index.equals(times)
    assert ds.breadth.index.equals(times)
    syn = SignalSynthesizer()
    fake = features.assign(youzi_flow=0.0, inst_flow=0.0, retail_chase=0.0,
                           final_ms=0.0, ofss=0.0, cps=0.0, big_flow=0.0)
    monkeypatch.setattr(syn, "synthesize", lambda data, _: fake)
    ev = TradingStateMachine(syn)._build_eval_table(ds, features)
    assert ev["index_close"].notna().all()
    assert ev["index_vwap"].notna().all()
    assert ev["veto_flag"].all()


def test_historical_st_and_strict_data_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(research, "AssumedNonSTLoader", lambda: pytest.fail("assumed loader used"))
    assert isinstance(research._loader("historical"), research.RealDataLoader)
    with pytest.raises(ValueError, match="缓存来源标记"):
        research._check_chunk_modes(tmp_path, "historical", (("600171", "2024Q1"),))
    missing = {"is_st": 0.0, "index_min": 0.0, "breadth": 0.0, "mrs": 0.0,
               "missing_dates": {name: ["2024-01-02"] for name in
                                 ("is_st", "index_min", "breadth", "mrs")}}
    with pytest.raises(ValueError, match="strict.*historical"):
        research.validate_quality("strict", "assumed_non_st", missing,
                                  {name: .9 for name in ("is_st", "index_min", "breadth", "mrs")})
    with pytest.raises(ValueError, match="覆盖率"):
        research.validate_quality("strict", "historical", missing,
                                  {name: .9 for name in ("is_st", "index_min", "breadth", "mrs")})
    kline = _frame([pd.Timestamp("2024-01-02 10:00")], "600171", KLINE_COLS)
    kline["is_st"] = float("nan")
    with pytest.raises(ValueError, match="历史 ST"):
        research._validate_historical_st(DataSlice(kline=kline), ("600171",))


def test_market_coverage_counts_decision_bars_not_unique_minutes():
    first, second = pd.Timestamp("2024-01-02 10:00"), pd.Timestamp("2024-01-02 10:01")
    kline = pd.concat([_frame([first], "600171", KLINE_COLS),
                       _frame([first], "600732", KLINE_COLS),
                       _frame([second], "600888", KLINE_COLS)]).sort_index()
    index = _frame([first], None, INDEX_MIN_COLS)
    breadth = _frame([first], None, BREADTH_COLS)
    features = pd.DataFrame({"symbol": kline.symbol.to_numpy(), "mrs": [1., 1., 1.]},
                            index=kline.index)
    coverage = research.data_coverage(DataSlice(kline=kline, index_min=index,
                                                 breadth=breadth), features)
    assert coverage["decision_bars"] == 3
    assert coverage["index_min"] == pytest.approx(2 / 3)
    assert coverage["breadth"] == pytest.approx(2 / 3)
