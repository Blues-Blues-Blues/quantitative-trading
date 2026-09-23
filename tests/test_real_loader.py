"""Historical field and independent-market behavior of the real-data adapter."""

import numpy as np
import pandas as pd

from data.real_loader import RealDataLoader


def _bars():
    ts = pd.to_datetime(["2024-01-02 10:00", "2024-01-03 10:00"])
    return pd.DataFrame({
        "symbol": ["600000", "600000"],
        "open": [10.0, 11.0], "high": [10.0, 11.0],
        "low": [10.0, 11.0], "close": [10.0, 11.0],
        "volume": [1000.0, 1000.0], "amount": [10000.0, 11000.0],
    }, index=ts)


def test_historical_st_and_float_shares_are_effective_dated(tmp_path):
    data2 = tmp_path / "data2"
    data2.mkdir()
    pd.DataFrame({
        "symbol": ["600000", "600000"],
        "trade_date": ["2024-01-01", "2024-01-03"],
        "is_st": [0, 1], "float_shares": [1e8, 2e8],
    }).to_csv(data2 / "stock_history.csv", index=False)
    loader = RealDataLoader(data_root=tmp_path)
    loader.l2.load_kline = lambda *args, **kwargs: _bars()
    out = loader._load_kline(["600000"], "2024-01-02", "2024-01-03")
    assert out["is_st"].tolist() == [False, True]
    assert out["float_market_cap"].tolist() == [1e9, 2.2e9]
    assert np.isnan(out["up_limit"].iloc[0])
    assert out["up_limit"].iloc[1] == 10.5


def test_missing_history_and_market_feeds_remain_missing(tmp_path):
    loader = RealDataLoader(data_root=tmp_path)
    loader.l2.load_kline = lambda *args, **kwargs: _bars()
    out = loader._load_kline(["600000"], "2024-01-02", "2024-01-03")
    assert out["is_st"].isna().all()
    assert out["float_market_cap"].isna().all()
    assert out["up_limit"].isna().all()
    assert loader._load_market_table("index_min", ["index_code"], out.index) is None
