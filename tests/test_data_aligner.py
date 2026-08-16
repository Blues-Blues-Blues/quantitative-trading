"""多源异构数据时间对齐管道单元测试（全部使用合成 mock 数据，不联网）。

覆盖点：
- 宏观外部数据 T-1 全量对齐（当日数据绝不参与当日计算）
- 海外休市/缺数据时 asof + ffill 容错，输出 Warning 且不中断
- 龙虎榜 T+1 隔离（披露次日才可用，T 日使用会触发 LookaheadError）
- verify_no_lookahead 防未来函数校验
- DataLoader（CSV/Parquet）读写回环组装 DataSlice
- DataSlice schema 校验与 breadth.adr 自动补齐
"""

import logging

import numpy as np
import pandas as pd
import pytest

import config.data_sources as data_sources
from data.aligner import LookaheadError, TimeAligner
from data.dataloader import CsvDataLoader, ParquetDataLoader
from data.dataslice import DataSlice, KLINE_COLS, MACRO_COLS
from data import storage


# ----------------------------------------------------------------------
# Mock 数据构造
# ----------------------------------------------------------------------

def cn_minutes(dates, freq: str = "1min") -> pd.DatetimeIndex:
    """构造 A 股盘中分钟时间轴（09:30-11:30 / 13:00-15:00）。"""
    parts = []
    for d in pd.to_datetime(dates):
        parts.append(pd.date_range(f"{d:%Y-%m-%d} 09:30", f"{d:%Y-%m-%d} 11:30", freq=freq))
        parts.append(pd.date_range(f"{d:%Y-%m-%d} 13:00", f"{d:%Y-%m-%d} 15:00", freq=freq))
    return pd.DatetimeIndex(np.concatenate([p.values for p in parts])).sort_values()


def mock_macro(dates, base: float = 100.0) -> pd.DataFrame:
    """每外部交易日一行的宏观表（trade_date + MACRO_COLS 数值列）。"""
    rows = []
    for i, d in enumerate(pd.to_datetime(dates)):
        row = {"trade_date": d}
        for j, col in enumerate(MACRO_COLS):
            row[col] = base + i * 10 + j
        rows.append(row)
    return pd.DataFrame(rows)


def mock_kline(axis: pd.DatetimeIndex, symbols=("600000", "000001")) -> pd.DataFrame:
    rows = []
    for t in axis:
        for s in symbols:
            rows.append({
                "symbol": s, "open": 10.0, "high": 10.5, "low": 9.8,
                "close": 10.2, "volume": 1000.0, "amount": 10200.0,
                "vwap": 10.2, "float_market_cap": 1e9, "up_limit": 11.0,
                "down_limit": 9.0, "is_st": False,
            })
    df = pd.DataFrame(rows).set_index(axis.repeat(len(symbols)))
    df.index.name = "ts"
    return df


def mock_dragon_tiger(dates) -> pd.DataFrame:
    rows = []
    for i, d in enumerate(pd.to_datetime(dates)):
        rows.append({
            "symbol": "600000", "trade_date": d,
            "buy_amount": 1e8 + i, "sell_amount": 5e7, "net_amount": 5e7, "side": 1,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# T-1 对齐
# ----------------------------------------------------------------------

class TestExternalAlignment:
    """宏观外部数据 T-1 全量对齐。"""

    def test_t_minus_1_uses_previous_day(self):
        dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
        axis = cn_minutes(dates, freq="1h")
        macro = mock_macro(dates)
        out = TimeAligner().align_external(macro, axis, MACRO_COLS)

        assert list(out.index) == list(axis)
        d2 = pd.Timestamp("2024-01-03 09:30")   # 第 2 个交易日
        d3 = pd.Timestamp("2024-01-04 09:30")   # 第 3 个交易日
        # cn 第 2 日使用外部第 1 日值；第 3 日使用外部第 2 日值
        assert out.loc[d2, "us_spx"] == macro.iloc[0]["us_spx"]
        assert out.loc[d3, "us_spx"] == macro.iloc[1]["us_spx"]

    def test_same_day_external_never_visible(self):
        dates = ["2024-01-02", "2024-01-03"]
        axis = cn_minutes(dates, freq="1h")
        macro = mock_macro(dates)
        out = TimeAligner().align_external(macro, axis, MACRO_COLS)

        # 外部第 1 日（2024-01-02）的值绝不出现在 cn 第 1 日（T-1 策略）
        d1 = pd.Timestamp("2024-01-02")
        same_day = out.loc[out.index.normalize() == d1, "us_spx"]
        assert same_day.isna().all()
        # 外部第 1 日的值应从 cn 第 2 日才开始出现
        d2 = pd.Timestamp("2024-01-03")
        next_day = out.loc[out.index.normalize() == d2, "us_spx"]
        assert (next_day == macro.iloc[0]["us_spx"]).all()

    def test_missing_day_asof_uses_prior(self, caplog):
        """海外休市整行缺失：asof 自动取前值，不中断流程。"""
        dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
        axis = cn_minutes(dates, freq="1h")
        macro = mock_macro(dates)
        macro = macro[macro["trade_date"] != "2024-01-03"].copy()  # 01-03 休市缺行

        with caplog.at_level(logging.WARNING):
            out = TimeAligner().align_external(macro, axis, MACRO_COLS)

        assert len(out) == len(axis)
        # cn 01-04：asof 跳过缺失日，取 01-02 整行值
        d3 = pd.Timestamp("2024-01-04 09:30")
        assert out.loc[d3, "us_spx"] == macro.iloc[0]["us_spx"]
        assert out.loc[d3, "gold"] == macro.iloc[0]["gold"]
        # 已发出缺失告警
        assert any("缺失" in r.message for r in caplog.records)

    def test_partial_nan_ffill(self):
        """行内部分 NaN：asof 命中该日行，缺失列由 ffill 回填前一有效值。"""
        dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
        axis = cn_minutes(dates, freq="1h")
        macro = mock_macro(dates)
        macro.loc[macro["trade_date"] == "2024-01-03", "us_spx"] = np.nan

        out = TimeAligner().align_external(macro, axis, MACRO_COLS)

        # cn 01-04 命中 01-03 行：us_spx 缺失 → ffill 到 01-02 值；gold 取 01-03 值
        d3 = pd.Timestamp("2024-01-04 09:30")
        assert out.loc[d3, "us_spx"] == macro.iloc[0]["us_spx"]
        assert out.loc[d3, "gold"] == macro.iloc[1]["gold"]

    def test_macro_has_no_leading_nan_after_ffill(self):
        dates = ["2024-01-02", "2024-01-03"]
        axis = cn_minutes(dates, freq="1h")
        macro = mock_macro(dates)
        out = TimeAligner().align_external(macro, axis, MACRO_COLS)
        # 从第 2 个交易日开始，所有列均无 NaN
        tail = out.loc[out.index.normalize() == "2024-01-03"]
        assert tail.notna().all().all()


# ----------------------------------------------------------------------
# 龙虎榜 T+1
# ----------------------------------------------------------------------

class TestDragonTiger:
    """龙虎榜 T+1 隔离。"""

    def test_avail_date_is_next_trading_day(self):
        dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
        axis = cn_minutes(dates)
        dts = mock_dragon_tiger(["2024-01-02"])
        out = TimeAligner().align_dragon_tiger(dts, axis)

        assert out["avail_date"].iloc[0] == pd.Timestamp("2024-01-03")

    def test_using_before_avail_raises(self):
        """T 日实时特征若误用当日榜单 → 触发未来函数校验。"""
        axis = cn_minutes(["2024-01-02", "2024-01-03"])
        dts = TimeAligner().align_dragon_tiger(
            mock_dragon_tiger(["2024-01-02"]), axis)
        row = dts.iloc[0]

        # T 日（01-02 盘中）使用 → 违规
        with pytest.raises(LookaheadError):
            TimeAligner.verify_no_lookahead(
                [pd.Timestamp("2024-01-02 10:00")], [row["avail_date"]],
                name="dragon_tiger")
        # T+1（01-03 盘中）使用 → 合规
        assert TimeAligner.verify_no_lookahead(
            [pd.Timestamp("2024-01-03 10:00")], [row["avail_date"]],
            name="dragon_tiger")

    def test_verify_no_lookahead_length_mismatch(self):
        with pytest.raises(ValueError):
            TimeAligner.verify_no_lookahead([pd.Timestamp("2024-01-02 10:00")], [])


# ----------------------------------------------------------------------
# 完整管道
# ----------------------------------------------------------------------

class TestAlignSlice:
    """align_slice 端到端。"""

    def test_pipeline(self):
        dates = ["2024-01-02", "2024-01-03"]
        axis = cn_minutes(dates, freq="30min")
        ds = DataSlice(
            kline=mock_kline(axis),
            macro=mock_macro(dates),
            dragon_tiger=mock_dragon_tiger(["2024-01-02"]),
            breadth=pd.DataFrame(
                {"advancers": [3000, 3500], "decliners": [1500, 1000]},
                index=axis[:2],
            ),
            meta={"symbols": ["600000", "000001"]},
        )
        out = TimeAligner().align_slice(ds)

        # macro 已对齐：第 2 个交易日用外部第 1 日值
        assert out.macro.loc["2024-01-03 09:30", "us_spx"] == \
            ds.macro.iloc[0]["us_spx"]
        # 龙虎榜带 avail_date
        assert "avail_date" in out.dragon_tiger.columns
        # breadth 自动补齐 adr
        assert "adr" in out.breadth.columns
        assert out.breadth["adr"].iloc[0] == pytest.approx(3000 / 1500)
        # 元数据标记
        assert out.meta["t_minus_1_external"] is True
        # 整体 schema 校验通过
        out.validate()


# ----------------------------------------------------------------------
# DataLoader 读写回环
# ----------------------------------------------------------------------

@pytest.fixture
def tmp_sources(tmp_path):
    """把逻辑数据源临时指向临时目录，测试后恢复注册表。"""
    saved = {}

    def _set(key: str) -> str:
        saved[key] = data_sources._registry.get(key)
        data_sources.set_data_path(key, tmp_path)
        return str(tmp_path)

    yield _set
    for k, v in saved.items():
        if v is None:
            data_sources._registry.pop(k, None)
        else:
            data_sources._registry[k] = v


class TestDataLoader:
    """DataLoader 抽象层读写回环。"""

    def test_csv_roundtrip(self, tmp_sources):
        root = tmp_sources("daily_cache")
        root2 = tmp_sources("macro")

        axis = cn_minutes(["2024-01-02"], freq="30min")
        kline = mock_kline(axis, symbols=("600000",))
        kline.index.name = "ts"
        macro = mock_macro(["2024-01-02"])
        macro.index = pd.DatetimeIndex(macro["trade_date"])
        macro.index.name = "ts"

        storage.write_frame(kline, "daily_cache", "kline.csv")
        storage.write_frame(macro, "macro", "macro.csv")

        loader = CsvDataLoader()
        ds = loader.load_slice(
            ["600000"], "2024-01-02", "2024-01-02",
            include={"l2_snapshot": False, "tick_trades": False,
                     "index_min": False, "breadth": False,
                     "industry": False, "macro": True, "dragon_tiger": False},
        )
        assert not ds.kline.empty
        assert set(ds.symbols()) == {"600000"}
        assert not ds.macro.empty
        ds.validate(required_cols={"kline": KLINE_COLS, "macro": MACRO_COLS})

    def test_parquet_roundtrip(self, tmp_sources):
        pytest.importorskip("pyarrow")
        tmp_sources("daily_cache")
        axis = cn_minutes(["2024-01-02"], freq="30min")
        kline = mock_kline(axis, symbols=("600000",))
        kline.index.name = "ts"
        storage.write_frame(kline, "daily_cache", "kline.parquet")

        ds = ParquetDataLoader().load_slice(
            ["600000"], "2024-01-02", "2024-01-02",
            include={"l2_snapshot": False, "tick_trades": False,
                     "index_min": False, "breadth": False,
                     "industry": False, "macro": False, "dragon_tiger": False},
        )
        assert not ds.kline.empty
        assert isinstance(ds.kline.index, pd.DatetimeIndex)

    def test_missing_file_returns_empty(self, tmp_sources, caplog):
        tmp_sources("daily_cache")
        loader = CsvDataLoader()
        with caplog.at_level(logging.WARNING):
            ds = loader.load_slice(
                ["600000"], "2024-01-02", "2024-01-02",
                include={"l2_snapshot": False, "tick_trades": False,
                         "index_min": False, "breadth": False,
                         "industry": False, "macro": False, "dragon_tiger": False},
            )
        assert ds.kline.empty
        assert any("数据文件不存在" in r.message for r in caplog.records)


# ----------------------------------------------------------------------
# DataSlice 校验
# ----------------------------------------------------------------------

class TestDataSliceValidate:
    """DataSlice 结构校验。"""

    def test_validate_ok(self):
        axis = cn_minutes(["2024-01-02"], freq="30min")
        ds = DataSlice(kline=mock_kline(axis))
        ds.validate()  # 不抛异常

    def test_validate_duplicate_index_fails(self):
        axis = cn_minutes(["2024-01-02"], freq="30min")
        kline = mock_kline(axis)
        # 制造重复的 (时间戳, symbol) 行并保持排序
        kline = pd.concat([kline, kline.iloc[[1]]]).sort_index()
        with pytest.raises(ValueError, match="重复"):
            DataSlice(kline=kline).validate()

    def test_dragon_tiger_without_avail_date_fails(self):
        axis = cn_minutes(["2024-01-02"], freq="30min")
        dts = mock_dragon_tiger(["2024-01-02"])
        with pytest.raises(ValueError, match="avail_date"):
            DataSlice(kline=mock_kline(axis), dragon_tiger=dts).validate()
