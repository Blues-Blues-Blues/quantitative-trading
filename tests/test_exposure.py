"""任务书 E2/T07：有符号市值加权、现金影响、逐因子缺失与空仓时间轴。"""
import numpy as np
import pandas as pd
import pytest

from analytics.exposure import ExposureAnalyzer, FACTORS, POSITION_COLUMNS


def exposure_inputs():
    ts = pd.Timestamp("2024-01-02 15:00")
    positions = pd.DataFrame([
        dict(ts=ts, symbol="A", shares=20, mark_price=10, market_value=200, total_equity=1000),
        dict(ts=ts, symbol="B", shares=30, mark_price=10, market_value=300, total_equity=1000)])
    factors = pd.DataFrame([dict(ts=ts, symbol="A", global_mod=.5, chain_mod=-.2, agent_ms=0),
                            dict(ts=ts, symbol="B", global_mod=-.1, chain_mod=.4, agent_ms=.8)])
    equity = pd.DataFrame([dict(ts=ts, total_equity=1000)])
    return positions, factors, equity


def test_e2_signed_equity_weights_and_conservation():
    positions, factors, equity = exposure_inputs()
    saved = factors.copy(deep=True)
    contributions, exposure = ExposureAnalyzer.compute(positions, factors, equity)
    assert exposure.exposure.tolist() == pytest.approx([.07, .08, .24], abs=1e-12)
    assert exposure.invested_weight.tolist() == [.5] * 3
    assert exposure.coverage.tolist() == [1] * 3
    for f in FACTORS:
        assert contributions.loc[contributions.factor.eq(f), "contribution"].sum() == pytest.approx(
            exposure.set_index("factor").loc[f, "exposure"], abs=1e-12)
    pd.testing.assert_frame_equal(saved, factors)


@pytest.mark.parametrize("missing", [None, np.nan, np.inf, "bad"])
def test_e2_missing_factor_is_not_zero(missing):
    positions, factors, equity = exposure_inputs()
    factors["global_mod"] = factors.global_mod.astype(object)
    factors.loc[1, "global_mod"] = missing
    contributions, exposure = ExposureAnalyzer.compute(positions, factors, equity)
    global_row = exposure.iloc[0]
    assert np.isnan(global_row.exposure)
    assert global_row.observed_exposure == pytest.approx(.1)
    assert global_row.coverage == pytest.approx(.4)
    assert global_row.status == "partial"
    assert exposure.exposure.iloc[1:].tolist() == pytest.approx([.08, .24])
    assert np.isnan(contributions.loc[contributions.symbol.eq("B") & contributions.factor.eq("global_mod"), "contribution"].iloc[0])


def test_flat_axis_and_partial_sale():
    positions, factors, equity = exposure_inputs()
    curve = pd.concat([equity, equity.assign(ts=pd.Timestamp("2024-01-03 15:00"))])
    positions.loc[1, ["shares", "market_value"]] = [15, 150]
    _, result = ExposureAnalyzer.compute(positions, factors, curve)
    assert result.exposure.iloc[:3].tolist() == pytest.approx([.085, .02, .12])
    assert result.exposure.iloc[3:].tolist() == [0, 0, 0]
    assert result.status.iloc[3:].tolist() == ["flat"] * 3
    assert result.coverage.iloc[3:].tolist() == [1] * 3


def test_future_factors_cannot_fill_current_holdings():
    positions, factors, equity = exposure_inputs()
    factors["ts"] += pd.Timedelta(days=1)
    _, result = ExposureAnalyzer.compute(positions, factors, equity)
    assert result.exposure.isna().all()
    assert (result.coverage == 0).all()


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_equity_raises(value):
    positions, factors, equity = exposure_inputs()
    equity["total_equity"] = value
    with pytest.raises(ValueError, match="权益"):
        ExposureAnalyzer.compute(positions, factors, equity)


def test_duplicate_factors_raise_and_empty_positions_schema():
    positions, factors, equity = exposure_inputs()
    with pytest.raises(ValueError, match="重复"):
        ExposureAnalyzer.compute(positions, pd.concat([factors, factors]), equity)
    contributions, result = ExposureAnalyzer.compute(pd.DataFrame(columns=POSITION_COLUMNS), factors, equity)
    assert contributions.empty
    assert len(result) == 3 and (result.exposure == 0).all()
