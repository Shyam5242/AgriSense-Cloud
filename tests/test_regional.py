import numpy as np
import pandas as pd
import pytest

import regional as R


def test_simulated_shapes_and_ranges():
    s, m, d = R.generate_simulated()
    assert list(s.columns) == R.SUMMARY_COLS and list(m.columns) == R.MONTHLY_COLS and list(d.columns) == R.DISEASE_COLS
    assert len(s) == 7 * 6 * 2 and len(m) == len(s) * 6
    assert not s.isna().any().any() and not m.isna().any().any()
    assert (s["yield_kg_ha"] >= 600).all() and s["ndvi_mean"].between(-1, 1).all() and (m["ndvi"].between(-1, 1)).all()
    assert (d["cases"] >= 0).all()


def test_deterministic():
    a, _, _ = R.generate_simulated(seed=3)
    b, _, _ = R.generate_simulated(seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_yield_correlates_with_ndvi():
    s, _, _ = R.generate_simulated()
    assert R.correlations(s).loc["ndvi_mean", "yield_kg_ha"] > 0.3


def test_underperformance_finds_penalised_district():
    s, _, _ = R.generate_simulated()
    u = R.underperformance(s)
    assert u.iloc[0]["district"] == "Villupuram" and u.iloc[0]["status"] == "Below peers"


def test_underperformance_small_input():
    s, _, _ = R.generate_simulated()
    assert R.underperformance(s.head(3)).empty


def test_trend_ordering():
    s, _, _ = R.generate_simulated()
    _, order = R.yield_trend(s)
    assert order[0] == "2019 Kharif" and order[1] == "2019 Rabi" and order[-1] == "2024 Rabi"


def test_kpis():
    s, _, _ = R.generate_simulated()
    k = R.kpis(s)
    assert k["districts"] == 7 and k["first_year"] == 2019 and k["last_year"] == 2024


def test_load_falls_back_to_simulated(tmp_path):
    assert R.load_regional(tmp_path)["simulated"] is True


def test_load_real_csv(tmp_path):
    s, m, d = R.generate_simulated()
    s.to_csv(tmp_path / "regional_summary.csv", index=False)
    out = R.load_regional(tmp_path)
    assert out["simulated"] is False and out["monthly"] is None and out["disease"] is None
    m.to_csv(tmp_path / "regional_monthly.csv", index=False)
    assert R.load_regional(tmp_path)["monthly"] is not None


def test_missing_column_message(tmp_path):
    s, _, _ = R.generate_simulated()
    s.drop(columns=["ndvi_mean"]).to_csv(tmp_path / "regional_summary.csv", index=False)
    with pytest.raises(ValueError, match="ndvi_mean"):
        R.load_regional(tmp_path)
