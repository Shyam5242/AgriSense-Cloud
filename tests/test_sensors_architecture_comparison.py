from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import architecture as A
import model_comparison as M
import sensors as S


# ---------------------------------------------------------------- sensors
def test_sensor_shape_and_bounds():
    df = S.simulate_sensors(hours=72, seed=1, start=datetime(2026, 6, 1))
    assert list(df.columns) == S.COLUMNS and len(df) == 72
    assert df["soil_moisture_pct"].between(8, 50).all() and df["humidity_pct"].between(25, 98).all()
    assert not df.isna().any().any()


def test_sensor_daily_cycle_and_humidity_link():
    df = S.simulate_sensors(hours=24 * 14, seed=2, start=datetime(2026, 5, 1))
    by_hour = df.groupby(df["timestamp"].dt.hour)["temperature_c"].mean()
    assert 12 <= by_hour.idxmax() <= 17 and 0 <= by_hour.idxmin() <= 6
    assert np.corrcoef(df["temperature_c"], df["humidity_pct"])[0, 1] < -0.5


def test_sensor_deterministic_and_validation():
    a = S.simulate_sensors(hours=10, seed=5, start=datetime(2026, 1, 1))
    b = S.simulate_sensors(hours=10, seed=5, start=datetime(2026, 1, 1))
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError):
        S.simulate_sensors(hours=0)


def test_sensor_monsoon_wetter_than_dry_season():
    wet = S.simulate_sensors(hours=24 * 60, seed=3, start=datetime(2026, 7, 1))["soil_moisture_pct"].mean()
    dry = S.simulate_sensors(hours=24 * 60, seed=3, start=datetime(2026, 2, 1))["soil_moisture_pct"].mean()
    assert wet > dry


# ---------------------------------------------------------------- architecture
def test_architecture_graph_is_consistent():
    for a, b in A.EDGES:
        assert a in A.NODES and b in A.NODES
    assert {v[4] for v in A.NODES.values()} == set(A.LAYER_COLORS)
    for model in ("yield", "leaf", "crop", "land"):
        assert ("gateway", model) in A.EDGES and (model, "app") in A.EDGES
    xs, ys, zs = A.edge_coordinates()
    assert len(xs) == len(ys) == len(zs) == len(A.EDGES) * 3


# ---------------------------------------------------------------- model comparison
def _fake_data(tmp_path, n=90, seed=0):
    rng = np.random.default_rng(seed)
    soil = pd.DataFrame({
        "field_id": range(n), "nitrogen": rng.uniform(20, 120, n), "phosphorus": rng.uniform(10, 80, n),
        "potassium": rng.uniform(10, 80, n), "rainfall": rng.uniform(200, 900, n),
        "temperature": rng.uniform(20, 34, n), "humidity": rng.uniform(40, 90, n), "ph": rng.uniform(5.5, 7.8, n)})
    peak = rng.uniform(0.4, 0.9, n)
    soil["yield_kg_ha"] = 1500 + 0.8 * soil["rainfall"] + 3000 * peak + rng.normal(0, 80, n)
    rows = []
    for i in range(n):
        for d in range(8):
            rows.append([i, d * 10, peak[i] * np.sin(np.pi * (d + 0.5) / 8) ** 1.2 + rng.normal(0, 0.01)])
    soil.to_csv(tmp_path / "soil_weather_yield.csv", index=False)
    pd.DataFrame(rows, columns=["field_id", "day", "ndvi"]).to_csv(tmp_path / "ndvi_timeseries.csv", index=False)


def test_comparison_runs_and_ndvi_helps(tmp_path):
    _fake_data(tmp_path)
    res = M.run_comparison(tmp_path, folds=3)
    df = pd.DataFrame(res["results"])
    expected_models = 3 if M.xgboost_available() else 2
    assert len(df) == expected_models * 2 and set(df["features"]) == {"Ground only", "Ground + NDVI"}
    assert np.isfinite(df[["MAE", "RMSE", "R2"]].to_numpy()).all()
    gain = M.ndvi_gain(res).set_index("model")
    assert gain.loc["Random Forest", "R2 gain"] > 0


def test_comparison_save_load(tmp_path):
    assert M.load_comparison(tmp_path / "none.json") is None
    M.save_comparison({"results": [], "n_rows": 0}, tmp_path / "x.json")
    assert M.load_comparison(tmp_path / "x.json")["n_rows"] == 0
