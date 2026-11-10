import json
import sys
import types

import joblib
import numpy as np
import pandas as pd
import pytest

import retrain_yield as R
import yield_models as Y


def _write_tables(d, n=150, seed=0):
    rng = np.random.default_rng(seed)
    soil = pd.DataFrame({
        "field_id": range(n), "nitrogen": rng.uniform(20, 120, n), "phosphorus": rng.uniform(10, 80, n),
        "potassium": rng.uniform(10, 80, n), "rainfall": rng.uniform(200, 900, n),
        "temperature": rng.uniform(20, 34, n), "humidity": rng.uniform(40, 90, n), "ph": rng.uniform(5.5, 7.8, n)})
    peak = rng.uniform(0.4, 0.9, n)
    soil["yield_kg_ha"] = 1500 + 0.8 * soil["rainfall"] + 3000 * peak + rng.normal(0, 80, n)
    rows = [[i, k * 10, peak[i] * np.sin(np.pi * (k + 0.5) / 8) ** 1.2] for i in range(n) for k in range(8)]
    soil.to_csv(d / "soil_weather_yield.csv", index=False)
    pd.DataFrame(rows, columns=["field_id", "day", "ndvi"]).to_csv(d / "ndvi_timeseries.csv", index=False)


def _fake_xgboost(monkeypatch=None):
    from sklearn.ensemble import GradientBoostingRegressor
    mod = types.ModuleType("xgboost")
    mod.XGBRegressor = lambda **kw: GradientBoostingRegressor(n_estimators=40, random_state=1)
    sys.modules["xgboost"] = mod


def test_random_forest_when_xgboost_missing(tmp_path):
    sys.modules.pop("xgboost", None)
    _write_tables(tmp_path)
    if Y.xgboost_available():
        pytest.skip("real xgboost installed")
    res = Y.train_yield(tmp_path, prefer_xgboost=True)
    assert res["name"] == "Random Forest" and "pip install xgboost" in res["note"]
    assert res["metrics"]["model"] == "Random Forest" and 0 < res["metrics"]["R2"] <= 1


def test_xgboost_is_used_when_available(tmp_path):
    _write_tables(tmp_path)
    saved = sys.modules.get("xgboost")
    _fake_xgboost()
    try:
        res = Y.train_yield(tmp_path, prefer_xgboost=True)
        assert res["name"] == "XGBoost" and res["note"] is None
        assert res["metrics"]["model"] == "XGBoost" and "random_forest" in res["metrics"]
        forced = Y.train_yield(tmp_path, prefer_xgboost=False)
        assert forced["name"] == "Random Forest"
    finally:
        if saved is None:
            sys.modules.pop("xgboost", None)
        else:
            sys.modules["xgboost"] = saved


def test_retrain_updates_only_yield_parts(tmp_path):
    data, models = tmp_path / "data", tmp_path / "models"
    data.mkdir(); models.mkdir()
    _write_tables(data)
    meta = {"leaf_classes": ["a", "b"], "landcover_classes": ["x"], "crop_defaults": {"N": 1},
            "metrics": {"leaf": {"accuracy": 0.95}, "landcover": {"accuracy": 0.94}, "crop": {"accuracy": 0.9},
                        "yield": {"R2": 0.1, "RMSE": 9}}}
    (models / "meta.json").write_text(json.dumps(meta))
    saved = sys.modules.get("xgboost")
    _fake_xgboost()
    try:
        R.retrain(data, models, prefer_xgboost=True)
    finally:
        if saved is None:
            sys.modules.pop("xgboost", None)
        else:
            sys.modules["xgboost"] = saved
    new = json.loads((models / "meta.json").read_text())
    assert new["leaf_classes"] == ["a", "b"] and new["crop_defaults"] == {"N": 1}          # untouched
    assert new["metrics"]["leaf"] == {"accuracy": 0.95} and new["metrics"]["landcover"] == {"accuracy": 0.94}
    assert new["metrics"]["yield"]["model"] == "XGBoost" and new["metrics"]["yield"]["R2"] != 0.1
    assert len(new["yield_defaults"]) == 10 and hasattr(joblib.load(models / "yield_rf.joblib"), "predict")


def test_status_table_names_yield_model():
    import architecture as A
    rows = {c: impl for c, impl, _ in A.status_table(False, None, "XGBoost")}
    assert rows["Yield regressor"].startswith("XGBoost")
