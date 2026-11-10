"""
yield_models.py - trains the yield regressor, preferring XGBoost when it is installed.

Both Random Forest and XGBoost are trained on the same split and scored on the same held-out test rows;
the metrics of both are saved so the result is transparent. XGBoost becomes the app's yield model when available.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from common import YIELD_FEATURES
from ndvi import build_ndvi_feature_table
from preprocessing import preprocess_yield_table

GROUND = YIELD_FEATURES[:7]


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


def fit_yield_model(kind: str, X, y, seed: int = 42):
    if kind == "rf":
        return RandomForestRegressor(n_estimators=300, max_depth=10, random_state=seed).fit(X, y)
    if kind == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9,
                            random_state=seed).fit(X, y)
    raise ValueError(f"unknown model kind: {kind}")


def _score(model, X, y) -> dict:
    pred = model.predict(X)
    return {"R2": round(float(r2_score(y, pred)), 4), "RMSE": round(float(mean_squared_error(y, pred) ** 0.5), 2)}


def train_yield(data_dir="data", prefer_xgboost: bool = True, seed: int = 42) -> dict:
    """Returns {model, name, metrics, note, defaults, ranges}. metrics also holds the Random Forest result."""
    data_dir = Path(data_dir)
    y_df = preprocess_yield_table(pd.read_csv(data_dir / "soil_weather_yield.csv"), GROUND)
    ndvi = build_ndvi_feature_table(pd.read_csv(data_dir / "ndvi_timeseries.csv"))
    merged = y_df.merge(ndvi, on="field_id")
    X, y = merged[YIELD_FEATURES].values, merged["yield_kg_ha"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)

    rf = fit_yield_model("rf", Xtr, ytr, seed)
    rf_metrics = _score(rf, Xte, yte)
    model, name, chosen, note = rf, "Random Forest", rf_metrics, None
    if prefer_xgboost:
        if xgboost_available():
            model = fit_yield_model("xgboost", Xtr, ytr, seed)
            name, chosen = "XGBoost", _score(model, Xte, yte)
        else:
            note = "XGBoost is not installed (run: pip install xgboost), so Random Forest was kept."
    return {"model": model, "name": name, "note": note,
            "metrics": {**chosen, "model": name, "random_forest": rf_metrics},
            "defaults": {c: float(merged[c].median()) for c in YIELD_FEATURES},
            "ranges": {c: [float(merged[c].quantile(.02)), float(merged[c].quantile(.98))] for c in YIELD_FEATURES}}
