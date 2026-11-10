"""
model_comparison.py - yield-model comparison from the paper's "Baseline Comparison" section.

Random Forest, XGBoost and a shallow ANN are each trained twice (ground features only, then ground + NDVI)
and scored with K-fold cross-validation (MAE, RMSE, R2).

    python model_comparison.py            # writes models/model_comparison.json

Notes
  * Uses data/soil_weather_yield.csv and data/ndvi_timeseries.csv, so the results are only as real as that data.
  * XGBoost is skipped (not substituted) if the xgboost package is not installed: pip install xgboost
  * Imputation and outlier clipping are applied before cross-validation, as in train_and_save.py.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import MODELS, YIELD_FEATURES
from ndvi import build_ndvi_feature_table
from preprocessing import preprocess_yield_table

GROUND = YIELD_FEATURES[:7]
TARGET = "yield_kg_ha"
RESULT_PATH = MODELS / "model_comparison.json"


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


def _models(seed: int) -> dict:
    models = {"Random Forest": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=seed, n_jobs=-1)}
    if xgboost_available():
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9,
                                         random_state=seed, n_jobs=-1)
    models["ANN (MLP)"] = TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800,
                                                               early_stopping=True, random_state=seed)),
        transformer=StandardScaler())
    return models


def load_table(data_dir) -> pd.DataFrame:
    data_dir = Path(data_dir)
    soil = preprocess_yield_table(pd.read_csv(data_dir / "soil_weather_yield.csv"), GROUND)
    ndvi = build_ndvi_feature_table(pd.read_csv(data_dir / "ndvi_timeseries.csv"))
    return soil.merge(ndvi, on="field_id")


def run_comparison(data_dir, folds: int = 5, seed: int = 42) -> dict:
    table = load_table(data_dir)
    y = table[TARGET].to_numpy(float)
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    rows = []
    for set_name, feats in {"Ground only": GROUND, "Ground + NDVI": YIELD_FEATURES}.items():
        X = table[feats].to_numpy(float)
        for name, model in _models(seed).items():
            s = cross_validate(model, X, y, cv=cv, scoring=scoring)
            rows.append({
                "model": name, "features": set_name,
                "MAE": float(-s["test_mae"].mean()), "MAE_std": float(s["test_mae"].std()),
                "RMSE": float(-s["test_rmse"].mean()), "RMSE_std": float(s["test_rmse"].std()),
                "R2": float(s["test_r2"].mean()), "R2_std": float(s["test_r2"].std()),
            })
    return {"data_dir": str(data_dir), "n_rows": int(len(table)), "folds": folds,
            "xgboost_available": xgboost_available(), "created": datetime.now().isoformat(timespec="seconds"),
            "results": rows}


def ndvi_gain(result: dict) -> pd.DataFrame:
    """Per model: R2 and RMSE change when NDVI features are added."""
    df = pd.DataFrame(result["results"])
    out = []
    for name, grp in df.groupby("model", sort=False):
        g = grp.set_index("features")
        if {"Ground only", "Ground + NDVI"} <= set(g.index):
            out.append({"model": name,
                        "R2 ground only": g.loc["Ground only", "R2"], "R2 ground + NDVI": g.loc["Ground + NDVI", "R2"],
                        "R2 gain": g.loc["Ground + NDVI", "R2"] - g.loc["Ground only", "R2"],
                        "RMSE change": g.loc["Ground + NDVI", "RMSE"] - g.loc["Ground only", "RMSE"]})
    return pd.DataFrame(out)


def save_comparison(result: dict, path=None) -> Path:
    path = Path(path or RESULT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return path


def load_comparison(path=None):
    path = Path(path or RESULT_PATH)
    return json.loads(path.read_text()) if path.exists() else None


if __name__ == "__main__":
    res = run_comparison(Path(__file__).resolve().parent / "data")
    print("Saved to", save_comparison(res))
    print(pd.DataFrame(res["results"]).round(3).to_string(index=False))
    if not res["xgboost_available"]:
        print("\nXGBoost was skipped. Install it with: pip install xgboost")
