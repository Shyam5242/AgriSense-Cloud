"""
retrain_yield.py - retrain ONLY the yield model (XGBoost when installed).

    python retrain_yield.py          # XGBoost if installed, otherwise Random Forest
    python retrain_yield.py --rf     # force Random Forest

It leaves the crop, leaf-disease and land-cover models and the rest of models/meta.json untouched,
so your real image models trained in Colab are safe. Restart the app and API afterwards.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

from common import MODELS
from yield_models import train_yield


def retrain(data_dir="data", models_dir=MODELS, prefer_xgboost: bool = True) -> dict:
    models_dir = Path(models_dir)
    res = train_yield(data_dir, prefer_xgboost=prefer_xgboost)
    meta_path = models_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["yield_defaults"], meta["yield_ranges"] = res["defaults"], res["ranges"]
    meta.setdefault("metrics", {})["yield"] = res["metrics"]
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(res["model"], models_dir / "yield_rf.joblib")      # file name kept for compatibility
    meta_path.write_text(json.dumps(meta, indent=2))
    return res


if __name__ == "__main__":
    out = retrain(prefer_xgboost="--rf" not in sys.argv)
    if out["note"]:
        print(out["note"])
    m = out["metrics"]
    print(f"Yield model: {out['name']}  R2 {m['R2']}  RMSE {m['RMSE']} kg/ha")
    print(f"Random Forest on the same split: R2 {m['random_forest']['R2']}  RMSE {m['random_forest']['RMSE']} kg/ha")
    print("Restart the app and API to use it.")
