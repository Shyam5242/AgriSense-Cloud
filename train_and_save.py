"""
train_and_save.py - trains every model and saves it to ./models

    python train_and_save.py          (retrains EVERYTHING, including the image models)
    python retrain_yield.py           (retrains only the yield model)

Uses the synthetic datasets from generate_data.py / generate_images.py unless
you drop your own files into ./data (same column names) - see README.md.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import generate_data
import generate_images
from common import CROP_FEATURES, MODELS, YIELD_FEATURES, build_cnn
from ndvi import build_ndvi_feature_table
from preprocessing import preprocess_yield_table
from yield_models import train_yield

GROUND = YIELD_FEATURES[:7]
if (MODELS / "image_meta.json").exists() and "--yes" not in sys.argv:
    print("STOP: models/image_meta.json exists, so your real image models (trained in Colab) would be overwritten "
          "by synthetic ones.\nTo refresh only the yield model run:  python retrain_yield.py\n"
          "To really retrain everything run:  python train_and_save.py --yes")
    sys.exit(1)
t0 = time.time()
MODELS.mkdir(exist_ok=True)
meta, metrics = {}, {}

# ---------- 1. data ----------
# Synthetic data is created only for files that are missing, so your own real
# files in ./data are never overwritten. Use --force-synthetic to regenerate.
force = "--force-synthetic" in sys.argv
D = Path("data")
D.mkdir(exist_ok=True)
tab_files = ["soil_weather_yield.csv", "ndvi_timeseries.csv"]
if force or not all((D / f).exists() for f in tab_files):
    print("[1/5] Generating synthetic yield + NDVI data")
    generate_data.make_ndvi_series(generate_data.make_soil_weather_yield())
else:
    print("[1/5] Using existing yield + NDVI files in ./data")
if force or not (D / "crop_recommendation.csv").exists():
    generate_data.make_crop_recommendation()
if force or not (D / "leaf_images.npz").exists():
    generate_images.make_leaf_images()
if force or not (D / "landcover_images.npz").exists():
    generate_images.make_landcover_images()

# ---------- 2. yield regression ----------
print("[2/5] Yield model (XGBoost if installed, otherwise Random Forest; ground + NDVI)")
res = train_yield("data")
if res["note"]:
    print("      ", res["note"])
metrics["yield"] = res["metrics"]
print("      ", metrics["yield"])
joblib.dump(res["model"], MODELS / "yield_rf.joblib")      # file name kept for compatibility
meta["yield_defaults"], meta["yield_ranges"] = res["defaults"], res["ranges"]

# ---------- 3. crop recommender ----------
print("[3/5] Crop recommender (Random Forest, 22 classes)")
cdf = pd.read_csv("data/crop_recommendation.csv")
Xc, yc = cdf[CROP_FEATURES].values, np.asarray(cdf["label"].tolist())
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
crop = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42).fit(Xtr, ytr)
metrics["crop"] = {"accuracy": round(accuracy_score(yte, crop.predict(Xte)), 4)}
print("      ", metrics["crop"])
joblib.dump(crop, MODELS / "crop_rf.joblib")
meta["crop_defaults"] = {c: float(cdf[c].median()) for c in CROP_FEATURES}
meta["crop_ranges"] = {c: [float(cdf[c].quantile(.02)), float(cdf[c].quantile(.98))] for c in CROP_FEATURES}


# ---------- 4/5. CNNs ----------
def train_cnn(npz, name, step):
    print(f"[{step}/5] {name} CNN")
    d = np.load(npz)
    Xi, yr = d["X"], d["y"]
    enc = LabelEncoder().fit(yr)
    yi = enc.transform(yr)
    Xtr, Xte, ytr, yte = train_test_split(Xi, yi, test_size=0.2, random_state=42, stratify=yi)
    model = build_cnn(Xi.shape[1:], len(enc.classes_))
    model.fit(Xtr, ytr, validation_split=0.15, epochs=12, batch_size=32, verbose=0)
    acc = float((model.predict(Xte, verbose=0).argmax(1) == yte).mean())
    metrics[name] = {"accuracy": round(acc, 4)}
    print("      ", metrics[name])
    model.save(MODELS / f"{name}_cnn.keras")
    return list(map(str, enc.classes_))


meta["leaf_classes"] = train_cnn("data/leaf_images.npz", "leaf", 4)
meta["landcover_classes"] = train_cnn("data/landcover_images.npz", "landcover", 5)

meta["metrics"] = metrics
(MODELS / "meta.json").write_text(json.dumps(meta, indent=2))
print(f"\nDone in {time.time() - t0:.0f}s. Models saved in ./models")
print("Next:  streamlit run app.py     (web app)    or    uvicorn api:app --reload     (REST API)")
