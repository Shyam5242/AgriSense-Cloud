"""
api.py - AgriSense Cloud REST API (FastAPI)

    uvicorn api:app --reload        then open  http://127.0.0.1:8000/docs
"""
import io
import json

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

import os

import cloud
from common import MODELS, Predictor, models_ready

app = FastAPI(title="AgriSense Cloud API", version="1.0")
_predictor = None


def predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        if not models_ready():
            raise HTTPException(503, "Models not found. Run: python train_and_save.py")
        _predictor = Predictor()
    return _predictor


class YieldInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    rainfall: float
    temperature: float
    humidity: float
    ph: float
    ndvi_mean: float
    ndvi_peak: float
    ndvi_slope: float


class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


def _image(upload: UploadFile):
    try:
        return io.BytesIO(upload.file.read())
    except Exception:
        raise HTTPException(400, "Could not read the uploaded file")


@app.get("/")
def health():
    return {"status": "ok", "models_ready": models_ready()}


class YieldBatch(BaseModel):
    rows: list[YieldInput]


@app.get("/meta")
def meta():
    """Class lists, default values and metrics saved by the training scripts."""
    if not models_ready():
        raise HTTPException(503, "Models not found. Run: python train_and_save.py")
    return json.loads((MODELS / "meta.json").read_text())


@app.post("/predict-yield/batch")
def predict_yield_batch(batch: YieldBatch):
    if not 1 <= len(batch.rows) <= 5000:
        raise HTTPException(422, "Send between 1 and 5000 rows.")
    preds = predictor().predict_yield([r.model_dump() for r in batch.rows])
    return {"predictions": [round(float(p), 1) for p in preds]}


@app.post("/predict-yield")
def predict_yield(data: YieldInput):
    out = {"predicted_yield_kg_ha": round(predictor().predict_yield([data.model_dump()])[0], 1)}
    cloud.log_prediction("yield", data.model_dump(), out)
    return out


@app.post("/recommend-crop")
def recommend_crop(data: CropInput):
    top = predictor().recommend_crop(data.model_dump())
    out = {"recommended": top[0][0], "top5": [{"crop": c, "probability": round(p, 4)} for c, p in top]}
    cloud.log_prediction("crop", data.model_dump(), out)
    return out


@app.post("/detect-disease")
def detect_disease(file: UploadFile = File(...)):
    res = predictor().detect_disease(_image(file))
    out = {"disease": res[0][0], "confidence": round(res[0][1], 4),
           "all": [{"class": c, "probability": round(p, 4)} for c, p in res]}
    cloud.log_prediction("disease", {"file": file.filename}, {"disease": out["disease"], "confidence": out["confidence"]})
    return out


@app.post("/field-landcover")
def field_landcover(file: UploadFile = File(...)):
    res = predictor().classify_landcover(_image(file))
    out = {"landcover_class": res[0][0], "confidence": round(res[0][1], 4),
           "all": [{"class": c, "probability": round(p, 4)} for c, p in res]}
    cloud.log_prediction("landcover", {"file": file.filename}, {"class": out["landcover_class"], "confidence": out["confidence"]})
    return out


@app.get("/ndvi")
def ndvi(lat: float, lon: float, days: int = 30):
    """Mean Sentinel-2 NDVI around a location via Google Earth Engine (set GEE_PROJECT)."""
    if not os.getenv("GEE_PROJECT"):
        raise HTTPException(501, "Set GEE_PROJECT to enable Earth Engine NDVI.")
    from gee_ndvi import mean_ndvi
    v = mean_ndvi(lat, lon, days)
    if v is None:
        raise HTTPException(404, "No cloud-free Sentinel-2 scene in that window.")
    return {"lat": lat, "lon": lon, "days": days, "ndvi_mean": v}


@app.get("/sensor/simulate")
def simulate(n: int = 5):
    """Simulated IoT readings (daily temperature cycle, seasonal moisture, rain events). Returns the last n hours."""
    from sensors import simulate_sensors
    n = max(1, min(n, 168))
    df = simulate_sensors(hours=n)
    return {"simulated": True, "readings": [{
        "timestamp": r.timestamp.isoformat(), "soil_moisture": r.soil_moisture_pct,
        "temperature": r.temperature_c, "humidity": r.humidity_pct} for r in df.itertuples()]}
