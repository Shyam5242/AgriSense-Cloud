"""common.py - shared helpers for training, the Streamlit app and the FastAPI service."""
import json
import os
from pathlib import Path

import joblib

import cloud
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
IMG_SIZE = 64

YIELD_FEATURES = ["nitrogen", "phosphorus", "potassium", "rainfall", "temperature",
                  "humidity", "ph", "ndvi_mean", "ndvi_peak", "ndvi_slope"]
CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def build_cnn(input_shape, num_classes):
    """Compact CNN used for both leaf-disease and land-cover classification."""
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, 3, activation="relu", padding="same"), layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu", padding="same"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"), layers.Dropout(0.35),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def prepare_image(file_or_path, size=IMG_SIZE):
    """Open an image (path, file object or bytes stream) -> float array (1, size, size, 3) in [0, 1]."""
    img = file_or_path if isinstance(file_or_path, Image.Image) else Image.open(file_or_path)
    img = img.convert("RGB").resize((size, size))
    return np.asarray(img, dtype="float32")[None] / 255.0


def models_ready():
    needed = ["yield_rf.joblib", "crop_rf.joblib", "leaf_cnn.keras", "landcover_cnn.keras", "meta.json"]
    for n in needed:
        cloud.ensure_model(MODELS / n)   # no-op unless MODEL_BUCKET is set
    return all((MODELS / n).exists() for n in needed)


class Predictor:
    """Loads every trained model once and exposes simple predict_* methods."""

    def __init__(self):
        if not models_ready():
            raise RuntimeError("Models not found. Run:  python train_and_save.py")
        from tensorflow import keras
        self.meta = json.loads((MODELS / "meta.json").read_text())
        self.yield_model = joblib.load(MODELS / "yield_rf.joblib")
        self.crop_model = joblib.load(MODELS / "crop_rf.joblib")
        self.leaf = keras.models.load_model(MODELS / "leaf_cnn.keras")
        self.land = keras.models.load_model(MODELS / "landcover_cnn.keras")

    def predict_yield(self, rows):
        """rows: list of dicts or 2-D array in YIELD_FEATURES order -> list of kg/ha."""
        X = np.array([[r[f] for f in YIELD_FEATURES] if isinstance(r, dict) else r for r in rows], dtype=float)
        return self.yield_model.predict(X).tolist()

    def recommend_crop(self, row, top=5):
        x = np.array([[row[f] for f in CROP_FEATURES]], dtype=float)
        p = self.crop_model.predict_proba(x)[0]
        idx = np.argsort(p)[::-1][:top]
        return [(str(self.crop_model.classes_[i]), float(p[i])) for i in idx]

    def _cnn(self, model, classes, image):
        size = int(model.input_shape[1])          # works for 64px CNNs and larger transfer-learning models
        p = model.predict(prepare_image(image, size), verbose=0)[0]
        order = np.argsort(p)[::-1]
        return [(classes[i], float(p[i])) for i in order]

    def detect_disease(self, image):
        return self._cnn(self.leaf, self.meta["leaf_classes"], image)

    def classify_landcover(self, image):
        return self._cnn(self.land, self.meta["landcover_classes"], image)
