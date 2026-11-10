"""
api_client.py - lets the Streamlit app talk to the FastAPI gateway over HTTP instead of loading the models itself.

Set AGRISENSE_API_URL (for example http://127.0.0.1:8000) and the app uses ApiPredictor, which has the same methods
as common.Predictor: predict_yield, recommend_crop, detect_disease, classify_landcover, and a .meta attribute.
"""
from __future__ import annotations

import io

from common import CROP_FEATURES, YIELD_FEATURES


class ApiError(RuntimeError):
    pass


class ApiPredictor:
    def __init__(self, base_url: str, timeout: float = 180, session=None):
        if session is None:
            import requests
            session = requests.Session()
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session
        health = self._get("/")
        if not health.get("models_ready"):
            raise ApiError("The API is running but its models are missing. Run: python train_and_save.py")
        self.meta = self._get("/meta")

    # ------------------------------------------------------------------ transport
    def _check(self, resp):
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise ApiError(f"API returned {resp.status_code}: {detail}")
        return resp.json()

    def _get(self, path):
        try:
            return self._check(self.session.get(self.base + path, timeout=self.timeout))
        except ApiError:
            raise
        except Exception as e:
            raise ApiError(f"Cannot reach the API at {self.base}: {e}")

    def _post(self, path, **kw):
        try:
            return self._check(self.session.post(self.base + path, timeout=self.timeout, **kw))
        except ApiError:
            raise
        except Exception as e:
            raise ApiError(f"Cannot reach the API at {self.base}: {e}")

    @staticmethod
    def _png(image) -> bytes:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()

    # ------------------------------------------------------------------ same interface as Predictor
    def predict_yield(self, rows):
        payload = [dict(r) if isinstance(r, dict) else dict(zip(YIELD_FEATURES, map(float, r))) for r in rows]
        payload = [{f: float(p[f]) for f in YIELD_FEATURES} for p in payload]
        return list(self._post("/predict-yield/batch", json={"rows": payload})["predictions"])

    def recommend_crop(self, row, top=5):
        out = self._post("/recommend-crop", json={f: float(row[f]) for f in CROP_FEATURES})
        return [(d["crop"], float(d["probability"])) for d in out["top5"]][:top]

    def detect_disease(self, image):
        out = self._post("/detect-disease", files={"file": ("leaf.png", self._png(image), "image/png")})
        return [(d["class"], float(d["probability"])) for d in out["all"]]

    def classify_landcover(self, image):
        out = self._post("/field-landcover", files={"file": ("tile.png", self._png(image), "image/png")})
        return [(d["class"], float(d["probability"])) for d in out["all"]]
