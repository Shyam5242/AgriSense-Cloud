"""
ndvi.py
--------
Implements Algorithm 2 (NDVI Computation from Satellite Bands) and the
seasonal aggregation (mean / peak / slope) used as yield-model features.

compute_ndvi() works on real Sentinel-2 NIR/RED band arrays (e.g. read via
rasterio) exactly as described in the report; here it is exercised on the
synthetic per-field NDVI time series produced by generate_data.py.
"""
import numpy as np
import pandas as pd


def compute_ndvi(nir: np.ndarray, red: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """NDVI = (NIR - RED) / (NIR + RED), clipped to [-1, 1]."""
    nir = nir.astype("float32")
    red = red.astype("float32")
    ndvi = (nir - red) / (nir + red + eps)
    return np.clip(ndvi, -1, 1)


def seasonal_ndvi_features(ndvi_series: np.ndarray) -> dict:
    """Aggregate a per-field NDVI time series into model-ready features."""
    x = np.asarray(ndvi_series, dtype="float64")
    slope = np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0.0
    return {
        "ndvi_mean": float(np.mean(x)),
        "ndvi_peak": float(np.max(x)),
        "ndvi_slope": float(slope),
    }


def build_ndvi_feature_table(ndvi_df: pd.DataFrame) -> pd.DataFrame:
    """Turns the long-format NDVI time series (field_id, day, ndvi) into
    one row of aggregated NDVI features per field."""
    records = []
    for field_id, grp in ndvi_df.groupby("field_id"):
        feats = seasonal_ndvi_features(grp.sort_values("day")["ndvi"].values)
        feats["field_id"] = field_id
        records.append(feats)
    return pd.DataFrame(records)
