"""
examples.py - ready-made example data for showcasing every feature of AgriSense Cloud.

Tabular examples are built from the project's own data files, so they always match what the models were trained on.
Image examples are read from examples/leaf and examples/landcover (create them with colab_make_example_images.py).
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import CROP_FEATURES, YIELD_FEATURES

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# Used only when data/crop_recommendation.csv is missing. Typical, approximate conditions.
FALLBACK_CROPS = {
    "Typical rice": dict(N=80, P=48, K=40, temperature=24, humidity=82, ph=6.4, rainfall=236),
    "Typical maize": dict(N=78, P=48, K=20, temperature=22, humidity=65, ph=6.2, rainfall=85),
    "Typical chickpea": dict(N=40, P=68, K=80, temperature=19, humidity=17, ph=7.3, rainfall=80),
    "Typical cotton": dict(N=118, P=46, K=20, temperature=24, humidity=79, ph=6.9, rainfall=80),
    "Typical banana": dict(N=100, P=82, K=50, temperature=27, humidity=80, ph=6.0, rainfall=105),
    "Typical coconut": dict(N=22, P=17, K=31, temperature=27, humidity=95, ph=5.9, rainfall=175),
}

# Approximate points chosen for land-cover contrast. Always confirm on the satellite map.
EXAMPLE_FIELDS = [
    {"field_id": "Ex Chennai urban", "lat": 13.0827, "lon": 80.2707, "note": "Built-up city area: expect low NDVI."},
    {"field_id": "Ex Pulicat lake", "lat": 13.6667, "lon": 80.1667, "note": "Lagoon and wetland: expect low or patchy NDVI."},
    {"field_id": "Ex Thanjavur delta", "lat": 10.8, "lon": 79.1, "note": "Cauvery delta, mostly paddy: NDVI depends on the crop stage. Move the marker onto a field."},
    {"field_id": "Ex Nilgiris hills", "lat": 11.38, "lon": 76.62, "note": "Forest and tea hills: expect high NDVI. Check the marker is not on a town."},
]

SHOWCASE = [
    ("Crop recommendation", "Crop recommendation", "Load an example scenario (for example Typical rice)", "The recommended crop matches the scenario name."),
    ("Yield prediction", "Yield prediction", "Load an example field (low, typical, high)", "Predicted yield follows the recorded yield; then drag NDVI in the what-if chart."),
    ("Batch prediction", "Yield prediction", "Tick 'Use the example batch'", "Predicted vs actual yield, with MAE and R²."),
    ("Leaf disease", "Leaf disease (image)", "Pick an example leaf image", "Predicted class matches the known class shown under it."),
    ("Land cover", "Land cover (image)", "Pick an example satellite tile", "Predicted class matches the known class shown under it."),
    ("Satellite NDVI", "Satellite NDVI", "Add the example fields, then Run ingestion now", "Urban and water low, forest high; markers turn green or amber."),
    ("Regional analytics", "Regional analytics", "Opens with simulated data automatically", "Trends, correlations and the underperforming-district chart."),
    ("Model comparison", "Model comparison", "Click Run comparison now", "Random Forest, XGBoost and ANN, with and without NDVI."),
    ("Sensors", "Sensors (simulated)", "Change hours of history", "Daily temperature cycle and rain events."),
    ("System architecture", "System architecture", "Rotate the 3D diagram", "Every component marked Implemented, Partial or Simulated."),
]


# ----------------------------------------------------------------------------- tabular examples
def crop_scenarios(data_dir) -> dict:
    """{name: {feature: value}} with one 'Typical <crop>' scenario per crop (median of that crop in the dataset)."""
    path = Path(data_dir) / "crop_recommendation.csv"
    if path.exists():
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        if set(CROP_FEATURES) | {"label"} <= set(df.columns):
            med = df.groupby("label")[CROP_FEATURES].median()
            return {f"Typical {label}": {f: round(float(row[f]), 3) for f in CROP_FEATURES} for label, row in med.iterrows()}
    return {k: {f: float(v[f]) for f in CROP_FEATURES} for k, v in FALLBACK_CROPS.items()}


def _yield_table(data_dir) -> pd.DataFrame:
    from model_comparison import load_table      # imported here so crop examples work without scikit-learn
    return load_table(data_dir)


def yield_examples(data_dir) -> dict:
    """{name: {"values": {...}, "actual": kg/ha}} for a low, a typical and a high-yield field."""
    t = _yield_table(data_dir).dropna(subset=YIELD_FEATURES + ["yield_kg_ha"]).reset_index(drop=True)
    out = {}
    for name, q in [("Low-yield field (bottom 10%)", 0.10), ("Typical field (median)", 0.50), ("High-yield field (top 10%)", 0.90)]:
        target = t["yield_kg_ha"].quantile(q)
        row = t.iloc[(t["yield_kg_ha"] - target).abs().argmin()]
        out[name] = {"values": {f: round(float(row[f]), 3) for f in YIELD_FEATURES},
                     "actual": float(row["yield_kg_ha"]), "field_id": row["field_id"]}
    return out


def yield_batch(data_dir, n: int = 40, seed: int = 0) -> pd.DataFrame:
    t = _yield_table(data_dir).dropna(subset=YIELD_FEATURES + ["yield_kg_ha"])
    t = t.sample(min(n, len(t)), random_state=seed)
    df = t[YIELD_FEATURES].round(3).reset_index(drop=True)
    df["actual_yield_kg_ha"] = t["yield_kg_ha"].round(1).to_numpy()
    return df


def regional_examples() -> dict:
    """{file name: DataFrame} in the formats the Regional analytics page reads (all simulated)."""
    import regional
    summary, monthly, disease = regional.generate_simulated()
    return {"regional_summary.csv": summary, "regional_monthly.csv": monthly, "disease_cases.csv": disease}


def sensor_example(hours: int = 72) -> pd.DataFrame:
    from sensors import simulate_sensors
    return simulate_sensors(hours=hours, seed=7)


def fields_example() -> pd.DataFrame:
    return pd.DataFrame(EXAMPLE_FIELDS)


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


# ----------------------------------------------------------------------------- image examples
def list_example_images(root, kind: str) -> list:
    """Images in <root>/examples/<kind>/. File names look like '<class>__<number>.jpg'; the class is the known label."""
    folder = Path(root) / "examples" / kind
    if not folder.is_dir():
        return []
    items = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            label = p.stem.rsplit("__", 1)[0] if "__" in p.stem else p.stem
            items.append({"name": f"{label.replace('___', ' - ').replace('_', ' ')} ({p.stem.rsplit('__', 1)[-1]})",
                          "path": str(p), "label": label})
    return items


def zip_examples(root, kind: str) -> bytes:
    folder = Path(root) / "examples" / kind
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.iterdir()) if folder.is_dir() else []:
            if p.suffix.lower() in IMAGE_EXTS:
                z.write(p, p.name)
    return buf.getvalue()
