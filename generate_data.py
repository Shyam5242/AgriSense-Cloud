"""
generate_data.py
-----------------
Creates realistic SYNTHETIC datasets that mirror the schema of the real
datasets named in the project (India Agriculture Yield, Crop Recommendation,
PlantVillage leaf images, EuroSAT satellite patches, Sentinel-2 NDVI series).

Why synthetic data: this environment has no internet access to Kaggle /
Google Earth Engine / Sentinel Hub, so the real datasets cannot be
downloaded here. The data-generation logic below encodes realistic
relationships (e.g. yield depends on rainfall/N/NDVI with noise) so that
every downstream model, metric, and chart in this project is a genuine
result of code execution -- not a fabricated number.

Replace this file's output with the real downloaded datasets (same column
names) to get real-world results; the rest of the pipeline is unchanged.
"""
import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)
OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)


def make_soil_weather_yield(n=1500):
    """Mimics the India Agriculture Yield dataset (district-season records)."""
    nitrogen = RNG.normal(80, 20, n).clip(10, 150)
    phosphorus = RNG.normal(45, 15, n).clip(5, 100)
    potassium = RNG.normal(45, 15, n).clip(5, 100)
    rainfall = RNG.gamma(shape=6, scale=150, size=n).clip(100, 3000)
    temperature = RNG.normal(26, 4, n).clip(10, 42)
    humidity = RNG.normal(65, 12, n).clip(20, 100)
    ph = RNG.normal(6.5, 0.7, n).clip(4.5, 9)

    # true but noisy relationship: yield rises with rainfall, N, and a
    # vegetation-health proxy; this proxy becomes the "ground truth" that
    # NDVI (computed later) is designed to correlate with
    veg_health = (0.35 * (rainfall / 1500) + 0.3 * (nitrogen / 120)
                  + 0.2 * (humidity / 100) - 0.15 * abs(temperature - 27) / 10)
    veg_health = veg_health.clip(0, 1)

    yield_kg_ha = (1800 + 2600 * veg_health + 8 * nitrogen + 1.1 * rainfall / 10
                   - 25 * abs(ph - 6.5) * 10 + RNG.normal(0, 220, n))
    yield_kg_ha = yield_kg_ha.clip(300, None)

    field_id = [f"F{str(i).zfill(4)}" for i in range(n)]
    df = pd.DataFrame({
        "field_id": field_id, "nitrogen": nitrogen, "phosphorus": phosphorus,
        "potassium": potassium, "rainfall": rainfall, "temperature": temperature,
        "humidity": humidity, "ph": ph, "veg_health_true": veg_health,
        "yield_kg_ha": yield_kg_ha
    })
    # inject a few missing values + outliers, like real agri data
    for col in ["nitrogen", "rainfall", "ph"]:
        idx = RNG.choice(n, size=int(0.02 * n), replace=False)
        df.loc[idx, col] = np.nan
    df.to_csv(f"{OUT}/soil_weather_yield.csv", index=False)
    return df


def make_ndvi_series(yield_df, days_per_season=12):
    """Simulates Sentinel-2 NDVI time series per field, correlated with
    the hidden veg_health_true used above (standing in for the real
    Sentinel-2 NIR/RED band retrieval + NDVI computation)."""
    rows = []
    for _, r in yield_df.iterrows():
        base = 0.15 + 0.65 * r["veg_health_true"]
        t = np.linspace(0, np.pi, days_per_season)
        seasonal_curve = base * np.sin(t) + RNG.normal(0, 0.03, days_per_season)
        for day, ndvi in zip(range(days_per_season), seasonal_curve):
            rows.append({"field_id": r["field_id"], "day": day,
                         "ndvi": float(np.clip(ndvi, -1, 1))})
    ndvi_df = pd.DataFrame(rows)
    ndvi_df.to_csv(f"{OUT}/ndvi_timeseries.csv", index=False)
    return ndvi_df


def make_crop_recommendation(n=2200):
    """Mimics the Kaggle Crop Recommendation dataset (22 crop classes)."""
    crops = ["rice", "maize", "chickpea", "kidneybeans", "pigeonpeas", "mothbeans",
              "mungbean", "blackgram", "lentil", "pomegranate", "banana", "mango",
              "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya",
              "coconut", "cotton", "jute", "coffee"]
    rows = []
    # deterministic, well-separated per-crop centers (mimics how distinct
    # real crops are in N-P-K/climate space) rather than overlapping
    # random centers, so the classifier has genuine signal to learn from
    for ci, crop in enumerate(crops):
        n_c = n // len(crops)
        n_mean = 20 + (ci * 97 / len(crops)) % 100
        p_mean = 20 + ((ci * 53) % len(crops)) * 4
        k_mean = 20 + ((ci * 71) % len(crops)) * 4
        rain_mean = 400 + ((ci * 37) % len(crops)) * 100
        temp_mean = 18 + ((ci * 13) % len(crops)) * 0.7
        humidity_mean = 40 + ((ci * 19) % len(crops)) * 2.5
        ph_mean = 5.5 + ((ci * 7) % len(crops)) * 0.13
        for _ in range(n_c):
            rows.append({
                "N": RNG.normal(n_mean, 13), "P": RNG.normal(p_mean, 13),
                "K": RNG.normal(k_mean, 13), "temperature": RNG.normal(temp_mean, 2.8),
                "humidity": RNG.normal(humidity_mean, 10), "ph": RNG.normal(ph_mean, 0.45),
                "rainfall": RNG.normal(rain_mean, 130), "label": crop
            })
    df = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
    df.to_csv(f"{OUT}/crop_recommendation.csv", index=False)
    return df


def make_leaf_image_features(n_per_class=220, classes=("healthy", "blight", "rust", "mildew")):
    """Stand-in for PlantVillage leaf images. Real pixels aren't available
    offline, so each 'image' is represented by a feature vector of texture
    + color-channel statistics in the same numeric range CNN pooled
    features would occupy -- enough to demonstrate a genuine train/test
    classification pipeline end to end."""
    X, y = [], []
    for ci, cls in enumerate(classes):
        center = RNG.normal(0, 1, 32) + ci * 0.6
        feats = RNG.normal(center, 1.15, size=(n_per_class, 32))
        X.append(feats)
        y += [cls] * n_per_class
    X = np.vstack(X)
    y = np.array(y)
    np.savez(f"{OUT}/leaf_features.npz", X=X, y=y)
    return X, y


def make_landcover_patch_features(n_per_class=150,
                                    classes=("annual_crop", "forest", "pasture",
                                             "residential", "water")):
    """Stand-in for EuroSAT Sentinel-2 patches (band-statistic features)."""
    X, y = [], []
    for ci, cls in enumerate(classes):
        center = RNG.normal(0, 1, 24) + ci * 0.7
        feats = RNG.normal(center, 1.1, size=(n_per_class, 24))
        X.append(feats)
        y += [cls] * n_per_class
    X = np.vstack(X)
    y = np.array(y)
    np.savez(f"{OUT}/landcover_features.npz", X=X, y=y)
    return X, y


if __name__ == "__main__":
    yield_df = make_soil_weather_yield()
    ndvi_df = make_ndvi_series(yield_df)
    crop_df = make_crop_recommendation()
    make_leaf_image_features()
    make_landcover_patch_features()
    print("Synthetic datasets written to:", OUT)
    print(" - soil_weather_yield.csv  :", yield_df.shape)
    print(" - ndvi_timeseries.csv     :", ndvi_df.shape)
    print(" - crop_recommendation.csv :", crop_df.shape)
    print(" - leaf_features.npz       : (880, 32) across 4 classes")
    print(" - landcover_features.npz  : (750, 24) across 5 classes")
