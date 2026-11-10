"""
regional.py - regional / seasonal analytics for AgriSense Cloud.

No Streamlit imports here, so it can be tested on its own.

DATA
  Put your own files in the data/ folder to replace the simulated demo data:
    regional_summary.csv  (required)  district,year,season,rainfall_mm,temperature_c,soil_moisture_pct,
                                      ndvi_mean,ndvi_peak,yield_kg_ha,disease_incidence_pct
    regional_monthly.csv  (optional)  district,year,season,month,ndvi,soil_moisture_pct
    disease_cases.csv     (optional)  district,year,disease,cases
  If regional_summary.csv is missing, SIMULATED data is generated (clearly labelled in the app).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SUMMARY_COLS = ["district", "year", "season", "rainfall_mm", "temperature_c", "soil_moisture_pct",
                "ndvi_mean", "ndvi_peak", "yield_kg_ha", "disease_incidence_pct"]
MONTHLY_COLS = ["district", "year", "season", "month", "ndvi", "soil_moisture_pct"]
DISEASE_COLS = ["district", "year", "disease", "cases"]
SEASONS = ["Kharif", "Rabi"]
CORR_COLS = ["rainfall_mm", "temperature_c", "soil_moisture_pct", "ndvi_mean", "disease_incidence_pct", "yield_kg_ha"]
DISEASES = ["Early blight", "Late blight", "Leaf spot", "Rust", "Powdery mildew"]

# district: (typical seasonal rainfall mm, typical temperature C, yield penalty fraction)
# These numbers are INVENTED for the demo. Villupuram gets a deliberate penalty so the
# "underperforming region" analysis has something to find.
_DISTRICTS = {
    "Thanjavur": (520, 28.5, 0.00),
    "Tiruvallur": (460, 29.5, 0.00),
    "Kancheepuram": (480, 29.2, 0.00),
    "Villupuram": (430, 29.0, -0.10),
    "Madurai": (330, 30.0, 0.00),
    "Coimbatore": (310, 27.0, 0.00),
    "Salem": (380, 28.0, 0.00),
}


def generate_simulated(seed: int = 7, years=range(2019, 2025)):
    """Return (summary, monthly, disease) DataFrames of SIMULATED regional data."""
    rng = np.random.default_rng(seed)
    summary_rows, monthly_rows, disease_rows = [], [], []
    for district, (base_rain, base_temp, penalty) in _DISTRICTS.items():
        for year in years:
            for s_idx, season in enumerate(SEASONS):
                rain = max(80.0, base_rain * (1.0 if s_idx == 0 else 0.8) * rng.normal(1.0, 0.18))
                temp = base_temp + (1.0 if s_idx == 0 else -1.5) + rng.normal(0, 0.6)
                soil = float(np.clip(14 + 0.05 * rain + rng.normal(0, 2.5), 8, 48))
                peak = float(np.clip(0.42 + 0.00045 * rain + 0.004 * (soil - 25) + rng.normal(0, 0.04), 0.3, 0.9))
                ndvi_months = []
                for m in range(1, 7):
                    curve = np.sin(np.pi * (m - 0.5) / 6) ** 1.4
                    ndvi_m = float(np.clip(0.18 + (peak - 0.18) * curve + rng.normal(0, 0.015), 0.05, 0.95))
                    soil_m = float(np.clip(soil * (0.8 + 0.4 * curve) + rng.normal(0, 1.2), 5, 55))
                    ndvi_months.append(ndvi_m)
                    monthly_rows.append([district, year, season, m, round(ndvi_m, 3), round(soil_m, 1)])
                ndvi_mean = float(np.mean(ndvi_months))
                yld = (1500 + 4200 * (ndvi_mean - 0.35) + 1.1 * (rain - 350) - 55 * abs(temp - 27.5)
                       + rng.normal(0, 120)) * (1 + penalty)
                yld = max(600.0, yld)
                dis = float(np.clip(6 + 0.012 * rain - 9 * (ndvi_mean - 0.5) + rng.normal(0, 2.0), 1, 35))
                summary_rows.append([district, year, season, round(rain, 1), round(temp, 1), round(soil, 1),
                                     round(ndvi_mean, 3), round(max(ndvi_months), 3), round(yld, 0), round(dis, 1)])
                total = int(dis * rng.uniform(8, 12))
                counts = rng.multinomial(total, rng.dirichlet([3, 2, 2, 1.5, 1]))
                for d_name, c in zip(DISEASES, counts):
                    disease_rows.append([district, year, d_name, int(c)])
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLS)
    monthly = pd.DataFrame(monthly_rows, columns=MONTHLY_COLS)
    disease = pd.DataFrame(disease_rows, columns=DISEASE_COLS).groupby(["district", "year", "disease"], as_index=False)["cases"].sum()
    return summary, monthly, disease


def _read_checked(path: Path, required: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing column(s): {', '.join(missing)}. Required: {', '.join(required)}")
    return df


def load_regional(data_dir) -> dict:
    """Load real CSVs from data_dir if regional_summary.csv exists, otherwise simulate."""
    data_dir = Path(data_dir)
    path = data_dir / "regional_summary.csv"
    if path.exists():
        summary = _read_checked(path, SUMMARY_COLS)
        monthly = _read_checked(data_dir / "regional_monthly.csv", MONTHLY_COLS) if (data_dir / "regional_monthly.csv").exists() else None
        disease = _read_checked(data_dir / "disease_cases.csv", DISEASE_COLS) if (data_dir / "disease_cases.csv").exists() else None
        return {"summary": summary, "monthly": monthly, "disease": disease, "simulated": False}
    summary, monthly, disease = generate_simulated()
    return {"summary": summary, "monthly": monthly, "disease": disease, "simulated": True}


# ----------------------------------------------------------------------------- analytics
def kpis(df: pd.DataFrame) -> dict:
    first, last = df["year"].min(), df["year"].max()
    y_first = df.loc[df["year"] == first, "yield_kg_ha"].mean()
    y_last = df.loc[df["year"] == last, "yield_kg_ha"].mean()
    return {
        "districts": int(df["district"].nunique()),
        "mean_yield": float(df["yield_kg_ha"].mean()),
        "mean_ndvi": float(df["ndvi_mean"].mean()),
        "mean_soil_moisture": float(df["soil_moisture_pct"].mean()),
        "mean_disease": float(df["disease_incidence_pct"].mean()),
        "yield_change_pct": float((y_last - y_first) / y_first * 100) if first != last and y_first else 0.0,
        "first_year": int(first), "last_year": int(last),
    }


def yield_trend(df: pd.DataFrame):
    """Rows ordered by year then season, plus the ordered list of period labels."""
    d = df.copy()
    d["period"] = d["year"].astype(str) + " " + d["season"].astype(str)
    d["_s"] = d["season"].map({s: i for i, s in enumerate(SEASONS)}).fillna(9)
    d = d.sort_values(["year", "_s", "district"])
    return d[["district", "period", "yield_kg_ha"]], list(dict.fromkeys(d["period"]))


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in CORR_COLS if c in df.columns]
    return df[cols].corr().round(2)


def underperformance(df: pd.DataFrame, threshold_pct: float = -5.0) -> pd.DataFrame:
    """Compare each district's yield with what rainfall, temperature and NDVI predict across ALL districts.
    A negative gap means the district yields less than comparable conditions would suggest."""
    if len(df) < 8:
        return pd.DataFrame(columns=["district", "gap_pct", "status"])
    X = np.column_stack([np.ones(len(df)), df["rainfall_mm"], df["temperature_c"], df["ndvi_mean"]]).astype(float)
    y = df["yield_kg_ha"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = np.clip(X @ beta, 1, None)
    tmp = pd.DataFrame({"district": df["district"].to_numpy(), "gap_pct": (y - pred) / pred * 100})
    out = tmp.groupby("district", as_index=False)["gap_pct"].mean().round(1)
    out["status"] = np.where(out["gap_pct"] < threshold_pct, "Below peers", "On par or above")
    return out.sort_values("gap_pct").reset_index(drop=True)


def district_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["rainfall_mm", "temperature_c", "soil_moisture_pct", "ndvi_mean", "disease_incidence_pct", "yield_kg_ha"]
    return df.groupby("district")[cols].mean().round(2).reset_index()
