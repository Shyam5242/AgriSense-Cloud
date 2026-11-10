"""
ingest_satellite.py - scheduled satellite ingestion (paper: "a scheduled cloud job pulls satellite imagery per
registered field, computes NDVI, and stores results").

For every field in data/fields.csv it fetches the mean Sentinel-2 NDVI from Google Earth Engine and appends it to
data/ndvi_observations.csv (one row per field per day; running twice on a day replaces that day's row).

    python ingest_satellite.py                   # run once
    python ingest_satellite.py --loop-hours 24   # keep running, once every 24 hours
Needs:  set GEE_PROJECT=your-project-id   and   earthengine authenticate
To run it automatically every day on Windows, see run_ingestion.bat and INSTRUCTIONS.md.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
FIELDS_PATH = DATA / "fields.csv"
OBS_PATH = DATA / "ndvi_observations.csv"
FIELD_COLS = ["field_id", "lat", "lon"]
OBS_COLS = ["field_id", "date", "ndvi", "status"]


def load_fields(path=FIELDS_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=FIELD_COLS)
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in FIELD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing column(s): {', '.join(missing)}. Required: {', '.join(FIELD_COLS)}")
    df["field_id"] = df["field_id"].astype(str).str.strip()
    return df[FIELD_COLS]


def add_field(path, field_id, lat, lon) -> pd.DataFrame:
    field_id = str(field_id).strip()
    if not field_id:
        raise ValueError("Field name cannot be empty.")
    if not re.fullmatch(r"[\w\- ]{1,40}", field_id):
        raise ValueError("Use letters, numbers, spaces, - or _ only (max 40 characters).")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Latitude must be between -90 and 90, longitude between -180 and 180.")
    df = load_fields(path)
    if field_id in set(df["field_id"]):
        raise ValueError(f"A field called '{field_id}' already exists.")
    df = pd.concat([df, pd.DataFrame([{"field_id": field_id, "lat": float(lat), "lon": float(lon)}])], ignore_index=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def remove_field(path, field_id, obs_path=None, delete_history: bool = False) -> pd.DataFrame:
    """Remove a registered field. With delete_history=True its stored NDVI observations are deleted too."""
    df = load_fields(path)
    if field_id not in set(df["field_id"]):
        raise ValueError(f"No field called '{field_id}'.")
    df = df[df["field_id"] != field_id].reset_index(drop=True)
    df.to_csv(path, index=False)
    if delete_history and obs_path is not None and Path(obs_path).exists():
        obs = load_observations(obs_path)
        obs[obs["field_id"] != field_id].to_csv(obs_path, index=False)
    return df


def _empty_observations() -> pd.DataFrame:
    return pd.DataFrame({"field_id": pd.Series(dtype="object"), "date": pd.Series(dtype="object"),
                         "ndvi": pd.Series(dtype="float64"), "status": pd.Series(dtype="object")})


def load_observations(path=OBS_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return _empty_observations()
    df = pd.read_csv(path)
    for c in OBS_COLS:
        if c not in df.columns:
            df[c] = None
    df["field_id"] = df["field_id"].astype(str).str.strip()     # ids like "1" must not become integers
    df["date"] = df["date"].astype(str)
    df["ndvi"] = pd.to_numeric(df["ndvi"], errors="coerce")
    return df[OBS_COLS]


def ingest(fields: pd.DataFrame, fetch, days: int = 30, out=OBS_PATH, today: date | None = None) -> pd.DataFrame:
    """fetch(lat, lon, days) -> NDVI float, or None when there is no cloud-free scene. Returns the new rows."""
    day = (today or date.today()).isoformat()
    rows = []
    for r in fields.itertuples():
        try:
            v = fetch(float(r.lat), float(r.lon), days)
            rows.append({"field_id": r.field_id, "date": day, "ndvi": v,
                         "status": "ok" if v is not None else "no cloud-free scene"})
        except Exception as e:  # one failing field must not stop the others
            rows.append({"field_id": r.field_id, "date": day, "ndvi": None, "status": f"error: {e}"[:200]})
    new = pd.DataFrame(rows, columns=OBS_COLS)
    old = load_observations(out)
    combined = new if old.empty else pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates(["field_id", "date"], keep="last").sort_values(["field_id", "date"])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    return new


def latest_by_field(obs: pd.DataFrame) -> pd.DataFrame:
    ok = obs[obs["ndvi"].notna()]
    if ok.empty:
        return pd.DataFrame({"field_id": pd.Series(dtype="object"), "latest_ndvi": pd.Series(dtype="float64"),
                             "latest_date": pd.Series(dtype="object")})
    last = ok.sort_values("date").groupby("field_id", as_index=False).last()
    return last.rename(columns={"ndvi": "latest_ndvi", "date": "latest_date"})[["field_id", "latest_ndvi", "latest_date"]]


def observation_summary(path=OBS_PATH) -> dict:
    obs = load_observations(path)
    ok = obs[obs["ndvi"].notna()]
    return {"rows": int(len(ok)), "fields": int(ok["field_id"].nunique()), "dates": int(ok["date"].nunique()),
            "last_date": str(ok["date"].max()) if len(ok) else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fetch NDVI for every registered field and store it.")
    ap.add_argument("--days", type=int, default=30, help="look-back window for Sentinel-2 scenes")
    ap.add_argument("--loop-hours", type=float, default=0, help="repeat every N hours (0 = run once)")
    ap.add_argument("--fields", default=str(FIELDS_PATH))
    ap.add_argument("--out", default=str(OBS_PATH))
    args = ap.parse_args(argv)
    if not os.getenv("GEE_PROJECT"):
        print("Set GEE_PROJECT first (and run: earthengine authenticate).")
        return 1
    from gee_ndvi import mean_ndvi
    while True:
        fields = load_fields(args.fields)
        if fields.empty:
            print(f"No fields registered in {args.fields}. Add some in the app (Satellite NDVI page).")
            return 1
        new = ingest(fields, mean_ndvi, days=args.days, out=args.out)
        print(f"{date.today()}: {int((new['status'] == 'ok').sum())} of {len(new)} fields updated")
        for r in new[new["status"] != "ok"].itertuples():
            print(f"  {r.field_id}: {r.status}")
        if args.loop_hours <= 0:
            return 0
        time.sleep(args.loop_hours * 3600)


if __name__ == "__main__":
    sys.exit(main())
