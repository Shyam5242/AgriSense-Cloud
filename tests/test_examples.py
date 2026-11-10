import numpy as np
import pandas as pd
import pytest
from PIL import Image

import examples as E
from common import CROP_FEATURES, YIELD_FEATURES


def _write_tables(d, n=120, seed=0):
    rng = np.random.default_rng(seed)
    soil = pd.DataFrame({
        "field_id": range(n), "nitrogen": rng.uniform(20, 120, n), "phosphorus": rng.uniform(10, 80, n),
        "potassium": rng.uniform(10, 80, n), "rainfall": rng.uniform(200, 900, n),
        "temperature": rng.uniform(20, 34, n), "humidity": rng.uniform(40, 90, n), "ph": rng.uniform(5.5, 7.8, n)})
    peak = rng.uniform(0.4, 0.9, n)
    soil["yield_kg_ha"] = 1500 + 0.8 * soil["rainfall"] + 3000 * peak + rng.normal(0, 80, n)
    rows = [[i, d_ * 10, peak[i] * np.sin(np.pi * (d_ + 0.5) / 8) ** 1.2] for i in range(n) for d_ in range(8)]
    soil.to_csv(d / "soil_weather_yield.csv", index=False)
    pd.DataFrame(rows, columns=["field_id", "day", "ndvi"]).to_csv(d / "ndvi_timeseries.csv", index=False)


def test_crop_scenarios_from_data_use_medians(tmp_path):
    df = pd.DataFrame({"N": [10, 20, 30, 100], "P": 1, "K": 1, "temperature": 20, "humidity": 50, "ph": 6, "rainfall": 100,
                       "label": ["rice", "rice", "rice", "maize"]})
    df.to_csv(tmp_path / "crop_recommendation.csv", index=False)
    s = E.crop_scenarios(tmp_path)
    assert set(s) == {"Typical rice", "Typical maize"} and s["Typical rice"]["N"] == 20.0 and s["Typical maize"]["N"] == 100.0
    assert set(s["Typical rice"]) == set(CROP_FEATURES)


def test_crop_scenarios_fallback(tmp_path):
    s = E.crop_scenarios(tmp_path)
    assert len(s) >= 5 and all(set(v) == set(CROP_FEATURES) for v in s.values())


def test_yield_examples_and_batch(tmp_path):
    _write_tables(tmp_path)
    ex = E.yield_examples(tmp_path)
    low, mid, high = (ex[k]["actual"] for k in ex)
    assert low < mid < high and all(set(v["values"]) == set(YIELD_FEATURES) for v in ex.values())
    b = E.yield_batch(tmp_path, n=30)
    assert len(b) == 30 and list(b.columns) == YIELD_FEATURES + ["actual_yield_kg_ha"] and not b.isna().any().any()
    pd.testing.assert_frame_equal(b, E.yield_batch(tmp_path, n=30))           # deterministic


def test_yield_examples_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        E.yield_examples(tmp_path)


def test_example_fields_are_valid_for_registration(tmp_path):
    import ingest_satellite as I
    for f in E.EXAMPLE_FIELDS:
        I.add_field(tmp_path / "f.csv", f["field_id"], f["lat"], f["lon"])
    assert len(I.load_fields(tmp_path / "f.csv")) == len(E.EXAMPLE_FIELDS)


def test_regional_and_sensor_examples():
    files = E.regional_examples()
    assert set(files) == {"regional_summary.csv", "regional_monthly.csv", "disease_cases.csv"}
    assert len(E.sensor_example(24)) == 24 and len(E.fields_example()) == len(E.EXAMPLE_FIELDS)
    assert isinstance(E.csv_bytes(E.fields_example()), bytes)


def test_image_listing_and_zip(tmp_path):
    folder = tmp_path / "examples" / "leaf"
    folder.mkdir(parents=True)
    for n in ["Tomato___Early_blight__1.jpg", "Tomato___Early_blight__2.jpg", "healthy__1.png", "notes.txt"]:
        if n.endswith("txt"):
            (folder / n).write_text("x")
        else:
            Image.new("RGB", (8, 8)).save(folder / n)
    items = E.list_example_images(tmp_path, "leaf")
    assert [i["label"] for i in items] == ["Tomato___Early_blight", "Tomato___Early_blight", "healthy"] or len(items) == 3
    assert {i["label"] for i in items} == {"Tomato___Early_blight", "healthy"}
    assert "Tomato - Early blight" in items[0]["name"]
    assert E.list_example_images(tmp_path, "landcover") == []
    import io, zipfile
    names = zipfile.ZipFile(io.BytesIO(E.zip_examples(tmp_path, "leaf"))).namelist()
    assert len(names) == 3 and "notes.txt" not in names
