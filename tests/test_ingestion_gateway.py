from datetime import date

import pandas as pd
import pytest

import architecture as A
import ingest_satellite as I
from api_client import ApiError, ApiPredictor
from common import CROP_FEATURES, YIELD_FEATURES


# ---------------------------------------------------------------- ingestion
def test_add_field_validation(tmp_path):
    p = tmp_path / "fields.csv"
    df = I.add_field(p, "North plot", 13.0, 80.0)
    assert list(df.columns) == I.FIELD_COLS and len(df) == 1
    with pytest.raises(ValueError, match="already exists"):
        I.add_field(p, "North plot", 13.1, 80.1)
    with pytest.raises(ValueError, match="empty"):
        I.add_field(p, "  ", 13, 80)
    with pytest.raises(ValueError, match="Latitude"):
        I.add_field(p, "x", 95, 80)
    with pytest.raises(ValueError, match="letters"):
        I.add_field(p, "<script>", 13, 80)


def test_ingest_statuses_and_dedupe(tmp_path):
    fields = pd.DataFrame({"field_id": ["a", "b", "c"], "lat": [1.0, 2.0, 3.0], "lon": [1.0, 2.0, 3.0]})

    def fetch(lat, lon, days):
        if lat == 2.0:
            return None
        if lat == 3.0:
            raise RuntimeError("boom")
        return 0.55

    out = tmp_path / "obs.csv"
    new = I.ingest(fields, fetch, out=out, today=date(2026, 10, 1))
    assert dict(zip(new["field_id"], new["status"])) == {"a": "ok", "b": "no cloud-free scene", "c": "error: boom"}
    I.ingest(fields, fetch, out=out, today=date(2026, 10, 1))          # same day: replaced, not duplicated
    assert len(I.load_observations(out)) == 3
    I.ingest(fields, fetch, out=out, today=date(2026, 10, 2))
    obs = I.load_observations(out)
    assert len(obs) == 6
    s = I.observation_summary(out)
    assert s["rows"] == 2 and s["fields"] == 1 and s["dates"] == 2 and s["last_date"] == "2026-10-02"
    latest = I.latest_by_field(obs)
    assert list(latest["field_id"]) == ["a"] and latest.iloc[0]["latest_ndvi"] == 0.55


def test_load_fields_missing_column(tmp_path):
    (tmp_path / "f.csv").write_text("field_id,lat\na,1\n")
    with pytest.raises(ValueError, match="lon"):
        I.load_fields(tmp_path / "f.csv")
    assert I.load_fields(tmp_path / "none.csv").empty


def test_remove_field_keeps_or_deletes_history(tmp_path):
    fp, op = tmp_path / "f.csv", tmp_path / "o.csv"
    I.add_field(fp, "a", 1, 1)
    I.add_field(fp, "b", 2, 2)
    fields = I.load_fields(fp)
    I.ingest(fields, lambda lat, lon, d: 0.5, out=op, today=date(2026, 10, 1))
    left = I.remove_field(fp, "a", op, delete_history=False)
    assert list(left["field_id"]) == ["b"] and set(I.load_observations(op)["field_id"]) == {"a", "b"}
    I.remove_field(fp, "b", op, delete_history=True)
    assert I.load_fields(fp).empty and set(I.load_observations(op)["field_id"]) == {"a"}
    with pytest.raises(ValueError, match="No field"):
        I.remove_field(fp, "zzz")


def test_numeric_field_ids_and_empty_history_merge(tmp_path):
    fp, op = tmp_path / "f.csv", tmp_path / "o.csv"
    I.add_field(fp, "Avadi", 13, 80)
    I.add_field(fp, "1", 12, 79)                                   # numeric-looking id
    fields = I.load_fields(fp)
    # no observations yet: merge must work
    assert len(fields.merge(I.latest_by_field(I.load_observations(op)), on="field_id", how="left")) == 2
    op.write_text("field_id,date,ndvi,status\n1,2026-10-01,0.5,ok\n")   # pandas would read the id as an integer
    obs = I.load_observations(op)
    assert obs["field_id"].tolist() == ["1"]
    merged = fields.merge(I.latest_by_field(obs), on="field_id", how="left")
    assert merged.loc[merged["field_id"] == "1", "latest_ndvi"].iloc[0] == 0.5


# ---------------------------------------------------------------- gateway client
class _Resp:
    def __init__(self, data, status=200):
        self._d, self.status_code, self.text = data, status, str(data)

    def json(self):
        return self._d


class _FakeSession:
    def __init__(self, ready=True):
        self.ready, self.calls = ready, []

    def get(self, url, timeout=None):
        self.calls.append(("GET", url))
        if url.endswith("/meta"):
            return _Resp({"leaf_classes": ["a", "b"]})
        return _Resp({"status": "ok", "models_ready": self.ready})

    def post(self, url, timeout=None, json=None, files=None):
        self.calls.append(("POST", url))
        if url.endswith("/predict-yield/batch"):
            return _Resp({"predictions": [1000.0 + i for i in range(len(json["rows"]))]})
        if url.endswith("/recommend-crop"):
            assert set(json) == set(CROP_FEATURES)
            return _Resp({"top5": [{"crop": "rice", "probability": 0.7}, {"crop": "maize", "probability": 0.3}]})
        if url.endswith("/detect-disease"):
            assert files["file"][2] == "image/png" and files["file"][1][:4] == b"\x89PNG"
            return _Resp({"all": [{"class": "healthy", "probability": 0.9}, {"class": "rust", "probability": 0.1}]})
        return _Resp({"detail": "nope"}, 400)


def test_api_predictor_matches_predictor_interface():
    from PIL import Image
    p = ApiPredictor("http://x:8000/", session=_FakeSession())
    assert p.meta["leaf_classes"] == ["a", "b"] and p.base == "http://x:8000"
    row = {f: 1.0 for f in YIELD_FEATURES}
    assert p.predict_yield([row, row]) == [1000.0, 1001.0]
    assert p.predict_yield([[1.0] * len(YIELD_FEATURES)]) == [1000.0]          # arrays as used by the CSV page
    assert p.recommend_crop({f: 1.0 for f in CROP_FEATURES}, top=1) == [("rice", 0.7)]
    assert p.detect_disease(Image.new("RGB", (8, 8)))[0] == ("healthy", 0.9)


def test_api_predictor_errors():
    with pytest.raises(ApiError, match="models are missing"):
        ApiPredictor("http://x", session=_FakeSession(ready=False))
    p = ApiPredictor("http://x", session=_FakeSession())
    from PIL import Image
    with pytest.raises(ApiError, match="400"):
        p.classify_landcover(Image.new("RGB", (8, 8)))


# ---------------------------------------------------------------- status table
def test_status_reflects_reality():
    base = {c: s for c, _, s in A.status_table(False, None)}
    assert base["API gateway"] == "Partial" and base["Satellite ingestion"] == "Partial" and base["Cloud storage"] == "Partial"
    live = {c: s for c, _, s in A.status_table(True, {"rows": 3, "last_date": "2026-10-18"})}
    assert live["API gateway"] == "Implemented" and live["Satellite ingestion"] == "Implemented"
    assert live["Cloud storage"] == "Partial"        # stays partial: nothing is deployed to cloud storage
