import pytest
from common import models_ready
pytestmark = pytest.mark.skipif(not models_ready(), reason="train models first: python train_and_save.py")
from fastapi.testclient import TestClient
from api import app
c = TestClient(app)
def test_health(): assert c.get("/").json()["status"] == "ok"
def test_crop():
    r = c.post("/recommend-crop", json=dict(N=60, P=45, K=45, temperature=26, humidity=65, ph=6.5, rainfall=1200))
    assert r.status_code == 200 and len(r.json()["top5"]) == 5
def test_missing_field(): assert c.post("/recommend-crop", json={"N": 1}).status_code == 422
