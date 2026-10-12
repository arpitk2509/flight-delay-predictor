"""Run with:  pytest -q   (needs artifacts/ to exist, i.e. run `python -m src.train` first)."""
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # `with` triggers the startup event that loads the model
        yield c


def _first_route(client):
    origin = client.get("/api/meta").json()["airports"][0]["code"]
    dest = client.get("/api/destinations", params={"origin": origin}).json()[0]["code"]
    carrier = client.get("/api/carriers", params={"origin": origin, "dest": dest}).json()[0]["code"]
    return origin, dest, carrier


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_predict_ok(client):
    o, d, c = _first_route(client)
    r = client.post("/api/predict", json={"flight_date": "2025-07-03", "dep_time": "18:30",
                                          "carrier": c, "origin": o, "dest": d})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["risk"] in {"low", "medium", "high"}
    assert len(body["reasons"]) > 0
    assert len(body["context"]["hourly_origin_rates"]) == 24


def test_unknown_route_404(client):
    r = client.post("/api/predict", json={"flight_date": "2025-07-03", "dep_time": "18:30",
                                          "carrier": "ZZ", "origin": "ORD", "dest": "ATL"})
    assert r.status_code == 404


def test_bad_time_422(client):
    r = client.post("/api/predict", json={"flight_date": "2025-07-03", "dep_time": "25:99",
                                          "carrier": "AA", "origin": "ORD", "dest": "ATL"})
    assert r.status_code == 422
