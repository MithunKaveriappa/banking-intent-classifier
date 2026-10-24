import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # `with` triggers model loading at startup
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_top_k(client):
    r = client.post("/predict", json={"text": "My card has not arrived yet", "top_k": 3})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 3
    scores = [p["score"] for p in preds]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= s <= 1 for s in scores)


def test_predict_sensible_label(client):
    r = client.post("/predict", json={"text": "My card has not arrived yet", "top_k": 3})
    labels = [p["label"] for p in r.json()["predictions"]]
    assert any(l in labels for l in ["card_arrival", "card_delivery_estimate"])


def test_empty_text_rejected(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_blank_text_rejected(client):
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_invalid_top_k_rejected(client):
    assert client.post("/predict", json={"text": "hello", "top_k": 50}).status_code == 422