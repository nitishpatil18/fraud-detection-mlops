"""tests for the fraud detection api.

uses fastapi's TestClient, which runs the app in-process without a real
server. we patch the model loader so tests don't need mlflow running.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """spin up the app with a fake model."""
    fake_model = MagicMock()
    fake_model.run_id = "test_run_id"
    fake_model.feature_names = ["TransactionAmt", "ProductCD", "card1"]
    fake_model.threshold = 0.5
    fake_model.predict.return_value = (0.73, True)

    with patch("src.api.main.load_from_env", return_value=fake_model):
        from src.api.main import app
        with TestClient(app) as c:
            yield c


def test_health(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_info(client) -> None:
    r = client.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_run_id"] == "test_run_id"
    assert body["n_features"] == 3
    assert body["threshold"] == 0.5


def test_predict_valid_request(client) -> None:
    r = client.post("/predict", json={"TransactionAmt": 50.0, "ProductCD": "W"})
    assert r.status_code == 200
    body = r.json()
    assert body["fraud_probability"] == 0.73
    assert body["is_fraud"] is True
    assert body["model_run_id"] == "test_run_id"


def test_predict_negative_amount_rejected(client) -> None:
    r = client.post("/predict", json={"TransactionAmt": -10})
    assert r.status_code == 422


def test_predict_accepts_extra_fields(client) -> None:
    r = client.post("/predict", json={
        "TransactionAmt": 50.0,
        "C1": 1.0,
        "V100": 0.5,
    })
    assert r.status_code == 200