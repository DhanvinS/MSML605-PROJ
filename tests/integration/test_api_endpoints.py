"""Integration tests for the FastAPI inference application.

These tests spin up the app in-process using httpx + FastAPI TestClient.
The model is NOT loaded (no model dir set), so /predict returns 503.
Tests that require a loaded model are marked with @pytest.mark.requires_model
and skipped in CI unless a model artefact is present.
"""

import os

import pytest
from fastapi.testclient import TestClient

# Prevent the startup event from trying to load a model during tests
os.environ.setdefault("MODEL_DIR", "/nonexistent")

from src.inference.app import app  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)


class TestHealth:
    def test_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_schema(self):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data

    def test_status_ok(self):
        assert client.get("/health").json()["status"] == "ok"


class TestMetrics:
    def test_returns_200(self):
        resp = client.get("/metrics-summary")
        assert resp.status_code == 200

    def test_schema(self):
        data = client.get("/metrics-summary").json()
        for key in ("requests_total", "avg_latency_ms", "p95_latency_ms", "drift_alert"):
            assert key in data


class TestPredictWithoutModel:
    def test_returns_503(self):
        resp = client.post(
            "/predict",
            json={"ticker": "AAPL", "features": [0.0] * 23},
        )
        assert resp.status_code == 503


class TestPredictValidation:
    def test_wrong_feature_count_returns_422(self):
        resp = client.post(
            "/predict",
            json={"ticker": "AAPL", "features": [0.0] * 10},
        )
        assert resp.status_code == 422

    def test_invalid_ticker_returns_422(self):
        resp = client.post(
            "/predict",
            json={"ticker": "aapl123!", "features": [0.0] * 23},
        )
        assert resp.status_code == 422

    def test_nan_feature_returns_422(self):
        features = [0.0] * 23
        features[5] = float("nan")
        resp = client.post(
            "/predict",
            json={"ticker": "AAPL", "features": features},
        )
        assert resp.status_code == 422

    def test_inf_feature_returns_422(self):
        features = [0.0] * 23
        features[5] = float("inf")
        resp = client.post(
            "/predict",
            json={"ticker": "AAPL", "features": features},
        )
        assert resp.status_code == 422


class TestBatchPredictValidation:
    def test_empty_records_returns_422(self):
        resp = client.post(
            "/predict/batch",
            json={"ticker": "AAPL", "records": []},
        )
        assert resp.status_code == 422

    def test_oversized_batch_returns_422(self):
        resp = client.post(
            "/predict/batch",
            json={"ticker": "AAPL", "records": [[0.0] * 23] * 501},
        )
        assert resp.status_code == 422


class TestInferenceLog:
    def test_returns_200(self):
        resp = client.get("/inference-log")
        assert resp.status_code == 200

    def test_schema(self):
        data = client.get("/inference-log").json()
        assert "records" in data
        assert "count" in data


class TestDriftAlert:
    def test_set_alert_true(self):
        resp = client.post("/drift-alert", params={"alert": True})
        assert resp.status_code == 200
        assert resp.json()["drift_alert"] is True

    def test_set_alert_false(self):
        resp = client.post("/drift-alert", params={"alert": False})
        assert resp.status_code == 200
        assert resp.json()["drift_alert"] is False
