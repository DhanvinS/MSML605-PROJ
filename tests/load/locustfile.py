"""Locust load test scenarios for the Stock Prediction API.

Usage:
    # Headless stress ramp (5 → 50 users):
    locust -f tests/load/locustfile.py \
        --host http://your-ec2-ip:8000 \
        --headless -u 50 -r 5 --run-time 120s --html report.html

    # Interactive web UI:
    locust -f tests/load/locustfile.py --host http://localhost:8000

Scenarios:
    1. Baseline     — 5 users, 60s
    2. Stress ramp  — 5→50 users over 120s (find p95 > 200ms point)
    3. Spike        — 100 users instant, 30s
    4. Soak         — 20 users, 600s (detect memory/latency drift)
"""

import random

from locust import HttpUser, between, task


TICKERS = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
N_FEATURES = 30


def _random_features() -> list[float]:
    return [random.gauss(0, 1) for _ in range(N_FEATURES)]


class StockPredictorUser(HttpUser):
    """Simulates a typical API consumer hitting the prediction endpoints."""

    # 0.1–0.5 second think time → ~2–10 RPS per user
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict_single(self):
        """Primary endpoint: single prediction."""
        payload = {
            "ticker": random.choice(TICKERS),
            "features": _random_features(),
        }
        with self.client.post("/predict", json=payload, catch_response=True) as resp:
            if resp.status_code == 503:
                resp.failure("Model not loaded (503)")
            elif resp.status_code not in (200, 422):
                resp.failure(f"Unexpected status {resp.status_code}")
            else:
                resp.success()

    @task(2)
    def predict_batch(self):
        """Batch endpoint: 10 records at once."""
        payload = {
            "ticker": random.choice(TICKERS),
            "records": [_random_features() for _ in range(10)],
        }
        with self.client.post("/predict/batch", json=payload, catch_response=True) as resp:
            if resp.status_code == 503:
                resp.failure("Model not loaded (503)")
            elif resp.status_code not in (200, 422):
                resp.failure(f"Unexpected status {resp.status_code}")
            else:
                resp.success()

    @task(1)
    def health_check(self):
        """Health check: low-cost liveness probe."""
        self.client.get("/health")

    @task(1)
    def metrics_summary(self):
        """Metrics endpoint."""
        self.client.get("/metrics-summary")


class SpikeUser(HttpUser):
    """User profile for spike test: immediate bursts, no think time."""

    wait_time = between(0, 0.05)

    @task
    def predict_single(self):
        payload = {
            "ticker": random.choice(TICKERS),
            "features": _random_features(),
        }
        self.client.post("/predict", json=payload)


class SoakUser(HttpUser):
    """User profile for soak test: moderate sustained load."""

    wait_time = between(1, 3)

    @task(5)
    def predict_single(self):
        payload = {
            "ticker": random.choice(TICKERS),
            "features": _random_features(),
        }
        self.client.post("/predict", json=payload)

    @task(1)
    def health_check(self):
        self.client.get("/health")
