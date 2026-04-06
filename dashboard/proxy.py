"""Local proxy server — bridges the Streamlit dashboard to the SageMaker endpoint.

Run with:
    uvicorn dashboard.proxy:app --port 8000 --reload

The dashboard connects to http://localhost:8000.
This proxy forwards /predict and /invocations to SageMaker and serves
/health, /metrics-summary from local state.
"""

import json
import os
import time
import collections
from datetime import datetime, timezone

import boto3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "stock-prediction-endpoint")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

app = FastAPI(title="SageMaker Proxy", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_sm_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
_sm = boto3.client("sagemaker", region_name=AWS_REGION)

_request_latencies: collections.deque = collections.deque(maxlen=1000)
_request_count = 0
_start_time = time.time()


def _get_model_version() -> str:
    try:
        ep = _sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        config_name = ep["EndpointConfigName"]
        cfg = _sm.describe_endpoint_config(EndpointConfigName=config_name)
        model_name = cfg["ProductionVariants"][0]["ModelName"]
        return model_name
    except Exception:
        return "unknown"


def _invoke(body: dict) -> dict:
    global _request_count
    t0 = time.perf_counter()
    resp = _sm_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(body),
    )
    result = json.loads(resp["Body"].read())
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _request_latencies.append(elapsed_ms)
    _request_count += 1
    return result


# --- Health ---
@app.get("/health")
def health():
    try:
        status = _sm.describe_endpoint(EndpointName=ENDPOINT_NAME)["EndpointStatus"]
        loaded = status == "InService"
    except Exception:
        loaded = False
    return {
        "status": "ok" if loaded else "unavailable",
        "model_loaded": loaded,
        "model_version": _get_model_version(),
    }


# --- Predict ---
class PredictRequest(BaseModel):
    ticker: str
    features: list[float]


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return _invoke({"ticker": req.ticker, "features": req.features})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/invocations")
def invocations(req: PredictRequest):
    return predict(req)


# --- Metrics summary ---
@app.get("/metrics-summary")
def metrics_summary():
    lats = list(_request_latencies)
    avg = sum(lats) / len(lats) if lats else 0.0
    p95 = sorted(lats)[int(len(lats) * 0.95)] if lats else 0.0
    return {
        "requests_total": _request_count,
        "avg_latency_ms": round(avg, 2),
        "p95_latency_ms": round(p95, 2),
        "drift_alert": False,
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


# --- Latency history (buckets by minute) ---
_latency_buckets: list[dict] = []  # [{timestamp, p50_ms, p95_ms, p99_ms}]
_last_bucket_time: float = time.time()


@app.get("/latency-history")
def latency_history():
    global _last_bucket_time
    now = time.time()
    lats = sorted(_request_latencies)
    # Emit a new bucket every 60 seconds if there are requests
    if lats and (now - _last_bucket_time) >= 60:
        n = len(lats)
        _latency_buckets.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "p50_ms": round(lats[int(n * 0.50)], 2),
            "p95_ms": round(lats[int(n * 0.95)], 2),
            "p99_ms": round(lats[min(int(n * 0.99), n - 1)], 2),
        })
        _last_bucket_time = now
        # Keep last 1440 buckets (24 hours at 1/min)
        if len(_latency_buckets) > 1440:
            _latency_buckets.pop(0)
    return _latency_buckets


# --- Ping ---
@app.get("/ping")
def ping():
    return {"status": "ok"}
