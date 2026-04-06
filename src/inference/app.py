"""FastAPI inference application."""

import asyncio
import collections
import logging
import os
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from src.inference.predictor import StockPredictor
from src.inference.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Stock Prediction API",
    description="Real-time stock price direction prediction with drift monitoring.",
    version="1.0.0",
)

Instrumentator().instrument(app).expose(app)

predictor = StockPredictor()
_executor = ThreadPoolExecutor(max_workers=4)

# In-memory ring buffer for drift monitoring (last 1000 inference payloads)
_inference_log: collections.deque = collections.deque(maxlen=1000)
_request_latencies: collections.deque = collections.deque(maxlen=1000)
_request_count: int = 0
_drift_alert: bool = False
_start_time: float = time.time()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    """Load model artefacts once at startup."""
    model_dir = os.environ.get("MODEL_DIR", "models/latest")
    s3_bucket = os.environ.get("MODEL_S3_BUCKET")
    s3_prefix = os.environ.get("MODEL_S3_PREFIX", "models/latest")

    if s3_bucket:
        logger.info("Loading model from S3: s3://%s/%s", s3_bucket, s3_prefix)
        predictor.load_from_s3(s3_bucket, s3_prefix)
    elif os.path.isdir(model_dir):
        logger.info("Loading model from local dir: %s", model_dir)
        predictor.load_from_dir(model_dir)
    else:
        logger.warning(
            "MODEL_DIR '%s' not found and MODEL_S3_BUCKET not set. "
            "Model not loaded — /predict will return 503.",
            model_dir,
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded,
        model_version=predictor.get_version(),
    )


@app.get("/ping")
async def ping() -> dict:
    """SageMaker health check endpoint."""
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/invocations")
async def invocations(req: PredictRequest) -> PredictResponse:
    """SageMaker inference endpoint."""
    return await predict(req)


@app.get("/metrics-summary", response_model=MetricsResponse)
async def metrics_summary() -> MetricsResponse:
    latencies = list(_request_latencies)
    avg_latency = sum(latencies) / len(latencies) * 1000 if latencies else 0.0
    p95_latency = 0.0
    if latencies:
        sorted_lat = sorted(latencies)
        p95_latency = sorted_lat[int(len(sorted_lat) * 0.95)] * 1000

    return MetricsResponse(
        requests_total=_request_count,
        avg_latency_ms=round(avg_latency, 2),
        p95_latency_ms=round(p95_latency, 2),
        drift_alert=_drift_alert,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    global _request_count
    _check_model_loaded()
    t0 = time.perf_counter()

    loop = asyncio.get_event_loop()
    try:
        p50, p10, p90 = await loop.run_in_executor(
            _executor, predictor.predict, req.features
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.perf_counter() - t0
    _request_latencies.append(elapsed)
    _request_count += 1

    # Log to ring buffer for drift detection
    _inference_log.append(req.features)

    return PredictResponse(
        ticker=req.ticker,
        prediction=round(p50, 6),
        lower_bound=round(p10, 6),
        upper_bound=round(p90, 6),
        model_version=predictor.get_version(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    global _request_count
    _check_model_loaded()
    t0 = time.perf_counter()

    loop = asyncio.get_event_loop()
    try:
        p50s, p10s, p90s = await loop.run_in_executor(
            _executor, predictor.predict_batch, req.records
        )
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.perf_counter() - t0
    _request_latencies.append(elapsed)
    _request_count += len(req.records)

    for record in req.records:
        _inference_log.append(record)

    return BatchPredictResponse(
        ticker=req.ticker,
        predictions=[round(v, 6) for v in p50s],
        lower_bounds=[round(v, 6) for v in p10s],
        upper_bounds=[round(v, 6) for v in p90s],
        model_version=predictor.get_version(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        count=len(p50s),
    )


@app.get("/inference-log")
async def get_inference_log() -> dict:
    """Return current ring buffer contents (used by monitoring agent)."""
    return {
        "records": list(_inference_log),
        "count": len(_inference_log),
    }


@app.post("/drift-alert")
async def set_drift_alert(alert: bool) -> dict:
    """Allow monitoring agent to set the drift alert flag."""
    global _drift_alert
    _drift_alert = alert
    logger.warning("Drift alert set to %s", alert)
    return {"drift_alert": _drift_alert}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_model_loaded() -> None:
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check startup logs.",
        )
