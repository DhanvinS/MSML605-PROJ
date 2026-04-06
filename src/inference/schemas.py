"""Pydantic request/response schemas for the inference API."""

import re
from typing import Annotated

from pydantic import BaseModel, field_validator, model_validator

TICKER_RE = re.compile(r"^[A-Z]{1,5}$")
N_FEATURES = 30  # must match feature pipeline output


class PredictRequest(BaseModel):
    ticker: str
    features: list[float]

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        v = v.strip().upper()
        if not TICKER_RE.match(v):
            raise ValueError(f"Invalid ticker symbol: {v!r}")
        return v

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: list[float]) -> list[float]:
        if len(v) != N_FEATURES:
            raise ValueError(
                f"Expected {N_FEATURES} features, got {len(v)}"
            )
        import math
        for i, val in enumerate(v):
            if not math.isfinite(val):
                raise ValueError(f"Feature[{i}] is not finite: {val}")
        return v


class PredictResponse(BaseModel):
    ticker: str
    prediction: float       # p50 (median)
    lower_bound: float      # p10
    upper_bound: float      # p90
    model_version: str
    timestamp: str


class BatchPredictRequest(BaseModel):
    ticker: str
    records: list[list[float]]

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        v = v.strip().upper()
        if not TICKER_RE.match(v):
            raise ValueError(f"Invalid ticker symbol: {v!r}")
        return v

    @field_validator("records")
    @classmethod
    def validate_records(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) == 0:
            raise ValueError("records must not be empty")
        if len(v) > 500:
            raise ValueError("Batch size capped at 500 records")
        import math
        for i, row in enumerate(v):
            if len(row) != N_FEATURES:
                raise ValueError(
                    f"records[{i}]: expected {N_FEATURES} features, got {len(row)}"
                )
            for j, val in enumerate(row):
                if not math.isfinite(val):
                    raise ValueError(f"records[{i}][{j}] is not finite: {val}")
        return v


class BatchPredictResponse(BaseModel):
    ticker: str
    predictions: list[float]
    lower_bounds: list[float]
    upper_bounds: list[float]
    model_version: str
    timestamp: str
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class MetricsResponse(BaseModel):
    requests_total: int
    avg_latency_ms: float
    p95_latency_ms: float
    drift_alert: bool
    uptime_seconds: float
