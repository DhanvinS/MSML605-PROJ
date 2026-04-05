"""Capture and persist training data statistics for drift detection."""

import json
import logging
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

N_HISTOGRAM_BINS = 20


def capture_baseline_stats(
    X_train: pd.DataFrame,
    output_path: str | None = None,
    s3_bucket: str | None = None,
    s3_key: str = "baseline/baseline_stats.json",
) -> dict:
    """Compute per-feature distribution statistics from the training set.

    These statistics are loaded at inference time by the drift detector to
    compare against live inference data.

    Args:
        X_train: Feature DataFrame (no target column).
        output_path: Local path to write JSON (optional).
        s3_bucket: S3 bucket to upload JSON (optional).
        s3_key: S3 key for the JSON file.

    Returns:
        Dict mapping feature name -> stats dict.
    """
    stats: dict = {}

    for col in X_train.columns:
        series = X_train[col].dropna().values.astype(float)

        if len(series) == 0:
            logger.warning("Column %s has no non-null values, skipping", col)
            continue

        counts, bin_edges = np.histogram(series, bins=N_HISTOGRAM_BINS)

        stats[col] = {
            "mean": float(np.mean(series)),
            "std": float(np.std(series)),
            "min": float(np.min(series)),
            "max": float(np.max(series)),
            "p5": float(np.percentile(series, 5)),
            "p25": float(np.percentile(series, 25)),
            "p50": float(np.percentile(series, 50)),
            "p75": float(np.percentile(series, 75)),
            "p95": float(np.percentile(series, 95)),
            "n_samples": int(len(series)),
            "histogram": {
                "counts": counts.tolist(),
                "bin_edges": bin_edges.tolist(),
            },
        }

    baseline = {
        "features": stats,
        "n_features": len(stats),
        "n_train_samples": len(X_train),
        "feature_names": list(X_train.columns),
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(baseline, f, indent=2)
        logger.info("Baseline stats written to %s", output_path)

    if s3_bucket:
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=s3_bucket,
            Key=s3_key,
            Body=json.dumps(baseline).encode(),
            ContentType="application/json",
        )
        logger.info("Baseline stats uploaded to s3://%s/%s", s3_bucket, s3_key)

    logger.info(
        "Captured baseline stats for %d features from %d samples",
        len(stats),
        len(X_train),
    )
    return baseline


def load_baseline_stats(
    local_path: str | None = None,
    s3_bucket: str | None = None,
    s3_key: str = "baseline/baseline_stats.json",
) -> dict:
    """Load baseline stats from a local file or S3."""
    if local_path and Path(local_path).exists():
        with open(local_path) as f:
            return json.load(f)

    if s3_bucket:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
        return json.loads(obj["Body"].read().decode())

    raise FileNotFoundError("No local path or S3 bucket provided for baseline stats")
