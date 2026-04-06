"""Drift check SageMaker Processing Job script.

Reads SageMaker Data Capture JSONL files, computes drift vs baseline,
pushes metrics to CloudWatch, and writes a drift report to S3.

SageMaker mounts:
  /opt/ml/processing/input/captures/   ← S3 captures/ prefix (last N days)
  /opt/ml/processing/input/baseline/   ← S3 baseline/ prefix
  /opt/ml/processing/output/           ← written report is uploaded to S3
"""

import base64
import json
import logging
import os
from pathlib import Path

import boto3
import pandas as pd

from src.monitoring.cloudwatch_logger import push_drift_metrics
from src.monitoring.drift_detector import analyze_drift
from src.training.baseline_capture import load_baseline_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_CAPTURES = Path(os.environ.get("SM_CAPTURES_DIR", "/opt/ml/processing/input/captures"))
INPUT_BASELINE = Path(os.environ.get("SM_BASELINE_DIR", "/opt/ml/processing/input/baseline"))
OUTPUT_DIR = Path(os.environ.get("SM_OUTPUT_DIR", "/opt/ml/processing/output"))
MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", "50"))


def _parse_capture_files(captures_dir: Path) -> list[list[float]]:
    """Parse SageMaker Data Capture JSONL files and extract feature vectors.

    SageMaker stores each inference as a JSONL line with base64-encoded request body.
    The request body is: {"ticker": "AAPL", "features": [f1, f2, ...]}
    """
    all_features: list[list[float]] = []
    jsonl_files = list(captures_dir.rglob("*.jsonl"))
    logger.info("Found %d capture files in %s", len(jsonl_files), captures_dir)

    for fpath in jsonl_files:
        try:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        capture_data = record.get("captureData", {})
                        endpoint_input = capture_data.get("endpointInput", {})

                        encoding = endpoint_input.get("encoding", "BASE64")
                        raw_data = endpoint_input.get("data", "")

                        if encoding == "BASE64":
                            decoded = base64.b64decode(raw_data).decode("utf-8")
                        else:
                            decoded = raw_data

                        payload = json.loads(decoded)
                        features = payload.get("features")
                        if features and isinstance(features, list):
                            all_features.append(features)
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.debug("Skipping malformed capture line: %s", e)
        except Exception as e:
            logger.warning("Could not read capture file %s: %s", fpath, e)

    logger.info("Parsed %d inference records from captures", len(all_features))
    return all_features


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load baseline stats -----------------------------------------------
    baseline_path = INPUT_BASELINE / "baseline_stats.json"
    if not baseline_path.exists():
        logger.error("Baseline stats not found at %s", baseline_path)
        raise FileNotFoundError(f"Missing baseline stats: {baseline_path}")

    baseline_stats = load_baseline_stats(local_path=str(baseline_path))
    feature_names = baseline_stats.get("feature_names", [])
    logger.info("Loaded baseline stats: %d features", len(feature_names))

    # --- Parse capture files -----------------------------------------------
    raw_features = _parse_capture_files(INPUT_CAPTURES)

    if len(raw_features) < MIN_SAMPLES:
        logger.warning(
            "Only %d samples in captures (need >= %d). Writing empty report.",
            len(raw_features),
            MIN_SAMPLES,
        )
        report_dict = {
            "trigger_retraining": False,
            "overall_drift_detected": False,
            "max_psi": 0.0,
            "n_live_samples": len(raw_features),
            "reason": f"Insufficient samples ({len(raw_features)} < {MIN_SAMPLES})",
        }
        report_path = OUTPUT_DIR / "latest_drift_report.json"
        report_path.write_text(json.dumps(report_dict, indent=2))
        logger.info("Wrote empty report to %s", report_path)
        return

    # --- Build live feature DataFrame --------------------------------------
    if feature_names:
        live_df = pd.DataFrame(raw_features, columns=feature_names)
    else:
        live_df = pd.DataFrame(raw_features)
        live_df.columns = [f"feature_{i}" for i in range(live_df.shape[1])]

    logger.info("Live window: %d samples × %d features", len(live_df), live_df.shape[1])

    # --- Run drift analysis ------------------------------------------------
    report = analyze_drift(baseline_stats, live_df)
    report_dict = report.to_dict()

    # --- Push metrics to CloudWatch ----------------------------------------
    try:
        push_drift_metrics(report, ticker="ALL")
        logger.info("Drift metrics pushed to CloudWatch")
    except Exception as e:
        logger.warning("CloudWatch push failed (non-fatal): %s", e)

    # --- Write drift report ------------------------------------------------
    report_json = json.dumps(report_dict, indent=2)

    # Write to /tmp/ first (always writable), then try the SageMaker output dir
    tmp_path = Path("/tmp/latest_drift_report.json")
    tmp_path.write_text(report_json)

    # Try SageMaker output mount (may fail if non-root user)
    try:
        report_path = OUTPUT_DIR / "latest_drift_report.json"
        report_path.write_text(report_json)
        logger.info("Drift report written to %s", report_path)
    except PermissionError:
        logger.warning("Cannot write to %s (permission denied) — uploading directly to S3", OUTPUT_DIR)

    # Always upload directly to S3 so Lambda can read it regardless
    s3_bucket = os.environ.get("S3_BUCKET", "")
    if s3_bucket:
        try:
            import boto3
            boto3.client("s3").put_object(
                Bucket=s3_bucket,
                Key="monitoring/latest_drift_report.json",
                Body=report_json.encode(),
                ContentType="application/json",
            )
            logger.info("Drift report uploaded directly to s3://%s/monitoring/latest_drift_report.json", s3_bucket)
        except Exception as e:
            logger.error("Failed to upload drift report to S3: %s", e)
    logger.info(
        "Drift result — trigger_retraining=%s  max_psi=%.4f  drifted_features=%d/%d",
        report.trigger_retraining,
        report.max_psi,
        report.n_features_drifted,
        len(report.feature_results),
    )


if __name__ == "__main__":
    run()
