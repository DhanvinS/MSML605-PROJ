"""Push drift and model metrics to AWS CloudWatch."""

import logging
import os
from datetime import datetime, timezone

import boto3

from src.monitoring.drift_detector import DriftReport

logger = logging.getLogger(__name__)

NAMESPACE = os.environ.get("CW_NAMESPACE", "StockPredictor/DriftMetrics")


def _cw_client():
    return boto3.client("cloudwatch", region_name=os.environ.get("AWS_REGION", "us-east-1"))


def push_drift_metrics(report: DriftReport, ticker: str = "UNKNOWN") -> None:
    """Push all drift metrics from a DriftReport to CloudWatch.

    Args:
        report: DriftReport returned by drift_detector.analyze_drift().
        ticker: Ticker symbol used as a CloudWatch dimension.
    """
    cw = _cw_client()
    ts = datetime.now(timezone.utc)
    dimensions = [{"Name": "Ticker", "Value": ticker}]

    metric_data = [
        {
            "MetricName": "Overall_Drift_Score",
            "Dimensions": dimensions,
            "Timestamp": ts,
            "Value": report.drift_fraction,
            "Unit": "None",
        },
        {
            "MetricName": "Max_PSI",
            "Dimensions": dimensions,
            "Timestamp": ts,
            "Value": report.max_psi,
            "Unit": "None",
        },
        {
            "MetricName": "Drift_Triggered",
            "Dimensions": dimensions,
            "Timestamp": ts,
            "Value": 1.0 if report.overall_drift_detected else 0.0,
            "Unit": "None",
        },
        {
            "MetricName": "Retrain_Triggered",
            "Dimensions": dimensions,
            "Timestamp": ts,
            "Value": 1.0 if report.trigger_retraining else 0.0,
            "Unit": "None",
        },
        {
            "MetricName": "N_Features_Drifted",
            "Dimensions": dimensions,
            "Timestamp": ts,
            "Value": float(report.n_features_drifted),
            "Unit": "Count",
        },
        {
            "MetricName": "Live_Sample_Count",
            "Dimensions": dimensions,
            "Timestamp": ts,
            "Value": float(report.n_live_samples),
            "Unit": "Count",
        },
    ]

    # Per-feature PSI and KS metrics (batch in groups of 20 — CW limit)
    for feat in report.feature_results:
        feat_dims = dimensions + [{"Name": "Feature", "Value": feat.feature}]
        metric_data.extend(
            [
                {
                    "MetricName": "Feature_PSI",
                    "Dimensions": feat_dims,
                    "Timestamp": ts,
                    "Value": feat.psi,
                    "Unit": "None",
                },
                {
                    "MetricName": "Feature_KS_Statistic",
                    "Dimensions": feat_dims,
                    "Timestamp": ts,
                    "Value": feat.ks_statistic,
                    "Unit": "None",
                },
            ]
        )

    # CloudWatch accepts max 1000 metrics per call; batch by 20
    for i in range(0, len(metric_data), 20):
        batch = metric_data[i : i + 20]
        cw.put_metric_data(Namespace=NAMESPACE, MetricData=batch)

    logger.info(
        "Pushed %d drift metrics to CloudWatch namespace '%s'",
        len(metric_data),
        NAMESPACE,
    )


def push_model_metrics(
    rmse: float,
    mae: float,
    directional_accuracy: float,
    ticker: str = "UNKNOWN",
    model_version: str = "unknown",
) -> None:
    """Push model performance metrics to CloudWatch."""
    cw = _cw_client()
    ts = datetime.now(timezone.utc)
    dimensions = [
        {"Name": "Ticker", "Value": ticker},
        {"Name": "ModelVersion", "Value": model_version},
    ]

    cw.put_metric_data(
        Namespace=NAMESPACE,
        MetricData=[
            {
                "MetricName": "Model_RMSE",
                "Dimensions": dimensions,
                "Timestamp": ts,
                "Value": rmse,
                "Unit": "None",
            },
            {
                "MetricName": "Model_MAE",
                "Dimensions": dimensions,
                "Timestamp": ts,
                "Value": mae,
                "Unit": "None",
            },
            {
                "MetricName": "Directional_Accuracy",
                "Dimensions": dimensions,
                "Timestamp": ts,
                "Value": directional_accuracy,
                "Unit": "None",
            },
        ],
    )
    logger.info("Pushed model performance metrics to CloudWatch")


def create_drift_alarm(
    alarm_name: str,
    sns_topic_arn: str,
    threshold: float = 0.2,
) -> None:
    """Create a CloudWatch alarm that fires when Overall_Drift_Score > threshold."""
    cw = _cw_client()
    cw.put_metric_alarm(
        AlarmName=alarm_name,
        AlarmDescription="Stock prediction model drift detected — trigger retraining",
        MetricName="Overall_Drift_Score",
        Namespace=NAMESPACE,
        Statistic="Maximum",
        Period=3600,  # 1 hour
        EvaluationPeriods=1,
        Threshold=threshold,
        ComparisonOperator="GreaterThanThreshold",
        TreatMissingData="notBreaching",
        AlarmActions=[sns_topic_arn],
        OKActions=[sns_topic_arn],
    )
    logger.info("CloudWatch alarm '%s' created/updated", alarm_name)
