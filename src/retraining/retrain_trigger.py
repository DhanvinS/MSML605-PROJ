"""AWS Lambda handler for triggering SageMaker retraining jobs."""

import json
import logging
import os
from datetime import datetime, timezone

import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RETRAIN_THRESHOLD = float(os.environ.get("RETRAIN_PSI_THRESHOLD", "0.2"))
ECR_IMAGE_URI = os.environ.get("ECR_TRAINING_IMAGE_URI", "")
S3_INPUT_URI = os.environ.get("S3_TRAINING_DATA_URI", "")
S3_OUTPUT_URI = os.environ.get("S3_MODEL_OUTPUT_URI", "")
SAGEMAKER_ROLE = os.environ.get("SAGEMAKER_ROLE_ARN", "")
ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "stock-prediction-endpoint")


def lambda_handler(event: dict, context) -> dict:
    """Entry point for the Lambda function.

    Triggered by:
    - CloudWatch Alarm → SNS → Lambda  (source: 'drift_alarm')
    - EventBridge schedule             (source: 'scheduled')
    """
    logger.info("Event: %s", json.dumps(event))

    source = _determine_source(event)
    drift_score = _extract_drift_score(event)

    if source == "drift_alarm" and drift_score is not None:
        if drift_score < RETRAIN_THRESHOLD:
            msg = f"Drift score {drift_score:.3f} below threshold {RETRAIN_THRESHOLD} — skipping"
            logger.info(msg)
            return {"status": "skipped", "reason": msg}

    job_name = _launch_training_job()
    return {"status": "launched", "training_job_name": job_name}


def _launch_training_job() -> str:
    sm = boto3.client("sagemaker")
    job_name = f"stock-retrain-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    sm.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": ECR_IMAGE_URI,
            "TrainingInputMode": "File",
        },
        RoleArn=SAGEMAKER_ROLE,
        InputDataConfig=[
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": S3_INPUT_URI,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/x-parquet",
                "InputMode": "File",
            }
        ],
        OutputDataConfig={"S3OutputPath": S3_OUTPUT_URI},
        ResourceConfig={
            "InstanceType": "ml.m5.xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
        HyperParameters={
            "config": "configs/model_config.yaml",
        },
        Tags=[
            {"Key": "Project", "Value": "StockPrediction"},
            {"Key": "TriggeredBy", "Value": "DriftMonitor"},
        ],
    )

    logger.info("SageMaker training job launched: %s", job_name)
    return job_name


def _determine_source(event: dict) -> str:
    if "source" in event:
        return event["source"]
    if "Records" in event:
        # SNS notification from CloudWatch alarm
        return "drift_alarm"
    return "scheduled"


def _extract_drift_score(event: dict) -> float | None:
    try:
        if "Records" in event:
            sns_msg = json.loads(event["Records"][0]["Sns"]["Message"])
            return float(sns_msg.get("drift_score", RETRAIN_THRESHOLD + 1))
        return event.get("drift_score")
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Post-training update handler (second Lambda, triggered by EventBridge
# when the training job state changes to "Completed")
# ---------------------------------------------------------------------------

def update_endpoint_handler(event: dict, context) -> dict:
    """Update the SageMaker endpoint with the newly trained model."""
    detail = event.get("detail", {})
    job_name = detail.get("TrainingJobName", "")
    status = detail.get("TrainingJobStatus", "")

    if status != "Completed":
        logger.info("Training job '%s' status '%s' — no action", job_name, status)
        return {"status": "skipped"}

    sm = boto3.client("sagemaker")

    # Get model artifact location from completed training job
    job_desc = sm.describe_training_job(TrainingJobName=job_name)
    model_s3_uri = job_desc["ModelArtifacts"]["S3ModelArtifacts"]

    # Create new SageMaker model
    model_name = f"stock-model-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": ECR_IMAGE_URI,
            "ModelDataUrl": model_s3_uri,
            "Environment": {"SAGEMAKER_PROGRAM": "train_xgboost.py"},
        },
        ExecutionRoleArn=SAGEMAKER_ROLE,
    )

    # Create new endpoint config
    config_name = f"stock-config-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": "ml.t3.medium",
                "InitialInstanceCount": 1,
            }
        ],
    )

    # Update the endpoint in-place (zero-downtime blue/green)
    sm.update_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=config_name,
    )

    logger.info(
        "Endpoint '%s' updated to model '%s' from job '%s'",
        ENDPOINT_NAME,
        model_name,
        job_name,
    )
    return {"status": "updated", "model_name": model_name}
