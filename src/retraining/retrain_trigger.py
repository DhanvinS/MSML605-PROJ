"""AWS Lambda handler — orchestrates the full retraining pipeline.

Event flow (all handled by this single function):
  1. scheduled / drift_check_complete → SageMaker Processing Job (ingestion)
  2. ingestion_complete               → SageMaker Training Job
  3. training_complete                → Update SageMaker Endpoint

Triggered by:
  - EventBridge schedule (daily)
  - EventBridge: SageMaker Processing Job state=Completed
  - EventBridge: SageMaker Training Job state=Completed
"""

import json
import logging
import os
from datetime import datetime, timezone

import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Environment variables (set by 06_setup_automation.sh)
S3_BUCKET = os.environ.get("S3_BUCKET", "")
ECR_INFERENCE_IMAGE_URI = os.environ.get("ECR_INFERENCE_IMAGE_URI", "")
ECR_TRAINING_IMAGE_URI = os.environ.get("ECR_TRAINING_IMAGE_URI", "")
ECR_INGESTION_IMAGE_URI = os.environ.get("ECR_INGESTION_IMAGE_URI", "")
SAGEMAKER_ROLE = os.environ.get("SAGEMAKER_ROLE_ARN", "")
ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "stock-prediction-endpoint")
TICKERS = os.environ.get("TICKERS", "AAPL,MSFT,GOOGL").split(",")
LOOKBACK_DAYS = os.environ.get("LOOKBACK_DAYS", "730")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
RETRAIN_THRESHOLD = float(os.environ.get("RETRAIN_PSI_THRESHOLD", "0.2"))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def lambda_handler(event: dict, context) -> dict:
    logger.info("Event: %s", json.dumps(event))
    source = _determine_source(event)
    logger.info("Determined source: %s", source)

    # Step 1: Daily schedule → run drift check first
    if source == "scheduled":
        if not _captures_exist():
            logger.info("No captures in S3 yet — skipping drift check until endpoint receives traffic")
            return {"status": "skipped", "reason": "no_captures_yet"}
        job_name = _run_drift_check_job()
        return {"status": "drift_check_launched", "processing_job_name": job_name}

    # Step 1b: Drift check completed → read report, decide whether to retrain
    if source == "drift_check_complete":
        report = _read_drift_report()
        logger.info("Drift report: trigger=%s max_psi=%.4f",
                    report.get("trigger_retraining"), report.get("max_psi", 0))
        if report.get("trigger_retraining", False):
            job_name = _run_ingestion_job()
            _notify(f"Drift detected (max_PSI={report.get('max_psi', 0):.3f}) — ingestion launched: {job_name}")
            return {"status": "ingestion_launched", "processing_job_name": job_name, "drift_report": report}
        return {"status": "no_drift", "max_psi": report.get("max_psi", 0)}

    # Step 2: Ingestion completed → start training
    if source == "ingestion_complete":
        job_name = _launch_training_job()
        return {"status": "training_launched", "training_job_name": job_name}

    # Step 3: Training completed → update endpoint
    if source == "training_complete":
        detail = event.get("detail", {})
        job_name = detail.get("TrainingJobName", "")
        result = _update_endpoint(job_name)
        _notify(f"Model retrained and endpoint updated. Training job: {job_name}")
        return result

    logger.warning("Unknown source: %s", source)
    return {"status": "skipped", "source": source}


# ---------------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------------

def _determine_source(event: dict) -> str:
    # Direct invocation with explicit source
    if "source" in event and not event["source"].startswith("aws."):
        return event["source"]

    # EventBridge SageMaker events
    detail_type = event.get("detail-type", "")
    if detail_type == "SageMaker Processing Job State Change":
        job_name = event.get("detail", {}).get("ProcessingJobName", "")
        if job_name.startswith("drift-check-"):
            return "drift_check_complete"
        if job_name.startswith("stock-ingest-"):
            return "ingestion_complete"
        # Unknown processing job — skip
        return f"unknown_processing_job:{job_name}"

    if detail_type == "SageMaker Training Job State Change":
        return "training_complete"

    # EventBridge scheduled event
    if event.get("source") == "aws.events":
        return "scheduled"

    return "scheduled"


# ---------------------------------------------------------------------------
# Step 1a: Drift check Processing Job
# ---------------------------------------------------------------------------

def _run_drift_check_job() -> str:
    sm = boto3.client("sagemaker")
    job_name = f"drift-check-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    sm.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingResources={
            "ClusterConfig": {
                "InstanceType": "ml.t3.medium",
                "InstanceCount": 1,
                "VolumeSizeInGB": 20,
            }
        },
        AppSpecification={
            "ImageUri": ECR_INFERENCE_IMAGE_URI,
            "ContainerEntrypoint": ["python", "-m", "src.monitoring.run_drift_check"],
        },
        ProcessingInputs=[
            {
                "InputName": "captures",
                "S3Input": {
                    "S3Uri": f"s3://{S3_BUCKET}/captures",
                    "LocalPath": "/opt/ml/processing/input/captures",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "baseline",
                "S3Input": {
                    "S3Uri": f"s3://{S3_BUCKET}/baseline",
                    "LocalPath": "/opt/ml/processing/input/baseline",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
        ],
        ProcessingOutputConfig={
            "Outputs": [
                {
                    "OutputName": "report",
                    "S3Output": {
                        "S3Uri": f"s3://{S3_BUCKET}/monitoring",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        },
        RoleArn=SAGEMAKER_ROLE,
        Environment={
            "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
            "S3_BUCKET": S3_BUCKET,
        },
    )
    logger.info("Drift check job launched: %s", job_name)
    return job_name


# ---------------------------------------------------------------------------
# Step 1b: Read drift report from S3
# ---------------------------------------------------------------------------

def _captures_exist() -> bool:
    """Check if any inference captures have been written to S3."""
    s3 = boto3.client("s3")
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix="captures/", MaxKeys=1)
    return resp.get("KeyCount", 0) > 0


def _read_drift_report() -> dict:
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key="monitoring/latest_drift_report.json")
        return json.loads(obj["Body"].read().decode())
    except Exception as e:
        logger.warning("Could not read drift report: %s. Defaulting to no retrain.", e)
        return {"trigger_retraining": False, "max_psi": 0.0}


# ---------------------------------------------------------------------------
# Step 2: Data ingestion Processing Job
# ---------------------------------------------------------------------------

def _run_ingestion_job() -> str:
    sm = boto3.client("sagemaker")
    job_name = f"stock-ingest-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    s3_output = f"s3://{S3_BUCKET}/processed"

    sm.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingResources={
            "ClusterConfig": {
                "InstanceType": "ml.t3.medium",
                "InstanceCount": 1,
                "VolumeSizeInGB": 20,
            }
        },
        AppSpecification={
            "ImageUri": ECR_INGESTION_IMAGE_URI,
            "ContainerArguments": ["--tickers", *TICKERS, "--lookback-days", LOOKBACK_DAYS],
        },
        ProcessingOutputConfig={
            "Outputs": [
                {
                    "OutputName": "processed",
                    "S3Output": {
                        "S3Uri": s3_output,
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        },
        RoleArn=SAGEMAKER_ROLE,
        Environment={"LOOKBACK_DAYS": LOOKBACK_DAYS},
    )
    logger.info("Ingestion job launched: %s", job_name)
    return job_name


# ---------------------------------------------------------------------------
# Step 3: Training Job
# ---------------------------------------------------------------------------

def _launch_training_job() -> str:
    sm = boto3.client("sagemaker")
    job_name = f"stock-retrain-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    sm.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": ECR_TRAINING_IMAGE_URI,
            "TrainingInputMode": "File",
        },
        RoleArn=SAGEMAKER_ROLE,
        InputDataConfig=[
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{S3_BUCKET}/processed",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/x-parquet",
                "InputMode": "File",
            }
        ],
        OutputDataConfig={"S3OutputPath": f"s3://{S3_BUCKET}/models"},
        ResourceConfig={
            "InstanceType": "ml.m5.xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
        HyperParameters={"config": "configs/model_config.yaml"},
        Environment={
            "S3_BUCKET": S3_BUCKET,
            "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
        },
        Tags=[
            {"Key": "Project", "Value": "StockPrediction"},
            {"Key": "TriggeredBy", "Value": "DriftMonitor"},
        ],
    )
    logger.info("Training job launched: %s", job_name)
    return job_name


# ---------------------------------------------------------------------------
# Step 4: Update endpoint (blue/green, zero downtime)
# ---------------------------------------------------------------------------

def _update_endpoint(job_name: str) -> dict:
    sm = boto3.client("sagemaker")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    # Get model artifact from completed job
    job_desc = sm.describe_training_job(TrainingJobName=job_name)
    model_s3_uri = job_desc["ModelArtifacts"]["S3ModelArtifacts"]

    model_name = f"stock-model-{ts}"
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": ECR_INFERENCE_IMAGE_URI,
            "ModelDataUrl": model_s3_uri,
            "Environment": {
                "SAGEMAKER_PROGRAM": "serve",
                "MODEL_DIR": "/opt/ml/model",
                "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
            },
        },
        ExecutionRoleArn=SAGEMAKER_ROLE,
    )

    config_name = f"stock-config-{ts}"
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": "ml.t2.medium",
                "InitialInstanceCount": 1,
            }
        ],
        DataCaptureConfig={
            "EnableCapture": True,
            "InitialSamplingPercentage": 100,
            "DestinationS3Uri": f"s3://{S3_BUCKET}/captures",
            "CaptureOptions": [
                {"CaptureMode": "Input"},
                {"CaptureMode": "Output"},
            ],
        },
    )

    try:
        sm.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name,
        )
    except sm.exceptions.ClientError as e:
        if "Cannot update in-progress endpoint" in str(e):
            logger.warning("Endpoint already updating — new config queued, will apply after current update completes")
            return {"status": "queued", "model_name": model_name, "training_job": job_name}
        raise

    logger.info("Endpoint '%s' updated to model '%s'", ENDPOINT_NAME, model_name)
    return {"status": "updated", "model_name": model_name, "training_job": job_name}


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

def _notify(message: str) -> None:
    if not SNS_TOPIC_ARN:
        return
    try:
        sns = boto3.client("sns")
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject="[StockPrediction] Retraining Pipeline Update",
            Message=message,
        )
    except Exception as e:
        logger.warning("SNS notification failed: %s", e)
