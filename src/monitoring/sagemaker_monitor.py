"""SageMaker Model Monitor configuration helpers."""

import logging
import os

import boto3

logger = logging.getLogger(__name__)


def enable_data_capture(
    endpoint_name: str,
    s3_capture_uri: str,
    sampling_percentage: int = 100,
) -> None:
    """Enable Data Capture on a SageMaker endpoint.

    Args:
        endpoint_name: Name of the deployed SageMaker endpoint.
        s3_capture_uri: S3 URI to store captured requests/responses.
        sampling_percentage: Percentage of requests to capture (0-100).
    """
    sm = boto3.client("sagemaker", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    sm.update_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=_get_or_create_endpoint_config(
            endpoint_name, s3_capture_uri, sampling_percentage, sm
        ),
    )
    logger.info(
        "Data capture enabled on endpoint '%s' (%.0f%% sampling → %s)",
        endpoint_name,
        sampling_percentage,
        s3_capture_uri,
    )


def _get_or_create_endpoint_config(
    endpoint_name: str,
    s3_capture_uri: str,
    sampling_percentage: int,
    sm_client,
) -> str:
    """Return an endpoint config name with data capture enabled."""
    import datetime

    config_name = f"{endpoint_name}-capture-{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    # Fetch current config to clone
    ep_desc = sm_client.describe_endpoint(EndpointName=endpoint_name)
    current_config_name = ep_desc["EndpointConfigName"]
    current_config = sm_client.describe_endpoint_config(
        EndpointConfigName=current_config_name
    )

    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=current_config["ProductionVariants"],
        DataCaptureConfig={
            "EnableCapture": True,
            "InitialSamplingPercentage": sampling_percentage,
            "DestinationS3Uri": s3_capture_uri,
            "CaptureOptions": [
                {"CaptureMode": "Input"},
                {"CaptureMode": "Output"},
            ],
            "CaptureContentTypeHeader": {
                "JsonContentTypes": ["application/json"],
            },
        },
    )
    return config_name


def create_monitoring_schedule(
    endpoint_name: str,
    baseline_s3_uri: str,
    output_s3_uri: str,
    role_arn: str,
    schedule_cron: str = "cron(0 * ? * * *)",
) -> str:
    """Create a SageMaker Model Monitor schedule.

    Args:
        endpoint_name: Endpoint to monitor.
        baseline_s3_uri: S3 URI of the baseline statistics (from baseline_capture).
        output_s3_uri: S3 URI where monitor results are written.
        role_arn: IAM role ARN with SageMaker + S3 permissions.
        schedule_cron: Cron expression (default: every hour).

    Returns:
        Monitoring schedule name.
    """
    from sagemaker.model_monitor import DefaultModelMonitor
    from sagemaker.model_monitor.dataset_format import DatasetFormat

    monitor = DefaultModelMonitor(
        role=role_arn,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=20,
        max_runtime_in_seconds=1800,
    )

    schedule_name = f"{endpoint_name}-monitor"
    monitor.create_monitoring_schedule(
        monitor_schedule_name=schedule_name,
        endpoint_input=endpoint_name,
        output_s3_uri=output_s3_uri,
        statistics=f"{baseline_s3_uri}/statistics.json",
        constraints=f"{baseline_s3_uri}/constraints.json",
        schedule_cron_expression=schedule_cron,
        enable_cloudwatch_metrics=True,
    )

    logger.info(
        "Model Monitor schedule '%s' created (schedule=%s)",
        schedule_name,
        schedule_cron,
    )
    return schedule_name


def suggest_baseline(
    training_dataset_s3_uri: str,
    output_s3_uri: str,
    role_arn: str,
) -> str:
    """Run a SageMaker baselining job to generate statistics + constraints.

    Returns:
        S3 URI where baseline outputs were saved.
    """
    from sagemaker.model_monitor import DefaultModelMonitor

    monitor = DefaultModelMonitor(
        role=role_arn,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=20,
    )

    monitor.suggest_baseline(
        baseline_dataset=training_dataset_s3_uri,
        dataset_format={"csv": {"header": True}},
        output_s3_uri=output_s3_uri,
        wait=True,
        logs=False,
    )

    logger.info("Baseline suggestion complete → %s", output_s3_uri)
    return output_s3_uri
