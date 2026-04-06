#!/usr/bin/env bash
# 03_initial_ingest.sh — run first SageMaker Processing Job to fetch & engineer features
source "$(dirname "$0")/common.sh"

echo "========================================"
echo " Initial Data Ingestion"
echo "========================================"

SAGEMAKER_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${SAGEMAKER_ROLE_NAME}"
JOB_NAME="stock-ingest-initial-$(date -u +%Y%m%d-%H%M%S)"
S3_OUTPUT="s3://${S3_BUCKET}/processed"

echo "Job name:   $JOB_NAME"
echo "Tickers:    $TICKERS"
echo "Lookback:   ${LOOKBACK_DAYS} days"
echo "Output:     $S3_OUTPUT"
echo ""

# Build ContainerArguments JSON array using jq
# Result: ["--tickers", "AAPL", "MSFT", ... , "--lookback-days", "730"]
CONTAINER_ARGS=$(echo "$TICKERS" | tr ',' '\n' | \
  jq -R . | \
  jq -s --arg ld "$LOOKBACK_DAYS" '["--tickers"] + . + ["--lookback-days", $ld]')

APP_SPEC=$(jq -n \
  --arg image "$ECR_INGESTION" \
  --argjson args "$CONTAINER_ARGS" \
  '{ImageUri: $image, ContainerArguments: $args}')

aws sagemaker create-processing-job \
  --processing-job-name "$JOB_NAME" \
  --processing-resources '{
    "ClusterConfig": {
      "InstanceType": "ml.t3.medium",
      "InstanceCount": 1,
      "VolumeSizeInGB": 20
    }
  }' \
  --app-specification "$APP_SPEC" \
  --processing-output-config "{
    \"Outputs\": [{
      \"OutputName\": \"processed\",
      \"S3Output\": {
        \"S3Uri\": \"${S3_OUTPUT}\",
        \"LocalPath\": \"/opt/ml/processing/output\",
        \"S3UploadMode\": \"EndOfJob\"
      }
    }]
  }" \
  --role-arn "$SAGEMAKER_ROLE_ARN" \
  --environment "{\"LOOKBACK_DAYS\": \"${LOOKBACK_DAYS}\"}"

echo "Processing job submitted."
wait_for_sagemaker_job processing "$JOB_NAME"

echo ""
echo "Processed features uploaded to: $S3_OUTPUT"
echo "Verify:"
echo "  aws s3 ls s3://$S3_BUCKET/processed/"
echo ""
echo "========================================"
echo " Ingestion complete."
echo "========================================"
