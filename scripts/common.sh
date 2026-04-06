#!/usr/bin/env bash
# common.sh — sourced by all deployment scripts
# Usage: source "$(dirname "$0")/common.sh"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load .env
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
else
  echo "ERROR: .env not found at $PROJECT_ROOT/.env" >&2
  exit 1
fi

# Resolve account ID at runtime
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_ACCOUNT_ID

# Core variables
AWS_REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET_NAME}"
ECR_BASE="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_INGESTION="${ECR_BASE}/stock-prediction/ingestion:latest"
ECR_TRAINING="${ECR_BASE}/stock-prediction/training:latest"
ECR_INFERENCE="${ECR_BASE}/stock-prediction/inference:latest"

ENDPOINT_NAME="stock-prediction-endpoint"
LAMBDA_FUNCTION="stock-prediction-orchestrator"
SAGEMAKER_ROLE_NAME="stock-prediction-sagemaker-role"
LAMBDA_ROLE_NAME="stock-prediction-lambda-role"
CODEBUILD_ROLE_NAME="stock-prediction-codebuild-role"
CODEBUILD_PROJECT="stock-prediction-build"
SNS_TOPIC_NAME="stock-prediction-alerts"
TICKERS="${TICKERS:-AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-730}"

# Helper: wait for a SageMaker job to complete
# Usage: wait_for_sagemaker_job training stock-retrain-20240101-120000
wait_for_sagemaker_job() {
  local job_type="$1"   # training | processing
  local job_name="$2"
  local describe_cmd="describe-${job_type}-job"
  local status_key="${job_type^}JobStatus"   # TrainingJobStatus | ProcessingJobStatus

  echo "Waiting for SageMaker $job_type job: $job_name"
  while true; do
    STATUS=$(aws sagemaker "${describe_cmd}" \
      --"${job_type}-job-name" "$job_name" \
      --query "${status_key}" \
      --output text 2>/dev/null || echo "UNKNOWN")
    echo "  $(date -u '+%H:%M:%S') Status: $STATUS"
    case "$STATUS" in
      Completed) echo "Job completed successfully."; return 0 ;;
      Failed|Stopped) echo "ERROR: Job $job_name $STATUS"; exit 1 ;;
      *) sleep 30 ;;
    esac
  done
}

# Helper: wait for SageMaker endpoint to reach InService
wait_for_endpoint() {
  local endpoint_name="$1"
  echo "Waiting for endpoint '$endpoint_name' to reach InService..."
  while true; do
    STATUS=$(aws sagemaker describe-endpoint \
      --endpoint-name "$endpoint_name" \
      --query EndpointStatus \
      --output text 2>/dev/null || echo "UNKNOWN")
    echo "  $(date -u '+%H:%M:%S') Status: $STATUS"
    case "$STATUS" in
      InService) echo "Endpoint is InService."; return 0 ;;
      Failed) echo "ERROR: Endpoint failed"; exit 1 ;;
      *) sleep 30 ;;
    esac
  done
}

echo "common.sh loaded — AWS_ACCOUNT_ID=$AWS_ACCOUNT_ID  REGION=$AWS_REGION  BUCKET=$S3_BUCKET"
