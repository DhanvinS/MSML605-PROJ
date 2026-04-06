#!/usr/bin/env bash
# 05_deploy.sh — create SageMaker model, endpoint config (with data capture), endpoint
source "$(dirname "$0")/common.sh"

echo "========================================"
echo " Deploying SageMaker Endpoint"
echo "========================================"

SAGEMAKER_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${SAGEMAKER_ROLE_NAME}"
TS=$(date -u +%Y%m%d-%H%M%S)
MODEL_NAME="stock-model-${TS}"
CONFIG_NAME="stock-config-${TS}"
CAPTURE_S3="s3://${S3_BUCKET}/captures"

# --- Get model artifact from last training job ---------------------------
if [[ -f /tmp/last_training_job.txt ]]; then
  TRAINING_JOB=$(cat /tmp/last_training_job.txt)
else
  # Find the most recent completed training job
  TRAINING_JOB=$(aws sagemaker list-training-jobs \
    --name-contains "stock-retrain" \
    --status-equals Completed \
    --sort-by CreationTime \
    --sort-order Descending \
    --max-results 1 \
    --query 'TrainingJobSummaries[0].TrainingJobName' \
    --output text)
fi

if [[ -z "$TRAINING_JOB" || "$TRAINING_JOB" == "None" ]]; then
  echo "ERROR: No completed training job found. Run 04_initial_train.sh first."
  exit 1
fi

echo "Using training job: $TRAINING_JOB"

ARTIFACT_URI=$(aws sagemaker describe-training-job \
  --training-job-name "$TRAINING_JOB" \
  --query 'ModelArtifacts.S3ModelArtifacts' \
  --output text)
echo "Model artifact:    $ARTIFACT_URI"
echo "Capture output:    $CAPTURE_S3"
echo ""

# --- Create SageMaker Model -----------------------------------------------
echo "[1/4] Creating SageMaker model: $MODEL_NAME"
aws sagemaker create-model \
  --model-name "$MODEL_NAME" \
  --primary-container "{
    \"Image\": \"${ECR_INFERENCE}\",
    \"ModelDataUrl\": \"${ARTIFACT_URI}\",
    \"Environment\": {
      \"SAGEMAKER_PROGRAM\": \"serve\",
      \"MODEL_DIR\": \"/opt/ml/model\",
      \"AWS_REGION\": \"${AWS_REGION}\"
    }
  }" \
  --execution-role-arn "$SAGEMAKER_ROLE_ARN"
echo "  Model created."

# --- Create Endpoint Config with Data Capture ----------------------------
echo "[2/4] Creating endpoint config: $CONFIG_NAME"
aws sagemaker create-endpoint-config \
  --endpoint-config-name "$CONFIG_NAME" \
  --production-variants "[{
    \"VariantName\": \"AllTraffic\",
    \"ModelName\": \"${MODEL_NAME}\",
    \"InstanceType\": \"ml.t2.medium\",
    \"InitialInstanceCount\": 1,
    \"InitialVariantWeight\": 1
  }]" \
  --data-capture-config "{
    \"EnableCapture\": true,
    \"InitialSamplingPercentage\": 100,
    \"DestinationS3Uri\": \"${CAPTURE_S3}\",
    \"CaptureOptions\": [{\"CaptureMode\": \"Input\"}, {\"CaptureMode\": \"Output\"}],
    \"CaptureContentTypeHeader\": {
      \"JsonContentTypes\": [\"application/json\"]
    }
  }"
echo "  Endpoint config created (Data Capture → $CAPTURE_S3)"

# --- Create or update endpoint -------------------------------------------
ENDPOINT_EXISTS=$(aws sagemaker describe-endpoint \
  --endpoint-name "$ENDPOINT_NAME" \
  --query 'EndpointStatus' \
  --output text 2>/dev/null || echo "MISSING")

if [[ "$ENDPOINT_EXISTS" == "MISSING" ]]; then
  echo "[3/4] Creating endpoint: $ENDPOINT_NAME"
  aws sagemaker create-endpoint \
    --endpoint-name "$ENDPOINT_NAME" \
    --endpoint-config-name "$CONFIG_NAME"
else
  echo "[3/4] Updating endpoint: $ENDPOINT_NAME (current status: $ENDPOINT_EXISTS)"
  aws sagemaker update-endpoint \
    --endpoint-name "$ENDPOINT_NAME" \
    --endpoint-config-name "$CONFIG_NAME"
fi

echo "[4/4] Waiting for endpoint to reach InService..."
wait_for_endpoint "$ENDPOINT_NAME"

echo ""
echo "Endpoint URL (for SageMaker Runtime invocation):"
echo "  https://runtime.sagemaker.${AWS_REGION}.amazonaws.com/endpoints/${ENDPOINT_NAME}/invocations"
echo ""
echo "Test command:"
echo "  aws sagemaker-runtime invoke-endpoint \\"
echo "    --endpoint-name $ENDPOINT_NAME \\"
echo "    --content-type application/json \\"
echo "    --body '{\"ticker\":\"AAPL\",\"features\":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3]}' \\"
echo "    /tmp/response.json && cat /tmp/response.json"
echo ""
echo "========================================"
echo " Deployment complete."
echo "========================================"
