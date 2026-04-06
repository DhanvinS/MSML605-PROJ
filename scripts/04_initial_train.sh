#!/usr/bin/env bash
# 04_initial_train.sh — run first SageMaker Training Job
source "$(dirname "$0")/common.sh"

echo "========================================"
echo " Initial Model Training"
echo "========================================"

SAGEMAKER_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${SAGEMAKER_ROLE_NAME}"
JOB_NAME="stock-retrain-initial-$(date -u +%Y%m%d-%H%M%S)"
S3_INPUT="s3://${S3_BUCKET}/processed"
S3_OUTPUT="s3://${S3_BUCKET}/models"

echo "Job name:   $JOB_NAME"
echo "Input:      $S3_INPUT"
echo "Output:     $S3_OUTPUT"
echo ""

aws sagemaker create-training-job \
  --training-job-name "$JOB_NAME" \
  --algorithm-specification "{
    \"TrainingImage\": \"${ECR_TRAINING}\",
    \"TrainingInputMode\": \"File\"
  }" \
  --role-arn "$SAGEMAKER_ROLE_ARN" \
  --input-data-config "[{
    \"ChannelName\": \"train\",
    \"DataSource\": {
      \"S3DataSource\": {
        \"S3DataType\": \"S3Prefix\",
        \"S3Uri\": \"${S3_INPUT}\",
        \"S3DataDistributionType\": \"FullyReplicated\"
      }
    },
    \"ContentType\": \"application/x-parquet\",
    \"InputMode\": \"File\"
  }]" \
  --output-data-config "{
    \"S3OutputPath\": \"${S3_OUTPUT}\"
  }" \
  --resource-config '{
    "InstanceType": "ml.m5.xlarge",
    "InstanceCount": 1,
    "VolumeSizeInGB": 30
  }' \
  --stopping-condition '{"MaxRuntimeInSeconds": 3600}' \
  --hyper-parameters '{
    "config": "configs/model_config.yaml"
  }' \
  --environment "{
    \"S3_BUCKET\": \"${S3_BUCKET}\",
    \"AWS_REGION\": \"${AWS_REGION}\"
  }" \
  --tags '[
    {"Key": "Project", "Value": "StockPrediction"},
    {"Key": "Stage",   "Value": "InitialTraining"}
  ]'

echo "Training job submitted."
wait_for_sagemaker_job training "$JOB_NAME"

# Save job name for next step
echo "$JOB_NAME" > /tmp/last_training_job.txt
echo "Training job name saved to /tmp/last_training_job.txt"

# Print artifact location
ARTIFACT_URI=$(aws sagemaker describe-training-job \
  --training-job-name "$JOB_NAME" \
  --query 'ModelArtifacts.S3ModelArtifacts' \
  --output text)
echo ""
echo "Model artifact: $ARTIFACT_URI"
echo "Baseline stats: s3://$S3_BUCKET/baseline/baseline_stats.json"
echo ""
echo "========================================"
echo " Training complete. Proceed to 05_deploy.sh"
echo "========================================"
