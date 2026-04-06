#!/usr/bin/env bash
# 00_cleanup.sh — delete all existing AWS resources before a fresh deploy
source "$(dirname "$0")/common.sh"

echo "========================================"
echo " Cleaning up AWS resources"
echo "========================================"

# --- SageMaker endpoint --------------------------------------------------
echo "[1/8] SageMaker endpoint..."
aws sagemaker delete-endpoint \
  --endpoint-name "$ENDPOINT_NAME" 2>/dev/null \
  && echo "  Endpoint deleted. Waiting for deletion..." \
  || echo "  Endpoint not found, skipping."

# Wait for endpoint deletion
if aws sagemaker describe-endpoint --endpoint-name "$ENDPOINT_NAME" &>/dev/null; then
  while aws sagemaker describe-endpoint --endpoint-name "$ENDPOINT_NAME" &>/dev/null; do
    echo "  Waiting for endpoint deletion..."
    sleep 15
  done
  echo "  Endpoint deleted."
fi

# Delete all endpoint configs with our prefix
echo "  Deleting endpoint configs..."
aws sagemaker list-endpoint-configs --name-contains "stock-" \
  --query 'EndpointConfigs[*].EndpointConfigName' \
  --output text 2>/dev/null | tr '\t' '\n' | while read -r name; do
  [[ -z "$name" ]] && continue
  aws sagemaker delete-endpoint-config --endpoint-config-name "$name" 2>/dev/null \
    && echo "  Deleted config: $name" || true
done

# Delete all models with our prefix
echo "  Deleting SageMaker models..."
aws sagemaker list-models --name-contains "stock-" \
  --query 'Models[*].ModelName' \
  --output text 2>/dev/null | tr '\t' '\n' | while read -r name; do
  [[ -z "$name" ]] && continue
  aws sagemaker delete-model --model-name "$name" 2>/dev/null \
    && echo "  Deleted model: $name" || true
done

# --- Lambda ---------------------------------------------------------------
echo "[2/8] Lambda function..."
aws lambda delete-function \
  --function-name "$LAMBDA_FUNCTION" 2>/dev/null \
  && echo "  Lambda deleted." || echo "  Lambda not found, skipping."

# --- EventBridge rules ----------------------------------------------------
echo "[3/8] EventBridge rules..."
for RULE in stock-prediction-daily-drift stock-prediction-processing-complete stock-prediction-training-complete; do
  # Remove targets first
  TARGET_IDS=$(aws events list-targets-by-rule \
    --rule "$RULE" \
    --query 'Targets[*].Id' \
    --output text 2>/dev/null || echo "")
  if [[ -n "$TARGET_IDS" ]]; then
    aws events remove-targets \
      --rule "$RULE" \
      --ids $TARGET_IDS 2>/dev/null || true
  fi
  aws events delete-rule --name "$RULE" 2>/dev/null \
    && echo "  Deleted rule: $RULE" || echo "  Rule $RULE not found, skipping."
done

# --- CloudWatch alarms ----------------------------------------------------
echo "[4/8] CloudWatch alarms..."
aws cloudwatch delete-alarms \
  --alarm-names StockPredictorDriftAlarm 2>/dev/null \
  && echo "  Alarms deleted." || echo "  No alarms found, skipping."

# --- SNS topic ------------------------------------------------------------
echo "[5/8] SNS topic..."
TOPIC_ARN=$(aws sns list-topics \
  --query "Topics[?contains(TopicArn, '$SNS_TOPIC_NAME')].TopicArn" \
  --output text 2>/dev/null | head -1)
if [[ -n "$TOPIC_ARN" ]]; then
  aws sns delete-topic --topic-arn "$TOPIC_ARN" \
    && echo "  SNS topic deleted: $TOPIC_ARN"
else
  echo "  SNS topic not found, skipping."
fi

# --- IAM roles ------------------------------------------------------------
echo "[6/8] IAM roles..."
for ROLE in "$SAGEMAKER_ROLE_NAME" "$LAMBDA_ROLE_NAME" "$CODEBUILD_ROLE_NAME"; do
  # Detach managed policies
  aws iam list-attached-role-policies \
    --role-name "$ROLE" \
    --query 'AttachedPolicies[*].PolicyArn' \
    --output text 2>/dev/null | tr '\t' '\n' | while read -r arn; do
    [[ -z "$arn" ]] && continue
    aws iam detach-role-policy --role-name "$ROLE" --policy-arn "$arn" 2>/dev/null || true
  done
  # Delete inline policies
  aws iam list-role-policies \
    --role-name "$ROLE" \
    --query 'PolicyNames[*]' \
    --output text 2>/dev/null | tr '\t' '\n' | while read -r pname; do
    [[ -z "$pname" ]] && continue
    aws iam delete-role-policy --role-name "$ROLE" --policy-name "$pname" 2>/dev/null || true
  done
  aws iam delete-role --role-name "$ROLE" 2>/dev/null \
    && echo "  Deleted role: $ROLE" || echo "  Role $ROLE not found, skipping."
done

# --- ECR repos ------------------------------------------------------------
echo "[7/8] ECR repositories..."
for REPO in stock-prediction/ingestion stock-prediction/training stock-prediction/inference; do
  aws ecr delete-repository \
    --repository-name "$REPO" \
    --force 2>/dev/null \
    && echo "  Deleted ECR repo: $REPO" || echo "  Repo $REPO not found, skipping."
done

# --- CodeBuild project ----------------------------------------------------
echo "  Deleting CodeBuild project..."
aws codebuild delete-project \
  --name "$CODEBUILD_PROJECT" 2>/dev/null \
  && echo "  CodeBuild project deleted." || echo "  Project not found, skipping."

# --- S3 bucket contents ---------------------------------------------------
echo "[8/8] S3 bucket contents..."
BUCKET_EXISTS=$(aws s3api head-bucket --bucket "$S3_BUCKET" 2>&1 || echo "MISSING")
if [[ "$BUCKET_EXISTS" != *"MISSING"* ]]; then
  echo ""
  read -r -p "  WARNING: Delete ALL contents of s3://$S3_BUCKET? [y/N] " CONFIRM
  if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    aws s3 rm "s3://$S3_BUCKET" --recursive \
      && echo "  Bucket contents deleted."
    # Delete versioned objects too
    aws s3api delete-objects \
      --bucket "$S3_BUCKET" \
      --delete "$(aws s3api list-object-versions \
        --bucket "$S3_BUCKET" \
        --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}' \
        --output json 2>/dev/null)" 2>/dev/null || true
  else
    echo "  Skipping S3 wipe."
  fi
else
  echo "  Bucket $S3_BUCKET not found, skipping."
fi

echo ""
echo "========================================"
echo " Cleanup complete."
echo "========================================"
