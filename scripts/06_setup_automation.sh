#!/usr/bin/env bash
# 06_setup_automation.sh — create Lambda, EventBridge rules, CloudWatch alarm, SNS topic
source "$(dirname "$0")/common.sh"

echo "========================================"
echo " Setting up Automation"
echo "========================================"

SAGEMAKER_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${SAGEMAKER_ROLE_NAME}"
LAMBDA_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${LAMBDA_ROLE_NAME}"
ALERT_EMAIL="${ALERT_EMAIL:-}"

# --- SNS topic ------------------------------------------------------------
echo "[1/5] SNS topic..."
TOPIC_ARN=$(aws sns create-topic \
  --name "$SNS_TOPIC_NAME" \
  --query TopicArn \
  --output text)
echo "  Topic ARN: $TOPIC_ARN"

if [[ -n "$ALERT_EMAIL" ]]; then
  aws sns subscribe \
    --topic-arn "$TOPIC_ARN" \
    --protocol email \
    --notification-endpoint "$ALERT_EMAIL" 2>/dev/null || true
  echo "  Subscription sent to $ALERT_EMAIL (check your inbox to confirm)"
fi

# --- Package Lambda -------------------------------------------------------
echo "[2/5] Packaging Lambda function..."
LAMBDA_ZIP="/tmp/stock-prediction-lambda.zip"
LAMBDA_BUILD_DIR="/tmp/lambda-build"
rm -rf "$LAMBDA_BUILD_DIR"
mkdir -p "$LAMBDA_BUILD_DIR/src/retraining" \
         "$LAMBDA_BUILD_DIR/src/monitoring" \
         "$LAMBDA_BUILD_DIR/src/training" \
         "$LAMBDA_BUILD_DIR/src"

# Copy Lambda source files
cp "$PROJECT_ROOT/src/retraining/retrain_trigger.py" "$LAMBDA_BUILD_DIR/src/retraining/"
cp "$PROJECT_ROOT/src/monitoring/drift_detector.py"   "$LAMBDA_BUILD_DIR/src/monitoring/"
cp "$PROJECT_ROOT/src/monitoring/cloudwatch_logger.py" "$LAMBDA_BUILD_DIR/src/monitoring/"
cp "$PROJECT_ROOT/src/training/baseline_capture.py"   "$LAMBDA_BUILD_DIR/src/training/"

# __init__.py files
touch "$LAMBDA_BUILD_DIR/src/__init__.py"
touch "$LAMBDA_BUILD_DIR/src/retraining/__init__.py"
touch "$LAMBDA_BUILD_DIR/src/monitoring/__init__.py"
touch "$LAMBDA_BUILD_DIR/src/training/__init__.py"

cd "$LAMBDA_BUILD_DIR"
zip -r "$LAMBDA_ZIP" . > /dev/null
cd "$PROJECT_ROOT"
echo "  Lambda zip: $LAMBDA_ZIP ($(du -sh "$LAMBDA_ZIP" | cut -f1))"

# --- Create or update Lambda function -------------------------------------
echo "[3/5] Lambda function: $LAMBDA_FUNCTION"

LAMBDA_EXISTS=$(aws lambda get-function \
  --function-name "$LAMBDA_FUNCTION" \
  --query 'Configuration.FunctionName' \
  --output text 2>/dev/null || echo "MISSING")

if [[ "$LAMBDA_EXISTS" == "MISSING" ]]; then
  aws lambda create-function \
    --function-name "$LAMBDA_FUNCTION" \
    --runtime python3.11 \
    --role "$LAMBDA_ROLE_ARN" \
    --handler "src.retraining.retrain_trigger.lambda_handler" \
    --zip-file "fileb://$LAMBDA_ZIP" \
    --timeout 300 \
    --memory-size 512 \
    --description "Orchestrates stock prediction retraining pipeline"
  echo "  Lambda created."
else
  aws lambda update-function-code \
    --function-name "$LAMBDA_FUNCTION" \
    --zip-file "fileb://$LAMBDA_ZIP" > /dev/null
  echo "  Lambda code updated."
  # Wait for code update to finish before updating configuration
  aws lambda wait function-updated --function-name "$LAMBDA_FUNCTION"
fi

# Set environment variables
aws lambda update-function-configuration \
  --function-name "$LAMBDA_FUNCTION" \
  --environment "{
    \"Variables\": {
      \"S3_BUCKET\":               \"${S3_BUCKET}\",
      \"ECR_INFERENCE_IMAGE_URI\": \"${ECR_INFERENCE}\",
      \"ECR_TRAINING_IMAGE_URI\":  \"${ECR_TRAINING}\",
      \"ECR_INGESTION_IMAGE_URI\": \"${ECR_INGESTION}\",
      \"SAGEMAKER_ROLE_ARN\":      \"${SAGEMAKER_ROLE_ARN}\",
      \"SAGEMAKER_ENDPOINT_NAME\": \"${ENDPOINT_NAME}\",
      \"TICKERS\":                 \"${TICKERS}\",
      \"LOOKBACK_DAYS\":           \"${LOOKBACK_DAYS}\",
      \"SNS_TOPIC_ARN\":           \"${TOPIC_ARN}\",
      \"RETRAIN_PSI_THRESHOLD\":   \"0.2\"
    }
  }" > /dev/null
echo "  Lambda environment configured."

LAMBDA_ARN=$(aws lambda get-function \
  --function-name "$LAMBDA_FUNCTION" \
  --query 'Configuration.FunctionArn' \
  --output text)

# --- EventBridge rules ----------------------------------------------------
echo "[4/5] EventBridge rules..."

create_eb_rule_and_target() {
  local rule_name="$1"
  local description="$2"
  local event_pattern="$3"
  # No Input override — full SageMaker event is passed to Lambda as-is
  # Lambda reads detail-type + detail.TrainingJobName / detail.ProcessingJobName

  RULE_ARN=$(aws events put-rule \
    --name "$rule_name" \
    --description "$description" \
    --event-pattern "$event_pattern" \
    --state ENABLED \
    --query RuleArn \
    --output text)
  echo "  Rule created: $rule_name"

  aws lambda add-permission \
    --function-name "$LAMBDA_FUNCTION" \
    --statement-id "eb-${rule_name}" \
    --action "lambda:InvokeFunction" \
    --principal "events.amazonaws.com" \
    --source-arn "$RULE_ARN" 2>/dev/null || true

  aws events put-targets \
    --rule "$rule_name" \
    --targets "[{\"Id\": \"target-1\", \"Arn\": \"$LAMBDA_ARN\"}]"
}

# Rule 1: Daily drift check at 6 AM UTC
DAILY_RULE_ARN=$(aws events put-rule \
  --name "stock-prediction-daily-drift" \
  --description "Daily drift check trigger" \
  --schedule-expression "cron(0 6 * * ? *)" \
  --state ENABLED \
  --query RuleArn \
  --output text)
aws lambda add-permission \
  --function-name "$LAMBDA_FUNCTION" \
  --statement-id "eb-stock-prediction-daily-drift" \
  --action "lambda:InvokeFunction" \
  --principal "events.amazonaws.com" \
  --source-arn "$DAILY_RULE_ARN" 2>/dev/null || true
aws events put-targets \
  --rule "stock-prediction-daily-drift" \
  --targets "[{\"Id\": \"target-1\", \"Arn\": \"$LAMBDA_ARN\", \"Input\": \"{\\\"source\\\": \\\"scheduled\\\"}\"}]"
echo "  Rule: stock-prediction-daily-drift (cron 6 AM UTC)"

# Rule 2: SageMaker Processing Job completed (full event passed to Lambda)
create_eb_rule_and_target \
  "stock-prediction-processing-complete" \
  "SageMaker Processing Job completed" \
  '{
    "source": ["aws.sagemaker"],
    "detail-type": ["SageMaker Processing Job State Change"],
    "detail": {"ProcessingJobStatus": ["Completed"]}
  }'

# Rule 3: SageMaker Training Job completed (full event passed to Lambda)
create_eb_rule_and_target \
  "stock-prediction-training-complete" \
  "SageMaker Training Job completed" \
  '{
    "source": ["aws.sagemaker"],
    "detail-type": ["SageMaker Training Job State Change"],
    "detail": {"TrainingJobStatus": ["Completed"]}
  }'

# --- CloudWatch alarm (backup trigger) ------------------------------------
echo "[5/5] CloudWatch alarm..."
aws cloudwatch put-metric-alarm \
  --alarm-name "StockPredictorDriftAlarm" \
  --alarm-description "Max PSI > 0.2 — model drift detected" \
  --namespace "StockPredictor/DriftMetrics" \
  --metric-name "Max_PSI" \
  --statistic Maximum \
  --period 86400 \
  --evaluation-periods 1 \
  --threshold 0.2 \
  --comparison-operator GreaterThanThreshold \
  --treat-missing-data notBreaching \
  --alarm-actions "$TOPIC_ARN" \
  --ok-actions "$TOPIC_ARN"
echo "  CloudWatch alarm created: StockPredictorDriftAlarm"

rm -f "$LAMBDA_ZIP"
rm -rf "$LAMBDA_BUILD_DIR"

echo ""
echo "========================================"
echo " Automation setup complete."
echo ""
echo " Drift check runs daily at 6 AM UTC."
echo " Retraining triggers automatically when max_PSI > 0.2."
echo ""
echo " Manual trigger:"
echo "   aws lambda invoke \\"
echo "     --function-name $LAMBDA_FUNCTION \\"
echo "     --payload '{\"source\":\"scheduled\"}' \\"
echo "     /tmp/out.json && cat /tmp/out.json"
echo "========================================"
