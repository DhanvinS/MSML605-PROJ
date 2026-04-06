#!/usr/bin/env bash
# 02_build.sh — zip repo, upload to S3, trigger CodeBuild, wait for completion
source "$(dirname "$0")/common.sh"

echo "========================================"
echo " Building Docker images via CodeBuild"
echo "========================================"

ZIP_PATH="/tmp/stock-prediction-source.zip"

# --- Zip source code (exclude large/unnecessary directories) -------------
echo "[1/3] Creating source zip..."
cd "$PROJECT_ROOT"
zip -r "$ZIP_PATH" . \
  --exclude ".git/*" \
  --exclude "data/*" \
  --exclude "models/*" \
  --exclude ".venv/*" \
  --exclude "__pycache__/*" \
  --exclude "*.pyc" \
  --exclude ".pytest_cache/*" \
  --exclude "output/*" \
  --exclude "*.egg-info/*" \
  > /dev/null
echo "  Source zip created: $ZIP_PATH ($(du -sh "$ZIP_PATH" | cut -f1))"

# --- Upload zip to S3 -----------------------------------------------------
echo "[2/3] Uploading source zip to S3..."
aws s3 cp "$ZIP_PATH" "s3://$S3_BUCKET/codebuild/source.zip"
echo "  Uploaded to s3://$S3_BUCKET/codebuild/source.zip"
rm -f "$ZIP_PATH"

# --- Start CodeBuild and wait --------------------------------------------
echo "[3/3] Starting CodeBuild build..."
BUILD_ID=$(aws codebuild start-build \
  --project-name "$CODEBUILD_PROJECT" \
  --query 'build.id' \
  --output text)
echo "  Build ID: $BUILD_ID"
echo "  Monitor: https://console.aws.amazon.com/codesuite/codebuild/projects/$CODEBUILD_PROJECT/build/$BUILD_ID/log"

echo "  Polling build status every 20s..."
while true; do
  STATUS=$(aws codebuild batch-get-builds \
    --ids "$BUILD_ID" \
    --query 'builds[0].buildStatus' \
    --output text)
  PHASE=$(aws codebuild batch-get-builds \
    --ids "$BUILD_ID" \
    --query 'builds[0].currentPhase' \
    --output text 2>/dev/null || echo "")
  echo "  $(date -u '+%H:%M:%S') Status: $STATUS  Phase: $PHASE"
  case "$STATUS" in
    SUCCEEDED)
      echo ""
      echo "Build succeeded! Images pushed to ECR:"
      echo "  $ECR_INGESTION"
      echo "  $ECR_TRAINING"
      echo "  $ECR_INFERENCE"
      break
      ;;
    FAILED|STOPPED|FAULT|TIMED_OUT)
      echo "ERROR: CodeBuild $STATUS. Check logs at:"
      echo "  https://console.aws.amazon.com/codesuite/codebuild/projects/$CODEBUILD_PROJECT/build/$BUILD_ID/log"
      exit 1
      ;;
    *)
      sleep 20
      ;;
  esac
done

echo ""
echo "========================================"
echo " Build complete. All 3 images in ECR."
echo "========================================"
