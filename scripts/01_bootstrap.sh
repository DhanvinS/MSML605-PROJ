#!/usr/bin/env bash
# 01_bootstrap.sh — create S3 bucket, ECR repos, IAM roles, CodeBuild project
source "$(dirname "$0")/common.sh"

echo "========================================"
echo " Bootstrapping AWS infrastructure"
echo "========================================"

# --- S3 bucket ------------------------------------------------------------
echo "[1/5] S3 bucket: $S3_BUCKET"
if aws s3api head-bucket --bucket "$S3_BUCKET" 2>/dev/null; then
  echo "  Bucket already exists."
else
  if [[ "$AWS_REGION" == "us-east-1" ]]; then
    aws s3api create-bucket --bucket "$S3_BUCKET" --region "$AWS_REGION"
  else
    aws s3api create-bucket --bucket "$S3_BUCKET" --region "$AWS_REGION" \
      --create-bucket-configuration LocationConstraint="$AWS_REGION"
  fi
  aws s3api put-bucket-versioning \
    --bucket "$S3_BUCKET" \
    --versioning-configuration Status=Enabled
  echo "  Bucket created with versioning enabled."
fi

# --- ECR repositories -----------------------------------------------------
echo "[2/5] ECR repositories..."
for REPO in stock-prediction/ingestion stock-prediction/training stock-prediction/inference; do
  if aws ecr describe-repositories --repository-names "$REPO" &>/dev/null; then
    echo "  Repo already exists: $REPO"
  else
    aws ecr create-repository \
      --repository-name "$REPO" \
      --image-scanning-configuration scanOnPush=true \
      --region "$AWS_REGION"
    echo "  Created: $REPO"
  fi
done

# --- Helper: create IAM role with trust policy ----------------------------
create_role_if_missing() {
  local role_name="$1"
  local trust_service="$2"  # sagemaker.amazonaws.com | lambda.amazonaws.com | codebuild.amazonaws.com

  if aws iam get-role --role-name "$role_name" &>/dev/null; then
    echo "  Role already exists: $role_name"
    return
  fi

  TRUST_DOC=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "$trust_service"},
    "Action": "sts:AssumeRole"
  }]
}
EOF
)
  aws iam create-role \
    --role-name "$role_name" \
    --assume-role-policy-document "$TRUST_DOC" \
    --description "Auto-created by 01_bootstrap.sh"
  echo "  Created role: $role_name"
}

# --- IAM: SageMaker execution role ----------------------------------------
echo "[3/5] IAM roles..."
create_role_if_missing "$SAGEMAKER_ROLE_NAME" "sagemaker.amazonaws.com"
for POLICY in \
  arn:aws:iam::aws:policy/AmazonSageMakerFullAccess \
  arn:aws:iam::aws:policy/AmazonS3FullAccess \
  arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly \
  arn:aws:iam::aws:policy/CloudWatchFullAccess; do
  aws iam attach-role-policy \
    --role-name "$SAGEMAKER_ROLE_NAME" \
    --policy-arn "$POLICY" 2>/dev/null || true
done
echo "  SageMaker role ready: $SAGEMAKER_ROLE_NAME"

# --- IAM: Lambda execution role -------------------------------------------
create_role_if_missing "$LAMBDA_ROLE_NAME" "lambda.amazonaws.com"
for POLICY in \
  arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole \
  arn:aws:iam::aws:policy/AmazonSageMakerFullAccess \
  arn:aws:iam::aws:policy/AmazonS3FullAccess \
  arn:aws:iam::aws:policy/AmazonSNSFullAccess \
  arn:aws:iam::aws:policy/CloudWatchFullAccess; do
  aws iam attach-role-policy \
    --role-name "$LAMBDA_ROLE_NAME" \
    --policy-arn "$POLICY" 2>/dev/null || true
done
echo "  Lambda role ready: $LAMBDA_ROLE_NAME"

# --- IAM: CodeBuild service role ------------------------------------------
create_role_if_missing "$CODEBUILD_ROLE_NAME" "codebuild.amazonaws.com"
for POLICY in \
  arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess \
  arn:aws:iam::aws:policy/CloudWatchLogsFullAccess \
  arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess; do
  aws iam attach-role-policy \
    --role-name "$CODEBUILD_ROLE_NAME" \
    --policy-arn "$POLICY" 2>/dev/null || true
done
echo "  CodeBuild role ready: $CODEBUILD_ROLE_NAME"

# Allow CodeBuild to read the source zip from S3
aws iam put-role-policy \
  --role-name "$CODEBUILD_ROLE_NAME" \
  --policy-name "S3SourceAccess" \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [{
      \"Effect\": \"Allow\",
      \"Action\": [\"s3:GetObject\", \"s3:GetObjectVersion\"],
      \"Resource\": \"arn:aws:s3:::${S3_BUCKET}/*\"
    }]
  }" 2>/dev/null || true

# Wait for roles to propagate
echo "  Waiting 10s for IAM role propagation..."
sleep 10

SAGEMAKER_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${SAGEMAKER_ROLE_NAME}"
CODEBUILD_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${CODEBUILD_ROLE_NAME}"

# --- CodeBuild project ----------------------------------------------------
echo "[5/5] CodeBuild project: $CODEBUILD_PROJECT"
if aws codebuild batch-get-projects --names "$CODEBUILD_PROJECT" \
  --query 'projects[0].name' --output text 2>/dev/null | grep -q "$CODEBUILD_PROJECT"; then
  echo "  CodeBuild project already exists."
else
  aws codebuild create-project \
    --name "$CODEBUILD_PROJECT" \
    --description "Builds stock prediction Docker images" \
    --source "{
      \"type\": \"S3\",
      \"location\": \"${S3_BUCKET}/codebuild/source.zip\",
      \"buildspec\": \"infra/buildspec.yml\"
    }" \
    --artifacts "{\"type\": \"NO_ARTIFACTS\"}" \
    --environment "{
      \"type\": \"LINUX_CONTAINER\",
      \"image\": \"aws/codebuild/standard:7.0\",
      \"computeType\": \"BUILD_GENERAL1_MEDIUM\",
      \"privilegedMode\": true,
      \"environmentVariables\": [
        {\"name\": \"AWS_ACCOUNT_ID\", \"value\": \"${AWS_ACCOUNT_ID}\", \"type\": \"PLAINTEXT\"},
        {\"name\": \"AWS_REGION\",     \"value\": \"${AWS_REGION}\",     \"type\": \"PLAINTEXT\"},
        {\"name\": \"S3_BUCKET\",      \"value\": \"${S3_BUCKET}\",      \"type\": \"PLAINTEXT\"}
      ]
    }" \
    --service-role "$CODEBUILD_ROLE_ARN"
  echo "  CodeBuild project created."
fi

echo ""
echo "========================================"
echo " Bootstrap complete."
echo " SageMaker role ARN: $SAGEMAKER_ROLE_ARN"
echo "========================================"
