#!/usr/bin/env bash
# EC2 user-data bootstrap script.
# Run once on first launch via EC2 User Data, or manually: sudo bash ec2_setup.sh

set -euo pipefail

LOG=/var/log/ec2_setup.log
exec > >(tee -a "$LOG") 2>&1
echo "=== EC2 setup started at $(date) ==="

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
apt-get update -y
apt-get install -y \
    docker.io \
    docker-compose-plugin \
    awscli \
    git \
    curl \
    jq \
    unzip

# Start Docker
systemctl enable docker
systemctl start docker
usermod -aG docker ubuntu

# ---------------------------------------------------------------------------
# 2. AWS environment variables (set in EC2 launch template or SSM Parameter Store)
# ---------------------------------------------------------------------------
# These are expected to be present via IAM instance role or environment:
# - AWS_REGION
# - ECR_URI
# - S3_BUCKET_NAME
# - SNS_TOPIC_ARN  (optional)

ECR_URI="${ECR_URI:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"

if [ -z "$ECR_URI" ]; then
    echo "WARNING: ECR_URI not set. Skipping ECR login."
else
    echo "Logging in to ECR..."
    aws ecr get-login-password --region "$AWS_REGION" \
        | docker login --username AWS --password-stdin "$ECR_URI"
fi

# ---------------------------------------------------------------------------
# 3. Clone repo (or copy from S3)
# ---------------------------------------------------------------------------
APP_DIR=/home/ubuntu/stock-prediction-mlops

if [ ! -d "$APP_DIR" ]; then
    REPO_URL="${REPO_URL:-}"
    if [ -n "$REPO_URL" ]; then
        git clone "$REPO_URL" "$APP_DIR"
    else
        # Fallback: download artefacts from S3
        mkdir -p "$APP_DIR"
        aws s3 sync "s3://${S3_BUCKET_NAME}/deploy/" "$APP_DIR/"
    fi
fi

chown -R ubuntu:ubuntu "$APP_DIR"

# ---------------------------------------------------------------------------
# 4. Create .env file for docker-compose
# ---------------------------------------------------------------------------
cat > "$APP_DIR/.env" <<EOF
AWS_REGION=${AWS_REGION}
ECR_URI=${ECR_URI}
S3_BUCKET_NAME=${S3_BUCKET_NAME:-}
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
EOF
# Note: Prefer IAM instance role over explicit keys. Keys intentionally left blank.

# ---------------------------------------------------------------------------
# 5. Pull images and start services
# ---------------------------------------------------------------------------
cd "$APP_DIR"
if [ -n "$ECR_URI" ]; then
    docker pull "${ECR_URI}/stock-prediction/inference:latest" || true
    docker pull "${ECR_URI}/stock-prediction/dashboard:latest" || true
fi

docker compose -f infra/docker-compose.yml up -d

echo "=== EC2 setup complete at $(date) ==="
echo "Services running:"
docker compose -f infra/docker-compose.yml ps
