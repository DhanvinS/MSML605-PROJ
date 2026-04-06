# Stock Prediction MLOps — AWS Deployment Guide

Containerized real-time stock prediction system with drift detection and automated retraining.
**MSML605 — University of Maryland**

---

## System Architecture

```
DefeatBeta API (defeatbeta-api)
       |
  fetch_yahoo.py  ──────────────────────────────────────────────┐
       |                                                         |
  feature_pipeline.py  (RSI, MACD, Bollinger Bands, etc.)       |
       |                                                         |
  train_xgboost.py  (walk-forward CV + Optuna tuning)           |
       |                                                         ▼
  models/latest/model.xgb                              S3 Bucket
       |                                                 ├── raw/
  FastAPI inference-api  (port 8000)                    ├── processed/
       |                                                 ├── models/
       ├── /predict                                      ├── baseline/
       ├── /inference-log  (ring buffer of 500)          └── monitoring/
       └── /health
       |
  monitoring-agent (hourly KS + PSI drift check)
       |
  CloudWatch Metrics ──► Alarm (PSI > 0.2) ──► SNS Topic
                                                    |
                                               Lambda Function
                                                    |
                                           SageMaker Training Job
                                                    |
                                           SageMaker Endpoint
                                                    |
                                          Streamlit Dashboard (port 8501)
```

---

## AWS Services Overview

| Service | Purpose | Estimated Cost |
|---|---|---|
| **EC2** (t3.small) | Runs all Docker containers (API + dashboard + monitoring agent) | ~$15/mo |
| **S3** | Stores raw data, processed features, trained models, drift reports | Free tier: 5 GB |
| **ECR** | Hosts Docker images for inference, training, dashboard | Free tier: 500 MB |
| **SageMaker** | Managed training jobs + real-time inference endpoint | ~$0.05/hr on ml.t3.medium |
| **Lambda** | Triggered by SNS alarm to launch SageMaker retraining job | Free tier: 1M requests |
| **CloudWatch** | Receives drift metrics, fires alarms when PSI exceeds threshold | Free tier: 10 metrics |
| **SNS** | Bridges CloudWatch alarm to Lambda + email notifications | Free tier: 1M publishes |
| **IAM** | Roles and permission policies for all services | Free |

---

## Phase 0 — Prerequisites

### 0.1 Install required tools locally

```bash
# macOS
brew install awscli git python@3.11

# Verify
aws --version        # must be 2.x
python3.11 --version
docker --version     # Desktop 24+
```

### 0.2 Create an AWS account and IAM admin user

1. Go to https://aws.amazon.com and create an account.
2. In the AWS Console → **IAM** → **Users** → **Create user**.
3. Name it `mlops-admin`, enable **Programmatic access**.
4. Attach policy: `AdministratorAccess` (you can tighten this later).
5. Download the **CSV** with `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. Keep this file safe.

### 0.3 Configure AWS CLI

```bash
aws configure
# AWS Access Key ID:     <from the CSV>
# AWS Secret Access Key: <from the CSV>
# Default region name:   us-east-1
# Default output format: json
```

Confirm it works:

```bash
aws sts get-caller-identity
# Should return your AccountId and UserId
```

### 0.4 Note your AWS Account ID

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo $AWS_ACCOUNT_ID
```

Save this — you will substitute `YOUR_ACCOUNT` with this value throughout this guide.

---

## Phase 1 — Local Development & Model Training

### 1.1 Clone and set up the environment

```bash
git clone <your-repo-url>
cd MSML605-PROJ

python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements-dev.txt
```

### 1.2 Create your local `.env` file

```bash
cat > .env <<EOF
# AWS
AWS_ACCOUNT_ID=YOUR_ACCOUNT
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Filled in later
S3_BUCKET_NAME=
ECR_URI=
SNS_TOPIC_ARN=
SAGEMAKER_ROLE_ARN=

# Data source
# defeatbeta-api is used by src/ingestion/fetch_yahoo.py (no API key required)
EOF
```

### 1.3 Fetch training data

```bash
python -c "
from src.ingestion.fetch_yahoo import fetch_ohlcv
df = fetch_ohlcv('AAPL', '2019-01-01', '2024-01-01')
df.to_parquet('data/raw/AAPL_daily.parquet')
print('Downloaded', len(df), 'rows')
"
```

Notes:
- The ingestion module now uses the `defeatbeta-api` Python package under the hood.
- This pipeline currently expects daily bars (`interval='1d'`).

### 1.4 Build the feature matrix

```bash
python -c "
from src.features.feature_pipeline import build_feature_matrix, split_and_scale
df = build_feature_matrix('AAPL', '2019-01-01', '2024-01-01')
train, val, test, scaler = split_and_scale(df, scaler_save_path='models/latest/scaler.pkl')
train.to_parquet('data/processed/AAPL_train.parquet')
print('Train:', len(train), '| Val:', len(val), '| Test:', len(test))
"
```

### 1.5 Capture baseline statistics (needed for drift detection)

```bash
python -c "
import pandas as pd
from src.training.baseline_capture import capture_baseline_stats
train = pd.read_parquet('data/processed/AAPL_train.parquet')
X_train = train.drop(columns=['target'])
capture_baseline_stats(X_train, output_path='data/baseline/baseline_stats.json')
print('Baseline stats saved')
"
```

### 1.6 Train the XGBoost model

```bash
python src/training/train_xgboost.py \
    --config configs/model_config.yaml \
    --ticker AAPL
# Output: models/latest/model.xgb + output/metrics.json
```

### 1.7 (Optional) Hyperparameter tuning with Optuna

```bash
python -c "
import pandas as pd, json
from src.training.hyperparameter_tuning import run_tuning

df = pd.read_parquet('data/processed/AAPL_train.parquet')
best_params = run_tuning(df, n_trials=50)
with open('configs/best_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)
print('Best params:', best_params)
"
```

### 1.8 Verify locally with Docker Compose (optional but recommended)

```bash
docker compose -f infra/docker-compose.yml up --build -d
curl http://localhost:8000/health
# Expected: {"status":"ok","model_loaded":true}
# Open dashboard: http://localhost:8501
docker compose -f infra/docker-compose.yml down
```

---

## Phase 2 — AWS Infrastructure Setup

All commands below use the AWS CLI. Run them in order.

### 2.1 Create the S3 bucket

```bash
# Choose a globally unique bucket name
export S3_BUCKET_NAME=stock-prediction-mlops-$(echo $AWS_ACCOUNT_ID | tail -c 6)
echo "Bucket name: $S3_BUCKET_NAME"

aws s3 mb s3://$S3_BUCKET_NAME --region us-east-1

# Create folder structure
for prefix in raw processed models baseline monitoring deploy; do
    aws s3api put-object --bucket $S3_BUCKET_NAME --key $prefix/
done

echo "S3 bucket ready: s3://$S3_BUCKET_NAME"
```

Update `.env`: set `S3_BUCKET_NAME=$S3_BUCKET_NAME`

### 2.2 Create ECR repositories

```bash
export ECR_URI=$AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
echo "ECR URI: $ECR_URI"

for repo in inference training dashboard; do
    aws ecr create-repository \
        --repository-name stock-prediction/$repo \
        --region us-east-1
done
```

Update `.env`: set `ECR_URI=$ECR_URI`

### 2.3 Create IAM roles

#### EC2 instance role (allows EC2 to pull from ECR, read/write S3, push to CloudWatch)

```bash
# Create the trust policy document
cat > /tmp/ec2-trust.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ec2.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF

aws iam create-role \
    --role-name stock-prediction-ec2-role \
    --assume-role-policy-document file:///tmp/ec2-trust.json

for policy in \
    arn:aws:iam::aws:policy/AmazonS3FullAccess \
    arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly \
    arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy \
    arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore; do
    aws iam attach-role-policy \
        --role-name stock-prediction-ec2-role \
        --policy-arn $policy
done

# Create instance profile and attach the role
aws iam create-instance-profile \
    --instance-profile-name stock-prediction-ec2-profile
aws iam add-role-to-instance-profile \
    --instance-profile-name stock-prediction-ec2-profile \
    --role-name stock-prediction-ec2-role
```

#### SageMaker execution role

```bash
cat > /tmp/sm-trust.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "sagemaker.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF

aws iam create-role \
    --role-name stock-prediction-sagemaker-role \
    --assume-role-policy-document file:///tmp/sm-trust.json

for policy in \
    arn:aws:iam::aws:policy/AmazonSageMakerFullAccess \
    arn:aws:iam::aws:policy/AmazonS3FullAccess \
    arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess; do
    aws iam attach-role-policy \
        --role-name stock-prediction-sagemaker-role \
        --policy-arn $policy
done

export SAGEMAKER_ROLE_ARN=$(aws iam get-role \
    --role-name stock-prediction-sagemaker-role \
    --query Role.Arn --output text)
echo "SageMaker role: $SAGEMAKER_ROLE_ARN"
```

Update `.env`: set `SAGEMAKER_ROLE_ARN=$SAGEMAKER_ROLE_ARN`

#### Lambda execution role

```bash
cat > /tmp/lambda-trust.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "lambda.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF

aws iam create-role \
    --role-name stock-prediction-lambda-role \
    --assume-role-policy-document file:///tmp/lambda-trust.json

for policy in \
    arn:aws:iam::aws:policy/AmazonSageMakerFullAccess \
    arn:aws:iam::aws:policy/AmazonS3FullAccess \
    arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole; do
    aws iam attach-role-policy \
        --role-name stock-prediction-lambda-role \
        --policy-arn $policy
done

export LAMBDA_ROLE_ARN=$(aws iam get-role \
    --role-name stock-prediction-lambda-role \
    --query Role.Arn --output text)
echo "Lambda role: $LAMBDA_ROLE_ARN"
```

### 2.4 Create SNS topic and subscribe your email

```bash
export SNS_TOPIC_ARN=$(aws sns create-topic \
    --name stock-prediction-alerts \
    --region us-east-1 \
    --query TopicArn --output text)
echo "SNS topic: $SNS_TOPIC_ARN"

# Subscribe your email (you will receive a confirmation email — click the link)
aws sns subscribe \
    --topic-arn $SNS_TOPIC_ARN \
    --protocol email \
    --notification-endpoint your@email.com
```

Update `.env`: set `SNS_TOPIC_ARN=$SNS_TOPIC_ARN`

### 2.5 Upload baseline data and initial model to S3

```bash
# Upload training data
aws s3 sync data/processed/ s3://$S3_BUCKET_NAME/processed/

# Upload baseline stats
aws s3 cp data/baseline/baseline_stats.json \
    s3://$S3_BUCKET_NAME/baseline/baseline_stats.json

# Upload the trained model
aws s3 sync models/latest/ s3://$S3_BUCKET_NAME/models/latest/

echo "S3 uploads complete"
```

---

## Phase 3 — Build and Push Docker Images

### 3.1 Log in to ECR

```bash
aws ecr get-login-password --region us-east-1 \
    | docker login --username AWS --password-stdin $ECR_URI
```

### 3.2 Build and push all three images

```bash
# Inference API image
docker build -f infra/docker/Dockerfile.inference \
    -t $ECR_URI/stock-prediction/inference:latest .
docker push $ECR_URI/stock-prediction/inference:latest

# Dashboard image
docker build -f infra/docker/Dockerfile.dashboard \
    -t $ECR_URI/stock-prediction/dashboard:latest .
docker push $ECR_URI/stock-prediction/dashboard:latest

# Training image (used by SageMaker jobs)
docker build -f infra/docker/Dockerfile.training \
    -t $ECR_URI/stock-prediction/training:latest .
docker push $ECR_URI/stock-prediction/training:latest

echo "All images pushed to ECR"
```

---

## Phase 4 — Deploy to EC2

### 4.1 Create a security group

```bash
export SG_ID=$(aws ec2 create-security-group \
    --group-name stock-prediction-sg \
    --description "Stock prediction app" \
    --query GroupId --output text)

# Allow SSH from your IP only
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp --port 22 --cidr $MY_IP/32

# Allow API and dashboard traffic
aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp --port 8000 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp --port 8501 --cidr 0.0.0.0/0

echo "Security group: $SG_ID"
```

### 4.2 Create an SSH key pair

```bash
aws ec2 create-key-pair \
    --key-name stock-prediction-key \
    --query KeyMaterial \
    --output text > ~/.ssh/stock-prediction-key.pem

chmod 400 ~/.ssh/stock-prediction-key.pem
echo "Key saved to ~/.ssh/stock-prediction-key.pem"
```

### 4.3 Launch the EC2 instance

```bash
# Get the latest Ubuntu 22.04 AMI ID for us-east-1
export AMI_ID=$(aws ec2 describe-images \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
              "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text)

echo "Using AMI: $AMI_ID"

export INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type t3.small \
    --key-name stock-prediction-key \
    --security-group-ids $SG_ID \
    --iam-instance-profile Name=stock-prediction-ec2-profile \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":20}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=stock-prediction-app}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance launched: $INSTANCE_ID"

# Wait for it to be running
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

export EC2_HOST=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "EC2 public IP: $EC2_HOST"
```

### 4.4 Bootstrap the instance via SSH

```bash
# Wait a minute for SSH daemon to start, then connect
ssh -i ~/.ssh/stock-prediction-key.pem ubuntu@$EC2_HOST

# Inside the EC2 instance, run:
export ECR_URI=YOUR_ECR_URI          # e.g. 123456789.dkr.ecr.us-east-1.amazonaws.com
export AWS_REGION=us-east-1
export S3_BUCKET_NAME=your-bucket-name
export REPO_URL=https://github.com/your-org/MSML605-PROJ.git

bash -c "$(curl -fsSL https://raw.githubusercontent.com/your-org/MSML605-PROJ/main/infra/ec2_setup.sh)"
# OR copy and paste the contents of infra/ec2_setup.sh and run it directly
```

Alternatively, pass the setup script as **EC2 User Data** at launch time:

```bash
aws ec2 run-instances \
    ... \
    --user-data file://infra/ec2_setup.sh
```

### 4.5 Verify the deployment

```bash
# From your local machine
curl http://$EC2_HOST:8000/health
# Expected: {"status":"ok","model_loaded":true,"model_version":"..."}

# Open in browser
echo "Dashboard: http://$EC2_HOST:8501"
echo "API docs:  http://$EC2_HOST:8000/docs"
```

---

## Phase 5 — SageMaker Training & Endpoint

### 5.1 Launch a SageMaker training job

```bash
python -c "
import boto3, os

sm = boto3.client('sagemaker', region_name='us-east-1')
sm.create_training_job(
    TrainingJobName='stock-prediction-v2',
    AlgorithmSpecification={
        'TrainingImage': '${ECR_URI}/stock-prediction/training:latest',
        'TrainingInputMode': 'File',
    },
    RoleArn='${SAGEMAKER_ROLE_ARN}',
    InputDataConfig=[{
        'ChannelName': 'train',
        'DataSource': {'S3DataSource': {
            'S3DataType': 'S3Prefix',
            'S3Uri': 's3://${S3_BUCKET_NAME}/processed/',
            'S3DataDistributionType': 'FullyReplicated',
        }},
    }],
    OutputDataConfig={
        'S3OutputPath': 's3://${S3_BUCKET_NAME}/models/',
    },
    ResourceConfig={
        'InstanceType': 'ml.m5.xlarge',
        'InstanceCount': 1,
        'VolumeSizeInGB': 30,
    },
    StoppingCondition={'MaxRuntimeInSeconds': 3600},
)
print('Training job launched — monitor at: https://console.aws.amazon.com/sagemaker')
"
```

Monitor the job:
```bash
aws sagemaker describe-training-job \
    --training-job-name stock-prediction-v1 \
    --query '[TrainingJobStatus, SecondaryStatus]'
```

### 5.2 Deploy the trained model as a SageMaker endpoint

```bash
python -c "
import sagemaker

role = '${SAGEMAKER_ROLE_ARN}'
model = sagemaker.model.Model(
    image_uri='${ECR_URI}/stock-prediction/inference:latest',
    model_data='s3://${S3_BUCKET_NAME}/models/stock-prediction-v2/output/model.tar.gz',
    role=role,
)
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium',
    endpoint_name='stock-prediction-endpoint',
)
print('Endpoint live:', predictor.endpoint_name)
"
```

Monitor the endpoint:
```bash
aws sagemaker describe-endpoint \
    --endpoint-name stock-prediction-endpoint \
    --query EndpointStatus
```

### 5.3 Enable SageMaker Model Monitor

```bash
python -c "
from src.monitoring.sagemaker_monitor import enable_data_capture, create_monitoring_schedule

enable_data_capture(
    endpoint_name='stock-prediction-endpoint',
    s3_capture_uri='s3://${S3_BUCKET_NAME}/monitoring/data-capture',
)

create_monitoring_schedule(
    endpoint_name='stock-prediction-endpoint',
    baseline_s3_uri='s3://${S3_BUCKET_NAME}/baseline',
    output_s3_uri='s3://${S3_BUCKET_NAME}/monitoring/monitor-output',
    role_arn='${SAGEMAKER_ROLE_ARN}',
)
print('Model Monitor schedule created')
"
```

---

## Phase 6 — Automated Retraining Pipeline

This phase wires together: **CloudWatch alarm → SNS → Lambda → SageMaker**.

### 6.1 Create the CloudWatch drift alarm

Run this **after** the first drift report has been pushed to CloudWatch (the monitoring agent does this automatically once 50+ predictions have been made):

```bash
python -c "
from src.monitoring.cloudwatch_logger import create_drift_alarm

create_drift_alarm(
    alarm_name='StockPredictorDriftAlarm',
    sns_topic_arn='${SNS_TOPIC_ARN}',
    threshold=0.2,
)
print('Alarm created')
"
```

### 6.2 Package and deploy the Lambda function

```bash
# Package Lambda with its dependencies
cd /path/to/MSML605-PROJ
zip -r lambda.zip \
    infra/lambda/retrain_handler.py \
    src/ \
    configs/ \
    requirements.txt

# Create the Lambda function
aws lambda create-function \
    --function-name stock-prediction-retrain \
    --runtime python3.11 \
    --role $LAMBDA_ROLE_ARN \
    --handler retrain_handler.lambda_handler \
    --zip-file fileb://lambda.zip \
    --timeout 120 \
    --environment Variables="{
        ECR_TRAINING_IMAGE_URI=$ECR_URI/stock-prediction/training:latest,
        S3_TRAINING_DATA_URI=s3://$S3_BUCKET_NAME/processed/,
        S3_MODEL_OUTPUT_URI=s3://$S3_BUCKET_NAME/models/,
        SAGEMAKER_ROLE_ARN=$SAGEMAKER_ROLE_ARN,
        SAGEMAKER_ENDPOINT_NAME=stock-prediction-endpoint,
        RETRAIN_PSI_THRESHOLD=0.2
    }"
```

### 6.3 Subscribe Lambda to SNS

```bash
# Grant SNS permission to invoke Lambda
aws lambda add-permission \
    --function-name stock-prediction-retrain \
    --statement-id sns-invoke \
    --action lambda:InvokeFunction \
    --principal sns.amazonaws.com \
    --source-arn $SNS_TOPIC_ARN

# Subscribe Lambda to the SNS topic
aws sns subscribe \
    --topic-arn $SNS_TOPIC_ARN \
    --protocol lambda \
    --notification-endpoint \
        arn:aws:lambda:us-east-1:$AWS_ACCOUNT_ID:function:stock-prediction-retrain

echo "Pipeline wired: CloudWatch → SNS → Lambda → SageMaker"
```

### 6.4 Full retraining pipeline flow

When PSI drift exceeds 0.2:
1. Monitoring agent pushes `Overall_Drift_Score` metric to CloudWatch
2. CloudWatch alarm fires and publishes to SNS
3. SNS invokes Lambda and sends email to you
4. Lambda launches a new SageMaker training job
5. After training completes, Lambda updates the SageMaker endpoint with the new model
6. EC2 inference API picks up the new model on next restart (or via `/reload` endpoint if implemented)

---

## Phase 7 — CI/CD with GitHub Actions

### 7.1 Add GitHub Secrets

Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | Your AWS access key |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key |
| `ECR_REGISTRY` | `YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com` |
| `EC2_HOST` | EC2 public IP or hostname |
| `EC2_SSH_KEY` | Full contents of `~/.ssh/stock-prediction-key.pem` |
| `S3_BUCKET_NAME` | Your S3 bucket name |

### 7.2 Workflow triggers

| Git event | Workflow | What it does |
|---|---|---|
| Pull request to `main` | `ci.yml` | Lint (ruff) → unit tests (pytest) → Docker build (no push) |
| Push / merge to `main` | `cd.yml` | Build images → push to ECR → SSH into EC2 → pull new images → restart containers → health check |

### 7.3 Test CI locally

```bash
# Install act (runs GitHub Actions workflows locally)
brew install act

act pull_request -j test
```

---

## Phase 8 — Monitoring & Drift Detection

### 8.1 Manual drift check

```bash
python -c "
import pandas as pd, json
from src.training.baseline_capture import load_baseline_stats
from src.monitoring.drift_detector import analyze_drift

baseline = load_baseline_stats(local_path='data/baseline/baseline_stats.json')
live = pd.read_parquet('data/processed/AAPL_train.parquet') \
         .drop(columns=['target']).tail(200)
report = analyze_drift(baseline, live)
print(json.dumps(report.to_dict(), indent=2))
"
```

### 8.2 View CloudWatch drift metrics

```bash
aws cloudwatch get-metric-statistics \
    --namespace StockPredictor/DriftMetrics \
    --metric-name Overall_Drift_Score \
    --dimensions Name=Ticker,Value=AAPL \
    --start-time $(date -u -v-24H +%Y-%m-%dT%H:%M:%SZ) \
    --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
    --period 3600 \
    --statistics Maximum
```

### 8.3 Inject artificial drift (for testing alarms)

```bash
python -c "
import numpy as np, requests

# Generate features shifted 5 standard deviations — guarantees alarm fires
records = (np.random.normal(5, 3, (200, 23))).tolist()
for rec in records:
    requests.post('http://localhost:8000/predict',
                  json={'ticker': 'AAPL', 'features': rec})
print('Injected 200 drifted samples — monitoring agent will detect on next cycle')
"
```

### 8.4 Check alarm state

```bash
aws cloudwatch describe-alarms \
    --alarm-names StockPredictorDriftAlarm \
    --query 'MetricAlarms[0].StateValue'
```

---

## Phase 9 — Load Testing

### 9.1 Run against EC2

```bash
# Basic test: 5 users for 60 seconds
locust -f tests/load/locustfile.py \
    --host http://$EC2_HOST:8000 \
    --headless -u 5 -r 1 --run-time 60s \
    --html reports/baseline.html

# Stress test: ramp to 50 users over 120 seconds
locust -f tests/load/locustfile.py \
    --host http://$EC2_HOST:8000 \
    --headless -u 50 -r 5 --run-time 120s \
    --html reports/stress.html

# Interactive UI (open http://localhost:8089)
locust -f tests/load/locustfile.py --host http://$EC2_HOST:8000
```

### 9.2 Performance targets

| Metric | Target |
|---|---|
| p95 latency | < 200 ms |
| Error rate at peak load | < 1% |
| Throughput before degradation | > 100 RPS |
| Memory growth over 10-min soak | < 50 MB |

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `AWS_ACCOUNT_ID` | Yes | 12-digit AWS account ID |
| `AWS_REGION` | Yes | AWS region (default: `us-east-1`) |
| `AWS_ACCESS_KEY_ID` | Yes (local) | AWS credentials — prefer IAM role on EC2 |
| `AWS_SECRET_ACCESS_KEY` | Yes (local) | AWS credentials |
| `S3_BUCKET_NAME` | Yes | Main project S3 bucket |
| `ECR_URI` | Yes | ECR registry URI, e.g. `123456789.dkr.ecr.us-east-1.amazonaws.com` |
| `SNS_TOPIC_ARN` | Yes | SNS topic ARN for alerts and Lambda trigger |
| `SAGEMAKER_ROLE_ARN` | Yes | IAM role ARN for SageMaker jobs |
| `SAGEMAKER_ENDPOINT_NAME` | Yes | Name of SageMaker endpoint (default: `stock-prediction-endpoint`) |
| `MODEL_DIR` | No | Local model directory (default: `models/latest`) |
| `MODEL_S3_BUCKET` | No | If set, loads model from S3 instead of local disk |
| `MODEL_S3_PREFIX` | No | S3 prefix for model artifacts (default: `models/latest`) |
| `DEFEATBETA_DATA_SOURCE` | No | Optional marker variable only (default source is defeatbeta-api) |
| `CW_NAMESPACE` | No | CloudWatch namespace (default: `StockPredictor/DriftMetrics`) |
| `DRIFT_CHECK_INTERVAL` | No | Seconds between drift checks (default: `3600`) |
| `RETRAIN_PSI_THRESHOLD` | No | PSI threshold for triggering retraining (default: `0.2`) |
| `API_URL` | No | Inference API URL used by dashboard and monitoring agent |

---

## Estimated AWS Costs

| Scenario | Monthly Cost |
|---|---|
| EC2 t3.small running 24/7 | ~$15 |
| S3 (< 5 GB data + model storage) | < $1 |
| ECR (< 500 MB images) | < $1 |
| SageMaker endpoint (ml.t3.medium, 8 hrs/day) | ~$12 |
| SageMaker training (ml.m5.xlarge, 1 hr/month) | ~$0.23 |
| Lambda (triggered by drift, < 100 invocations) | Free tier |
| CloudWatch (< 10 custom metrics) | Free tier |
| SNS (< 1M publishes) | Free tier |
| **Total (typical)** | **~$28–$30/month** |

To minimize costs during development: stop the SageMaker endpoint when not actively testing (`aws sagemaker delete-endpoint --endpoint-name stock-prediction-endpoint`).

---

## Troubleshooting

### Model not loading on startup
- Verify `MODEL_DIR` contains `model.xgb` **or** `MODEL_S3_BUCKET` + `MODEL_S3_PREFIX` point to the correct S3 location.
- Check container logs: `docker logs inference-api`

### defeatbeta-api import or runtime issues
- Ensure dependencies are installed in the active environment: `pip install -r requirements-dev.txt`
- Verify package import: `python -c "from defeatbeta_api.data.ticker import Ticker; print(Ticker('AAPL').price().head())"`
- If Pylance still shows unresolved imports after install, reload the Python environment in VS Code.

### Docker containers on EC2 can't reach S3
- Confirm the EC2 IAM instance profile is attached: `aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].IamInstanceProfile'`
- The profile should be `stock-prediction-ec2-profile`.

### SageMaker training job fails with "AccessDenied"
- Verify the SageMaker role has `AmazonSageMakerFullAccess` + `AmazonS3FullAccess`.
- Check that ECR image has a resource-based policy allowing SageMaker to pull it: AWS Console → ECR → repository → Permissions.

### CloudWatch alarm never fires after drift injection
- Confirm the monitoring agent is running: `docker ps | grep monitoring-agent`
- The agent needs >= 50 inference log entries before it runs a drift check.
- Check that `S3_BUCKET_NAME` and AWS credentials are set in the monitoring agent container.

### Lambda fails with "ResourceNotFound" for SageMaker endpoint
- The first training job must complete and the endpoint must be deployed (Phase 5) before Lambda can update it.
- Check Lambda CloudWatch logs: AWS Console → Lambda → Monitor → View CloudWatch logs.
