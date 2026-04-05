# Setup & Deployment Guide

Containerized Real-Time Stock Prediction System with Drift Detection and Automated Retraining
MSML605 — University of Maryland

---

## Table of Contents

1. [Services Overview](#1-services-overview)
2. [Prerequisites](#2-prerequisites)
3. [Local Development Setup](#3-local-development-setup)
4. [Train the Model Locally](#4-train-the-model-locally)
5. [Run with Docker Compose](#5-run-with-docker-compose)
6. [AWS Infrastructure Setup](#6-aws-infrastructure-setup)
7. [Deploy to EC2](#7-deploy-to-ec2)
8. [SageMaker Integration](#8-sagemaker-integration)
9. [CI/CD via GitHub Actions](#9-cicd-via-github-actions)
10. [Load Testing](#10-load-testing)
11. [Monitoring & Drift Detection](#11-monitoring--drift-detection)
12. [Environment Variables Reference](#12-environment-variables-reference)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Services Overview

| Service | What it does | Port |
|---|---|---|
| **inference-api** | FastAPI REST endpoint serving XGBoost predictions | 8000 |
| **dashboard** | Streamlit observability UI (predictions, drift, system metrics) | 8501 |
| **monitoring-agent** | Background process running hourly KS+PSI drift checks | — |
| **AWS Lambda** | Triggers SageMaker retraining when drift is detected | — |
| **SageMaker Endpoint** | Managed real-time inference endpoint (production option) | — |

### Data Flow

```
Yahoo Finance → feature engineering → XGBoost train → FastAPI
                                                          |
                                               inference log (ring buffer)
                                                          |
                                              drift detector (KS + PSI)
                                                          |
                                             CloudWatch Alarm > PSI 0.2
                                                          |
                                              SNS → Lambda → SageMaker retrain
                                                          |
                                             Updated model deployed to endpoint
                                                          |
                                                   Streamlit dashboard
```

---

## 2. Prerequisites

### Required Software

| Tool | Version | Install |
|---|---|---|
| Python | 3.11+ | [python.org](https://python.org) |
| Docker Desktop | 24+ | [docker.com](https://docker.com) |
| AWS CLI v2 | 2.x | `brew install awscli` |
| Git | any | `brew install git` |

### Required AWS Services (Free Tier eligible where noted)

| Service | Purpose | Notes |
|---|---|---|
| EC2 (t3.small) | Runs Docker containers | ~$15/month |
| S3 | Stores data, models, drift reports | Free tier: 5 GB |
| ECR | Stores Docker images | Free tier: 500 MB |
| SageMaker | Managed training + endpoint | `ml.t3.medium` ~$0.05/hr |
| Lambda | Retraining trigger | Free tier: 1M requests |
| CloudWatch | Metrics, alarms, logs | Free tier: 10 metrics |
| SNS | Alert notifications | Free tier: 1M publishes |
| IAM | Roles and permissions | Free |

### API Keys

| Key | Where to get | Used by |
|---|---|---|
| Yahoo Finance | Free, no key needed | `fetch_yahoo.py` |
| Alpha Vantage | [alphavantage.co](https://www.alphavantage.co/support/#api-key) (free) | `fetch_alphavantage.py` (fallback) |

---

## 3. Local Development Setup

### 3.1 Clone the repository

```bash
git clone <your-repo-url>
cd stock-prediction-mlops
```

### 3.2 Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate.bat    # Windows

pip install -r requirements-dev.txt
```

### 3.3 Configure AWS credentials (for S3 access)

```bash
aws configure
# AWS Access Key ID: <your key>
# AWS Secret Access Key: <your secret>
# Default region: us-east-1
# Default output format: json
```

Or use environment variables:

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1
```

### 3.4 Set required environment variables

Copy the template and fill in your values:

```bash
cp .env.example .env   # (create manually if not present)
```

Minimum required variables:

```bash
export S3_BUCKET_NAME=your-stock-prediction-bucket
export ALPHAVANTAGE_API_KEY=your_key_here   # optional fallback
```

### 3.5 Verify setup

```bash
python -c "import xgboost, fastapi, streamlit; print('All packages OK')"
pytest tests/unit/ -v
```

---

## 4. Train the Model Locally

### 4.1 Fetch training data

```bash
python -c "
from src.ingestion.fetch_yahoo import fetch_ohlcv
df = fetch_ohlcv('AAPL', '2019-01-01', '2024-01-01')
df.to_parquet('data/raw/AAPL_daily.parquet')
print('Downloaded', len(df), 'rows')
"
```

### 4.2 Build feature matrix

```bash
python -c "
from src.features.feature_pipeline import build_feature_matrix, split_and_scale
df = build_feature_matrix('AAPL', '2019-01-01', '2024-01-01')
train, val, test, scaler = split_and_scale(df, scaler_save_path='models/latest/scaler.pkl')
train.to_parquet('data/processed/AAPL_train.parquet')
print('Train:', len(train), 'rows | Val:', len(val), 'rows | Test:', len(test), 'rows')
"
```

### 4.3 Capture baseline stats (for drift detection)

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

### 4.4 Train with walk-forward CV

```bash
python src/training/train_xgboost.py --config configs/model_config.yaml --ticker AAPL
# Outputs: models/latest/model.xgb + output/metrics.json
```

### 4.5 Optional: Run Optuna hyperparameter tuning

```bash
python -c "
import pandas as pd
from src.training.hyperparameter_tuning import run_tuning
import json

df = pd.read_parquet('data/processed/AAPL_train.parquet')
best_params = run_tuning(df, n_trials=50)
with open('configs/best_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)
print('Best params:', best_params)
"
```

---

## 5. Run with Docker Compose

### 5.1 Create the `.env` file

```bash
cat > .env <<EOF
S3_BUCKET_NAME=your-bucket-name
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
ECR_URI=123456789.dkr.ecr.us-east-1.amazonaws.com
EOF
```

### 5.2 Build and start all services

```bash
# From the project root
docker compose -f infra/docker-compose.yml up --build -d
```

### 5.3 Verify services are running

```bash
docker compose -f infra/docker-compose.yml ps
curl http://localhost:8000/health
# Expected: {"status":"ok","model_loaded":true,"model_version":"..."}
```

Open the dashboard: http://localhost:8501

### 5.4 Test a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "features": [0.1, -0.2, 0.05, 1.2, 0.8, 0.3, -0.1, 0.4, 0.9, 1.1,
                 -0.3, 0.7, 0.2, -0.5, 1.0, 0.6, -0.2, 0.8, 0.3, 1.4,
                 -0.1, 0.5, 0.9]
  }'
```

### 5.5 Stop services

```bash
docker compose -f infra/docker-compose.yml down
```

---

## 6. AWS Infrastructure Setup

### 6.1 Create S3 bucket

```bash
aws s3 mb s3://your-stock-prediction-bucket --region us-east-1

# Create required prefixes
aws s3api put-object --bucket your-stock-prediction-bucket --key raw/
aws s3api put-object --bucket your-stock-prediction-bucket --key processed/
aws s3api put-object --bucket your-stock-prediction-bucket --key models/
aws s3api put-object --bucket your-stock-prediction-bucket --key baseline/
aws s3api put-object --bucket your-stock-prediction-bucket --key monitoring/
```

### 6.2 Create ECR repositories

```bash
aws ecr create-repository --repository-name stock-prediction/inference --region us-east-1
aws ecr create-repository --repository-name stock-prediction/training --region us-east-1
aws ecr create-repository --repository-name stock-prediction/dashboard --region us-east-1

# Note the repository URIs from the output
```

### 6.3 Create IAM role for EC2 and SageMaker

```bash
# Create EC2 instance profile with S3 + ECR + CloudWatch permissions
# (attach policies: AmazonS3FullAccess, AmazonEC2ContainerRegistryReadOnly,
#  CloudWatchAgentServerPolicy, AmazonSSMManagedInstanceCore)

# Create SageMaker execution role
# (attach policies: AmazonSageMakerFullAccess, AmazonS3FullAccess)
```

### 6.4 Create SNS topic for alerts

```bash
aws sns create-topic --name stock-prediction-alerts --region us-east-1
# Subscribe your email:
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:YOUR_ACCOUNT:stock-prediction-alerts \
  --protocol email \
  --notification-endpoint your@email.com
```

### 6.5 Create CloudWatch alarm

```bash
# After first drift report is pushed, create the alarm:
python -c "
from src.monitoring.cloudwatch_logger import create_drift_alarm
create_drift_alarm(
    alarm_name='StockPredictorDriftAlarm',
    sns_topic_arn='arn:aws:sns:us-east-1:YOUR_ACCOUNT:stock-prediction-alerts',
    threshold=0.2
)
"
```

### 6.6 Create Lambda function

```bash
# Package the Lambda
cd /path/to/project
zip -r lambda.zip infra/lambda/retrain_handler.py src/ configs/ requirements.txt

# Create function
aws lambda create-function \
  --function-name stock-prediction-retrain \
  --runtime python3.11 \
  --role arn:aws:iam::YOUR_ACCOUNT:role/lambda-sagemaker-role \
  --handler retrain_handler.lambda_handler \
  --zip-file fileb://lambda.zip \
  --timeout 60 \
  --environment Variables="{
    ECR_TRAINING_IMAGE_URI=YOUR_ECR_URI/stock-prediction/training:latest,
    S3_TRAINING_DATA_URI=s3://your-bucket/processed/,
    S3_MODEL_OUTPUT_URI=s3://your-bucket/models/,
    SAGEMAKER_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT:role/sagemaker-execution-role,
    SAGEMAKER_ENDPOINT_NAME=stock-prediction-endpoint,
    RETRAIN_PSI_THRESHOLD=0.2
  }"

# Connect SNS to Lambda
aws lambda add-permission \
  --function-name stock-prediction-retrain \
  --statement-id sns-invoke \
  --action lambda:InvokeFunction \
  --principal sns.amazonaws.com \
  --source-arn arn:aws:sns:us-east-1:YOUR_ACCOUNT:stock-prediction-alerts

aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:YOUR_ACCOUNT:stock-prediction-alerts \
  --protocol lambda \
  --notification-endpoint arn:aws:lambda:us-east-1:YOUR_ACCOUNT:function:stock-prediction-retrain
```

---

## 7. Deploy to EC2

### 7.1 Launch EC2 instance

- AMI: Ubuntu 22.04 LTS
- Instance type: t3.small (dev) or t3.medium (prod)
- IAM role: attach the EC2 instance profile from step 6.3
- Security group: allow inbound TCP 8000 (API) and 8501 (dashboard) from your IP

### 7.2 Build and push Docker images

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin YOUR_ECR_URI

# Build and push inference image
docker build -f infra/docker/Dockerfile.inference \
  -t YOUR_ECR_URI/stock-prediction/inference:latest .
docker push YOUR_ECR_URI/stock-prediction/inference:latest

# Build and push dashboard image
docker build -f infra/docker/Dockerfile.dashboard \
  -t YOUR_ECR_URI/stock-prediction/dashboard:latest .
docker push YOUR_ECR_URI/stock-prediction/dashboard:latest
```

### 7.3 Bootstrap EC2

SSH into your instance and run:

```bash
# Set required variables
export ECR_URI=YOUR_ECR_URI
export AWS_REGION=us-east-1
export S3_BUCKET_NAME=your-bucket-name
export REPO_URL=https://github.com/your-org/stock-prediction-mlops.git

# Run setup script
curl -fsSL https://raw.githubusercontent.com/your-org/stock-prediction-mlops/main/infra/ec2_setup.sh | sudo bash
```

Or use EC2 User Data during instance launch (paste the script contents).

### 7.4 Verify deployment

```bash
# From your local machine
curl http://YOUR_EC2_PUBLIC_IP:8000/health
# Open dashboard: http://YOUR_EC2_PUBLIC_IP:8501
```

---

## 8. SageMaker Integration

### 8.1 Push training image to ECR

```bash
docker build -f infra/docker/Dockerfile.training \
  -t YOUR_ECR_URI/stock-prediction/training:latest .
docker push YOUR_ECR_URI/stock-prediction/training:latest
```

### 8.2 Upload training data to S3

```bash
aws s3 sync data/processed/ s3://your-bucket/processed/
```

### 8.3 Run a training job from Python

```bash
python -c "
import boto3

sm = boto3.client('sagemaker', region_name='us-east-1')
sm.create_training_job(
    TrainingJobName='stock-prediction-v1',
    AlgorithmSpecification={
        'TrainingImage': 'YOUR_ECR_URI/stock-prediction/training:latest',
        'TrainingInputMode': 'File'
    },
    RoleArn='arn:aws:iam::YOUR_ACCOUNT:role/sagemaker-execution-role',
    InputDataConfig=[{
        'ChannelName': 'train',
        'DataSource': {'S3DataSource': {
            'S3DataType': 'S3Prefix',
            'S3Uri': 's3://your-bucket/processed/',
            'S3DataDistributionType': 'FullyReplicated'
        }}
    }],
    OutputDataConfig={'S3OutputPath': 's3://your-bucket/models/'},
    ResourceConfig={'InstanceType': 'ml.m5.xlarge', 'InstanceCount': 1, 'VolumeSizeInGB': 30},
    StoppingCondition={'MaxRuntimeInSeconds': 3600}
)
print('Training job launched')
"
```

### 8.4 Deploy as SageMaker endpoint

```bash
python -c "
import sagemaker, boto3

role = 'arn:aws:iam::YOUR_ACCOUNT:role/sagemaker-execution-role'
model = sagemaker.model.Model(
    image_uri='YOUR_ECR_URI/stock-prediction/inference:latest',
    model_data='s3://your-bucket/models/stock-prediction-v1/output/model.tar.gz',
    role=role,
)
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.t3.medium',
    endpoint_name='stock-prediction-endpoint',
)
print('Endpoint deployed:', predictor.endpoint_name)
"
```

### 8.5 Enable Model Monitor

```bash
python -c "
from src.monitoring.sagemaker_monitor import enable_data_capture, create_monitoring_schedule

enable_data_capture(
    endpoint_name='stock-prediction-endpoint',
    s3_capture_uri='s3://your-bucket/monitoring/data-capture',
)

create_monitoring_schedule(
    endpoint_name='stock-prediction-endpoint',
    baseline_s3_uri='s3://your-bucket/baseline',
    output_s3_uri='s3://your-bucket/monitoring/monitor-output',
    role_arn='arn:aws:iam::YOUR_ACCOUNT:role/sagemaker-execution-role',
)
"
```

---

## 9. CI/CD via GitHub Actions

### 9.1 Add GitHub Secrets

Go to your GitHub repo → Settings → Secrets → Actions. Add:

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | Your AWS access key |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key |
| `ECR_REGISTRY` | `123456789.dkr.ecr.us-east-1.amazonaws.com` |
| `EC2_HOST` | EC2 public IP or hostname |
| `EC2_SSH_KEY` | Contents of your EC2 private key (.pem) |

### 9.2 Workflow behavior

| Event | Workflow | Actions |
|---|---|---|
| PR to main | `ci.yml` | Lint → Unit tests → Integration tests → Docker build (no push) |
| Push/merge to main | `cd.yml` | Build → Push to ECR → SSH deploy to EC2 → Health check |

### 9.3 Run CI locally (act)

```bash
# Install act: brew install act
act pull_request -j test
```

---

## 10. Load Testing

### 10.1 Install Locust

```bash
pip install locust==2.29.0
```

### 10.2 Run load test scenarios

```bash
# Scenario 1: Baseline (5 users, 60 seconds)
locust -f tests/load/locustfile.py \
  --host http://your-ec2-ip:8000 \
  --headless -u 5 -r 1 --run-time 60s \
  --html reports/baseline.html

# Scenario 2: Stress ramp (5 → 50 users, 120 seconds)
locust -f tests/load/locustfile.py \
  --host http://your-ec2-ip:8000 \
  --headless -u 50 -r 5 --run-time 120s \
  --html reports/stress.html

# Scenario 3: Spike (100 users instant, 30 seconds)
locust -f tests/load/locustfile.py --class-picker SpikeUser \
  --host http://your-ec2-ip:8000 \
  --headless -u 100 -r 100 --run-time 30s \
  --html reports/spike.html

# Scenario 4: Soak (20 users, 10 minutes)
locust -f tests/load/locustfile.py --class-picker SoakUser \
  --host http://your-ec2-ip:8000 \
  --headless -u 20 -r 2 --run-time 600s \
  --html reports/soak.html

# Interactive web UI (open http://localhost:8089)
locust -f tests/load/locustfile.py --host http://your-ec2-ip:8000
```

### 10.3 What to measure

| Metric | Target |
|---|---|
| p95 latency | < 200 ms |
| Error rate at peak load | < 1% |
| Max RPS before degradation | > 100 RPS |
| Memory growth over soak test | < 50 MB |

---

## 11. Monitoring & Drift Detection

### 11.1 Trigger a manual drift check

```bash
python -c "
import pandas as pd, json
from src.training.baseline_capture import load_baseline_stats
from src.monitoring.drift_detector import analyze_drift

baseline = load_baseline_stats(local_path='data/baseline/baseline_stats.json')
live = pd.read_parquet('data/processed/AAPL_train.parquet').drop(columns=['target']).tail(200)
report = analyze_drift(baseline, live)

print(json.dumps(report.to_dict(), indent=2))
"
```

### 11.2 Inject drift for testing

To verify that alarms fire correctly, inject artificial drift:

```bash
python -c "
import numpy as np, pandas as pd, json, requests

# Generate highly drifted features (shifted by 5 standard deviations)
records = (np.random.normal(5, 3, (200, 23))).tolist()

# Push to the inference log via the API
for rec in records:
    requests.post('http://localhost:8000/predict',
                  json={'ticker': 'AAPL', 'features': rec})

print('Injected', len(records), 'drifted samples')
"
# Then wait for the monitoring agent to run (or trigger manually)
```

### 11.3 View drift metrics in CloudWatch

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

---

## 12. Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | `models/latest` | Local path to model artefacts |
| `MODEL_S3_BUCKET` | — | S3 bucket for model artefacts (overrides MODEL_DIR) |
| `MODEL_S3_PREFIX` | `models/latest` | S3 prefix for model artefacts |
| `S3_BUCKET_NAME` | — | Main project S3 bucket |
| `AWS_REGION` | `us-east-1` | AWS region |
| `AWS_ACCESS_KEY_ID` | — | AWS credentials (prefer IAM role on EC2) |
| `AWS_SECRET_ACCESS_KEY` | — | AWS credentials |
| `ALPHAVANTAGE_API_KEY` | — | Alpha Vantage API key (optional fallback) |
| `CW_NAMESPACE` | `StockPredictor/DriftMetrics` | CloudWatch metric namespace |
| `ECR_TRAINING_IMAGE_URI` | — | ECR URI for training Docker image |
| `S3_TRAINING_DATA_URI` | — | S3 prefix for training parquet files |
| `S3_MODEL_OUTPUT_URI` | — | S3 prefix where trained models are saved |
| `SAGEMAKER_ROLE_ARN` | — | IAM role ARN for SageMaker jobs |
| `SAGEMAKER_ENDPOINT_NAME` | `stock-prediction-endpoint` | SageMaker endpoint name |
| `RETRAIN_PSI_THRESHOLD` | `0.2` | PSI threshold above which retraining fires |
| `DRIFT_CHECK_INTERVAL` | `3600` | Seconds between drift checks (monitoring agent) |
| `API_URL` | `http://localhost:8000` | Inference API URL (used by dashboard + monitoring agent) |

---

## 13. Troubleshooting

### Model not loading

```
ERROR: Model not loaded. Check startup logs.
```
- Ensure `MODEL_DIR` points to a directory containing `model.xgb`
- Or set `MODEL_S3_BUCKET` + `MODEL_S3_PREFIX` and verify S3 permissions

### yfinance download returns empty DataFrame

- Yahoo Finance occasionally blocks requests. Add a delay or use `proxy` parameter
- Fallback: use Alpha Vantage (`ALPHAVANTAGE_API_KEY` must be set)

### Docker container can't reach S3

- On EC2: attach an IAM instance profile with S3 access (no keys needed)
- Locally: ensure `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are in `.env`

### SageMaker training job fails with "Permission denied"

- Verify the SageMaker execution role has `AmazonSageMakerFullAccess` and `AmazonS3FullAccess`
- Check that the ECR repository policy allows SageMaker to pull the image

### Locust reports high error rate

- Check that the inference API is healthy: `curl http://host:8000/health`
- If `model_loaded: false`, the model artefacts are missing
- Increase instance size or reduce `--users` count

### Drift alarm fires immediately after deployment

- This happens if the live window is populated with data from a different ticker or time period
- Clear the inference log: restart the inference-api container
- Ensure the baseline stats match the same feature pipeline version as production
