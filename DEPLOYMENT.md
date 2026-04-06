# Stock Prediction MLOps — AWS Deployment Plan

## Objective
Stock price prediction model fully deployed on AWS with automated drift detection, fresh data ingestion, model retraining, and endpoint updates. All computation (Docker builds, ingestion, training, retraining) runs on AWS. Local machine only runs bash scripts.

---

## Architecture

```
Local machine  →  bash scripts only (no Docker, no compute)

[S3 bucket: stock-prediction-mlops-{account_id}]
  raw/          OHLCV parquet files
  processed/    feature matrices (parquet)
  models/       XGBoost artifacts (.xgb, scaler.pkl)
  baseline/     baseline_stats.json (used for drift detection)
  captures/     SageMaker Data Capture output (JSONL)
  monitoring/   latest_drift_report.json

[ECR — 3 repositories]
  stock-prediction/ingestion:latest
  stock-prediction/training:latest
  stock-prediction/inference:latest

[CodeBuild: stock-prediction-build]
  Builds all 3 images on linux/amd64, pushes to ECR
  Source: S3 zip upload from local repo
  No local Docker needed — avoids OCI manifest issues with SageMaker

[SageMaker Processing Jobs]
  drift-check-*   reads S3 captures → PSI/KS analysis → S3 report
  stock-ingest-*  fetches OHLCV via defeatbeta-api → features → S3

[SageMaker Training Job: stock-retrain-*]
  Input:  s3://.../processed/
  Output: s3://.../models/ + s3://.../baseline/baseline_stats.json

[SageMaker Endpoint: stock-prediction-endpoint]
  Port 8080, /ping, /invocations
  Data Capture → s3://.../captures/ (input features logged for drift detection)
  Instance: ml.t2.medium

[Lambda: stock-prediction-orchestrator]
  Single function handles all pipeline stages:
  scheduled        → start drift-check Processing Job
  drift_complete   → read S3 report, if drifted → start ingestion
  ingestion_done   → start training job
  training_done    → update endpoint (zero-downtime blue/green)

[EventBridge — 3 rules]
  Rule 1: cron(0 6 * * ? *)              → Lambda {source: scheduled}   (daily)
  Rule 2: Processing Job state=Completed → Lambda {source: drift_complete|ingestion_complete}
  Rule 3: Training Job state=Completed   → Lambda {source: training_complete}

[CloudWatch + SNS]
  Metric: StockPredictor/DriftMetrics/max_psi
  Alarm:  max_psi > 0.2 → SNS email alert (backup trigger)
```

**AWS Services:** S3, ECR, CodeBuild, SageMaker, Lambda, EventBridge, CloudWatch, SNS, IAM  
**Not used:** EC2, ECS, EKS, RDS, SageMaker Model Monitor (built-in)

---

## Repository Structure (after cleanup)

```
MSML605-PROJ/
├── .env                          # AWS credentials (not committed)
├── .env.example
├── pyproject.toml
├── requirements.txt              # inference + training deps (no defeatbeta-api)
├── requirements-ingestion.txt    # defeatbeta-api (isolated, separate Docker image)
├── DEPLOYMENT.md                 # this file
│
├── configs/
│   ├── model_config.yaml
│   ├── feature_config.yaml
│   └── aws_config.yaml
│
├── src/
│   ├── features/
│   │   ├── feature_pipeline.py
│   │   └── technical_indicators.py
│   ├── ingestion/
│   │   ├── fetch_yahoo.py            # defeatbeta-api data fetching
│   │   ├── data_validator.py
│   │   └── ingest_and_upload.py      # SageMaker Processing Job entrypoint
│   ├── training/
│   │   ├── train_xgboost.py          # SageMaker Training Job entrypoint
│   │   ├── baseline_capture.py       # saves baseline_stats.json to S3
│   │   ├── evaluate.py
│   │   └── time_series_split.py
│   ├── inference/
│   │   ├── app.py                    # FastAPI: /ping /invocations /predict /health
│   │   ├── predictor.py
│   │   └── schemas.py
│   ├── monitoring/
│   │   ├── drift_detector.py         # KS test + PSI
│   │   ├── cloudwatch_logger.py
│   │   └── run_drift_check.py        # SageMaker Processing Job entrypoint (NEW)
│   └── retraining/
│       └── retrain_trigger.py        # Lambda handler (orchestrates all stages)
│
├── infra/
│   ├── buildspec.yml                 # CodeBuild: builds + pushes all 3 images
│   └── docker/
│       ├── Dockerfile.ingestion
│       ├── Dockerfile.training
│       ├── Dockerfile.inference
│       └── Dockerfile.dashboard      # local use only
│
├── scripts/
│   ├── common.sh                     # shared vars + wait_for_sagemaker_job()
│   ├── 00_cleanup.sh                 # delete all AWS resources
│   ├── 01_bootstrap.sh               # create S3, ECR, IAM, CodeBuild
│   ├── 02_build.sh                   # zip repo → S3 → trigger CodeBuild → wait
│   ├── 03_initial_ingest.sh          # first SageMaker Processing Job (ingestion)
│   ├── 04_initial_train.sh           # first SageMaker Training Job
│   ├── 05_deploy.sh                  # SageMaker model + endpoint with data capture
│   └── 06_setup_automation.sh        # Lambda + EventBridge + CloudWatch + SNS
│
├── tests/
│   ├── unit/
│   └── integration/
│
└── dashboard/                        # Streamlit (run locally only)
    └── app.py
```

**Files deleted vs previous version:** `infra/ec2_setup.sh`, `infra/docker-compose.yml`, `PLAN.md`, `SETUP.md`, `src/monitoring/sagemaker_monitor.py`

---

## Files to Create / Modify

### New files
| File | Purpose |
|------|---------|
| `scripts/common.sh` | Shared env vars, `wait_for_sagemaker_job()` helper |
| `scripts/00_cleanup.sh` | Delete all existing AWS resources before fresh deploy |
| `scripts/01_bootstrap.sh` | Create S3, ECR repos, IAM roles, CodeBuild project |
| `scripts/02_build.sh` | Upload source zip to S3, start CodeBuild, poll until done |
| `scripts/03_initial_ingest.sh` | Run ingestion Processing Job, wait for completion |
| `scripts/04_initial_train.sh` | Run training job, wait for completion |
| `scripts/05_deploy.sh` | Deploy endpoint with data capture enabled |
| `scripts/06_setup_automation.sh` | Lambda + EventBridge + CloudWatch + SNS |
| `infra/buildspec.yml` | CodeBuild buildspec (3 images) |
| `src/monitoring/run_drift_check.py` | Drift check Processing Job script |

### Modified files
| File | Change |
|------|--------|
| `src/retraining/retrain_trigger.py` | Restructure for 4-stage event flow; add `_run_drift_check_job()` |
| `src/training/train_xgboost.py` | Add `capture_baseline_stats()` → S3 after training |
| `configs/aws_config.yaml` | Fix inference instance `ml.t3.medium` → `ml.t2.medium` |

---

## .env Variables Required

```bash
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=<your key>
AWS_SECRET_ACCESS_KEY=<your secret>
S3_BUCKET_NAME=stock-prediction-mlops-<account_id>
ALERT_EMAIL=your@email.com
TICKERS=AAPL,MSFT,GOOGL
LOOKBACK_DAYS=730
```

---

## Step-by-Step Deployment

### Prerequisites
```bash
# AWS CLI authenticated
aws sts get-caller-identity

# .env populated (see above)
```

### Step 0 — Delete old AWS resources
```bash
bash scripts/00_cleanup.sh
# Deletes: SageMaker endpoint/config/model, Lambda, EventBridge rules,
#          CloudWatch alarms, SNS topic, IAM roles, ECR repos, CodeBuild project
# Prompts before wiping S3 bucket contents
```

### Step 1 — Create fresh infrastructure (~2 min)
```bash
bash scripts/01_bootstrap.sh
# Creates:
#   S3 bucket with versioning
#   ECR repos: stock-prediction/{ingestion,training,inference}
#   IAM role: stock-prediction-sagemaker-role (SageMakerFullAccess + S3 + ECR)
#   IAM role: stock-prediction-lambda-role (LambdaBasic + SageMaker + S3 + SNS)
#   IAM role: stock-prediction-codebuild-role (ECR + CodeBuild + CloudWatchLogs)
#   CodeBuild project: stock-prediction-build (privileged, standard:7.0)
```

### Step 2 — Build Docker images on AWS (~10-15 min)
```bash
bash scripts/02_build.sh
# Zips repo (excludes .git/, data/, models/)
# Uploads zip to s3://$S3_BUCKET/codebuild/source.zip
# Starts CodeBuild build, polls until SUCCEEDED
# CodeBuild: plain `docker build` on linux/amd64 → no OCI manifest issues
# Pushes 3 images to ECR
```

### Step 3 — Initial data ingestion (~5 min)
```bash
bash scripts/03_initial_ingest.sh
# Starts SageMaker Processing Job (ingestion image)
# Fetches 730 days OHLCV for AAPL, MSFT, GOOGL via defeatbeta-api
# Builds feature matrix (RSI, MACD, Bollinger, SMA, EMA, lags)
# Writes parquet to s3://.../processed/{AAPL,MSFT,GOOGL}.parquet
# Waits for Completed
```

### Step 4 — Initial model training (~15 min)
```bash
bash scripts/04_initial_train.sh
# Starts SageMaker Training Job (training image, ml.m5.xlarge)
# Reads s3://.../processed/, trains XGBoost p10/p50/p90 models
# Outputs: model.xgb, model_p10.xgb, model_p90.xgb, scaler.pkl → s3://.../models/
# Also writes baseline_stats.json → s3://.../baseline/  (required for drift detection)
# Waits for Completed
```

### Step 5 — Deploy inference endpoint (~10 min)
```bash
bash scripts/05_deploy.sh
# Gets model artifact URI from completed training job
# Creates SageMaker Model (inference:latest image + model.tar.gz)
# Creates Endpoint Config with Data Capture:
#   - Captures inference inputs → s3://.../captures/ (JSONL, base64 encoded)
#   - 100% sampling rate
# Creates endpoint stock-prediction-endpoint (ml.t2.medium)
# Waits for InService
```

### Step 6 — Set up automation (~2 min)
```bash
bash scripts/06_setup_automation.sh
# Creates SNS topic + subscribes ALERT_EMAIL
# Packages Lambda (retrain_trigger.py + monitoring/ + baseline_capture.py)
# Creates Lambda: stock-prediction-orchestrator (Python 3.11, 512MB, 300s)
# Sets Lambda env vars: S3_BUCKET, ECR_*_IMAGE_URI, SAGEMAKER_ROLE_ARN, etc.
# Creates EventBridge rules:
#   Rule 1: cron(0 6 * * ? *) → Lambda {source: scheduled}
#   Rule 2: Processing Job Completed → Lambda (routes by job name prefix)
#   Rule 3: Training Job Completed → Lambda {source: training_complete}
# Creates CloudWatch alarm: max_psi > 0.2 → SNS
```

### Step 7 — Verify
```bash
# Test endpoint (replace features with 23 real values)
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name stock-prediction-endpoint \
  --content-type application/json \
  --body '{"ticker":"AAPL","features":[0.1,0.2,...]}' \
  /tmp/response.json
cat /tmp/response.json

# Trigger manual drift check
aws lambda invoke \
  --function-name stock-prediction-orchestrator \
  --payload '{"source":"scheduled"}' \
  /tmp/lambda_out.json
cat /tmp/lambda_out.json

# Check CloudWatch for drift metrics
aws cloudwatch get-metric-statistics \
  --namespace StockPredictor/DriftMetrics \
  --metric-name max_psi \
  --start-time $(date -u -v-1d +%Y-%m-%dT%H:%M:%SZ) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --period 86400 \
  --statistics Maximum
```

---

## Automated Retraining Flow (once deployed)

```
Every day 6 AM UTC
  └─ EventBridge → Lambda {source: scheduled}
       └─ SageMaker Processing Job: drift-check-YYYYMMDD
            reads s3://captures/ (last 7 days)
            computes PSI + KS test vs baseline
            writes s3://monitoring/latest_drift_report.json
            pushes max_psi to CloudWatch

  └─ EventBridge (Processing Job Completed)
       └─ Lambda reads drift report
            if trigger_retraining=true:
              └─ SageMaker Processing Job: stock-ingest-YYYYMMDD
                   fetches fresh OHLCV → new feature matrices → S3

  └─ EventBridge (Processing Job Completed)
       └─ Lambda → SageMaker Training Job: stock-retrain-YYYYMMDD
            trains on fresh data
            saves new model artifacts + baseline_stats to S3

  └─ EventBridge (Training Job Completed)
       └─ Lambda → creates new SageMaker Model + Endpoint Config
            updates endpoint in-place (zero-downtime blue/green)
            SNS email: "Model retrained and endpoint updated"
```

---

## Key Design Decisions

| Problem | Solution |
|---------|----------|
| OCI manifest rejected by SageMaker | Plain `docker build` on CodeBuild (linux/amd64 native) — no buildx needed |
| `defeatbeta-api` conflicts with XGBoost/FastAPI deps | Isolated to `requirements-ingestion.txt` + `Dockerfile.ingestion` only |
| No EC2 to run compute | All compute on SageMaker (Processing + Training Jobs) + Lambda |
| Lambda ZIP > 50MB (scipy) | Drift check runs as SageMaker Processing Job (inference image has scipy/pandas) |
| In-memory ring buffer lost on container restart | SageMaker Data Capture writes to S3 (durable) |
| `ml.t3.medium` not valid endpoint instance | Use `ml.t2.medium` |
| Platform mismatch (Mac ARM vs AWS AMD64) | Docker builds happen on CodeBuild (linux/amd64 natively) |

---

## Key Source Files

| File | Role |
|------|------|
| `src/retraining/retrain_trigger.py` | Lambda handler — orchestrates all 4 pipeline stages |
| `src/monitoring/run_drift_check.py` | Drift check Processing Job — reads captures, computes drift, writes report |
| `src/ingestion/ingest_and_upload.py` | Ingestion Processing Job — fetches OHLCV, builds features, writes parquet |
| `src/training/train_xgboost.py` | Training Job — trains XGBoost, saves model + baseline stats |
| `src/inference/app.py` | FastAPI server — `/ping`, `/invocations`, `/predict`, `/health` |
| `src/monitoring/drift_detector.py` | KS test + PSI drift analysis |
| `src/training/baseline_capture.py` | Computes and saves baseline feature stats to S3 |
| `infra/buildspec.yml` | CodeBuild buildspec — builds all 3 Docker images |
