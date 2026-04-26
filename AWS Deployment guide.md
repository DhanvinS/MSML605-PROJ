# Deployment Guide — Stock Prediction MLOps on AWS

All computation (Docker builds, ingestion, training, inference, drift detection) runs on AWS.  
Your local machine only runs bash scripts that call the AWS CLI.

---

## Prerequisites

### 1. Install AWS CLI
```bash
brew install awscli        # macOS
# or: pip install awscli
aws --version              # should show 2.x
```

### 2. Configure AWS credentials
```bash
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1), Output (json)
aws sts get-caller-identity   # verify — shows your account ID
```

### 3. Set up .env file
Create `.env` in the project root (copy from `.env.example`):
```bash
# .env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=<your access key>
AWS_SECRET_ACCESS_KEY=<your secret key>
S3_BUCKET_NAME=stock-prediction-mlops-<your-account-id>   # must be globally unique
ALERT_EMAIL=your@email.com
TICKERS=AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA
LOOKBACK_DAYS=365
```

> Never commit `.env` to git — it is already in `.gitignore`.

### 4. Make scripts executable
```bash
chmod +x scripts/*.sh
```

---

## AWS IAM Permissions Required

Your AWS user/role must have these permissions to run the scripts:

| Permission | Why |
|------------|-----|
| `iam:CreateRole`, `iam:AttachRolePolicy` | Bootstrap creates service roles |
| `s3:CreateBucket`, `s3:PutObject`, `s3:GetObject` | Data + model storage |
| `ecr:CreateRepository`, `ecr:PutImage` | Container image storage |
| `codebuild:CreateProject`, `codebuild:StartBuild` | Docker builds on AWS |
| `sagemaker:*` | Training jobs, Processing jobs, Endpoints |
| `lambda:CreateFunction`, `lambda:UpdateFunctionCode` | Orchestration function |
| `events:PutRule`, `events:PutTargets` | Scheduling rules |
| `cloudwatch:PutMetricAlarm` | Drift alarm |
| `sns:CreateTopic`, `sns:Subscribe` | Email alerts |

**Quickest option:** Attach `AdministratorAccess` to your IAM user for initial deployment, then lock down after.

---

## Step-by-Step Deployment

### STEP 0 — Delete old AWS resources (skip if fresh account)
```bash
bash scripts/00_cleanup.sh
```
Deletes: SageMaker endpoint, Lambda, EventBridge rules, CloudWatch alarms,  
SNS topic, IAM roles, ECR repos, CodeBuild project.  
Prompts before wiping S3 contents.

Expected output:
```
[1/8] SageMaker endpoint... Endpoint not found, skipping.
...
[8/8] S3 bucket contents... Bucket not found, skipping.
Cleanup complete.
```

---

### STEP 1 — Bootstrap infrastructure (~2 min)
```bash
bash scripts/01_bootstrap.sh
```
Creates:
- S3 bucket with versioning
- 3 ECR repositories: `stock-prediction/{ingestion,training,inference}`
- 3 IAM roles: sagemaker, lambda, codebuild
- CodeBuild project: `stock-prediction-build`

Expected output:
```
[1/5] S3 bucket: stock-prediction-mlops-...  Bucket created.
[2/5] ECR repositories... Created: stock-prediction/ingestion ...
[3/5] IAM roles... SageMaker role ready. Lambda role ready.
[5/5] CodeBuild project created.
Bootstrap complete.
SageMaker role ARN: arn:aws:iam::740291410880:role/stock-prediction-sagemaker-role
```

---

### STEP 2 — Build Docker images on AWS (~10-15 min)
```bash
bash scripts/02_build.sh
```
What it does:
1. Zips the project (excludes `.git/`, `data/`, `models/`, `.venv/`)
2. Uploads zip to `s3://<bucket>/codebuild/source.zip`
3. Starts a CodeBuild build
4. Polls until SUCCEEDED

CodeBuild runs on `linux/amd64` natively — builds 3 images with plain `docker build`  
(no `buildx`, no OCI manifest issues with SageMaker).

Expected output:
```
[1/3] Creating source zip... (2.1M)
[2/3] Uploading to S3...
[3/3] Starting CodeBuild build... Build ID: stock-prediction-build:abc123
  14:22:01 Status: IN_PROGRESS  Phase: BUILD
  14:22:21 Status: IN_PROGRESS  Phase: POST_BUILD
  14:24:45 Status: SUCCEEDED    Phase: COMPLETED
Build succeeded! Images pushed to ECR.
```

**If build fails:** Check logs at the URL printed in the output (CodeBuild console).

---

### STEP 3 — Initial data ingestion (~5-10 min)
```bash
bash scripts/03_initial_ingest.sh
```
Starts a SageMaker Processing Job using the `ingestion` image:
- Fetches 730 days of OHLCV data for AAPL, MSFT, GOOGL via `defeatbeta-api`
- Computes technical features (RSI, MACD, Bollinger, SMA, EMA, lags)
- Writes parquet files to `s3://<bucket>/processed/`

Expected output:
```
Job name: stock-ingest-initial-20240401-142500
  14:25:30 Status: InProgress
  14:28:00 Status: Completed
Processed features uploaded to: s3://.../processed/
```

Verify:
```bash
aws s3 ls s3://$S3_BUCKET_NAME/processed/
# AAPL.parquet  MSFT.parquet  GOOGL.parquet
```

---

### STEP 4 — Initial model training (~15-20 min)
```bash
bash scripts/04_initial_train.sh
```
Starts a SageMaker Training Job using the `training` image:
- Reads parquets from `s3://<bucket>/processed/`
- Applies RobustScaler, runs walk-forward CV
- Trains 3 XGBoost models: p50 (median), p10 (lower bound), p90 (upper bound)
- Saves `model.xgb`, `model_p10.xgb`, `model_p90.xgb`, `scaler.pkl`, `version.txt`
- Uploads `baseline_stats.json` to `s3://<bucket>/baseline/` (used for drift detection)
- All artifacts saved to `s3://<bucket>/models/<job-name>/output/model.tar.gz`

Expected output:
```
Job name: stock-retrain-initial-20240401-150000
  15:00:30 Status: InProgress
  15:16:45 Status: Completed
Model artifact: s3://.../models/stock-retrain-initial-.../output/model.tar.gz
Baseline stats: s3://.../baseline/baseline_stats.json
```

Verify:
```bash
aws s3 ls s3://$S3_BUCKET_NAME/baseline/
# baseline_stats.json
```

---

### STEP 5 — Deploy inference endpoint (~10 min)
```bash
bash scripts/05_deploy.sh
```
1. Gets model artifact from the last completed training job
2. Creates a SageMaker Model (inference image + model.tar.gz)
3. Creates an Endpoint Config with **Data Capture** enabled:
   - Captures every inference request (input features) to `s3://<bucket>/captures/`
   - This captured data is used daily for drift detection
4. Creates the endpoint `stock-prediction-endpoint` on `ml.t2.medium`
5. Waits until InService

Expected output:
```
[1/4] Creating SageMaker model: stock-model-20240401-151000
[2/4] Creating endpoint config with Data Capture → s3://.../captures/
[3/4] Creating endpoint: stock-prediction-endpoint
[4/4] Waiting for InService...
  15:10:30 Status: Creating
  15:18:00 Status: InService
Endpoint is InService.
```

Test it:
```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name stock-prediction-endpoint \
  --content-type application/json \
  --body '{"ticker":"AAPL","features":[0.5,0.3,-0.2,1.1,0.8,0.4,0.6,1.2,-0.1,0.9,0.7,0.3,0.5,1.0,0.2,0.8,0.4,0.6,1.1,0.3,0.7,0.9,0.5]}' \
  /tmp/response.json
cat /tmp/response.json
# {"ticker":"AAPL","prediction":0.012345,"lower_bound":0.008,"upper_bound":0.018,...}
```

> Note: The 23 feature values should be your actual scaled features. Use the feature  
> pipeline to generate them: `src/features/feature_pipeline.py → build_feature_matrix()`

---

### STEP 6 — Set up automation (~2 min)
```bash
bash scripts/06_setup_automation.sh
```
Creates:
1. **SNS topic** `stock-prediction-alerts` + email subscription
2. **Lambda function** `stock-prediction-orchestrator` with all env vars
3. **EventBridge rule 1**: `cron(0 6 * * ? *)` → Lambda `{source: scheduled}` (daily 6 AM UTC)
4. **EventBridge rule 2**: SageMaker Processing Job Completed → Lambda (routing by job name prefix)
5. **EventBridge rule 3**: SageMaker Training Job Completed → Lambda (update endpoint)
6. **CloudWatch alarm**: `Max_PSI > 0.2` → SNS email alert

> Check your email inbox and confirm the SNS subscription.

Expected output:
```
[1/5] SNS topic... arn:aws:sns:us-east-1:...:stock-prediction-alerts
  Subscription sent to your@email.com (check your inbox to confirm)
[2/5] Packaging Lambda...
[3/5] Lambda function created.
  Lambda environment configured.
[4/5] EventBridge rules... 3 rules created.
[5/5] CloudWatch alarm created.
Automation setup complete.
```

---

## Verification

### Test the endpoint
```bash
# Quick smoke test
echo '{"ticker":"AAPL","features":[0.5,0.3,-0.2,1.1,0.8,0.4,0.6,1.2,-0.1,0.9,0.7,0.3,0.5,1.0,0.2,0.8,0.4,0.6,1.1,0.3,0.7,0.9,0.5,0.1,-0.3,0.4,0.8,-0.1,0.6,0.2]}' > /tmp/body.json

aws sagemaker-runtime invoke-endpoint \
  --endpoint-name stock-prediction-endpoint \
  --content-type application/json \
  --body fileb:///tmp/body.json \
  /tmp/response.json && cat /tmp/response.json

```

### Manually trigger drift check
```bash
aws lambda invoke \
  --function-name stock-prediction-orchestrator \
  --payload '{"source":"scheduled"}' \
  --cli-binary-format raw-in-base64-out \
  /tmp/out.json && cat /tmp/out.json
```
Expected: `{"status": "drift_check_launched", "processing_job_name": "drift-check-..."}`

### Manually trigger full retraining (skip drift check)
```bash
aws lambda invoke \
  --function-name stock-prediction-orchestrator \
  --payload '{"source":"ingestion_complete"}' \
  --cli-binary-format raw-in-base64-out \
  /tmp/out.json && cat /tmp/out.json
```

### Check drift report (after drift check runs)
```bash
aws s3 cp s3://$S3_BUCKET_NAME/monitoring/latest_drift_report.json /tmp/report.json
cat /tmp/report.json
```

### Monitor CloudWatch metrics
```bash
aws cloudwatch get-metric-statistics \
  --namespace StockPredictor/DriftMetrics \
  --metric-name Max_PSI \
  --dimensions Name=Ticker,Value=ALL \
  --start-time $(date -u -v-7d +%Y-%m-%dT%H:%M:%SZ) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --period 86400 \
  --statistics Maximum
```

### Check Lambda logs
```bash
aws logs tail /aws/lambda/stock-prediction-orchestrator --follow
```

---

## Automated Retraining Flow

Once all 6 steps are complete, this runs automatically every day:

```
6 AM UTC
  └─ EventBridge → Lambda {source: scheduled}
       └─ SageMaker Processing Job: drift-check-YYYYMMDD-HHMMSS
            • Reads s3://.../captures/ (all captured inferences)
            • Computes PSI + KS test vs baseline_stats.json
            • Writes s3://.../monitoring/latest_drift_report.json
            • Pushes Max_PSI to CloudWatch

  Processing Job Completes
  └─ EventBridge → Lambda {source: drift_check_complete}
       • Reads latest_drift_report.json
       IF trigger_retraining = true (max_PSI > 0.2 OR > 30% features drifted):
         └─ SageMaker Processing Job: stock-ingest-YYYYMMDD-HHMMSS
              • Fetches fresh OHLCV data via defeatbeta-api
              • Rebuilds feature matrices → S3 processed/

  Ingestion Job Completes
  └─ EventBridge → Lambda {source: ingestion_complete}
       └─ SageMaker Training Job: stock-retrain-YYYYMMDD-HHMMSS
            • Trains new p50/p10/p90 XGBoost models
            • Saves new baseline_stats.json
            • Saves model.tar.gz to S3

  Training Job Completes
  └─ EventBridge → Lambda {source: training_complete}
       └─ Creates new SageMaker Model + Endpoint Config
          Updates stock-prediction-endpoint (zero-downtime blue/green)
          SNS email: "Model retrained and endpoint updated"
```

---

## Useful Commands

### Check SageMaker jobs
```bash
# Recent training jobs
aws sagemaker list-training-jobs \
  --name-contains stock-retrain \
  --sort-by CreationTime --sort-order Descending \
  --max-results 5

# Recent processing jobs
aws sagemaker list-processing-jobs \
  --name-contains stock \
  --sort-by CreationTime --sort-order Descending \
  --max-results 5

# Endpoint status
aws sagemaker describe-endpoint \
  --endpoint-name stock-prediction-endpoint \
  --query '[EndpointStatus, CreationTime]'
```

### Check S3 data
```bash
# Latest captures (for drift detection)
aws s3 ls s3://$S3_BUCKET_NAME/captures/ --recursive | tail -20

# Drift report
aws s3 cp s3://$S3_BUCKET_NAME/monitoring/latest_drift_report.json - | python3 -m json.tool
```

### Rebuild and redeploy after code changes
```bash
bash scripts/02_build.sh      # rebuild images
bash scripts/05_deploy.sh     # redeploy endpoint with new image
```

### Force a full retraining right now
```bash
bash scripts/03_initial_ingest.sh   # fetch fresh data
bash scripts/04_initial_train.sh    # train new model
bash scripts/05_deploy.sh           # redeploy endpoint
```

---

## Troubleshooting

### CodeBuild fails
```bash
# View build logs
aws logs get-log-events \
  --log-group-name /aws/codebuild/stock-prediction-build \
  --log-stream-name <build-id>/phase_context \
  --query 'events[*].message' --output text
```

### SageMaker job fails
```bash
# Get failure reason
aws sagemaker describe-training-job \
  --training-job-name <job-name> \
  --query '[TrainingJobStatus, FailureReason]'

# View training logs
aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix <job-name>
```

### Lambda errors
```bash
aws logs tail /aws/lambda/stock-prediction-orchestrator
```

### Endpoint fails to start
```bash
aws sagemaker describe-endpoint \
  --endpoint-name stock-prediction-endpoint \
  --query '[EndpointStatus, FailureReason]'
```
Common cause: model image doesn't have `/ping` or `/invocations` on port 8080.  
The `Dockerfile.inference` is already correctly configured.

### "No module named 'src'" inside Docker container
This is fixed by `ENV PYTHONPATH=/opt/ml/code` in `Dockerfile.training` and  
`ENV PYTHONPATH=/opt/ml/processing` in `Dockerfile.ingestion` and  
`ENV PYTHONPATH=/app` in `Dockerfile.inference`.

### Endpoint instance type error
Use exactly: `ml.t2.medium` (not `ml.t3.medium`).  
Defined in `configs/aws_config.yaml` and `scripts/05_deploy.sh`.
