# Stock Prediction MLOps — MSML605 Final Project

An end-to-end MLOps system for multi-ticker stock return prediction, featuring automated drift detection, continuous retraining, and a live monitoring dashboard deployed on AWS.

## System Architecture

```
defeatbeta-api (OHLCV data)
        │
        ▼
Ingestion (SageMaker Processing Job / local)
  └─ 30 technical features (RSI, MACD, Bollinger Bands, SMA, EMA, lags)
        │
        ▼
Training (SageMaker Training Job / local)
  └─ p50 (median) + p10/p90 (confidence bounds) XGBoost models
  └─ baseline_stats.json → drift detection reference
        │
        ▼
Inference API (SageMaker Endpoint / local FastAPI)
  └─ /predict, /predict/batch, /health, /metrics-summary
  └─ Data Capture → S3 captures/
        │
        ▼
EventBridge (daily 6 AM UTC) → Lambda Orchestrator
  └─ Drift check (KS test + PSI vs baseline)
  └─ IF drift: re-ingest → retrain → redeploy (zero-downtime blue/green)
  └─ CloudWatch alarm → SNS email alert
```

---

## Local Deployment

### 1. Environment Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -r requirements-ingestion.txt   # defeatbeta-api for data fetching
```

### 2. Fetch Data and Engineer Features

```bash
bash scripts/ingest_local.sh
# Output: data/processed/<TICKER>.parquet (30 features + target)
```

Override defaults:
```bash
TICKERS="AAPL MSFT GOOGL" LOOKBACK_DAYS=365 bash scripts/ingest_local.sh
```

### 3. Train XGBoost Models

```bash
bash scripts/train_local.sh
# Output: models/latest/model.xgb, model_p10.xgb, model_p90.xgb, scaler.pkl
#         output/metrics.json
#         models/latest/baseline_stats.json
```

### 4. Run Baseline Comparison

```bash
python scripts/run_baseline_comparison.py
# Compares Naive Mean, Linear Regression, Ridge vs XGBoost
# Uses same walk-forward CV and 90/10 test split as training
```

### 5. Run ARIMA SOTA Comparison

```bash
python scripts/run_prophet_comparison.py --ticker AAPL
```

### 6. Start Inference Server

```bash
bash scripts/run_local.sh
# API: http://127.0.0.1:8000
# Docs: http://127.0.0.1:8000/docs
```

Test:
```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","features":[0.1,0.2,-0.1,0.3,0.5,0.4,0.6,-0.2,0.1,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.1,0.2,0.3,0.4,0.5]}'
```

### 7. Run the Monitoring Dashboard (server must be running)

```bash
streamlit run dashboard/app.py
# Dashboard: http://localhost:8501
```

The dashboard connects to the inference server at `http://localhost:8000` by default and auto-refreshes every 60 seconds. It has three tabs:

- **Live Predictions** — actual vs predicted closing price with p10–p90 confidence band for any ticker and date range
- **Drift Metrics** — per-feature PSI bar chart, drift trend over time, and full KS/PSI feature table
- **System Metrics** — request count, avg/p95 latency, uptime, and retraining event history

Override defaults via environment variables:

```bash
API_URL=http://localhost:8000 \
DRIFT_REPORT_PATH=data/baseline/latest_drift_report.json \
streamlit run dashboard/app.py
```

### 8. Run Load Tests (server must be running)

```bash
# Baseline — 5 users, 60 seconds
locust -f tests/load/locustfile.py --host http://localhost:8000 \
  --headless -u 5 -r 1 --run-time 60s --html output/load_baseline.html

# Stress ramp — 5 → 50 users over 2 minutes
locust -f tests/load/locustfile.py --host http://localhost:8000 \
  --headless -u 50 -r 5 --run-time 120s --html output/load_stress.html

# Spike — 100 concurrent users, 30 seconds
locust -f tests/load/locustfile.py --host http://localhost:8000 \
  --headless -u 100 -r 100 --run-time 30s --html output/load_spike.html

# Soak — 20 users sustained for 10 minutes
locust -f tests/load/locustfile.py --host http://localhost:8000 \
  --headless -u 20 -r 2 --run-time 600s --html output/load_soak.html
```

Performance targets: p95 latency < 200 ms, error rate < 1%, throughput > 100 RPS.

---

## AWS Deployment

### Prerequisites
- AWS CLI configured (`aws configure`)
- `.env` file created from `.env.example`
- Scripts made executable: `chmod +x scripts/*.sh`

See [AWS Deployment guide.md](AWS%20Deployment%20guide.md) for IAM permissions required.

### Step 1 — Bootstrap Infrastructure (~2 min)
```bash
bash scripts/01_bootstrap.sh
# Creates: S3 bucket, ECR repos, IAM roles, CodeBuild project
```

### Step 2 — Build Docker Images (~10-15 min)
```bash
bash scripts/02_build.sh
# Zips source, uploads to S3, triggers CodeBuild, polls until SUCCEEDED
```

### Step 3 — Initial Data Ingestion (~5-10 min)
```bash
bash scripts/03_initial_ingest.sh
# Runs SageMaker Processing Job → data/features written to s3://<bucket>/processed/
```

### Step 4 — Initial Model Training (~15-20 min)
```bash
bash scripts/04_initial_train.sh
# Runs SageMaker Training Job → model.tar.gz + baseline_stats.json to S3
```

### Step 5 — Deploy Inference Endpoint (~10 min)
```bash
bash scripts/05_deploy.sh
# Creates SageMaker endpoint with Data Capture enabled (captures → s3://<bucket>/captures/)
```

Test:
```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name stock-prediction-endpoint \
  --content-type application/json \
  --body '{"ticker":"AAPL","features":[0.5,0.3,...]}' \
  /tmp/response.json && cat /tmp/response.json
```

### Step 6 — Set Up Automated Retraining (~2 min)
```bash
bash scripts/06_setup_automation.sh
# Creates: SNS alerts, Lambda orchestrator, EventBridge rules, CloudWatch alarm
# Check email inbox to confirm SNS subscription
```

After Step 6, the pipeline runs automatically every day at 6 AM UTC:
```
EventBridge → drift check → (if drift) ingest → train → redeploy endpoint
```

### Step 7 — Run Load Tests Against AWS Endpoint

The SageMaker endpoint requires AWS Signature V4 authentication, so Locust cannot target it directly. Instead, start the local proxy (`dashboard/proxy.py`) which forwards requests to SageMaker via boto3, then run Locust against the proxy.

```bash
# Terminal 1 — start the SageMaker proxy (requires AWS CLI configured + endpoint InService)
PYTHONPATH=$(pwd) uvicorn dashboard.proxy:app --host 127.0.0.1 --port 8000 --workers 4

# Verify the proxy reaches the endpoint
curl http://127.0.0.1:8000/health
```

```bash
# Terminal 2 — run load tests (same commands as local, traffic goes to SageMaker)

# Baseline — 5 users, 60 seconds
locust -f tests/load/locustfile.py --host http://localhost:8000 \
  --headless -u 5 -r 1 --run-time 60s --html output/load_aws_baseline.html

# Stress ramp — 5 → 50 users over 2 minutes
locust -f tests/load/locustfile.py --host http://localhost:8000 \
  --headless -u 50 -r 5 --run-time 120s --html output/load_aws_stress.html

# Spike — 100 concurrent users, 30 seconds
locust -f tests/load/locustfile.py --host http://localhost:8000 \
  --headless -u 100 -r 100 --run-time 30s --html output/load_aws_spike.html

# Soak — 20 users sustained for 10 minutes
locust -f tests/load/locustfile.py --host http://localhost:8000 \
  --headless -u 20 -r 2 --run-time 600s --html output/load_aws_soak.html
```

Performance targets: p95 latency < 200 ms, error rate < 1%, throughput > 100 RPS.

AWS latency will be higher than local due to the SageMaker network round-trip (~20–80 ms additional), but the endpoint scales horizontally under load whereas the local server is single-instance.

### Step 8 — Run the Monitoring Dashboard Against AWS

The dashboard points at whatever `API_URL` serves the `/predict` and `/metrics-summary` endpoints. For AWS, keep the proxy from Step 7 running and launch the dashboard in a second terminal:

```bash
# Terminal 1 — proxy must already be running (from Step 7)
PYTHONPATH=$(pwd) uvicorn dashboard.proxy:app --host 127.0.0.1 --port 8000 --workers 4

# Terminal 2 — launch dashboard
S3_BUCKET_NAME=<your-bucket-name> \
API_URL=http://localhost:8000 \
streamlit run dashboard/app.py
# Dashboard: http://localhost:8501
```

Setting `S3_BUCKET_NAME` enables the dashboard to:
- Pull the latest processed parquet files for each ticker directly from S3
- Download the latest drift report from `s3://<bucket>/monitoring/latest_drift_report.json`
- Show retraining events pulled from CloudWatch Lambda logs

Without `S3_BUCKET_NAME` the dashboard falls back to local files in `data/` (useful for offline inspection).

### Useful AWS Commands

```bash
# Check endpoint status
aws sagemaker describe-endpoint \
  --endpoint-name stock-prediction-endpoint \
  --query '[EndpointStatus, CreationTime]'

# Manually trigger drift check
aws lambda invoke \
  --function-name stock-prediction-orchestrator \
  --payload '{"source":"scheduled"}' \
  --cli-binary-format raw-in-base64-out /tmp/out.json && cat /tmp/out.json

# Check latest drift report
aws s3 cp s3://$S3_BUCKET_NAME/monitoring/latest_drift_report.json - | python3 -m json.tool

# Follow Lambda logs
aws logs tail /aws/lambda/stock-prediction-orchestrator --follow
```

---

## Project Structure

```
src/
  ingestion/        Data fetching and feature engineering
  features/         Technical indicators (RSI, MACD, Bollinger Bands, etc.)
  training/         XGBoost training, baselines, evaluation, walk-forward CV
  inference/        FastAPI serving app and predictor
  monitoring/       Drift detection (KS test + PSI) and CloudWatch logging
  retraining/       Lambda orchestrator for automated retraining pipeline
scripts/
  ingest_local.sh               Fetch data locally → data/processed/
  train_local.sh                Train models locally → models/latest/
  run_local.sh                  Start inference server locally
  run_baseline_comparison.py    Baseline model evaluation
  run_prophet_comparison.py     ARIMA SOTA comparison
  01_bootstrap.sh               AWS infrastructure setup
  02_build.sh                   Build and push Docker images
  03_initial_ingest.sh          AWS data ingestion
  04_initial_train.sh           AWS model training
  05_deploy.sh                  Deploy SageMaker endpoint
  06_setup_automation.sh        EventBridge + Lambda + CloudWatch
tests/
  unit/             Unit tests
  integration/      API endpoint tests
  load/             Locust load test scenarios (3 user profiles)
configs/
  model_config.yaml    XGBoost hyperparameters and Optuna search space
  feature_config.yaml  Feature engineering parameters
  aws_config.yaml      AWS resource names and instance types
infra/
  docker/           Dockerfiles for ingestion, training, inference, dashboard
  lambda/           Lambda function entry point
dashboard/
  app.py            Streamlit monitoring dashboard (3 tabs: predictions, drift, system)
  proxy.py          FastAPI proxy — forwards requests to SageMaker for load tests and dashboard
  components/       Reusable Plotly chart components (prediction_chart, drift_panel, metrics_panel)
```

---

## Tools and Libraries

| Category | Library |
|---|---|
| ML | XGBoost 2.1, scikit-learn 1.5, Optuna 3.6 |
| SOTA Comparison | ARIMA (statsmodels 0.14) |
| Data | pandas 2.2, numpy 1.26, PyArrow 1.6 |
| Feature Engineering | scipy 1.13 |
| Inference API | FastAPI 0.111, uvicorn 0.30, Pydantic 2.7 |
| Monitoring | Prometheus, CloudWatch |
| Dashboard | Streamlit 1.35, Plotly 5.22 |
| Load Testing | Locust 2.29 |
| AWS | boto3 1.34, SageMaker SDK 2.220 |
| Drift Detection | KS test (scipy), PSI (custom) |
