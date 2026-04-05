# Containerized Real-Time Stock Prediction System
## MSML605 Project — Full Implementation Plan

---

## Context

Building a complete MLOps pipeline from scratch for a UMD graduate course project.
The system ingests real-time stock data, trains XGBoost models with time-series feature
engineering, deploys them as containerized REST APIs on AWS, monitors production for
data drift (KS test + PSI), and automatically retrains when drift is detected.
The user (Sarthak Garg) owns the ML Lead role: feature engineering, model training,
baseline capture, and evaluation metrics.

---

## Repository Structure

```
stock-prediction-mlops/
├── .github/workflows/
│   ├── ci.yml                     # lint + test + Docker build on PR
│   └── cd.yml                     # push to ECR + SSH deploy to EC2 on main
├── data/
│   ├── raw/                       # gitignored downloaded CSVs
│   ├── processed/                 # feature-engineered parquet files
│   └── baseline/baseline_stats.json
├── src/
│   ├── ingestion/
│   │   ├── fetch_yahoo.py         # yfinance downloader
│   │   ├── fetch_alphavantage.py  # Alpha Vantage fallback
│   │   └── data_validator.py      # schema + null checks
│   ├── features/
│   │   ├── technical_indicators.py  # RSI, SMA, EMA, Bollinger, Volume, VWAP
│   │   ├── feature_pipeline.py      # orchestrates full feature build + scaling
│   │   └── feature_store.py         # S3 read/write helpers
│   ├── training/
│   │   ├── train_xgboost.py         # XGBoost training entry point
│   │   ├── train_lstm.py            # optional PyTorch LSTM variant
│   │   ├── hyperparameter_tuning.py # Optuna HPO
│   │   ├── time_series_split.py     # walk-forward CV (no shuffle leakage)
│   │   ├── evaluate.py              # RMSE, MAE, MAPE, directional accuracy
│   │   └── baseline_capture.py      # saves training distribution stats for drift
│   ├── inference/
│   │   ├── app.py                   # FastAPI application
│   │   ├── predictor.py             # model load + predict + confidence interval
│   │   └── schemas.py               # Pydantic request/response validation
│   ├── monitoring/
│   │   ├── drift_detector.py        # KS test + PSI per feature
│   │   ├── cloudwatch_logger.py     # push drift metrics to CloudWatch
│   │   └── sagemaker_monitor.py     # SageMaker Model Monitor config
│   └── retraining/
│       └── retrain_trigger.py       # Lambda handler logic
├── dashboard/
│   ├── app.py                       # Streamlit entry point (3 tabs)
│   └── components/
│       ├── prediction_chart.py      # actual vs predicted + confidence band
│       ├── drift_panel.py           # PSI bar chart + KS heatmap
│       └── metrics_panel.py         # system metrics from CloudWatch
├── infra/
│   ├── docker/
│   │   ├── Dockerfile.inference     # multi-stage, uvicorn, port 8000
│   │   ├── Dockerfile.training      # SageMaker-compatible entry point
│   │   └── Dockerfile.dashboard     # Streamlit container
│   ├── docker-compose.yml           # local dev: inference + dashboard + monitoring
│   ├── ec2_setup.sh                 # EC2 user-data bootstrap
│   └── lambda/retrain_handler.py    # AWS Lambda deployment package
├── tests/
│   ├── unit/
│   │   ├── test_indicators.py
│   │   ├── test_drift_detector.py
│   │   └── test_predictor.py
│   ├── integration/test_api_endpoints.py
│   └── load/locustfile.py           # 4 Locust scenarios
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_drift_analysis.ipynb
├── configs/
│   ├── model_config.yaml            # XGBoost hyperparameter search space
│   ├── feature_config.yaml          # window sizes, tickers list
│   └── aws_config.yaml              # S3 bucket names, region, ECR URI
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Component Implementation Details

### 1. Data Ingestion (`src/ingestion/`)

**`fetch_yahoo.py`**
- `fetch_ohlcv(ticker, start, end, interval) -> pd.DataFrame`
- Uses `yfinance.download()` with `auto_adjust=True`
- Intervals: `1d` for training, `1h` for live drift monitoring
- Fallback: `fetch_alphavantage.py` using `TIME_SERIES_DAILY_ADJUSTED` endpoint
  with `tenacity` exponential backoff (5 calls/min rate limit)
- Raw JSON stored to `s3://bucket/raw/{ticker}/{date}.json` before parsing

---

### 2. Feature Engineering (`src/features/technical_indicators.py`) — ML Lead

**23 total features, all functions operate on OHLCV DataFrame:**

```
compute_rsi(close, period=14)            -> RSI 0-100 via Wilder's smoothing
compute_sma(close, windows=[5,10,20,50]) -> sma_5, sma_10, sma_20, sma_50
compute_ema(close, windows=[12,26])      -> ema_12, ema_26 (adjust=False)
compute_bollinger_bands(close, window=20) -> bb_upper, bb_lower, bb_width, bb_pct_b
compute_volume_features(ohlcv)           -> volume_sma_20, volume_ratio, OBV, VWAP
compute_price_features(ohlcv)            -> returns_1d, returns_5d, hl_range, gap
build_target(close, horizon=1)           -> forward_return (regression) or direction (0/1)
```

**`feature_pipeline.py`** — `build_feature_matrix(ticker, start, end, config_path)`
1. Fetch OHLCV -> call all compute_* functions -> concatenate
2. Drop first 50 rows (NaN from 50-day SMA)
3. Fit `RobustScaler` on train split only, transform all splits
4. Persist `scaler.pkl` to S3 alongside model
5. Save parquet to `s3://bucket/processed/{ticker}/features_{date}.parquet`

---

### 3. Model Training (`src/training/`) — ML Lead

**`time_series_split.py`** — walk-forward CV with `TimeSeriesSplit(n_splits=3, gap=5)`
- 5-day gap prevents lookahead from lag features
- Hold out final 10% strictly for production simulation

**`train_xgboost.py`** — `train(X_train, y_train, X_val, y_val, params) -> xgb.Booster`
- Starting params: `objective=reg:squarederror`, `n_estimators=500`, `lr=0.05`,
  `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `early_stopping_rounds=30`
- Log feature importances via `booster.get_score(importance_type='gain')`
- Save `model.xgb` (portable binary) to `s3://bucket/models/{version}/`
- Also train p10/p90 quantile models for confidence intervals

**`hyperparameter_tuning.py`** — Optuna with TPESampler + MedianPruner
- Search: `lr` log-uniform(0.005, 0.3), `max_depth` int[3,9], `subsample` [0.5,1.0]
- 50 trials; integrates with XGBoost callbacks for mid-trial pruning

**`evaluate.py`** — `compute_metrics(y_true, y_pred) -> dict`
- RMSE, MAE, MAPE, directional_accuracy, Sharpe proxy
- Directional accuracy and Sharpe are primary domain metrics for the report

**`baseline_capture.py`** — `capture_baseline_stats(X_train, output_path) -> dict`
- Per feature: mean, std, min, max, p5/p25/p50/p75/p95, histogram (20 bins)
- Saved to `s3://bucket/baseline/baseline_stats.json`
- Must regenerate on every retrain — bridges to SageMaker Model Monitor

---

### 4. Inference API (`src/inference/`)

**`app.py`** — FastAPI endpoints:
- `POST /predict` — single prediction with confidence interval (p10, p50, p90)
- `POST /predict/batch` — batch predictions
- `GET /health` — model loaded status + version
- `GET /metrics` — Prometheus-format metrics via `prometheus_fastapi_instrumentator`
- Model loaded at startup via `@app.on_event("startup")`, never per-request
- Inference payloads logged to in-memory ring buffer (last 1000 requests)
  periodically flushed to `s3://bucket/monitoring/live_window_{ts}.parquet`

**`predictor.py`** — `StockPredictor` class
- Loads `.xgb` + `scaler.pkl` from S3 on init
- Applies same scaler as training, returns (p10, p50, p90) predictions

**`schemas.py`** — Pydantic validators: exactly 23 finite floats, valid ticker regex

---

### 5. Drift Detection (`src/monitoring/drift_detector.py`)

**KS Test:**
```
run_ks_test(baseline, live, alpha=0.05) -> DriftResult
    scipy.stats.ks_2samp(baseline, live)
    Flag overall drift if >30% of features show p < alpha
```

**PSI:**
```
compute_psi(baseline, current, n_bins=10) -> float
    Bin edges from baseline percentiles, clip to 1e-6
    Thresholds: <0.1 stable, 0.1-0.2 investigate, >0.2 retrain
```

**`analyze_drift(baseline_path, live_window_df, n_live=500) -> DriftReport`**
- Runs both KS and PSI per feature
- Returns: `{feature_results, overall_drift_detected, trigger_retraining, timestamp}`

**`cloudwatch_logger.py`** — pushes per-feature KS/PSI + `Overall_Drift_Score` to
namespace `StockPredictor/DriftMetrics`. CloudWatch Alarm: `Overall_Drift_Score > 0.2`
triggers SNS -> Lambda retraining.

---

### 6. Automated Retraining (`infra/lambda/retrain_handler.py`)

- Triggered by CloudWatch Alarm (drift) -> SNS or EventBridge schedule (weekly)
- Launches `sm_client.create_training_job()` with ECR training image
- SageMaker reads from `s3://bucket/processed/`, writes to `s3://bucket/models/`
- Second Lambda (on `TrainingJobStatusChanged`) calls `update_endpoint()` + regenerates baseline

---

### 7. Streamlit Dashboard (`dashboard/`)

Three tabs:
- **Tab 1 — Live Predictions:** actual vs predicted line chart + p10/p90 shaded band,
  RMSE/MAE metric cards, 60s auto-refresh
- **Tab 2 — Drift Metrics:** per-feature PSI bar chart (green/yellow/red), KS heatmap
  (features x time), 30-day drift score trend, red banner on drift detection
- **Tab 3 — System Metrics:** CloudWatch API throughput, p95 latency, EC2 CPU, retraining event table

---

### 8. AWS Deployment

**`Dockerfile.inference`** — multi-stage python:3.11-slim, uvicorn 2 workers, port 8000

**`Dockerfile.training`** — SageMaker-compatible: reads `/opt/ml/input/data/train/`,
writes to `/opt/ml/model/`, entrypoint is `train_xgboost.py`

**`docker-compose.yml`** — three services: `inference-api`, `dashboard`, `monitoring-agent`
(runs drift checks on cron), Docker bridge network, only API+dashboard ports exposed

**EC2:** `ec2_setup.sh` runs on User Data — installs Docker, pulls ECR image, starts compose

**SageMaker:** `ml.t3.medium` real-time endpoint, `DataCaptureConfig(sampling_percentage=100)`,
`DefaultModelMonitor` with hourly schedule and S3 baseline from `baseline_stats.json`

---

### 9. CI/CD (GitHub Actions)

**`ci.yml`** — on PR to main:
1. `ruff check src/ dashboard/` — linting
2. `pytest tests/unit/ --cov=src` — unit tests
3. `docker build Dockerfile.inference` — image build test
4. `pytest tests/integration/` — API integration tests against container

**`cd.yml`** — on push to main:
1. Configure AWS credentials from secrets
2. Build + push inference image to ECR with `$GITHUB_SHA` and `latest` tags
3. SSH to EC2: `docker pull latest && docker-compose up -d --no-deps inference-api`

Secrets needed: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `ECR_URI`, `EC2_HOST`, `EC2_SSH_KEY`

---

### 10. Load Testing (`tests/load/locustfile.py`)

```python
class StockPredictorUser(HttpUser):
    wait_time = between(0.1, 0.5)
    # POST /predict         weight 10
    # POST /predict/batch   weight 2  (10 records)
    # GET  /health          weight 1
```

Four scenarios:
1. **Baseline** — 5 users, 60s -> establish p50/p95/p99 latency baseline
2. **Stress ramp** — 5->50 users over 120s -> find p95 > 200ms threshold
3. **Spike** — 100 users instant, 30s -> measure error rate under burst
4. **Soak** — 20 users, 600s -> detect memory leaks or latency degradation

---

## Key Python Packages

**requirements.txt:**
```
yfinance==0.2.54
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
xgboost==2.1.0
optuna==3.6.1
scipy==1.13.1
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.4
boto3==1.34.120
sagemaker==2.220.0
streamlit==1.35.0
plotly==5.22.0
tenacity==8.3.0
prometheus-fastapi-instrumentator==7.0.0
pyarrow==16.1.0
```

**requirements-dev.txt:**
```
pytest==8.2.2
pytest-cov==5.0.0
pytest-asyncio==0.23.7
httpx==0.27.0
ruff==0.4.9
locust==2.29.0
```

---

## Data Flow

```
Yahoo Finance API
  -> fetch_yahoo.py -> S3 raw/
  -> feature_pipeline.py -> baseline_capture.py -> S3 baseline_stats.json
  -> S3 processed/features.parquet
  -> train_xgboost.py + hyperparameter_tuning.py
  -> evaluate.py -> CloudWatch (RMSE/MAE)
  -> S3 models/v{N}/model.xgb + scaler.pkl
  -> FastAPI inference endpoint (EC2 Docker / SageMaker)
    -> ring buffer -> S3 monitoring/live_window.parquet
    -> drift_detector.py (KS + PSI)
    -> cloudwatch_logger.py -> CloudWatch Alarm
    -> SNS -> Lambda retrain_handler.py
    -> SageMaker Training Job -> updated model -> loop
  -> dashboard/app.py reads S3 + CloudWatch -> Streamlit UI
```

---

## Sprint Plan

| Week | Sarthak (ML Lead) | Dhanvin (Infra) | Basil (MLOps) |
|------|-------------------|-----------------|----------------|
| 1 | fetch_yahoo, technical_indicators, feature_pipeline + unit tests | Docker setup, docker-compose, AWS IAM/S3 | Streamlit skeleton |
| 2 | train_xgboost, time_series_split, evaluate, baseline_capture | Dockerfile.inference, Dockerfile.training, ECR | FastAPI app.py, predictor.py |
| 3 | hyperparameter_tuning, SageMaker endpoint | Lambda, CloudWatch alarms, EC2 deploy, cd.yml | drift_detector, cloudwatch_logger, Streamlit live data |
| 4 | Final eval on held-out test, metrics write-up | ci.yml, scalability experiments | Drift injection experiment, alarms validation |

---

## Verification

1. **Unit tests:** `pytest tests/unit/` — all indicator functions, KS/PSI, predictor
2. **Integration test:** `docker-compose up` -> hit `POST /predict` -> check response schema
3. **Drift test:** replace live_window with OOD data -> verify CloudWatch alarm fires + Lambda triggers SageMaker job
4. **Load test:** `locust -f tests/load/locustfile.py --host http://localhost:8000 --headless -u 50 -r 5 --run-time 120s --html report.html`
5. **End-to-end:** run full pipeline: fetch -> feature -> train -> deploy -> monitor -> drift trigger -> retrain
