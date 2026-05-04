"""Streamlit observability dashboard.

Tabs:
  1. Live Predictions — actual vs predicted + confidence band
  2. Drift Metrics    — per-feature PSI + KS heatmap + trend
  3. System Metrics   — API latency, request rate, retraining events
"""

import json
import os
from datetime import datetime, timedelta, timezone, date
from pathlib import Path

import boto3
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from components.drift_panel import (
    render_drift_banner,
    render_drift_trend,
    render_ks_heatmap,
    render_psi_bar_chart,
)
from components.metrics_panel import (
    render_latency_chart,
    render_retraining_table,
    render_system_metrics,
)
from components.prediction_chart import (
    render_metrics_cards,
    render_prediction_chart,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL = os.environ.get("API_URL", "http://localhost:8000")
DRIFT_REPORT_PATH = os.environ.get("DRIFT_REPORT_PATH", "data/baseline/latest_drift_report.json")
PREDICTIONS_LOG_PATH = os.environ.get("PREDICTIONS_LOG_PATH", "data/monitoring/predictions_log.parquet")
S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "")
REFRESH_INTERVAL_MS = 60_000  # 60 seconds

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]

st.set_page_config(
    page_title="Stock Prediction Monitor",
    page_icon="📈",
    layout="wide",
)

# Auto-refresh every 60 seconds
st_autorefresh(interval=REFRESH_INTERVAL_MS, key="dashboard_refresh")


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def fetch_health() -> dict:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.json()
    except Exception:
        return {"status": "unreachable", "model_loaded": False, "model_version": "N/A"}


@st.cache_data(ttl=60)
def fetch_api_metrics() -> dict:
    try:
        resp = requests.get(f"{API_URL}/metrics-summary", timeout=5)
        return resp.json()
    except Exception:
        return {}


@st.cache_data(ttl=300)
def fetch_processed_data(ticker: str) -> pd.DataFrame:
    """Download latest processed parquet from S3 for a ticker."""
    local_path = Path(f"data/processed/{ticker}.parquet")
    if S3_BUCKET:
        try:
            s3 = boto3.client("s3")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(S3_BUCKET, f"processed/{ticker}.parquet", str(local_path))
        except Exception as e:
            st.warning(f"Could not download {ticker} from S3: {e}")
    if local_path.exists():
        df = pd.read_parquet(local_path)
        df.index = pd.to_datetime(df.index)
        return df
    return pd.DataFrame()


@st.cache_data(ttl=300)
def _fetch_all_close_prices(ticker: str) -> dict:
    """Return {Timestamp -> close_price} for all available dates.

    Primary: raw OHLCV parquet from S3 (s3://bucket/raw/{ticker}.parquet).
    Fallback: reconstruct exact close prices from Bollinger Band features in
              the processed parquet using:
                close[T] = bb_lower[T] + bb_pct_b[T] * (bb_upper[T] - bb_lower[T])
              This identity is exact because bb_pct_b was computed from close.
    """
    # Primary: try raw parquet from S3
    if S3_BUCKET:
        try:
            s3 = boto3.client("s3")
            local = Path(f"data/raw/{ticker}.parquet")
            local.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(S3_BUCKET, f"raw/{ticker}.parquet", str(local))
            raw = pd.read_parquet(local)
            raw.index = pd.to_datetime(raw.index)
            return {ts: float(p) for ts, p in raw["Close"].items()}
        except Exception:
            pass

    # Fallback: reconstruct from processed features
    df = fetch_processed_data(ticker)
    if df.empty:
        return {}
    required = {"bb_lower", "bb_upper", "bb_pct_b"}
    if not required.issubset(df.columns):
        return {}
    # close = bb_lower + bb_pct_b * (bb_upper - bb_lower)
    close_series = df["bb_lower"] + df["bb_pct_b"] * (df["bb_upper"] - df["bb_lower"])
    return {ts: round(float(p), 4) for ts, p in close_series.items() if pd.notna(p)}


def predict_price_range(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Walk-forward backtest: for each trading day T in [start_date, end_date],
    predict the closing price for day T using features computed from data up to T-1.

    Model mechanics:
      - Processed parquet row at date T-1 has features from T-1 and target = (close[T] - close[T-1]) / close[T-1]
      - So: predicted_close[T] = close[T-1] * (1 + model_output)
      - We compare predicted_close[T] with actual_close[T]

    Result dataframe is indexed at the prediction date T (not the feature date T-1).
    """
    df = fetch_processed_data(ticker)
    if df.empty:
        return pd.DataFrame()

    feature_cols = [c for c in df.columns if c != "target"]
    all_close = _fetch_all_close_prices(ticker)

    # We need feature rows from T-1 to predict close for T in [start_date, end_date].
    # Feature row T-1 is available one business day before each prediction date T.
    # Select feature rows where next business day falls in [start_date, end_date].
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # Feature rows whose NEXT business day is in [start_date, end_date]
    # i.e., feature row date in [start_date - 1 bday, end_date - 1 bday]
    feature_start = start_ts - pd.offsets.BDay(1)
    feature_end = end_ts - pd.offsets.BDay(1)

    window = df[(df.index >= feature_start) & (df.index <= feature_end)]
    if window.empty:
        return pd.DataFrame()

    rows = []
    for feature_date, row in window.iterrows():
        features = row[feature_cols].tolist()
        close_prev = all_close.get(feature_date)  # close at T-1

        # Prediction date is the next business day after the feature row
        pred_date = feature_date + pd.offsets.BDay(1)
        actual_close = all_close.get(pred_date)  # what actually happened at T

        try:
            resp = requests.post(
                f"{API_URL}/predict",
                json={"ticker": ticker, "features": features},
                timeout=10,
            )
            result = resp.json()
            pred_return = result.get("prediction", 0.0)
            low_return = result.get("lower_bound", 0.0)
            high_return = result.get("upper_bound", 0.0)

            predicted_close = round(close_prev * (1 + pred_return), 2) if close_prev else None
            lower_price = round(close_prev * (1 + low_return), 2) if close_prev else None
            upper_price = round(close_prev * (1 + high_return), 2) if close_prev else None

            rows.append({
                "date": pred_date,             # date we're predicting FOR
                "ticker": ticker,
                "actual_close": actual_close,  # what actually happened on pred_date
                "predicted_close": predicted_close,  # model's prediction for pred_date
                "lower_price": lower_price,
                "upper_price": upper_price,
                "predicted_return": round(pred_return, 6),
                "actual_return": float(row["target"]) if "target" in row else None,
                "prev_close": close_prev,
                "model_version": result.get("model_version", ""),
            })
        except Exception:
            pass

    if not rows:
        return pd.DataFrame()

    pred_df = pd.DataFrame(rows).set_index("date")
    pred_df.sort_index(inplace=True)
    Path(PREDICTIONS_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(PREDICTIONS_LOG_PATH)
    return pred_df




@st.cache_data(ttl=60)
@st.cache_data(ttl=300)
def load_drift_report() -> dict:
    """Download latest drift report from S3, fallback to local file."""
    if S3_BUCKET:
        try:
            s3 = boto3.client("s3")
            local = Path(DRIFT_REPORT_PATH)
            local.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(S3_BUCKET, "monitoring/latest_drift_report.json", str(local))
        except Exception:
            pass
    if os.path.exists(DRIFT_REPORT_PATH):
        with open(DRIFT_REPORT_PATH) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=300)
def load_drift_history() -> pd.DataFrame:
    """Build drift history from all monitoring reports in S3.

    Each drift check run writes latest_drift_report.json to S3. We accumulate
    history by listing all versioned reports (or just using the current report
    as a single data point if no history parquet exists yet).
    """
    history_path = Path(os.environ.get("DRIFT_HISTORY_PATH", "data/monitoring/drift_history.parquet"))

    # Try to load existing history
    rows = []
    if history_path.exists():
        existing = pd.read_parquet(history_path)
        rows = existing.to_dict("records")

    # Append current report if not already in history
    report = load_drift_report()
    if report and "timestamp" in report:
        ts = report["timestamp"]
        already_in_history = any(r.get("timestamp") == ts for r in rows)
        if not already_in_history:
            rows.append({
                "timestamp": ts,
                "max_psi": report.get("max_psi", 0.0),
                "drift_fraction": report.get("drift_fraction", 0.0),
                "n_features_drifted": report.get("n_features_drifted", 0),
                "trigger_retraining": report.get("trigger_retraining", False),
            })
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            history_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(history_path, index=False)
            return df

    if rows:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp")

    return pd.DataFrame(columns=["timestamp", "drift_fraction", "max_psi"])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Stock Prediction Monitor")
health = fetch_health()
api_status = "🟢 Online" if health.get("model_loaded") else "🔴 Model not loaded"
st.sidebar.markdown(f"**API Status:** {api_status}")
st.sidebar.markdown(f"**Model Version:** `{health.get('model_version', 'N/A')}`")
st.sidebar.markdown(f"**API URL:** `{API_URL}`")

selected_ticker = st.sidebar.selectbox("Ticker", TICKERS)

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["📈 Live Predictions", "🔍 Drift Metrics", "⚙️ System Metrics"])

with tab1:
    st.header(f"Price Predictions — {selected_ticker}")

    # Date range selector — predictions run automatically on change
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input(
            "From",
            value=date.today() - timedelta(days=30),
            max_value=date.today(),
        )
    with col_end:
        end_date = st.date_input(
            "To",
            value=date.today(),
            max_value=date.today(),
        )

    with st.spinner(f"Loading predictions for {selected_ticker}..."):
        pred_df = predict_price_range(selected_ticker, start_date, end_date)

    if pred_df.empty:
        st.warning("No data available for the selected range.")
    else:
        import numpy as np

        # --- Metrics row ---
        valid = pred_df.dropna(subset=["predicted_close", "actual_close", "prev_close"]) \
            if "predicted_close" in pred_df.columns and "actual_close" in pred_df.columns else pd.DataFrame()

        if not valid.empty:
            # Directional accuracy: did model correctly predict up/down from prev_close?
            actual_dir = np.sign(valid["actual_close"].values - valid["prev_close"].values)
            pred_dir = np.sign(valid["predicted_return"].values)
            dir_acc = float(np.mean(actual_dir == pred_dir)) * 100

            # MAE in dollar terms
            mae_price = float(np.mean(np.abs(valid["actual_close"].values - valid["predicted_close"].values)))

            last_actual = valid["actual_close"].iloc[-1]
            last_pred = valid["predicted_close"].iloc[-1]
            last_date = valid.index[-1].date()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Directional Accuracy", f"{dir_acc:.1f}%",
                      help="% of days where model correctly predicted up/down direction")
            m2.metric("MAE (price)", f"${mae_price:.2f}",
                      help="Mean absolute error between predicted and actual closing price")
            m3.metric(f"Actual Close ({last_date})", f"${last_actual:.2f}")
            m4.metric(f"Predicted Close ({last_date})", f"${last_pred:.2f}",
                      delta=f"${last_pred - last_actual:+.2f}")

        # --- Price chart ---
        # Both actual_close and predicted_close are already indexed at the SAME prediction date T.
        # No date shift needed — the model used features from T-1 to predict close at T.
        fig = go.Figure()

        if "actual_close" in pred_df.columns and pred_df["actual_close"].notna().any():
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df["actual_close"],
                name="Actual Close",
                line=dict(color="#4C9BE8", width=2),
            ))

        if "predicted_close" in pred_df.columns and pred_df["predicted_close"].notna().any():
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df["predicted_close"],
                name="Predicted Close",
                line=dict(color="#F28C38", width=2, dash="dash"),
            ))

            if "upper_price" in pred_df.columns and "lower_price" in pred_df.columns:
                idx = pred_df.index
                upper = pred_df["upper_price"]
                lower = pred_df["lower_price"]
                fig.add_trace(go.Scatter(
                    x=pd.concat([pd.Series(idx), pd.Series(idx[::-1])]),
                    y=pd.concat([upper, lower[::-1]]),
                    fill="toself",
                    fillcolor="rgba(242,140,56,0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Confidence Band (p10–p90)",
                    showlegend=True,
                ))

        fig.update_layout(
            title=f"{selected_ticker} — Actual vs Predicted Closing Price",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Data table ---
        with st.expander("Show raw data"):
            display_cols = ["actual_close", "predicted_close", "lower_price", "upper_price",
                            "predicted_return", "actual_return", "prev_close"]
            display_cols = [c for c in display_cols if c in pred_df.columns]
            st.dataframe(pred_df[display_cols].round(4), use_container_width=True)

    drift_report = load_drift_report()
    if drift_report.get("overall_drift_detected"):
        # Only warn if the endpoint hasn't been updated since drift was detected
        drift_ts = pd.Timestamp(drift_report.get("timestamp", "2000-01-01"), tz="UTC")
        try:
            # model_version format: "stock-model-20260406-193744"
            mv = health.get("model_version", "").replace("stock-model-", "")
            endpoint_ts = pd.Timestamp(datetime.strptime(mv, "%Y%m%d-%H%M%S"), tz=timezone.utc)
        except Exception:
            endpoint_ts = pd.Timestamp("2000-01-01", tz="UTC")
        if endpoint_ts < drift_ts:
            st.error("DRIFT DETECTED — model may be stale. Retraining has been triggered.")

with tab2:
    st.header("Data Drift Analysis")

    drift_report = load_drift_report()
    drift_history = load_drift_history()

    max_psi = drift_report.get("max_psi", 0.0)
    drift_detected = drift_report.get("overall_drift_detected", False)

    render_drift_banner(drift_detected, max_psi)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        feature_psi = {
            r["feature"]: r["psi"]
            for r in drift_report.get("features", [])
        }
        render_psi_bar_chart(feature_psi)

    with col_right:
        render_drift_trend(drift_history)

    st.subheader("Feature Drift Details")
    if drift_report.get("features"):
        features_df = pd.DataFrame(drift_report["features"])
        st.dataframe(
            features_df.style.background_gradient(subset=["psi"], cmap="RdYlGn_r"),
            use_container_width=True,
        )

    # KS heatmap requires history
    ks_history_path = os.environ.get("KS_HISTORY_PATH", "data/monitoring/ks_history.parquet")
    if os.path.exists(ks_history_path):
        ks_df = pd.read_parquet(ks_history_path)
        render_ks_heatmap(ks_df)

with tab3:
    st.header("System & Infrastructure Metrics")

    api_metrics = fetch_api_metrics()

    render_system_metrics(
        requests_total=api_metrics.get("requests_total", 0),
        avg_latency_ms=api_metrics.get("avg_latency_ms", 0.0),
        p95_latency_ms=api_metrics.get("p95_latency_ms", 0.0),
        uptime_seconds=api_metrics.get("uptime_seconds", 0.0),
    )

    # Latency history — fetched from proxy in-memory buckets (1 bucket/minute)
    try:
        resp = requests.get(f"{API_URL}/latency-history", timeout=5)
        latency_records = resp.json()
        if latency_records:
            latency_df = pd.DataFrame(latency_records)
            latency_df["timestamp"] = pd.to_datetime(latency_df["timestamp"])
            render_latency_chart(latency_df)
        else:
            st.info("Latency history accumulates in 1-minute buckets — check back after more requests.")
    except Exception:
        st.info("Proxy not reachable — latency history unavailable.")

    # Retraining events — pulled from CloudWatch Lambda logs
    st.subheader("Retraining Event History")
    try:
        logs = boto3.client("logs", region_name="us-east-1")
        streams = logs.describe_log_streams(
            logGroupName="/aws/lambda/stock-prediction-orchestrator",
            orderBy="LastEventTime",
            descending=True,
            limit=10,
        )["logStreams"]

        keywords = {"training_launched", "ingestion_launched", "drift_check_launched",
                    "endpoint updated", "updated", "no_drift", "skipped"}
        retrain_events = []
        seen = set()
        for stream in streams:
            events = logs.get_log_events(
                logGroupName="/aws/lambda/stock-prediction-orchestrator",
                logStreamName=stream["logStreamName"],
                limit=100,
            )["events"]
            for e in events:
                msg = e["message"].strip()
                ts = e["timestamp"]
                # Pick up Lambda REPORT lines (show each invocation) and status lines
                if "Determined source:" in msg or "status" in msg.lower():
                    if ts not in seen and any(k in msg for k in keywords | {"source:", "status"}):
                        seen.add(ts)
                        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                        # Extract just the useful part
                        if "Determined source:" in msg:
                            label = msg.split("Determined source:")[-1].strip()
                        else:
                            label = msg[:150]
                        retrain_events.append({"timestamp": dt, "event": label})

        if retrain_events:
            st.dataframe(pd.DataFrame(retrain_events).drop_duplicates(), use_container_width=True)
        else:
            st.info("No retraining events recorded yet.")
    except Exception as ex:
        st.info(f"Could not fetch retraining events: {ex}")
