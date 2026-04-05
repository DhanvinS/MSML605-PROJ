"""Streamlit observability dashboard.

Tabs:
  1. Live Predictions — actual vs predicted + confidence band
  2. Drift Metrics    — per-feature PSI + KS heatmap + trend
  3. System Metrics   — API latency, request rate, retraining events
"""

import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from dashboard.components.drift_panel import (
    render_drift_banner,
    render_drift_trend,
    render_ks_heatmap,
    render_psi_bar_chart,
)
from dashboard.components.metrics_panel import (
    render_latency_chart,
    render_retraining_table,
    render_system_metrics,
)
from dashboard.components.prediction_chart import (
    render_metrics_cards,
    render_prediction_chart,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL = os.environ.get("API_URL", "http://localhost:8000")
DRIFT_REPORT_PATH = os.environ.get("DRIFT_REPORT_PATH", "data/baseline/latest_drift_report.json")
PREDICTIONS_LOG_PATH = os.environ.get("PREDICTIONS_LOG_PATH", "data/monitoring/predictions_log.parquet")
REFRESH_INTERVAL_MS = 60_000  # 60 seconds

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


@st.cache_data(ttl=60)
def load_predictions_log() -> pd.DataFrame:
    if os.path.exists(PREDICTIONS_LOG_PATH):
        return pd.read_parquet(PREDICTIONS_LOG_PATH).tail(200)
    # Return empty DataFrame with expected schema
    return pd.DataFrame(
        columns=["timestamp", "actual", "predicted", "lower_bound", "upper_bound"]
    )


@st.cache_data(ttl=60)
def load_drift_report() -> dict:
    if os.path.exists(DRIFT_REPORT_PATH):
        with open(DRIFT_REPORT_PATH) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=300)
def load_drift_history() -> pd.DataFrame:
    history_path = os.environ.get("DRIFT_HISTORY_PATH", "data/monitoring/drift_history.parquet")
    if os.path.exists(history_path):
        return pd.read_parquet(history_path)
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

ticker_options = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
selected_ticker = st.sidebar.selectbox("Ticker", ticker_options)

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["📈 Live Predictions", "🔍 Drift Metrics", "⚙️ System Metrics"])

with tab1:
    st.header("Live Predictions vs Actual")

    pred_df = load_predictions_log()
    drift_report = load_drift_report()
    api_metrics = fetch_api_metrics()

    # Metrics cards
    if not pred_df.empty and "actual" in pred_df.columns:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np

        y_true = pred_df["actual"].dropna().values
        y_pred = pred_df["predicted"].dropna().values
        n = min(len(y_true), len(y_pred))
        if n > 0:
            rmse = float(np.sqrt(mean_squared_error(y_true[:n], y_pred[:n])))
            mae = float(mean_absolute_error(y_true[:n], y_pred[:n]))
            dir_acc = float(np.mean(np.sign(y_true[:n]) == np.sign(y_pred[:n])))
            render_metrics_cards(rmse, mae, dir_acc)

    render_prediction_chart(pred_df)

    if drift_report.get("overall_drift_detected"):
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

    # Latency history (if available)
    latency_path = os.environ.get("LATENCY_HISTORY_PATH", "data/monitoring/latency_history.parquet")
    if os.path.exists(latency_path):
        latency_df = pd.read_parquet(latency_path)
        render_latency_chart(latency_df)
    else:
        st.info("Latency history not yet available — starts accumulating after first requests.")

    # Retraining events
    retrain_path = os.environ.get("RETRAIN_EVENTS_PATH", "data/monitoring/retrain_events.json")
    retrain_events = []
    if os.path.exists(retrain_path):
        with open(retrain_path) as f:
            retrain_events = json.load(f)

    render_retraining_table(retrain_events)
