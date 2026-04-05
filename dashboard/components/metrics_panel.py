"""System metrics panel: API throughput, latency, EC2 CPU, retraining events."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_system_metrics(
    requests_total: int,
    avg_latency_ms: float,
    p95_latency_ms: float,
    uptime_seconds: float,
) -> None:
    """Render top-level API performance metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Requests", f"{requests_total:,}")
    col2.metric("Avg Latency", f"{avg_latency_ms:.1f} ms")
    col3.metric("p95 Latency", f"{p95_latency_ms:.1f} ms")
    hours = uptime_seconds / 3600
    col4.metric("Uptime", f"{hours:.1f} h")


def render_latency_chart(latency_history: pd.DataFrame) -> None:
    """Render latency percentile trend.

    Args:
        latency_history: DataFrame with columns [timestamp, p50_ms, p95_ms, p99_ms].
    """
    if latency_history.empty:
        st.info("No latency history available.")
        return

    fig = go.Figure()
    for col, color, name in [
        ("p50_ms", "#2ca02c", "p50"),
        ("p95_ms", "#ff7f0e", "p95"),
        ("p99_ms", "#d62728", "p99"),
    ]:
        if col in latency_history.columns:
            fig.add_trace(
                go.Scatter(
                    x=latency_history["timestamp"],
                    y=latency_history[col],
                    name=name,
                    line={"color": color},
                )
            )

    fig.add_hline(y=200, line_dash="dash", line_color="red", annotation_text="SLA (200 ms)")
    fig.update_layout(
        title="API Latency Over Time",
        xaxis_title="Time",
        yaxis_title="Latency (ms)",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_retraining_table(retrain_events: list[dict]) -> None:
    """Render a table of all retraining events.

    Args:
        retrain_events: List of dicts with keys: timestamp, trigger, job_name,
                        before_rmse, after_rmse, status.
    """
    st.subheader("Retraining Event History")
    if not retrain_events:
        st.info("No retraining events recorded yet.")
        return

    df = pd.DataFrame(retrain_events)
    if "before_rmse" in df.columns and "after_rmse" in df.columns:
        df["rmse_delta"] = (df["after_rmse"] - df["before_rmse"]).round(6)

    st.dataframe(df, use_container_width=True)
