"""Drift metrics panel component."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_drift_banner(drift_detected: bool, max_psi: float) -> None:
    """Show a coloured banner based on drift severity."""
    if drift_detected:
        st.error(
            f"DRIFT DETECTED — Retraining triggered (max PSI = {max_psi:.3f})"
        )
    elif max_psi > 0.1:
        st.warning(
            f"Moderate drift detected — monitoring (max PSI = {max_psi:.3f})"
        )
    else:
        st.success(f"Model stable (max PSI = {max_psi:.3f})")


def render_psi_bar_chart(feature_psi: dict[str, float]) -> None:
    """Render a colour-coded bar chart of per-feature PSI scores."""
    if not feature_psi:
        st.info("No PSI data available yet.")
        return

    df = pd.DataFrame(
        {"Feature": list(feature_psi.keys()), "PSI": list(feature_psi.values())}
    ).sort_values("PSI", ascending=False)

    colors = [
        "#d62728" if v > 0.2 else "#ff7f0e" if v > 0.1 else "#2ca02c"
        for v in df["PSI"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df["Feature"],
            y=df["PSI"],
            marker_color=colors,
            text=[f"{v:.3f}" for v in df["PSI"]],
            textposition="outside",
        )
    )

    fig.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Investigate (0.1)")
    fig.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Retrain (0.2)")

    fig.update_layout(
        title="Per-Feature PSI Scores",
        xaxis_title="Feature",
        yaxis_title="PSI",
        height=400,
        xaxis={"tickangle": -45},
        margin={"t": 60, "b": 100},
    )

    st.plotly_chart(fig, use_container_width=True)


def render_ks_heatmap(ks_history: pd.DataFrame) -> None:
    """Render a KS statistic heatmap (features × time).

    Args:
        ks_history: DataFrame with columns [timestamp, feature, ks_statistic].
    """
    if ks_history.empty:
        st.info("Not enough history for KS heatmap yet.")
        return

    pivot = ks_history.pivot_table(
        index="feature", columns="timestamp", values="ks_statistic"
    )

    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn_r",
        zmin=0,
        zmax=1,
        title="KS Statistic Heatmap (features × time)",
        labels={"color": "KS statistic"},
        aspect="auto",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_drift_trend(drift_history: pd.DataFrame) -> None:
    """Render overall drift score over time.

    Args:
        drift_history: DataFrame with columns [timestamp, drift_fraction, max_psi].
    """
    if drift_history.empty:
        st.info("No drift history available yet.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drift_history["timestamp"],
            y=drift_history["drift_fraction"],
            name="Drift Fraction",
            line={"color": "#1f77b4"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=drift_history["timestamp"],
            y=drift_history["max_psi"],
            name="Max PSI",
            line={"color": "#d62728", "dash": "dash"},
            yaxis="y2",
        )
    )
    fig.add_hline(y=0.3, line_dash="dot", line_color="orange", annotation_text="Drift threshold (30%)")

    fig.update_layout(
        title="Drift Score History (Last 30 Days)",
        xaxis_title="Date",
        yaxis_title="Drift Fraction",
        yaxis2={
            "title": "Max PSI",
            "overlaying": "y",
            "side": "right",
        },
        height=350,
        legend={"orientation": "h"},
    )
    st.plotly_chart(fig, use_container_width=True)
