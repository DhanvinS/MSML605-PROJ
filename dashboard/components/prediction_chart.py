"""Prediction chart component: actual vs predicted with confidence band."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_prediction_chart(df: pd.DataFrame) -> None:
    """Render a Plotly line chart of actual vs predicted prices.

    Args:
        df: DataFrame with columns: timestamp, actual, predicted, lower_bound, upper_bound.
    """
    if df.empty:
        st.info("No prediction data available yet.")
        return

    fig = go.Figure()

    # Confidence band (shaded region)
    fig.add_trace(
        go.Scatter(
            x=pd.concat([df["timestamp"], df["timestamp"][::-1]]),
            y=pd.concat([df["upper_bound"], df["lower_bound"][::-1]]),
            fill="toself",
            fillcolor="rgba(255, 165, 0, 0.15)",
            line={"color": "rgba(255,255,255,0)"},
            name="p10–p90 CI",
            showlegend=True,
        )
    )

    # Actual price
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["actual"],
            mode="lines",
            name="Actual",
            line={"color": "#1f77b4", "width": 2},
        )
    )

    # Predicted price
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["predicted"],
            mode="lines",
            name="Predicted",
            line={"color": "#ff7f0e", "width": 2, "dash": "dash"},
        )
    )

    fig.update_layout(
        title="Live Predictions vs Actual Returns",
        xaxis_title="Timestamp",
        yaxis_title="Forward Return",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        height=400,
        margin={"t": 60, "b": 40},
    )

    st.plotly_chart(fig, use_container_width=True)


def render_metrics_cards(rmse: float, mae: float, directional_acc: float) -> None:
    """Render metric summary cards."""
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse:.5f}")
    col2.metric("MAE", f"{mae:.5f}")
    col3.metric("Directional Accuracy", f"{directional_acc:.1%}")
