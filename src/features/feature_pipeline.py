"""Orchestrates the full feature engineering pipeline."""

import logging
import os
import pickle
from pathlib import Path

import pandas as pd
import yaml
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.features.technical_indicators import (
    build_target,
    compute_bollinger_bands,
    compute_ema,
    compute_lag_features,
    compute_macd,
    compute_price_features,
    compute_rsi,
    compute_sma,
    compute_volume_features,
)
from src.ingestion.fetch_yahoo import fetch_ohlcv

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: list[str] = []  # populated dynamically


def build_feature_matrix(
    ticker: str,
    start: str,
    end: str,
    config_path: str = "configs/feature_config.yaml",
    ohlcv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the complete feature matrix for a ticker.

    Args:
        ticker: Stock symbol.
        start: Start date string "YYYY-MM-DD".
        end: End date string "YYYY-MM-DD".
        config_path: Path to feature_config.yaml.
        ohlcv: Optional pre-fetched OHLCV DataFrame (skips download).

    Returns:
        DataFrame with all features + target column, NaN rows dropped.
    """
    cfg = _load_config(config_path)

    if ohlcv is None:
        ohlcv = fetch_ohlcv(
            ticker,
            start,
            end,
            interval=cfg["data"]["default_interval"],
        )

    logger.info("Building features for %s (%d raw rows)", ticker, len(ohlcv))

    close = ohlcv["Close"]
    parts = [
        compute_rsi(close, period=cfg["features"]["rsi_period"]),
        compute_sma(close, windows=cfg["features"]["sma_windows"]),
        compute_ema(close, windows=cfg["features"]["ema_windows"]),
        compute_macd(close),
        compute_bollinger_bands(
            close,
            window=cfg["features"]["bollinger_window"],
            num_std=cfg["features"]["bollinger_std"],
        ),
        compute_volume_features(ohlcv, window=cfg["features"]["volume_sma_window"]),
        compute_price_features(ohlcv),
        compute_lag_features(close, periods=cfg["features"]["lag_periods"]),
        build_target(close, horizon=cfg["features"]["target_horizon"]),
    ]

    df = pd.concat(parts, axis=1)

    # Drop the last `horizon` rows whose target is NaN by design
    df = df.dropna()

    min_rows = cfg["pipeline"]["min_rows_after_dropna"]
    if len(df) < min_rows:
        raise ValueError(
            f"{ticker}: only {len(df)} rows after dropna, need ≥{min_rows}"
        )

    logger.info(
        "Feature matrix built: %d rows × %d features (+ target)",
        len(df),
        df.shape[1] - 1,
    )
    return df


def split_and_scale(
    df: pd.DataFrame,
    test_size: float = 0.10,
    scaler_type: str = "RobustScaler",
    scaler_save_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """Temporal train / val / test split with scaler fit on train only.

    Returns:
        (train_df, val_df, test_df, fitted_scaler)
        Each DataFrame retains all columns including 'target'.
    """
    n = len(df)
    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * 0.875)  # ~12.5% of remaining for val

    train = df.iloc[:val_idx].copy()
    val = df.iloc[val_idx:test_idx].copy()
    test = df.iloc[test_idx:].copy()

    feature_cols = [c for c in df.columns if c != "target"]

    scaler = RobustScaler() if scaler_type == "RobustScaler" else StandardScaler()
    scaler.fit(train[feature_cols])

    for split in (train, val, test):
        split[feature_cols] = scaler.transform(split[feature_cols])

    if scaler_save_path:
        Path(scaler_save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_save_path, "wb") as f:
            pickle.dump(scaler, f)
        logger.info("Scaler saved to %s", scaler_save_path)

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d", len(train), len(val), len(test)
    )
    return train, val, test, scaler


def load_scaler(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)
