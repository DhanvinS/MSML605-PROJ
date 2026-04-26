"""Baseline models for comparison against the XGBoost stock prediction model.

Each baseline exposes the same .fit(X_train, y_train) / .predict(X_test) interface
so they can be evaluated with the same walk-forward CV pipeline used for XGBoost.

Run via: python scripts/run_baseline_comparison.py
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from src.training.evaluate import compute_metrics
from src.training.time_series_split import train_test_split_temporal, walk_forward_splits

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline model classes
# ---------------------------------------------------------------------------

class NaiveMeanBaseline:
    """Always predicts the mean of training returns.

    More meaningful than zero-forecast for directional accuracy — the mean
    return on a training set is typically a small positive number, so this
    gives a slight directional signal (bullish bias) rather than always
    predicting no change.
    """

    name = "naive_mean"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "NaiveMeanBaseline":
        self._mean = float(y_train.mean())
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return np.full(len(X_test), self._mean)


class LinearRegressionBaseline:
    """Ordinary least-squares linear regression on all 30 features.

    A fresh StandardScaler is fit on each training fold to prevent leakage.
    """

    name = "linear_regression"

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._model = LinearRegression()

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LinearRegressionBaseline":
        X_scaled = self._scaler.fit_transform(X_train)
        self._model.fit(X_scaled, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self._model.predict(self._scaler.transform(X_test))


class RidgeRegressionBaseline:
    """L2-regularised linear regression — more robust than OLS on financial data."""

    name = "ridge_regression"

    def __init__(self, alpha: float = 1.0) -> None:
        self._scaler = StandardScaler()
        self._model = Ridge(alpha=alpha)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RidgeRegressionBaseline":
        X_scaled = self._scaler.fit_transform(X_train)
        self._model.fit(X_scaled, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self._model.predict(self._scaler.transform(X_test))


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_baselines(
    df: pd.DataFrame,
    baselines: list,
    n_splits: int = 3,
    gap: int = 5,
    test_size: float = 0.10,
) -> list[dict]:
    """Run walk-forward CV + test holdout for every baseline, using the same
    splits as the XGBoost training pipeline.

    Args:
        df: Feature DataFrame with 'target' column and DatetimeIndex.
            Typically the concatenated processed parquets across all tickers.
        baselines: List of baseline instances (NaiveMeanBaseline, etc.)
        n_splits: Number of walk-forward CV folds (matches XGBoost default).
        gap: Samples skipped between train end and val start (matches XGBoost default).
        test_size: Fraction of data held out as the final test set.

    Returns:
        List of result dicts, each with keys:
            model, split (fold_N | cv_avg | test), and all compute_metrics keys.
    """
    feature_cols = [c for c in df.columns if c != "target"]

    train_df, test_df = train_test_split_temporal(df, test_size=test_size)
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    results: list[dict] = []

    for baseline in baselines:
        fold_metrics: list[dict] = []

        # --- Walk-forward CV on train portion ---
        for split in walk_forward_splits(train_df, n_splits=n_splits, gap=gap):
            baseline.fit(split.X_train, split.y_train)
            y_pred = baseline.predict(split.X_val)
            m = compute_metrics(split.y_val.values, y_pred)
            fold_metrics.append(m)
            results.append({"model": baseline.name, "split": f"fold_{split.fold}", **m})

        # --- CV average ---
        avg = {
            k: round(float(np.mean([f[k] for f in fold_metrics])), 6)
            for k in fold_metrics[0]
            if isinstance(fold_metrics[0][k], (int, float))
        }
        results.append({"model": baseline.name, "split": "cv_avg", **avg})

        # --- Final test holdout: refit on full train set ---
        baseline.fit(train_df[feature_cols], train_df["target"])
        y_pred_test = baseline.predict(X_test)
        test_m = compute_metrics(y_test.values, y_pred_test)
        results.append({"model": baseline.name, "split": "test", **test_m})

        logger.info(
            "%s — test RMSE=%.4f dir_acc=%.4f sharpe=%.4f",
            baseline.name,
            test_m["rmse"],
            test_m["directional_accuracy"],
            test_m["sharpe_proxy"],
        )

    return results
