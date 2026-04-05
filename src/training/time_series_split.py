"""Walk-forward cross-validation split for time-series data.

IMPORTANT: Never shuffle time-series data. All splits preserve temporal order.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


@dataclass
class CVSplit:
    fold: int
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    train_start: str
    train_end: str
    val_start: str
    val_end: str


def walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 3,
    gap: int = 5,
) -> list[CVSplit]:
    """Generate walk-forward (expanding window) CV splits.

    Args:
        df: Feature DataFrame with a 'target' column and DatetimeIndex.
        n_splits: Number of CV folds.
        gap: Number of samples to skip between train end and val start,
             preventing lookahead from lagged features.

    Returns:
        List of CVSplit dataclasses.
    """
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols]
    y = df["target"]

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    splits = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        split = CVSplit(
            fold=fold,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            train_start=str(X_train.index[0].date()),
            train_end=str(X_train.index[-1].date()),
            val_start=str(X_val.index[0].date()),
            val_end=str(X_val.index[-1].date()),
        )
        splits.append(split)
        logger.debug(
            "Fold %d: train [%s – %s] (%d rows), val [%s – %s] (%d rows)",
            fold,
            split.train_start,
            split.train_end,
            len(X_train),
            split.val_start,
            split.val_end,
            len(X_val),
        )

    logger.info("Created %d walk-forward CV folds (gap=%d)", n_splits, gap)
    return splits


def train_test_split_temporal(
    df: pd.DataFrame,
    test_size: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hard holdout split — test set is always the last `test_size` fraction."""
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    logger.info(
        "Temporal split: train=%d rows, test=%d rows (%.0f%% holdout)",
        len(train),
        len(test),
        test_size * 100,
    )
    return train, test
