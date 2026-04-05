"""XGBoost model training entry point.

This file is also the SageMaker training script — it reads from
/opt/ml/input/data/train/ and writes to /opt/ml/model/ when running in SM.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb
import yaml

from src.training.evaluate import compute_metrics
from src.training.time_series_split import walk_forward_splits, train_test_split_temporal

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
) -> xgb.Booster:
    """Train a single XGBoost model.

    Args:
        X_train, y_train: Training features and target.
        X_val, y_val: Validation features and target (for early stopping).
        params: XGBoost hyperparameter dict.

    Returns:
        Trained xgb.Booster.
    """
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=list(X_train.columns))

    n_estimators = params.pop("n_estimators", 500)
    early_stopping_rounds = params.pop("early_stopping_rounds", 30)

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50,
    )

    # Restore popped keys so the dict is not mutated for callers
    params["n_estimators"] = n_estimators
    params["early_stopping_rounds"] = early_stopping_rounds

    val_pred = booster.predict(dval)
    metrics = compute_metrics(y_val.values, val_pred)
    logger.info("Val metrics: %s", metrics)

    return booster


def train_with_cv(
    df: pd.DataFrame,
    params: dict,
    n_splits: int = 3,
    gap: int = 5,
) -> tuple[xgb.Booster, dict]:
    """Train using walk-forward CV; return the final model trained on full train set.

    Returns:
        (final_booster, averaged_cv_metrics)
    """
    feature_cols = [c for c in df.columns if c != "target"]
    train_df, test_df = train_test_split_temporal(df)

    splits = walk_forward_splits(train_df, n_splits=n_splits, gap=gap)

    all_metrics: list[dict] = []
    for split in splits:
        booster = train(split.X_train, split.y_train, split.X_val, split.y_val, dict(params))
        dval = xgb.DMatrix(split.X_val, feature_names=feature_cols)
        val_pred = booster.predict(dval)
        m = compute_metrics(split.y_val.values, val_pred)
        m["fold"] = split.fold
        all_metrics.append(m)
        logger.info("Fold %d — %s", split.fold, m)

    # Average CV metrics (exclude 'fold' key)
    cv_avg = {
        k: sum(d[k] for d in all_metrics) / len(all_metrics)
        for k in all_metrics[0]
        if k != "fold"
    }
    logger.info("CV average metrics: %s", cv_avg)

    # Train final model on full training set (no val split)
    X_full = train_df[feature_cols]
    y_full = train_df["target"]
    # Use last val split as a rough validation proxy for early stopping
    last_split = splits[-1]
    final_booster = train(X_full, y_full, last_split.X_val, last_split.y_val, dict(params))

    # Evaluate on held-out test set
    dtest = xgb.DMatrix(test_df[feature_cols], feature_names=feature_cols)
    test_pred = final_booster.predict(dtest)
    test_metrics = compute_metrics(test_df["target"].values, test_pred)
    logger.info("Test set metrics: %s", test_metrics)
    cv_avg["test"] = test_metrics

    # Log feature importances
    importances = final_booster.get_score(importance_type="gain")
    top_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("Top-10 feature importances (gain): %s", top_feats)

    return final_booster, cv_avg


def save_model(booster: xgb.Booster, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(path)
    logger.info("Model saved to %s", path)


def load_model(path: str) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(path)
    return booster


# ---------------------------------------------------------------------------
# SageMaker entry point
# ---------------------------------------------------------------------------

def _sagemaker_paths() -> tuple[str, str, str]:
    data_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output")
    return data_dir, model_dir, output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--ticker", default="AAPL")
    args, _ = parser.parse_known_args()

    data_dir, model_dir, output_dir = _sagemaker_paths()

    # Load feature parquet from train channel
    parquet_files = list(Path(data_dir).glob("*.parquet"))
    if not parquet_files:
        logger.error("No parquet files found in %s", data_dir)
        sys.exit(1)

    df = pd.concat([pd.read_parquet(f) for f in parquet_files])
    df.sort_index(inplace=True)
    logger.info("Loaded %d rows from %s", len(df), data_dir)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    params = dict(cfg["xgboost"]["base_params"])
    booster, metrics = train_with_cv(
        df,
        params,
        n_splits=cfg["training"]["n_cv_splits"],
        gap=cfg["training"]["cv_gap"],
    )

    save_model(booster, os.path.join(model_dir, "model.xgb"))

    metrics_path = os.path.join(output_dir, "metrics.json")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics written to %s", metrics_path)
