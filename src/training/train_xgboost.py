"""XGBoost model training entry point — SageMaker Training Job script.

SageMaker injects:
  /opt/ml/input/data/train/   ← parquet files from 03_initial_ingest.sh
  /opt/ml/model/              ← output: model.xgb, model_p10.xgb, model_p90.xgb, scaler.pkl
  /opt/ml/output/             ← metrics.json

Also writes baseline_stats.json to S3 for drift detection.
"""

import json
import logging
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xgboost as xgb
import yaml
from sklearn.preprocessing import RobustScaler

from src.training.baseline_capture import capture_baseline_stats
from src.training.evaluate import compute_metrics
from src.training.time_series_split import walk_forward_splits, train_test_split_temporal

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# SageMaker paths
# ---------------------------------------------------------------------------

DATA_DIR = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
OUTPUT_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output")
S3_BUCKET = os.environ.get("S3_BUCKET", "")


# ---------------------------------------------------------------------------
# Core training functions
# ---------------------------------------------------------------------------

def _make_dmatrix(X: pd.DataFrame, y: pd.Series) -> xgb.DMatrix:
    return xgb.DMatrix(X, label=y, feature_names=list(X.columns))


def train_single(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
) -> xgb.Booster:
    """Train one XGBoost model with early stopping."""
    params = dict(params)
    n_estimators = params.pop("n_estimators", 500)
    early_stopping_rounds = params.pop("early_stopping_rounds", 30)

    dtrain = _make_dmatrix(X_train, y_train)
    dval = _make_dmatrix(X_val, y_val)

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50,
    )
    return booster


def run_cv(
    df: pd.DataFrame,
    params: dict,
    n_splits: int = 3,
    gap: int = 5,
) -> dict:
    """Walk-forward CV — returns averaged metrics across folds."""
    feature_cols = [c for c in df.columns if c != "target"]
    train_df, _ = train_test_split_temporal(df)
    splits = walk_forward_splits(train_df, n_splits=n_splits, gap=gap)

    fold_metrics = []
    for split in splits:
        b = train_single(split.X_train, split.y_train, split.X_val, split.y_val, dict(params))
        dval = xgb.DMatrix(split.X_val, feature_names=feature_cols)
        m = compute_metrics(split.y_val.values, b.predict(dval))
        m["fold"] = split.fold
        fold_metrics.append(m)
        logger.info("Fold %d — %s", split.fold, m)

    cv_avg = {
        k: sum(d[k] for d in fold_metrics) / len(fold_metrics)
        for k in fold_metrics[0]
        if k != "fold"
    }
    logger.info("CV average: %s", cv_avg)
    return cv_avg


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Load all parquet files from train channel -------------------------
    data_dir = Path(DATA_DIR)
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        logger.error("No parquet files in %s", data_dir)
        sys.exit(1)

    df = pd.concat([pd.read_parquet(f) for f in parquet_files])
    df.sort_index(inplace=True)
    logger.info("Loaded %d rows × %d cols from %d files", len(df), df.shape[1], len(parquet_files))

    # --- Load config -------------------------------------------------------
    with open("configs/model_config.yaml") as f:
        cfg = yaml.safe_load(f)

    base_params = dict(cfg["xgboost"]["base_params"])
    quantile_params = dict(cfg["xgboost"].get("quantile_params", {}))
    n_cv_splits = cfg["training"]["n_cv_splits"]
    cv_gap = cfg["training"]["cv_gap"]

    feature_cols = [c for c in df.columns if c != "target"]

    # --- Temporal split and scaling ----------------------------------------
    n = len(df)
    test_idx = int(n * 0.90)
    val_idx = int(test_idx * 0.875)

    train_df = df.iloc[:val_idx].copy()
    val_df = df.iloc[val_idx:test_idx].copy()
    test_df = df.iloc[test_idx:].copy()

    scaler = RobustScaler()
    scaler.fit(train_df[feature_cols])

    for split in (train_df, val_df, test_df):
        split[feature_cols] = scaler.transform(split[feature_cols])

    logger.info("Split — train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df))

    # Save scaler
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Scaler saved to %s", scaler_path)

    # --- Walk-forward CV on scaled train data ------------------------------
    logger.info("Running walk-forward CV (%d splits)...", n_cv_splits)
    cv_metrics = run_cv(
        pd.concat([train_df, val_df]),
        dict(base_params),
        n_splits=n_cv_splits,
        gap=cv_gap,
    )

    # --- Train final p50 model on train set (val for early stopping) -------
    logger.info("Training final p50 model...")
    booster_p50 = train_single(
        train_df[feature_cols], train_df["target"],
        val_df[feature_cols], val_df["target"],
        dict(base_params),
    )

    # --- Train p10 quantile model ------------------------------------------
    logger.info("Training p10 quantile model...")
    p10_params = dict(base_params)
    p10_params.update({
        "objective": quantile_params.get("objective", "reg:quantileerror"),
        "quantile_alpha": quantile_params.get("quantile_alpha_low", 0.1),
    })
    p10_params.pop("eval_metric", None)
    booster_p10 = train_single(
        train_df[feature_cols], train_df["target"],
        val_df[feature_cols], val_df["target"],
        p10_params,
    )

    # --- Train p90 quantile model ------------------------------------------
    logger.info("Training p90 quantile model...")
    p90_params = dict(base_params)
    p90_params.update({
        "objective": quantile_params.get("objective", "reg:quantileerror"),
        "quantile_alpha": quantile_params.get("quantile_alpha_high", 0.9),
    })
    p90_params.pop("eval_metric", None)
    booster_p90 = train_single(
        train_df[feature_cols], train_df["target"],
        val_df[feature_cols], val_df["target"],
        p90_params,
    )

    # --- Evaluate on test set ----------------------------------------------
    dtest = xgb.DMatrix(test_df[feature_cols], feature_names=feature_cols)
    test_pred_p50 = booster_p50.predict(dtest)
    test_metrics = compute_metrics(test_df["target"].values, test_pred_p50)
    logger.info("Test set metrics: %s", test_metrics)

    # Feature importances
    importances = booster_p50.get_score(importance_type="gain")
    top10 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("Top-10 features (gain): %s", top10)

    # --- Save models -------------------------------------------------------
    booster_p50.save_model(os.path.join(MODEL_DIR, "model.xgb"))
    booster_p10.save_model(os.path.join(MODEL_DIR, "model_p10.xgb"))
    booster_p90.save_model(os.path.join(MODEL_DIR, "model_p90.xgb"))
    logger.info("All 3 models saved to %s", MODEL_DIR)

    # Write version.txt
    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    Path(MODEL_DIR, "version.txt").write_text(version)
    logger.info("Version: %s", version)

    # --- Capture baseline stats and upload to S3 ---------------------------
    logger.info("Capturing baseline stats...")
    capture_baseline_stats(
        X_train=train_df[feature_cols],
        output_path=os.path.join(MODEL_DIR, "baseline_stats.json"),
        s3_bucket=S3_BUCKET or None,
        s3_key="baseline/baseline_stats.json",
    )

    # --- Write metrics.json ------------------------------------------------
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    all_metrics = {
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "feature_importances": dict(top10),
        "model_version": version,
        "n_train_samples": len(train_df),
        "n_test_samples": len(test_df),
    }
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Metrics written to %s", metrics_path)
    logger.info("Training complete.")
