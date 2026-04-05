"""Optuna-based hyperparameter optimisation for XGBoost."""

import logging
from typing import Callable

import optuna
import pandas as pd
import xgboost as xgb
import yaml

from src.training.evaluate import compute_metrics
from src.training.time_series_split import walk_forward_splits

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    n_splits: int,
    gap: int,
) -> float:
    """Optuna objective: return mean val RMSE across CV folds."""
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 42,
        "verbosity": 0,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
    }

    feature_cols = [c for c in df.columns if c != "target"]
    splits = walk_forward_splits(df, n_splits=n_splits, gap=gap)
    fold_rmses: list[float] = []

    for split in splits:
        dtrain = xgb.DMatrix(split.X_train, label=split.y_train, feature_names=feature_cols)
        dval = xgb.DMatrix(split.X_val, label=split.y_val, feature_names=feature_cols)

        n_est = params.pop("n_estimators")
        callbacks = [optuna.integration.XGBoostPruningCallback(trial, "val-rmse")]
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=n_est,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            callbacks=callbacks,
            verbose_eval=False,
        )
        params["n_estimators"] = n_est

        val_pred = booster.predict(dval)
        m = compute_metrics(split.y_val.values, val_pred)
        fold_rmses.append(m["rmse"])

    return float(sum(fold_rmses) / len(fold_rmses))


def run_tuning(
    df: pd.DataFrame,
    n_trials: int = 50,
    timeout: int = 3600,
    n_splits: int = 3,
    gap: int = 5,
    config_path: str = "configs/model_config.yaml",
) -> dict:
    """Run Optuna hyperparameter search.

    Args:
        df: Full training DataFrame with 'target' column.
        n_trials: Number of Optuna trials.
        timeout: Max wall-clock seconds for the study.
        n_splits: Walk-forward CV folds.
        gap: Gap samples between train/val in each fold.
        config_path: Path to model config (reads n_trials/timeout overrides).

    Returns:
        Best hyperparameter dict ready to pass to train_xgboost.train().
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    n_trials = cfg["xgboost"]["optuna"].get("n_trials", n_trials)
    timeout = cfg["xgboost"]["optuna"].get("timeout", timeout)

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="xgboost_stock_prediction",
    )

    study.optimize(
        lambda trial: _objective(trial, df, n_splits, gap),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_params["objective"] = "reg:squarederror"
    best_params["eval_metric"] = "rmse"
    best_params["seed"] = 42
    best_params["verbosity"] = 0

    logger.info("Best trial RMSE: %.6f", study.best_value)
    logger.info("Best params: %s", best_params)

    return best_params
