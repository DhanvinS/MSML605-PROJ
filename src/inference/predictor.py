"""Model loading and prediction logic."""

import logging
import os
import pickle
from pathlib import Path

import boto3
import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)


class StockPredictor:
    """Loads XGBoost model + scaler from disk or S3 and serves predictions.

    Trains three models: median (p50), lower bound (p10), upper bound (p90).
    The p10/p90 models use quantile regression for calibrated intervals.
    """

    def __init__(self) -> None:
        self._booster: xgb.Booster | None = None
        self._booster_low: xgb.Booster | None = None
        self._booster_high: xgb.Booster | None = None
        self._scaler = None
        self._feature_names: list[str] = []
        self._version: str = "unknown"

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_from_dir(self, model_dir: str) -> None:
        """Load all artefacts from a local directory."""
        model_path = Path(model_dir) / "model.xgb"
        scaler_path = Path(model_dir) / "scaler.pkl"
        low_path = Path(model_dir) / "model_p10.xgb"
        high_path = Path(model_dir) / "model_p90.xgb"
        version_path = Path(model_dir) / "version.txt"

        self._booster = _load_booster(model_path)
        self._feature_names = self._booster.feature_names or []

        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self._scaler = pickle.load(f)
            logger.info("Scaler loaded from %s", scaler_path)

        if low_path.exists():
            self._booster_low = _load_booster(low_path)
        if high_path.exists():
            self._booster_high = _load_booster(high_path)

        if version_path.exists():
            self._version = version_path.read_text().strip()

        logger.info("StockPredictor loaded (version=%s)", self._version)

    def load_from_s3(
        self,
        bucket: str,
        prefix: str,
        local_cache: str = "/tmp/model_cache",
    ) -> None:
        """Download model artefacts from S3 then load from local cache."""
        Path(local_cache).mkdir(parents=True, exist_ok=True)
        s3 = boto3.client("s3")

        for fname in ["model.xgb", "scaler.pkl", "model_p10.xgb", "model_p90.xgb", "version.txt"]:
            key = f"{prefix}/{fname}"
            local = Path(local_cache) / fname
            try:
                s3.download_file(bucket, key, str(local))
                logger.info("Downloaded s3://%s/%s", bucket, key)
            except Exception as exc:
                logger.warning("Could not download %s: %s", key, exc)

        self.load_from_dir(local_cache)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self, features: list[float]
    ) -> tuple[float, float, float]:
        """Predict for a single observation.

        Args:
            features: List of N_FEATURES float values (already scaled).

        Returns:
            (p50, p10, p90) — point estimate and confidence interval.
        """
        self._check_loaded()
        x = self._preprocess(np.array(features, dtype=float).reshape(1, -1))
        dm = xgb.DMatrix(x, feature_names=self._feature_names or None)

        p50 = float(self._booster.predict(dm)[0])
        p10 = float(self._booster_low.predict(dm)[0]) if self._booster_low else p50 * 0.9
        p90 = float(self._booster_high.predict(dm)[0]) if self._booster_high else p50 * 1.1

        return p50, p10, p90

    def predict_batch(
        self, records: list[list[float]]
    ) -> tuple[list[float], list[float], list[float]]:
        """Predict for a batch of observations.

        Returns:
            Three equal-length lists: (p50s, p10s, p90s).
        """
        self._check_loaded()
        X = self._preprocess(np.array(records, dtype=float))
        dm = xgb.DMatrix(X, feature_names=self._feature_names or None)

        p50s = self._booster.predict(dm).tolist()
        p10s = self._booster_low.predict(dm).tolist() if self._booster_low else [v * 0.9 for v in p50s]
        p90s = self._booster_high.predict(dm).tolist() if self._booster_high else [v * 1.1 for v in p50s]

        return p50s, p10s, p90s

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is not None:
            return self._scaler.transform(X)
        return X

    def _check_loaded(self) -> None:
        if self._booster is None:
            raise RuntimeError("Model not loaded. Call load_from_dir() or load_from_s3() first.")

    @property
    def is_loaded(self) -> bool:
        return self._booster is not None

    def get_version(self) -> str:
        return self._version


def _load_booster(path: Path) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(path))
    logger.info("Booster loaded from %s", path)
    return booster
