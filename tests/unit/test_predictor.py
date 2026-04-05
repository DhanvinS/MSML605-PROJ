"""Unit tests for the StockPredictor class."""

import os
import tempfile

import numpy as np
import pytest
import xgboost as xgb

from src.inference.predictor import StockPredictor
from src.inference.schemas import N_FEATURES


@pytest.fixture
def tiny_booster(tmp_path):
    """Train and save a minimal XGBoost model for testing."""
    np.random.seed(0)
    n, p = 100, N_FEATURES
    X = np.random.randn(n, p).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)

    dtrain = xgb.DMatrix(X, label=y)
    booster = xgb.train({"objective": "reg:squarederror"}, dtrain, num_boost_round=5)

    model_path = tmp_path / "model.xgb"
    booster.save_model(str(model_path))

    version_path = tmp_path / "version.txt"
    version_path.write_text("test-v1")

    return tmp_path, booster


class TestStockPredictor:
    def test_not_loaded_initially(self):
        p = StockPredictor()
        assert not p.is_loaded

    def test_load_from_dir(self, tiny_booster):
        model_dir, _ = tiny_booster
        predictor = StockPredictor()
        predictor.load_from_dir(str(model_dir))
        assert predictor.is_loaded

    def test_get_version(self, tiny_booster):
        model_dir, _ = tiny_booster
        predictor = StockPredictor()
        predictor.load_from_dir(str(model_dir))
        assert predictor.get_version() == "test-v1"

    def test_predict_returns_three_floats(self, tiny_booster):
        model_dir, _ = tiny_booster
        predictor = StockPredictor()
        predictor.load_from_dir(str(model_dir))

        features = [float(i) for i in range(N_FEATURES)]
        result = predictor.predict(features)

        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_predict_batch(self, tiny_booster):
        model_dir, _ = tiny_booster
        predictor = StockPredictor()
        predictor.load_from_dir(str(model_dir))

        records = [[float(i) for i in range(N_FEATURES)] for _ in range(5)]
        p50s, p10s, p90s = predictor.predict_batch(records)

        assert len(p50s) == 5
        assert len(p10s) == 5
        assert len(p90s) == 5

    def test_predict_raises_when_not_loaded(self):
        predictor = StockPredictor()
        with pytest.raises(RuntimeError, match="not loaded"):
            predictor.predict([0.0] * N_FEATURES)
