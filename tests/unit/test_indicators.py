"""Unit tests for technical indicator functions."""

import numpy as np
import pandas as pd
import pytest

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


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Synthetic OHLCV data with 200 daily bars."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 1)
    df = pd.DataFrame(
        {
            "Open": close * (1 + np.random.randn(n) * 0.002),
            "High": close * (1 + np.abs(np.random.randn(n)) * 0.005),
            "Low": close * (1 - np.abs(np.random.randn(n)) * 0.005),
            "Close": close,
            "Volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="B"),
    )
    return df


class TestRSI:
    def test_output_range(self, sample_ohlcv):
        rsi = compute_rsi(sample_ohlcv["Close"])
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_length_matches_input(self, sample_ohlcv):
        rsi = compute_rsi(sample_ohlcv["Close"])
        assert len(rsi) == len(sample_ohlcv)

    def test_first_n_are_nan(self, sample_ohlcv):
        rsi = compute_rsi(sample_ohlcv["Close"], period=14)
        assert rsi.iloc[:14].isna().all()

    def test_name(self, sample_ohlcv):
        assert compute_rsi(sample_ohlcv["Close"]).name == "rsi_14"


class TestSMA:
    def test_columns(self, sample_ohlcv):
        sma = compute_sma(sample_ohlcv["Close"], windows=[5, 20])
        assert list(sma.columns) == ["sma_5", "sma_20"]

    def test_sma_5_equals_rolling_mean(self, sample_ohlcv):
        sma = compute_sma(sample_ohlcv["Close"], windows=[5])
        expected = sample_ohlcv["Close"].rolling(5).mean()
        pd.testing.assert_series_equal(sma["sma_5"], expected, check_names=False)


class TestEMA:
    def test_columns(self, sample_ohlcv):
        ema = compute_ema(sample_ohlcv["Close"], windows=[12, 26])
        assert list(ema.columns) == ["ema_12", "ema_26"]

    def test_no_nans_after_first(self, sample_ohlcv):
        ema = compute_ema(sample_ohlcv["Close"])
        # EWM produces valid values from the first row
        assert not ema.iloc[1:].isna().any().any()


class TestBollingerBands:
    def test_columns(self, sample_ohlcv):
        bb = compute_bollinger_bands(sample_ohlcv["Close"])
        for col in ["bb_upper", "bb_lower", "bb_width", "bb_pct_b"]:
            assert col in bb.columns

    def test_upper_gt_lower(self, sample_ohlcv):
        bb = compute_bollinger_bands(sample_ohlcv["Close"]).dropna()
        assert (bb["bb_upper"] >= bb["bb_lower"]).all()

    def test_width_non_negative(self, sample_ohlcv):
        bb = compute_bollinger_bands(sample_ohlcv["Close"]).dropna()
        assert (bb["bb_width"] >= 0).all()


class TestVolumeFeatures:
    def test_columns(self, sample_ohlcv):
        vf = compute_volume_features(sample_ohlcv)
        for col in ["volume_sma_20", "volume_ratio", "obv", "vwap_dev"]:
            assert col in vf.columns

    def test_obv_monotone_with_sign(self, sample_ohlcv):
        # OBV differences should have same sign as close changes
        vf = compute_volume_features(sample_ohlcv)
        close_diff = sample_ohlcv["Close"].diff().iloc[1:]
        obv_diff = vf["obv"].diff().iloc[1:]
        same_sign = np.sign(close_diff) == np.sign(obv_diff)
        # Allow small floating-point discrepancies at zero
        nonzero = close_diff != 0
        assert same_sign[nonzero].mean() > 0.95


class TestPriceFeatures:
    def test_columns(self, sample_ohlcv):
        pf = compute_price_features(sample_ohlcv)
        for col in ["returns_1d", "returns_5d", "hl_range", "gap"]:
            assert col in pf.columns

    def test_hl_range_non_negative(self, sample_ohlcv):
        pf = compute_price_features(sample_ohlcv).dropna()
        assert (pf["hl_range"] >= 0).all()


class TestLagFeatures:
    def test_columns(self, sample_ohlcv):
        lf = compute_lag_features(sample_ohlcv["Close"], periods=[1, 2, 3])
        assert list(lf.columns) == ["lag_ret_1", "lag_ret_2", "lag_ret_3"]

    def test_lag_1_is_shifted(self, sample_ohlcv):
        lf = compute_lag_features(sample_ohlcv["Close"], periods=[1])
        pct = sample_ohlcv["Close"].pct_change()
        pd.testing.assert_series_equal(
            lf["lag_ret_1"].dropna(), pct.shift(1).dropna(), check_names=False
        )


class TestBuildTarget:
    def test_regression_mode(self, sample_ohlcv):
        target = build_target(sample_ohlcv["Close"], horizon=1, mode="regression")
        assert target.name == "target"
        # Last row should be NaN (no future data)
        assert pd.isna(target.iloc[-1])

    def test_classification_mode(self, sample_ohlcv):
        target = build_target(sample_ohlcv["Close"], horizon=1, mode="classification")
        valid = target.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})
