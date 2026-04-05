"""Unit tests for drift detection functions."""

import json

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detector import (
    analyze_drift,
    compute_psi,
    run_ks_test,
)
from src.training.baseline_capture import capture_baseline_stats


@pytest.fixture
def stable_baseline_and_live():
    """Same distribution — no drift expected."""
    np.random.seed(0)
    baseline = np.random.normal(0, 1, 1000)
    live = np.random.normal(0, 1, 200)
    return baseline, live


@pytest.fixture
def drifted_baseline_and_live():
    """Different distribution — drift expected."""
    np.random.seed(0)
    baseline = np.random.normal(0, 1, 1000)
    live = np.random.normal(3, 2, 200)  # shifted mean and wider std
    return baseline, live


class TestKSTest:
    def test_no_drift_on_same_distribution(self, stable_baseline_and_live):
        baseline, live = stable_baseline_and_live
        _, p_value, is_drifted = run_ks_test(baseline, live)
        assert not is_drifted
        assert p_value > 0.05

    def test_drift_detected_on_different_distribution(self, drifted_baseline_and_live):
        baseline, live = drifted_baseline_and_live
        _, p_value, is_drifted = run_ks_test(baseline, live)
        assert is_drifted
        assert p_value < 0.05

    def test_returns_tuple_of_three(self, stable_baseline_and_live):
        baseline, live = stable_baseline_and_live
        result = run_ks_test(baseline, live)
        assert len(result) == 3

    def test_insufficient_samples_returns_no_drift(self):
        baseline = np.random.normal(0, 1, 1000)
        live = np.random.normal(3, 2, 5)  # too few
        _, _, is_drifted = run_ks_test(baseline, live)
        assert not is_drifted


class TestPSI:
    def test_stable_distribution_has_low_psi(self, stable_baseline_and_live):
        baseline, live = stable_baseline_and_live
        psi = compute_psi(baseline, live)
        assert psi < 0.1

    def test_drifted_distribution_has_high_psi(self, drifted_baseline_and_live):
        baseline, live = drifted_baseline_and_live
        psi = compute_psi(baseline, live)
        assert psi > 0.2

    def test_psi_is_non_negative(self, stable_baseline_and_live):
        baseline, live = stable_baseline_and_live
        assert compute_psi(baseline, live) >= 0

    def test_insufficient_samples_returns_zero(self):
        baseline = np.random.normal(0, 1, 1000)
        live = np.array([1.0] * 5)  # too few
        assert compute_psi(baseline, live) == 0.0


class TestAnalyzeDrift:
    @pytest.fixture
    def baseline_stats(self):
        """Build baseline stats from a simple 2-feature DataFrame."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "feat_a": np.random.normal(0, 1, 500),
                "feat_b": np.random.normal(5, 2, 500),
            }
        )
        return capture_baseline_stats(df)

    def test_no_drift_same_distribution(self, baseline_stats):
        np.random.seed(42)
        live = pd.DataFrame(
            {
                "feat_a": np.random.normal(0, 1, 200),
                "feat_b": np.random.normal(5, 2, 200),
            }
        )
        report = analyze_drift(baseline_stats, live)
        assert not report.overall_drift_detected
        assert not report.trigger_retraining

    def test_drift_detected_on_shifted_data(self, baseline_stats):
        np.random.seed(42)
        live = pd.DataFrame(
            {
                "feat_a": np.random.normal(5, 1, 200),   # large shift
                "feat_b": np.random.normal(15, 2, 200),  # large shift
            }
        )
        report = analyze_drift(baseline_stats, live)
        assert report.overall_drift_detected
        assert report.trigger_retraining

    def test_report_has_all_features(self, baseline_stats):
        live = pd.DataFrame(
            {
                "feat_a": np.random.normal(0, 1, 100),
                "feat_b": np.random.normal(5, 2, 100),
            }
        )
        report = analyze_drift(baseline_stats, live)
        feature_names = {r.feature for r in report.feature_results}
        assert "feat_a" in feature_names
        assert "feat_b" in feature_names

    def test_to_dict_is_serialisable(self, baseline_stats):
        live = pd.DataFrame(
            {
                "feat_a": np.random.normal(0, 1, 100),
                "feat_b": np.random.normal(5, 2, 100),
            }
        )
        report = analyze_drift(baseline_stats, live)
        d = report.to_dict()
        # Must be JSON-serialisable
        json.dumps(d)
