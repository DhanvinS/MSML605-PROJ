"""Drift detection using Kolmogorov-Smirnov test and Population Stability Index."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# PSI thresholds
PSI_STABLE = 0.1
PSI_INVESTIGATE = 0.2  # above this → trigger retraining

# Fraction of features that must show drift to flag overall drift
DRIFT_FEATURE_FRACTION_THRESHOLD = 0.30


@dataclass
class FeatureDriftResult:
    feature: str
    ks_statistic: float
    ks_p_value: float
    ks_drifted: bool
    psi: float
    psi_severity: str   # "stable" | "investigate" | "severe"


@dataclass
class DriftReport:
    timestamp: str
    feature_results: list[FeatureDriftResult]
    n_features_drifted: int
    drift_fraction: float
    overall_drift_detected: bool
    max_psi: float
    trigger_retraining: bool
    n_live_samples: int
    n_baseline_samples: int

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "overall_drift_detected": self.overall_drift_detected,
            "trigger_retraining": self.trigger_retraining,
            "drift_fraction": round(self.drift_fraction, 4),
            "n_features_drifted": self.n_features_drifted,
            "max_psi": round(self.max_psi, 4),
            "n_live_samples": self.n_live_samples,
            "n_baseline_samples": self.n_baseline_samples,
            "features": [
                {
                    "feature": r.feature,
                    "ks_statistic": round(r.ks_statistic, 4),
                    "ks_p_value": round(r.ks_p_value, 4),
                    "ks_drifted": r.ks_drifted,
                    "psi": round(r.psi, 4),
                    "psi_severity": r.psi_severity,
                }
                for r in self.feature_results
            ],
        }


def run_ks_test(
    baseline_data: np.ndarray,
    live_data: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, bool]:
    """Run two-sample KS test.

    Returns:
        (statistic, p_value, is_drifted)
    """
    if len(live_data) < 10:
        logger.warning("Insufficient live samples for KS test (%d)", len(live_data))
        return 0.0, 1.0, False

    statistic, p_value = stats.ks_2samp(baseline_data, live_data)
    is_drifted = bool(p_value < alpha)
    return float(statistic), float(p_value), is_drifted


def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index.

    PSI < 0.1  : no significant drift
    PSI 0.1-0.2: moderate drift, investigate
    PSI > 0.2  : severe drift, trigger retraining

    Args:
        baseline: 1-D array of baseline (training) values.
        current: 1-D array of current (live) values.
        n_bins: Number of bins (edges derived from baseline percentiles).

    Returns:
        PSI scalar value.
    """
    if len(current) < 10:
        return 0.0

    # Use baseline percentiles to define bin edges
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.percentile(baseline, percentiles))

    if len(bin_edges) < 2:
        return 0.0

    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    current_counts, _ = np.histogram(current, bins=bin_edges)

    baseline_pcts = baseline_counts / len(baseline)
    current_pcts = current_counts / len(current)

    # Clip to avoid log(0)
    baseline_pcts = np.clip(baseline_pcts, 1e-6, None)
    current_pcts = np.clip(current_pcts, 1e-6, None)

    psi = float(
        np.sum((current_pcts - baseline_pcts) * np.log(current_pcts / baseline_pcts))
    )
    return psi


def _psi_severity(psi: float) -> str:
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_INVESTIGATE:
        return "investigate"
    return "severe"


def analyze_drift(
    baseline_stats: dict,
    live_window_df: pd.DataFrame,
    ks_alpha: float = 0.05,
    psi_retrain_threshold: float = PSI_INVESTIGATE,
) -> DriftReport:
    """Run KS + PSI drift analysis for all features.

    Args:
        baseline_stats: Dict returned by baseline_capture.capture_baseline_stats().
        live_window_df: DataFrame of recent inference feature vectors.
        ks_alpha: KS test significance level.
        psi_retrain_threshold: PSI above which retraining is triggered.

    Returns:
        DriftReport with per-feature and overall results.
    """
    feature_stats = baseline_stats["features"]
    feature_results: list[FeatureDriftResult] = []
    n_live = len(live_window_df)

    for feat_name, stats_dict in feature_stats.items():
        if feat_name not in live_window_df.columns:
            logger.warning("Feature '%s' not found in live window, skipping", feat_name)
            continue

        live_data = live_window_df[feat_name].dropna().values

        # Reconstruct baseline samples from histogram
        hist = stats_dict["histogram"]
        edges = np.array(hist["bin_edges"])
        counts = np.array(hist["counts"])
        bin_mids = (edges[:-1] + edges[1:]) / 2
        baseline_samples = np.repeat(bin_mids, counts)

        ks_stat, ks_pval, ks_drifted = run_ks_test(baseline_samples, live_data, alpha=ks_alpha)
        psi_val = compute_psi(baseline_samples, live_data)

        feature_results.append(
            FeatureDriftResult(
                feature=feat_name,
                ks_statistic=ks_stat,
                ks_p_value=ks_pval,
                ks_drifted=ks_drifted,
                psi=psi_val,
                psi_severity=_psi_severity(psi_val),
            )
        )

    n_drifted = sum(1 for r in feature_results if r.ks_drifted)
    drift_fraction = n_drifted / len(feature_results) if feature_results else 0.0
    overall_drift = drift_fraction >= DRIFT_FEATURE_FRACTION_THRESHOLD
    max_psi = max((r.psi for r in feature_results), default=0.0)
    trigger_retrain = max_psi >= psi_retrain_threshold or overall_drift

    report = DriftReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        feature_results=feature_results,
        n_features_drifted=n_drifted,
        drift_fraction=drift_fraction,
        overall_drift_detected=overall_drift,
        max_psi=max_psi,
        trigger_retraining=trigger_retrain,
        n_live_samples=n_live,
        n_baseline_samples=baseline_stats.get("n_train_samples", 0),
    )

    logger.info(
        "Drift analysis: %d/%d features drifted (%.0f%%), max_PSI=%.3f, retrain=%s",
        n_drifted,
        len(feature_results),
        drift_fraction * 100,
        max_psi,
        trigger_retrain,
    )

    return report
