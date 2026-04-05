"""Model evaluation metrics for regression and direction accuracy."""

import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute a comprehensive suite of regression metrics.

    Args:
        y_true: Ground-truth forward returns.
        y_pred: Predicted forward returns.

    Returns:
        Dict with rmse, mae, mape, directional_accuracy, sharpe_proxy.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Remove rows where ground-truth is exactly 0 to avoid division errors in MAPE
    nonzero_mask = y_true != 0

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    if nonzero_mask.sum() > 0:
        mape = float(
            np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]))
            * 100
        )
    else:
        mape = float("nan")

    # Directional accuracy: fraction where sign matches
    directional_accuracy = float(
        np.mean(np.sign(y_true) == np.sign(y_pred))
    )

    # Annualised Sharpe proxy: treat each prediction as a position signal
    # sharpe = mean daily return / std daily return * sqrt(252)
    signed_returns = np.sign(y_pred) * y_true
    std = float(np.std(signed_returns))
    sharpe_proxy = (
        float(np.mean(signed_returns) / std * np.sqrt(252))
        if std > 0
        else float("nan")
    )

    metrics = {
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "mape": round(mape, 4),
        "directional_accuracy": round(directional_accuracy, 4),
        "sharpe_proxy": round(sharpe_proxy, 4),
        "n_samples": len(y_true),
    }

    logger.info("Metrics: %s", metrics)
    return metrics
