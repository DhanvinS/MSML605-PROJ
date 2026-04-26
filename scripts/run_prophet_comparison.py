"""Compare ARIMA (classical SOTA time-series model) against XGBoost on stock returns.

ARIMA is a univariate baseline — it sees only the historical return series,
with no access to the 30 engineered features used by XGBoost. A meaningful
gap between XGBoost and ARIMA demonstrates the value of feature engineering.

Both models are evaluated on the same single-ticker test split so N is identical.
XGBoost is re-evaluated from the saved model files (models/latest/) rather than
reading from metrics.json, which aggregates all tickers.

Requires: pip install statsmodels

Usage:
    python scripts/run_prophet_comparison.py
    python scripts/run_prophet_comparison.py --ticker MSFT --data-dir data/processed
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.evaluate import compute_metrics
from src.training.time_series_split import train_test_split_temporal
from src.inference.predictor import StockPredictor

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
warnings.filterwarnings("ignore")  # suppress statsmodels convergence warnings

MODEL_DIR = Path("models/latest")


def load_ticker(data_dir: Path, ticker: str) -> pd.DataFrame:
    path = data_dir / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run ingestion first.")
    df = pd.read_parquet(path).sort_index()
    if "target" not in df.columns:
        raise ValueError(f"'target' column missing in {path}")
    return df


def run_arima(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Fit ARIMA(2,0,1) on training returns and forecast the test period.

    Returns are already stationary (d=0). Order (2,0,1) captures short-term
    autocorrelation typical in financial return series.
    Forecasts are made recursively — each actual value is appended to the
    history before the next step, mimicking real deployment conditions.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        print("ERROR: statsmodels is not installed.")
        print("Install it with: pip install statsmodels")
        sys.exit(1)

    history = list(train_df["target"].values)
    predictions = []

    for actual in test_df["target"].values:
        model = ARIMA(history, order=(2, 0, 1))
        result = model.fit()
        predictions.append(result.forecast(steps=1)[0])
        history.append(actual)

    return np.array(predictions)


def run_xgboost(test_df: pd.DataFrame) -> np.ndarray:
    """Evaluate the saved XGBoost model on the same test split."""
    if not (MODEL_DIR / "model.xgb").exists():
        raise FileNotFoundError(
            f"No model found in {MODEL_DIR}. Run bash scripts/train_local.sh first."
        )
    predictor = StockPredictor()
    predictor.load_from_dir(str(MODEL_DIR))
    feature_cols = [c for c in test_df.columns if c != "target"]
    X = test_df[feature_cols].values.tolist()
    p50, _, _ = predictor.predict_batch(X)
    return np.array(p50)


def print_comparison(ticker: str, arima_metrics: dict, xgb_metrics: dict | None) -> None:
    try:
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        use_tabulate = False

    rows = [
        ["arima(2,0,1) — univariate", "test",
         arima_metrics["rmse"], arima_metrics["mae"],
         arima_metrics["directional_accuracy"], arima_metrics["sharpe_proxy"],
         arima_metrics["n_samples"]],
    ]
    if xgb_metrics:
        rows.append([
            "xgboost — 30 features", "test",
            xgb_metrics["rmse"], xgb_metrics["mae"],
            xgb_metrics["directional_accuracy"], xgb_metrics["sharpe_proxy"],
            xgb_metrics["n_samples"],
        ])

    headers = ["Model", "Split", "RMSE", "MAE", "Dir. Acc.", "Sharpe", "N"]

    print(f"\n{'=' * 70}")
    print(f"ARIMA vs XGBOOST — {ticker} — TEST SET HOLDOUT")
    print("=" * 70)
    if use_tabulate:
        print(tabulate(rows, headers=headers, floatfmt=".4f", tablefmt="github"))
    else:
        print("\t".join(headers))
        for row in rows:
            print("\t".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row))

    print()
    print("Notes:")
    print("  - ARIMA(2,0,1): classical time-series model, uses only the return series")
    print("  - XGBoost: uses 30 engineered features (RSI, MACD, Bollinger Bands, lags, etc.)")
    print("  - Walk-forward evaluation: ARIMA refits on each new observation")
    print("  - Both models evaluated on identical single-ticker 90/10 temporal split")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="ARIMA vs XGBoost comparison")
    parser.add_argument("--ticker", default="AAPL", help="Ticker to evaluate (default: AAPL)")
    parser.add_argument("--data-dir", default="data/processed", help="Directory with processed parquet files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    df = load_ticker(data_dir, args.ticker)
    print(f"Loaded {len(df)} rows for {args.ticker}")

    train_df, test_df = train_test_split_temporal(df, test_size=0.10)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    print(f"Fitting ARIMA walk-forward over {len(test_df)} test steps (takes ~30-60 seconds)...")
    arima_pred = run_arima(train_df, test_df)
    arima_metrics = compute_metrics(test_df["target"].values, arima_pred)

    print("Evaluating XGBoost on the same test split...")
    xgb_metrics = None
    try:
        xgb_pred = run_xgboost(test_df)
        xgb_metrics = compute_metrics(test_df["target"].values, xgb_pred)
    except FileNotFoundError as e:
        print(f"Warning: {e} — XGBoost row will be omitted.")

    print_comparison(args.ticker, arima_metrics, xgb_metrics)


if __name__ == "__main__":
    main()
