"""Fetch fresh OHLCV data, build feature matrix, and upload to S3.

Runs as a SageMaker Processing Job before each retraining job.
Output is written to /opt/ml/processing/output/ and SageMaker uploads it to S3.
"""

import argparse
import logging
import os
from datetime import date, timedelta
from pathlib import Path

from src.features.feature_pipeline import build_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(os.environ.get("SM_OUTPUT_DIR", "/opt/ml/processing/output"))


def run(tickers: list[str], lookback_days: int) -> None:
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        logger.info("Ingesting %s (%s → %s)", ticker, start, end)
        df = build_feature_matrix(ticker, start, end)
        out_path = OUTPUT_DIR / f"{ticker}.parquet"
        df.to_parquet(out_path, index=True)
        logger.info("Wrote %d rows × %d cols to %s", len(df), df.shape[1], out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL"])
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=int(os.environ.get("LOOKBACK_DAYS", 730)),
        help="How many calendar days of history to fetch",
    )
    args = parser.parse_args()
    run(args.tickers, args.lookback_days)
