"""Alpha Vantage data downloader with automatic rate-limit retry."""

import logging
import os
import time

import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://www.alphavantage.co/query"
# Free tier: 5 requests/minute, 500/day
_RATE_LIMIT_PAUSE = 12  # seconds between calls to stay under 5/min


class AlphaVantageError(Exception):
    pass


@retry(
    retry=retry_if_exception_type(AlphaVantageError),
    wait=wait_exponential(multiplier=1, min=12, max=60),
    stop=stop_after_attempt(5),
)
def _request(params: dict) -> dict:
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise EnvironmentError("ALPHAVANTAGE_API_KEY environment variable not set")

    params["apikey"] = api_key
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "Note" in data:
        # Rate limit note from Alpha Vantage
        raise AlphaVantageError(f"Rate limit hit: {data['Note']}")
    if "Error Message" in data:
        raise AlphaVantageError(f"API error: {data['Error Message']}")

    return data


def fetch_ohlcv(
    ticker: str,
    outputsize: str = "full",
) -> pd.DataFrame:
    """Fetch daily adjusted OHLCV data from Alpha Vantage.

    Args:
        ticker: Stock symbol, e.g. "AAPL".
        outputsize: "full" (20 years) or "compact" (last 100 days).

    Returns:
        DataFrame with columns Open, High, Low, Close, Volume indexed by date.
    """
    logger.info("Fetching %s from Alpha Vantage (outputsize=%s)", ticker, outputsize)

    data = _request(
        {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": outputsize,
            "datatype": "json",
        }
    )

    ts_key = "Time Series (Daily)"
    if ts_key not in data:
        raise AlphaVantageError(f"Unexpected response keys: {list(data.keys())}")

    time_series = data[ts_key]
    records = []
    for date_str, values in time_series.items():
        records.append(
            {
                "Date": pd.to_datetime(date_str),
                "Open": float(values["1. open"]),
                "High": float(values["2. high"]),
                "Low": float(values["3. low"]),
                "Close": float(values["5. adjusted close"]),  # adjusted
                "Volume": float(values["6. volume"]),
            }
        )

    df = pd.DataFrame(records).set_index("Date").sort_index()
    time.sleep(_RATE_LIMIT_PAUSE)  # respect free-tier rate limit
    logger.info("Fetched %d rows for %s from Alpha Vantage", len(df), ticker)
    return df
