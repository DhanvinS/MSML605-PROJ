"""Yahoo Finance data downloader using yfinance."""

import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}


def fetch_ohlcv(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV data for a single ticker from Yahoo Finance.

    Args:
        ticker: Stock symbol, e.g. "AAPL".
        start: Start date string "YYYY-MM-DD".
        end: End date string "YYYY-MM-DD".
        interval: Bar interval — "1d", "1h", "5m", etc.

    Returns:
        DataFrame with columns Open, High, Low, Close, Volume indexed by datetime.

    Raises:
        ValueError: If the download returns empty or malformed data.
    """
    logger.info("Fetching %s from %s to %s (interval=%s)", ticker, start, end, interval)

    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker} [{start} – {end}]")

    # yfinance may return MultiIndex columns when fetching multiple tickers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    _validate(df, ticker)
    logger.info("Fetched %d rows for %s", len(df), ticker)
    return df


def fetch_multiple(
    tickers: list[str],
    start: str,
    end: str,
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple tickers. Returns {ticker: DataFrame}."""
    return {t: fetch_ohlcv(t, start, end, interval) for t in tickers}


def _validate(df: pd.DataFrame, ticker: str) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing}")

    null_pct = df.isnull().mean().max()
    if null_pct > 0.05:
        raise ValueError(
            f"{ticker}: null fraction {null_pct:.1%} exceeds 5% threshold"
        )

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
