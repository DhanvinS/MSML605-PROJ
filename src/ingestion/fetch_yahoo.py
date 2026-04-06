"""Price data downloader using defeat-beta-api."""

import logging

import pandas as pd
from defeatbeta_api.data.ticker import Ticker

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}


def fetch_ohlcv(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV data for a single ticker from defeat-beta-api.

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
    logger.info("Fetching %s from defeatbeta-api (%s to %s)", ticker, start, end)

    if interval != "1d":
        raise ValueError(
            "defeatbeta-api price() currently supports daily data in this pipeline; "
            f"received interval={interval}"
        )

    raw_df = Ticker(ticker).price()
    df = _normalize_price_df(raw_df)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            f"{ticker}: expected DatetimeIndex after normalization, got {type(df.index).__name__}. "
            "Check the ticker symbol is valid (e.g. Meta is now 'META', not 'FB')."
        )
    df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]

    if df.empty:
        raise ValueError(f"No data returned for {ticker} [{start} - {end}]")

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


def _normalize_price_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    column_map = {
        "report_date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    missing = [c for c in column_map if c not in raw_df.columns]
    if missing:
        raise ValueError(f"defeatbeta-api response missing columns: {missing}")

    df = raw_df.rename(columns=column_map)[list(column_map.values())].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]]
