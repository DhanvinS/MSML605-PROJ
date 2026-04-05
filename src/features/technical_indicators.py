"""Technical indicator computation functions.

All functions accept pandas Series/DataFrame inputs and return Series or DataFrames.
None of these functions modify the input data in place.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothed moving average.

    Args:
        close: Closing price series.
        period: Look-back period (default 14).

    Returns:
        RSI series in range [0, 100].
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's EMA: equivalent to EWM with alpha = 1/period, adjust=False
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename("rsi_14")


# ---------------------------------------------------------------------------
# Trend — Moving Averages
# ---------------------------------------------------------------------------

def compute_sma(close: pd.Series, windows: list[int] = None) -> pd.DataFrame:
    """Simple Moving Averages for multiple windows.

    Returns:
        DataFrame with columns sma_5, sma_10, sma_20, sma_50.
    """
    if windows is None:
        windows = [5, 10, 20, 50]
    return pd.DataFrame(
        {f"sma_{w}": close.rolling(window=w).mean() for w in windows}
    )


def compute_ema(close: pd.Series, windows: list[int] = None) -> pd.DataFrame:
    """Exponential Moving Averages (adjust=False for true Wilder-style EMA).

    Returns:
        DataFrame with columns ema_12, ema_26.
    """
    if windows is None:
        windows = [12, 26]
    return pd.DataFrame(
        {f"ema_{w}": close.ewm(span=w, adjust=False).mean() for w in windows}
    )


def compute_macd(close: pd.Series) -> pd.DataFrame:
    """MACD line and signal line.

    Returns:
        DataFrame with columns macd_line, macd_signal, macd_hist.
    """
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "macd_signal": signal,
            "macd_hist": macd_line - signal,
        }
    )


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def compute_bollinger_bands(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands.

    Returns:
        DataFrame with columns bb_upper, bb_lower, bb_width, bb_pct_b.
    """
    rolling = close.rolling(window=window)
    mid = rolling.mean()
    std = rolling.std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame(
        {
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_width": width,
            "bb_pct_b": pct_b,
        }
    )


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def compute_volume_features(ohlcv: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Volume-based features: SMA ratio, On-Balance Volume, VWAP deviation.

    Args:
        ohlcv: DataFrame with Open, High, Low, Close, Volume columns.
        window: Rolling window for volume SMA and VWAP.

    Returns:
        DataFrame with volume_sma_20, volume_ratio, obv, vwap_dev.
    """
    close = ohlcv["Close"]
    volume = ohlcv["Volume"]

    vol_sma = volume.rolling(window=window).mean()
    vol_ratio = volume / vol_sma.replace(0, np.nan)

    # On-Balance Volume
    sign = np.sign(close.diff()).fillna(0)
    obv = (sign * volume).cumsum()

    # VWAP deviation: (close - VWAP) / VWAP
    typical_price = (ohlcv["High"] + ohlcv["Low"] + close) / 3
    vwap = (typical_price * volume).rolling(window=window).sum() / \
           volume.rolling(window=window).sum().replace(0, np.nan)
    vwap_dev = (close - vwap) / vwap.replace(0, np.nan)

    return pd.DataFrame(
        {
            "volume_sma_20": vol_sma,
            "volume_ratio": vol_ratio,
            "obv": obv,
            "vwap_dev": vwap_dev,
        }
    )


# ---------------------------------------------------------------------------
# Price / Returns
# ---------------------------------------------------------------------------

def compute_price_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Price-derived features: returns, range, gap.

    Returns:
        DataFrame with returns_1d, returns_5d, hl_range, gap.
    """
    close = ohlcv["Close"]
    open_ = ohlcv["Open"]
    high = ohlcv["High"]
    low = ohlcv["Low"]

    returns_1d = close.pct_change(1)
    returns_5d = close.pct_change(5)
    hl_range = (high - low) / close.replace(0, np.nan)
    gap = (open_ - close.shift(1)) / close.shift(1).replace(0, np.nan)

    return pd.DataFrame(
        {
            "returns_1d": returns_1d,
            "returns_5d": returns_5d,
            "hl_range": hl_range,
            "gap": gap,
        }
    )


def compute_lag_features(
    close: pd.Series, periods: list[int] = None
) -> pd.DataFrame:
    """Lagged percentage-change features.

    Args:
        close: Closing price series.
        periods: List of lag periods (default 1..8).

    Returns:
        DataFrame with columns lag_ret_1 .. lag_ret_N.
    """
    if periods is None:
        periods = list(range(1, 9))
    pct = close.pct_change()
    return pd.DataFrame(
        {f"lag_ret_{p}": pct.shift(p) for p in periods}
    )


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

def build_target(
    close: pd.Series,
    horizon: int = 1,
    mode: str = "regression",
) -> pd.Series:
    """Construct the prediction target.

    Args:
        close: Closing price series.
        horizon: Number of bars forward (default 1).
        mode: "regression" returns forward return; "classification" returns
              direction (1 = up, 0 = down/flat).

    Returns:
        Target series (NaN for the last `horizon` rows).
    """
    forward_return = (close.shift(-horizon) - close) / close.replace(0, np.nan)
    if mode == "classification":
        return (forward_return > 0).astype(float).rename("target")
    return forward_return.rename("target")
