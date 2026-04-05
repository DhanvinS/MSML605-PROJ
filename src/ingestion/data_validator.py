"""Schema and data-quality validation for OHLCV DataFrames."""

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_ohlcv(
    df: pd.DataFrame,
    ticker: str = "unknown",
    max_null_pct: float = 0.05,
    min_rows: int = 100,
) -> ValidationResult:
    """Run all OHLCV quality checks and return a ValidationResult.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        ticker: Symbol name used in messages.
        max_null_pct: Maximum allowed fraction of null values per column.
        min_rows: Minimum required row count.

    Returns:
        ValidationResult with .valid flag and lists of errors / warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"{ticker}: missing required columns: {missing}")

    if errors:
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    # 2. Row count
    if len(df) < min_rows:
        errors.append(f"{ticker}: only {len(df)} rows, need at least {min_rows}")

    # 3. Null fractions
    for col in REQUIRED_COLUMNS:
        null_pct = df[col].isnull().mean()
        if null_pct > max_null_pct:
            errors.append(
                f"{ticker}.{col}: null fraction {null_pct:.1%} > {max_null_pct:.0%}"
            )
        elif null_pct > 0:
            warnings.append(f"{ticker}.{col}: {null_pct:.1%} nulls present")

    # 4. Price sanity
    neg_prices = (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).sum()
    if neg_prices > 0:
        errors.append(f"{ticker}: {neg_prices} rows with non-positive prices")

    # 5. OHLC relationship
    bad_hl = (df["High"] < df["Low"]).sum()
    if bad_hl > 0:
        errors.append(f"{ticker}: {bad_hl} rows where High < Low")

    # 6. Index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append(f"{ticker}: index must be DatetimeIndex, got {type(df.index)}")
    else:
        # 7. Check for duplicate timestamps
        dups = df.index.duplicated().sum()
        if dups > 0:
            warnings.append(f"{ticker}: {dups} duplicate timestamps in index")

    # 8. Negative volume
    neg_vol = (df["Volume"] < 0).sum()
    if neg_vol > 0:
        errors.append(f"{ticker}: {neg_vol} rows with negative volume")

    valid = len(errors) == 0
    if valid:
        logger.info("%s: validation passed (%d rows)", ticker, len(df))
    else:
        for err in errors:
            logger.error(err)
    for warn in warnings:
        logger.warning(warn)

    return ValidationResult(valid=valid, errors=errors, warnings=warnings)


def assert_valid(df: pd.DataFrame, ticker: str = "unknown") -> None:
    """Validate and raise ValueError on any errors."""
    result = validate_ohlcv(df, ticker)
    if not result.valid:
        raise ValueError(
            f"OHLCV validation failed for {ticker}:\n"
            + "\n".join(result.errors)
        )
