"""Shared OHLCV schema — every data source (real or synthetic) and every
cache read must agree on this shape so downstream candidates never branch
on where the data came from.
"""
from __future__ import annotations

import pandas as pd

OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def validate_ohlcv_df(df: pd.DataFrame) -> bool:
    """True if `df` is a well-formed OHLCV frame.

    Checked, not assumed: this is what a cache-read validates before
    trusting a file on disk (see `forecasting_core.cache`) — a corrupt or
    truncated cache file must fail this, not be silently used.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    if list(df.columns) != OHLCV_COLUMNS:
        return False
    if df.empty:
        return False
    if df[OHLCV_COLUMNS].isna().any().any():
        return False
    if not (df["high"] >= df["low"]).all():
        return False
    if not (df["high"] >= df["open"]).all() or not (df["high"] >= df["close"]).all():
        return False
    if not (df["low"] <= df["open"]).all() or not (df["low"] <= df["close"]).all():
        return False
    if (df["volume"] < 0).any():
        return False
    dates = pd.to_datetime(df["date"])
    if not dates.is_monotonic_increasing:
        return False
    return True
