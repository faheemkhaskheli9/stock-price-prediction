"""Deterministic synthetic OHLCV generator (issue #2).

The default `OHLCV_SOURCE` — lets the whole pipeline run and be tested with
no network access or API key. Same seed always produces the same series
(byte-for-byte), matching the schema in `forecasting_core.schema` exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import OHLCV_COLUMNS


def generate_synthetic_ohlcv(
    seed: int = 0,
    n_bars: int = 252,
    start_price: float = 100.0,
    start_date: str = "2020-01-01",
    daily_drift: float = 0.0003,
    daily_vol: float = 0.015,
) -> pd.DataFrame:
    """Generate a deterministic OHLCV series via a seeded geometric random walk.

    `seed` fully determines the output — no wall-clock, no global RNG state
    read. `high`/`low` are derived from `close` with a small seeded
    intrabar range so `high >= max(open, close)` and `low <= min(open, close)`
    always hold (see `schema.validate_ohlcv_df`).
    """
    if n_bars <= 0:
        raise ValueError("n_bars must be positive")
    if start_price <= 0:
        raise ValueError("start_price must be positive")

    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=daily_drift, scale=daily_vol, size=n_bars)
    close = start_price * np.exp(np.cumsum(log_returns))
    open_ = np.empty(n_bars)
    open_[0] = start_price
    open_[1:] = close[:-1]

    intrabar_range = np.abs(rng.normal(loc=0.0, scale=daily_vol / 2, size=n_bars)) * close
    high = np.maximum(open_, close) + intrabar_range
    low = np.minimum(open_, close) - intrabar_range
    low = np.clip(low, a_min=0.01, a_max=None)

    volume = rng.integers(low=1_000, high=1_000_000, size=n_bars)
    dates = pd.date_range(start=start_date, periods=n_bars, freq="D")

    df = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    return df[OHLCV_COLUMNS]
