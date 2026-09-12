from __future__ import annotations

import pandas as pd
import pytest

from forecasting_core.schema import OHLCV_COLUMNS, validate_ohlcv_df
from forecasting_core.synthetic import generate_synthetic_ohlcv


def test_same_seed_is_deterministic():
    a = generate_synthetic_ohlcv(seed=42, n_bars=50)
    b = generate_synthetic_ohlcv(seed=42, n_bars=50)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_differs():
    a = generate_synthetic_ohlcv(seed=1, n_bars=50)
    b = generate_synthetic_ohlcv(seed=2, n_bars=50)
    assert not a["close"].equals(b["close"])


def test_output_matches_ohlcv_schema():
    df = generate_synthetic_ohlcv(seed=0, n_bars=100)
    assert list(df.columns) == OHLCV_COLUMNS
    assert validate_ohlcv_df(df) is True


def test_n_bars_controls_length():
    df = generate_synthetic_ohlcv(seed=0, n_bars=30)
    assert len(df) == 30


def test_invalid_n_bars_raises():
    with pytest.raises(ValueError):
        generate_synthetic_ohlcv(n_bars=0)


def test_invalid_start_price_raises():
    with pytest.raises(ValueError):
        generate_synthetic_ohlcv(start_price=0)
