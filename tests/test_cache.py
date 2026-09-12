from __future__ import annotations

import pandas as pd
import pytest

from forecasting_core.cache import read_ohlcv_cache, write_ohlcv_cache_atomic
from forecasting_core.synthetic import generate_synthetic_ohlcv


def test_write_then_read_roundtrips(tmp_path):
    df = generate_synthetic_ohlcv(seed=0, n_bars=20)
    path = tmp_path / "cache.csv"

    write_ohlcv_cache_atomic(path, df)
    loaded = read_ohlcv_cache(path)

    assert loaded is not None
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True), df.reset_index(drop=True), check_dtype=False
    )


def test_missing_file_is_a_cache_miss(tmp_path):
    assert read_ohlcv_cache(tmp_path / "does-not-exist.csv") is None


def test_write_does_not_leave_a_tmp_file_behind(tmp_path):
    df = generate_synthetic_ohlcv(seed=0, n_bars=5)
    path = tmp_path / "cache.csv"
    write_ohlcv_cache_atomic(path, df)
    assert not (tmp_path / "cache.csv.tmp").exists()


def test_failed_write_cleans_up_tmp_and_reraises(tmp_path, monkeypatch):
    df = generate_synthetic_ohlcv(seed=0, n_bars=5)
    path = tmp_path / "cache.csv"

    def boom(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", boom)
    with pytest.raises(OSError):
        write_ohlcv_cache_atomic(path, df)

    assert not path.exists()
    assert not (tmp_path / "cache.csv.tmp").exists()


def test_corrupt_cache_file_is_discarded_not_trusted(tmp_path):
    path = tmp_path / "cache.csv"
    path.write_text("not,a,valid,ohlcv,csv\n1,2,3,4,5\n", encoding="utf-8")

    result = read_ohlcv_cache(path)

    assert result is None
    assert not path.exists()  # evicted so future reads don't hit the same bad file


def test_cache_with_missing_column_is_discarded(tmp_path):
    df = generate_synthetic_ohlcv(seed=0, n_bars=10).drop(columns=["volume"])
    path = tmp_path / "cache.csv"
    df.to_csv(path, index=False)

    assert read_ohlcv_cache(path) is None
    assert not path.exists()


def test_cache_with_nan_values_is_discarded(tmp_path):
    df = generate_synthetic_ohlcv(seed=0, n_bars=10)
    df.loc[2, "close"] = None
    path = tmp_path / "cache.csv"
    df.to_csv(path, index=False)

    assert read_ohlcv_cache(path) is None
