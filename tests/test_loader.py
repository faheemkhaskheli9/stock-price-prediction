from __future__ import annotations

import pandas as pd
import pytest

from forecasting_core.loader import load_ohlcv
from forecasting_core.schema import validate_ohlcv_df
from forecasting_core.synthetic import generate_synthetic_ohlcv


def test_defaults_to_synthetic_with_no_network(tmp_path, monkeypatch):
    monkeypatch.delenv("OHLCV_SOURCE", raising=False)
    df = load_ohlcv(cache_dir=tmp_path, seed=0, n_bars=30)
    assert validate_ohlcv_df(df) is True


def test_env_var_selects_source(tmp_path, monkeypatch):
    monkeypatch.setenv("OHLCV_SOURCE", "synthetic")
    df = load_ohlcv(cache_dir=tmp_path, seed=1, n_bars=10)
    assert len(df) == 10


def test_explicit_source_overrides_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("OHLCV_SOURCE", "csv:/nonexistent/path.csv")
    df = load_ohlcv("synthetic", cache_dir=tmp_path, seed=0, n_bars=5)
    assert len(df) == 5


def test_second_load_hits_cache_without_regenerating(tmp_path, monkeypatch):
    calls = []
    original = generate_synthetic_ohlcv

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr("forecasting_core.loader.generate_synthetic_ohlcv", spy)

    first = load_ohlcv("synthetic", cache_dir=tmp_path, seed=0, n_bars=15)
    second = load_ohlcv("synthetic", cache_dir=tmp_path, seed=0, n_bars=15)

    assert len(calls) == 1  # second load served from cache
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True), second.reset_index(drop=True), check_dtype=False
    )


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        load_ohlcv("not-a-real-source")


def test_missing_csv_source_raises_not_silently_falls_back(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ohlcv("csv:/nonexistent/path.csv", cache_dir=tmp_path)


def test_invalid_csv_source_raises_not_silently_falls_back(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_ohlcv(f"csv:{bad_csv}", cache_dir=tmp_path / "cache")


def test_valid_csv_source_loads_and_caches(tmp_path):
    source_df = generate_synthetic_ohlcv(seed=7, n_bars=12)
    csv_path = tmp_path / "public_dump.csv"
    source_df.to_csv(csv_path, index=False)

    df = load_ohlcv(f"csv:{csv_path}", cache_dir=tmp_path / "cache")

    assert validate_ohlcv_df(df) is True
    assert len(df) == 12
