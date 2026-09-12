"""OHLCV loader (issue #1 / #14): source-agnostic, cached, atomic, validated.

`OHLCV_SOURCE` selects where data comes from:
- `"synthetic"` (default) — the deterministic generator in
  `forecasting_core.synthetic`, no network access required.
- `"csv:<path>"` — a local CSV already in the OHLCV schema, standing in for
  a downloaded public dataset. Fetching a *live* public source over the
  network is out of scope for this offline, CPU-only pipeline and is not
  implemented; `csv:<path>` is the documented boundary a real fetch would
  sit behind (mocked here per the project's CPU-only/no-network constraint).

Every source is written through the same atomic, validated cache
(`forecasting_core.cache`) so repeated loads don't re-generate/re-read the
source file, and a corrupted cache entry is never trusted.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .cache import read_ohlcv_cache, write_ohlcv_cache_atomic
from .schema import validate_ohlcv_df
from .synthetic import generate_synthetic_ohlcv

DEFAULT_CACHE_DIR = Path("data/ohlcv_cache")


def _cache_key(source: str, seed: int, n_bars: int) -> str:
    raw = f"{source}::{seed}::{n_bars}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_ohlcv(
    source: Optional[str] = None,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    seed: int = 0,
    n_bars: int = 252,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load an OHLCV frame from `source` (or `OHLCV_SOURCE`, default
    "synthetic"), through the atomic/validated cache.

    A `csv:<path>` source that doesn't exist or fails schema validation
    raises loudly rather than silently falling back to synthetic data —
    the caller named that source, so hiding the mistake would be worse
    than a crash (see CLAUDE.md rule 7: strict with explicit input).
    """
    source = source or os.environ.get("OHLCV_SOURCE", "synthetic")
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{_cache_key(source, seed, n_bars)}.csv"

    if use_cache:
        cached = read_ohlcv_cache(cache_path)
        if cached is not None:
            return cached

    if source == "synthetic":
        df = generate_synthetic_ohlcv(seed=seed, n_bars=n_bars)
    elif source.startswith("csv:"):
        csv_path = Path(source[len("csv:"):])
        if not csv_path.exists():
            raise FileNotFoundError(f"OHLCV_SOURCE csv path not found: {csv_path}")
        df = pd.read_csv(csv_path)
        if not validate_ohlcv_df(df):
            raise ValueError(
                f"OHLCV_SOURCE csv file failed schema validation: {csv_path}"
            )
    else:
        raise ValueError(
            f"Unknown OHLCV_SOURCE {source!r}; supported: 'synthetic', 'csv:<path>'"
        )

    if use_cache:
        write_ohlcv_cache_atomic(cache_path, df)
    return df
