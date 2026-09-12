"""Atomic, validated on-disk cache for OHLCV frames.

- Writes are atomic: a temp file in the same directory, then `os.replace`
  onto the target — an interrupted write (retry exhausted, disk full,
  process killed) never leaves a truncated file at the real path.
- Reads are validated: a cached file that fails `schema.validate_ohlcv_df`
  is discarded and treated as a cache miss, not silently trusted — one bad
  write must not poison every future run.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .schema import validate_ohlcv_df

logger = logging.getLogger(__name__)


def read_ohlcv_cache(path: Path) -> Optional[pd.DataFrame]:
    """Return the cached frame at `path`, or None on a miss/invalid file.

    An invalid cached file (corrupt CSV, wrong columns, NaNs, ...) is
    deleted here so the caller re-fetches/regenerates instead of the bad
    file being read again on every future run.
    """
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except Exception as exc:  # noqa: BLE001 - any parse failure = cache miss
        logger.warning("Discarding unreadable OHLCV cache %s: %s", path, exc)
        path.unlink(missing_ok=True)
        return None

    if not validate_ohlcv_df(df):
        logger.warning("Discarding invalid OHLCV cache %s (failed schema validation)", path)
        path.unlink(missing_ok=True)
        return None
    return df


def write_ohlcv_cache_atomic(path: Path, df: pd.DataFrame) -> None:
    """Write `df` to `path` atomically (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
