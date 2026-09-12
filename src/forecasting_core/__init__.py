"""Shared data/feature/evaluation layer for the forecasting-candidate
benchmark.

Phase 1 covers OHLCV loading with an atomic, validated on-disk cache
(`forecasting_core.cache`, `forecasting_core.loader`), a deterministic
offline data source (`forecasting_core.synthetic`), and the shared schema
they agree on (`forecasting_core.schema`). Feature construction, the
walk-forward split, and the candidate registry land in later Phase 1/2
issues on top of this layer.
"""
