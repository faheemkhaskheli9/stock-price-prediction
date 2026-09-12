# Architecture Notes: Stock Price Prediction

## Pipeline

```text
OHLCV data (public/synthetic) -> point-in-time feature build
    -> [Naive/Ridge baselines | LSTM | GRU | Transformer]
    -> walk-forward evaluation -> leaderboard (skill-vs-naive, MAE, cost)
```

## Components

- `forecasting_core.data` — OHLCV loading/caching, reusing `trading-ai-suite`'s
  atomic/validated disk-cache pattern rather than re-implementing it
- `forecasting_core.features` — point-in-time feature construction (no
  look-ahead leakage), following the `ml/configurable-model-training`
  knowledge-base pattern's availability-timestamp rule
- `forecasting_core.candidates` — a registered candidate per architecture
  (naive, ridge, lstm, gru, transformer) behind one train/score interface
- `forecasting_core.evaluate` — walk-forward split utility (not a single
  trailing holdout) and a leaderboard builder with a Pareto-front flag over
  accuracy vs. cost, following the `ml/model-search-and-selection`
  knowledge-base pattern

## Design Notes

- Baselines (naive, Ridge) are first-class candidates in the same registry
  as the deep-learning ones — the whole project is a comparison, so the
  baseline must go through the identical pipeline, not a special-cased path.
- Build the feature set once per split and reuse it across every candidate
  scored on that split, per the model-search-and-selection KB pattern's
  efficiency argument.
- Record fit time, predict latency, and model size per candidate the same
  way the KB pattern's cost metrics are computed (perf_counter + serialized
  size), so the leaderboard's cost columns are directly comparable.
- Torch-based candidates (LSTM/GRU/Transformer) should soft-import torch and
  register as `available = False` if it's missing, per the
  `configurable-model-training` KB pattern's optional-heavy-deps rule.
- Reuse `trading-ai-suite`'s OHLCV ingestion/caching code as a dependency
  where practical instead of re-implementing it — see that suite's
  `trading_core` app. **Update (Phase 1, issue #1):** as of this issue,
  `trading-ai-suite`'s `src/` is empty (flagged in `STATUS.md` — issues
  closed there with no matching code), so there was nothing to import from.
  `forecasting_core.cache` reuses the *pattern* described above (atomic
  temp-file-then-rename write, validate-on-read with eviction of a bad
  cache file) implemented fresh here. If `trading-ai-suite` gets real code
  later, reconcile the two rather than keeping duplicate cache
  implementations.

## Consolidation note

This project is standalone (not a suite) — it depends on `trading-ai-suite`
for OHLCV data plumbing but is not being merged into it, since its purpose
(a deep-learning benchmark study) is distinct from that suite's production
forecasting feature.
