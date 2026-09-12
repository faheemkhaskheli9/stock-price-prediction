# Evaluation Notes: Stock Price Prediction

## Metrics

- **Skill vs. naive** — `1 - model_err / naive_err` on the walk-forward test
  windows (MAE-based).
- **MAE / RMSE** — standard regression error on price (or return) targets.
- **Directional accuracy** — fraction of windows where the predicted
  up/down direction matches the actual move.
- **Cost** — fit time, per-window predict latency, and serialized model size
  per candidate (wall-clock, relative — not an absolute SLA).
- **Pareto front** — candidates not dominated on (skill↑, fit_time↓,
  predict_latency↓, size↓), reported alongside the ranked leaderboard rather
  than folded into one blended score.

## Protocol

Walk-forward, not a single trailing holdout: the evaluation harness rolls
the train/test window forward across the series and aggregates per-window
metrics, so a lucky/unlucky single split can't dominate the ranking.

## Result Log

_Fill in as phases land — one row per evaluation run._

| Date | Phase | Candidate | Skill vs. naive | MAE | Fit time | Notes |
|------|-------|-----------|------------------|-----|----------|-------|
| _TBD_ | | | | | | |
