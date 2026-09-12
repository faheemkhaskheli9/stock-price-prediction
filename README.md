# Stock Price Prediction

> Time-Series/Finance portfolio project — independent open-source implementation.
> This is an original, from-scratch build. It is not affiliated with, and does not
> contain any code, prompts, data, or business logic from, any employer or client.

![status](https://img.shields.io/badge/status-planned-lightgrey)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## Relationship to `trading-ai-suite`

This portfolio's `trading-ai-suite` already ships a **Stock Price
Forecasting** feature — a Ridge-regression baseline on moving-average/
return/lag features. This project is deliberately a different angle, not a
duplicate: it's a **deep-learning sequence-model benchmark** —
LSTM/GRU/Transformer architectures compared against classical baselines
(including the suite's own Ridge approach) on the same walk-forward
evaluation protocol, reporting an accuracy-vs-compute-cost trade-off rather
than shipping a single production forecaster. Where useful, it reuses
`trading-ai-suite`'s OHLCV ingestion/caching code as a dependency rather
than re-implementing it.

## 1. Problem

Simple baselines (moving average, Ridge regression) are fast and hard to
beat on noisy financial series, but it's not obvious *how much*, if
anything, a sequence model buys you, and at what training/inference cost.
This project answers that empirically: train several sequence-model
architectures on the same point-in-time-correct features, evaluate them
walk-forward, and rank them for accuracy **and** cost against the
classical baselines — an honest "is deep learning worth it here" study.

## 2. Architecture

```text
OHLCV data (public/synthetic) -> point-in-time feature build
    -> [Naive/Ridge baselines | LSTM | GRU | Transformer]
    -> walk-forward evaluation -> leaderboard (skill-vs-naive, MAE, cost)
```

A shared `forecasting_core` layer owns data loading/caching, point-in-time
feature construction, and the walk-forward evaluation harness; each model
architecture is a registered candidate the harness can train and score
without any of the pipeline code changing.

## 3. Technology Stack

- Python, PyTorch (LSTM/GRU/Transformer sequence models)
- scikit-learn (baselines: naive, Ridge)
- Pandas, Matplotlib (data handling, charts)
- Pytest for tests; CPU-only by default (small models, short sequences)

## 4. Feature List

- Point-in-time OHLCV feature pipeline (no look-ahead leakage)
- Naive and Ridge baseline forecasters (first-class candidates, not an
  afterthought — the whole point is comparing against them)
- LSTM, GRU, and Transformer sequence-model candidates behind a shared
  training/scoring interface
- Walk-forward evaluation harness (not a single trailing holdout) producing
  a leaderboard: skill-vs-naive, MAE/RMSE, directional accuracy, fit time,
  inference latency, and a Pareto-front flag over accuracy vs. cost
- Actual-vs-predicted charts per candidate

## 5. Implementation Plan

1. Phase 1: `forecasting_core` — OHLCV loader/cache (reuse `trading-ai-suite`
   patterns), point-in-time feature builder, walk-forward split utility,
   project skeleton
2. Phase 2: Naive and Ridge baseline candidates wired through the shared
   training/scoring interface (establishes the comparison floor first)
3. Phase 3: LSTM and GRU sequence-model candidates
4. Phase 4: Transformer sequence-model candidate
5. Phase 5: Leaderboard (skill/cost/Pareto front), charts, Docker/compose,
   CI, docs, evaluation write-up

## Task Tracking

Work will be broken into phase-tagged user stories tracked as GitHub Issues,
not in this file. Implement Phase 1 issues first (later phases depend on it).
When you start one, add label `status:in-progress`. When you finish, close it
referencing the commit (e.g. `git commit -m "... Closes #4"`) and push.

## 6. Repository Structure

```text
stock-price-prediction/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── .env.example
├── docker/
├── docs/
│   ├── architecture.md
│   └── evaluation.md
├── src/
├── tests/
├── configs/
├── scripts/
├── notebooks/
├── examples/
├── assets/
└── .github/
    └── workflows/
```

## 7. Setup

```bash
git clone <this-repo-url>
cd stock-price-prediction
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt   # or: pip install -e .
cp .env.example .env
```

## 8. Dataset

Public OHLCV data (or a deterministic synthetic generator when no network
access is available/desired) — no proprietary, employer-owned, or
client-identifiable data is used. Same offline-by-default policy as
`trading-ai-suite`: runs on CPU with synthetic data with no API key needed.

## 9. Training / Execution

```bash
# Once Phase 1-2 land:
python -m forecasting_core.run --candidate ridge --ticker SYNTH
python -m forecasting_core.leaderboard   # once Phase 5 lands
```

## 10. Evaluation

Document evaluation metrics and how to reproduce them here (see
`docs/evaluation.md`): walk-forward skill-vs-naive, MAE/RMSE, directional
accuracy, and the accuracy-vs-cost Pareto front across all candidates.

## 11. Results

_To be filled in as the implementation progresses — leaderboard table and
actual-vs-predicted charts per candidate go here._

## 12. API

_This project is a benchmarking study, not a served API — no endpoints are
planned. If that changes, document it here._

## 13. Docker

```bash
docker build -t stock-price-prediction .
docker run stock-price-prediction
```

## 14. Tests

```bash
pytest tests/
```

## 15. Limitations

- This is a from-scratch, independent recreation built for portfolio
  purposes, not a trading system to run real money through.
- Performance numbers, once added, are based on public/synthetic datasets
  and are not representative of any production system's real-world results.
- Scaffold stage: no code has been implemented yet — see §5 Implementation
  Plan.

## 16. Future Work

- A real fundamentals/news feature layer once `trading-ai-suite`'s sentiment
  work matures enough to share.
- Ensemble candidates combining a baseline and a sequence model.
- Track open items as GitHub Issues.

## 17. Disclosure

This repository is an **independent open-source recreation inspired by the
kind of production systems I have worked on professionally**. It contains no
employer or client source code, prompts, datasets, credentials, architecture
diagrams, or business logic. All code, data, and documentation here are
original or built on publicly available datasets and open-source tools.

---
_Last updated: 2026-09-12_
