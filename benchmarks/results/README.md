# Benchmark results — v5 fleet (Feb-12 cutoff bundles)

This directory contains machine-readable artifacts from the four validation
protocols run against the v5 LightGBM ensemble on 42 smart meters.  All
three JSON/CSV pairs are reproducible from the scripts in `benchmarks/`.

## Index

| File | Produced by | Meters × cells | Purpose |
|---|---|---|---|
| `v5_benchmark_s1.{json,csv}` | `benchmarks/v5_benchmark_s1.py` | 42 × 1 holdout slice | S1 chronological: per-meter last-15% holdout MAPE vs v4 |
| `v5_benchmark_s2.{json,csv}` | `benchmarks/v5_benchmark_s2.py` | 42 × every 4th complete day | S2 stratified: every-4th-day holdout |
| `v5_benchmark_s3.{json,csv}` | `benchmarks/v5_benchmark_s3.py` | 42 × 10 weekly origins × 336-slot horizon | S3 rolling-origin (light, batch-mode, leaky) |
| `v5_benchmark_s3_by_meter.csv` | same | 42 meter rollups | per-meter stability σ across 10 origins |
| `v5_benchmark_s3_by_origin.csv` | same | 10 origin rollups | fleet cohort distribution per origin |

## How to read these numbers honestly

Each benchmark answers a **different operational question**.  Do not
compare MAPE across them as if they measured the same thing.

### S1 — `holdout_mape` (4.83% median, 13/42 under 4%)
> *"If I give the model real lag features and hold out the last 15% of
>  history, how accurate is its regression?"*

This is the **model-quality** number.  At each held-out slot, the feature
builder has full access to the meter's true lag values (lag_1, lag_24,
lag_336) because all actuals are present in the context frame.  The model
is not being asked to forecast — it's being asked to regress known inputs
to an unknown target.

**Use this to compare v4 vs v5, tier-to-tier, or model-version changes.**

### S2 — leaky stratified (3.77% median, 23/42 under 4%)
> *"If every 4th complete day is held out, with training days before AND
>  after each holdout day, how accurate is the regression?"*

Same pattern as S1 — lag features see real actuals.  Better than S1
because both sides of each holdout day carry valid context.  "Leaky" here
means the v5 bundle was chronologically trained through Feb-12, so holdout
days within that window overlap with training data (lag values the model
has effectively memorised).  A clean S2 requires retraining bundles with
stratified holdout masks.

**Use this as an upper bound on steady-state regression accuracy.**

### S3 — rolling-origin 7-day forward forecast (~28% median, 0/420 under 4%)
> *"If deployed right now and asked to forecast the next 7 days from
>  origin t, how accurate is the operational forecast?"*

At each origin, the model has actuals through t − 30 min.  It must predict
336 half-hour slots **without** knowing the actuals at those slots.  Lag
features rely on either seasonal priming (batch) or previously-emitted
predictions (recursive).  Validated on a sample meter: batch 23.82%,
recursive 24.66% at h=7d on msn 65045250.

**Use this as the honest deployment MAPE for multi-day forecasts.**

### Forward (Apr 21 → May 20 2026) — see `outputs/forward_v5_feb12/`
> *"What does the engine output for the month after our last actuals?"*

No actuals available to score against.  The four strategies (A seasonal
anchor, B v4 batch, C v5 batch Feb-12, D ensemble blend median) produce
calibration-comparable but unverifiable projections.  D (ensemble median)
is the most defensible default because the per-row median is robust to
strategy-specific failures.

## Cohort-MAPE report — where v5 actually lands

Reading S1 + S3 together:

| Cohort bar | Protocol | Definition | v5 result |
|---|---|---|---|
| **Model quality** | S1 holdout | regression on held-out 15% with full feature access | **median 4.83%** / 13 of 42 under 4% |
| **Steady state** | S2 leaky stratified | regression on every-4th-day with full-timeline feature access | **median 3.77%** / 23 of 42 under 4% |
| **Deployment 24h** | S3 h=0-24h mean (batch) | 24h forward, production inference path | **~53%** fleet mean (noisy due to zero-MAPE inflation on low-demand meters) |
| **Deployment 7d** | S3 h=72h-7d mean (batch) | 7-day forward | **~88%** fleet mean; median ≈ 28%  |

Honest headline for the EdgeGrid dashboard:

> **v5 achieves regression quality of 4.83% median MAPE and 13/42 meters
>  under the 4% bar.**  Its operational 7-day forecast MAPE is substantially
> worse (~28% median), which is the gap between **what the model knows
> when scoring** and **what it has access to when forecasting**.  Closing
> this gap requires either shorter effective horizons (daily retrain +
> rolling-forecast) or stronger recursive priming.

## Reproducibility

```bash
# S1 — ~15s serial
PYTHONPATH=src python benchmarks/v5_benchmark_s1.py

# S2 — ~120s serial (batch or split into --msns chunks for sandboxed envs)
PYTHONPATH=src python benchmarks/v5_benchmark_s2.py

# S3-light (batch) — ~3min serial / 7 chunks of ~30s each
PYTHONPATH=src python benchmarks/v5_benchmark_s3.py

# Forward forecast — ~25s
PYTHONPATH=src python benchmarks/forward_v5_feb12_strategies.py
```

All scripts write to `benchmarks/results/` (this dir) or
`outputs/forward_v5_feb12/` and are safe to re-run.
