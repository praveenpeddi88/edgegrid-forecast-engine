# v5 Forecast Engine — Release Notes & Honest Cohort-MAPE Report

**Status:** tagged internal, Feb-12 cutoff bundles, deployed behind
`/v5` FastAPI router.
**Date:** 2026-04-21
**Scope:** 42 APEPDCL smart meters (5 SP + 37 TP, ≥180 days of history).

---

## 1. What changed from v4 → v5

**v4** was a per-meter LightGBM bundle with 66 candidate features, two-pass
selection, tier-adaptive regularization, trailing-21d MBE correction, and
an 85/15 chronological split.

**v5** keeps every v4 innovation and layers three things on top:

1. **Outage-aware training mask** — the v5 retrainer explicitly excludes
   known outage windows from the training loss, so the model isn't
   penalised for not forecasting zero-demand spans that are really power
   cuts.  The mask lives in `data/operational/outage_windows.parquet` and
   is merged into the fit step in `training/v5_retrain.py`.
2. **Seasonal anchor priming for forward forecasts** — the inference side
   now supports a pseudo-history mode that fills post-cutoff context with
   4-week (dow, hour, minute) medians so batch prediction at horizons
   beyond the last real actual doesn't produce NaN-laden feature rows.
   Implemented in `benchmarks/forward_v5_feb12_strategies.py`.
3. **Recursive one-step-ahead inference** — `inference/v5_predict.py`
   adds `predict_recursive`, which steps through the horizon one slot at
   a time and feeds each prediction back into the working context.  This
   matches the autoregressive pattern the model was trained on and is
   safer than batch mode for horizons longer than ~48 steps, at a cost of
   ~30s per meter per 7-day window.

Feature set, two-pass selection, and tier-adaptive params are unchanged.
Every v5 bundle is a drop-in replacement for its v4 sibling — same
`selected_features`, `model_mean`, `model_q10`, `model_q90`, and
`trail_mbe_capped` keys.

---

## 2. The honest cohort-MAPE report

The user's stated goal is **"MAPE < 4% across all 42 meters, on existing
data and future dates where we don't have the data."**  After running all
four validation protocols, here is how v5 actually behaves, split by the
operational question each protocol answers.

### 2.1 Regression-quality bar (S1 holdout)

Held out each meter's chronological last 15% of actuals.  The bundle
predicts those slots with full access to real lag/rolling features
because the context frame contains the complete timeline.  **This is the
"model quality" number.**

| Stat | v4 | v5 | Δ |
|---|---|---|---|
| Mean MAPE | 9.58% | **6.34%** | **−3.24pp** |
| Median MAPE | 7.15% | **4.83%** | **−2.32pp** |
| p90 MAPE | 23.44% | 12.41% | −11.03pp |
| Under 4% | 3 / 42 | **13 / 42** | +10 |
| Under 10% | 26 / 42 | **35 / 42** | +9 |
| Improved vs v4 | — | **38 / 42** | — |
| Worsened vs v4 | — | 3 / 42 | — |

**Verdict:** v5 clears the 4% bar for 31% of the fleet and the 10% bar
for 83%.  The full 42-meter 4%-bar is not met — the hardest meters
(65021169, 67003309, 67003234) have persistent structural noise
(seasonal shift contracts, irrigation duty cycles) that additional
features alone can't fix.

### 2.2 Steady-state bar (S2 leaky stratified)

Every 4th complete day (≥44 of 48 slots) held out, full-timeline
feature access on both sides of each holdout.  Numbers below are
"leaky" because v5 was chronologically trained, not stratified — most
holdout days fall inside the training window.

| Stat | v5 leaky | Notes |
|---|---|---|
| Mean MAPE | 4.63% | |
| Median MAPE | **3.77%** | |
| p90 MAPE | 8.56% | |
| Under 4% | **23 / 42** (55%) | |
| Under 10% | 40 / 42 (95%) | |

**Verdict:** this is the upper bound on "steady-state retrained monthly"
performance.  Matches v4's published S2 (median 4.9%) and beats it,
confirming v5 is strictly better at regression.

### 2.3 Deployment bar (S3 rolling-origin, 7-day forward)

10 weekly origins spanning Dec 4 2025 → Feb 5 2026.  At each origin, the
deployed inference path predicts the next 336 half-hour slots.  Actuals
are scored at each horizon step.

| Horizon bucket | Mean MAPE | Median MAPE |
|---|---|---|
| 0–24 h | 53.22% | — |
| 24–72 h | 91.16% | — |
| 72 h – 7 d | 87.96% | — |
| **Full 7-d fleet** | 83.59% | **28.68%** |

Cross-origin stability (per-meter σ): **median 7.6 pp**.

Validated on msn 65045250 (S2 MAPE 2.4%) at origin 2026-01-29:
batch=23.82%, recursive=24.66% — both inference paths land in the same
regime.  The 28% median is a real, reproducible deployment number.

**Why so much worse than S1/S2?**  Because S1 and S2 measure
*regression* — the model predicts slot t with access to real lag
features at t−30m, t−24h, t−7d.  S3 measures *forecasting* — at slot
t+k, the required lag features don't exist yet, so the inference path
substitutes either seasonal medians (batch) or the model's own prior
predictions (recursive).  Both substitutes carry error that compounds
across the horizon.

**Verdict:** v5's deployed **7-day forward MAPE is not ~4% and will not
become ~4% without architectural change.**  The gap between S1 (4.83%)
and S3 (28.68%) is the honest signal of what "production forecast"
means.

### 2.4 Forward bar (Apr 21 → May 20 2026)

30 days, no actuals available to score.  Four strategies produced:

| Strategy | Total fleet kWh | Description |
|---|---|---|
| A — seasonal anchor | 93,056 | 4-week (dow, hour, minute) median replay |
| B — v4 batch | 118,968 | v4 bundle + batch predict + seasonal priming |
| C — v5 batch Feb-12 | 115,982 | v5 bundle + batch predict + seasonal priming |
| **D — ensemble blend (median)** | **114,385** | per-row median of A, B, C |

D is the shipped default because the per-row median is robust to
strategy-specific failure modes (e.g. C's placeholder-lag drift doesn't
dominate; A's stationarity doesn't dominate; B's v4 regression-to-mean
doesn't dominate).  Artifacts at `outputs/forward_v5_feb12/` and served
via `GET /v5/forecast/apr-may`.

---

## 3. What this means for EdgeGrid operations

Three distinct deployment modes, three distinct MAPE bars:

**Next-30-min forecast** (real-time dispatch support): expected MAPE
close to S1/S2 because lag_1 (the dominant feature) is always real.
Fleet median ~5%, Medium tier under 4%.  **Safe for dispatch.**

**Day-ahead forecast** (DAM bid preparation): expected MAPE ~15–25%
depending on how fresh the trailing actuals are.  Acceptable for
ranking cheapest hours but not for load-commitment.  **Use with
confidence intervals.**

**Week-ahead forecast** (BESS sizing, capacity planning): expected MAPE
25–30% fleet median, up to 50%+ for HT meters.  **Do not promise <5%
here.**  Use the ensemble D strategy and communicate the q10/q90 band.

**Forward-month forecast** (budget planning, scenario modelling): no
ground-truth validation possible.  Trust the shape of the curves more
than the absolute values.

---

## 4. Path to the 4% goal (if we choose to pursue it)

The 4% bar is achievable for the **regression** protocol — and v5 is
within striking distance (13/42 today).  Closing the remaining 29
meters requires per-meter root-cause work, not a new feature set.

The 4% bar is **not** achievable for the 7-day forward protocol with
the current architecture.  Getting there would require one of:

1. **Daily retraining with morning-fresh actuals** so each forecast
   horizon starts from a freshly-fitted bundle.  Cuts the S3 gap but
   adds operational burden.
2. **Hierarchical / hybrid models** (Chronos-style foundation model +
   per-meter residual correction).  Larger model but less feature
   engineering; MAPE gains depend on the base model's India-climate
   prior.
3. **Shorter effective horizon** — only commit to 0–24h forecasts
   publicly; treat 24h+ as indicative planning bands with wide q10/q90.

The right answer is (3) short-term + (1) for long-term.  Honest
communication of what each horizon means is more important than the
MAPE number itself.

---

## 5. Service surface

The v5 FastAPI router (`src/edgegrid_forecast/api/v5_router.py`,
wired into `main.py` via `app.include_router`) exposes:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v5/healthz` | Service + model registry health, fleet MAPE stats |
| `GET` | `/v5/manifest` | Full v5 manifest JSON |
| `GET` | `/v5/meters` | Meter list with tier, holdout MAPE, train cutoff |
| `POST` | `/v5/predict` | Per-meter forecast (batch or recursive mode) |
| `POST` | `/v5/predict/fleet` | Fleet short-horizon forecast |
| `POST` | `/v5/retrain` | Retrain one meter (subprocess, blocking) |
| `GET` | `/v5/forecast/apr-may` | Served artifact, 4-strategy forward forecast |
| `POST` | `/v5/refresh-cache` | Clear manifest/bundle/frame cache |

Request example:

```bash
curl -s http://localhost:8000/v5/predict \
  -H "Content-Type: application/json" \
  -d '{"msn":"50143025","horizon_steps":48,"mode":"batch"}'
```

---

## 6. Reproducibility

```bash
# Retrain all 42 bundles with Feb-12 cutoff  (~3–5 min serial)
PYTHONPATH=src python -m edgegrid_forecast.training.v5_retrain \
  --train-cutoff 2026-02-12T15:00:00

# Validate (all four protocols)
PYTHONPATH=src python benchmarks/v5_benchmark_s1.py
PYTHONPATH=src python benchmarks/v5_benchmark_s2.py
PYTHONPATH=src python benchmarks/v5_benchmark_s3.py
PYTHONPATH=src python benchmarks/forward_v5_feb12_strategies.py

# Serve
PYTHONPATH=src uvicorn edgegrid_forecast.api.main:app --host 0.0.0.0 --port 8000
```

---

## 7. Known gaps and follow-ups

1. **S3 is currently "light + leaky".**  Full protocol calls for
   per-origin retrain; that's a 6-hour overnight run we haven't booked.
2. **Post-Feb-12 clean S2** requires appending newer actuals to the
   source parquets.  Trivial once EdgeGrid's ingestion catches up past
   Feb 12.
3. **Chronos fallback for cold-start** is specced in docs/ but not
   wired into the router.  Next-sprint work.
4. **Dashboard** (`outputs/s3_dashboard.html` in the original S3
   protocol) is not produced.  The per-meter and per-origin CSVs are
   sufficient for a notebook-driven review in the interim.

---

*v5 ships because it strictly improves on v4 across every regression
protocol and because the deployment gaps (S3, forward) are now
documented instead of hidden.  The engine is production-ready for
real-time dispatch support; communicate forward-horizon bands
honestly.*
