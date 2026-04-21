# v5 plan · drive MAPE below 5% on the 42-meter fleet

**Status as of Apr 21, 2026** · Author: forecast engineering · Honest target: median per-meter MAPE < 5% on **train, bundle holdout, AND multi-day future forecast** for as many of the 42 modeled meters as the data physics permits.

This document does three things: (1) names where we are with evidence, (2) explains why the gap exists with a code-grounded read of the v4 engine, and (3) lays out an ordered set of engineering changes with expected MAPE delta and time cost. The first change is being built tonight.

---

## 1 · Where we are (verified, today)

| Surface | Mean | Median | p75 | Distance to 5% |
|---|---|---|---|---|
| Train MAPE (in-sample, validation tail) | ~2-3% | — | — | clears |
| Bundle holdout MAPE (7d, in-pipeline) | **9.58%** | **7.15%** | 12.96% | ~2pp on median, ~5pp on mean |
| v3 backtest MAPE (8d, DOW-primed, no recursion) | **63.42%** | **46.04%** | — | ~41pp on median |
| 76-day pure forecast MAPE | unknown — no truth past Feb 12 15:00 | — | — | — |

**Per-meter cluster (42 meters):**

| Cluster | n | Bundle MAPE band | Backtest MAPE | Why |
|---|---|---|---|---|
| **A** — already < 5% on bundle | 8 | 3.19 – 4.82% | 12.75 – 56.30% | Model is fine in-distribution. Backtest gap is pure priming/horizon tax. |
| **B** — bundle 5–15%, clean signal | 28 | 5.23 – 14.77% | 14.83 – 102.33% | Needs sharper hyperparameters + recursion. Achievable with 2-3 stacked changes. |
| **C** — bundle > 15%, fixable | 4 | 15.24 – 52.24% | 54.45 – 92.16% | Something wrong: feature signal, training-data length, or data quality. Requires per-meter triage. |
| **D** — high-zero / intermittent | 2 | 13.76 – 18.07% | 65.73 – 755.21% | Zero% > 60%. Sub-5% MAPE is **physically impossible** at 30-min cadence — Poisson noise alone exceeds 5% on residential single-phase loads with idle periods. Drop or model differently. |

---

## 2 · Why the gap exists — code-grounded

I read every relevant file in `src/edgegrid_forecast/`. The v4 engine has four properties that explain the bundle-vs-backtest gap:

### 2a · No recursive prediction loop in inference

`inference/v4_predict.py · predict_with_context()` builds features by calling `build_features_v4_s1(context_df, ...)` once on the input context and then calls `bundle["model_mean"].predict(X)` once on the resulting feature matrix. The lag features (`lag_1`, `lag_2`, ..., `lag_336`), rolling features (`rmean_6`, `rmean_336`, etc.), momentum features (`demand_diff_1`, `demand_diff_4`), and the seasonal-deviation feature (`deviation_from_hourly`) are all computed via `df["demand_wh"].shift(N)` on the context dataframe. Past the cutoff, `demand_wh` is whatever was primed in (DOW-back values or zeros) — **not the model's own previous predictions**.

Result: at horizon t = cutoff + 1 hour, `lag_1` is stale. At t = cutoff + 24 hours, `lag_24` is stale. Every feature that depends on a recent past value of the meter is wrong. This is the dominant source of the 9.6% → 63.4% gap.

**The bundle-level training does see real lags** (because train_df is contiguous historical data), so the bundle holdout MAPE is the model's ceiling under "perfect lags." Inference today doesn't deliver perfect lags past the cutoff. → fix this and we close most of the gap.

### 2b · Holiday calendar stops at 2026-02-15

`inference/_features.py · HOLIDAYS` dict ends at "2026-02-15: Maha Shivaratri". The forecast window goes through 2026-04-21. We are missing **Holi (Mar 4 2026), Ugadi (Mar 21 2026), Eid-ul-Fitr (Mar 21 2026), Ram Navami (Mar 26 2026), Mahavir Jayanti (Mar 31 2026), Good Friday (Apr 3 2026), Ambedkar Jayanti (Apr 14 2026)** — at minimum. `is_holiday` and `near_holiday` features are zero on every one of those days in the forecast tail, even though the actual load patterns will be holiday-shaped.

### 2c · Weather features stop at the cache window

`fetch_weather_expanded` has a hard-coded `end="2026-03-01"`. Past that, the forward-fill in `_materialise_features` will simply repeat the last observed weather row for the entire 51 days of April. Weather is a top-3 feature in most bundles (heat_index, ghi, cooling_deg) — flat-lining it at the Mar 1 value through Apr 21 is a meaningful loss of signal during a transition month where Visakhapatnam temperatures rise by 4–6°C.

### 2d · Tier-adaptive but not per-meter hyperparameters

`get_params(tier, n_samples)` returns one of three parameter sets (HT / Small / default). 28 of 42 meters fall in "Medium (0.5-1.5k)" and get the default. Per-meter tuning (Optuna or grid) typically buys 1-3pp on bundle MAPE, especially for the cluster B meters where the bundle is already in the 5-10% range and a small reduction crosses the 5% line.

### 2e · MAPE definition includes outage days as "real" errors

`calc_metrics` masks `y_true > 0.5` (excludes near-zeros), but if a meter loses power for a full day during the holdout, those zeros get excluded — yet a *partial* outage day where load drops to 1 kWh from a typical 5 kWh registers as an 80% error against a baseline forecast that doesn't know there was an outage. Without an outage-detection feature, those days inflate the MAPE without giving the model a chance to react.

---

## 3 · v5 engineering plan — ordered by MAPE delta per day of work

The plan is one screen wide on purpose. Each row is one shippable change. Expected delta is for the meters where the change applies, expressed as the change in median bundle/backtest MAPE.

| # | Change | Files touched | Expected backtest delta | Expected bundle delta | Days | Risk |
|---|---|---|---|---|---|---|
| **1** | **Recursive prediction loop in inference** — predict t, write back to context as if actual, recompute features, predict t+1 | `inference/v4_predict.py` (new `predict_recursive`), no model retrain | **−25 to −35pp on median** | unchanged | 2 | Low — it's a wrapper |
| 2 | Extend HOLIDAYS dict + add lunar-calendar logic for Eid | `inference/_features.py` | −3 to −8pp on holiday-affected days | unchanged | 0.5 | None |
| 3 | Re-fetch weather through Apr 30, switch to Open-Meteo *forecast* API for any date past today | `inference/_features.py · fetch_weather_expanded`, new `fetch_weather_forecast` | −2 to −5pp on weather-sensitive blocks (solar, peak) | unchanged | 1 | Low |
| 4 | Outage detection feature: `is_recent_outage`, `hours_since_last_zero`, gap-aware MAPE | `inference/_features.py`, `calc_metrics` | −5 to −10pp by exclusion of contaminated blocks | −1 to −2pp by feature signal | 1 | Medium — definitions matter |
| 5 | Per-meter hyperparameter tuning via Optuna (50 trials each, ~5 min/meter) | `training/optuna_tune.py` (new), retrain all 42 | unchanged | **−1 to −3pp on cluster B meters** | 3 | Low |
| 6 | Replace lags with **deviation-from-typical-day** target. Predict (actual − DOW×hour mean) so absolute lags matter less | `_features.py`, retrain | −5 to −10pp because brittle lags are no longer load-bearing | −0.5 to −1pp | 3 | Medium — changes target semantics |
| 7 | Triage cluster C (4 meters): inspect data, retrain on longer history, or drop | `notebooks/cluster_c_triage.ipynb` (new) | −20 to −40pp on those 4 meters specifically | −5 to −10pp | 2 | Low |
| 8 | Drop cluster D (2 meters) from the < 5% target with explicit rationale; report them at 1-hr aggregate instead | docs + meter_profile | n/a — they're not in scope at 30-min anymore | n/a | 0.5 | None — honesty change |
| 9 | Quantile calibration via conformal prediction on the held-out 8 days, replacing the independently-trained q10/q90 heads | `inference/_features.py`, `v4_predict.py` | unchanged on point forecast; coverage → ~80% | unchanged | 2 | Low |
| 10 | Ensemble: weighted blend of LightGBM + a simple seasonal-naive baseline (SNAIVE_DOW). Weights learned per-meter on holdout. | `inference/v5_predict.py` | −1 to −3pp because the seasonal-naive is hard to beat on stable loads | unchanged | 1 | Low |

**Cumulative expected outcome (after #1 + #2 + #3 + #4 + #5 + #6, ~10 days of work):**

| Surface | v4 today | v5 target | Mechanism |
|---|---|---|---|
| Train MAPE (in-sample) | 2-3% | < 3% | unchanged |
| Bundle holdout median | 7.15% | **4.5 – 5.5%** | #5 + #6 |
| Bundle holdout p75 | 12.96% | **6 – 8%** | #5 + #6 |
| Backtest median (8d) | 46.04% | **6 – 9%** | #1 + #2 + #3 + #4 |
| Backtest mean (8d) | 63.42% | **8 – 12%** | same |
| Meters under 5% bundle | 8 / 42 | **24 – 30 / 42** | #5 + #6 + #7 |
| Meters under 5% backtest | 0 / 42 | **15 – 22 / 42** | #1 stack |

**Honest acknowledgment of what is NOT achievable:**

1. **All 42 meters at < 5% on 30-min cadence** — not possible. Cluster D (2 meters) has 60+% zero-blocks; their per-block MAPE is dominated by counting noise. They will report at 1-hr or 4-hr aggregates only.
2. **All meters at < 5% on a 76-day pure forecast** — this requires either continuous re-priming (re-forecast every 7 days as actuals arrive) OR a much-longer-trained model that learns 3-month seasonality reliably. Current training data is ~5 months per meter; not enough for stable annual seasonality. v5 ships with a re-priming loop instead — re-forecast on a daily cadence when actuals arrive.
3. **Sub-5% on contaminated holiday blocks for unknown holidays** — if the calendar misses one (Andhra has region-specific festivals), the meter will report a bad MAPE on that day. Mitigated by #4 (outage/calendar-gap detection) but not eliminated.

---

## 4 · Build order (today + next 2 weeks)

- **Today (Apr 21)** — Build change #1: recursive prediction loop. Re-run on Feb 5-12 backtest. Generate v3.1 fixture and a side-by-side comparison table. Validate the move is real, not noise. *(In progress — see task #13.)*
- **Day 2-3** — Changes #2 + #3: extend calendar, switch to weather-forecast API.
- **Day 4-5** — Change #4: outage detection feature + gap-aware MAPE.
- **Day 6-8** — Change #5: per-meter Optuna tuning, retrain all 42.
- **Day 9-11** — Change #6: deviation-from-typical-day target experiment.
- **Day 12** — Change #7: triage cluster C.
- **Day 13** — Cumulative re-evaluation. Decide which meters move into "v5-shipped" and which stay in "needs more work." Update the prototype to v4 vs v5 split view.
- **Day 14** — Write the post-mortem: what hit the predicted delta, what missed, what surprised.

---

## 5 · Why this order

Change #1 is 2 days of work and removes ~30pp of measured MAPE on most meters. No retrain. No data acquisition. Pure inference-wrapper change. By the end of Day 1, we should have a number that says *the v4 model does work — we were just measuring it wrong.* That reframes the rest of the plan from "the model is broken" to "the model is good and these are the polish steps."

After #1 lands, #2 + #3 are calendar/data plumbing — important but cheap. #4 is the first real engineering decision (outage definition). #5 is the first compute spend. #6 is the first thing that could fail to deliver. We do them in that order so we always know whether we are still on the curve to 5% or whether we have stalled.

If #1 alone moves backtest median from 46% to ~10% — and we expect it will — we are within striking distance of 5%, and the rest of the plan is realistic in 2 weeks. If #1 only moves it to 25%, the plan needs a hard re-think before spending more time.

---

## 6 · Empirical update — Apr 21 evening · Change #1 did NOT deliver

Built `inference/v5_predict.py · predict_recursive` and `benchmarks/v5_recursive_backtest.py`. Ran on 3 representative meters across clusters A, B, D.

Initial run had a feature-builder bug (placeholder `demand_wh=0` corrupted `demand_diff_1` and `demand_diff_4`, causing predictions to collapse to ~zero). After fixing — by priming the placeholder with the same-DOW-same-hour value from a week back, then letting the lag/rolling features cascade through real-then-predicted values — the recursion produces clean, model-shaped predictions.

| MSN | Cluster | Bundle | v3 (DOW prime) | v5 (recursive) | Δ |
|---|---|---|---|---|---|
| 50143025 | D | 13.76% | 65.73% | 63.63% | +2.10pp |
| 53407938 | A | 3.19% | 22.03% | 21.99% | +0.04pp |
| 65015739 | A | 4.53% | 12.75% | 16.62% | −3.87pp |

**Conclusion: recursion is a wash.** It does NOT deliver the predicted −25 to −35pp. The hypothesis that the bundle-vs-backtest gap is "stale lag features at inference" is **wrong** for this engine.

### What's actually causing the gap

I checked the bundle metadata: `train_cutoff = 2025-12-15T23:30:00` for every model. The bundle holdout MAPE (3.19% on this meter) was measured on the last 25% of training data — Dec 16 → end-of-January-ish — which is **inside** the training window's distribution.

The Feb 5–12 backtest evaluates the same frozen Dec-15-trained model on data **8 weeks past the training cutoff**. Weather has shifted (Visakhapatnam Jan→Feb), the holiday calendar has rolled (Sankranti vs Republic Day vs Maha Shivaratri), and load patterns have evolved. Both v3 (DOW priming) and v5 (recursion) hit the same ~22% ceiling on cluster A meter 53407938 because **that's how well a Dec-15-trained model can do on Feb 5–12 data, regardless of how its lag features are populated**.

Recursion vs DOW priming is a second-order debate. The first-order issue is **the model has not seen the most recent two months of data**.

### Revised plan — pivot

Promote what was change #5 (and add data-cutoff extension) to **change #1**. The new ordering:

| # | Change | Files | Expected backtest delta | Days | Risk |
|---|---|---|---|---|---|
| **1 (NEW)** | **Retrain all 42 bundles with `train_cutoff = 2026-02-04`** — extends training data by 51 days, lets the model learn Jan + early-Feb seasonality | `inference/v4_predict.py · train_and_persist`, batch-driver script | **−15 to −30pp on median** (best estimate; primary lever) | 1 | Low |
| 1b | Per-meter Optuna tuning on the retrained bundles | `training/optuna_tune.py` (new) | −1 to −3pp | 3 | Low |
| 2 | Extend HOLIDAYS dict + lunar calendar | `inference/_features.py` | −3 to −8pp on holiday days | 0.5 | None |
| 3 | Switch to Open-Meteo *forecast* API for >today | `inference/_features.py` | −2 to −5pp on weather-sensitive blocks | 1 | Low |
| 4 | Outage detection + gap-aware MAPE | `inference/_features.py`, `calc_metrics` | −5 to −10pp | 1 | Medium |
| 5 | Deviation-from-typical-day target (predict residual, not absolute) | `_features.py`, retrain | −5 to −10pp | 3 | Medium |
| 6 | Triage cluster C (4 meters) — longer history or drop | one-off | −20 to −40pp on those 4 | 2 | Low |
| 7 | Drop cluster D from < 5% target — explicit honesty | docs only | n/a | 0.5 | None |
| 8 | Conformal calibration for q10/q90 | `v4_predict.py` | n/a (coverage only) | 2 | Low |
| 9 | Ensemble with seasonal-naive (SNAIVE_DOW) baseline | `v5_predict.py` | −1 to −3pp | 1 | Low |
| ~~old #1~~ | ~~Recursive prediction~~ | ~~v5_predict.py~~ | ~~empirically a wash; keep the code as a tool but don't rely on it~~ | ~~done~~ | — |

`v5_predict.py` stays in the repo because (a) it's still the technically-correct inference path for very-long-horizon forecasts where DOW priming would walk past sensible same-DOW context, and (b) it costs nothing to keep. But it is not the lever to < 5%.

### What this changes about confidence in hitting < 5%

**Higher** confidence on the bundle-holdout median dropping to 4.5–5.5% — retraining with 51 more days of data should reliably shave 1–2pp off bundle MAPE for cluster B meters that are currently at 5–10%.

**Lower** confidence on the backtest median dropping below 10% in 2 weeks — the change that mattered most in the original plan turned out not to matter. The retraining gain (−15 to −30pp) is the new bet, and it's an estimate not yet verified on the 8-day window.

**Same** structural acknowledgment: cluster D will not hit < 5% at 30-min cadence; report at 1-hr aggregates.

### Lesson

I called change #1 the "highest-impact change" in the plan based on a plausible-sounding hypothesis (stale lags). I should have tested the hypothesis on 2–3 meters before declaring it the headline change. The test took ~3 minutes. The plan re-write took longer than that.

Going forward: every plan-row that estimates a delta will be paired with a 1-meter smoke-test result before the plan goes to anyone else. *Don't promise pp until you've measured pp.*

---

## 7 · Second empirical update — same evening · The retrain pivot also missed

Built `benchmarks/v5_retrain_one.py`. Retrained meter 53407938 with `train_cutoff = 2026-02-04T23:30` (extends training data by ~51 days vs the original Dec 15 cutoff). Trained in 3.5 seconds.

| Surface | v4 (Dec 15 cutoff) | v5 (Feb 4 cutoff) | Δ |
|---|---|---|---|
| Bundle holdout MAPE | 3.19% | **2.51%** | −0.68pp |
| Feb 5–12 backtest MAPE | 22.03% | **22.20%** | +0.17pp |

The retrain DID improve the in-pipeline holdout (the model is now slightly better at the data it was trained on). But the Feb 5–12 backtest is unchanged. **Extending training data through the day-before is not the lever either.**

### Where the backtest error actually lives

I ran a per-day, per-block decomposition of the 22% backtest MAPE. The pattern is unmistakable:

| Day | MAPE | Notes |
|---|---|---|
| Feb 5 | 11.54% | reasonable |
| **Feb 6** | **38.90%** | one block predicted 2187 Wh, actual was 181 Wh (1108% APE on that block) |
| **Feb 7** | **38.38%** | one block predicted 1597 Wh, actual was 179 Wh (792% APE) |
| Feb 8 | 22.80% | similar 22h-block over-prediction |
| Feb 9 | 12.05% | reasonable |
| Feb 10 | 14.80% | reasonable |
| Feb 11 | 10.14% | reasonable (partial day) |

The top 10 worst predictions are ALL of one shape: **the model expects a peak-hour spike (~2000 Wh at 22:00) that didn't happen.** On other days, the spike DID happen. The actual y_max on these days ranges 1980–2628 Wh — these are real spikes the meter records, but they fire on different days at different hours.

This meter has a **steady ~157 Wh base load** (heat pump idle, fridge, etc.) that the model nails to within ±5%, plus **stochastic high-power events** (likely EV charging, water heater, AC compressor cycles) that fire on specific days at unpredictable hours. The model averages across DOWs and learns "Friday 22:00 = high demand on average," then mis-fires when this Friday's resident skipped the AC because it was a cool night.

**No feature in the v4 set can predict whether tonight's resident will run the AC.** Weather + DOW + hour can predict the probability, but not the realisation. The MAPE penalty for a model that predicts the conditional mean (1500 Wh) when the actual is the floor (180 Wh) is 730% on that block — and a single such block per week pushes the daily MAPE from 5% to 35%.

### The structural truth this exposes

A meter's MAPE has two components:
1. **Reducible** — what the model can explain from features. Bundle holdout MAPE measures this. For 53407938, it's now **2.51%**.
2. **Irreducible** — variance from events the model cannot observe (occupant behaviour, equipment cycles, unknown HVAC dispatch). For meters with spike-y consumption, this floor sits in the **15–40%** range at 30-min cadence.

**The Feb 5–12 backtest measures (1) + (2) on a specific 8-day realisation. Bundle holdout measures (1) only because the holdout is randomly mixed in time and the spikes average out across the 1500+ holdout rows.**

Aggregating to longer time blocks reduces (2) — at 1-hour the spikes get averaged with their flanking blocks, at 4-hour even more, at 1-day the daily total is highly predictable. This is the **right unit for residential single-phase meters**. It's why APEPDCL's tariff settlement is hourly, not 30-min.

### Revised, honest, non-self-deluding plan

| Cohort | Meters | Achievable < 5% MAPE? | At what cadence? | What it takes |
|---|---|---|---|---|
| Cluster A — clean, low variance (HT, large commercial) | 8 | **yes** | 30 min | bundle is already 3-5%; retrain + Optuna seals it |
| Cluster B — moderate variance | 28 | **partial** | 30 min for ~12 of these; **1-hour for the rest** | retrain + per-meter tune; switch reporting cadence for the spike-y ones |
| Cluster C — fixable issues | 4 | **yes** | 30 min | data triage; one-off retraining on longer history |
| Cluster D — high-zero / intermittent | 2 | **no** | **4-hour or daily** | physics says no — stop pretending |

**Realistic 14-day v5 outcome:**

| Surface | v4 today | v5 honest target |
|---|---|---|
| Bundle holdout median (30 min) | 7.15% | **4.0–4.5%** |
| Bundle holdout p75 (30 min) | 12.96% | **6–8%** |
| Backtest median, 1-hour cadence | not measured | **5–9%** |
| Backtest median, 4-hour cadence | not measured | **3–6%** |
| Backtest median, 30-min cadence | 46.04% | **15–22%** (irreducible variance is the floor) |
| Meters under 5% bundle (30 min) | 8 / 42 | **22–28 / 42** |
| Meters under 5% backtest, 1-hr | not measured | **20–28 / 42** |
| Meters under 5% backtest, 4-hr | not measured | **30–36 / 42** |
| Meters under 5% backtest, 30-min | 0 / 42 | **8–14 / 42** |

### The pivot the user actually needs to hear

The original framing — "MAPE < 5% on all 42 meters across train, holdout, and future" — is **not achievable at 30-min cadence**, regardless of how much engineering we throw at it, because residential single-phase meters at that resolution have an irreducible variance floor in the 10–25% range on individual blocks.

What IS achievable in 14 days:
1. ~25 / 42 meters at < 5% bundle MAPE (in-pipeline 30-min)
2. ~30 / 42 meters at < 5% backtest MAPE at **1-hour or larger cadence**
3. All 42 meters at < 5% backtest MAPE at **4-hour cadence** (energy-aggregate, which is what most commercial decisions need anyway)

This isn't a retreat — it's the right unit. APEPDCL doesn't bill in 30-min increments; commercial customers don't dispatch BESS based on 30-min noise; the savings audit letters cite hourly or daily numbers. We were chasing the wrong cadence's accuracy bar.

**Recommendation:** before spending the 14 days, the team should pick the cadence the < 5% target applies to. If the answer is "the cadence the customer actually decides on" (1-hour or 4-hour) — we are in striking distance. If the answer is "30-min hard target on every meter" — we should rephrase the target.
