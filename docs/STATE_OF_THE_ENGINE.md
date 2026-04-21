# EdgeGrid Forecast Engine — State of the Engine

**Last updated:** 2026-04-21
**Owner:** Praveen Peddi (peddipra@gmail.com)
**Single source of truth.** Everything else cross-links into here.

---

## 1. The Outcome We Are Building

> **A production-grade demand forecasting engine that predicts demand accurately, per meter, for each of the 42 smart meters EdgeGrid operates — deployable as a service.**

The engine is the product. Everything downstream — BESS sizing, dispatch optimisation, savings audits, commercial economics — consumes this engine. It must stand on its own as production-grade.

Production-grade unpacks into three properties, all measured per meter:

1. **Accurate** — per-meter MAPE under the cohort-specific bar across the four cadences the engine serves (15-min DAM, 30-min native, hourly, ToD-4h).
2. **Stable** — that accuracy holds across many forecast origins, not just one cherry-picked cutoff. The engine is run daily in production; the daily-MAPE distribution must stay tight.
3. **Operable** — deterministic training, versioned bundles, clean predict API, documented feature inputs, reproducible by someone other than the person who trained it.

### Pass bar — the gate v4 must clear to ship

| Cohort | Members | MAPE bar |
|---|---|---|
| **A** | 8 meters | every meter, every origin, every horizon step ≤ 5% |
| **B** | 29 meters | fleet median ≤ 5%, p95 ≤ 10% (every origin) |
| **C** | 4 meters | fleet median ≤ 15%, cross-origin σ ≤ 5pp |
| **D** | 1 meter (50186364) | cross-origin σ ≤ 10pp; absolute MAPE not gated — bar is *predictably bad, not erratic* |

If the surface passes all four bars across all rolling origins, **v4 ships as production**. Where it fails, the failure surface names the retraining priorities for v5.

---

## 2. TL;DR — Where We Are Right Now (2026-04-21)

- **v4 is the production engine.** 42 LightGBM quantile bundles (mean + q10 + q90 per meter), two-pass training, trailing-21d MBE bias correction. Bundle manifest stamped 2026-04-17.
- **Validation status**: S1 (chronological cutoff) ✅, S2 (stratified temporal) ✅, S3 (rolling-origin) — design locked, dry-run on 1 origin complete, full 10-origin sweep queued.
- **Data gap (Feb 13 → Apr 20) filled** with 4 forecasting strategies. Strategy D (hybrid 7d-recursive + 60d-anchor) is the operational choice (+2.6% fleet vs seasonal anchor; v4 batch alone drifts +28.7%).
- **Forward forecast (Apr 21 → May 21) generated** across the same 4 strategies.
- **Block-accuracy framework** built (15-min DAM / 30-min native / hourly / ToD-4h cadences) and skeleton table for the gap pre-staged for instant join when AMI feed arrives.
- **Three live HTML viewers** shipped: block-accuracy dashboard, forward-forecast viewer, continuous-history viewer.

---

## 3. Engine Evolution — v1 → v4

| Version | Approach | Best Fleet MAPE | Status |
|---|---|---|---|
| **v1** | Persistence + DOW priming | ~22% | Retired |
| **v2** | Single global LightGBM | ~14% | Retired |
| **v3** | Per-meter LightGBM mean only | ~9% | Prototype, kept for reference |
| **v4** | Per-meter quantile bundles (mean + q10 + q90), two-pass training, trailing-21d MBE correction, recursive + batch inference paths | **~6.5%** (S1), **~5.8%** (S2) | **Production** |
| v5 (recursive) | Same model, autoregressive inference; production safeguards on | Used inside v4 stack | In production for 7d-window forecasts |

**Key v4 design decisions:**
- Two-pass training: quick-fit screen (top 55% by feature gain) → full-train (800 rounds). Cuts overfitting on noisy meters without losing signal on clean ones.
- Trailing-21d MBE correction: val-gated, capped at 30%, dampened 0.5×. Catches systematic drift without amplifying noise.
- Per-meter bundles, not one global model. Cohort-D meter (50186364) was destroying the global loss; isolating it freed the rest.
- Quantile triplet (mean + q10 + q90) provides calibrated uncertainty bands for downstream BESS dispatch.

---

## 4. Validation Strategies

Three parallel ways of validating the same engine. The engine doesn't ship until all three pass.

### S1 — Chronological Cutoff ✅

Train up to date X, test on date X onwards. Mimics one production deploy.

- **Doc**: `docs/STRATEGY_1_CHRONOLOGICAL_CUTOFF.md`
- **Dashboard**: `edgegrid-strategy1-v3-dashboard.html` (uploaded artifact)
- **Status**: Complete. v4 cleared the bar at the chosen cutoff.

### S2 — Stratified Temporal ✅

Every-4th-day holdout, seasonally balanced. Tests that the model isn't memorising specific weeks.

- **Doc**: `docs/STRATEGY_2_STRATIFIED_TEMPORAL.md`
- **Dashboard**: `edgegrid-strategy2-v4-dashboard.html` (uploaded artifact)
- **Status**: Complete. v4 cleared the bar with a slightly tighter MAPE than S1.

### S3 — Rolling-Origin Validation 🔄

Advance the cutoff across many origins; retrain at each; measure MAPE as a surface over (meter × origin × horizon). Tests stability under daily re-forecasting — the actual production cadence.

- **Doc**: `docs/S3_ROLLING_ORIGIN_PROTOCOL.md` (design + dry-run findings)
- **Schedule**: 10 weekly origins, Dec 4 2025 → Feb 5 2026; 7-day horizon at 30-min native.
- **Compute budget**: ~26 min/origin × 10 = ~3.7 hr serial (fits one overnight run).
- **Dry-run status (origin 2026-01-29)**: Complete. Wiring validated. Eyeball MAPE A 9.5% / B 15.8% / C 21.4% / D 41.4% at native-30min — pass bars look hard at this single origin; full sweep will tell us if Jan 29 is representative or an outlier.
- **Next**: full 10-origin sweep (queued as task #26).

---

## 5. Block-Accuracy Framework

Per-meter, per-block predicted/actual/APE/MBE metrics across four operational cadences.

| Cadence | Block size | Blocks/day | Use case |
|---|---|---|---|
| `dam_15min` | 15 min | 96 | DAM bidding, sub-hourly dispatch |
| `native_30min` | 30 min | 48 | Meter-native cadence, default for engine I/O |
| `hourly` | 60 min | 24 | Procurement and capacity planning |
| `tod_4h` | 4–6 h | 6 | APEPDCL ToD tariff blocks (tariff-aware optimisation) |

**Code**: `src/edgegrid_forecast/accuracy/block_accuracy.py`. Public API: `BlockSpec`, `BLOCK_SPECS`, `broadcast_to_dam_15min`, `assign_blocks`, `per_meter_block_mape`, `cohort_rollup`, `fleet_mape`, `assign_cohorts`. EPSILON = 0.0005 kWh (matches v4 calc_metrics mask).

**Backfill on Feb 5–12 actuals (52,088 rows)** computed and persisted; this was the validation that the framework wires correctly to actuals.

---

## 6. Gap + Forward Forecasting (4 Strategies)

The Feb 13 → Apr 20 data gap (67 days) and the forward window (Apr 21 → May 20, 30 days) are forecasted across four parallel strategies. **These are forecasting strategies, not training strategies — separate from S1/S2/S3.**

| Strategy | Mechanism | When it wins | Fleet vs A |
|---|---|---|---|
| **A — Seasonal Anchor + Weather** | Median of last 60d same-DOW-same-hour, scaled by Open-Meteo weather | Drift-free baseline; long horizons; insulated against bad lag inputs | — (baseline) |
| **B — v4 Batch (anchor-seeded lags)** | v4 LightGBM run in single batch pass; lags fabricated once from seasonal anchor | Diagnostic only — shows model bias when fed synthetic lags | **+28.7%** (drifts high) |
| **C — v5 Recursive** | v4 model in autoregressive mode (production inference path) | Short horizons (≤7d) with real recent lags | Sampled to 4 cohort-representative meters only (~128s/meter compute too high for full sweep) |
| **D — Hybrid 7d Recursive + 60d Anchor** | v5 recursive for first 7 days, seasonal anchor for the next 60 | **Operational choice.** Combines model sharpness early with drift-free baseline late | **+2.6%** |

**Why D is the operational pick**: model is most accurate when its lag inputs are real (first 7 days), then becomes structurally biased when lags go synthetic. Anchor handles the long tail without drift. Splicing B into the tail would inherit B's +28.7% systematic bloat — exactly what we built D to avoid.

---

## 7. Documentation Index

Every document, with path + one-line summary.

### Strategic / outcome docs
| Doc | Path | What it is |
|---|---|---|
| **State of the Engine** (this file) | `docs/STATE_OF_THE_ENGINE.md` | Single source of truth, master index |
| Foundation | `FOUNDATION.md` (uploaded) | Original engine principles + design constraints |
| Progress log | `PROGRESS.md` (uploaded) | Session-by-session journal v1 → v4 |
| Session 12 progress | `docs/SESSION_12_PROGRESS.md` | Dispatch console + Forecast Accuracy showcase build log |
| Truth board | `docs/TRUTH_BOARD_5PCT_REACHABILITY.md` | Per-meter × per-cadence reachability table for the <5% MAPE goal |

### Engine architecture
| Doc | Path | What it is |
|---|---|---|
| v4 architecture | `V4_ENGINE_ARCHITECTURE.md` (uploaded) | Module map, data flow, training loop, inference paths |
| v4 teardown | `v4_engine_teardown.md` (uploaded) | Honest assessment: what works, what breaks, what to fix in v5 |
| v5 plan | `docs/v5_plan_mape_below_5.md` | Engineering plan to push fleet MAPE below 5%; recursion table; retrain pivot |
| Engineering doc | `EdgeGrid_Engineering_Documentation.docx` (uploaded) | High-level engineering overview |
| v4 technical doc | `edgegrid-forecast-engine-v4-technical-documentation.docx` (uploaded) | Implementation specifics |
| Chronos ensemble eval | `CHRONOS_ENSEMBLE_EVALUATION.md` (uploaded) | Eval of Amazon Chronos as ensemble member; conclusion + tradeoff |

### Validation strategy docs
| Doc | Path | What it is |
|---|---|---|
| S1 chronological | `docs/STRATEGY_1_CHRONOLOGICAL_CUTOFF.md` | S1 protocol + v1 baseline + v4 results + per-cohort breakdown |
| S2 stratified temporal | `docs/STRATEGY_2_STRATIFIED_TEMPORAL.md` | S2 protocol + v4 results + per-cohort breakdown |
| S3 rolling-origin | `docs/S3_ROLLING_ORIGIN_PROTOCOL.md` | S3 protocol + dry-run findings (origin 2026-01-29) |

---

## 8. Artifact Index — Live Dashboards

Open these by clicking the links. Each is a self-contained HTML file.

| Artifact | Path | What it shows |
|---|---|---|
| **Continuous-history viewer** | `outputs/continuous_history_viewer.html` | Per-meter timeline: actuals → 67d gap forecast → 30d forward forecast, with strategy overlay (A/B/C/D), cohort filter, smoothing, time-range presets |
| **Block-accuracy dashboard** | `outputs/block_accuracy_dashboard.html` | Fleet MAPE tiles + per-meter table + cohort bars across the 4 cadences (Feb 5–12 backfill) |
| **Forward-forecast viewer** | `outputs/forward_forecast_viewer.html` | Strategy A vs B disagreement viewer for the 12 most-divergent meters |
| **S1 dashboard** | `edgegrid-strategy1-v3-dashboard.html` (uploaded) | v1→v3 walk under chronological-cutoff validation |
| **S2 dashboard** | `edgegrid-strategy2-v4-dashboard.html` (uploaded) | v4 results under stratified-temporal validation |

---

## 9. Artifact Index — Data (Parquet / CSV)

### Block-accuracy outputs
| File | Rows | What |
|---|---|---|
| `outputs/block_accuracy_backfill.parquet` | 52,088 | Per-meter per-block predicted/actual/APE/MBE for Feb 5–12, 4 cadences |
| `outputs/block_accuracy_summary.csv` | 168 | Per-meter per-cadence summary (42 × 4) |
| `outputs/block_accuracy_cohort.csv` | 16 | Cohort × cadence rollup |
| `outputs/block_accuracy_fleet.csv` | 4 | Fleet × cadence rollup |
| `outputs/block_accuracy_gap_skeleton.parquet` | 1,515,540 | Pre-staged accuracy table for the Feb 13 → Apr 20 gap, NaN actuals (instant-join when AMI arrives) |

### Forward-forecast outputs (Apr 21 → May 21)
| File | Rows | What |
|---|---|---|
| `outputs/forward_forecast_30d.parquet` | 60,480 | Strategy A baseline at 30-min native |
| `outputs/forward_forecast_dam_15min.parquet` | 120,960 | Strategy A broadcast to 15-min DAM |
| `outputs/forward_forecast_strategy_A_seasonal.parquet` | 60,480 | A only |
| `outputs/forward_forecast_strategy_B_v4batch.parquet` | 60,480 | B only |
| `outputs/forward_forecast_all_strategies.parquet` | 120,960 | A + B + D combined long-format |

### Gap-forecast outputs (Feb 13 → Apr 20)
| File | Rows | What |
|---|---|---|
| `outputs/gap_strategy_A_seasonal.parquet` | 135,072 | A — seasonal anchor + weather |
| `outputs/gap_strategy_B_v4batch.parquet` | 135,072 | B — v4 batch with anchor-seeded lags |
| `outputs/gap_strategy_C_v5recursive.parquet` | 12,864 | C — v5 recursive (4 cohort-rep meters × 7 days) |
| `outputs/gap_strategy_D_hybrid.parquet` | 135,072 | D — hybrid 7d recursive + 60d anchor (operational pick) |
| `outputs/gap_all_strategies.parquet` | ~417,000 | Long-format union of A/B/C/D |
| `outputs/full_forecast_feb13_may21.parquet` | 605,280 | Stitched 98-day forecast across all 4 strategies |

### S3 dry-run outputs
| File | What |
|---|---|
| `outputs/s3/2026-01-29T0000/forecast.parquet` | Per-meter 7-day forecast (14,112 rows) |
| `outputs/s3/2026-01-29T0000/metrics_long.parquet` | Per-meter per-step MAPE/MBE/coverage |
| `outputs/s3/2026-01-29T0000/cohort_summary.csv` | Cohort × cadence rollup at this origin |
| `outputs/s3/2026-01-29T0000/per_meter_summary.csv` | One row per meter |
| `outputs/s3/2026-01-29T0000/timing.json` | Wall-clock per stage |
| `models/v4_s3/2026-01-29T0000/<msn>.joblib` | Refitted bundles for this origin (production `models/v4/` untouched) |

### Raw data + reference
| File | What |
|---|---|
| `data/raw/sp_data.parquet` | Single-phase actuals (5 meters, ts → 2026-02-12) |
| `data/raw/tp_data.parquet` | Three-phase actuals (45 meters, ts → 2026-02-12) |
| `data/raw/meter_profile.parquet` | Meter metadata (location, phase, MSN) |
| `data/external/weather/*.parquet` | Open-Meteo cache, 2024-10-01 → 2026-04-21, three locations |
| `data/external/nasa_power/*.parquet` | NASA POWER 3-yr archive |
| `data/external/air_quality/*.parquet` | Air-quality cache (FY24-25) |
| `prototypes/forecast_engine_v3/oracle_floor.csv` | 42-meter oracle floor (bundle_mape used for cohort assignment) |

---

## 10. Code Index — Key Modules

### Engine core
| Path | Role |
|---|---|
| `src/edgegrid_forecast/inference/v4_predict.py` | Batch prediction (one shot, lags seeded once) |
| `src/edgegrid_forecast/inference/v5_predict.py` | Recursive prediction (autoregressive, production path) |
| `src/edgegrid_forecast/inference/_features.py` | Feature builder (lags, rolls, calendar, weather joins) |
| `src/edgegrid_forecast/training/v5_retrain_one.py` | Per-meter retrain (Pass 1 quick-fit + Pass 2 full-train + quantile fit) |
| `src/edgegrid_forecast/data/loaders.py` | Meter list, actuals loader, train/holdout splitter |
| `src/edgegrid_forecast/data/collectors/open_meteo.py` | Historical + forecast weather collector |
| `src/edgegrid_forecast/accuracy/__init__.py` | Public API for block accuracy |
| `src/edgegrid_forecast/accuracy/block_accuracy.py` | Per-meter per-block MAPE/MBE/coverage at 4 cadences |

### Benchmarks (executable runs)
| Path | What it produces |
|---|---|
| `benchmarks/block_accuracy_backfill.py` | Feb 5–12 block accuracy table |
| `benchmarks/forward_forecast_apr_may.py` | Apr 21 → May 21 strategy A baseline |
| `benchmarks/forward_forecast_multi_strategy.py` | Apr 21 → May 21 strategies A/B (C deferred for compute) |
| `benchmarks/gap_forecast_feb_apr.py` | Feb 13 → Apr 20 gap fill, 4 strategies |
| `benchmarks/s3_rolling_origin.py` | S3 single-origin runner (`--origin YYYY-MM-DD --mode {refit,full_retrain}`) |

### Dashboard / viewer builders
| Path | Output |
|---|---|
| `scripts/build_block_accuracy_dashboard.py` | `outputs/block_accuracy_dashboard.html` |
| `scripts/build_forward_forecast_viewer.py` | `outputs/forward_forecast_viewer.html` |
| `scripts/build_continuous_history_viewer.py` | `outputs/continuous_history_viewer.html` |

---

## 11. Decision Log — Key Calls Made

1. **Per-meter bundles, not global model.** Cohort-D meter was destroying global loss. Isolating freed the rest.
2. **Two-pass training.** Quick-fit screen at 55% feature retention catches overfit on noisy meters without losing signal on clean ones.
3. **Trailing-21d MBE bias correction**, val-gated, 30% cap, 0.5× dampen. Without it, forecasts drift after 30+ days.
4. **Quantile triplet (mean + q10 + q90)** instead of point forecast. Downstream BESS dispatch needs uncertainty bands, not point estimates.
5. **4 cadence framework** (15-min DAM / 30-min native / hourly / ToD-4h). Anything narrower than DAM is below meter resolution; anything wider than ToD loses tariff signal.
6. **Cohort assignment via `bundle_mape` thresholds** (A<5, 5≤B<15, 15≤C<25, D≥25). Sets honest expectations per meter group.
7. **Forecasting strategy D = 7d recursive + 60d anchor.** Anchor is drift-free; recursive is sharpest in week 1; D combines both. Splicing B (batch) into the tail would inherit +28.7% bias.
8. **S3 = 10 weekly origins, 7-day horizon, full retrain (refit unavailable).** Matches production cadence; fits one overnight run.
9. **Continuous-history viewer aggregates to hourly** (not 30-min native) to keep file size under 4 MB while preserving daily-pattern fidelity.
10. **Production bundles at `models/v4/` are sacred.** S3 retrains write to `models/v4_s3/<origin>/`. Never overwrite.

---

## 12. Open Work

| Task | Status | What |
|---|---|---|
| #25 | ✅ done | Design S3 protocol — locked in `docs/S3_ROLLING_ORIGIN_PROTOCOL.md` |
| #26 | 🔄 in progress | Execute full 10-origin S3 sweep + build dashboard + write `docs/S3_RESULTS.md` verdict |

Backlog (not yet on the task list, surface when needed):
- v4.1: port trailing-21d MBE correction into batch inference (would likely cut Strategy B's +28.7% drift; would make a `Strategy E` viable as a hybrid tail).
- v5 retraining priorities — populated from S3 results once we know which (meter × origin) triples fail.
- Continuous-AMI ingestion: when meter feed resumes, join into `outputs/block_accuracy_gap_skeleton.parquet` and grade each strategy's gap forecast against actuals.
- Dispatch console linkage: pipe Strategy D forward forecast into BESS Explorer for live dispatch simulation.

---

## 13. Risks & Known Issues

1. **No refit path** — every S3 origin requires full retrain. Acceptable at 5.2 s/meter but worth a v4.1 ticket to add a true refit (faster S3 iteration, faster v5 development loop).
2. **v5 recursive compute** — ~128 s/meter for 30-day horizons. Not feasible for full-fleet long-window operations. Hybrid (D) hides this; if we ever need pure recursive at full fleet, parallelisation work is required.
3. **Per-step MAPE inflates near zero** — single sub-EPS slot can balloon the per-meter average. Block-aggregated cohort numbers are more honest. Reading rule baked into the S3 protocol.
4. **Cohort D = n=1.** Statistical claims about D are weak by definition. Treat 50186364 as an individual case, not a cohort.
5. **Bundle manifest stamped 2026-04-17** — implies last full retrain was Apr 17, after which no new actuals arrived (data ends Feb 12). When the AMI feed resumes, a full retrain on the new actuals is the first thing to do, ahead of the S3 sweep.

---

## 14. Glossary

- **MAPE** — Mean Absolute Percentage Error. Per-block, then aggregated.
- **MBE** — Mean Bias Error in kWh. Tells you direction (over- vs under-forecast) where MAPE only tells magnitude.
- **Cohort** — meter grouping by bundle_mape: A <5%, B 5–15%, C 15–25%, D ≥25%.
- **Cadence** — granularity of evaluation: 15-min DAM, 30-min native, hourly, ToD-4h.
- **Origin** (S3) — the timestamp at which the model is "frozen" and asked to forecast forward; production analogue is "today at 00:00".
- **Horizon step** — distance from origin in 30-min slots; h=1 is "30 min ahead", h=336 is "7 days ahead".
- **Bundle** — one meter's complete LightGBM artifact: mean model + q10 model + q90 model + bias-correction state.
- **Anchor** — seasonal median of last 60 days same-DOW-same-hour, scaled by weather.
- **Occupant noise** — irreducible variance in a meter's behaviour caused by human/process activity that no signal in our feature set can explain.

---

*This document is the canonical source for "what is the state of the engine right now." When in doubt, update this file first, then the satellite docs.*
