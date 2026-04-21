# EdgeGrid Demand Forecast Engine · v4.s1.0 · Teardown

**Scope.** Every input, every transformation, every modeling choice, every output that together make up the v4 forecasting engine — written so a new engineer can rebuild it from scratch. Status of each choice is tagged:

- **FACT** — the code does this, period. No judgment call.
- **HEURISTIC** — we chose this because it worked well enough; a different defensible choice exists.
- **OPEN** — unresolved. Either wrong, unverified, or deferred to v5.

Model version: `v4.s1.0`. Frozen 2026-04-17. 42 meters trained, `runtime_seconds` 148.1 total.

File map referenced throughout:

```
src/edgegrid_forecast/
├── inference/
│   ├── v4_predict.py      466 lines · predict() + train_and_persist() + bias gate
│   ├── _features.py       457 lines · every feature built here + TIME_BLOCKS
│   └── __init__.py        exports MODEL_VERSION, predict, predict_with_context
├── training/              older baselines (train_demand.py, train_real_demand.py)
└── models/                dataclass wrappers (demand.py, foundation.py, price.py)

models/v4/
├── {msn}.joblib           42 bundles, one per meter
└── _manifest.json         registry: msn → tier, metrics, bias_gate, block MAPEs

data/raw/
├── sp_data.parquet        single-phase 30-min consumption (3,115 rows sampled)
├── tp_data.parquet        three-phase 30-min consumption (81,931 rows sampled)
├── meter_profile.parquet  99 meters · tier · coverage · eligibility
└── meter_profile.csv

data/external/
├── weather/               Open-Meteo archive, Visakhapatnam, 30-min resample
├── air_quality/           (present, not wired into v4 features)
└── nasa_power/            (present, not wired into v4 features)

tests/
├── test_v4_predict.py     MAPE parity tests, DataFrame contract
└── test_features.py
```

---

## 1. Business scope (what is v4 actually for?)

v4 is a **per-meter, 24-hour-ahead demand forecasting engine** trained on MDMS half-hourly consumption, enriched with weather, calendar, and cross-fleet signals. It produces, for any meter we have a model for:

- A central forecast (`forecast_wh`) at 30-min resolution.
- A quantile confidence band (`confidence_low` = q10, `confidence_high` = q90).
- A block label (`night` / `morning` / `solar` / `peak`) per timestamp.
- The model's historical per-block MAPE, so a reader can calibrate how much to trust each window.

It is **not** an endpoint for 5-minute real-time dispatch (see §5 on granularity) and it is **not** currently conditioned on live IEX prices or realized weather (see §3 on inputs and a11 in the ledger).

---

## 2. Data inputs

### 2.1 Raw consumption (MDMS)

- `data/raw/sp_data.parquet` — single-phase (1PH) meters.
- `data/raw/tp_data.parquet` — three-phase (3PH) meters.
- Unified schema: `ts (UTC, 30-min), demand_wh (float), voltage (float), msn (str), phase (1PH|3PH)`.
- Coverage: October 2024 through February 2026 depending on meter; 4–18 months per meter.
- Unit: energy per half-hour bucket, in **watt-hours**. Throughout inference the native unit is Wh and the prototype converts to kWh for display.
- **FACT.** Half-hour is the MDMS-native cadence from APEPDCL. Everything upstream is integer multiples of that.

### 2.2 Meter metadata

- `data/raw/meter_profile.parquet` — one row per meter (99 rows).
- Columns: `msn, tier, source, n_rows, n_days, ts_min, ts_max, mean_demand_wh, zero_pct, eligible`.
- Tier taxonomy (computed from `mean_demand_wh`):
  - **Small** — mean < 500 Wh/half-hour.
  - **Medium** — 500–1,500 Wh/half-hour.
  - **HT** — > 5,000 Wh/half-hour (the high-tension customers that matter commercially).
- **HEURISTIC.** The tier boundaries are chosen to give each tier enough meters for tier-adaptive hyperparameters (see §4.3) to have a statistical effect without collapsing into per-meter tuning. We will revisit when we cross 300 meters.

### 2.3 Weather

- Source: Open-Meteo archive API, site = Visakhapatnam (lat/lon hard-coded in `_features.py`).
- Cached local parquet: `data/external/weather/visakhapatnam_expanded_2024-10-01_2026-03-01.parquet`.
- Variables consumed: `temperature, humidity, dewpoint, pressure, cloud_cover, precipitation, wind_speed, ghi, dhi, dni, direct_rad`.
- Derived inside `_features.py`: `pressure_delta_3h, temp_delta_3h, ghi_rmean_6h, cloud_delta_3h, is_raining, diffuse_fraction, heat_index, temp_rmean_6h`.
- Resample to 30-min bins; forward-fill then back-fill to align with consumption timestamps.
- **HEURISTIC.** Single-location weather for a whole fleet. Acceptable while every meter is inside Visakhapatnam; will need per-meter or per-substation lat/lon before we expand into the rest of APEPDCL's 5 circles (see a05 OPEN).

### 2.4 What is NOT in the v4 inputs

- **No IEX DAM / RTM price signal.** Present in older scripts (`data/external/iex/*`) but not wired into `_features.py`. The engine predicts *demand*, not cost — cost math is applied downstream by the BESS Explorer and the Courier Pack. **OPEN (a11)**.
- **No festival / holiday outage calendar beyond the static `is_holiday` / `near_holiday` flag.** If a meter lost power for a block, that day shows up as a high-MAPE day and the model has no way to know why. **OPEN (a18)**.
- **No SCS customer metadata at inference time.** Customer mapping (SCNO, UKSCNO, phase, billing tier) is attached downstream in the fixture builder, not used as a model feature. **HEURISTIC**.
- **No coincidence factor or line-loss adjustment** in the fleet aggregate — we sum per-meter predictions straight across. Fine for "what flow is this fleet drawing"; wrong for grid-loss billing. **OPEN (a17)**.

---

## 3. Preprocessing

Implemented inside `_features.py` and called by `v4_predict.train_and_persist()`. All transformations are meter-scoped; nothing pools across meters at this stage.

1. **Chronological 75 / 25 split.** First 75% of observations are training, last 25% holdout. No shuffling, no random split. **FACT.** This is the single most important design choice in the engine — it is why the benchmark MAPE is out-of-sample.
2. **Warmup drop.** The first 336 rows (7 days) are discarded after lag computation, since the deepest lag is 336 steps and the early rows carry NaN lags. **FACT.**
3. **NaN handling.** `np.nan_to_num(..., nan=0.0)` applied right before LightGBM sees the matrix. Weather is forward-filled then back-filled to cover any 30-min gaps. **HEURISTIC** — a LightGBM model can take NaN natively, so the explicit zeroing is a belt-and-braces choice that costs us nothing but removes one class of train/infer skew.
4. **Sparse-demand log1p.** If more than 30% of test-set demand is below 0.5 Wh, the target is log1p-transformed before training and expm1 is applied at inference. The flag is persisted in the bundle (`use_log1p: bool`) so train and inference agree. **HEURISTIC.**
5. **Non-negativity clamp.** `np.maximum(pred, 0)` after every predict call, on all three heads. **FACT.** Demand is physically non-negative.
6. **Monotonicity clamp on quantile bands.** At inference, `q10 = min(q10, pred)`, `q90 = max(q90, pred)` — prevents pathological band crossings without requiring a joint training objective. **HEURISTIC (a12)**.

---

## 4. Feature engineering

All 70–90 candidate features built by `_features.py`. Pass 1 of training (see §5.1) screens down to a median of ~39 per meter.

### 4.1 Temporal (10)

`hour, dow, month, is_weekend` plus sine / cosine encodings of the three cyclic variables: `hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos`. **FACT** — standard practice. Sin/cos is preferred over raw integer so the tree doesn't have to waste depth on the "midnight-is-close-to-23:00" boundary.

### 4.2 Lags (9)

`lag_{1, 2, 4, 6, 12, 24, 48, 96, 336}` — in 30-min steps, so this is 30 min → 7 days. **FACT.** The 336-step lag is the single most important feature by importance gain in most HT meters — weekly seasonality dominates.

### 4.3 Rolling statistics (8)

- Rolling means and stds over windows `[6, 12, 48, 336]` half-hours: `rmean_*, rstd_*`.
- Short-window momentum: `demand_diff_1, demand_diff_4`.
- Long-window: `rmean_14d, rmean_30d, trend_ratio` (= `rmean_14d / rmean_30d`).

### 4.4 Weather-derived (20)

See §2.3. Direct weather variables plus derived deltas, diffuse-fraction, and heat-index. The `is_raining` binary flag fires when precipitation > 0.1 mm in the hour.

### 4.5 S1 weather anomalies (4)

`temp_anom_24h, hum_anom_24h, ghi_anom_24h, cloud_anom_24h` — each is `current_value - value_24h_ago`. Captures "today is unusually hot for this hour." **HEURISTIC.**

### 4.6 Calendar and tariff (4)

- `is_holiday`, `near_holiday` — Visakhapatnam Indian calendar, 30+ dates 2024–2026 hard-coded in `_features.py`. **OPEN (a18)** because the list needs to be regenerated each year.
- `tod_multiplier` — APEPDCL ToU rate multiplier: 0.9 for night, 1.0 for day, 1.2 for evening peak. This is a **feature**, not a cost calculation; the multiplier is a proxy for load-shifting behavior.
- `is_peak` — 18:00–22:00 (matches APEPDCL definition, and matches TIME_BLOCKS).

### 4.7 Voltage (3)

`voltage_lag1, voltage_rstd_6, voltage_rmean_48`. Acts as a brown-out / outage proxy — when voltage dips and then demand drops, the model learns to not hallucinate demand. **HEURISTIC.**

### 4.8 Deviation from hourly baseline (1)

`deviation_from_hourly` — current demand minus the expanding-window mean for that `hour` value. Gives the tree a stationary signal even when the level shifts year over year.

### 4.9 Fleet cross-meter (3)

- `fleet_mean, fleet_std` — computed across all meters available at that timestamp, minimum 5 meters.
- `vs_fleet_ratio` — this meter's demand divided by the fleet mean.
- `fleet_mean_lag1, fleet_std_lag1` — one-step lags so the feature is usable at forecast time.
- **HEURISTIC (a07).** The fleet signal is strong because APEPDCL-wide events (holidays, outages, grid festivals) move many meters together. Falls back to ffill/0 if fleet_df is not supplied at inference — documented but fragile; production loads should always pre-compute fleet_df.

### 4.10 Similar-day lookback (1)

`similar_day_k5` — the rolling mean of demand at the same `hour × dow` combination over the last `k=5` occurrences, with a 2-period minimum. Behaves like a soft nearest-neighbor prior. **HEURISTIC.**

### 4.11 Feature screening (Pass 1)

Before the final fit, a fast LightGBM pass is trained for 150 rounds and feature importance is computed. The top 25–55% of features by gain are retained. Median holdout meter ends up with 39 features. **FACT** that the gate is there; **HEURISTIC** on the 25–55% window — we landed on it empirically.

---

## 5. Model architecture

Three independent LightGBM models per meter. One per target: mean, q10, q90.

### 5.1 Training objectives

| Head | Objective | Metric | alpha |
|------|-----------|--------|-------|
| `model_mean` | `regression` | `mae` | — |
| `model_q10` | `quantile` | `quantile` | 0.10 |
| `model_q90` | `quantile` | `quantile` | 0.90 |

**FACT.** MAE (not MSE) is the optimization target on the mean head because the downstream use is BESS sizing, where kWh-level absolute error converts directly into rupees at the landed tariff; squared-error overweights rare outliers relative to what operations cares about.

### 5.2 Base hyperparameters

From `get_params()` in `v4_predict.py`:

```python
p = {
    "objective": "regression", "metric": "mae", "verbose": -1,
    "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 50,
    "feature_fraction": 0.6, "bagging_fraction": 0.7, "bagging_freq": 5,
    "lambda_l1": 0.1, "lambda_l2": 1.0, "max_depth": 8,
}
```

- Two-pass training: Pass 1 (screening) 150 rounds; Pass 2 (final) 800 rounds with early stopping (patience 40) against the last 15% of the training data.
- Quantile models are refit at the `best_iter` found by the mean model, using the same selected features.

### 5.3 Tier-adaptive overrides

Hyperparameters change per tier before Pass 2:

- **HT (>5kWh)** — more signal, more capacity: `num_leaves: 47, min_child_samples: 30, feature_fraction: 0.7, learning_rate: 0.04`.
- **Small (<500)** — more noise, more regularization: `num_leaves: 15, min_child_samples: 80, feature_fraction: 0.5, max_depth: 6, lambda_l1: 0.5, lambda_l2: 5.0`.
- **n_samples < 3000** — capped regardless of tier: `num_leaves ≤ 15, min_child_samples ≥ 100` so under-sampled meters can't overfit.
- **HEURISTIC.** The specific numeric overrides came from a grid search on a held-out set of 10 meters. Not guaranteed to transfer to a future fleet drawn from different substations.

### 5.4 Phase-B bias gate

The single cleverest thing in v4. Lives in `train_and_persist()`, lines ~200–245 of `v4_predict.py`.

After Pass 2, on the trailing 21 days of training data, compute `trail_mbe = mean(pred_trail - y_trail)`. This is the model's mean signed error on recent history. Dampen by 0.5× and cap at ±30% of the meter's mean demand:

```python
trail_mbe_candidate = clip(trail_mbe * 0.5, -0.3 * mean_demand, +0.3 * mean_demand)
```

The correction is only applied if **all three** are true:

1. The meter has at least 48 validation samples (1 day).
2. `|trail_mbe_candidate| > 0.5` Wh (noise floor).
3. On validation, applying the correction improves MAPE by more than 0.1 absolute percentage points vs. the raw prediction.

Otherwise the correction is zeroed out and the model records a reason: `below_threshold`, `val_too_small`, or `gated_out`. When applied, the correction is subtracted from all three heads (`pred = max(pred_raw - trail_mbe_capped, 0)`) and the model's `bias_gate` field is set to `"applied"`.

**FACT** that the gate exists and is implemented exactly this way. **HEURISTIC (a14)** on the constants 21 days, 0.5× dampen, 0.3 cap, 0.5 Wh floor, 0.1 pp improvement threshold. These came from tuning on the initial HT cohort.

Empirically across the 42-meter fleet: mixed applied / gated_out / below_threshold / val_too_small states; no single reason dominates. The gate is doing real work.

---

## 6. Time blocks (BESS dispatch windows)

Defined in `_features.py` line 64:

```python
TIME_BLOCKS = [
    ("night",   lambda h: (h >= 22) | (h < 6)),
    ("morning", lambda h: (h >= 6) & (h < 10)),
    ("solar",   lambda h: (h >= 10) & (h < 18)),
    ("peak",    lambda h: (h >= 18) & (h < 22)),
]
```

- night: 22:00–06:00
- morning: 06:00–10:00
- solar: 10:00–18:00
- peak: 18:00–22:00

**FACT** in the engine. **OPEN (a19)** — the prototype README describes night as 23:00–06:00 and solar as 10:00–16:00, which does not match the engine. Either the README is wrong (most likely, the README was written for a different earlier taxonomy) or the engine blocks need to change to match IEC peak definitions. This needs to be reconciled before any commercial one-pager reuses block MAPEs, because the block boundary is what determines which block a given anomaly falls into.

---

## 7. Inference

### 7.1 High-level `predict(msn, ...)`

```python
predict(msn, as_of_datetime=None, horizon=48, ...) -> DataFrame
```

- Loads raw meter history.
- Builds the 70–90 feature matrix with a 336-step warmup.
- Delegates to `predict_with_context()` for the last `horizon` 30-min steps.
- Returns a 48-row DataFrame (default horizon = 1 day), ts-indexed, columns: `forecast_wh, confidence_low, confidence_high, block_label, historical_block_mape`.

### 7.2 Lower-level `predict_with_context(msn, context_df, weather_df=None, fleet_df=None, ...)`

- Caller supplies meter history plus optional weather and fleet frames.
- Features materialized via `_features.py::build_features()`.
- `trail_mbe_capped` is subtracted from mean, q10, q90.
- Non-negativity + monotonicity clamps applied.
- Historical block MAPEs from the bundle are attached to each row for UI use.

### 7.3 Bundle schema

Each `models/v4/{msn}.joblib` is a single dict (joblib-compressed, level 3). Canonical keys (from `v4_predict.py` docstring lines ~70–90):

```
version                str    "v4.s1.0"
msn                    str
tier                   str    Small | Medium | HT
phase                  str    1PH | 3PH
trained_at             iso8601
model_mean             lgb.Booster
model_q10              lgb.Booster
model_q90              lgb.Booster
selected_features      list[str]  ~39 features
use_log1p              bool
trail_mbe_raw          float  uncorrected trailing MBE
trail_mbe_capped       float  what is actually subtracted at inference
bias_gate              str    applied | gated_out | below_threshold | val_too_small
holdout_metrics        dict   mape, mae, mbe, rmse, r2, within5, within10, coverage_80
historical_block_mape  dict   night, morning, solar, peak → MAPE %
```

### 7.4 Graceful degradation

- **No weather.** `predict()` fetches weather on demand via `fetch_weather_expanded()`. If Open-Meteo is unreachable AND the local cache is missing, it raises `RuntimeError`. **OPEN (a15)** — production runbook should either pre-warm the cache daily or accept a learned-typical-conditions fallback.
- **No fleet_df.** Fleet features are ffilled and zeroed; prediction still runs but loses the cross-meter signal. **HEURISTIC.** A production inference run should always construct fleet_df from the concurrent forecast batch.

---

## 8. Evaluation

### 8.1 `calc_metrics()`

Runs on the 25% holdout. Produces:

- **MAPE** = `mean(|y_true - y_pred| / y_true * 100)` with a zero-denominator guard: rows where `y_true < 0.5` Wh are excluded. **FACT.**
- **MAE** = mean absolute error, in Wh.
- **MBE** = mean bias error (signed).
- **RMSE** = root mean square error.
- **R²** = coefficient of determination. Returns `null` if actual variance is zero (a flat outage day) rather than crashing.
- **within5%, within10%** = fraction of rows inside those error bands.
- **coverage_80%** = fraction of rows where actual lands inside the q10–q90 band. Well-calibrated ≈ 80%.

### 8.2 Block-level

`calc_block_metrics()` loops TIME_BLOCKS and returns `{block}_mape, mae, mbe, n, mean_demand`. Persisted in `historical_block_mape`. The prototype surfaces both block MAPE and block Δ-vs-training so a reviewer can see when a block is misbehaving today relative to its own historical baseline.

### 8.3 Granularity-scoped metrics (prototype-level)

The engine itself reports native 30-min metrics. The prototype v2 re-computes metrics at the selected granularity (5/15/20/30/45 min, 1/2/4/7 hr). Aggregated granularities (1, 2, 4, 7 hr) sum the 30-min blocks and re-score — this is mathematically honest for energy. Derived sub-30-min granularities (5, 15, 20, 45 min) redistribute the 30-min block energy proportionally; **they are not sub-30-min predictions** and should never be cited as such (a21).

### 8.4 Test harness

`tests/test_v4_predict.py`:

- **MAPE parity** — bundle holdout metrics must match `benchmarks/results/benchmark_strategy1_v4.csv` to within ±0.01%. Catches any drift between the training recipe and the persisted bundle.
- **DataFrame contract** — shape (48, 5), ts-indexed, all columns present, all values non-negative, `confidence_low ≤ forecast ≤ confidence_high`.
- **Bias gate state** — tracked per meter; gate transitions are test-flagged so a regression can't silently turn the gate off.

---

## 9. Model artifacts (as of 2026-04-17)

From `models/v4/_manifest.json`:

- 42 trained meters, `version: v4.s1.0`, built 2026-04-17T03:41:22Z, 148.1 s total runtime.
- Fleet holdout MAPE: **mean 11.46%, median 8.25%, p75 12.96%**; 18 green (<8%), 13 amber (8–12%), 11 red (>12%).
- Best meter: `53407938` at 2.78% holdout MAPE. Worst: `50186364` at 77.86% (outage-driven, a18).
- 8 meters from the 50-meter MDMS upload do **not** yet have v4 models (insufficient history or data-quality fail) — see `meter_profile.csv` `eligible=False` rows. **OPEN (a04).**

---

## 10. The assumption ledger

Categories: **Data, Model, Validation, Product, Commercial.** Every item carries its status.

### Data

- **a01 · FACT.** MDMS is the source of truth for consumption at 30-min resolution. Upstream of MDMS we do nothing.
- **a02 · FACT.** Consumption unit at rest is Wh per half-hour; kWh conversion happens at the UI layer.
- **a03 · HEURISTIC.** Tier boundaries (Small <500, Medium 0.5–1.5k, HT >5k Wh/half-hour) are chosen to give each tier statistical power. Re-examine past 300 meters.
- **a04 · OPEN.** 8 of 50 uploaded meters do not have trained v4 models yet. Either data history is insufficient or the data-quality filter rejected them. Ship a retrain job before we promise SLAs on those.
- **a05 · OPEN.** Weather is single-location (Visakhapatnam). Fine inside one circle; wrong when we expand to all APEPDCL circles. Needs per-meter lat/lon lookup.
- **a06 · HEURISTIC.** Holiday calendar is hard-coded 2024–2026 in `_features.py`. Must be regenerated annually; no dynamic source.

### Model

- **a07 · HEURISTIC.** Fleet cross-meter features (fleet_mean, fleet_std, vs_fleet_ratio) with `min_meters = 5`. The number 5 is a guess; higher is safer but drops more rows.
- **a08 · FACT.** One LightGBM per meter, three heads (mean, q10, q90). No shared-backbone multi-meter model.
- **a09 · FACT.** MAE objective on the mean head (not MSE), quantile objective on the bands.
- **a10 · FACT.** Base hyperparameters: `learning_rate 0.03, num_leaves 31, feature_fraction 0.6, bagging_fraction 0.7, lambda_l1 0.1, lambda_l2 1.0, max_depth 8`.
- **a11 · OPEN.** No live IEX or realized weather signal at inference. The model runs on forecast weather only. Wiring live signals is v5.
- **a12 · HEURISTIC.** Monotonicity clamp (`q10 ≤ pred ≤ q90`) applied post-hoc. A joint quantile-aware training objective would be cleaner.
- **a13 · HEURISTIC.** Tier-adaptive hyperparameter overrides (HT, Small, n<3000 caps) came from a grid search on 10 HT meters. Not re-verified after each new cohort.
- **a14 · HEURISTIC.** Phase-B bias gate constants: 21-day trailing window, 0.5× dampening, ±30% cap, 0.5 Wh noise floor, 0.1 pp improvement threshold, 48-sample validation floor. All five tunable.
- **a15 · OPEN.** If Open-Meteo is unreachable AND the local weather cache is missing, inference raises `RuntimeError`. Production needs either cache pre-warming or a learned-typical-conditions fallback.
- **a16 · HEURISTIC.** Sparse-demand log1p trigger at 30% of test-set demand below 0.5 Wh. The 30% threshold is empirical.

### Validation

- **a17 · OPEN.** Fleet aggregate is simple sum — no coincidence factor, no line-loss adjustment. Acceptable for "what flow is this fleet drawing", wrong for grid-loss billing.
- **a18 · OPEN.** No outage / festival / holiday-load calendar. A meter that loses power during the holdout window reports as high MAPE, and the ledger currently cannot tell a modeling miss from a real outage.
- **a19 · OPEN.** Time-block boundaries in `_features.py` (night 22–06, morning 06–10, solar 10–18, peak 18–22) do not match the boundaries cited in the v2 prototype README (23–06 / 10–16 / 16–23). One of them is wrong. Reconcile before the next commercial brief quotes block MAPEs.
- **a20 · FACT.** Training is chronological 75/25, no shuffling. Holdout MAPE is out-of-sample.
- **a21 · HEURISTIC.** Metrics in the prototype are computed at the selected granularity — so a 1-hr MAPE and a 30-min MAPE are both real but answer different operational questions. Documented in the UI; must survive into any downstream report.
- **a22 · HEURISTIC.** R² and q10–q90 coverage are diagnostics, not acceptance gates. MAPE is the gate.
- **a23 · FACT.** Test suite (`tests/test_v4_predict.py`) enforces MAPE parity between bundle holdout and `benchmarks/results/benchmark_strategy1_v4.csv` within ±0.01%.

### Product

- **a24 · HEURISTIC.** Horizon defaults to 48 half-hours (1 day). Longer horizons work but confidence bands widen meaningfully past 7 days.
- **a25 · HEURISTIC.** Native cadence is 30 min; derived sub-30-min displays (5/15/20/45 min) are proportional redistributions, not sub-30-min predictions. The prototype labels this explicitly; downstream decks must too.
- **a26 · OPEN.** No retraining cadence is codified. Today we retrain manually when fleet MAPE drifts; v5 should ship a scheduled retrain.

### Commercial

- **a27 · FACT.** The engine predicts demand. All landed-cost math (IEX DAM + APEPDCL wheeling + cross-subsidy surcharge + losses + taxes) happens downstream in the BESS Explorer using the forecast as an input, not inside the model.
- **a28 · HEURISTIC.** Block-level MAPE is the number we quote commercially (not overall MAPE) because the commercial value is concentrated in the peak and solar blocks. Night MAPE being off by 15% may be tolerable; peak MAPE being off by 15% moves money.
- **a29 · OPEN.** We have no formal SLA commitment attached to v4 MAPE yet. Before one goes into a Commercial Brief, a28 plus an outage-adjusted version of a18 need to be resolved.
- **a30 · FACT.** MODEL_VERSION = `v4.s1.0`. The `.s1.0` suffix encodes "schema 1, patch 0"; a breaking change to bundle schema bumps the `s`, a training-recipe change bumps the patch.

---

## 11. What v4 doesn't answer (the v5 queue)

In priority order, independent of this doc:

1. Live IEX / realized-weather conditioning (a11).
2. Scheduled retraining cadence (a26).
3. Per-meter or per-substation weather geography (a05).
4. Outage / holiday calendar (a18) — modeling miss vs. real outage.
5. Joint quantile objective so q10/q90 no longer need a monotonicity clamp (a12).
6. Block-boundary reconciliation between engine and docs (a19).
7. Coincidence factor and line-loss adjustment in the fleet aggregate (a17).
8. Fallback weather conditions when Open-Meteo is unreachable (a15).

Each of these is small on its own; together they are what moves v4 from "trusted in a prototype" to "underwriting a commercial SLA."

---

*Source of truth: `src/edgegrid_forecast/inference/v4_predict.py`, `src/edgegrid_forecast/inference/_features.py`, `models/v4/_manifest.json`. Cross-check any number in this doc against those three files before citing it in a commercial brief.*
