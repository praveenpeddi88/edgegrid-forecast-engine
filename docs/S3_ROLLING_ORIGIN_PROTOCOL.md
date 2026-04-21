# S3 — Rolling-Origin Validation Protocol

**Status:** design locked, execution starting 2026-04-21
**Owner:** Praveen + engine team
**Depends on:** v4 LightGBM bundle interface, per-meter actuals through 2026-02-12, Open-Meteo weather cache through 2026-04-21.

## 1. Purpose

Prove that the v4 forecasting engine is **production-grade** — not just accurate at one chronological cutoff (S1) or one stratified holdout pattern (S2), but **stable across the operational cadence of daily re-forecasting** that production demands. S3 is the gate before v4 can be declared deployable.

## 2. Outcome we're testing against

> *A production-grade demand forecasting engine that predicts demand accurately, per meter, for each of the 42 smart meters — deployable as a service.*

Three properties:

1. **Accurate** — per-meter MAPE under the cohort-specific bar.
2. **Stable** — bar holds across many forecast origins, not just one.
3. **Operable** — reproducible, versioned, callable as a daily predict API.

S1 and S2 gave us (1). S3 gives us (2). (3) is engineering hygiene that lives alongside.

## 3. Pass bar — per cohort, per origin

| Cohort | Meters | Bar |
|---|---|---|
| **A** | 8 | Every meter, every origin, every 30-min horizon step ≤ **5% MAPE** |
| **B** | 29 | Fleet median ≤ **5% MAPE**, p95 ≤ **10% MAPE** — across every origin |
| **C** | 4 | Fleet median ≤ **15% MAPE**, cross-origin σ ≤ **5pp** (stability > absolute accuracy) |
| **D** | 1 (50186364) | Cross-origin σ ≤ **10pp**; absolute MAPE not gated — bar is *predictably bad, not erratic* |

If the surface passes all four bars across all 10 origins, **v4 ships as production**. If it fails, the failure surface tells us exactly which (meter × origin × horizon) triple broke and whether it's a model, data, or weather problem.

## 4. Origin schedule

- **10 weekly origins**, spaced 7 days apart.
- Origins: `2025-12-04, 2025-12-11, 2025-12-18, 2025-12-25, 2026-01-01, 2026-01-08, 2026-01-15, 2026-01-22, 2026-01-29, 2026-02-05`.
- Last origin (2026-02-05) forecasts 2026-02-05 → 2026-02-12, matching the last 7 days of actuals we have.
- At each origin t, the engine sees actuals through `t - 30 min` (no leakage) and forecasts the next **7 days = 336 × 30-min slots**.

## 5. Per-origin procedure

1. **Refit, not retrain.** At each origin, refit the 42 v4 quantile bundles against a trailing window of actuals ending at `t - 30 min`. Tree structure preserved; leaf weights updated. This matches what we'd do in production — we don't retrain trees daily.
2. **Recompute trailing-21d MBE bias correction** at each origin. Mirror production.
3. **Forecast horizon = 7 days at 30-min native cadence.** Inference path = v5 recursive (production path), with the production safeguards on (val-gated, 30% cap, 0.5× dampening).
4. **Broadcast to 15-min DAM** for the dashboard view; also aggregate to hourly and ToD-4h for cadence comparability.

**Post-mortem escalation:** after the 10-origin sweep, pick the 2–3 worst origins by any cohort bar and do a **full retrain** (not just refit) to see whether instability is a model-capacity issue or a data-drift issue.

## 6. Metrics captured per (meter × origin × horizon step × cadence)

- `mape` — absolute percentage error, actuals > 0.5 Wh mask.
- `mbe_kwh` — mean bias error in kWh.
- `q10_coverage`, `q90_coverage` — fraction of actuals inside q10–q90 band (calibration check).
- `predicted_kwh`, `actual_kwh` — raw values for forensic drill-down.
- `horizon_step` — 1..336 (30-min), so we can see horizon-decay curves.

## 7. Aggregations

- **Per-meter per-origin** — 7-day MAPE summary (one number per meter per origin).
- **Per-cohort per-origin** — fleet median + p95 of meter MAPE.
- **Per-meter per-horizon-step** — does accuracy degrade as horizon extends from h=1 (30 min ahead) to h=336 (7 days ahead)?
- **Per-cadence** — same surface at native-30, hourly, ToD-4h, 15-min DAM.

## 8. Deliverables

1. **`outputs/s3_rolling_origin.parquet`** — long format: `meter_id × origin × horizon_step × cadence → metrics`.
2. **`outputs/s3_pass_fail_summary.csv`** — one row per `(cohort × origin)` with pass/fail flag and offending meter list.
3. **`outputs/s3_dashboard.html`** — interactive surface viewer: select cohort → see 10-origin × 336-horizon MAPE heatmap per meter, with cadence toggle and pass-bar overlay.
4. **`docs/S3_RESULTS.md`** — written verdict: *production-grade yes/no*, cohort-by-cohort, named v5 retraining priorities if any.

## 9. Compute budget

- **Refit-only**: ~30–45 s per bundle × 42 bundles × 10 origins ≈ **2.1–3.1 hr compute**. Backgroundable.
- **Forecast**: recursive at ~128 s per meter for 30 days ≈ ~30 s per meter for 7 days × 42 meters × 10 origins ≈ **3.5 hr**. Parallelisable.
- **Aggregation + dashboard build**: ≤ 15 min.

Total realistic wall-clock: **6–7 hr** if run serially, **1–2 hr** if meters parallelised. Fits inside an overnight run.

## 10. Done-when

- Parquet + summary CSV + dashboard HTML exist and are reproducible from `benchmarks/s3_rolling_origin.py`.
- `docs/S3_RESULTS.md` contains the verdict and — if v4 fails — the named retrain priorities.
- A dry-run on 1 origin has been executed first to catch wiring issues before the 10-origin sweep is committed.

## 11. Dry-run findings (origin 2026-01-29, executed 2026-04-21)

Findings that supersede the original §5/§9 design assumptions:

1. **No refit path exists.** LightGBM `.refit()` is not used anywhere in the repo. Production training does a full Pass1+Pass2+quantile retrain via `v5_retrain_one.retrain_meter`. The `--mode refit` flag falls back to full retrain. **The protocol now is: full retrain at every origin.** This is acceptable because measured retrain time is only 5.2 s/meter.
2. **Real per-origin timing**: retrain ~218 s, forecast (recursive 7d) ~1300 s, metrics ~6 s → **~26 min/origin serial**.
3. **10-origin sweep extrapolates to ~3.7 hr serial.** Fits one overnight run. No parallelisation required.
4. **Per-step MAPE inflates on near-zero actuals** (single sub-EPS slot can balloon the simple-average MAPE for a meter to 100%+). The block-aggregated cohort numbers from `block_accuracy.cohort_rollup` are more honest. Reading rule for §3 pass bars: evaluate the per-cohort cadence rollups, not the per-step means.
5. **Eyeball at this single origin**: A median 9.5%, B median 15.8%, C median 21.4%, D 41.4% at native-30min. Pass bars look hard without v5 work — but the surface is intact and the full sweep will tell us whether Jan 29 is representative or an outlier.

Artifacts from this dry-run live at `outputs/s3/2026-01-29T0000/` and refit bundles at `models/v4_s3/2026-01-29T0000/`. Production bundles at `models/v4/` are untouched.
