# Forecast Engine · v2 · real v4 LightGBM, 9 granularities

A disposable single-file HTML prototype. Open `index.html` directly in any browser — no dev server, no build step, ~1.6 MB self-contained (D3 from CDN, all forecast data inlined).

This is the v2 of the forecast engine prototype. v1 used a seasonal-naive extrapolation as a placeholder. **v2 uses the real production v4 LightGBM models** — one model per meter, evaluated on actuals the model never saw during training.

## Three-question framing

- **What question is this answering?** Can a non-ML stakeholder *see* that the engine actually predicts the future, not just regurgitates the past — and can they slice that prediction at the time-block size their workflow uses (15 min for ramping, 1 hr for ToU contracts, 4 hr for shift planning)?
- **Whose hands is this for?** Cross-functional. APEPDCL ops, EdgeGrid commercial, design partners, board observers. Anyone who needs to trust a forecast number before quoting it. Not the ML team — they read the joblib bundle.
- **What does success look like?** A reader picks a meter, switches granularity, and sees the teal forecast line sitting on top of the dashed actual line in the held-out window. They say one of two things — "oh, it works" (question closed, scale this to v5) or "the [night/morning/solar/peak] block is consistently off" (question closed in a different direction — fix the block, then re-show).

## The aha moment

Each meter's chart shows two periods:

- **History** (last 14 days, gray line) — actuals before the train cutoff.
- **Forecast** (next 7 days, teal line + q10/q90 band) — produced by the LightGBM model with NO knowledge of those 7 days.
- **Held-out actual** (dashed white line, overlaid on the forecast window) — the truth the model is being scored against.

When the dashed line tracks the teal line, the model is right about the future. The fleet header shows mean / median real MAPE across all 42 modeled meters.

## What the granularity switcher means

Nine choices: **5 · 15 · 20 · 30 · 45 min · 1 · 2 · 4 · 7 hr**. Tagged in the UI as one of three honesty levels:

- **NATIVE (30 min)** — what the model actually predicts. The teal line and the dashed actual line are the model's exact outputs and the exact MDMS observations.
- **AGGREGATED (1, 2, 4, 7 hr)** — sums of consecutive 30-min blocks. Mathematically exact for energy. Use this when you need hourly contract math, ramp-window planning, or shift-level demand.
- **DERIVED (5, 15, 20, 45 min)** — *not* a sub-30-min prediction. We proportionally redistribute each 30-min block's energy across the smaller bins by overlap fraction. Useful for visual smoothness and for talking through what a smart meter at finer cadence *would look like* under uniform within-block load. Do not cite as a sub-30-min forecast.

Each granularity choice carries a colored tag (teal/green/amber) in the chart header and a context box in the right panel that re-states the honesty level. The reader cannot accidentally believe a 5-min number is a 5-min prediction.

## What's in the fixture

Generated from `models/v4/{msn}.joblib` × `_smdata_t_blp_tp/sp_*.txt` × `SCS_PMSGVENDOR.xlsx`:

- 42 meters with trained v4 LightGBM models (8 of 50 uploaded meters await retrain — see assumption a04)
- For each meter: 14 days of history + 7 days of forecast + matching held-out actuals
- Per-meter `holdout_mape_real`, `block_mape_real` (night / morning / solar / peak), `historical_block_mape` (the model's own self-reported training MAPE)
- Q10 / Q90 quantile band per meter
- Customer mapping: SCNO, UKSCNO, Phase from SCS export
- Fleet aggregate: per-timestamp simple sum across all 42 meters

Headline: **mean MAPE 11.46%, median 8.25%, p75 12.96%** — 18 green, 13 amber, 11 red.

## What's intentionally NOT in the fixture

- **No live IEX or weather signals** — the v4 inference path falls back to learned typical conditions. Wiring real-time signals lands in v5. Documented as assumption a10.
- **No outage / festival / holiday calendar** — a meter that lost power for 4 hours of the holdout window will show a high MAPE here, and that is a data-truth problem, not a model problem. (a18)
- **No coincidence factor or line-loss adjustment** in the fleet aggregate. Simple sum, suitable for the "what flow is this fleet drawing" view but not for grid-loss billing. (a17)

## Day picker + live metrics (new in v2.1)

Above the chart there is now a **Day** segmented control — "All 7" plus one button per day in the held-out window. Pick a day and the forecast / actual / q10-q90 band and all metrics re-scope to just that day.

Below the chart sits an **8-card metrics strip**, recomputed on every granularity or day change:

- **MAPE** — mean absolute percent error, excluding blocks where actual ≤ 0.001 kWh (zero-denominator guard)
- **MAE** — mean absolute error in kWh-per-block (so a 1-hr MAE reads in kWh/hr, a 7-hr MAE in kWh/7-hr-block)
- **RMSE** — root mean square error, same units as MAE; surfaces outlier blocks
- **MBE** — mean bias error (signed). Over-forecasting (MBE > 0) vs under-forecasting (MBE < 0) is labeled
- **sMAPE** — symmetric MAPE, robust when actuals and predictions are both small
- **R²** — coefficient of determination. Null when actual variance is zero (e.g. a flat outage day); otherwise 0…1
- **Coverage** — fraction of actuals that landed inside the q10-q90 band. Well-calibrated → ~80%
- **Worst block** — the single largest absolute error in the window, with its timestamp, so reviewers can jump to it

**Metrics are computed AT the selected granularity.** This is the deliberate choice — not a bug. If you pick 7 hr, MAE is kWh-per-7-hr-block and MAPE is measured on those aggregated sums. Larger buckets usually show lower MAPE because errors cancel inside the bucket. A 30-min MAPE is the strictest truth test; 7-hr is the operational-contract truth test. Both are real, they just answer different questions. See assumption a21.

The block-quad (Night / Morning / Solar / Peak) also re-scopes to the selected day and granularity, with Δ-vs-training badges so a reviewer can see immediately if the block is behaving worse on this day than the model's historical self-reported performance.

## The assumption ledger

Twenty-two assumptions across five categories — Data, Model, Validation, Product, Commercial. Each tagged FACT / HEURISTIC / OPEN. The right-side panel surfaces the assumption IDs that apply to whatever meter and granularity you're looking at, so a reader can trace any number on screen back to its assumption in one click.

The ledger now reflects the real v4 path (q10/q90 quantile heads, tier-adaptive hyperparameters, bias gate, held-out MAPE on data the model never saw) — not the v1 extrapolation. a21 documents the metric-follows-granularity choice; a22 positions R² and coverage as diagnostics, not acceptance gates.

## Time box

Built in one session on top of the v4 inference wrapper. If a cross-domain reader can't pick a meter, switch to "1 hr" granularity, see the forecast vs actual overlay, and reference an assumption ID inside 90 seconds of opening this file — the chart is too busy, the granularity tags are not loud enough, or the ledger is too long. We redesign before any of these numbers go into a Commercial Brief.

## Files

- `index.html` — the prototype (~1.6 MB, self-contained)
- `forecasts.json` — the fixture (~1.5 MB, also embedded into index.html so the file works offline / from file://)
- `index_template.html` — the template used to generate index.html (`__FIXTURE_JSON__` placeholder)
- `README.md` — this file
