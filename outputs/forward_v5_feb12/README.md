# Forward Forecast — Apr 21 → May 21 2026 (4 strategies)

- Generated: `2026-04-21T11:34:28.033553`
- Meters: **42** (v5 fleet, Feb-12 train cutoff)
- Horizon: `2026-04-21 00:00:00` → `2026-05-21 00:00:00` (30 days × 48 half-hourly = 1440 slots)
- Rows per strategy: **60,480** (42 × 1440)

## Strategies

| id | name | engine | priming | notes |
|----|------|--------|---------|-------|
| A  | seasonal_anchor  | none (median lookup) | 4-week (dow,h,min) | no weather, no model — baseline |
| B  | v4_batch         | v4 LightGBM quantile | 4-week seasonal pseudo-history + real pre | pre-v5 reference engine |
| C  | v5_batch_feb12   | v5 LightGBM quantile | 4-week seasonal pseudo-history + real pre | v5 bundles retrained through 2026-02-12 15:00 |
| D  | ensemble_blend   | per-(meter,ts) median(A, B, C) | — | robust to single-strategy outliers |

## Why priming?
Last verified actuals are 2026-02-12 ~15:00 — that's 67 days before the
forecast-window start (2026-04-21 00:00). A recursive forecast across 67+30 days
of prediction-fed lags accumulates massive drift. Instead we lay down a 35-day
synthetic pseudo-history built from (dow, hour, minute) medians of the last 4
weeks of real actuals, so the lag/roll/similar-day features evaluate against
in-distribution values.

## Output files

All parquets share the schema `meter_id, ts, predicted_kwh, q10_kwh, q90_kwh, strategy`.

- `strategy_A.parquet` — seasonal-anchor forecasts (42 × 1440 = 60,480 rows)
- `strategy_B.parquet` — v4 batch forecasts
- `strategy_C.parquet` — v5 batch forecasts (Feb-12 bundles)
- `strategy_D.parquet` — per-row median ensemble
- `all_strategies.parquet` — long-form union (241,920 rows)
- `per_strategy_rollup.csv` — fleet totals per strategy

## Known caveats

- **Batch vs recursive.** Strategies B and C use single-shot batch prediction on
  the primed pseudo-history. This matches how the bundles are trained (lags are
  seen directly) and is the same regime the bundle-holdout MAPE measures, but
  does NOT re-feed predictions back into lags mid-horizon. For recursive
  inference (one-step-ahead), use `edgegrid_forecast.inference.v5_predict.predict_recursive`.
- **Weather extrapolation.** Weather beyond `2026-04-21` is padded from
  same-day-of-year 2025, then ffilled. This is climatological, not a real forecast.
- **No ground truth.** These are forecasts, not backtests. When Apr-May 2026
  actuals arrive, per-meter MAPE can be computed by joining on `meter_id, ts`.
