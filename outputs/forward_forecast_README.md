# Forward Forecast — Apr 21 -> May 21, 2026

- Generated: `2026-04-21T07:53:49.781365+00:00`
- Model version: `v5.s1.0-seasonal-anchor`
- Horizon: `2026-04-21T00:00:00` -> `2026-05-21T00:00:00` (30 days, 1440 x 30-min slots)
- Meters forecast: **42/42**
- Rows written:
  - `outputs/forward_forecast_30d.parquet` -- **60,480** rows
  - `outputs/forward_forecast_dam_15min.parquet` -- **120,960** rows (half-kWh broadcast of each 30-min row into two 15-min sub-slots)

## Methodology -- Seasonal-Anchor Priming (option B)

The last verified actual in `data/raw/{sp,tp}_data.parquet` is **2026-02-12 ~15:00** -- 67 days before the forecast window starts (2026-04-21) and 97 days before it ends (2026-05-21). Rolling a pure recursive forecast forward through 67+30 = **97 days of prediction-fed lags** accumulates massive drift: every lag_N / rmean_N / diff_N feature ends up reading from earlier predictions, not real meter data.

Instead we use a **seasonal-anchor priming** strategy:

1. For each meter, take the last **4 weeks** of real 30-min actuals ending 2026-02-12 and compute the **(day-of-week, hour, minute) median** of `demand_wh`. This gives us a canonical weekly profile for each meter with weather and behaviour from the most recent observed window.
2. Lay down a **35-day synthetic pseudo-history** ending at `2026-04-20 23:30` whose values are that (dow, hour, minute) median aligned to the pseudo-history dates. This yields a fully-populated lag buffer so `lag_1..lag_336` (7 days), `rmean_6..rmean_1440` (30 days) and the similar-day features are all stable and in-distribution when the recursive predictor starts at 2026-04-21 00:00.
3. Run `edgegrid_forecast.inference.v5_predict.predict_recursive` over the 1440 half-hourly horizon slots. v5 predicts one step at a time, feeding its own output back as demand_wh for the next step, producing mean + q10 + q90 per slot.
4. Weather features: the Open-Meteo cache covers history through 2026-04-21. For the forward window we pad weather values from the **same day-of-year in 2025** (shifted by 365 days); any remaining gaps are ffilled. This climatological extrapolation keeps the temperature/humidity/GHI feature distributions realistic for Vizag in late April / May.
5. Output is converted from Wh-per-30min to **kWh-per-30min**, clamped >=0 with q10 <= mean <= q90, and written long-format.

### Why seasonal-anchor beats pure recursion across a 97-day gap

- The v5 recursive predictor was designed to absorb **prediction-fed lags over 24-168 hour horizons** (its training distribution). Beyond that, errors compound and the model drifts toward whatever happens to dominate its priors.
- By resetting the priming window to the *most recent observed weekly profile*, we anchor the forecast's level + shape to the last four weeks of actuals -- eliminating the 67-day compounded-error tail that pure recursion would incur.
- Weather sensitivity is preserved (climatology-shifted Apr/May features feed in), so the model still differentiates between cooler mornings and peak afternoons; only the *priming distribution* is seasonally anchored.

## Methods actually used
- `seasonal_anchor_weather_adjusted` -- **42 meter(s)**

## Data freshness caveat

Actuals end **2026-02-12 ~15:00**. The forecast window starts **2026-04-21**, leaving a 67-day observation gap. MAPE vs actuals cannot be computed for the forecast window until new actuals arrive. Block-accuracy can be computed retroactively once any fresh actuals land in `data/raw/{sp,tp}_data.parquet` -- the dashboard can join on `(meter_id, ts)` to build rolling MAPE.

## Sanity checks
- Meters with any NaN prediction: **0**
- Meters with any negative prediction: **0**
- Meters with any forecast > 10x historical 95th-percentile demand: **0**

## Output schema -- `forward_forecast_30d.parquet`

| column | dtype | description |
|---|---|---|
| meter_id | str | Meter serial number (UKSCNO/MSN) |
| ts | timestamp(ns) | 30-min block start, Asia/Kolkata naive |
| predicted_kwh | float64 | mean forecast, kWh per 30-min block |
| q10_kwh | float64 | 10th-percentile quantile, kWh |
| q90_kwh | float64 | 90th-percentile quantile, kWh |
| model_version | str | versioning tag for downstream audit |
| generated_at | str (ISO8601) | UTC timestamp at generation |

## Output schema -- `forward_forecast_dam_15min.parquet`

Same schema as above. Each 30-min row is broadcast to two 15-min rows (`ts` and `ts + 15min`), with `predicted_kwh`, `q10_kwh`, `q90_kwh` each divided by 2 so the 15-min totals sum back to the 30-min value. This matches the IEX DAM block cadence.

## Per-meter summary

| meter_id | status | method | total kWh | mean kWh/30m | last actual | wall s |
|---|---|---|---:|---:|---|---:|
| `67001151` | ok | seasonal_anchor_weather_adjusted | 14,047.4 | 9.755 | 2026-02-12 11:00:00 | 0.0 |
| `67003234` | ok | seasonal_anchor_weather_adjusted | 13,889.7 | 9.646 | 2026-02-12 11:30:00 | 0.0 |
| `67003309` | ok | seasonal_anchor_weather_adjusted | 12,058.9 | 8.374 | 2026-02-12 11:30:00 | 0.0 |
| `67001818` | ok | seasonal_anchor_weather_adjusted | 9,509.7 | 6.604 | 2026-02-12 11:30:00 | 0.0 |
| `67003694` | ok | seasonal_anchor_weather_adjusted | 8,959.9 | 6.222 | 2026-02-12 11:30:00 | 0.0 |
| `65021169` | ok | seasonal_anchor_weather_adjusted | 698.7 | 0.485 | 2026-02-12 11:00:00 | 0.0 |
| `65044028` | ok | seasonal_anchor_weather_adjusted | 1,922.2 | 1.335 | 2026-02-12 11:00:00 | 0.0 |
| `53401842` | ok | seasonal_anchor_weather_adjusted | 899.8 | 0.625 | 2026-02-12 10:00:00 | 0.0 |
| `65049719` | ok | seasonal_anchor_weather_adjusted | 1,009.0 | 0.701 | 2026-02-12 11:00:00 | 0.0 |
| `65007036` | ok | seasonal_anchor_weather_adjusted | 1,802.8 | 1.252 | 2026-02-12 11:30:00 | 0.0 |
| `53408407` | ok | seasonal_anchor_weather_adjusted | 1,428.0 | 0.992 | 2026-02-12 11:00:00 | 0.0 |
| `65011155` | ok | seasonal_anchor_weather_adjusted | 1,490.4 | 1.035 | 2026-02-12 11:30:00 | 0.0 |
| `65017058` | ok | seasonal_anchor_weather_adjusted | 1,217.7 | 0.846 | 2026-02-12 11:30:00 | 0.0 |
| `65023784` | ok | seasonal_anchor_weather_adjusted | 808.5 | 0.561 | 2026-02-12 11:30:00 | 0.0 |
| `65004669` | ok | seasonal_anchor_weather_adjusted | 1,016.0 | 0.706 | 2026-02-12 11:30:00 | 0.0 |
| `65021124` | ok | seasonal_anchor_weather_adjusted | 971.0 | 0.674 | 2026-02-12 11:30:00 | 0.0 |
| `65025443` | ok | seasonal_anchor_weather_adjusted | 1,069.2 | 0.743 | 2026-02-12 11:30:00 | 0.0 |
| `65022501` | ok | seasonal_anchor_weather_adjusted | 1,128.5 | 0.784 | 2026-02-12 11:00:00 | 0.0 |
| `65003175` | ok | seasonal_anchor_weather_adjusted | 1,125.0 | 0.781 | 2026-02-12 11:00:00 | 0.0 |
| `53403416` | ok | seasonal_anchor_weather_adjusted | 645.7 | 0.448 | 2026-02-12 11:30:00 | 0.0 |
| `65000228` | ok | seasonal_anchor_weather_adjusted | 656.0 | 0.456 | 2026-02-12 11:00:00 | 0.0 |
| `53407938` | ok | seasonal_anchor_weather_adjusted | 1,273.5 | 0.884 | 2026-02-11 10:00:00 | 0.0 |
| `65002231` | ok | seasonal_anchor_weather_adjusted | 1,061.3 | 0.737 | 2026-02-12 11:30:00 | 0.0 |
| `65001891` | ok | seasonal_anchor_weather_adjusted | 1,013.6 | 0.704 | 2026-02-12 11:30:00 | 0.0 |
| `65022302` | ok | seasonal_anchor_weather_adjusted | 757.6 | 0.526 | 2026-02-12 11:30:00 | 0.0 |
| `53408951` | ok | seasonal_anchor_weather_adjusted | 1,048.8 | 0.728 | 2026-02-12 11:30:00 | 0.0 |
| `53401885` | ok | seasonal_anchor_weather_adjusted | 751.8 | 0.522 | 2026-02-12 11:00:00 | 0.0 |
| `65024185` | ok | seasonal_anchor_weather_adjusted | 606.5 | 0.421 | 2026-02-12 11:30:00 | 0.0 |
| `65023781` | ok | seasonal_anchor_weather_adjusted | 944.3 | 0.656 | 2026-02-12 11:30:00 | 0.0 |
| `65021964` | ok | seasonal_anchor_weather_adjusted | 771.3 | 0.536 | 2026-02-12 11:00:00 | 0.0 |
| `50154700` | ok | seasonal_anchor_weather_adjusted | 792.5 | 0.550 | 2026-02-12 15:00:00 | 0.0 |
| `65015026` | ok | seasonal_anchor_weather_adjusted | 763.9 | 0.530 | 2026-02-12 11:00:00 | 0.0 |
| `65041990` | ok | seasonal_anchor_weather_adjusted | 843.7 | 0.586 | 2026-02-12 11:00:00 | 0.0 |
| `65030697` | ok | seasonal_anchor_weather_adjusted | 554.2 | 0.385 | 2026-02-12 11:30:00 | 0.0 |
| `65045250` | ok | seasonal_anchor_weather_adjusted | 831.3 | 0.577 | 2026-02-12 11:30:00 | 0.0 |
| `65024487` | ok | seasonal_anchor_weather_adjusted | 542.0 | 0.376 | 2026-02-12 11:00:00 | 0.0 |
| `65015739` | ok | seasonal_anchor_weather_adjusted | 591.0 | 0.410 | 2026-02-12 11:00:00 | 0.0 |
| `51057607` | ok | seasonal_anchor_weather_adjusted | 637.6 | 0.443 | 2026-02-12 11:30:00 | 0.0 |
| `65012662` | ok | seasonal_anchor_weather_adjusted | 604.7 | 0.420 | 2026-02-12 11:30:00 | 0.0 |
| `65003102` | ok | seasonal_anchor_weather_adjusted | 508.2 | 0.353 | 2026-02-12 11:00:00 | 0.0 |
| `50186364` | ok | seasonal_anchor_weather_adjusted | 835.1 | 0.580 | 2026-02-12 14:30:00 | 0.0 |
| `50143025` | ok | seasonal_anchor_weather_adjusted | 58.5 | 0.041 | 2026-02-12 14:30:00 | 0.0 |
