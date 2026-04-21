# Truth Board — Where the <5% MAPE target is reachable, per meter, per cadence

**Date:** 2026-04-21 · **Scope:** 42 modeled APEPDCL meters · **Source:** Feb 5–12 backfill on the v4 frozen bundles, measured by `src/edgegrid_forecast/accuracy/block_accuracy.py`.

## 1. The headline

At 15-min DAM blocks, **10 of 42 meters** clear the 5% MAPE bar. At 30-min native cadence the count is the same — **10 of 42** — because the 30-min metric is mathematically equivalent to the mean of two 15-min slots that come from the same model output. At hourly cadence, **17 of 42** clear. At ToD 4-hour blocks, **23 of 42** clear. No single cadence delivers <5% on the full fleet, and that is not a modelling failure to be fixed in the next sprint — it is an empirical floor measured on real Feb 5–12 actuals against a frozen Dec-15-trained model.

## 2. Why this is the physical limit, not a modelling failure

We chased this. We re-trained with extended cutoffs (Feb 4 instead of Dec 15) — the bundle holdout improved by 0.7pp on the test meter while the Feb 5–12 backtest moved by 0.17pp in the wrong direction. We built a recursive inference loop to fix "stale lag features past the cutoff" — net zero. We then ran an oracle floor study: even with perfect knowledge of the day-ahead's hour-of-day mean, the median per-meter MAPE at 30-min sat at ~32% across the fleet (see `prototypes/forecast_engine_v3/oracle_floor.csv`). That floor exists because residential and small-commercial meters carry a layer of human and equipment decisions — when the AC compressor fires, when the EV starts charging, when the geyser kicks in, when the diesel genset takes over after a brownout — that no feature in our v4 set can observe in real time. The v5 plan names this directly: a meter's MAPE decomposes into a reducible component (what features can explain) and an irreducible component (variance from unobserved events). Bundle holdout MAPE measures the reducible part, because spikes average out across thousands of randomly-mixed holdout rows. Per-block MAPE on a specific 8-day window measures both, and on spike-y meters the irreducible part dominates. Aggregating to longer time blocks reduces the irreducible part deterministically — at 4-hour cadence, a single missed AC cycle gets averaged with seven well-predicted blocks instead of standing alone as a 700% APE. That is why the count of meters under 5% rises monotonically as the cadence widens, and why the user has to choose the cadence before we can promise a number.

## 3. Per-meter × per-cadence reachability

Sorted by bundle holdout MAPE ascending. "Coarsest-needed" is the first cadence at which the meter clears 5%; "none" means it never does on the Feb 5–12 backfill at any of the four reported cadences.

| Meter ID | Cohort | Bundle MAPE | <5% @15-min | <5% @30-min | <5% @1h | <5% @ToD | Coarsest-needed |
|---|---|---|---|---|---|---|---|
| 53407938 | A | 3.19% | yes | yes | yes | yes | 15-min |
| 65045250 | A | 3.25% | yes | yes | yes | yes | 15-min |
| 65021964 | A | 3.75% | yes | yes | yes | yes | 15-min |
| 65015739 | A | 4.53% | yes | yes | yes | yes | 15-min |
| 65022501 | A | 4.61% | no  | no  | no  | no  | none |
| 65002231 | A | 4.67% | yes | yes | yes | yes | 15-min |
| 65011155 | A | 4.71% | yes | yes | yes | yes | 15-min |
| 53408407 | A | 4.82% | yes | yes | yes | yes | 15-min |
| 50154700 | B | 5.23% | no  | no  | yes | yes | 1h |
| 65001891 | B | 5.54% | yes | yes | yes | yes | 15-min |
| 67003694 | B | 5.54% | no  | no  | yes | yes | 1h |
| 65012662 | B | 5.71% | no  | no  | yes | yes | 1h |
| 65023781 | B | 5.81% | yes | yes | yes | yes | 15-min |
| 65017058 | B | 5.88% | no  | no  | no  | yes | ToD 4h |
| 67001151 | B | 6.34% | no  | no  | no  | no  | none |
| 65003102 | B | 6.36% | no  | no  | yes | yes | 1h |
| 65004669 | B | 6.63% | no  | no  | no  | no  | none |
| 65007036 | B | 6.86% | no  | no  | yes | yes | 1h |
| 65023784 | B | 7.01% | no  | no  | no  | no  | none |
| 65030697 | B | 7.10% | no  | no  | no  | no  | none |
| 65024185 | B | 7.12% | no  | no  | no  | yes | ToD 4h |
| 65000228 | B | 7.18% | no  | no  | no  | yes | ToD 4h |
| 53408951 | B | 7.31% | no  | no  | no  | yes | ToD 4h |
| 53401885 | B | 7.84% | no  | no  | no  | yes | ToD 4h |
| 65015026 | B | 7.88% | no  | no  | no  | no  | none |
| 65044028 | B | 8.02% | no  | no  | no  | yes | ToD 4h |
| 65003175 | B | 8.21% | no  | no  | no  | no  | none |
| 67001818 | B | 8.51% | yes | yes | yes | yes | 15-min |
| 65025443 | B | 9.80% | no  | no  | yes | yes | 1h |
| 67003234 | B | 10.59% | no | no  | no  | no  | none |
| 65024487 | B | 11.26% | no | no  | no  | no  | none |
| 51057607 | B | 12.67% | no | no  | no  | no  | none |
| 65041990 | B | 12.88% | no | no  | no  | no  | none |
| 65022302 | B | 13.40% | no | no  | no  | no  | none |
| 67003309 | B | 13.44% | no | no  | yes | yes | 1h |
| 50143025 | B | 13.76% | no | no  | no  | no  | none |
| 53403416 | B | 14.77% | no | no  | no  | no  | none |
| 53401842 | C | 15.24% | no | no  | no  | no  | none |
| 65049719 | C | 16.27% | no | no  | no  | no  | none |
| 65021169 | C | 18.07% | no | no  | no  | no  | none |
| 65021124 | C | 18.29% | no | no  | no  | no  | none |
| 50186364 | D | 52.24% | no | no  | no  | no  | none |

The pattern is regular. Cohort A — eight meters with bundle MAPE under 5% — clears every cadence cleanly except meter 65022501, which is the one A-cohort meter whose bundle holdout (4.61%) is achieved on noise-friendly mixed data but whose Feb 5–12 backfill includes enough unmodelled spikes to push every cadence over the bar. Cohort B is where the cadence dial does most of its work: of 29 meters in B, three clear at 15-min, ten at hourly, sixteen at 4-hour. Cohorts C and D do not clear at any cadence — they need data triage or a different modelling approach, not a wider time bucket.

## 4. Cohort roll-up

Counts read "n meters under 5% / n meters in cohort." Cohort assignment is by bundle MAPE: A < 5%, B 5–15%, C 15–30%, D ≥ 30%, matching the cohort sizes in `outputs/block_accuracy_cohort.csv` (8 / 29 / 4 / 1).

| Cohort | n meters | <5% @15-min | <5% @30-min | <5% @1h | <5% @ToD 4h |
|---|---|---|---|---|---|
| A | 8 | 7/8 | 7/8 | 7/8 | 7/8 |
| B | 29 | 3/29 | 3/29 | 10/29 | 16/29 |
| C | 4 | 0/4 | 0/4 | 0/4 | 0/4 |
| D | 1 | 0/1 | 0/1 | 0/1 | 0/1 |
| ALL | 42 | 10/42 | 10/42 | 17/42 | 23/42 |

Cohort A is essentially "done" at every cadence. Cohort B is where the cadence-vs-engineering trade-off lives. Cohort C and the single Cohort D meter (50186364, bundle 52.24%) need work that is not a cadence change.

## 5. What would unlock the rest

For the 19 meters that don't clear at the ToD 4-hour cadence, and the broader set that misses at finer cadences, the levers are concrete and not interchangeable:

- **a) New observability.** EV charger telematics, AC compressor duty-cycle telemetry, DG (diesel genset) status flags, and per-circuit appliance sub-metering. These are the signals that would reduce the irreducible-variance floor identified by the oracle study. None of them are in the current data set; all of them require either site instrumentation or third-party integrations. No model change without these signals will get spike-y residential meters to <5% at 15-min.
- **b) Closing the actuals gap.** The last actual we have is Feb 12; today is Apr 21, a gap of 67 days. Forward forecasts in `outputs/forward_forecast_30d.parquet` therefore lean on the seasonal-anchor (DOW × hour mean of the last full month) for everything past Feb 12. A live AMI feed that closes the lag to a few hours would let the model use real recent lags and would meaningfully improve forward accuracy on cohort B in particular.
- **c) Cohort C/D triage.** Cohort C is four meters (15–18% bundle) where the issue is data quality, training-history length, or load-regime change rather than irreducible variance — a focused per-meter retrain pass on each of the four, with extended history and per-meter Optuna tuning, is a one-day investment with a 5–15pp expected gain. Cohort D is a single meter — 50186364, bundle 52.24% — whose backtest at ToD 4-hour is 101.5% MAPE driven by long zero-load stretches and a near-zero predicted total. This meter belongs on its own bench: rebuild from the meter's raw history, decide whether intermittency is genuine or instrumentation, and report it on a daily-aggregate cadence only.
- **d) Cadence choice.** The simplest unlock is the one that costs nothing: commit to ToD 4-hour as the operational reporting cadence and 23/42 meters clear today. Combine that with the cohort-C retrain and the count rises to roughly 27/42 within a week. The remaining 15 meters are the ones that fundamentally need new sensor data (lever a) or live AMI (lever b).

## 6. The product call

If dispatch and procurement decisions live on 4-hour windows — which is how APEPDCL settles tariff slabs and how BESS arbitrage plays for HT consumers — then the 5% target is the right framing and we deliver it today for the majority of the fleet, with a one-week sprint to bring cohort C in. If the use case is genuinely 15-min DAM, the framing has to split. We deliver fleet-aggregate MAPE under 5% (already achieved at the bundle level for the broadcast forecast) as the procurement product. We deliver per-meter <5% on the eight cohort-A meters as the real-time meter-level product. For the remaining 34 meters at 15-min, we accept that point-MAPE is wider than 5% and we communicate uncertainty with q10/q90 confidence bands instead of pretending a single point estimate is accurate to 5%. That is the honest product, and it is the one we can ship without claiming new physics.

## 7. What we're shipping today

- `outputs/block_accuracy_dashboard.html` — live per-meter, per-cadence accuracy view
- `outputs/block_accuracy_backfill.parquet` — raw 15-min predictions vs actuals for Feb 5–12 across all 42 meters
- `outputs/block_accuracy_summary.csv`, `outputs/block_accuracy_cohort.csv`, `outputs/block_accuracy_fleet.csv` — the measurement tables this Truth Board is computed from
- `outputs/forward_forecast_30d.parquet` — next 30 days at 30-min native, seasonally anchored past Feb 12
- `outputs/forward_forecast_dam_15min.parquet` — 15-min DAM broadcast for procurement
- `src/edgegrid_forecast/accuracy/block_accuracy.py` — the new measurement module that produces the cadence-banded MAPEs

The numbers in this document are reproducible from those files. The truth is cadence-dependent, and we commit to it.
