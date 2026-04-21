# Forecast Engine · v3 · 76-day horizon, DOW priming, month/day picker

A disposable single-file HTML prototype. Open `index.html` directly in any browser — no dev server, no build step, ~8.4 MB self-contained (D3 from CDN, all 76 days × 42 meters of forecast data inlined).

v1 used seasonal-naive extrapolation. v2 was the real v4 LightGBM engine evaluated on a 7-day held-out window. **v3 pushes the horizon to 76 days (Feb 5 – Apr 21, 2026)** using same-day-of-week priming, splits the window into an 8-day verified backtest and a 70-day pure-forecast tail, and puts a month + day picker on top so a reader can interrogate any day in the window.

## Three-question framing

- **What question is this answering?** Now that we have v4 models that score ~8% MAPE on their in-pipeline 7-day holdout, what happens when we actually run them 76 days into the future — the horizon that matters for commercial contracts, APEPDCL's April billing cycle, and the summer peak-season conversation? And can a non-ML stakeholder *see* that gap for themselves at any granularity (15 min for ramping, 1 hr for ToU, 4 hr for shift planning)?
- **Whose hands is this for?** EdgeGrid commercial + APEPDCL ops + design partners. The test reader is someone deciding whether they'd quote a savings number off the forecast engine for an April or May contract. Not the ML team — they read the joblib bundle.
- **What does success look like?** Reader picks a meter, lands on a Feb day (verified), sees the forecast line sitting near the actual. Then picks an Apr day (pure forecast) and sees the forecast line alone — the UI says "pure forecast, no truth" explicitly. They walk away knowing (a) the in-pipeline MAPE and the multi-week-horizon MAPE are not the same number, (b) *how different* they are for their specific meter, and (c) that the assumption ledger covers why. Question closed — either scale this with confidence bounded by the real numbers, or fix the gap (tighter priming, cascade, re-fit) before quoting.

## The aha moment

For each meter, the chart shows two periods:

- **History** (last 14 days before Feb 5, gray line) — actuals before the context cutoff.
- **Forecast** (Feb 5 → Apr 21, teal line + q10/q90 band) — produced by the v4 LightGBM bundle. Each day is forecast independently using same-day-of-week lag priming (see "DOW priming" below).
- **Held-out actual** (dashed white line, only Feb 5 → Feb 12 15:00) — the truth the model is being scored against for the verified window. Past Feb 12 15:00, there is no dashed line because there is no actual yet.

The day banner above the chart tells the reader which regime they're in: **VERIFIED** (teal, in the 8-day window), **PURE FORECAST** (violet, Feb 13 onwards), or **MIXED** (edge case on Feb 12).

Fleet header shows **mean 63.4% / median 46.0% real backtest MAPE** across the 42 modeled meters. That is the honest 76-day number — and it is very different from the bundles' self-reported ~8% (see "Reference MAPEs" below).

## The month + day picker

Above the chart sits a **Month** segmented control (Feb / Mar / Apr) and a **Day grid** that snaps to the selected month. Days inside the verified window carry a teal dot. The reader can pick any of the 76 days and the forecast / q10-q90 band / metrics strip / block quad / right-panel ledger all re-scope to that one day.

This is the core v3 ask — 76 days is too many for a single flat segmented control. Month-then-day keeps the picker honest about the scale of the window without hiding the tail.

## What the granularity switcher means (same as v2)

Nine choices: **5 · 15 · 20 · 30 · 45 min · 1 · 2 · 4 · 7 hr**. Default is **1 hr** (not 30 min). At a 76-day horizon, 30-min MAPE is noisy and 1-hr is the honest contract-sized number. Tags remain:

- **NATIVE (30 min)** — the model's actual predictions.
- **AGGREGATED (1, 2, 4, 7 hr)** — sums of consecutive 30-min blocks. Exact for energy.
- **DERIVED (5, 15, 20, 45 min)** — proportional redistribution of 30-min blocks. Not a sub-30-min forecast. Do not cite as one.

The honesty-level tag and the right-panel context box carry forward from v2.

## Reference MAPEs — and why they differ

The meter title pill now shows **two** numbers side by side: `bundle XX%` (what the model scored on its in-pipeline 7-day holdout, before training was frozen) and `XX% backtest` (what this prototype measured across the 8 verified days of v3 at 30-min granularity).

**Both are real. They answer different questions.** The bundle number is "how well does this model do on a week of data drawn from the same distribution as training." The v3 backtest is "how well does this model do when we prime it with same-day-of-week lags from the context cutoff and run it at a 1–8 day horizon." When those two numbers diverge — and they do, a lot — the gap is the DOW-priming + horizon tax, and it is exactly the number a commercial reader needs. The right-panel now surfaces both with per-block breakdowns (night / morning / solar / peak) so you can see *where* the gap lives. See assumption a24.

## Pure-forecast tail — honest empty state

For days past Feb 12 15:00 there is no actual, so:

- The dashed actual line does not render.
- The metrics strip shows one violet-toned empty-state card: *"No held-out actuals on this day · pure forecast (a25)"*.
- The block quad is muted and labeled "pure forecast — no truth".
- The day banner is violet with the PURE FORECAST tag.

No phantom MAPE, no misleading small numbers, no "don't worry about it" styling. The reader who lands on April 15 sees unambiguously that the system is forecasting but not being scored.

## What's intentionally NOT in the fixture

- **No live IEX or weather signals through the tail.** The 76-day forecast falls back to learned typical conditions past the latest weather observation. Wiring real-time exogenous feeds lands in v5. Documented as assumption a10.
- **No horizon-decay correction in the UI.** Mar 20 is treated the same as Feb 10 in terms of how we render it. This is deliberate — we want the reader to notice whether the forecast line degrades with horizon and to push back if it does. (a26)
- **No outage / festival / holiday calendar.** Ugadi, Holi, Ram Navami, Eid, Mahavir Jayanti all sit inside Feb 5 – Apr 21 and will show up as big misses if the meter's customer actually shut down. Not a model problem, a calendar-feature problem. (a18)
- **No coincidence factor or line-loss adjustment in the fleet aggregate.** Simple sum. (a17)

## Metrics at the selected granularity (same discipline as v2)

MAPE, MAE, RMSE, MBE, sMAPE, R², coverage, worst block — all recomputed at whatever granularity and day you picked. Larger buckets usually show lower MAPE because errors cancel inside the bucket. A 30-min 76-day backtest MAPE is the strictest truth test; 1-hr is the contract-sized test. (a21)

## The assumption ledger — now 26 entries

22 from v2 carry over. Four new ones anchor the v3 honesty story:

- **a23** · 76-day horizon with DOW priming, no cascade. Why independent-day forecasts instead of a rolling cascade — and what that costs us.
- **a24** · Backtest MAPE ≠ Bundle holdout MAPE. This is not a bug, this is the finding.
- **a25** · Pure-forecast tail (Feb 13 → Apr 21) has no truth. No metrics will be shown for those days, deliberately.
- **a26** · No horizon-decay caveat surfaced. A test of whether the reader notices drift on their own.

Plus a18 now lists the specific Feb–Apr festivals and a20 carries the v3 honest-finding language.

## Time box

Built directly on top of the v4 inference wrapper and the v2 prototype scaffold. If a cross-domain reader can't pick a meter, switch to an April date, see the violet "pure forecast" banner, and walk through what the backtest vs bundle numbers imply inside 2 minutes — the day picker is too crowded, the banner is too quiet, or the two-pill title is confusing. We redesign before any of these numbers go into a Commercial Brief or Strategy-1 Courier Pack.

## Fleet headline

- **Mean backtest MAPE** (8 verified days, 30 min, across 42 modeled meters): **63.42%**
- **Median:** 46.04%
- **Health distribution:** 0 green, 8 amber, 34 red
- **Bundle holdout mean** (for comparison, 7d, in-pipeline): ~9.6%

The gap *is* the story. This prototype is how we show it.

## Files

- `index.html` — the prototype (~8.4 MB, self-contained)
- `forecasts_v3.json` — the fixture (~8.3 MB, also embedded into `index.html` so the file works offline / from `file://`)
- `index_template.html` — the template used to generate `index.html` (`__FIXTURE_JSON__` placeholder)
- `README.md` — this file
