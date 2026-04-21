# Demand Forecast Engine · design prototype v0

A disposable single-file HTML prototype. Open `index.html` directly in any browser — no dev server, no build step, ~660 KB self-contained.

Built as a **cross-domain communication reference**: the chart is for the analyst, the assumption ledger on the right is for the commercial / design / exec / DISCOM reader who has to trust the numbers without reading the Python.

## Three-question framing

- **What question is this answering?** When a non-forecasting stakeholder sees a per-meter forecast curve, what do they need *next to it* to trust the number? Is it the chart, the MAPE pill, the block-level breakdown, or the assumption ledger?
- **Whose hands is this for?** Cross-functional. APEPDCL ops engineers, EdgeGrid commercial leads, design partners, board observers. Not the ML team — they read code, not prototypes.
- **What does success look like?** Reader can do three things in 60 seconds — pick a meter, read the forecast, and explain in one sentence what assumption the number depends on. If they can't do that third thing, the assumption ledger needs to be re-authored before we ship anything that quotes these numbers in a brief.

## What's in the fixture

Generated from the real APEPDCL MDMS dump (`data/raw/tp_data.parquet`, 653,713 rows across 45 MSNs):

- 42 meters that match the demo network graph
- 38 with real 7-day MDMS history (336 blocks at 30-min cadence)
- 4 with synthetic tier-baseline curves — flagged with a `SYNTHETIC` pill
- 48-hour forward projection for each, with ±MAPE band
- 3 substation aggregates (sum of member meters · assumption a19 applies)
- Block-level MAPE (Night / Morning / Solar / Peak) per meter
- Global headline: 9.58% mean MAPE, 7.18% median, 25 green / 6 amber / 11 red

## What's intentionally NOT in the fixture

- **The forward curve is NOT the live LightGBM forecast.** It's an honest extrapolation from each meter's own day-of-week × time-of-day median, scaled by that meter's holdout MAPE. Documented as assumption a15. Swapping in `/forecast/{msn}` from the production API is a one-endpoint change.
- No live IEX prices, no weather forecast, no holiday calendar — those affect the *production* forecast but don't affect the *story* this prototype is testing.
- No retraining queue, no drift detection, no model lineage — those belong to a different prototype.

## The assumption ledger

Twenty assumptions across five categories — Data, Model, Validation, Product, Commercial. Each is tagged:

- **FACT** — true today, by design or observation (e.g. one model per meter, holdout window length)
- **HEURISTIC** — a defensible choice that could move (e.g. health bucket thresholds, gap interpolation depth)
- **OPEN** — a known limitation or hand-wave (e.g. straight-sum substation aggregation, no outage calendar yet)

The ledger is filterable by category. Each meter's KPI strip names the specific assumption IDs that apply to its number.

## Time box

Built in one session. If a cross-domain reader can't pick a meter, read the chart, and reference an assumption inside 60 seconds of their first reaction, the ledger is too long, the chart is too busy, or both — and we redesign before any of these numbers go into a Commercial Brief.
