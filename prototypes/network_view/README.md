# Network View · design prototype v0

A disposable, single-file HTML prototype. Open `index.html` directly in any browser — no dev server, no build step.

## Three-question framing

- **What question is this answering?** Does graph-first storytelling change the reaction of a DISCOM GM persona, or is it intellectually satisfying but commercially flat?
- **Whose hands is this for?** An APEPDCL GM / substation engineer / EdgeGrid commercial lead — someone who should understand the EdgeGrid pitch in 30 seconds of looking at one screen.
- **What does success look like?** One reaction back from a real-or-stand-in reader that either says *"oh, that's what a routing node means"* (question closed, build the real thing) or *"this is pretty but so what"* (question closed, kill the graph metaphor and return to the substation card view).

## What's in the fixture

Real-shape demo network generated from `edgegrid_forecast.graph.demo_data.build_demo_network()`:

- 69 nodes · 112 edges · 6 edge types
- 42 meters · 3 substations (canonical: Madhurawada VSKP-MW-01) · 9 feeders
- 4 BESS units · 3 solar plants · 6 C&I loads · 1 weather cell · 1 IEX price curve
- Mean holdout MAPE 9.58% (matches v4 model headline)

## What the UI shows

- Force-directed graph, scrollable / zoomable / draggable
- Meter health coloring (green <8%, amber 8–12%, red >12%) driven by real `holdout_mape` values
- Click a node → drilldown panel on the right with commercial context:
  - substation: BESS MWh, solar kWp, landed cost ₹/kWh, contracted kVA, forecast-health mix
  - meter: tier, feeder, block-MAPE (night/morning/solar/peak)
  - BESS: capacity, duration, SOC, RTE, CapEx
- Edge styling encodes relationship type (consumes, feeds, serves, dispatches, observes, prices)

## What's intentionally missing

- No forecast time series, no dispatch schedule, no IRR heatmap — those belong to the next prototype.
- No real GPS — lat/lon are jittered around substation centroids.
- No production frontend primitives — this is tailwind-ish CSS by hand, not shadcn/Vite.
- No tests — if the graph metaphor survives the reaction test, the real thing gets re-built under `frontend/` with the design system and tests.

## Time box

Built in one session. If this prototype doesn't close the question within one day of feedback, we kill the graph view and go back to the substation-card-first layout.
