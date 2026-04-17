# EdgeGrid Dispatch Console — Prototype v1

Interactive full-stack prototype that exercises the EdgeGrid forecasting engine
(C1–C3, already shipped) plus the new dispatch (C4) and commercial-quantification
(C5) layers, per the handover brief in [`docs/EDGEGRID_PRODUCT_SPEC.md`](../../docs/EDGEGRID_PRODUCT_SPEC.md) §Part V.

## Prerequisites

- Python ≥ 3.11 with the project installed (`pip install -e .` or `uv sync`).
- Node ≥ 20 (first-run `npm install` happens automatically).
- `models/v4/*.joblib` and `models/v4/_manifest.json` — shipped by Session 11.
- Optional: `data/sp_data.parquet` and `data/tp_data.parquet` for real `predict()` output.
  If missing, the API falls back to a synthetic substation-level demand profile keyed
  off the per-meter MAPE in the manifest — the demo still tells a coherent story.

## Run

From repo root:

```bash
./prototypes/dispatch_console/run_backend.sh     # → http://localhost:8000
./prototypes/dispatch_console/run_frontend.sh    # → http://localhost:5173
```

Open <http://localhost:5173> and you should see the EdgeGrid network render
within a few seconds.

## The 90-second demo

1. **Network home.** Click the Madhurawada 33/11kV substation node (teal, upper-right of the Visakhapatnam cluster). Side panel shows its dispatch summary.
2. **Dispatch.** Click *Open dispatch*. You see a 48-hour × 15-minute schedule with grid import, BESS charge/discharge, SOC, and an **audit column** on the right — every non-hold row explains itself in plain English.
3. **Commercial brief.** Click *Commercial brief* (top-right). IRR heatmap (capacity × duration), sensitivity, and an FLS quote generator (enter a buyer kW, click *Generate quote* — firmness % derives from the substation's best HT-meter peak-block MAPE).
4. **PDF.** Click *Export PDF (print view)* → opens the self-contained brief HTML in a new tab. Use browser Print → Save as PDF.
5. **Model version** lives in the footer — matches `models/v4/_manifest.json`.

## Acceptance criteria (spec §Part V) — where each lives

| # | Criterion | Where it's exercised |
|---|---|---|
| 1 | `/network` returns ≥ 3 substations × 5–15 meters × 1–2 BESS | `src/edgegrid_forecast/graph/demo_data.py` |
| 2 | Meter click renders forecast in <2 s via real `predict()` | `GET /meter/{msn}/forecast`, `MeterDetail` screen |
| 3 | 48h × 15-min schedule with kVA tracking + audit string per row | `src/edgegrid_forecast/dispatch/optimizer_v2.py`, `DispatchConsole` screen |
| 4 | Commercial Brief IRR heatmap + readable PDF | `src/edgegrid_forecast/commercial/brief.py`, `GET /substation/{id}/brief.html` |
| 5 | All 14 existing `test_v4_predict.py` tests pass; 5 new dispatch tests added | `tests/test_dispatch_mip.py`, `tests/test_audit.py`, `tests/test_commercial.py` |
| 6 | `MODEL_VERSION` in UI footer matches `_manifest.json` | `/model/version` → `<Footer />` |
| 7 | Canonical demo substation (HT + Medium + Small mix) | `CANONICAL_SUBSTATION_ID = "ss-vskp-01"` |
| 8 | README + `run_backend.sh` + `run_frontend.sh` | this file |

## Running the tests

```bash
pytest -q tests/test_audit.py tests/test_dispatch_mip.py tests/test_commercial.py
pytest -q tests/test_v4_predict.py    # 14 existing, still green
```

## Architecture map

```
frontend (React + Vite + Tailwind + D3 + Recharts)
    │
    │  /api/*  (proxied via vite.config.ts)
    ▼
FastAPI (src/edgegrid_forecast/api/main.py + routers.py)
    │
    ├──► graph/        (EdgeGridNetwork, 3 substations + 42 real meters)
    ├──► inference/    (v4_predict.predict()  — untouched source of truth)
    ├──► dispatch/
    │       optimizer.py     (scipy, legacy — existing tests unchanged)
    │       optimizer_v2.py  (PuLP MILP, 192×15-min, confidence-weighted)
    │       audit.py         (natural-language strings)
    └──► commercial/
             irr.py, quote.py, brief.py  (IRR heatmap, FLS quote, PDF brief)
```

## Known v1 limitations

- Solar generation is treated as zero in the MILP (pvlib stub — spec §"Out of scope").
- IEX prices are synthetic (cheap-night / peak-evening sine). Live scraping is out of scope.
- The 30-min → 15-min bridge inside the MILP is documented linear-interpolation +
  half-and-ffill (DEBT-6 in ROADMAP tracks the eventual 15-min retrain).
- Single-operator demo — no auth, no multi-tenant.
- No WebSocket live re-solve (stretch goal).
