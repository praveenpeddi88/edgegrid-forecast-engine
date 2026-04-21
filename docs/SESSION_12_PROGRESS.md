# Session 12 — Dispatch Console prototype + Forecast Accuracy showcase

**Date:** 2026-04-17
**Starting point:** Session 11 tip — forecasting engine shipped (C1-C3), 42 per-meter v4 joblib bundles at `models/v4/`, `inference/v4_predict.py` + 14 passing tests.
**Outcome:** Functional full-stack prototype pushed to GitHub as feature branch; Forecast Accuracy showcase rendering live forecast-vs-actual overlay on all 42 real APEPDCL meters.

---

## TL;DR

- **What shipped:** `feat/dispatch-console-prototype` branch on GitHub with 3 commits — inference module bundling + Dispatch Console v1 (C4 + C5) + **Forecast Accuracy showcase v2** (the landing screen).
- **PR:** https://github.com/praveenpeddi88/edgegrid-forecast-engine/pull/new/feat/dispatch-console-prototype
- **Runnable locally:** backend on `:8000`, frontend on `:5173`. See [Resume section](#resume-next-session).
- **Key pivot:** After the initial dispatch-first UI shipped, user feedback ("always give me good frontend experience… the current links look stupid to use") refocused the primary screen from a dispatch demo to a **forecast-accuracy showcase**. v2 replaces the landing.

---

## Timeline / commit chain on the feature branch

| Commit | Title | Files | LOC |
|---|---|---|---|
| `dc2a4ab` | chore: include v4 inference module (Session 11 wrapper) for prototype | `src/edgegrid_forecast/inference/{__init__,_features,v4_predict}.py` + `tests/test_v4_predict.py` | +1,172 |
| `d0644c0` | feat: EdgeGrid Dispatch Console prototype v1 (C4 + C5) | graph/, commercial/, dispatch/{audit,optimizer_v2}.py, api/routers.py, frontend/*, prototypes/, tests | +12,941 |
| `6e12809` | feat(frontend): forecast accuracy showcase — 42 meters, live replay | api/showcase.py, App.tsx rewrite, showcase-api.ts, main.tsx (StrictMode off) | +1,316 / −577 |

Branch diverges from `main` at `5e8b38b` (remote's latest). Local `main` still has 3 unpushed commits from Session 10/11 work — reconcile separately when ready.

---

## Phase 1 — Dispatch Console v1 (plan → build)

### Plan

Approved plan lives at [`~/.claude/plans/witty-drifting-tarjan.md`](../../../../.claude/plans/witty-drifting-tarjan.md).
User-confirmed scope: React + Vite + Tailwind + shadcn/ui + D3 + Recharts · full spec acceptance criteria · PuLP MILP upgrade.

### Backend (Python, ~2.1k LOC)

| Module | Path | Purpose |
|---|---|---|
| Graph abstraction | [`src/edgegrid_forecast/graph/`](../src/edgegrid_forecast/graph) — `nodes.py`, `edges.py`, `network.py`, `demo_data.py` | Typed Node/Edge dataclasses; deterministic demo synthesis from the real manifest → 3 substations × 14 meters × 1-2 BESS; canonical `ss-vskp-01` (Madhurawada 33/11kV) has HT+Medium+Small tier mix |
| MILP optimizer v2 | [`src/edgegrid_forecast/dispatch/optimizer_v2.py`](../src/edgegrid_forecast/dispatch/optimizer_v2.py) | PuLP MILP, 48h × 15-min (192 intervals), SOC ∈ [10%, 90%], charge/discharge mutex, round-trip η², kVA peak tracking, **confidence-weighted aggression** scaled by `historical_block_mape` from `predict()`. Legacy scipy `optimizer.py` preserved for `tests/test_dispatch.py` compat |
| Audit strings | [`src/edgegrid_forecast/dispatch/audit.py`](../src/edgegrid_forecast/dispatch/audit.py) | Spec-format `[HH:MM] action N kWh because reason (confidence P%)` — regex-validated; every dispatch row carries one |
| Commercial | [`src/edgegrid_forecast/commercial/`](../src/edgegrid_forecast/commercial) — `irr.py`, `quote.py`, `brief.py` | IRR + NPV + payback + sensitivity grid (±20% IEX, ±10% demand, ±15% CAPEX); FLSQuoteGenerator turning peak-block MAPE into firmness %; printable HTML brief in EdgeGrid teal + Poppins |
| API routers (dispatch) | [`src/edgegrid_forecast/api/routers.py`](../src/edgegrid_forecast/api/routers.py) | 8 graph-centric endpoints mounted additively on existing FastAPI app: `/network`, `/meter/{msn}/forecast`, `/substation/{id}/{dispatch,commercial,fls-quote,brief.html}`, `/portfolio`, `/model/version` |

### Frontend (React + Vite + TS + Tailwind + D3 + Recharts)

Initial v1 at [`frontend/`](../frontend) with 5 screens (Network Home, Meter Detail, Dispatch Console, Commercial Brief, Portfolio) — all rewritten in v2 (see [Phase 3](#phase-3--forecast-accuracy-showcase-v2)).

### Tests (24 new, pandas 3.x compatible)

| File | Cases | What it proves |
|---|---|---|
| [`tests/test_audit.py`](../tests/test_audit.py) | 10 | Format regex, all 5 action branches, confidence mapping, negative-kwh rejection, action classifier |
| [`tests/test_dispatch_mip.py`](../tests/test_dispatch_mip.py) | 5 | Determinism, SOC + mutex constraints, confidence weighting reduces aggression, audit coverage, schema |
| [`tests/test_commercial.py`](../tests/test_commercial.py) | 9 | IRR sanity, heatmap shape, monotonicity, sensitivity ranges, FLS quote firmness floor/cap/pricing |

```bash
uv run pytest tests/test_audit.py tests/test_commercial.py tests/test_dispatch_mip.py -q
# 24 passed
```

### Glue

- [`prototypes/dispatch_console/README.md`](../prototypes/dispatch_console/README.md) — 90-second demo path + acceptance criteria map.
- [`prototypes/dispatch_console/run_backend.sh`](../prototypes/dispatch_console/run_backend.sh)
- [`prototypes/dispatch_console/run_frontend.sh`](../prototypes/dispatch_console/run_frontend.sh)
- [`prototypes/dispatch_console/demo_data.json`](../prototypes/dispatch_console/demo_data.json) — offline graph snapshot.

---

## Phase 2 — Environment + GitHub setup

### Install chain (fresh laptop state, no dev tools)

| Blocker found | Resolution |
|---|---|
| Python 3.9.6 only (need ≥3.10); no brew/pyenv/conda/uv | Installed `uv` via one-shot `curl -LsSf https://astral.sh/uv/install.sh \| sh` → `~/.local/bin/uv` |
| Python + 40+ deps needed | `uv sync --python 3.12 --extra dev` — pulls lightgbm, pulp, pandas 3.x, prophet, pvlib, pyarrow, pytest |
| LightGBM requires `libomp.dylib` (not on Apple Silicon by default, no brew) | Downloaded `llvm-openmp` from conda-forge (`conda.anaconda.org/conda-forge/osx-arm64/llvm-openmp-22.1.3-hc7d1edf_0.conda`), extracted the `.tar.zst` inside the `.conda` zip with zstandard, copied `libomp.dylib` to `~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib/` |
| Parquets vanished from working tree after `git stash --include-untracked` | Recovered from `stash@{0}^3` (untracked-parent commit) — `sp_data.parquet`, `tp_data.parquet`, `meter_profile.parquet`, `visakhapatnam_expanded_*.parquet` |
| pandas 3.x removed `fillna(method='bfill')` + refuses interpolate on str-column DataFrame | Patched `optimizer_v2.py`: split numeric/non-numeric columns in the 30→15-min resampler; use `.bfill()` directly |
| Demo substation sizing blew past 5-15 meter spec ceiling | Capped at 14 meters/substation in `graph/demo_data.py`; 42 meters fit cleanly across 3 subs |

### GitHub auth

| Blocker | Resolution |
|---|---|
| No `gh` CLI, no SSH key, osxkeychain empty for github.com; `credential.https://github.com.helper` pointed to a stale `/tmp/gh_.../gh` binary that no longer exists | Removed the broken per-host helper from local `.git/config`; stored user-provided PAT in osxkeychain via `git credential-osxkeychain store` |
| Push to main rejected (divergence — local has Session 10/11 commits; remote has Strategy 1 benchmark + docs update) with PROGRESS.md/ROADMAP.md conflicts in rebase | Chose path **C** (feature branch + PR): reset feature branch to `origin/main`, cherry-picked just the inference module + prototype commit, pushed |

**Branch on GitHub:** `feat/dispatch-console-prototype`.
**PR URL:** https://github.com/praveenpeddi88/edgegrid-forecast-engine/pull/new/feat/dispatch-console-prototype.

---

## Phase 3 — Forecast Accuracy showcase (v2)

### Why v2

User feedback on v1: *"always give me good frontend experience. The current links look stupid to use. How do you prioritise what to show in the frontend?"*

Saved as [`~/.claude/projects/.../feedback_frontend_quality.md`](../../../../.claude/projects/-Users-praveenkumarpeddi-Documents-Claude-Projects-Life-at-Edgegrid/memory/feedback_frontend_quality.md) — durable rule: land on an outcome not a canvas; labels are verbs about outcomes, not mechanisms; hide raw IDs; one aha per screen.

User-stated goal for v2: *"we just want to see the forecasting engine forecasting accurately live based on the training"* → refocus from dispatch demo to **forecast-accuracy proof**.

### Backend additions

[`src/edgegrid_forecast/api/showcase.py`](../src/edgegrid_forecast/api/showcase.py) — 5 new endpoints:

| Endpoint | What it returns |
|---|---|
| `GET /fleet/summary` | Headline accuracy numbers from the v4 manifest: 9.58% mean · 7.15% median · 1.61% HT peak · health counts (25 green / 6 amber / 11 red) |
| `GET /meters` | List of 42 meters with tier + block MAPE, sorted HT-first then by accuracy |
| `GET /fleet/actuals-range` | Latest available actual timestamp (2026-02-12) — drives scrubber default + bounds |
| `POST /fleet/replay` | Batch forecast-vs-actual for N meters at a given `as_of`; runs `predict()` in an 8-worker pool + joins with raw parquet actuals on 30-min timestamp (one round-trip, not 42). ~11s for all 42 meters on first call |
| `GET /meter/{msn}/history` | 48-hour overlay + block-level MAPE comparison (training vs this replay window) |

**Critical fix** (`as_of` timezone): browser `.toISOString()` ships UTC with `Z` suffix; raw MDMS parquets are tz-naive local wall clock. Without normalization the join silently misses every actual. Backend now strips tzinfo on `as_of` — both for `/fleet/replay` and `/meter/{msn}/history`.

### Frontend rewrite

[`frontend/src/App.tsx`](../frontend/src/App.tsx) — full rewrite + [`frontend/src/showcase-api.ts`](../frontend/src/showcase-api.ts) typed client.

**New landing screen — "Forecasting accuracy, live":**
- Hero proof strip: **mean error this window** (live replay MAPE, not training) + training median + HT peak-hour accuracy + fleet health breakdown
- Time scrubber with 7/14/30-day quick picks, bounded by actual availability
- Tier filter chips (All 42 / HT 5 / Medium 27 / Small 10)
- 42-tile responsive grid — health-colored border + dot per tile, forecast/actual dual-line sparkline, live MAPE

**Meter detail redesigned:** full 48h ComposedChart with confidence band + forecast (teal) + actual (white dashed) lines, tooltip aligned on all three. Plus Trained-vs-This-window bar chart by time of day.

**Navigation:** Accuracy (primary) · Network (placeholder) · Portfolio (kept). Dispatch/Commercial demoted out of the primary flow.

### Stability fixes

1. **StrictMode race** in the batch replay effect. First mount fires POST #1 (10s); unmount runs cleanup; remount fires POST #2; browser aborts both with `ERR_NETWORK_IO_SUSPENDED`. Fix: generation-counter (`replayGenRef`) so only the latest response can setState, + `useMemo` on `msnsKey` to stabilize the effect dep, + disabled StrictMode in `main.tsx` (dev-only, documented inline) since the 10s batch makes the double-invocation pattern UX-unfriendly.
2. **Timezone mismatch** (described above).

### Visual proof of working

End-to-end verified via `preview_screenshot` + fiber introspection:
- Hero reads `11.18% · this window` with `42 meters live` (live fleet MAPE on real replay window, not training number).
- 42 tiles render with forecast (solid line) + actual (dashed white) overlaid, color-coded by health.
- Per-tile MAPEs: 2.1%, 2.2%, 3.3%, 3.8%, 4.1%, 4.6%, 5.6%, 6.3%, 6.9%, 7.5% (sample from scroll view).
- MeterDetail: `HT meter · forecast vs actual`, `Training 5.54%` vs `This window 4.92%`, tooltip aligned.

---

## Artifacts index

### In the repo (committed)

```
edgegrid-forecast-engine/
├── docs/
│   ├── EDGEGRID_PRODUCT_SPEC.md              (input spec)
│   └── SESSION_12_PROGRESS.md                (this file)
├── src/edgegrid_forecast/
│   ├── inference/                            (Session 11, bundled in dc2a4ab)
│   │   ├── v4_predict.py
│   │   └── _features.py
│   ├── graph/                                (v1)
│   │   ├── nodes.py, edges.py, network.py, demo_data.py
│   ├── dispatch/
│   │   ├── optimizer_v2.py                   (v1 — PuLP MILP)
│   │   └── audit.py                          (v1)
│   ├── commercial/                           (v1)
│   │   ├── irr.py, quote.py, brief.py
│   └── api/
│       ├── main.py                           (2 router mounts added)
│       ├── routers.py                        (v1 dispatch router)
│       └── showcase.py                       (v2 NEW — forecast accuracy)
├── tests/
│   ├── test_audit.py, test_commercial.py, test_dispatch_mip.py   (v1)
│   └── test_v4_predict.py                    (Session 11)
├── frontend/                                 (v2 rewrite)
│   ├── src/App.tsx                           (Forecast Showcase landing)
│   ├── src/showcase-api.ts                   (typed client for showcase endpoints)
│   ├── src/main.tsx                          (StrictMode off)
│   ├── src/api.ts, src/index.css
│   └── package.json, tsconfig.json, tailwind.config.js, vite.config.ts, postcss.config.js
└── prototypes/dispatch_console/
    ├── README.md, run_backend.sh, run_frontend.sh, demo_data.json
```

### Outside the repo (user environment)

| Path | What it is |
|---|---|
| `~/.claude/plans/witty-drifting-tarjan.md` | Approved implementation plan (signed off before building) |
| `~/.claude/projects/.../memory/project_dispatch_console.md` | Project memory for future session continuity |
| `~/.claude/projects/.../memory/feedback_frontend_quality.md` | Durable UX rule (no raw IDs, outcome-labels, etc.) |
| `~/.claude/projects/.../memory/MEMORY.md` | Index — updated with both of the above |
| `~/.local/bin/uv`, `uvx` | uv 0.11.7 installed this session |
| `~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib/libomp.dylib` | libomp manually placed so LightGBM loads |
| `macOS Keychain` — internet password for `github.com` / `praveenpeddi88` | GitHub PAT stored for future pushes |

### On GitHub

- **Repo:** https://github.com/praveenpeddi88/edgegrid-forecast-engine
- **Feature branch tip:** [`6e12809`](https://github.com/praveenpeddi88/edgegrid-forecast-engine/commit/6e12809) on `feat/dispatch-console-prototype`
- **Open the PR:** https://github.com/praveenpeddi88/edgegrid-forecast-engine/pull/new/feat/dispatch-console-prototype

---

## Headline numbers reproduced in the UI

| Metric | Value | Source |
|---|---|---|
| Meters live | 42 | `models/v4/_manifest.json` |
| Mean MAPE (training holdout) | 9.58% | manifest |
| Median MAPE | 7.15% | manifest |
| HT peak-hour median MAPE | 1.61% | manifest — aha #2 source |
| Fleet MAPE on 48h replay window (2026-02-08 → 2026-02-10) | 11.18% | live `/fleet/replay` |
| Green meters (<8% error) | 25 / 42 | derived |
| Amber (8–12%) | 6 / 42 | derived |
| Red (>12%, retrain) | 11 / 42 | derived |

---

## Known limitations (v2)

1. **First load is ~10s** — 42-meter batch `predict()` runs cold. InitialLoading state sets expectations; follow-up would be a `(msns, as_of)` response cache at the backend.
2. **StrictMode disabled in dev** (documented inline in `main.tsx`). The batch POST's expense makes the double-invocation pattern a UX problem; production builds unaffected.
3. **Network view is a placeholder** (demoted from v1). The D3 graph still exists in git history at commit `d0644c0`; can be re-introduced as a secondary "see by substation" lens in v3.
4. **Dispatch Console + Commercial Brief screens from v1 were removed from the route table** when v2's App.tsx rewrote the router — their routers (`api/routers.py`) are still mounted and callable, but no UI hits them. Either restore as secondary nav or delete in a v3 cleanup pass.
5. **Local `main` has 3 unpushed commits** (Session 10, Session 11 inference wrapper, and the original Dispatch Console v1 commit `2ce08b8`). These don't need pushing now — the feature branch has what we want. But reconcile before doing any future work on `main`.
6. **No `uv.lock` test** — `pytest -q` requires `uv sync --extra dev` to be run first; without it pytest is missing.

---

## Resume next session

### Current running services (at hand-off time)

| Task ID | What it is | Port |
|---|---|---|
| `bvwzoryus` | uvicorn (backend) | 8000 |
| `1fd82e9a-731c-4045-a5a4-b2a6996dd5e1` | Vite preview (frontend) | 5173 |

Both may no longer be running by the time you read this — restart with the commands below.

### Quick start

```bash
REPO="/Users/praveenkumarpeddi/Library/Application Support/Claude/local-agent-mode-sessions/f5ba8df3-f08b-4818-93e9-93beb3c3f370/ae9e93f2-4016-4710-992c-0df6fd3ffb08/local_b8d896bb-963a-4954-9bdb-fb4688dd7d7f/outputs/edgegrid-forecast-engine"
cd "$REPO"

# First time only (uv already installed this session):
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra dev

# Terminal 1 — backend
uv run uvicorn edgegrid_forecast.api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — frontend
cd frontend && npm run dev -- --port 5173 --strictPort
# Open http://localhost:5173
```

### Smoke checks

```bash
# Backend alive, correct model version
curl -s http://localhost:8000/model/version
# → {"model_version":"v4.s1.0", ...}

# Fleet accuracy numbers
curl -s http://localhost:8000/fleet/summary
# → {"n_meters":42, "mean_mape_pct":9.58, "median_mape_pct":7.15, ...}

# Live replay for one meter
curl -s -X POST http://localhost:8000/fleet/replay \
  -H "Content-Type: application/json" \
  -d '{"msns":["67003694"], "horizon":48, "include_actuals":true}' \
  | python3 -c "import json,sys;d=json.load(sys.stdin);m=d['meters'][0];print(f'status={m[\"status\"]} mape={m[\"mape\"]:.2f}%' if m['mape'] else 'no mape')"
# → status=ok mape=3.70%
```

### Candidate v3 work

Prioritized if we continue:
1. **Cache `/fleet/replay` responses** keyed by `(msns, as_of, model_version)` — would make the landing < 500ms on reload.
2. **Rolling MAPE trendline** — daily fleet accuracy over the last 30 days, added to the proof strip or a secondary "Accuracy over time" chart.
3. **Retraining queue** — red-health meters get a "Queue for retraining" button that writes to a simple jobs file; M2 v5 solar-window fix as a natural follow-up.
4. **Network view rebuild** — re-introduce D3 graph as a secondary drilldown (`/substation/:id`) that lives under the Accuracy landing rather than replacing it.
5. **15-min native forecast** (DEBT-6) — resampling bridge in `optimizer_v2.py` goes away when the model retrains at 15-min cadence.
6. **Dispatch Console UI wiring** — the API routers still exist but v2's App.tsx doesn't consume them. Either resurrect those screens or remove the dead routes.

### Reconciling local main

```bash
cd "$REPO"
git log --oneline origin/main..main
# 2ce08b8 feat: Dispatch Console prototype v1 (local-only — superseded by feature branch)
# 0e2f8ac feat: Production inference wrapper (already in feature branch as dc2a4ab)
# 89909a3 feat(S1 v4): ... (Session 10 — has PROGRESS.md/ROADMAP.md conflicts with remote)
```

Options when you get to this: (a) rebase with manual conflict resolution (pick longer hunk on each side — session notes are additive), (b) reset local main to origin/main and accept those commits are lost to the feature branch instead, (c) cherry-pick only `89909a3`'s benchmark files (not the PROGRESS/ROADMAP edits) onto main.

---

*End of Session 12 progress. Next session: resume from [Candidate v3 work](#candidate-v3-work) or open the PR for merge review.*
