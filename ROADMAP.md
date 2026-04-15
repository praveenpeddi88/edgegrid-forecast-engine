# EdgeGrid Forecast Engine — Phased Roadmap

> **Mission:** Predictive dispatch engine for BESS + solar at APEPDCL HT consumer substations
> **Outcome:** Reduce landed cost of electricity for C&I consumers by 12-18% through optimal battery dispatch
> **Owner:** Praveen Peddi | **Repo:** `praveenpeddi88/edgegrid-forecast-engine`
> **Last updated:** 2026-04-15

---

## How to Use This Document

Each phase has a clear **outcome**, **acceptance criteria**, and **task breakdown**. Phases are sequential — each builds on the previous. Within a phase, tasks are ordered by dependency. Check off tasks as they're completed. Every session should start by reading this file and picking up from the current phase.

**Status key:** ✅ Done | 🔧 In Progress | ⬚ Not Started | ⛔ Blocked

---

## Phase 0 — Foundation (COMPLETE ✅)

**Outcome:** A working forecast + dispatch prototype on synthetic data that proves the architecture.

**What we proved:**
- LightGBM demand forecasting works: 2.27% val MAPE, 7.14% CV MAPE on synthetic data
- Chronos-Bolt zero-shot forecasting works: 8.9% avg MAPE with zero training
- Weather data pipeline pulls 131K rows from 4 free APIs with zero nulls
- 112-feature engineering pipeline across 8 families is modular and toggleable
- BESS dispatch optimizer runs 24h simulations with 3 charging strategies
- FastAPI serves 6 endpoints for forecasts + dispatch
- IEX price infrastructure handles CSV import + synthetic fallback

**Architecture decisions locked in:**
- LightGBM primary, Chronos cold-start, Prophet sanity check (AD-1, AD-9)
- Parquet storage, Open-Meteo + NASA POWER for weather, Asia/Kolkata timezone (AD-2, AD-3, AD-7)
- Expanding window CV with min_train_size=4000 (AD-10)

### Tasks (all complete)

- [x] LightGBM demand forecaster with train/predict/cross-validate/save/load
- [x] Prophet secondary model + ensemble with inverse-MAPE weighting
- [x] Chronos-Bolt zero-shot forecaster with autoregressive rollout
- [x] 112-feature pipeline: temporal, lag, rolling, price, consumption, weather, solar, AQ, interactions
- [x] Open-Meteo weather + solar + AQ collectors with chunked pulls
- [x] NASA POWER collector for cross-validation
- [x] IEX price collector: CSV parser + synthetic generator
- [x] Synthetic demand generator for 6 consumer profiles
- [x] BESS dispatch optimizer with solar/grid/IEX arbitrage
- [x] FastAPI with health, consumers, demand/solar/price forecast, dispatch endpoints
- [x] Landed cost calculator with APEPDCL transmission/distribution losses + network charges
- [x] Test suite: 35+ tests across constants, dispatch, solar, price, features, API
- [x] PROGRESS.md tracking document

---

## Phase 1 — Solar Generation Model (NEXT ⬚)

**Outcome:** Accurate hourly solar kWh predictions for each substation, so the dispatch optimizer knows how much free energy is available before deciding when to charge/discharge the battery.

**Why this matters:**
Solar generation is the single biggest input to BESS dispatch economics. If you overestimate solar, the battery charges from grid when it shouldn't. If you underestimate, free solar energy gets curtailed. A 10% error in solar forecast at a 1 MW plant = ~150 kWh/day of suboptimal dispatch = ₹750-1500/day of avoidable cost.

**Acceptance criteria:**
- [ ] Solar model produces hourly kWh forecasts for each of the 3 locations
- [ ] Physics baseline (clear-sky GHI → panel output) matches pvlib within 5%
- [ ] ML correction layer trained on Open-Meteo weather improves over physics-only by >10%
- [ ] Cloud cover, temperature derating, and soiling accounted for
- [ ] 24h-ahead solar forecast MAPE < 15% (industry standard for Indian locations)
- [ ] Output format compatible with dispatch optimizer's `solar_kwh` input

### Task Breakdown

**1.1 — Validate existing solar.py physics layer** ⬚
- `clear_sky_irradiance()` currently uses a simplified solar position model
- Compare against pvlib's `clearsky.ineichen()` for Visakhapatnam coordinates
- Verify GHI magnitude (expect 800-1050 W/m² peak for coastal AP)
- Check: does the current model handle tilt angle and azimuth correctly?
- File: `src/edgegrid_forecast/models/solar.py` lines 50-120

**1.2 — Build panel output converter** ⬚
- Convert GHI (W/m²) → AC power output (kW) for a given panel configuration
- Inputs: GHI, ambient temperature, wind speed, panel specs (capacity, efficiency, temp coefficient)
- Apply temperature derating: efficiency drops ~0.4%/°C above 25°C (crystalline silicon)
- Apply inverter efficiency curve (typically 95-97% at rated load, drops at low/high load)
- Apply soiling factor from air quality features (PM2.5 → soiling_index already in features.py)
- Output: hourly kW per installed kW_peak (capacity factor)
- Reference: PVWatts methodology

**1.3 — Train ML correction model** ⬚
- Features: clear_sky_ghi (physics), cloud_cover, temperature_2m, humidity, wind_speed, aerosol_optical_depth, hour, month, day_of_week
- Target: actual_generation / clear_sky_generation (correction ratio, 0 to 1.2)
- Model: LightGBM regressor (consistent with demand model approach)
- Training data: 8,760 hours × 3 locations from Open-Meteo
- Note: without real generation data, train on synthetic generation = physics × cloud_factor × noise
- Validation: hold out last 2 months, check MAPE

**1.4 — Create hybrid forecast pipeline** ⬚
- Blend: `final_kwh = physics_baseline × ml_correction_ratio`
- The physics layer guarantees physical constraints (no generation at night, seasonal patterns)
- The ML layer learns local cloud/weather patterns that physics alone misses
- Currently solar.py has a 40%/60% blend hardcoded (line 205) — make this learnable
- Add uncertainty bounds: use quantile regression or bootstrap residuals

**1.5 — Integrate with dispatch optimizer** ⬚
- `dispatch/optimizer.py` already accepts `solar_kwh` as a 24-element array
- Create a helper: `forecast_solar_for_dispatch(location_id, date, capacity_kw)` → 24-element array
- This becomes the bridge between forecast and dispatch modules
- Test: run dispatch for 1 week with solar forecast vs zero solar — verify savings increase

**1.6 — Benchmark and document** ⬚
- Metrics: MAE, MAPE, RMSE for each location
- Compare: physics-only vs ML-only vs hybrid
- Compute: capacity factor distribution (should be 15-22% for coastal AP)
- Add results to PROGRESS.md Section 2

---

## Phase 2 — Dispatch Loop Integration ⬚

**Outcome:** A complete simulation loop that takes demand forecast + solar forecast + IEX prices and produces an optimal BESS charge/discharge schedule with ₹ savings quantified per consumer per day.

**Why this matters:**
This is the core product value. Everything before this (forecasting, data pipelines) is infrastructure. This phase connects the infrastructure into the actual decision engine that tells a consumer: "charge your battery at 2am when IEX is ₹3.2/kWh, discharge at 7pm when grid is ₹7.8/kWh, save ₹X today."

**Acceptance criteria:**
- [ ] End-to-end pipeline: raw data → features → forecasts → dispatch → savings report
- [ ] Dispatch runs for all 6 consumers across a full month
- [ ] Three strategies compared: solar_surplus, full_solar, cheap_grid
- [ ] Monthly savings report with breakdown: solar direct use, BESS arbitrage, demand charge reduction
- [ ] Fix known bug: efficiency loss calculation in optimizer.py (sqrt vs linear)
- [ ] Fix known gap: iex_arbitrage_savings currently hardcoded to 0 in economics.py

### Task Breakdown

**2.1 — Fix dispatch optimizer bugs** ⬚
- Bug 1: `optimizer.py` line ~193 applies `np.sqrt(efficiency)` for charge/discharge loss. Should be:
  - Charge: `energy_stored = energy_input × √η` (one-way efficiency)
  - Discharge: `energy_delivered = energy_stored × √η` (one-way efficiency)
  - Round-trip: `delivered = input × η` — verify this is what the current code achieves
  - If `efficiency=0.88`, one-way = `√0.88 = 0.938`. Current code may be correct but needs explicit documentation
- Bug 2: `economics.py` line ~198: `iex_arbitrage_savings_inr` is hardcoded to 0
  - Calculate: `Σ (grid_price - iex_landed_price) × iex_purchase_kwh` for hours where IEX is cheaper
  - This is the core BESS value proposition — it MUST work

**2.2 — Build forecast-to-dispatch bridge** ⬚
- Create `src/edgegrid_forecast/pipeline/run_dispatch.py`
- Input: consumer_id, date_range, bess_config
- Steps:
  1. Load or generate demand forecast (LightGBM or Chronos)
  2. Generate solar forecast (Phase 1 output)
  3. Get IEX prices for period (CSV or synthetic)
  4. Run dispatch optimizer for each day
  5. Compute economics for each day
  6. Aggregate into monthly/annual summary
- Output: DataFrame with daily dispatch schedules + cumulative savings

**2.3 — Run full-month simulation for all consumers** ⬚
- Simulate April 2025 for all 6 consumers
- BESS config: 500 kWh / 4h duration (standard C&I configuration)
- Compare 3 charging strategies side by side
- Expected output per consumer per month:
  - Total consumption (kWh), peak demand (kW)
  - Solar generation and direct use (kWh)
  - BESS cycles, charge/discharge volumes
  - Grid purchase (kWh), IEX purchase (kWh)
  - Total bill without BESS, total bill with BESS, ₹ savings
  - Demand charge savings from peak shaving

**2.4 — Generate savings report** ⬚
- Produce a structured summary: consumer × month × strategy → savings
- Compute: simple payback period, annual savings %, demand charge reduction %
- Identify: which strategy wins for which consumer type (manufacturing vs commercial vs IT)
- This becomes the core data asset for sales conversations

**2.5 — Write integration tests** ⬚
- Test: full pipeline from synthetic data → forecast → dispatch → economics
- Test: edge cases — zero solar day, IEX spike day, weekend vs weekday
- Test: BESS SoC never violates bounds across multi-day simulation
- Test: savings are always non-negative (dispatch should never be worse than no-BESS)

---

## Phase 3 — Real Data Validation ⬚

**Outcome:** Validate the entire pipeline against real APEPDCL meter data for at least one consumer, proving that forecasts and savings estimates are grounded in reality, not just synthetic patterns.

**Why this matters:**
Everything we've built so far runs on synthetic data. Synthetic data has deterministic weather→demand relationships (because we engineered them). Real meter data has noise, missing readings, anomalies, behavioral patterns, and the messy nonlinearity that makes weather features actually useful. Until we validate on real data, the model metrics are interesting but not trustworthy for customer-facing claims.

**Acceptance criteria:**
- [ ] At least 3 months of hourly meter data for at least 1 consumer loaded and cleaned
- [ ] Data quality pipeline handles real-world issues: gaps, frozen readings, outliers
- [ ] Weather features show measurable MAPE improvement on real data (expect 15-25% relative)
- [ ] Backtest: simulate BESS dispatch on historical data, compare to actual billing
- [ ] ₹ savings estimate validated within ±20% of actual billing comparison

### Task Breakdown

**3.1 — Obtain real meter data** ⛔ (Praveen action item)
- Need: hourly kWh readings for any APEPDCL HT consumer
- Minimum: 3 months (2,160 hours) — enough for train + validate
- Ideal: 12 months (8,760 hours) — captures seasonal patterns
- Format: Excel/CSV with timestamp + demand columns
- Source: APEPDCL MDAS portal, consumer billing system, or manual meter reads
- Fallback: even daily readings can be disaggregated using load profile templates

**3.2 — Run data quality pipeline** ⬚
- Apply `data/quality.py` detectors to real data:
  - Frozen readings (consecutive identical values)
  - Z-score outliers (|z| > 3)
  - IQR outliers (beyond 1.5× IQR)
  - Isolation forest anomalies (multivariate)
- Impute missing values: linear interpolation for gaps < 6h, similar-day fill for longer
- Document: % of data imputed, % flagged as anomalous
- Strengthen quality.py with any new patterns found in real data

**3.3 — Retrain models on real data** ⬚
- Train LightGBM with real demand data merged with weather
- Compare: baseline (no weather) vs enriched (with weather)
- Hypothesis: weather features should reduce MAPE by 15-25% on real data
- Compare: Chronos zero-shot vs trained LightGBM — quantify cold-start gap
- Run expanding window CV — report per-fold stability

**3.4 — Backtest dispatch economics** ⬚
- Take real historical demand + actual IEX prices (from CSV export)
- Run dispatch optimizer as if BESS was installed
- Calculate: what would the consumer have saved each month?
- Compare: simulated bill vs actual bill from APEPDCL
- This produces the key claim: "Consumer X would have saved ₹Y lakhs/year with BESS"

**3.5 — Calibrate uncertainty bounds** ⬚
- Current bounds: naive ±15% of point forecast (demand.py line 170)
- Replace with: conformal prediction or quantile regression on real residuals
- Target: 80% prediction interval coverage (P10-P90 should contain 80% of actuals)
- This directly feeds BESS sizing — wider bounds = larger battery needed for reliability

---

## Phase 4 — Production Pipeline ⬚

**Outcome:** A running system that automatically fetches fresh weather forecasts, generates demand/solar/price predictions, and recommends daily BESS dispatch schedules for each consumer.

**Why this matters:**
Phases 1-3 are offline/batch analysis. Phase 4 makes it operational. A BESS operator needs tomorrow's dispatch schedule by 6pm today — that requires automated data collection, inference, and delivery.

**Acceptance criteria:**
- [ ] Daily automated pipeline: weather pull → forecast → dispatch → recommendation
- [ ] Runs on schedule (cron or equivalent) without manual intervention
- [ ] Handles failures gracefully: API timeout → use last good forecast, model error → fallback to Chronos
- [ ] Dispatch recommendation delivered as structured output (JSON or dashboard)
- [ ] Latency: end-to-end pipeline completes in < 5 minutes for all 6 consumers

### Task Breakdown

**4.1 — Build daily pipeline orchestrator** ⬚
- Create `src/edgegrid_forecast/pipeline/daily_run.py`
- Steps:
  1. Pull 7-day weather forecast from Open-Meteo forecast API (already coded, not tested)
  2. Generate demand forecast (48h ahead) for each consumer
  3. Generate solar forecast (48h ahead) for each location
  4. Get IEX prices (today's actual + tomorrow's estimate from pattern)
  5. Run dispatch optimizer for next 24-48 hours
  6. Compute expected savings
  7. Output recommendation as JSON + optional email/Slack alert
- Error handling: each step has try/except with fallback strategy

**4.2 — Implement model versioning** ⬚
- Track: which model version produced which forecast
- Store: model artifact + training metadata + performance metrics alongside predictions
- Enable: A/B testing between model versions
- Use joblib serialization (already in demand.py save/load)

**4.3 — Add forecast monitoring** ⬚
- Log: forecast vs actual comparison (once actuals become available)
- Track: MAPE drift over time — alert if model accuracy degrades
- Implement: simple rolling MAPE tracker stored in a CSV or SQLite
- This is critical for maintaining trust in the system

**4.4 — Expand API for production use** ⬚
- Add endpoints:
  - `GET /dispatch/recommendation/{consumer_id}` — today's recommended schedule
  - `GET /dispatch/history/{consumer_id}` — past dispatch results
  - `GET /metrics/forecast-accuracy/{consumer_id}` — rolling accuracy
  - `POST /data/upload-meter-reading` — accept new meter data
- Add authentication (API key or JWT)
- Add rate limiting
- Deploy: containerize with Docker, deploy to cloud (AWS/GCP/Azure)

**4.5 — Write operational runbook** ⬚
- Document: how to start/stop the pipeline
- Document: how to retrain models when new data arrives
- Document: how to add a new consumer
- Document: how to troubleshoot common failures (API timeout, model load error, bad data)

---

## Phase 5 — Portfolio Optimization & Dashboard ⬚

**Outcome:** Multi-consumer portfolio dispatch that optimizes across all 6 consumers simultaneously (shared solar, coordinated battery dispatch), with a live dashboard for EdgeGrid operations and customer-facing reports.

**Why this matters:**
Individual consumer optimization leaves money on the table. When Consumer A has excess solar at 11am and Consumer B peaks at 11am, coordinated dispatch can shift energy between them (via virtual net metering or shared BESS) for greater total savings. This is EdgeGrid's competitive moat — no single-consumer tool does this.

**Acceptance criteria:**
- [ ] Portfolio optimizer dispatches across all consumers in a cluster
- [ ] Virtual net metering economics computed (prosumer credits)
- [ ] Dashboard shows: real-time forecasts, dispatch schedule, cumulative savings
- [ ] Customer-facing report: monthly savings statement per consumer
- [ ] Cluster-level metrics: total savings, peak demand reduction, solar utilization %

### Task Breakdown

**5.1 — Multi-consumer dispatch optimizer** ⬚
- Extend `dispatch/optimizer.py` to accept multiple consumers
- Shared constraints: single grid connection, shared solar plant, shared BESS
- Optimization: minimize total portfolio cost (not individual)
- `economics.py` already has `compute_network_value()` for prosumer credits — wire it in

**5.2 — Build interactive dashboard** ⬚
- Technology: React + Chart.js or Plotly Dash (TBD based on team skills)
- Views:
  - **Forecast view:** demand/solar/price predictions for next 48h with confidence bands
  - **Dispatch view:** today's charge/discharge schedule with SoC timeline
  - **Savings view:** cumulative ₹ saved, monthly trend, strategy comparison
  - **Portfolio view:** all consumers on one screen, cluster-level metrics
- Data source: FastAPI endpoints from Phase 4

**5.3 — Generate customer-facing reports** ⬚
- Monthly PDF/Excel report per consumer:
  - Consumption summary (kWh, peak kW, load factor)
  - Solar generation and utilization
  - BESS operation (cycles, SoC profile, peak shaving)
  - Savings breakdown (energy, demand charge, IEX arbitrage)
  - Comparison: with BESS vs without BESS
  - CO2 avoided
- Automated generation from dispatch history data

**5.4 — BESS sizing recommendation engine** ⬚
- `optimizer.py` already has `optimize_bess_size()` — but it uses synthetic data
- Feed real dispatch results to refine sizing recommendations
- Output: "For Consumer X, the optimal BESS is Y kWh / Z hours at ₹A capex with B% IRR"
- This is the sales tool: show prospects their specific ROI before signing

---

## Phase 6 — Advanced Models & Scale ⬚

**Outcome:** Best-in-class forecast accuracy through foundation model ensembles, probabilistic forecasting, and 15-minute resolution matching IEX settlement periods.

**Why this matters:**
Phase 1-5 gets us to a working product. Phase 6 makes it best-in-class. The difference between 10% MAPE and 5% MAPE at scale = millions of rupees in more accurate dispatch. 15-minute resolution matters because IEX DAM settles at 15-min blocks — hourly forecasts leave intra-hour arbitrage on the table.

### Task Breakdown

**6.1 — TimesFM 2.5 (Google) integration** ⬚
- Alternative foundation model, different architecture than Chronos
- Ensemble: Chronos + TimesFM for more robust zero-shot forecasts
- Expected: 1-2pp MAPE improvement from model diversity

**6.2 — MOIRAI-2 (Salesforce) integration** ⬚
- Probabilistic foundation model — native uncertainty quantification
- Use for: calibrated prediction intervals (replace naive ±15% bounds)

**6.3 — Conformal prediction framework** ⬚
- Model-agnostic uncertainty quantification
- Guarantee: if you say "80% prediction interval", it actually covers 80% of outcomes
- Applies to: demand, solar, and price forecasts
- Impact: BESS sizing becomes statistically grounded

**6.4 — 15-minute resolution upgrade** ⬚
- IEX settles at 15-min blocks; hourly forecasts miss intra-hour price spikes
- Interpolate weather data from hourly → 15-min (linear or spline)
- Retrain demand model at 15-min granularity
- Modify dispatch optimizer for 96-slot (vs 24-slot) days
- Expected impact: 5-10% better IEX arbitrage capture

**6.5 — Automated IEX price collection** ⬚
- Build Playwright/Selenium scraper for IEX DAM Market Snapshot
- Schedule: daily pull of yesterday's 96 time blocks
- Store: append to growing historical price database
- Fallback: synthetic prices from FY matrix if scraper fails

---

## Dependencies & Critical Path

```
Phase 0 (Foundation) ✅
    │
    ├── Phase 1 (Solar Model) ──────┐
    │                               │
    │                               ▼
    │                    Phase 2 (Dispatch Loop)
    │                               │
    │                               │
    ▼                               ▼
Phase 3 (Real Data) ◄──────── Phase 2 output validates
    │                          against real billing
    │
    ▼
Phase 4 (Production Pipeline)
    │
    ▼
Phase 5 (Portfolio + Dashboard)
    │
    ▼
Phase 6 (Advanced Models)
```

**Critical path:** Phase 1 → Phase 2 → Phase 3 → Phase 4
**Parallel track:** Phase 3 (real data) can start anytime — it's a data acquisition task, not code-dependent
**Phase 6** items can be pulled forward into any phase if time allows (e.g., conformal prediction could go into Phase 2)

---

## Known Bugs & Tech Debt (Fix During Relevant Phase)

| # | Issue | Location | Fix In | Severity |
|---|-------|----------|--------|----------|
| BUG-1 | Efficiency loss as `np.sqrt(η)` — needs documentation or fix | `optimizer.py` ~L193 | Phase 2 | Medium |
| BUG-2 | `iex_arbitrage_savings_inr` hardcoded to 0 | `economics.py` ~L198 | Phase 2 | High |
| BUG-3 | Prophet holidays list uses circular import workaround | `demand.py` L386-390 | Phase 2 | Low |
| BUG-4 | `scipy.optimize` imported but unused in optimizer | `optimizer.py` imports | Phase 2 | Low |
| DEBT-1 | Solar blend ratio hardcoded 40/60 | `solar.py` L205 | Phase 1 | Medium |
| DEBT-2 | API lag features filled with NaN for missing history | `api/main.py` ~L203 | Phase 4 | Medium |
| DEBT-3 | No tests for economics, foundation, loaders, quality | `tests/` | Phase 2-3 | Medium |
| DEBT-4 | Network charges not parametrized per consumer | `constants.py` | Phase 5 | Low |
| DEBT-5 | IRR calculation uses simplified fallback | `optimizer.py` L352 | Phase 2 | Medium |

---

## Metrics That Matter (Track Across Phases)

| Metric | Current (Phase 0) | Target (Phase 3) | Target (Phase 5) | Why |
|--------|-------------------|-------------------|-------------------|-----|
| Demand MAPE (1-step) | 2.27% (synthetic) | <5% (real data) | <3% (real data) | Drives dispatch accuracy |
| Demand MAPE (24h-ahead) | 9.53% (synthetic) | <12% (real data) | <8% (real data) | Production scenario |
| Solar MAPE | Not measured | <15% | <10% | Solar input to dispatch |
| Chronos zero-shot MAPE | 8.9% (synthetic) | <12% (real data) | <10% (real data) | Cold-start fallback |
| Dispatch savings (₹/month/consumer) | Not measured | Measured on real data | Validated against billing | The product KPI |
| P10-P90 coverage | 83.9% (Chronos) | >80% (all models) | >85% (all models) | Trust in uncertainty bounds |
| Pipeline latency | N/A | N/A | <5 min for 6 consumers | Operational requirement |
| Data quality (% clean) | 100% (synthetic) | >95% (real data) | >98% (real data) | Garbage in, garbage out |

---

## Session Workflow

Every session should follow this pattern:

1. **Read** ROADMAP.md — find current phase and next unchecked task
2. **Read** PROGRESS.md — check latest metrics and known issues
3. **Execute** — pick up the next task, write code, run tests
4. **Validate** — verify the task meets acceptance criteria
5. **Update** PROGRESS.md — log what was done, update metrics
6. **Update** ROADMAP.md — check off completed tasks
7. **Commit + push** — every session ends with code in GitHub

---

*This roadmap is a living document. Update it as we learn — the plan should evolve with the product.*
