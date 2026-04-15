# EdgeGrid Forecast Engine — Module Roadmap

> **Mission:** Predictive dispatch engine for BESS + solar at APEPDCL HT consumer substations
> **Outcome:** Reduce landed cost of electricity for C&I consumers by 12-18% through optimal battery dispatch
> **Owner:** Praveen Peddi | **Repo:** `praveenpeddi88/edgegrid-forecast-engine`
> **Architecture reference:** EIL PRD v1 (M1-M6 module architecture)
> **Last updated:** 2026-04-15

---

## How to Use This Document

This roadmap mirrors the EIL PRD's modular architecture (M1–M6). Modules M1–M3 are shared infrastructure; M4–M6 are asset-specific execution layers. Within each module, features are tagged M{x}-F{y} matching the PRD. Every session should start by reading this file and PROGRESS.md, then pick up the next unchecked task.

**Status key:** ✅ Done | 🔧 In Progress | ⬚ Not Started | ⛔ Blocked

---

## Module Map & Dependencies

```
M1 Data Quality Engine ─────────┐
  (clean, validate, correct)     │
                                 ▼
M2 Forecasting Engine ──────► M3 Optimization Engine (MPC)
  (demand, solar, price)         (dispatch, sizing)
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              M4 BESS        M5 DR        M6 Explain-
              Execution      Engine       ability
              (charge/       (baseline,   (SHAP, audit,
               discharge)     curtail)     reports)
```

**Critical path:** M1 → M2 → M3 → M4
**Phase 1 (EIL PRD, 10 weeks):** M1 complete + M2 demand + M3 greedy dispatch + M6 basic explainability
**Defensibility order:** M1 (voltage SOC = 6-12mo moat) → M2 (load forecast = 12-18mo moat)

---

## Foundation (Phase 0) — COMPLETE ✅

Everything built before the PRD alignment. This work maps across M2, M3, and M4 in the module architecture.

**What we proved:**
- LightGBM demand forecasting: 2.27% val MAPE, 7.14% CV MAPE on synthetic data
- Chronos-Bolt zero-shot: 8.9% avg MAPE with zero training
- Weather pipeline: 131K rows from 4 APIs, zero nulls
- 112-feature engineering pipeline: 8 families, modular, toggleable
- BESS dispatch optimizer: 24h simulation, 3 charging strategies
- FastAPI: 6 endpoints for forecasts + dispatch
- IEX price infrastructure: CSV import + synthetic fallback
- Landed cost calculator with APEPDCL losses + network charges

**Architecture decisions locked:** AD-1 through AD-12 (see PROGRESS.md)

---

## M1 — Data Quality Engine 🔧

**PRD reference:** Module M1, Features M1-F1 through M1-F6
**Why first:** Every downstream model is only as good as the data feeding it. Indian grid data has unique noise patterns (voltage swings, DG transitions, CT artefacts, AMI packet loss) that generic quality pipelines miss entirely. Building these India-specific detectors creates a 6-12 month replication barrier.

**Phase 1 acceptance criteria (from PRD):**
- [ ] AMI ingestion handles NULL, duplicates, late arrivals for 15-min resolution
- [ ] Voltage-compensated SOC within ±2% of lab-calibrated reference
- [ ] Demand noise filter removes CT artefacts correlated with frequency deviation
- [ ] DG transition detection flags grid→DG switchover events with <1 interval latency
- [ ] APFC events excluded from DR baselines automatically
- [ ] Data quality score computed per interval, per signal, per consumer

### M1-F1: Smart Meter AMI Ingestion ⬚

**Problem:** Indian AMI networks have 15-40% packet loss. Meters send 15-min interval data that arrives with gaps, duplicates, and late packets. Multi-channel meters report kW, kVAR, kVA, voltage, current, PF — all must be synchronized.

**Detection logic:**
- NULL check: any channel missing for an interval → flag
- Duplicate detection: same timestamp + same meter → keep latest by arrival time
- Late arrival: packet timestamp > 2 intervals behind wall clock → flag but accept
- Multi-channel sync: all channels present for an interval OR entire interval flagged
- Gap detection: missing intervals in 15-min sequence → interpolate if <4 intervals, flag if ≥4

**Implementation:**
- [ ] `ingest_ami_packet(meter_id, timestamp, channels: dict)` — single packet handler
- [ ] `detect_gaps(series, freq="15min")` — find missing intervals in a 15-min sequence
- [ ] `handle_duplicates(df, timestamp_col, meter_col)` — dedup keeping latest arrival
- [ ] `handle_late_arrivals(df, max_delay_intervals=8)` — flag late but accept within window
- [ ] `sync_channels(df, required_channels=["kw","kvar","kva","voltage","current","pf"])` — ensure multi-channel alignment
- [ ] `compute_interval_quality_score(row)` — 0-1 score: 1.0 = all channels present + on-time, degraded for late/imputed/partial

**Location:** `src/edgegrid_forecast/data/quality.py` (extend existing)

### M1-F2: Statistical Anomaly Detection ✅ (basic)

**Current state:** `quality.py` has z-score, IQR, and Isolation Forest detectors + frozen reading detection. These are generic and work.

**Gaps to fill:**
- [ ] Add contextual anomaly detection: a value normal at 2pm might be anomalous at 2am
- [ ] Time-of-day z-score: compute z-score within same hour-of-day band
- [ ] Rolling baseline z-score: z-score against 48h rolling window (not global)

### M1-F3: Voltage Compensation / SOC Correction ⬚

**Problem:** Indian grid voltage deviates ±10-15% from nominal (typical range: 380-440V on 415V nominal). BMS calculates SOC from terminal voltage using a curve calibrated at nominal voltage. When grid voltage is 430V, BMS over-reports SOC by 3-8%. When 390V, it under-reports. This makes dispatch decisions wrong — the optimizer thinks the battery has more/less energy than it actually does.

**Detection & correction logic:**
- Collect (voltage, BMS_SOC, actual_SOC) triplets during known-state periods (full charge, full discharge, rest periods)
- Build site-specific voltage→SOC_error regression (linear initially, polynomial if non-linear)
- Apply: `corrected_SOC = BMS_SOC - voltage_soc_error(current_voltage)`
- Calibration period: 30-60 days of operational data per site
- Re-calibrate trigger: if correction residual exceeds 2% for 7 consecutive days

**Implementation:**
- [ ] `VoltageSOCCorrector` class with `calibrate(voltage, bms_soc, actual_soc)` and `correct(voltage, bms_soc)`
- [ ] `detect_known_state_periods(soc_series, current_series)` — find full-charge (SOC>98%, I≈0), full-discharge (SOC<5%, I≈0), rest (I≈0 for >30min) periods
- [ ] `build_correction_model(voltage, soc_error)` — linear/polynomial regression
- [ ] `check_calibration_drift(recent_residuals, threshold=0.02)` — trigger re-calibration
- [ ] `apply_correction(voltage_series, bms_soc_series)` → corrected_soc_series

**Location:** `src/edgegrid_forecast/data/quality.py` (new class)

### M1-F4: Demand Signal Noise Filter ⬚

**Problem:** CT (current transformer) metering at Indian substations produces artefacts from 2-4 Hz grid frequency swings. These appear as sudden spikes in kVA readings that don't reflect real load changes. Additionally, high-impedance faults and capacitor switching on the 11kV feeder create transient distortions.

**Detection logic (from PRD):**
1. Rolling median filter on 15-min kVA readings (window = 5 intervals = 75 min)
2. Compute deviation: `|kVA_actual - kVA_rolling_median| / kVA_rolling_median`
3. Flag if deviation > 3σ from 48h rolling baseline AND frequency outside 49.5-50.5 Hz band
4. Secondary check: if flagged interval's kW is stable (deviation < 1σ) but kVA spikes → likely PF artefact, not real load

**Implementation:**
- [ ] `DemandNoiseFilter` class with configurable thresholds
- [ ] `compute_rolling_baseline(kva_series, window="48h")` → rolling median + rolling σ
- [ ] `detect_ct_artefacts(kva, kw, frequency, sigma_threshold=3.0)` → boolean mask
- [ ] `detect_pf_artefacts(kva, kw, pf)` → kVA spike with stable kW = PF transient
- [ ] `clean_demand_signal(kva, kw, frequency, pf)` → cleaned kVA with artefacts replaced by rolling median

**Location:** `src/edgegrid_forecast/data/quality.py` (new class)

### M1-F5: DG Transition Detection ⬚

**Problem:** Many C&I consumers have diesel generators (DG) as backup. When grid power fails, the site switches to DG — grid import drops to near-zero but site load continues. DG periods must be detected and excluded from load forecasting baselines because they represent supply-side events, not demand-side behavior. If left in training data, the model learns that "demand drops to zero sometimes" and produces biased forecasts.

**Detection logic (from PRD):**
1. Primary signal: grid import drops to <5% of rolling baseline within 1 interval (15 min)
2. Confirmation: site load (if available from separate meter) remains >50% of baseline
3. Secondary signal: voltage signature change — DG voltage is typically more variable (±5%) and at slightly different frequency
4. Transition detection: flag the interval where grid→DG or DG→grid switchover happens
5. Duration: all intervals from grid→DG to DG→grid are marked as DG period
6. Use: exclude DG periods from demand training data, DR baseline computation

**Implementation:**
- [ ] `DGTransitionDetector` class
- [ ] `detect_grid_to_dg(grid_import, rolling_baseline, threshold_pct=5.0)` → boolean mask of DG-on periods
- [ ] `detect_dg_to_grid(grid_import, rolling_baseline)` → transition-back points
- [ ] `detect_voltage_signature(voltage_series, frequency_series)` → secondary DG confirmation
- [ ] `mark_dg_periods(grid_import, voltage, frequency)` → full DG period mask with transition labels
- [ ] `exclude_dg_from_training(demand_df, dg_mask)` → filtered DataFrame safe for model training

**Location:** `src/edgegrid_forecast/data/quality.py` (new class)

### M1-F6: APFC Switching Event Detection ⬚

**Problem:** Automatic Power Factor Correction (APFC) panels switch capacitor banks on/off to maintain PF near 0.95-1.0. Each switching event causes a step change in kVAR (and therefore kVA) without changing real power (kW). If APFC events are included in DR baselines, they can be mistaken for demand curtailment — the kVA drops sharply when capacitors switch in, making it look like the consumer reduced load.

**Detection logic (from PRD):**
1. Step change: kVA drops/rises >50 kVA (for HT consumers) within 1 interval
2. Stable kW: kW change in same interval is <1σ of rolling baseline
3. PF jump: power factor jumps toward 0.95-1.0 (cap switch-in) or drops away (switch-out)
4. Coincidence: all three conditions must occur simultaneously
5. Characteristic: APFC events are discrete (step function), not gradual

**Implementation:**
- [ ] `APFCSwitchingDetector` class
- [ ] `detect_kva_step(kva_series, threshold_kva=50)` → intervals with sudden kVA change
- [ ] `detect_stable_kw(kw_series, kva_step_mask, sigma_threshold=1.0)` → confirm kW is stable during kVA step
- [ ] `detect_pf_jump(pf_series, kva_step_mask, target_pf_range=(0.95, 1.0))` → PF moved toward target
- [ ] `classify_apfc_events(kva, kw, pf)` → labeled events: cap_switch_in, cap_switch_out
- [ ] `normalize_for_dr_baseline(kva, kw, apfc_events)` → kVA adjusted to remove APFC effects

**Location:** `src/edgegrid_forecast/data/quality.py` (new class)

### M1 Integration: Enhanced Quality Pipeline ⬚

- [ ] Upgrade `run_quality_pipeline()` to orchestrate all M1 detectors
- [ ] Add per-interval quality score: `quality_score = f(completeness, timeliness, anomaly_flags)`
- [ ] Add per-consumer quality report: % intervals with each flag type
- [ ] Support 15-min resolution (current pipeline assumes hourly)
- [ ] Output: cleaned DataFrame with all detection columns + summary quality metrics

---

## M2 — Forecasting Engine 🔧 (partially complete)

**PRD reference:** Module M2, Features M2-F1 through M2-F4
**Status:** ~70% complete from Phase 0. Demand forecasting works well. Solar and price forecasting infrastructure exists but needs validation against real data.

**Phase 1 acceptance criteria (from PRD):**
- [ ] Demand MAPE <12% on real consumer data (24h ahead)
- [ ] Solar forecast available for dispatch (physics + ML hybrid)
- [ ] IEX price pattern forecast for day-ahead dispatch
- [ ] 15-minute resolution for all forecasts (currently hourly)

### M2-F1: Demand Forecasting ✅ (synthetic) / ⬚ (real data)

**Complete:**
- [x] LightGBM with 112 features, 8 families → 2.27% val MAPE
- [x] Chronos-Bolt zero-shot → 8.9% avg MAPE (cold-start fallback)
- [x] Prophet + ensemble with inverse-MAPE weighting
- [x] Expanding window CV with min_train_size=4000 → 7.14% MAPE

**Remaining:**
- [ ] Validate on real APEPDCL meter data (⛔ blocked on data acquisition)
- [ ] Retrain at 15-min resolution (96 intervals/day vs 24)
- [ ] Add month-to-date peak tracking feature for demand charge optimization
- [ ] Conformal prediction for calibrated uncertainty bounds

### M2-F2: Solar Generation Forecasting ⬚

- [ ] Validate physics layer (solar.py) against pvlib for Visakhapatnam
- [ ] Build panel output converter: GHI → AC kW with temperature derating + soiling
- [ ] Train ML correction layer on weather features
- [ ] Create hybrid pipeline: physics × ml_correction_ratio
- [ ] Integrate with dispatch optimizer's solar_kwh input
- [ ] Target: <15% MAPE (24h ahead, industry standard for Indian locations)

### M2-F3: IEX Price Forecasting ⬚

- [ ] Pattern-based day-ahead: weekday/weekend × month × hour from FY24-25 matrix
- [ ] ML enhancement: weather → price correlation (hot day → high demand → high price)
- [ ] Build 15-min block price forecast (IEX settles at 15-min, not hourly)
- [ ] Automated IEX DAM price collection (scraper or API when available)

### M2-F4: Ensemble & Model Selection ⬚

- [ ] Automated model selection per consumer: LightGBM vs Chronos vs Prophet
- [ ] Ensemble weighting: inverse-MAPE or stacking
- [ ] TimesFM 2.5 / MOIRAI-2 integration for model diversity
- [ ] Forecast monitoring: track MAPE drift, alert on degradation

---

## M3 — Optimization Engine ⬚

**PRD reference:** Module M3, Features M3-F1 through M3-F4
**Status:** ~40% complete. Current greedy dispatch works but PRD specifies MPC with multi-objective optimization. Major upgrade needed.

**Phase 1 acceptance criteria (from PRD):**
- [ ] Dispatch produces 15-min charge/discharge schedule for next 24h
- [ ] Demand charge saving estimate within ±5% of actual billing
- [ ] Month-to-date peak tracking prevents unnecessary battery cycles
- [ ] Objective: maximize [arbitrage + demand charge savings + DISCOM peak shaving] − [degradation cost + SOC buffer penalty]

### M3-F1: MPC Controller ⬚ (major upgrade from greedy)

- [ ] 48h horizon, 15-min timestep, re-solve every 30 min
- [ ] Multi-objective: arbitrage + demand charge savings + peak shaving − degradation − SOC penalty
- [ ] Constraints: SOC bounds (10-90%), max C-rate, grid import limits
- [ ] kVA-based demand charges (not kW — Indian billing uses apparent power)
- [ ] Month-to-date peak tracking: only dispatch when projected kVA will exceed billing period max
- [ ] Replace current scipy/greedy approach with PuLP or CVXPY formulation

### M3-F2: BESS Degradation Model ⬚

- [ ] Cycle-based: degradation = f(DOD, C-rate, temperature, cycle_count)
- [ ] Calendar aging: capacity loss per year at rest
- [ ] Economic dispatch must include degradation cost per cycle
- [ ] Rainflow cycle counting for irregular dispatch patterns

### M3-F3: BESS Sizing Engine ⬚ (upgrade from current)

**Current:** `optimize_bess_size()` exists but uses synthetic data and simplified economics
- [ ] Parametric sweep: capacity (0.5-20 MWh) × duration (2/4/6h) × strategy
- [ ] Economics: NPV, IRR, payback with real CAPEX curves (₹150L/MWh declining annually)
- [ ] Sensitivity: to IEX price volatility, demand growth, tariff changes
- [ ] Output per configuration: annual savings, IRR, payback, optimal strategy

### M3-F4: Real-Time Recalculation ⬚

- [ ] Re-optimize when actuals deviate >10% from forecast
- [ ] Handle contingencies: sudden cloud cover, grid outage, DG activation
- [ ] Fallback: if optimizer fails, execute conservative default schedule

---

## M4 — BESS Execution Layer ⬚

**PRD reference:** Module M4
**Depends on:** M1 (clean data), M2 (forecasts), M3 (optimal schedule)

- [ ] Convert M3 schedule → BMS commands (charge/discharge/idle per interval)
- [ ] Safety checks: SOC bounds, temperature limits, max C-rate enforcement
- [ ] Grid code compliance: ramp rate limits per CERC/SERC regulations
- [ ] Telemetry ingestion: real-time SOC, voltage, current, temperature from BMS
- [ ] Closed-loop: actual SOC → correction signal back to M3

---

## M5 — Demand Response Engine ⬚

**PRD reference:** Module M5
**Depends on:** M1 (clean data + DG/APFC exclusion), M2 (demand forecast)

### M5-F1: Adjusted Historical Baseline

- [ ] 10 most recent non-curtailment, non-DG, non-holiday business days
- [ ] Morning adjustment factor (ratio of today's first 4h to baseline's first 4h)
- [ ] DG period exclusion (from M1-F5)
- [ ] APFC normalization (from M1-F6)
- [ ] High-variance interval detection and exclusion

### M5-F2: Curtailment Verification

- [ ] Compare actual kVA during DR event vs adjusted baseline
- [ ] Minimum sustained duration check (typically 15 min per DISCOM requirement)
- [ ] Prevent overclaiming from coincidental APFC events

### M5-F3: DR Event Economics

- [ ] Incentive calculation per DISCOM program rules
- [ ] Penalty for non-performance
- [ ] Integration with BESS dispatch (pre-charge battery before DR event)

---

## M6 — Explainability & Audit Layer ⬚

**PRD reference:** Module M6
**Phase 1 target:** 100% dispatch explanations (every charge/discharge decision has a reason)

- [ ] SHAP values for demand forecast features (which features drove this forecast?)
- [ ] Dispatch decision audit: for each interval, log why charge/discharge/idle was chosen
- [ ] Natural language explanations: "Charging at 02:00 because IEX price is ₹2.8/kWh (₹4.1 below grid tariff)"
- [ ] Monthly savings attribution: break savings into solar direct use, arbitrage, demand charge, DR
- [ ] Anomaly explanations: when quality pipeline flags data, explain what was detected and how it was handled
- [ ] Regulatory compliance: generate audit-ready reports showing dispatch decisions + data quality

---

## Known Bugs & Tech Debt

| # | Issue | Location | Fix In | Severity |
|---|-------|----------|--------|----------|
| BUG-1 | Efficiency loss as `np.sqrt(η)` — needs audit | `optimizer.py` ~L193 | M3 | Medium |
| BUG-2 | `iex_arbitrage_savings_inr` hardcoded to 0 | `economics.py` ~L198 | M3 | **High** |
| BUG-3 | Prophet holidays circular import workaround | `demand.py` L386-390 | M2 | Low |
| BUG-4 | `scipy.optimize` imported unused | `optimizer.py` imports | M3 | Low |
| DEBT-1 | Solar blend ratio hardcoded 40/60 | `solar.py` L205 | M2-F2 | Medium |
| DEBT-2 | API lag features NaN for missing history | `api/main.py` ~L203 | M4 | Medium |
| DEBT-3 | No tests for economics, foundation, loaders, quality | `tests/` | M1-M3 | Medium |
| DEBT-4 | Network charges not parametrized per consumer | `constants.py` | M3-F3 | Low |
| DEBT-5 | IRR calculation uses simplified fallback | `optimizer.py` L352 | M3-F3 | Medium |
| DEBT-6 | Hourly resolution throughout — PRD requires 15-min | All modules | M1-M3 | **High** |
| DEBT-7 | kW-based demand charges — India uses kVA | `economics.py` | M3-F1 | **High** |

---

## Metrics That Matter

| Metric | Current (Phase 0) | Target (Phase 1) | Target (Production) | Module |
|--------|-------------------|-------------------|---------------------|--------|
| Demand MAPE (24h-ahead) | 9.53% (synthetic) | <12% (real data) | <8% | M2 |
| Chronos zero-shot MAPE | 8.9% (synthetic) | <12% (real data) | <10% | M2 |
| Solar MAPE (24h-ahead) | Not measured | <15% | <10% | M2 |
| SOC correction accuracy | N/A | ±2% of lab reference | ±1% | M1 |
| Data quality score | 100% (synthetic) | >95% (real data) | >98% | M1 |
| Demand charge saving accuracy | N/A | ±5% of actual billing | ±3% | M3 |
| Dispatch savings (₹/month) | Not measured | Measured on real data | Validated | M3+M4 |
| P10-P90 coverage | 83.9% (Chronos) | >80% | >85% | M2 |
| Dispatch explanation coverage | 0% | 100% | 100% | M6 |

---

## Session Workflow

1. **Read** ROADMAP.md — find current module and next unchecked task
2. **Read** PROGRESS.md — check latest metrics and known issues
3. **Execute** — pick up the next task, write code, run tests
4. **Validate** — verify the task meets acceptance criteria
5. **Update** PROGRESS.md — log what was done, update metrics
6. **Update** ROADMAP.md — check off completed tasks
7. **Commit + push** — every session ends with code in GitHub

---

*This roadmap mirrors the EIL PRD module architecture. Update it as we learn — the plan should evolve with the product.*
