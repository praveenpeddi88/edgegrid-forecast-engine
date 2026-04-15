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

## M1 — Data Quality Engine ✅ COMPLETE

**PRD reference:** Module M1, Features M1-F1 through M1-F6
**Why first:** Every downstream model is only as good as the data feeding it. Indian grid data has unique noise patterns (voltage swings, DG transitions, CT artefacts, AMI packet loss) that generic quality pipelines miss entirely. Building these India-specific detectors creates a 6-12 month replication barrier.
**Code review:** Security A, Performance A, Correctness A, Maintainability A, Test Coverage A
**Location:** `src/edgegrid_forecast/data/quality/` (10 files, 1,760 lines)

**Phase 1 acceptance criteria (from PRD):**
- [x] AMI ingestion handles NULL, duplicates, late arrivals for 15-min resolution
- [x] Voltage-compensated SOC within +/-2% of lab-calibrated reference
- [x] Demand noise filter removes CT artefacts correlated with frequency deviation
- [x] DG transition detection flags grid-to-DG switchover events with <1 interval latency
- [x] APFC events excluded from DR baselines automatically
- [x] Data quality score computed per interval, per signal, per consumer

### M1-F1: Smart Meter AMI Ingestion ✅

**Location:** `quality/ami.py` (295 lines)

- [x] `detect_gaps(series, freq)` — vectorized with diff+cumsum (no Python loops)
- [x] `handle_duplicates(df, timestamp_col, meter_col)` — dedup keeping latest arrival
- [x] `handle_late_arrivals(df, max_delay_intervals)` — flag late but accept within window
- [x] `sync_channels(df, required_channels)` — vectorized with .notna().all(axis=1)
- [x] `validate_physical_ranges(df, channel_ranges)` — configurable per-channel bounds
- [x] `check_physical_consistency(df)` — division-safe kW/kVA/PF cross-validation
- [x] `compute_interval_quality_score(row)` — weighted composite (completeness 0.4, timeliness 0.3, validity 0.2, consistency 0.1)

### M1-F2: Statistical Anomaly Detection ✅

**Location:** `quality/anomaly.py` (232 lines)

- [x] `detect_frozen_readings(series, threshold)` — vectorized groupby.transform
- [x] `detect_outliers_zscore(series, threshold)` — with .fillna(False) on output
- [x] `detect_outliers_iqr(series, factor)` — explicit NaN-safe handling
- [x] `detect_outliers_contextual(series, group_col, threshold)` — time-of-day z-score via groupby.transform
- [x] `detect_outliers_rolling(series, window, threshold)` — 48h baseline with edge handling
- [x] `detect_outliers_isolation_forest(df, columns, contamination)` — multivariate

### M1-F3: Voltage Compensation / SOC Correction ✅

**Location:** `quality/voltage.py` (227 lines)

- [x] `VoltageSOCCorrector` class with input validation (polynomial_degree >= 1, min_calibration_points >= 10)
- [x] `detect_known_state_periods(soc, current)` — vectorized with groupby.agg
- [x] `calibrate(voltage, bms_soc, actual_soc)` — local effective_degree (never mutates configured degree)
- [x] `correct(voltage, bms_soc)` — returns corrected SOC clipped to [0, 100]
- [x] `check_calibration_drift(residuals, threshold, days)` — re-calibration trigger

### M1-F4: Demand Signal Noise Filter ✅

**Location:** `quality/noise.py` (173 lines)

- [x] `DemandNoiseFilter` class with cached rolling baseline (identity-based invalidation)
- [x] `compute_rolling_baseline(kva_series)` — 48h rolling median + sigma
- [x] `detect_ct_artefacts(kva, kw, frequency)` — sigma threshold + frequency correlation
- [x] `detect_pf_artefacts(kva, kw, pf)` — kVA spike with stable kW
- [x] `clean_demand_signal(kva, kw, frequency, pf)` — replace artefacts with rolling median

### M1-F5: DG Transition Detection ✅

**Location:** `quality/dg.py` (173 lines)

- [x] `DGTransitionDetector` class with input validation (import_drop_threshold_pct in 0-100)
- [x] `detect_grid_to_dg(grid_import, rolling_baseline)` — .where() for safe division
- [x] `detect_voltage_signature(voltage, frequency)` — secondary DG confirmation
- [x] `mark_dg_periods(grid_import, voltage, frequency)` — dg_confidence column (none/medium/high)
- [x] `exclude_dg_from_training(df, dg_mask)` — filtered DataFrame safe for model training

### M1-F6: APFC Switching Event Detection ✅

**Location:** `quality/apfc.py` (156 lines)

- [x] `APFCSwitchingDetector` class with input validation and configurable thresholds
- [x] `detect_kva_step(kva_series, threshold_kva)` — intervals with sudden kVA change
- [x] `detect_stable_kw(kw_series, kva_step_mask)` — confirm kW stability during kVA step
- [x] `detect_pf_jump(pf_series, kva_step_mask, target_pf_range)` — PF classification
- [x] `classify_apfc_events(kva, kw, pf)` — labeled: cap_switch_in, cap_switch_out
- [x] `normalize_for_dr_baseline(kva, kw, apfc_events)` — vectorized with shift(1) + boolean masking

### M1 Integration: Quality Pipeline ✅

**Location:** `quality/pipeline.py` (206 lines) + `quality/imputation.py` (97 lines)

- [x] `run_quality_pipeline()` orchestrates all M1 detectors per consumer
- [x] Per-interval quality score via `compute_interval_quality_score()`
- [x] `QualityReport` dataclass with 16 fields per consumer
- [x] Hybrid imputation: linear for short gaps, seasonal (same hour last week) for long gaps
- [x] Single detector instantiation outside loop, cache clearing per consumer
- [x] All constants in `_constants.py` (83 lines) — single source of truth

### M1 Remaining Work

- [ ] **Real meter data** — test all M1 detectors on actual APEPDCL AMI data (P0 blocker)
- [ ] **15-min resolution validation** — pipeline handles 15-min but needs real data to validate
- [ ] **Contextual anomaly tuning** — thresholds need calibration per consumer type

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
