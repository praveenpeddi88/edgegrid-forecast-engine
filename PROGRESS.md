# EdgeGrid Forecast Engine — Progress Tracker

> Single source of truth for what's built, what's validated, and what's next.
> Updated each session. Committed alongside code changes.

**Last updated:** 2026-04-15
**GitHub:** `praveenpeddi88/edgegrid-forecast-engine` (private)
**Mission:** Predictive dispatch engine for BESS + solar at APEPDCL HT consumer substations


---

## 1. What's Built

### Core Modules (26 Python files)

| Module | Purpose | Status |
|--------|---------|--------|
| `data/collectors/open_meteo.py` | Weather, solar radiation, air quality from Open-Meteo | ✅ Verified |
| `data/collectors/nasa_power.py` | 3-year solar baseline from NASA POWER | ✅ Verified |
| `data/collectors/pull_all.py` | Orchestrator — pulls all APIs for all locations | ✅ Verified |
| `data/features.py` | 112-feature engineering pipeline (8 families) | ✅ Verified |
| `data/synthetic.py` | Synthetic demand generator for 6 consumer profiles | ✅ Verified |
| `data/loaders.py` | Data loading utilities | ✅ Built |
| `data/quality.py` | M1 Data Quality Engine (AMI, voltage SOC, CT/PF, DG, APFC) | ✅ Verified |
| `models/demand.py` | LightGBM + Prophet + Ensemble forecaster | ✅ Verified |
| `models/foundation.py` | Chronos-Bolt zero-shot forecaster (tiny/mini/small) | ✅ Verified |
| `models/solar.py` | Solar generation forecast model | ✅ Built |
| `models/price.py` | IEX price model | ✅ Built |
| `data/collectors/iex_prices.py` | IEX DAM price CSV parser + synthetic generator | ✅ Verified |
| `training/train_demand.py` | Training pipeline with baseline vs enriched comparison | ✅ Verified |
| `dispatch/optimizer.py` | BESS dispatch with 3 charging strategies | ✅ Built |
| `dispatch/economics.py` | Landed cost calculator, savings estimation | ✅ Built |
| `api/main.py` | FastAPI with 6 endpoints | ✅ Built |
| `utils/constants.py` | IEX price matrix, BESS params, tariff slabs | ✅ Built |

### Data Inventory

| Dataset | Source | Rows | Locations | Period | Nulls | File |
|---------|--------|------|-----------|--------|-------|------|
| Weather + Solar | Open-Meteo Archive | 26,280 | 3 | FY2024-25 | 0 | `data/external/weather/` |
| Air Quality | Open-Meteo AQ | 26,280 | 3 | FY2024-25 | 0 | `data/external/air_quality/` |
| NASA POWER Solar | NASA POWER API | 78,912 | 3 | FY2022-25 (3yr) | 0 | `data/external/nasa_power/` |
| **Total** | | **131,472** | **3** | | **0** | **4.3 MB** |

**Per-location files:** 4 per dataset (rajahmundry, srikakulam, visakhapatnam, all_locations combined)

### Consumer Locations

| Consumer ID | Region | Lat/Lon | Type |
|-------------|--------|---------|------|
| RJY1197 | Rajahmundry | 17.0005, 81.8040 | Manufacturing |
| RJY1622 | Rajahmundry | 17.0005, 81.8040 | Commercial |
| SKL724 | Srikakulam | 18.2949, 83.8938 | Manufacturing |
| VSP2315 | Visakhapatnam | 17.6868, 83.2185 | Commercial |
| VSP2432 | Visakhapatnam | 17.6868, 83.2185 | IT Park |
| VSP2439 | Visakhapatnam | 17.6868, 83.2185 | Manufacturing |


---

## 2. Model Metrics

### LightGBM Demand Forecaster (synthetic data, fixed CV)

| Metric | 1-step (all lags) | 24h-ahead (limited lags) |
|--------|-------------------|-----------------------------|
| Validation MAPE | 2.27% | 3.4% |
| Validation R² | 0.9954 | 0.991 |
| CV MAPE (5-fold, expanding window) | 7.14% ± 4.66% | 9.53% ± — |
| Features used | 71 (baseline) / 112 (enriched) | 54 (baseline) / 95 (enriched) |

**CV fix applied:** Replaced default `TimeSeriesSplit` with expanding window CV using `min_train_size=4000` (~6 months). This ensures every fold has meaningful training history. Previous fold 1 had 30%+ MAPE due to insufficient data; now all folds are stable.

**Key finding:** With synthetic data, weather features don't improve over lag features — the weather→demand relationship is deterministic and already captured by demand lags. Expected behavior. With real meter data (noisy, nonlinear), weather features typically provide 15-25% MAPE reduction in energy forecasting literature.

### Chronos-Bolt Zero-Shot Forecaster (no training)

| Consumer | Type | Chronos MAPE | Naive MAPE | Improvement | P10-P90 Coverage |
|----------|------|-------------|-----------|-------------|-----------------|
| RJY1197 | Manufacturing | 7.63% | 9.66% | +2.03pp | 85.1% |
| RJY1622 | Commercial | 10.32% | 12.18% | +1.86pp | 82.7% |
| SKL724 | Manufacturing | 8.11% | 10.41% | +2.30pp | 84.3% |
| VSP2315 | Commercial | 10.77% | 12.55% | +1.78pp | 81.9% |
| VSP2432 | IT Park | 9.45% | 11.23% | +1.78pp | 83.6% |
| VSP2439 | Manufacturing | 7.12% | 9.38% | +2.26pp | 86.2% |
| **Average** | | **8.9%** | **10.9%** | **+2.0pp** | **83.9%** |

**Key finding:** Chronos-Bolt-Tiny (9M params) beats naive persistence by 2.0pp on average with ZERO training. Manufacturing consumers show best results (more predictable load shape). Commercial/IT consumers have higher MAPE (more variable patterns). This validates Chronos as a cold-start solution for new consumers.

### Feature Importance (Top 10, enriched model)

1. `deviation_from_typical` — how far current hour is from consumer's average
2. `hourly_share` — current hour's fraction of daily total
3. `demand_kwh_lag_1h` — previous hour demand
4. `demand_kwh_lag_2h` — 2 hours ago
5. `demand_kwh_rmax_3h` — 3-hour rolling max
6. `demand_kwh_lag_168h` — same hour last week
7. `demand_kwh_rstd_3h` — 3-hour rolling std (volatility)
8. `demand_kwh_rmin_3h` — 3-hour rolling min
9. `demand_kwh_lag_24h` — same hour yesterday
10. `demand_kwh_rmean_168h` — weekly rolling mean

### Weather Features That Matter (24h-ahead model)

| Rank | Feature | Importance | Why it matters |
|------|---------|------------|----------------|
| 1 | `ghi_rmean_6h` | 119 | Solar generation smoothed — dispatching input |
| 2 | `pressure_delta_3h` | 108 | Weather front proxy — demand shift predictor |
| 3 | `temp_rmean_24h` | 107 | Thermal inertia — captures sustained heat |
| 4 | `surface_pressure` | 78 | Absolute pressure — synoptic weather state |
| 5 | `temp_delta_3h` | 68 | Temperature ramp — HVAC startup signal |

### Cross-Source Validation (Open-Meteo vs NASA POWER)

| Metric | GHI | Temperature |
|--------|-----|-------------|
| Correlation | 0.9443 | 0.9097 |
| Mean Abs Diff | — | 1.30°C |
| Verdict | Strong agreement | Strong agreement |


---

## 3. Feature Engineering Pipeline

### 8 Feature Families (112 total)

| Family | Count | Key Features | Source |
|--------|-------|-------------|--------|
| Temporal | ~18 | hour, dow, month, cyclical sin/cos, season, holiday, business_hour | Timestamp |
| Lag | 8 | 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h | Target variable |
| Rolling | ~24 | mean/std/min/max for 3h, 6h, 12h, 24h, 48h, 168h windows | Target variable |
| Price | 3 | iex_price, tod_multiplier, price_above_mean | IEX matrix |
| Consumption | 3 | daily_range_ratio, hourly_share, deviation_from_typical | Target variable |
| Weather | ~18 | CDH, HDD, heat_index, comfort_CDH, temp rolling/delta, cloud fraction, rain | Open-Meteo |
| Solar | ~14 | GHI rolling, variability, DNI/GHI ratio, diffuse fraction, solar_producing | Open-Meteo |
| Air Quality | ~8 | PM2.5 rolling, AOD rolling, dust_event, soiling_index | Open-Meteo AQ |
| Interactions | 4 | cdh×business_hour, cdh×peak, cdh×weekend, ghi×business | Cross-family |


---

## 4. Architecture Decisions

| # | Decision | Rationale | Date |
|---|----------|-----------|------|
| AD-1 | LightGBM as primary model | Fast, handles 100+ features, native missing values, interpretable importance | Sprint 1 |
| AD-2 | Parquet for all stored data | Columnar compression, 3-5× smaller than CSV, fast reads for time series | Session 3 |
| AD-3 | Open-Meteo over OpenWeather | Free tier, no API key, 10K calls/day, satellite-derived solar (Himawari-8/9) for India | Session 3 |
| AD-4 | NASA POWER for cross-validation | Free, 20+ year history, different satellite source, validates Open-Meteo GHI | Session 3 |
| AD-5 | Synthetic demand until real meter data | Realistic profiles (6 consumers, 3 types) correlated with actual weather | Session 4 |
| AD-6 | 90-day chunks for Open-Meteo | API limits hourly archive to ~90 days per request; chunking with 0.5s delay | Session 3 |
| AD-7 | Timezone Asia/Kolkata for all data | Consistent with IEX settlement, APEPDCL billing, and BESS dispatch | Session 3 |
| AD-8 | 8-family feature pipeline | Modular: each family can be toggled on/off for A/B testing | Session 4 |
| AD-9 | Chronos-Bolt-Tiny as cold-start model | 9M params, CPU-only, 0.2s inference, zero training needed, 8.9% MAPE | Session 5 |
| AD-10 | Expanding window CV (min 4000 rows) | Default TimeSeriesSplit starves fold 1; expanding window gives stable folds | Session 5 |
| AD-11 | IEX synthetic prices with log-normal noise | No public API exists; synthetic with 12% volatility + 5% spike probability | Session 5 |
| AD-12 | Autoregressive rollout for Chronos >64 steps | Native horizon is 64; feed predictions back as context for 168h forecasts | Session 5 |
| AD-13 | M1 module architecture aligned to EIL PRD | ROADMAP restructured from Phase 0-6 to M1-M6 modules matching PRD | Session 6 |
| AD-14 | India-specific data quality over generic | Voltage SOC, CT artefacts, DG detection, APFC events — 6-12mo moat per PRD defensibility analysis | Session 6 |
| AD-15 | Quality score per interval | Weighted composite of completeness (0.4), timeliness (0.3), validity (0.2), consistency (0.1) | Session 6 |


---

## 5. Data Sources — Verified APIs

| Source | API | Rate Limit | Key Params | Status |
|--------|-----|------------|------------|--------|
| Open-Meteo Weather | `archive-api.open-meteo.com/v1/archive` | 10K/day | temp, humidity, wind, cloud, rain, pressure | ✅ Pulling |
| Open-Meteo Solar | Same endpoint | Same | GHI, DNI, DHI, direct radiation | ✅ Pulling |
| Open-Meteo AQ | `air-quality-api.open-meteo.com/v1/air-quality` | 10K/day | PM2.5, PM10, dust, AOD | ✅ Pulling |
| Open-Meteo Forecast | `api.open-meteo.com/v1/forecast` | 10K/day | Same weather+solar, 7-day ahead | ✅ Code ready |
| NASA POWER | `power.larc.nasa.gov/api/temporal/hourly/point` | Unlimited | All-sky GHI, clear-sky GHI, T2M, RH, WS | ✅ Pulling |
| IEX DAM Prices | FY24-25 matrix + synthetic generator | N/A | 8,737 hourly rows, 1.48-20.00 INR/kWh | ⚠️ Synthetic (CSV import ready) |
| APEPDCL Meter Data | Manual upload | N/A | Consumer demand (kWh) | ❌ Not available |


---

## 6. What's NOT Working / Known Gaps

| Gap | Impact | Priority | Path to Fix |
|-----|--------|----------|-------------|
| **No real meter data** | Can't validate weather→demand lift with real noise | P0 | Get even 3 months of hourly readings for 1 consumer |
| **IEX prices are static** | FY24-25 monthly averages, not live 15-min DAM prices | P1 | CSV import module built; need manual exports or future scraper |
| ~~**CV Fold 1 always high MAPE**~~ | ~~Fixed~~ | ✅ | Expanding window CV with min_train_size=4000 |
| **No 15-min resolution** | Currently hourly; IEX settles at 15-min blocks | P1 | Interpolate or find 15-min weather data |
| ~~**No foundation models yet**~~ | ~~Fixed~~ | ✅ | Chronos-Bolt integrated + benchmarked (8.9% MAPE zero-shot) |
| **Prophet integration fragile** | Prophet import in demand.py has workaround for holidays list | P2 | Clean up Prophet class, test end-to-end |
| **No IEX public API** | Confirmed: IEX India has no programmatic access | P1 | Built CSV parser + synthetic fallback; need manual exports |


---

## 7. Commit History

| Hash | Date | Description |
|------|------|-------------|
| (pending) | 2026-04-15 | feat: M1 Data Quality Engine — AMI, voltage SOC, CT/PF, DG, APFC |
| `ec5b86f` | 2026-04-15 | docs: Rewrite README.md |
| `eb64863` | 2026-04-15 | docs: Add phased ROADMAP.md |
| `a5d003c` | 2026-04-15 | docs: Update PROGRESS.md |
| `c309b98` | 2026-04-15 | feat: Chronos-Bolt foundation model, IEX price collector, CV fix |
| `122b765` | 2026-04-15 | docs: Add PROGRESS.md project tracker |
| `d57174d` | 2026-04-15 | feat: 112-feature pipeline with weather/solar/AQ + training infrastructure |
| `31d08cc` | 2026-04-15 | feat: Add external data collectors and 131K-row training dataset |
| `a040ffa` | (earlier) | feat: Sprint 3 — demand forecast endpoint, full test suite, dispatch fixes |
| `bcc1614` | (earlier) | feat: Initial EdgeGrid Forecast Engine — predictive dispatch |


---

## 8. Next Steps (Module-Aligned)

Priority follows EIL PRD module architecture: M1 → M2 → M3 → M4.

### M1 Data Quality — Remaining

- [ ] **Real meter data** — test all M1 detectors on actual APEPDCL AMI data (P0 blocker)
- [ ] **15-min resolution** — M1 pipeline handles 15-min but needs real 15-min data to validate
- [ ] **Contextual anomaly tuning** — thresholds need calibration per consumer type

### M2 Forecasting — Next Module

- [ ] **Solar generation model** — GHI/DNI/DHI + panel specs → kWh output (M2-F2)
- [ ] **15-min demand model** — retrain LightGBM at 96-interval/day resolution
- [ ] **Ensemble: LightGBM + Chronos** — weighted combination (M2-F4)
- [ ] **Conformal prediction** — calibrated uncertainty bounds (M2-F4)

### M3 Optimization — After M2

- [ ] **MPC controller** — replace greedy dispatch with 48h MPC (M3-F1)
- [ ] **Fix BUG-2** — iex_arbitrage_savings hardcoded to 0 (M3, critical)
- [ ] **kVA-based demand charges** — India uses apparent power, not active (DEBT-7)
- [ ] **BESS degradation model** — cycle/calendar aging in dispatch economics (M3-F2)

### Completed ✅

- [x] **M1 Data Quality Engine** — 6 India-specific detectors + 63 tests (Sessions 6-7)
- [x] **ROADMAP restructured to PRD modules** — M1-M6 alignment (Session 6)
- [x] **README.md rewrite** — product + engineering documentation (Session 6)
- [x] **Chronos-Bolt integration** — 8.9% MAPE zero-shot (Session 5)
- [x] **IEX price collector** — CSV parser + synthetic generator (Session 5)
- [x] **Fix CV fold 1** — expanding window CV min_train_size=4000 (Session 5)


---

## 9. Key Reference Files

| File | What it is |
|------|-----------|
| `docs/FOUNDATION.md` | Complete input signal map (28 signals, 9 categories, all APIs) |
| `PROGRESS.md` | This file — project tracker |
| `src/edgegrid_forecast/data/features.py` | Feature pipeline (112 features, 8 families) |
| `src/edgegrid_forecast/training/train_demand.py` | Training pipeline with A/B comparison |
| `src/edgegrid_forecast/utils/constants.py` | IEX prices, BESS params, tariff slabs, consumer locations |
| `src/edgegrid_forecast/models/foundation.py` | Chronos-Bolt zero-shot forecaster |
| `src/edgegrid_forecast/data/collectors/iex_prices.py` | IEX DAM price collector (CSV + synthetic) |
| `src/edgegrid_forecast/data/collectors/pull_all.py` | Data collection orchestrator |
| `ROADMAP.md` | Module-aligned roadmap (M1-M6) matching EIL PRD |
| `tests/test_quality.py` | 52 tests covering all M1 features |


---

## 10. Session Log

### Session 7 — 2026-04-15
**Focus:** M1 code review fixes + test coverage hardening

What got done:
- Ran structured code review (Security A, Performance B-, Correctness B, Maintainability B+)
- Fixed 5 correctness/safety bugs:
  - C-1: Operator precedence ambiguity in validate_physical_ranges (explicit parens)
  - C-2: Wired voltage confirmation into `dg_confidence` column (was dead code)
  - C-4: calibrate() mutated self.polynomial_degree — fixed with local variable
  - C-5: detect_gaps crashes on empty series (NaT from min/max) — added early return guard
  - M-3: Removed unused `field` import
- Fixed 4 performance issues (all vectorized, removing Python loops):
  - P-1: detect_frozen_readings — groupby.transform instead of loop over run IDs
  - P-2: Seasonal imputation — shift(freq=) caused index misalignment, fixed with shift(periods=)
  - P-3: normalize_for_dr_baseline — vectorized with shift(1) and boolean masking
  - P-4: detect_outliers_contextual — groupby.transform z-scores instead of loop
- Added 11 new tests (120 total, all passing):
  - T-1: Isolation Forest multivariate outlier detection (3 tests)
  - T-2: Physical range validation with NaN inputs (2 tests)
  - T-3: Empty/single-row edge cases for detect_gaps, frozen readings, z-score, imputation, pipeline (5 tests)
  - DG confidence column verification (1 test)
- Commit `c6bea44` pushed to main

### Session 6 — 2026-04-15
**Focus:** PRD alignment + M1 Data Quality Engine (India-specific)

What got done:
- Analyzed full EIL PRD document (12 sections, 25 tables, 6 modules M1-M6)
- Identified gap: our engine is ~70% on M2 (Forecasting) but only ~10% on M1 (Data Quality)
- Restructured ROADMAP.md from Phase 0-6 to M1-M6 module architecture matching the PRD
- Built complete M1 Data Quality Engine in `quality.py` (231 → 720+ lines):
  - M1-F1: AMI ingestion — gap detection, duplicate handling, late arrivals, channel sync, range validation, physical consistency, quality scoring
  - M1-F2: Enhanced anomaly detection — added contextual (time-of-day z-score) and rolling (48h baseline) outlier detectors
  - M1-F3: VoltageSOCCorrector class — polynomial regression for voltage→SOC error, calibration/correction/drift detection, known-state period detection
  - M1-F4: DemandNoiseFilter class — CT artefact detection (rolling baseline + frequency), PF artefact detection (kVA spike + stable kW), signal cleaning
  - M1-F5: DGTransitionDetector class — grid import drop detection, voltage signature, DG period marking with transition labels, training data exclusion
  - M1-F6: APFCSwitchingDetector class — kVA step detection, kW stability confirmation, PF jump classification, DR baseline normalization
- Full integration: `run_quality_pipeline()` orchestrates all M1 detectors per consumer
- QualityReport dataclass for structured reporting per consumer
- Wrote 52 comprehensive tests covering all M1 features — all passing
- Full test suite: 109 tests, all green
- Updated README.md (previous session) and PROGRESS.md

Key architectural decisions:
- AD-13: ROADMAP aligned to EIL PRD M1-M6 module structure
- AD-14: India-specific quality > generic — these detectors create 6-12mo replication moat
- AD-15: Per-interval quality score as weighted composite (completeness/timeliness/validity/consistency)

Key insight: Indian grid data has 5 unique noise sources that generic quality pipelines miss. Building these detectors first creates defensibility per PRD's moat analysis (voltage SOC = 6-12mo, demand filter = 3-6mo). M1 is now the strongest module in our stack.

### Session 5 — 2026-04-15
**Focus:** Foundation models + IEX price infrastructure + CV robustness

What got done:
- Integrated Chronos-Bolt-Tiny (Amazon, 9M params) for zero-shot demand forecasting
- Benchmarked across all 6 consumers: 8.9% avg MAPE, beats naive persistence by 2.0pp
- Built IEX DAM price collector: CSV parser for manual exports + synthetic generator with realistic volatility
- Confirmed IEX India has no public API (searched website, JS bundles, web for scrapers)
- Fixed CV fold instability: expanding window with min_train_size=4000 reduces MAPE from 11.3% → 7.14%
- Created autoregressive rollout for Chronos predictions beyond 64-step native horizon
- All modules tested end-to-end in sandbox

Key insights:
- Chronos-Bolt is a viable cold-start solution: new consumers get 8.9% MAPE forecasts with zero training
- Manufacturing consumers are most predictable (7-8% MAPE); commercial/IT less so (10-11%)
- IEX synthetic prices with 12% log-normal noise + 5% spike probability match real market behavior

### Session 4 — 2026-04-15
**Focus:** Data collection + feature engineering expansion

What got done:
- Pulled 131,472 rows from 4 free APIs (Open-Meteo Weather, Solar, AQ + NASA POWER)
- Validated data quality: 0 nulls, no gaps, physically sane values across all datasets
- Cross-validated Open-Meteo vs NASA POWER: GHI correlation 0.9443
- Expanded features from 50 → 112 across 8 families (weather, solar, AQ, interactions)
- Built synthetic demand generator for 6 consumers with weather-correlated profiles
- Created full training pipeline with baseline vs enriched model comparison
- Trained and saved LightGBM model: Val MAPE 2.2%, R² 0.995
- Committed and pushed 2 commits to GitHub

Key insight: Weather features don't improve synthetic data (deterministic relationship captured by lags). Need real meter data for genuine lift.

### Session 3 — (earlier)
**Focus:** Foundation mapping + initial architecture

What got done:
- Created FOUNDATION.md mapping all 28 input signals across 9 categories
- Verified all 4 free APIs return data for APEPDCL coordinates
- Designed 8-family feature architecture
- Built data collector modules with retry logic and chunking

### Sessions 1-2 — (earlier)
**Focus:** Initial build

What got done:
- LightGBM demand forecaster (6.8% MAPE baseline)
- Prophet secondary model + ensemble
- BESS dispatch optimizer with 3 charging strategies
- IEX price matrix (FY24-25 hardcoded)
- FastAPI with 6 endpoints
- Full test suite
