# EdgeGrid Forecast Engine — Progress Tracker

> Single source of truth for what's built, what's validated, and what's next.
> Updated each session. Committed alongside code changes.

**Last updated:** 2026-04-15
**GitHub:** `praveenpeddi88/edgegrid-forecast-engine` (private)
**Mission:** Predictive dispatch engine for BESS + solar at APEPDCL HT consumer substations


---

## 1. What's Built

### Core Modules (24 Python files)

| Module | Purpose | Status |
|--------|---------|--------|
| `data/collectors/open_meteo.py` | Weather, solar radiation, air quality from Open-Meteo | ✅ Verified |
| `data/collectors/nasa_power.py` | 3-year solar baseline from NASA POWER | ✅ Verified |
| `data/collectors/pull_all.py` | Orchestrator — pulls all APIs for all locations | ✅ Verified |
| `data/features.py` | 112-feature engineering pipeline (8 families) | ✅ Verified |
| `data/synthetic.py` | Synthetic demand generator for 6 consumer profiles | ✅ Verified |
| `data/loaders.py` | Data loading utilities | ✅ Built |
| `data/quality.py` | Data quality checks | ✅ Built |
| `models/demand.py` | LightGBM + Prophet + Ensemble forecaster | ✅ Verified |
| `models/solar.py` | Solar generation forecast model | ✅ Built |
| `models/price.py` | IEX price model | ✅ Built |
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

### Current: LightGBM Demand Forecaster (synthetic data)

| Metric | 1-step (all lags) | 24h-ahead (limited lags) |
|--------|-------------------|-----------------------------|
| Validation MAPE | 2.2% | 3.4% |
| Validation R² | 0.995 | 0.991 |
| CV MAPE (5-fold) | 11.3% ± 9.8% | 14.5% ± 12.5% |
| Features used | 71 (baseline) / 112 (enriched) | 54 (baseline) / 95 (enriched) |

**Key finding:** With synthetic data, weather features don't improve over lag features — the weather→demand relationship is deterministic and already captured by demand lags. Expected behavior. With real meter data (noisy, nonlinear), weather features typically provide 15-25% MAPE reduction in energy forecasting literature.

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


---

## 5. Data Sources — Verified APIs

| Source | API | Rate Limit | Key Params | Status |
|--------|-----|------------|------------|--------|
| Open-Meteo Weather | `archive-api.open-meteo.com/v1/archive` | 10K/day | temp, humidity, wind, cloud, rain, pressure | ✅ Pulling |
| Open-Meteo Solar | Same endpoint | Same | GHI, DNI, DHI, direct radiation | ✅ Pulling |
| Open-Meteo AQ | `air-quality-api.open-meteo.com/v1/air-quality` | 10K/day | PM2.5, PM10, dust, AOD | ✅ Pulling |
| Open-Meteo Forecast | `api.open-meteo.com/v1/forecast` | 10K/day | Same weather+solar, 7-day ahead | ✅ Code ready |
| NASA POWER | `power.larc.nasa.gov/api/temporal/hourly/point` | Unlimited | All-sky GHI, clear-sky GHI, T2M, RH, WS | ✅ Pulling |
| IEX DAM Prices | Hardcoded FY24-25 matrix | N/A | 12×24 price grid (INR/kWh) | ⚠️ Static |
| APEPDCL Meter Data | Manual upload | N/A | Consumer demand (kWh) | ❌ Not available |


---

## 6. What's NOT Working / Known Gaps

| Gap | Impact | Priority | Path to Fix |
|-----|--------|----------|-------------|
| **No real meter data** | Can't validate weather→demand lift with real noise | P0 | Get even 3 months of hourly readings for 1 consumer |
| **IEX prices are static** | FY24-25 monthly averages, not live 15-min DAM prices | P0 | Build IEX web scraper or find API |
| **CV Fold 1 always high MAPE** | First fold has limited training history (time series issue) | P1 | Use expanding window CV or drop first fold |
| **No 15-min resolution** | Currently hourly; IEX settles at 15-min blocks | P1 | Interpolate or find 15-min weather data |
| **No foundation models yet** | Chronos-Bolt, TimesFM could give zero-shot forecasts | P2 | Install and benchmark against LightGBM |
| **Prophet integration fragile** | Prophet import in demand.py has workaround for holidays list | P2 | Clean up Prophet class, test end-to-end |


---

## 7. Commit History

| Hash | Date | Description |
|------|------|-------------|
| `d57174d` | 2026-04-15 | feat: 112-feature pipeline with weather/solar/AQ + training infrastructure |
| `31d08cc` | 2026-04-15 | feat: Add external data collectors and 131K-row training dataset |
| `a040ffa` | (earlier) | feat: Sprint 3 — demand forecast endpoint, full test suite, dispatch fixes |
| `bcc1614` | (earlier) | feat: Initial EdgeGrid Forecast Engine — predictive dispatch |


---

## 8. Next Steps (Prioritized)

### Immediate (Next Session)

- [ ] **Real meter data** — even partial data for one consumer unlocks real validation
- [ ] **IEX price scraper** — automate 15-min DAM price collection from IEX website
- [ ] **Fix CV fold 1** — switch to expanding window or minimum training size

### Short-term (Next 2-3 Sessions)

- [ ] **Chronos-Bolt integration** — zero-shot demand forecast, compare with LightGBM
- [ ] **TimesFM 2.5** — Google's foundation model for time series
- [ ] **15-min resolution** — interpolate weather data to match IEX settlement periods
- [ ] **Solar generation model** — use GHI/DNI/DHI + panel specs to predict kWh output
- [ ] **Dispatch optimizer v2** — feed actual forecasts into BESS dispatch loop

### Medium-term

- [ ] **MOIRAI-2** — Salesforce foundation model, probabilistic forecasts
- [ ] **Conformal prediction** — replace the naive ±15% uncertainty bounds
- [ ] **Real-time pipeline** — Open-Meteo forecast API → model inference → dispatch recommendation
- [ ] **Multi-consumer aggregation** — portfolio-level dispatch across all 6 consumers
- [ ] **Dashboard** — live forecast vs actual comparison for model monitoring


---

## 9. Key Reference Files

| File | What it is |
|------|-----------|
| `docs/FOUNDATION.md` | Complete input signal map (28 signals, 9 categories, all APIs) |
| `PROGRESS.md` | This file — project tracker |
| `src/edgegrid_forecast/data/features.py` | Feature pipeline (112 features, 8 families) |
| `src/edgegrid_forecast/training/train_demand.py` | Training pipeline with A/B comparison |
| `src/edgegrid_forecast/utils/constants.py` | IEX prices, BESS params, tariff slabs, consumer locations |
| `src/edgegrid_forecast/data/collectors/pull_all.py` | Data collection orchestrator |


---

## 10. Session Log

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
