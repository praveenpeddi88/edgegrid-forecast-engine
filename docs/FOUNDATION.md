# EdgeGrid Forecast Engine — Input Foundation Map

> Living document. Every signal, data source, model, and time block the engine needs to become a production-grade dispatch brain.

**Last updated:** April 2026 | **Engine version:** v0.2 planning | **Consumers:** 6 APEPDCL HT (3 locations)

---

## 1. Consumer Locations (Data Pull Targets)

All external API calls target these 3 coordinates:

| Location | Lat | Lon | Consumers | Region |
|----------|-----|-----|-----------|--------|
| Rajahmundry | 17.0005 | 81.8040 | RJY1197, RJY1622 | East Godavari |
| Srikakulam | 18.2949 | 83.8938 | SKL724 | North AP |
| Visakhapatnam | 17.6868 | 83.2185 | VSP2315, VSP2432, VSP2439 | Vizag metro |

---

## 2. Input Signals — Complete Registry

### 2.1 Weather Signals (Open-Meteo Weather API)

**Source:** `https://api.open-meteo.com/v1/forecast` (forecast) and `https://archive-api.open-meteo.com/v1/archive` (historical)
**Access:** Free, no API key, 10K calls/day. Non-commercial use.
**Coverage:** Global. All 3 AP locations confirmed returning data (April 2026).
**Historical depth:** 1940–present via archive API.

| Signal | Parameter name | Unit | Granularity | Forecast domain | Priority | Why it matters |
|--------|---------------|------|-------------|-----------------|----------|----------------|
| Temperature (2m) | `temperature_2m` | °C | Hourly | Demand, Solar | P0 | #1 driver of cooling load. Every 1°C above 30°C adds ~3-5% to C&I demand in AP summer. |
| Relative Humidity | `relative_humidity_2m` | % | Hourly | Demand | P1 | Drives apparent temperature and HVAC efficiency. Critical for coastal Vizag/Srikakulam. |
| Wind Speed (10m) | `wind_speed_10m` | km/h | Hourly | Solar, Demand | P1 | Higher wind = better panel cooling = higher efficiency. Also affects convective cooling demand. |
| Cloud Cover | `cloud_cover` | % | Hourly | Solar | P0 | Direct driver of solar irradiance attenuation. Open-Meteo also provides low/mid/high layers separately. |
| Precipitation | `precipitation` | mm | Hourly | Solar, Demand | P1 | Monsoon (Jun-Sep) reduces solar 40-60% in AP. Rain also washes panels (reduces soiling). |
| Surface Pressure | `surface_pressure` | hPa | Hourly | Solar | P2 | Used in air density calculations for Ineichen clear-sky model. Minor effect. |

**Verified endpoint (Visakhapatnam):**
```
GET https://archive-api.open-meteo.com/v1/archive
  ?latitude=17.6868&longitude=83.2185
  &start_date=2024-04-01&end_date=2025-03-31
  &hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover,precipitation,surface_pressure
  &timezone=Asia/Kolkata
```

### 2.2 Solar Radiation Signals (Open-Meteo Weather API + Satellite Radiation API)

**Source:** Same Open-Meteo endpoint for forecast; historical via archive API. Satellite Radiation API at `https://api.open-meteo.com/v1/satellite-radiation` for higher-res satellite data.
**Coverage:** India covered by Himawari-8/9 satellite (JAXA). 0.05° resolution.

| Signal | Parameter name | Unit | Granularity | Priority | Why it matters |
|--------|---------------|------|-------------|----------|----------------|
| Global Horizontal Irradiance | `shortwave_radiation` | W/m² | Hourly | P0 | THE primary solar input. Total radiation on horizontal surface. Everything else derives from this. |
| Direct Normal Irradiance | `direct_normal_irradiance` | W/m² | Hourly | P1 | Beam radiation perpendicular to sun. Needed for tracking systems and concentrating solar. |
| Diffuse Horizontal Irradiance | `diffuse_radiation` | W/m² | Hourly | P1 | Scattered radiation. On cloudy days, DHI can be 80-100% of GHI. Critical for monsoon accuracy. |
| Direct Radiation | `direct_radiation` | W/m² | Hourly | P1 | Direct beam on horizontal. GHI = Direct + Diffuse. Cross-check for data quality. |

**Verified sample (Vizag, April 13 2026 noon):** GHI=958 W/m², DNI=760 W/m², DHI=208 W/m², Direct=750 W/m². Consistent and physical.

### 2.3 Air Quality Signals (Open-Meteo Air Quality API)

**Source:** `https://air-quality-api.open-meteo.com/v1/air-quality`
**Access:** Free, same rate limits as weather API.
**Why:** India has some of the highest PM2.5 globally. Soiling can reduce solar output 5-25% depending on location and season.

| Signal | Parameter name | Unit | Granularity | Priority | Why it matters |
|--------|---------------|------|-------------|----------|----------------|
| PM2.5 | `pm2_5` | μg/m³ | Hourly | P1 | Fine particulate — primary driver of solar panel soiling losses. Also health-correlated demand patterns. |
| PM10 | `pm10` | μg/m³ | Hourly | P1 | Coarse dust — causes rapid soiling, especially in dry months (Oct-May). |
| Dust | `dust` | μg/m³ | Hourly | P2 | Atmospheric dust loading from Rajasthan/Deccan. Indicates soiling potential. |
| Aerosol Optical Depth | `aerosol_optical_depth` | dimensionless | Hourly | P2 | Each 0.1 increase reduces GHI ~2-3%. High in Indo-Gangetic plain and post-monsoon AP. |

**Verified sample (Vizag):** PM10=32, PM2.5=19.6, Dust=16.0, AOD=0.54. All parameters return real values (no nulls).

### 2.4 Long-Term Solar Baseline (NASA POWER)

**Source:** `https://power.larc.nasa.gov/api/temporal/hourly/point`
**Access:** Free, no API key. Fair-use rate limits.
**Historical depth:** 2001–near-real-time. Hourly resolution.
**Why:** 20+ year solar baseline for seasonal patterns, long-term capacity factor estimation, and validation of Open-Meteo satellite data.

| Signal | Parameter name | Unit | Granularity | Priority | Why it matters |
|--------|---------------|------|-------------|----------|----------------|
| All-Sky Solar Irradiance | `ALLSKY_SFC_SW_DWN` | Wh/m² | Hourly | P1 | Satellite-derived GHI under actual cloud conditions. 20+ year history. |
| Clear-Sky Solar Irradiance | `CLRSKY_SFC_SW_DWN` | Wh/m² | Hourly | P1 | Theoretical max irradiance with no clouds. Ratio ALLSKY/CLRSKY = cloud impact factor. |
| Temperature (2m) | `T2M` | °C | Hourly | P1 | Cross-validation with Open-Meteo. Also used for panel temperature correction. |
| Relative Humidity | `RH2M` | % | Hourly | P2 | Cross-validation source. |
| Wind Speed (10m) | `WS10M` | m/s | Hourly | P2 | Cross-validation source. Note: NASA uses m/s, Open-Meteo uses km/h. |

**Verified sample (Vizag, Jan 2025):** T2M=22.38°C, RH2M=89.13%, ALLSKY_SFC_SW_DWN=0.0 at midnight (correct). Fill value -999.0 (not observed in sample data).

### 2.5 ISRO MOSDAC (India-Specific Satellite)

**Source:** `https://www.mosdac.gov.in/`
**Access:** Free, open access. Registration required.
**Coverage:** India-specific INSAT-3D/3DR satellite data.
**Why:** 3-day solar/wind forecast at 15-min intervals. India-optimized. Complements Open-Meteo.

| Signal | Description | Granularity | Priority | Status |
|--------|-------------|-------------|----------|--------|
| Solar Radiation Forecast | 3-day ahead solar irradiance | 15-min | P1 | Needs registration + integration |
| Cloud Cover (INSAT) | India satellite cloud imagery | 30-min | P2 | Needs integration |

**Status:** Available but not yet connected. Requires registration and exploring their download API.

### 2.6 Energy Market Signals (IEX India)

**Source:** `https://www.iexindia.com/market-data/day-ahead-market/market-snapshot`
**Access:** No public API. Web scraping or daily manual CSV download.
**Granularity:** 15-minute time blocks since April 2022.

| Signal | Description | Unit | Granularity | Priority | Status |
|--------|-------------|------|-------------|----------|--------|
| IEX DAM Price | Day-ahead market clearing price | ₹/MWh | 15-min | P0 | Needs scraper — currently hardcoded FY24-25 matrix |
| IEX RTM Price | Real-time market price | ₹/MWh | 15-min | P1 | Needs scraper |
| IEX GDAM Price | Green day-ahead market | ₹/MWh | 15-min | P2 | Needs scraper |

**Current gap:** This is the biggest data gap. The hardcoded price matrix is an approximation. Real 15-min prices will dramatically improve dispatch optimization.

### 2.7 Grid System Signals (Grid-India / POSOCO)

**Source:** `https://posoco.in/en` (NLDC), `https://srldc.in/` (Southern Region)
**Access:** Published on website. No API. Needs scraping.

| Signal | Description | Unit | Granularity | Priority | Status |
|--------|-------------|------|-------------|----------|--------|
| Grid Frequency | Real-time supply-demand balance indicator | Hz | Real-time | P1 | Needs scraper |
| All-India Demand | National demand level | MW | 15-min | P1 | Needs scraper |
| Southern Region Demand | SR-specific demand (AP is in SR) | MW | 15-min | P1 | Needs scraper |

### 2.8 Consumer / DISCOM Signals (Internal)

| Signal | Description | Unit | Granularity | Priority | Status |
|--------|-------------|------|-------------|----------|--------|
| Smart Meter Consumption | Hourly/15-min readings from APEPDCL HT meters | kWh | 15-min | P0 | Integrated (annual Excel) |
| FLS Tariff / ToD Rates | Full Load Supply tariff + Time-of-Day multipliers | ₹/kWh | Annual | P0 | Integrated |
| Contract Demand | Max contracted demand with DISCOM | kVA | Static | P0 | Integrated |
| Open Access Charges | CSS (₹0.41) + Wheeling (₹0.31) + Addl (₹0.47) | ₹/kWh | Annual | P0 | Integrated |
| Load Factor | avg/peak demand ratio | ratio | Monthly | P1 | Integrated (computed) |

### 2.9 Calendar / Temporal Signals

| Signal | Description | Priority | Status |
|--------|-------------|----------|--------|
| Indian Holidays | 25+ state + national holidays (Diwali, Sankranti, etc.) | P0 | Integrated |
| Day of Week / Hour | Temporal encoding for daily/weekly patterns | P0 | Integrated |
| Season / Month | Seasonal encoding for monsoon/summer/winter | P0 | Integrated |
| ToD Slab | Peak(1.2x)/Normal(1.0x)/Off-peak(0.9x) classification | P0 | Integrated |

---

## 3. Forecasting Models — Ensemble Architecture

### 3.1 Currently Integrated (v0.1)

| Model | Type | Library | Domains | Status | Performance |
|-------|------|---------|---------|--------|-------------|
| LightGBM | Gradient boost | `lightgbm` | Demand, Solar | Integrated | 6.8% MAPE, R²=0.985 |
| Prophet | Classical | `prophet` | Demand | Integrated | Ensemble secondary (30% weight) |

### 3.2 Foundation Models (v0.2 — Zero-Shot Capable)

| Model | Library | Domains | GPU? | Key Strength |
|-------|---------|---------|------|-------------|
| Chronos-Bolt (Amazon) | `chronos-forecasting` | Demand, Price, Solar | No | Best on electricity benchmarks. 250x faster than Chronos v1. Zero-shot — no training needed. 47% error reduction over naive. |
| TimesFM 2.5 (Google) | `timesfm` | Demand, Price, Solar | No | Google production-tested. Best with long context (2048h). Supports covariates in v2.5. |
| MOIRAI-2 (Salesforce) | `uni2ts` | Demand, Solar | No | Native multivariate. Good for demand+weather joint forecasting. |

### 3.3 Deep Learning (v0.2+)

| Model | Library | Domains | GPU? | Key Strength |
|-------|---------|---------|------|-------------|
| Temporal Fusion Transformer | `pytorch-forecasting` | Demand, Price | Yes | Attention-based interpretability. Multi-horizon natively. SOTA for energy demand. |
| PatchTST | `Time-Series-Library` | Demand, Solar | Yes | Transfer learning across sites. Channel-independent patching. |
| N-BEATS / N-HiTS | `neuralforecast` / `darts` | Demand, Price | Yes | Pure feedforward. Fast inference. Hierarchical for multi-rate forecasting. |
| NeuralProphet | `neuralprophet` | Demand | No | Prophet-like API with neural backend. Auto-regression + lagged regressors. |

### 3.4 Ensemble Strategy

The v0.2 ensemble combines classical ML (fast, interpretable) with foundation models (zero-shot, robust):

```
For each consumer and horizon:
  1. Run LightGBM (trained on that consumer's history + weather features)
  2. Run Chronos-Bolt (zero-shot, just feed it the time series)
  3. Run TimesFM (zero-shot, with covariates if available)
  4. Combine via inverse-MAPE weighted average
  5. Use conformal prediction for proper uncertainty intervals
```

New consumers get instant forecasts from foundation models (no training data needed), which improve as history accumulates and LightGBM gets trained.

---

## 4. Time Block Architecture

Indian power market operates at multiple granularity levels. The engine must forecast at each:

| Time Block | Duration | Use Case | Regulatory Context |
|------------|----------|----------|-------------------|
| **5-minute** | 5 min | Future: CERC approved 5-min scheduling transition | Approved at 15th NPC. Implementation timeline TBD. Build readiness now. |
| **15-minute** | 15 min (96/day) | Current settlement. IEX DAM/RTM pricing. Deviation settlement. | Active since April 2022. This is the operational standard. |
| **Hourly** | 1 hour (24/day) | Current engine level. Weather API native resolution. | Legacy IEX. All weather APIs provide hourly. Most benchmarks use hourly. |
| **4-hour** | 4 hours (6/day) | Peak/shoulder/off-peak windows. Strategic BESS scheduling. | Operational, not regulatory. Night/Morning/Day/Afternoon/Evening/Night. |
| **Daily** | 24 hours | Day-ahead planning. BESS cycling. Solar yield. | DAM bidding is day-ahead. Generator scheduling is day-ahead. |
| **Weekly** | 7 days | Maintenance scheduling. Week-ahead procurement. | NLDC publishes weekly outlook. |
| **Monthly/Seasonal** | 30-90 days | Capacity planning. BESS sizing. Tariff arbitrage budgeting. | APERC tariff revisions annual. Seasonal forecasts for OA commitment. |

---

## 5. Data Volume & API Budget

### API Call Budget (Free Tier)
- Open-Meteo (weather + radiation + air quality): 10,000 calls/day each
- NASA POWER: Unlimited (fair use)
- For 3 locations × hourly refresh × 3 APIs = ~216 calls/day (well within limits)
- At 100 consumers (~30 unique locations): ~2,160 calls/day (still within limits)

### Data Volume Estimates (6 consumers, 3 locations)
- Weather: ~26K rows/year/location × 3 = ~78K rows
- Solar radiation: ~26K rows/year/location × 3 = ~78K rows
- Air quality: ~26K rows/year/location × 3 = ~78K rows
- Meter data: ~52K rows/year/consumer × 6 = ~315K rows
- IEX prices: ~35K rows/year (shared across consumers)
- **Total: ~585K rows/year** — fits comfortably in PostgreSQL/Parquet

### At Scale (100 consumers, ~30 locations)
- ~10M rows/year. Still fits in TimescaleDB/Parquet.

---

## 6. Feature Engineering Pipeline (v0.2 Target: 80+ Features)

### Currently Implemented (v0.1): 50+ features
- Temporal: hour, day_of_week, month, season, is_weekend, is_holiday, tod_slab
- Lag: lag_1h, lag_2h, lag_3h, lag_6h, lag_12h, lag_24h, lag_48h, lag_168h
- Rolling: roll_mean_6h, roll_mean_12h, roll_mean_24h, roll_std_6h, roll_std_24h
- Price: iex_price, landed_cost, price_spread, tod_multiplier
- Consumption pattern: diff_24h, ratio_to_daily_mean, peak_hour_flag

### New Features for v0.2 (Weather + Air Quality)
- Weather: temperature_2m, humidity_2m, wind_speed_10m, cloud_cover, precipitation, surface_pressure
- Derived: heat_index (temp × humidity interaction), cooling_degree_hours (base 26°C), heating_degree_hours
- Solar: GHI, DNI, DHI, clear_sky_index (GHI/clear_sky_GHI), cloud_impact_factor
- Air quality: pm2_5, pm10, dust, aod, soiling_index (cumulative PM2.5 since last rain)
- Cross-source: temp_x_cloud (interaction), wind_x_humidity, ghi_x_cloud_cover

---

## 7. Landed Cost Formula (Reference)

```
landed_cost = iex_price / ((1 - 0.039) × (1 - 0.0275) × (1 - 0.0272)) + 0.41 + 0.31 + 0.47
            = iex_price / 0.9078 + 1.19

Where:
  0.039  = CTU loss (3.9%)
  0.0275 = STU loss (2.75%)
  0.0272 = Distribution loss (2.72%)
  0.41   = SLDC charge (₹/kWh)
  0.31   = Cross-subsidy surcharge (₹/kWh)
  0.47   = Additional surcharge (₹/kWh)
```

---

## 8. Data Sources Quick Reference

| Source | Type | Access | Rate Limit | Key Parameters |
|--------|------|--------|------------|----------------|
| Open-Meteo Weather | REST API | Free, no key | 10K/day | temp, humidity, wind, cloud, rain, pressure |
| Open-Meteo Radiation | REST API | Free, no key | 10K/day | GHI, DNI, DHI (Himawari satellite for India) |
| Open-Meteo Air Quality | REST API | Free, no key | 10K/day | PM2.5, PM10, dust, AOD |
| NASA POWER | REST API | Free, no key | Fair use | 20yr solar/weather hourly history |
| ISRO MOSDAC | Download API | Free, registration | Unknown | India satellite solar/cloud forecast |
| IEX India | Web scrape | Free, no API | Daily | DAM/RTM/GDAM 15-min prices |
| Grid-India (NLDC) | Web scrape | Free, no API | Unknown | Frequency, demand, supply |
| APEPDCL | Internal | Excel/SFTP | On upload | Smart meter 15-min consumption |
| APERC | Open data | Annual PDF | Annual | FLS tariff, ToD rates, OA charges |

---

*This document is the single source of truth for the Forecast Engine's data dependencies. Update it whenever a new signal is added or an API changes.*
