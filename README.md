# EdgeGrid Forecast Engine

**Predictive dispatch engine for BESS + solar at APEPDCL HT consumer substations.**

Demand forecasting, solar generation prediction, IEX price intelligence, and battery dispatch optimization — built for India's commercial & industrial electricity consumers.

---

## The Problem

India's C&I electricity consumers overpay because procurement decisions are reactive, not predictive. At APEPDCL substations in Andhra Pradesh:

- **IEX day-ahead prices** swing 3× within a day (₹2.5/kWh at 3am → ₹9/kWh at 7pm)
- **Solar generation** is predictable but intermittent — dispatch must anticipate cloud cover hours ahead
- **Battery storage (BESS)** can arbitrage these spreads, but only with intelligent charge/discharge scheduling
- **FLS grid tariff** at ₹6.50/kWh is often more expensive than IEX landed cost during off-peak hours

Without accurate forecasts, BESS operators either leave money on the table (charging at peak prices) or oversize batteries for safety margins that better predictions would eliminate.

## The Solution

This engine takes weather data, demand history, and IEX prices as inputs, and outputs an optimal 24-hour BESS dispatch schedule that minimizes total energy cost per consumer per day.

```
Weather Forecast ──┐
                   ├──► Demand Forecast ──┐
Demand History ────┘                      │
                                          ├──► Dispatch Optimizer ──► ₹ Savings
Solar Irradiance ──────► Solar Forecast ──┘          │
                                                     │
IEX DAM Prices ────────► Price Forecast ─────────────┘
```

## Current Status

**Phase 0 complete.** The forecasting and dispatch infrastructure is built and validated on synthetic data. See [ROADMAP.md](ROADMAP.md) for the full 6-phase plan.

| Component | Status | Key Metric |
|-----------|--------|------------|
| Demand Forecasting (LightGBM) | ✅ Validated | 2.27% MAPE (1-step), 7.14% CV MAPE |
| Demand Forecasting (Chronos-Bolt) | ✅ Validated | 8.9% MAPE zero-shot, no training needed |
| Feature Engineering | ✅ 112 features | 8 families: temporal, lag, rolling, price, consumption, weather, solar, AQ |
| Weather Data Pipeline | ✅ 131K rows | Open-Meteo + NASA POWER, 3 locations, FY24-25 |
| IEX Price Infrastructure | ✅ CSV + synthetic | 8,737 hourly rows with realistic volatility |
| BESS Dispatch Optimizer | ✅ Built | 3 charging strategies, 24h windows |
| FastAPI Service | ✅ 7 endpoints | Demand, solar, price forecasts + dispatch |
| Test Suite | ✅ 35+ tests | Constants, dispatch, solar, price, features, API |

---

## Quick Start

### Install

```bash
git clone https://github.com/praveenpeddi88/edgegrid-forecast-engine.git
cd edgegrid-forecast-engine
pip install -e ".[dev]"
```

### Run the API

```bash
edgegrid-forecast
# → Swagger UI at http://localhost:8000/docs
# → ReDoc at http://localhost:8000/redoc
```

### Run Tests

```bash
pytest tests/ -v
```

### Train a Demand Model

```bash
python -m edgegrid_forecast.training.train_demand
```

This runs baseline vs weather-enriched model comparison on synthetic data and saves the production model to `models/demand/`.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       FastAPI Service                         │
│  GET /health  GET /consumers                                 │
│  POST /forecast/demand  /forecast/solar  /forecast/price     │
│  POST /dispatch/optimize                                     │
└────────┬───────────────┬──────────────────┬──────────────────┘
         │               │                  │
  ┌──────▼──────┐  ┌─────▼──────┐  ┌───────▼────────┐
  │   Demand    │  │   Solar    │  │   Dispatch     │
  │  Forecaster │  │ Forecaster │  │   Optimizer    │
  │             │  │            │  │                │
  │  LightGBM   │  │  Physics   │  │  Greedy sim    │
  │  Prophet    │  │  + ML      │  │  3 strategies  │
  │  Chronos    │  │  hybrid    │  │  BESS + solar  │
  │  Ensemble   │  │            │  │  + IEX arb     │
  └──────┬──────┘  └─────┬──────┘  └───────┬────────┘
         │               │                  │
  ┌──────▼───────────────▼──────────────────▼────────┐
  │              Feature Engineering (112 features)    │
  │  Temporal · Lag · Rolling · Price · Consumption    │
  │  Weather · Solar · Air Quality · Interactions      │
  └────────────────────────┬───────────────────────────┘
                           │
  ┌────────────────────────▼───────────────────────────┐
  │               Data Collection Layer                 │
  │  Open-Meteo Weather/Solar/AQ · NASA POWER          │
  │  IEX DAM Prices (CSV + synthetic) · APEPDCL Meters │
  └─────────────────────────────────────────────────────┘
```

---

## Project Structure

```
edgegrid-forecast-engine/
├── src/edgegrid_forecast/
│   ├── config.py                          # Central config (env vars, paths, params)
│   │
│   ├── data/
│   │   ├── collectors/
│   │   │   ├── open_meteo.py              # Weather + solar + AQ from Open-Meteo
│   │   │   ├── nasa_power.py              # 3-year solar baseline from NASA POWER
│   │   │   ├── iex_prices.py              # IEX DAM CSV parser + synthetic generator
│   │   │   └── pull_all.py                # Orchestrator — pulls all APIs, all locations
│   │   ├── features.py                    # 112-feature pipeline (8 families)
│   │   ├── synthetic.py                   # Synthetic demand for 6 consumer profiles
│   │   ├── loaders.py                     # DISCOM Excel + meter data ingestion
│   │   └── quality.py                     # Frozen detection, outliers, imputation
│   │
│   ├── models/
│   │   ├── demand.py                      # LightGBM + Prophet + Ensemble forecaster
│   │   ├── foundation.py                  # Chronos-Bolt zero-shot (9M/21M/48M params)
│   │   ├── solar.py                       # Physics + ML hybrid solar forecaster
│   │   └── price.py                       # IEX price forecaster with spread analysis
│   │
│   ├── dispatch/
│   │   ├── optimizer.py                   # BESS dispatch + sizing optimizer
│   │   └── economics.py                   # Landed cost, savings, demand charges, CO2
│   │
│   ├── training/
│   │   └── train_demand.py                # Baseline vs enriched A/B comparison
│   │
│   ├── api/
│   │   └── main.py                        # FastAPI with 7 endpoints
│   │
│   └── utils/
│       └── constants.py                   # IEX prices, tariffs, losses, BESS params
│
├── tests/                                 # 35+ tests across 6 modules
├── data/external/                         # Weather, AQ, NASA POWER parquets (131K rows)
├── docs/FOUNDATION.md                     # Complete input signal map (28 signals, 9 categories)
├── ROADMAP.md                             # Phased execution plan (7 phases)
├── PROGRESS.md                            # Session-by-session progress tracker
└── pyproject.toml                         # Python >=3.10, hatchling build
```

---

## Consumers

Six APEPDCL HT consumers across three locations in Andhra Pradesh:

| Consumer ID | Location | Type | Coordinates |
|-------------|----------|------|-------------|
| RJY1197 | Rajahmundry | Manufacturing | 17.0005°N, 81.8040°E |
| RJY1622 | Rajahmundry | Commercial | 17.0005°N, 81.8040°E |
| SKL724 | Srikakulam | Manufacturing | 18.2949°N, 83.8938°E |
| VSP2315 | Visakhapatnam | Commercial | 17.6868°N, 83.2185°E |
| VSP2432 | Visakhapatnam | IT Park | 17.6868°N, 83.2185°E |
| VSP2439 | Visakhapatnam | Manufacturing | 17.6868°N, 83.2185°E |

---

## Forecasting Models

### Demand Forecasting

Three approaches, designed to complement each other:

**LightGBM** (primary) — Gradient boosted trees trained on 112 features. Fast training (~seconds), handles missing values natively, interpretable feature importance. Best when training data is available.

**Chronos-Bolt** (cold-start) — Amazon's pretrained foundation model (9M parameters). Produces 168-hour forecasts with zero training by learning from millions of time series. Beats naive persistence by 2.0 percentage points on our 6 consumers. Autoregressive rollout extends beyond the native 64-step horizon.

**Prophet** (sanity check) — Facebook's structural time series model. Captures daily + weekly + annual seasonality and Indian holidays. Acts as an ensemble member with inverse-MAPE weighting.

| Model | MAPE (synthetic) | Training Required | Latency |
|-------|-----------------|-------------------|---------|
| LightGBM (1-step) | 2.27% | Yes (seconds) | <10ms |
| LightGBM (24h-ahead) | 9.53% | Yes | <10ms |
| Chronos-Bolt (168h) | 8.9% avg | No (zero-shot) | ~200ms on CPU |
| Naive persistence | 10.9% avg | No | Instant |

### Solar Generation

Physics-informed ML hybrid. A clear-sky irradiance model (solar position, air mass, atmospheric extinction) provides the physical baseline. An ML layer learns local cloud and weather corrections. The blend captures both the guaranteed physical constraints (no generation at night, seasonal arc) and the messy real-world patterns (cloud cover, temperature derating, panel soiling from PM2.5).

### Price Forecasting

Built on the FY24-25 IEX DAM monthly × hourly price matrix (12 months × 24 hours). Provides baseline prices, landed cost calculation (accounting for 9.1% cascaded transmission losses + ₹1.19/kWh network charges), temperature-adjusted premiums, and spread analysis for identifying BESS arbitrage windows.

---

## Dispatch Optimizer

The optimizer takes demand forecast, solar forecast, and IEX prices as inputs and produces a 24-hour BESS charge/discharge schedule. It runs a greedy simulation at hourly resolution:

1. **Solar direct use** — solar generation consumed directly by the load (free energy)
2. **BESS charging** — based on the selected strategy (see below)
3. **BESS discharge** — targeted at peak-price hours to maximize arbitrage
4. **Grid/IEX purchase** — remaining demand filled from cheapest available source

### Three Charging Strategies

| Strategy | Logic | Best For |
|----------|-------|----------|
| `solar_surplus` | Charge only from excess solar (solar > demand) | Maximum solar utilization |
| `full_solar` | Charge from all solar generation | Green energy branding |
| `cheap_grid` | Solar first, then grid when IEX landed < FLS tariff | Maximum ₹ savings |

### BESS Constraints

Default parameters from APEPDCL domain knowledge:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Min SoC | 10% | Protects battery health |
| Max SoC | 90% | Protects battery health |
| Round-trip efficiency | 88% | LFP chemistry typical |
| Degradation | 2.5%/year | Annual capacity loss |
| Calendar life | 15 years | Expected operational life |
| Cycle life | 6,000 | At 80% depth of discharge |
| Demand charge | ₹475/kVA/month | APEPDCL HT-I tariff |

### BESS Sizing

`optimize_bess_size()` sweeps across capacity (100-20,000 kWh), duration (2/4/6 hours), and all 3 strategies to find the configuration that maximizes IRR. Output includes simple payback period, annual savings, and capex for each configuration.

---

## Feature Engineering

112 features across 8 families, all toggleable for A/B testing:

| Family | Count | Key Features | Source |
|--------|-------|-------------|--------|
| Temporal | ~18 | hour, day_of_week, cyclical sin/cos, season, holiday, business_hour, ToD slab | Timestamp |
| Lag | 8 | 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h | Target variable |
| Rolling | ~24 | mean/std/min/max for 3h, 6h, 12h, 24h, 48h, 168h windows | Target variable |
| Price | 3 | iex_price, tod_multiplier, price_above_mean | IEX matrix |
| Consumption | 3 | daily_range_ratio, hourly_share, deviation_from_typical | Target variable |
| Weather | ~18 | CDH, HDD, heat_index, comfort_CDH, temp rolling/delta, cloud fraction, rain | Open-Meteo |
| Solar | ~14 | GHI rolling, variability, DNI/GHI ratio, diffuse fraction, solar_producing | Open-Meteo |
| Air Quality | ~8 | PM2.5 rolling, AOD rolling, dust_event, soiling_index | Open-Meteo AQ |
| Interactions | 4 | CDH×business_hour, CDH×peak, CDH×weekend, GHI×business | Cross-family |

Top features by importance (24h-ahead model): GHI 6-hour rolling mean, pressure delta 3h, temperature 24h rolling mean, surface pressure, temperature delta 3h.

---

## API Reference

Base URL: `http://localhost:8000`

### `GET /health`

Health check with available consumers.

```bash
curl http://localhost:8000/health
```
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "engine": "EdgeGrid Forecast Engine",
  "available_consumers": ["RJY1197", "RJY1622", "SKL724", "VSP2315", "VSP2432", "VSP2439"]
}
```

### `GET /consumers`

List consumers with metadata.

```bash
curl http://localhost:8000/consumers
```
```json
{
  "RJY1197": {"region": "Rajahmundry", "latitude": 17.0005, "longitude": 81.804},
  "VSP2315": {"region": "Visakhapatnam", "latitude": 17.6868, "longitude": 83.2185}
}
```

### `POST /forecast/demand`

Generate demand forecast for an HT consumer. Requires a pre-trained model at `models/<consumer_id>/lightgbm_demand.joblib`.

```bash
curl -X POST http://localhost:8000/forecast/demand \
  -H "Content-Type: application/json" \
  -d '{
    "consumer_id": "VSP2315",
    "horizon_hours": 48,
    "include_features": true
  }'
```

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `consumer_id` | string | required | One of the 6 HT consumer IDs |
| `horizon_hours` | int | 48 | Forecast horizon (1-168 hours) |
| `include_features` | bool | false | Include feature importance in response |

**Response:** `timestamps`, `point_forecast_kwh`, `lower_bound_kwh` (P10), `upper_bound_kwh` (P90), `model_name`, `metrics`

### `POST /forecast/solar`

Forecast solar generation for a given location and panel capacity.

```bash
curl -X POST http://localhost:8000/forecast/solar \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 17.6868,
    "longitude": 83.2185,
    "capacity_kw": 500,
    "horizon_hours": 48
  }'
```

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `latitude` | float | required | Location latitude (-90 to 90) |
| `longitude` | float | required | Location longitude (-180 to 180) |
| `capacity_kw` | float | required | Installed solar capacity in kW |
| `horizon_hours` | int | 48 | Forecast horizon (1-168 hours) |

**Response:** `timestamps`, `generation_kwh`, `clear_sky_kwh`, `capacity_factor`

### `POST /forecast/price`

Forecast IEX DAM prices and landed costs.

```bash
curl -X POST "http://localhost:8000/forecast/price?month=5&hours=24"
```

**Query parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `month` | int | 4 | Calendar month (1-12) |
| `hours` | int | 24 | Number of hours to forecast (1-168) |

**Response:** `timestamps`, `iex_price_inr_kwh`, `landed_cost_inr_kwh`, `cheapest_hours`, `expensive_hours`

### `POST /dispatch/optimize`

Optimize BESS dispatch for a 24-hour window.

```bash
curl -X POST http://localhost:8000/dispatch/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "demand_kwh": [180,170,165,160,158,165,200,280,350,400,420,410,390,370,350,330,310,340,420,450,400,350,280,220],
    "solar_kwh": [0,0,0,0,0,10,50,150,300,400,450,420,380,300,200,100,30,0,0,0,0,0,0,0],
    "iex_prices": [3.5,3.2,3.0,2.9,2.85,3.1,3.8,5.2,6.5,7.2,7.8,8.0,7.5,7.0,6.5,6.0,5.5,5.8,7.5,9.0,8.5,7.0,5.5,4.2],
    "bess_capacity_kwh": 500,
    "bess_duration_hours": 4,
    "fls_tariff": 6.50,
    "strategy": "cheap_grid"
  }'
```

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `demand_kwh` | float[24] | required | Hourly demand forecast (kWh) |
| `solar_kwh` | float[24] | required | Hourly solar forecast (kWh) |
| `iex_prices` | float[24] | required | IEX DAM prices (INR/kWh) |
| `bess_capacity_kwh` | float | null | Battery capacity (null = no BESS) |
| `bess_duration_hours` | int | 4 | Battery duration (C-rate) |
| `fls_tariff` | float | 6.50 | FLS grid tariff (INR/kWh) |
| `strategy` | string | "cheap_grid" | One of: solar_surplus, full_solar, cheap_grid |

**Response:** `solar_direct_use_kwh[24]`, `bess_charge_kwh[24]`, `bess_discharge_kwh[24]`, `bess_soc_kwh[24]`, `grid_purchase_kwh[24]`, `iex_purchase_kwh[24]`, `total_cost_inr`, `solar_savings_inr`, `bess_savings_inr`, `reliability_score`

---

## Data Pipeline

### External Data Sources

| Source | API | Rate Limit | Rows Collected | Period |
|--------|-----|------------|----------------|--------|
| Open-Meteo Weather | `archive-api.open-meteo.com` | 10K/day | 26,280 | FY24-25 |
| Open-Meteo Solar | Same endpoint | Same | Included above | FY24-25 |
| Open-Meteo Air Quality | `air-quality-api.open-meteo.com` | 10K/day | 26,280 | FY24-25 |
| NASA POWER | `power.larc.nasa.gov` | Unlimited | 78,912 | FY22-25 (3yr) |
| IEX DAM Prices | Hardcoded matrix + synthetic | N/A | 8,737 | FY24-25 |
| APEPDCL Meters | Manual upload | N/A | Not yet available | — |

All external data is stored as Parquet files in `data/external/` (4.3 MB total, zero nulls).

### Collecting Fresh Data

```bash
# Pull weather + solar + AQ for all 3 locations
python -m edgegrid_forecast.data.collectors.pull_all

# Pull NASA POWER baseline (3-year history)
python -m edgegrid_forecast.data.collectors.nasa_power
```

### IEX Price Import

IEX India has no public API. Prices are obtained via manual CSV export from [iexindia.com/market-data/day-ahead-market/market-snapshot](https://www.iexindia.com/market-data/day-ahead-market/market-snapshot).

```python
from edgegrid_forecast.data.collectors.iex_prices import load_iex_prices

# From CSV exports
prices = load_iex_prices(csv_dir="data/raw/iex/")

# Synthetic fallback (realistic noise around FY24-25 averages)
prices = load_iex_prices()  # No csv_dir → generates synthetic
```

---

## Domain Knowledge

### Landed Cost Formula

Converting IEX clearing price to what the consumer actually pays at the busbar:

```
landed_cost = iex_price / ((1 - 0.039) × (1 - 0.0275) × (1 - 0.0272)) + 1.19
            = iex_price / 0.9078 + 1.19

Losses:
  CTU  = 3.90%  (Central Transmission Utility)
  STU  = 2.75%  (State Transmission Utility)
  Dist = 2.72%  (Distribution)

Network charges (INR/kWh):
  SLDC              = ₹0.41
  Cross-subsidy     = ₹0.31
  Additional        = ₹0.47
  Total fixed addon = ₹1.19/kWh
```

### Time-of-Day Tariff

APEPDCL HT consumers pay different rates by time of day:

| Period | Hours | Multiplier |
|--------|-------|------------|
| Off-peak | 00:00-05:59, 22:00-23:59 | 0.90× |
| Normal | 06:00-09:59, 14:00-17:59 | 1.00× |
| Peak | 10:00-13:59, 18:00-21:59 | 1.20× |

### When IEX Beats Grid

The BESS value proposition exists because IEX landed cost is often cheaper than FLS tariff (₹6.50/kWh) during off-peak hours. For example, at IEX ₹3.00/kWh → landed cost = 3.00/0.9078 + 1.19 = ₹4.49/kWh (₹2.01 cheaper than grid). Charging the battery at ₹4.49 and discharging during peak hours when grid is at ₹7.80 (6.50 × 1.20) creates ₹3.31/kWh of arbitrage value, minus round-trip losses.

---

## Configuration

Environment variables (all optional, sensible defaults):

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGEGRID_DATA_DIR` | `./data` | Base data directory |
| `EDGEGRID_MODEL_DIR` | `./models` | Trained model storage |
| `EDGEGRID_API_HOST` | `0.0.0.0` | API bind host |
| `EDGEGRID_API_PORT` | `8000` | API bind port |
| `NSRDB_API_KEY` | (none) | NREL solar API key (optional) |
| `IEX_API_KEY` | (none) | IEX API key (not available) |

Create a `.env` file for local development:

```
EDGEGRID_DATA_DIR=./data
EDGEGRID_MODEL_DIR=./models
EDGEGRID_API_PORT=8000
```

---

## Development

### Prerequisites

- Python >= 3.10
- For Chronos-Bolt: `pip install chronos-forecasting torch` (CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`)

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_dispatch.py -v

# With coverage
pytest tests/ --cov=edgegrid_forecast --cov-report=term-missing
```

### Code Style

```bash
ruff check src/ tests/
ruff format src/ tests/
```

### Key Development Conventions

- **Timezone:** All timestamps in Asia/Kolkata (IST). IEX settlement, APEPDCL billing, and BESS dispatch all use IST.
- **Units:** Energy in kWh, power in kW, prices in INR/kWh, demand charges in INR/kVA/month.
- **Financial year:** April-March. Use `fy_month_index()` to convert calendar month → FY index. This is a common bug source.
- **Storage:** Parquet for all stored data (3-5× smaller than CSV, fast columnar reads).
- **Logging:** loguru throughout. Set `LOGURU_LEVEL=DEBUG` for verbose output.

---

## Architecture Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| AD-1 | LightGBM as primary demand model | Fast, handles 100+ features, native missing values, interpretable importance |
| AD-2 | Parquet for all stored data | Columnar compression, 3-5× smaller than CSV, fast time series reads |
| AD-3 | Open-Meteo over OpenWeather | Free tier, no API key, 10K calls/day, satellite-derived solar (Himawari-8/9) for India |
| AD-4 | NASA POWER for cross-validation | Free, 20+ year history, different satellite source — GHI correlation 0.9443 vs Open-Meteo |
| AD-5 | Synthetic demand until real meter data | Realistic profiles (6 consumers, 3 types) correlated with actual weather patterns |
| AD-6 | 90-day chunks for Open-Meteo | API limits hourly archive to ~90 days per request; chunking with 0.5s delay |
| AD-7 | Asia/Kolkata for all timestamps | Consistent with IEX settlement, APEPDCL billing, and BESS dispatch |
| AD-8 | 8-family feature pipeline | Modular: each family can be toggled on/off for A/B testing |
| AD-9 | Chronos-Bolt-Tiny as cold-start model | 9M params, CPU-only, 0.2s inference, zero training needed |
| AD-10 | Expanding window CV | Default TimeSeriesSplit starves fold 1; min_train_size=4000 ensures stable folds |
| AD-11 | IEX synthetic prices with log-normal noise | No public API exists; 12% volatility + 5% spike probability matches real market behavior |
| AD-12 | Autoregressive rollout for Chronos >64 steps | Native horizon is 64; feed predictions back as context for 168h forecasts |

---

## Related Documents

| Document | Purpose |
|----------|---------|
| [ROADMAP.md](ROADMAP.md) | Phased execution plan — 7 phases from foundation to scale, with tasks and acceptance criteria |
| [PROGRESS.md](PROGRESS.md) | Session-by-session progress tracker — metrics, commits, decisions, known gaps |
| [docs/FOUNDATION.md](docs/FOUNDATION.md) | Complete input signal map — 28 signals across 9 categories with API endpoints |

---

## License

MIT

---

Built for [EdgeGrid](https://edgegrid.in) — making India's distribution grid intelligent, one substation at a time.
