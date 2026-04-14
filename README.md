# EdgeGrid Forecast Engine

**Predictive dispatch engine for India's distribution grid** — demand forecasting, solar generation prediction, BESS optimization, and energy market intelligence.

This is the Intelligence Layer of [EdgeGrid](https://edgegrid.in)'s Energy OS. It transforms smart meter data from DISCOMs into actionable dispatch decisions that minimize energy costs, maximize renewable utilization, and create financial value for every participant in the network.

## Why This Exists

India's distribution grid faces a coordination problem: 6,500+ commercial buildings with smart meters generate massive amounts of consumption data, but that data doesn't yet inform real-time energy procurement decisions. Meanwhile:

- **IEX DAM prices** swing 3× within a day (₹2.5 at 3am → ₹9 at 7pm)
- **Solar generation** is predictable but intermittent
- **BESS** can arbitrage price spreads, but only with intelligent dispatch
- **DISCOMs** need demand response signals, but have no forecasting infrastructure

This engine closes that gap.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Service                       │
│  /forecast/demand  /forecast/solar  /dispatch/optimize   │
└──────────┬──────────────┬──────────────┬────────────────┘
           │              │              │
    ┌──────▼──────┐ ┌─────▼─────┐ ┌─────▼──────┐
    │   Demand    │ │   Solar   │ │  Dispatch   │
    │  Forecaster │ │ Forecaster│ │  Optimizer  │
    │  (LightGBM  │ │ (Physics  │ │ (LP-based,  │
    │  + Prophet  │ │  + ML     │ │  3 BESS     │
    │  ensemble)  │ │  hybrid)  │ │  strategies)│
    └──────┬──────┘ └─────┬─────┘ └─────┬──────┘
           │              │              │
    ┌──────▼──────────────▼──────────────▼──────┐
    │            Feature Engineering             │
    │  Temporal · Lag · Rolling · Price · Pattern │
    └──────────────────┬─────────────────────────┘
                       │
    ┌──────────────────▼─────────────────────────┐
    │          Data Quality Pipeline              │
    │  Frozen detection · Outliers · Imputation   │
    └──────────────────┬─────────────────────────┘
                       │
    ┌──────────────────▼─────────────────────────┐
    │              Data Loaders                    │
    │  DISCOM Excel · Meter Reports · Weather API │
    │  IEX Prices · Solar Irradiance              │
    └─────────────────────────────────────────────┘
```

## Data Sources

| Source | Type | What It Provides |
|--------|------|-----------------|
| APEPDCL HT Consumer Profile | DISCOM Excel | 8760 hourly demand for 6 HT consumers |
| Meter Quality Report | Anomaly analysis | Frozen readings, outliers, gaps, quality flags |
| Open-Meteo API | Weather (free) | Temperature, cloud cover, irradiance |
| IEX DAM | Energy market | Day-ahead hourly clearing prices |
| Demand Response Brief | Domain knowledge | AutoDR pilot: 173 consumers, 67MW, 10% reduction |

## Models

### Demand Forecasting
- **Primary**: LightGBM with 50+ engineered features (temporal, lag, rolling, price, consumption pattern)
- **Secondary**: Prophet for structural seasonality decomposition
- **Production**: Weighted ensemble — weights learned via time-series cross-validation

### Solar Generation
- **Physics layer**: Clear-sky irradiance model (simplified Ineichen)
- **ML layer**: LightGBM learns cloud/weather corrections
- **Hybrid**: 40% physics + 60% ML blend

### Price Forecasting
- FY24-25 monthly×hourly IEX DAM matrix as baseline
- Temperature-adjusted premium for extreme heat days
- Spread analysis for BESS arbitrage opportunity sizing

### Dispatch Optimization
Three BESS charging strategies:
1. **Solar Surplus** — charge from excess solar only (cheapest)
2. **Full Solar** — charge from all solar generation
3. **Cheap Grid** — solar first, then grid when IEX landed < FLS tariff (highest volume)

BESS sizing optimizer sweeps 720+ configurations across size × duration × strategy.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run API
edgegrid-forecast
# → http://localhost:8000/docs

# Run tests
pytest tests/ -v

# Example: forecast solar for Visakhapatnam, 500kW system
curl -X POST http://localhost:8000/forecast/solar \
  -H "Content-Type: application/json" \
  -d '{"latitude": 17.6868, "longitude": 83.2185, "capacity_kw": 500}'
```

## Domain Knowledge

Critical constants (from APEPDCL tariff orders):

```python
# Landed cost from IEX to busbar
landed = iex_price / ((1-0.039) × (1-0.0275) × (1-0.0272)) + 0.41 + 0.31 + 0.47

# Demand charge: ₹475/kVA/month
# BESS constraints: 10% min SoC, 90% max SoC, 88% round-trip efficiency
# India grid emission factor: 0.82 kg CO2/kWh
```

## Project Structure

```
edgegrid-forecast-engine/
├── src/edgegrid_forecast/
│   ├── config.py              # Central configuration
│   ├── data/
│   │   ├── loaders.py         # DISCOM, meter, weather data ingestion
│   │   ├── quality.py         # Anomaly detection + imputation
│   │   └── features.py        # 50+ feature engineering
│   ├── models/
│   │   ├── demand.py          # LightGBM + Prophet + Ensemble
│   │   ├── solar.py           # Physics-informed ML hybrid
│   │   └── price.py           # IEX price forecasting
│   ├── dispatch/
│   │   ├── optimizer.py       # BESS dispatch + sizing optimizer
│   │   └── economics.py       # Financial calculations
│   ├── api/
│   │   └── main.py            # FastAPI service
│   └── utils/
│       └── constants.py       # Domain constants (tariffs, losses, prices)
├── tests/
├── notebooks/
├── data/raw/
└── pyproject.toml
```

## Roadmap

- [ ] **v0.1** — Core forecasting + dispatch (this release)
- [ ] **v0.2** — Real-time weather integration, model retraining pipeline
- [ ] **v0.3** — Multi-consumer cluster optimization (VNM-aware dispatch)
- [ ] **v0.4** — Demand response orchestration (AutoDR signal generation)
- [ ] **v0.5** — Carbon accounting + credit issuance
- [ ] **v1.0** — Production deployment with MLOps (model registry, monitoring, A/B testing)

## License

MIT

---

Built for [EdgeGrid](https://edgegrid.in) — The Energy Cloud of India.
