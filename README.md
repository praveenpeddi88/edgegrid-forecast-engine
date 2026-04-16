# EdgeGrid Forecast Engine

**Demand forecasting engine for BESS + solar dispatch at APEPDCL substations in Andhra Pradesh.**

42 smart meters. 30-minute intervals. 4.9% median MAPE. Built for India's commercial and industrial electricity grid.

---

## The Problem

India's C&I electricity consumers overpay because procurement decisions are reactive, not predictive. At APEPDCL substations in Andhra Pradesh:

- **IEX day-ahead prices** swing 3x within a day (Rs 2.5/kWh at 3am to Rs 9/kWh at 7pm)
- **Battery storage (BESS)** can arbitrage these spreads, but only with intelligent charge/discharge scheduling
- **FLS grid tariff** at Rs 6.50/kWh is often more expensive than IEX landed cost during off-peak hours
- **Indian grid data is noisy** — voltage swings of +/-15%, CT metering artefacts, diesel generator transitions, and 15-40% AMI packet loss corrupt the demand signal

Without accurate demand forecasts, BESS operators either leave money on the table or oversize batteries for safety margins that better predictions would eliminate.

## The Solution

This engine takes meter data, weather forecasts, and IEX prices as inputs, cleans the data through India-specific quality detectors, and generates accurate 30-minute interval demand forecasts that feed into BESS dispatch optimization.

```
Raw Meter Data ────► M1 Data Quality ──┐
  (noisy, gaps,      (clean, validate,  │
   CT artefacts)      India-specific)   │
                                        ▼
Weather (11 vars) ─┐              ┌─── Dispatch
                   ├─► M2 Demand ─┤    Optimizer ──► Rs Savings
Demand History ────┤   Forecast   │    (BESS schedule,
Voltage Telemetry ─┤   (LightGBM │     sizing, econ)
Holiday Calendar ──┘    v4)       │
                                  │
IEX DAM Prices ──────────────────┘
```

## Results

The engine has been iterated through four major versions. Here is the progression:

| Version | Strategy | Features | Mean MAPE | Median MAPE | Key Change |
|---------|----------|----------|-----------|-------------|------------|
| **v1** | S1 Chronological | 18 temporal + lag | 55.0% | 42.2% | Baseline |
| **v2** | S1 Chronological | 78 (no selection) | 59.6% | 42.2% | Overfitting — 57% meters regressed |
| **v3** | S1 Chronological | 43 → ~25 selected | 10.1% | 7.6% | Two-pass selection + weather + holidays |
| **v4** | S2 Stratified | 66 → ~36 selected | 9.1% | **4.9%** | Expanded weather + voltage + ToD tariff |

**91% improvement** from v1 to v4. 35 of 42 meters now achieve under 10% MAPE.

### Results by Tier (v4 S2)

| Tier | Meters | Median MAPE | Note |
|------|--------|-------------|------|
| HT (>5kWh) | 5 | 19.9% | Industrial loads with temporal trends |
| Large (1.5-5k) | 2 | 15.2% | Limited sample |
| Medium (0.5-1.5k) | 30 | 4.6% | Engine sweet spot |
| Small (<500) | 5 | 7.2% | High zero-demand inflates MAPE |

---

## Architecture

### Evaluation Strategies

Two complementary strategies measure different aspects of production readiness:

**Strategy 1: Chronological Cutoff** — Train on first 75% of timeline, predict last 25%. Answers: "Can we forecast unseen future?" Hardest test. Best for HT meters with temporal trends.

**Strategy 2: Stratified Temporal** — Every 4th complete day held out. Train and test span full timeline. Answers: "How accurate is steady-state daily forecasting?" Best for Medium/Small meters. No seasonal bias.

Production recommendation: Hybrid routing — S1 for HT, S2 for Medium/Small.

### Feature Engineering (v4)

66 candidate features, narrowed to ~36 per meter via two-pass selection:

| Category | Count | Examples |
|----------|-------|---------|
| Temporal | 12 | hour, day_of_week, month, is_weekend, is_holiday, day_of_year_sin/cos |
| Lag demand | 6 | lag_1, lag_2, lag_48, lag_96, lag_336 |
| Rolling stats | 6 | rmean_48, rmean_336, rstd_48, rmin_48, rmax_48 |
| Weather (base) | 11 | temperature, humidity, pressure, GHI, DNI, DHI, cloud, precip |
| Weather (derived) | 9 | pressure_delta_3h, heat_index, diffuse_fraction, ghi_rmean_6h |
| Voltage | 3 | voltage_lag1, voltage_rstd_6, voltage_rmean_48 |
| ToD tariff | 2 | tod_multiplier (APEPDCL 3-tier), is_peak |
| Interaction | 3 | peak_x_temp, heating_deg_15C |

### Two-Pass Feature Selection

This was the breakthrough that separated v3 from v2's overfitting disaster:

1. **Pass 1** — Quick fit (150 rounds) on all 66 candidates. Rank by LightGBM gain. Select top 55% (min 25).
2. **Pass 2** — Full training (800 rounds) with tier-adaptive hyperparameters on selected features only.

### Per-Tier Adaptive Regularization

| Parameter | HT (>5kWh) | Medium/Large | Small (<500) |
|-----------|------------|-------------|-------------|
| num_leaves | 31 | 63 | 63 |
| learning_rate | 0.03 | 0.05 | 0.05 |
| min_child_samples | 30 | 20 | 15 |
| lambda_l1 | 0.1 | 0.05 | 0.02 |
| lambda_l2 | 0.2 | 0.1 | 0.05 |

HT meters get stronger regularization (small dataset, high variance). Small meters get lighter regularization (maximize signal extraction).

### Chronos-Bolt Evaluation

Amazon's Chronos-Bolt-Tiny (9M params) was evaluated as a zero-shot foundation model. Result: 44% MAPE vs LightGBM's 9%. Ensemble calibration chose pure LightGBM for 41/42 meters. Chronos serves as a cold-start fallback (days 1-60 of new meters), not a production co-pilot.

### Cold-Start Protocol

1. **Days 1-14:** Chronos-Bolt zero-shot (~44% MAPE)
2. **Days 15-60:** LightGBM with temporal + weather features only (no lags)
3. **Days 60+:** Full v4 feature set including 7-day lags and rolling stats

---

## Data

### Meter Data

| Dataset | Rows | Meters | Period | Source |
|---------|------|--------|--------|--------|
| SP meters (1PH) | 63,561 | 5 | Jan 2025 – Feb 2026 | APEPDCL MDMS |
| TP meters (3PH) | 653,713 | 45 | Oct 2024 – Feb 2026 | APEPDCL MDMS |
| **Total** | **717K** | **50** | **Oct 2024 – Feb 2026** | |

42 of 50 meters eligible for benchmarking (≥180 days of data). 30-minute intervals, Wh resolution.

### Weather Data

11 variables from Open-Meteo Historical API for Visakhapatnam (17.72°N, 83.30°E):

`temperature_2m, relative_humidity_2m, dewpoint_2m, surface_pressure, cloud_cover, precipitation, wind_speed_10m, shortwave_radiation (GHI), direct_radiation, diffuse_radiation, direct_normal_irradiance`

Hourly resolution, interpolated to 30-min, timezone-aligned to IST.

### APEPDCL ToD Tariff

| Period | Hours | Multiplier |
|--------|-------|------------|
| Off-peak | 22:00–06:00 | 0.9x |
| Normal | 06:00–18:00 | 1.0x |
| Peak | 18:00–22:00 | 1.2x |

---

## Modules

### M1: Data Quality Engine (Complete)

India-specific quality pipeline with 6 detectors:

- **AMI packet loss** — 15-40% loss rates in Indian MDMS
- **Voltage SOC errors** — +/-3-8% from grid voltage swings
- **CT metering artefacts** — 2-4 Hz frequency swings
- **DG transitions** — Diesel generator switchovers masquerading as zero demand
- **APFC switching** — Capacitor bank events corrupting baselines
- **Anomaly detection** — Statistical outlier identification

10-file subpackage. Code review: Performance A, Correctness A, Maintainability A. 127 tests passing.

### M2: Demand Forecasting (Current Focus)

LightGBM v4 with 66-feature pipeline, two-pass selection, per-tier adaptive regularization. Trained per-meter. 42 meters benchmarked across two strategies.

### M3: Dispatch Optimizer (Built)

BESS dispatch with 3 charging strategies, 24h optimization windows, landed cost calculator.

---

## Project Structure

```
edgegrid-forecast-engine/
├── README.md                          # This file
├── PROGRESS.md                        # Session-by-session build log
├── ROADMAP.md                         # M1-M6 module roadmap
├── docs/
│   ├── FOUNDATION.md                  # Input signal registry (all APIs, all variables)
│   ├── STRATEGY_1_CHRONOLOGICAL_CUTOFF.md  # S1 analysis (v1 baseline → v3 results)
│   ├── STRATEGY_2_STRATIFIED_TEMPORAL.md   # S2 analysis (v4 results)
│   ├── V4_ENGINE_ARCHITECTURE.md      # Feature engineering, selection, training details
│   └── CHRONOS_ENSEMBLE_EVALUATION.md # Foundation model evaluation
├── src/edgegrid_forecast/
│   ├── data/
│   │   ├── quality/                   # M1 — 7 detector modules
│   │   ├── collectors/                # Open-Meteo, NASA POWER, IEX
│   │   ├── features.py                # 112-feature pipeline
│   │   └── loaders.py
│   ├── models/
│   │   ├── demand.py                  # LightGBM + Prophet
│   │   └── foundation.py             # Chronos-Bolt integration
│   ├── training/
│   │   ├── holdout_benchmark.py       # v1 baseline benchmark
│   │   ├── train_demand.py            # Training pipeline
│   │   └── train_real_demand.py       # Real meter data training
│   ├── dispatch/
│   │   ├── optimizer.py               # BESS dispatch
│   │   └── economics.py              # Landed cost, savings
│   └── api/main.py                    # FastAPI service
├── benchmarks/
│   ├── build_strategy2_engine.py      # S2 v3 benchmark script
│   ├── build_v4_ensemble_engine.py    # v4 ensemble (LightGBM + Chronos)
│   ├── generate_s2_v4_dashboard_data.py  # Dashboard data generator
│   └── results/
│       ├── benchmark_chronological.csv     # v1 S1 results (42 meters)
│       ├── benchmark_strategy1_v2.csv      # v2 S1 results
│       ├── benchmark_strategy1_v3.csv      # v3 S1 results
│       ├── benchmark_strategy2.csv         # S2 v3 results
│       └── benchmark_v4_ensemble.csv       # v4 S2 results (current best)
├── tests/                             # 127 tests
└── pyproject.toml
```

---

## Quick Start

```bash
git clone https://github.com/praveenpeddi88/edgegrid-forecast-engine.git
cd edgegrid-forecast-engine
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/ -v
```

### Run Benchmarks

The benchmark scripts in `benchmarks/` train all 42 meters and produce per-meter CSV results. They require `sp_data.parquet` and `tp_data.parquet` in the working directory.

```bash
# Strategy 2, v4 engine (current best)
python benchmarks/build_v4_ensemble_engine.py

# Generate dashboard data
python benchmarks/generate_s2_v4_dashboard_data.py
```

---

## Interactive Dashboards

Two self-contained HTML dashboards (React + Recharts, ~5MB each) provide interactive exploration:

- **Strategy 1 v3 Dashboard** — Chronological cutoff, 42 meters, animated prediction replay
- **Strategy 2 v4 Dashboard** — Stratified temporal, 42 meters, per-meter drill-down

Each dashboard has four views: See It Predict (animated forecast reveal), Validation (metrics + hourly MAPE + feature importance), Fleet (cross-meter comparison), and Engine (methodology explanation).

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| ML framework | LightGBM 4.x |
| Foundation model | Chronos-Bolt-Tiny (Amazon, 9M params) |
| Weather API | Open-Meteo Historical v1 |
| Holiday calendar | `holidays` (Python) — India + AP state |
| Dashboard | React 18 + TypeScript + Recharts |
| Data format | Apache Parquet, pandas |
| API | FastAPI |
| Tests | pytest (127 passing) |

---

## Key Decisions and Trade-offs

1. **Per-meter models vs fleet model** — Each meter gets its own LightGBM. More models to manage, but 42 diverse consumption patterns can't share weights effectively.

2. **Two-pass selection vs manual feature engineering** — The selector acts as a data-driven requirements doc. Each meter adopts only the features that improve its specific predictions.

3. **LightGBM over deep learning** — Sub-10ms inference, no GPU, interpretable feature importance. Deep learning (TFT, PatchTST) is planned for v5 but current accuracy doesn't justify the complexity.

4. **Hybrid evaluation strategy** — Different meter tiers exhibit fundamentally different dynamics. HT meters have temporal trends (use S1). Medium/Small meters are stationary (use S2).

5. **Chronos as safety net, not co-pilot** — Foundation model loses to trained LightGBM on every meter with sufficient data. But it's invaluable for cold-start.

---

*Built for APEPDCL. Powered by LightGBM. 42 meters, 4.9% median MAPE.*
