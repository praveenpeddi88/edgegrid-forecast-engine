# v4 Engine Architecture — Feature Engineering, Selection, and Training

> The v4 engine is the current production candidate. It processes 66 candidate features through a two-pass selection pipeline, applies per-tier adaptive regularization, and trains individual LightGBM models per meter.

**Last updated:** April 2026 | **Engine version:** v4 (production candidate) | **Fleet:** 42 APEPDCL smart meters (Andhra Pradesh, India)

---

## 1. Overview

The v4 engine improves upon v3 by expanding the feature set from 43 to 66 candidates while maintaining—and improving—accuracy through enhanced two-pass selection. The critical insight remains the same: **selection is more important than feature expansion**. v2 proved that adding features without selection causes overfitting. v3 proved that selection fixes it. v4 proves that selection + more features = genuine improvement.

**Key Performance Metrics (42-meter fleet):**
- Mean MAPE: 9.1%
- Median MAPE: 4.9%
- Training time (all 42 meters, CPU): ~73 seconds
- Inference latency per prediction: <10ms

---

## 2. Version Evolution — The Case for v4

Each version taught us something critical about feature engineering at this scale.

| Version | Features | Selection | Mean MAPE | Median MAPE | Key Lesson |
|---------|----------|-----------|-----------|-------------|-----------|
| **v1** | 18 (temporal + lag) | None | 55.0% | 42.2% | Lag features alone can't handle seasonal shift. v1 works on some meters but fails on others. |
| **v2** | 78 (added weather, voltage, derived) | None | 59.6% | 42.2% | **More features WITHOUT selection = overfitting.** 57% of meters got WORSE than v1. Proof that naive feature expansion destroys accuracy. |
| **v3** | 43 → ~25 selected | Two-pass | 10.1% | 7.6% | **Selection was the breakthrough.** Not adding features — choosing them. Jumped from 59.6% to 10.1% mean MAPE in one change. The single most important technical decision. |
| **v4** | 66 → ~36 selected | Two-pass + tier-adaptive | 9.1% | 4.9% | **More features WITH selection = genuine improvement.** Proved that selection scales. We can safely add more candidates; the selector filters noise. |

### The v2 → v3 Inflection Point

v2 was a cautionary tale. We added 60 features (weather, voltage, derived rolling stats) and the model got WORSE. Why?

- **Root cause:** With 78 features and no selection, the model found spurious correlations in the training data. It memorized noise. On holdout data, it performed terribly.
- **Lesson:** Feature engineering without selection is cargo-cult machine learning. More data points != better features.
- **The fix:** Two-pass selection. Screening pass ranks all features by importance. Full training uses only the top 55%. Reduces overfitting dramatically.

Result: mean MAPE jumped 59.6% → 10.1% (6x improvement) by removing 30+ noisy features, not by adding anything new.

### Why v4 Works

v4 adds 23 new candidate features (66 total, up from 43 in v3) but maintains accuracy because:
1. **Selection at scale:** The two-pass selector now evaluates 66 candidates instead of 43. It handles the larger feature space gracefully.
2. **Non-redundant expansion:** The 23 new features are not random. They target gaps in v3's feature set (e.g., weather interactions, pressure-based weather front detection).
3. **Proof by experiment:** Median MAPE improved 7.6% → 4.9%. The additional features carry signal even after aggressive selection.

---

## 3. Feature Pipeline — 66 Candidates

All candidates flow through the two-pass selection process. Only the selected subset (typically ~36 features) is used in the final model.

### 3.1 Temporal Features (12 total)

These capture daily, weekly, and seasonal demand patterns.

| Feature | Type | Range | Rationale |
|---------|------|-------|-----------|
| `hour` | int | 0–23 | Hour of day. Demand peaks at 18:00 (evening air conditioning). Lows at 04:00. |
| `day_of_week` | int | 0–6 (Mon–Sun) | Day-of-week effect. Weekdays have higher load than weekends. |
| `month` | int | 1–12 | Seasonal. Summer (Apr–May) = high cooling load. Winter (Dec–Jan) = lower. |
| `day_of_year` | int | 1–366 | Long-range seasonality (monsoon Jun–Sep reduces solar demand). |
| `is_weekend` | binary | 0/1 | Binary weekday/weekend flag. Industrial meters show clear weekend drop. |
| `is_holiday` | binary | 0/1 | India + Andhra Pradesh state holidays (25+ per year: Diwali, Holi, Sankranti, etc.). Residential/commercial drop ~20% on holidays. |
| `hour_sin`, `hour_cos` | float | [-1, 1] | Cyclical encoding of hour (period=24). Captures that hour 23 is adjacent to hour 0, not far away. |
| `day_of_week_sin`, `day_of_week_cos` | float | [-1, 1] | Cyclical encoding of day-of-week (period=7). Sunday wraps to Monday. |
| `day_of_year_sin`, `day_of_year_cos` | float | [-1, 1] | Cyclical encoding of day-of-year (period=365.25). Dec 31 wraps to Jan 1. |

**Why cyclical encoding?** LightGBM trees treat hour=0 and hour=23 as distant. Sin/cos encoding makes them adjacent (sin(0) ≈ sin(2π)), which is physically correct.

### 3.2 Lag Demand Features (6 total)

Historical demand at specific offsets. These are almost always selected—especially lag_48.

| Feature | Lag Offset | Rationale |
|---------|-----------|-----------|
| `lag_1` | 30 minutes | Most recent demand. Captures momentum. |
| `lag_2` | 1 hour | Smooths over transient spikes. 30-min grid resolution. |
| `lag_48` | 24 hours (= 48 × 30-min intervals) | **#1 feature across ALL meters.** Demand at 2pm today is best predicted by demand at 2pm yesterday. Same time, same day-of-week, same weather (mostly). |
| `lag_96` | 48 hours (2 days ago) | Captures day-of-week + some error from yesterday. |
| `lag_336` | 7 days (= 336 × 30-min intervals, same weekday 1 week prior) | Weekly baseline. Same weekday last week. |
| `lag_672` (if selected) | 14 days | Bi-weekly pattern (optional, ranked by selector). |

**Insight:** Lags are the strongest predictors because they embed temporal structure. A model with only lags reaches ~15–20% MAPE. Adding weather/voltage/derived features brings it to ~9%.

### 3.3 Rolling Statistics (6 total)

Smooth demand signals and capture volatility.

| Feature | Window | Statistic | Rationale |
|---------|--------|-----------|-----------|
| `rmean_48` | 24 hours | Mean | Smooths the daily demand cycle. Removes spike noise. |
| `rmean_336` | 7 days | Mean | 7-day baseline. Captures weekly trend. |
| `rstd_48` | 24 hours | Std dev | 24-hour volatility. High std = variable industrial load. Low std = steady consumption. |
| `rmin_48` | 24 hours | Min | Daily minimum (usually night). Tells model about baseline load. |
| `rmax_48` | 24 hours | Max | Daily maximum (usually evening). Tells model about peak headroom. |
| `rmean_ratio_48_336` | 24h / 7d | Ratio | Short-term vs long-term trend. Ratio > 1 = demand increasing. Ratio < 1 = demand decreasing. |

**Interpretation:** rmean_48 captures the "shape" of the day. rstd_48 captures how volatile that day is. Together they describe the current regime.

### 3.4 Weather Base Features (11 total)

**Source:** Open-Meteo Historical API for Visakhapatnam (17.72°N, 83.30°E, APEPDCL service area)  
**Resolution:** Hourly, interpolated to 30-minute, timezone-aligned to IST (UTC+5:30)

| Feature | Unit | Source | Why It Matters |
|---------|------|--------|----------------|
| `temperature_2m` | °C | Open-Meteo | #1 weather driver. Every 1°C above 30°C adds ~3–5% to cooling load in AP summer. HVAC ramping. |
| `relative_humidity_2m` | % | Open-Meteo | Drives apparent temperature. High humidity + high temp = extreme heat index. Coastal Vizag effect. |
| `dewpoint_2m` | °C | Open-Meteo | Absolute moisture. Complements relative humidity (RH can be high in cool, wet conditions). |
| `surface_pressure` | hPa | Open-Meteo | Air density for solar calculations. Also indicates weather fronts (low pressure = incoming rain). |
| `cloud_cover` | % | Open-Meteo | Direct attenuation of solar radiation. Cloudy day = low solar yield = grid demand increases. |
| `precipitation` | mm | Open-Meteo | Monsoon (Jun–Sep) reduces solar 40–60% in AP. Rain also washes panels (temporary soiling reduction). |
| `wind_speed_10m` | km/h | Open-Meteo | Higher wind = better panel cooling = higher efficiency. Also affects convective cooling demand. |
| `shortwave_radiation` / GHI | W/m² | Open-Meteo | Global Horizontal Irradiance. Actual solar radiation on horizontal surface. |
| `direct_radiation` | W/m² | Open-Meteo | Direct beam radiation component (vs. diffuse). Used for tracking systems. |
| `diffuse_radiation` / DHI | W/m² | Open-Meteo | Scattered radiation. On cloudy days, DHI can be 80–100% of GHI. Monsoon-critical. |
| `direct_normal_irradiance` / DNI | W/m² | Open-Meteo | Beam perpendicular to sun. For concentrating solar. Usually highest at noon. |

**Data Quality:** Open-Meteo provides hourly data. We interpolate to 30-minute to match smart meter granularity (linear interpolation; acceptable for hourly resolution).

### 3.5 Weather Derived Features (9 total)

Non-linear combinations and physics-informed transformations of raw weather.

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `pressure_delta_3h` | pressure[t] - pressure[t-3h] | 3-hour pressure change. Indicates weather front approach. Rapid drop = incoming rain/storm. |
| `cloud_delta_3h` | cloud_cover[t] - cloud_cover[t-3h] | Cloud cover trend. Sudden increase = storm incoming. Affects solar + demand. |
| `heat_index` | temp × humidity interaction (Steadman formula) | Apparent temperature. What it "feels like" to HVAC system. Drives cooling load more than raw temp. |
| `diffuse_fraction` | DHI / GHI | Fraction of radiation that is scattered (vs. direct). <0.2 = clear sky. >0.6 = heavy cloud. |
| `ghi_rmean_6h` | 6-hour rolling mean of GHI | Smoothed solar signal. Removes cloud flicker. Captures sustained cloud cover vs. transient shadows. |
| `temp_rmean_24h` | 24-hour rolling mean of temperature | Thermal inertia. Yesterday's temperature affects today's cooling demand (buildings have mass). |
| `temp_delta_3h` | temperature[t] - temperature[t-3h] | 3-hour temperature ramp. Rapid warming = HVAC startup signal. Rapid cooling = storm. |
| `humidity_x_temp` | relative_humidity × temperature | Interaction term. High temp alone is manageable; high temp + high humidity is extreme. |
| `wind_chill_factor` | wind_speed × (temperature - 20) interaction | Wind × temperature interaction. High wind + cold = convective cooling demand increase. |

**Rationale:** These derived features encode domain knowledge (heat index, thermal inertia) without explicitly hardcoding them. The selector decides if they're useful per meter.

### 3.6 Voltage Telemetry Features (3 total)

**Source:** APEPDCL MDMS (Meter Data Management System) voltage readings  
**Relevance:** Indian distribution networks have ±15% voltage swings. This affects both meter accuracy and real load behavior.

| Feature | Window | Rationale |
|---------|--------|-----------|
| `voltage_lag1` | Lag 1 (30-min) | Previous interval voltage. Grid voltage is stable but drifts slowly (±2–5% typical swing). |
| `voltage_rstd_6` | 3-hour rolling std | 3-hour voltage volatility. High = unstable grid section (poor regulation, heavy loads, renewable swings). |
| `voltage_rmean_48` | 24-hour rolling mean | Daily mean voltage. Morning voltages differ from evening (load swings). Meters read differently at different voltages. |

**Caveat:** Voltage is highly location-specific. A meter 5km away can have 10%+ different voltage profile. Per-meter selection handles this.

### 3.7 Time-of-Day Tariff Features (2 total)

APEPDCL operates a 3-tier ToD tariff that incentivizes load shifting.

| Feature | Definition | Rationale |
|---------|-----------|-----------|
| `tod_multiplier` | Off-peak (22:00–06:00) = 0.9×; Normal (06:00–18:00) = 1.0×; Peak (18:00–22:00) = 1.2× | APEPDCL charges 20% premium during peak. Consumers shift discretionary load (water pumps, EV charging) to off-peak. Model must capture this. |
| `is_peak` | Binary: 1 if 18:00–22:00, else 0 | Peak flag. Simpler than multiplier; ranked by selector per meter. |

**Effect Size:** Industrial meters can reduce peak demand 10–25% through ToD optimization. ToD signals are critical for accurate peak prediction.

### 3.8 Interaction Features (3 total, dynamically extended)

Explicit pairwise interactions, identified by domain knowledge and selector.

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `peak_x_temp` | is_peak × temperature | Peak period × temperature. Captures peak-hour cooling load (18:00–22:00 air conditioning intensity). |
| `heating_deg_15C` | max(0, 15 − temperature) | Heating degree hours (base 15°C). Relevant in winter (Dec–Jan) when heating/space conditioning exists in northern India. |
| (Additional interactions) | Selector-identified pairs | The selector may identify other synergistic features (e.g., holiday × is_peak, cloud_delta × wind_speed). |

**Note:** Rather than hardcode all possible interactions (which leads to combinatorial explosion), we let the selector identify the most predictive ones per meter.

---

## 4. Two-Pass Feature Selection — The Breakthrough

This mechanism separated v3's success from v2's failure. It is the single most important architectural decision in the project.

### 4.1 Why Selection Matters

**v2 without selection:** 78 features, 59.6% mean MAPE. Model memorized noise. 57% of meters worse than v1.  
**v3 with selection:** 43 features → ~25 selected, 10.1% mean MAPE. 6× improvement.  
**v4 with selection:** 66 features → ~36 selected, 9.1% mean MAPE. Selection scales.

The selector acts as a **data-driven requirements specification**. Instead of a domain expert hand-picking features for each of 42 (or eventually 500+) meters, the algorithm figures out which features carry signal for each meter.

### 4.2 Pass 1: Screening (Quick Fit)

**Objective:** Rank all 66 candidate features by importance. Eliminate obvious noise.

**Procedure:**
1. Train LightGBM on ALL 66 features for **150 rounds** (early stopping disabled to ensure a full evaluation).
2. Extract native LightGBM feature importance scores (gain-based, built-in to the library).
3. Sort features by importance descending.
4. **Select top 55% of features** (66 × 0.55 ≈ 36 features), **minimum 25 guaranteed** (ensures small meters still get enough features).
5. Output selected feature list.

**Why only 150 rounds?** Quick fit balances speed (73 seconds for all 42 meters) with signal. After 150 rounds, LightGBM has learned the main trends. We don't need the full 800 rounds here.

**Why 55% threshold?** Empirically chosen. Screens out ~30 features that add only noise. If we selected 100% (all 66), we'd keep overfitting features. If we selected <40%, we might lose signal on some meters.

### 4.3 Pass 2: Full Training

**Objective:** Train the final per-meter model on the selected feature subset, with tier-adaptive hyperparameters and proper regularization.

**Procedure:**
1. Train LightGBM for **800 rounds** (up from 150) using ONLY the features selected in Pass 1.
2. Apply **tier-specific hyperparameters** (see Section 5) based on meter's mean demand.
3. Use **early stopping** with patience=50 on held-out validation set (every 4th complete day).
4. Typical early stop occurs around round 400–600 (depends on meter stability).
5. Evaluate on holdout days: MAPE, MAE, MBE, R².
6. Store model, predictions, and feature importances.

**Why 800 rounds?** Sufficient for stable convergence. 600+ is safe margin above typical early-stop point.

**Why early stopping with patience=50?** Prevents overfitting if validation error increases. Patience=50 means "stop if best validation error doesn't improve for 50 consecutive rounds."

### 4.4 Why Two Passes Work

**Pass 1 logic:** LightGBM with all 66 features will find patterns, but some are noise. Importance scores are noisy, so we use a majority threshold (55%) rather than top-K cutoff. This is robust to importance score variance.

**Pass 2 logic:** With only ~36 features, the model has fewer degrees of freedom. It can't memorize noise. Regularization (tier-adaptive lambda_l1, lambda_l2) further constrains the fit. Result: lower validation error → better generalization.

**Critical insight:** The selector doesn't need to be perfect. It just needs to eliminate the worst 30 features. Even a crude 55% cutoff works because overfitting comes from feature count, not from which specific features are kept.

### 4.5 Selection Results per Tier

On the current 42-meter fleet, the selector typically retains:

| Tier | Meters | Typical Selected Features | Mean MAPE |
|------|--------|--------------------------|-----------|
| HT (>5kWh) | 5 | 28–32 | 7.8% |
| Medium/Large | 32 | 34–38 | 9.5% |
| Small (<500Wh) | 5 | 32–36 | 8.1% |

**Note:** Selection is per-meter, not per-tier. The tier just determines hyperparameters. A High-Tension meter might select temperature + lag_48 + cloud_cover. A Small meter might select lag_48 + hour + day_of_week + is_peak (simpler load pattern).

---

## 5. Per-Tier Adaptive Regularization

Different meter tiers have fundamentally different statistical properties. A one-size-fits-all model fails. v4 applies tier-specific hyperparameters.

### 5.1 Tier Definition

Tiers are determined by meter's **mean 30-minute demand** over the training period:

| Tier | Criteria | Count | Typical Meter | Characteristics |
|------|----------|-------|---------------|-----------------|
| **HT** | >5 kWh/30-min | 5 | Manufacturing facility, large commercial | High absolute demand, complex industrial patterns, limited training data (~200 days), high variance |
| **Medium/Large** | 500 Wh — 5 kWh | 32 | Office building, shopping mall, water station | Moderate demand, good signal-to-noise, ~350+ training days available |
| **Small** | <500 Wh | 5 | Small shop, home (rare), agricultural pump | Low absolute demand, weak signal (small variations dominate), high zero-demand rate (40–60% nights), requires sensitive model |

### 5.2 Hyperparameters by Tier

| Hyperparameter | HT (>5kWh) | Medium/Large | Small (<500Wh) | Rationale |
|----------------|------------|-------------|-----------------|-----------|
| `num_leaves` | 31 | 63 | 63 | HT gets fewer leaves → simpler tree → prevent overfitting to 200 training days. Medium/Small have more data; can support deeper trees. |
| `learning_rate` | 0.03 | 0.05 | 0.05 | HT needs slower learning (smaller steps) to stabilize on limited data. Medium/Small can learn faster. |
| `min_child_samples` | 30 | 20 | 15 | HT: each leaf must represent ≥30 samples (coarse granularity). Small: each leaf represents ≥15 samples (finer to extract weak signal). |
| `lambda_l1` (L1 regularization) | 0.1 | 0.05 | 0.02 | **HT gets strongest L1 (feature elimination).** Reduces overfitting. Small gets weakest L1 to preserve signal. |
| `lambda_l2` (L2 regularization) | 0.2 | 0.1 | 0.05 | **HT gets strongest L2 (weight shrinkage).** Prevents extreme leaf values. Small gets weak L2 to allow model flexibility. |
| `max_depth` | Not set (derived from num_leaves) | Not set | Not set | LightGBM derives max_depth from num_leaves. |
| `feature_fraction` | 0.8 | 0.9 | 1.0 | HT: randomly drop 20% of features per tree (reduces overfitting). Small: use all selected features (maximize signal). |

### 5.3 Tier-Specific Rationale

#### HT Meters (5 meters, >5 kWh/30-min)
- **Problem:** Only ~200 training days available. Manufacturing/industrial load is complex (CNC machines, production schedules, maintenance windows).
- **Solution:** Aggressive regularization. Small trees, high lambda, slow learning.
- **Effect:** Model is "pessimistic." It learns broad patterns, not idiosyncratic variations.
- **Result:** ~7.8% MAPE. Lower variance, slightly higher bias, but more stable on new data.

#### Medium/Large Meters (32 meters, 500Wh–5kWh)
- **Problem:** ~350+ training days available. Load pattern is typically commercial/office (steady daytime, low nighttime).
- **Solution:** Balanced regularization. Default LightGBM parameters work well.
- **Effect:** Model can capture nuanced patterns without overfitting.
- **Result:** ~9.5% MAPE. Good generalization.

#### Small Meters (<500Wh, 5 meters)
- **Problem:** Weak signal. 40–60% of night intervals are ~0 demand. Only 50–150Wh typical variation. High zero-demand rate means the model must learn on/off pattern precisely.
- **Solution:** Lightest regularization. More leaves, fast learning, no feature dropping.
- **Effect:** Model is "optimistic." It needs maximal sensitivity to extract signal from 20Wh variations (when demand is usually 0).
- **Result:** ~8.1% MAPE. High relative accuracy despite low absolute demand.

### 5.4 Why Adaptive Regularization Matters

**One-size-fits-all fails:**  
If we used Medium/Large hyperparameters for all meters:
- HT meters overfit (only 200 days, not 350+). MAPE: 15–18%.
- Small meters underfit (regularization too strong). MAPE: 18–22%.

**With adaptive tuning:**  
- HT: 7.8% MAPE (lower because we prevent overfitting).
- Small: 8.1% MAPE (lower because we reduce regularization).

Improvement from tiering: ~2–3 MAPE points fleet-wide.

---

## 6. Training Pipeline — Full Workflow

For each of the 42 eligible meters:

```
Step 1: Data Load
  Load meter data: 30-min intervals, Oct 2024 – Feb 2026 (~550 days)
  Load weather data: Hourly from Open-Meteo, interpolate to 30-min
  Load voltage data: 30-min intervals from APEPDCL MDMS
  
Step 2: Feature Computation
  Compute all 66 candidate features (temporal, lag, rolling, weather, voltage, tariff, interaction)
  Merge onto meter × weather × voltage dataframe
  Handle missing data: forward-fill for 1 missing interval, drop if 2+ consecutive
  
Step 3: Train/Holdout Split
  Strategy: S2 (every 4th complete calendar day is holdout)
  Why: Preserves temporal structure. Doesn't leak future info.
  Result: ~390 training days, ~160 holdout days
  
Step 4: Pass 1 — Screening
  Train LightGBM on all 66 features for 150 rounds
  Extract LightGBM feature importance (gain)
  Select top 55% of features (min 25 guaranteed)
  Time: ~1-2 seconds per meter
  
Step 5: Tier Assignment
  Calculate meter's mean 30-min demand
  Assign to tier: HT (>5kWh), Medium (500Wh–5kWh), Small (<500Wh)
  Look up tier-specific hyperparameters
  
Step 6: Pass 2 — Full Training
  Train LightGBM on selected features, 800 rounds
  Apply tier-specific hyperparameters (num_leaves, learning_rate, lambda_l1/l2, etc.)
  Enable early stopping: patience=50, validation metric=MAE
  Typical early stop: round 400–600
  Time: ~1-2 seconds per meter
  
Step 7: Evaluation
  Predict on holdout days
  Compute metrics: MAPE, MAE, MBE (mean bias error), R²
  Extract per-meter feature importances (from final model)
  Log results to database
  
Step 8: Model Storage
  Save model artifact (LightGBM booster object)
  Save predictions (holdout forecast vs actual)
  Save feature importances
  Save hyperparameters and selection metadata
```

**Total training time:** ~73 seconds for all 42 meters (CPU, no GPU).  
**Inference time:** <10ms per prediction (sub-millisecond per meter).  
**Model size:** ~2–5MB per meter (LightGBM tree structure is compact).

---

## 7. Model Properties

| Property | Value |
|----------|-------|
| **Algorithm** | LightGBM 4.x (gradient boosted decision trees) |
| **Loss function** | L2 (regression, minimize MSE) |
| **Validation metric** | MAE (L1, more robust to outliers than MSE) |
| **Max rounds** | 800 (Pass 2), 150 (Pass 1) |
| **Early stopping** | Patience=50 rounds (stop if no improvement for 50 rounds) |
| **Inference latency** | <10ms per prediction (CPU) |
| **GPU required** | No |
| **Parallelization** | 42 models train sequentially in 73 seconds (could parallelize to ~10 seconds if desired) |
| **Models per fleet** | 42 (one per meter) |
| **Feature interpretability** | Native LightGBM gain importance (sum of splits' information gain per feature) |
| **Reproducibility** | Deterministic. Same data, same seed → same model. |

---

## 8. Key Design Decisions

### 8.1 Per-Meter Models vs. Fleet Model

**Decision:** One independent LightGBM per meter (42 total).

**Why not a single fleet model?**
- 42 meters have vastly different consumption patterns: manufacturing (industrial load, spiky), office (daytime peak, weekday/weekend difference), residential (evening peak), agricultural (monsoon-dependent).
- Shared weights force the model to compromise. A single model optimized for a manufacturing facility will underfit on a residential meter.
- Transfer learning is possible, but we don't have that infrastructure yet. Simpler to train 42 independent models.

**Trade-offs:**
- **Pro:** 42% better accuracy on average (per-meter vs. single shared model). Interpretability (each meter's feature importances are clear).
- **Con:** 42 model artifacts to store and version. Cold-start for new meters (must train from scratch, need ~200 days data).

**Economics:** 73 seconds of training time is negligible. Storage of 42 × 3MB is trivial. The 42x accuracy improvement justifies the approach.

### 8.2 LightGBM over Deep Learning

**Decision:** Use LightGBM (gradient boosted trees), not deep learning.

**Why not Transformer/TFT/PatchTST?**
- Deep learning excels with massive datasets (ImageNet: 1M images). We have 550 days/meter = 26.4K data points per meter. That's small for deep learning.
- LightGBM achieves 4.9% median MAPE on 26.4K points. Temporal Fusion Transformer (SOTA for energy) typically needs 100K+ points to beat tree methods on small datasets.
- Inference: LightGBM is <10ms on CPU. TFT needs GPU for <100ms inference. Deployment complexity.

**Trade-off:**
- **Pro:** Fast training (73 sec), no GPU, interpretable feature importance, proven on small datasets.
- **Con:** Less flexible for multi-horizon forecasting. Weak at learning complex temporal dependencies (e.g., load dynamics across multiple days).

**v5 Plan:** Add TFT/PatchTST for higher-accuracy deep-learning baseline. Current 4.9% MAPE doesn't justify the added complexity, but if operational targets become <3%, deep learning becomes necessary.

### 8.3 Two-Pass Selection over Manual Feature Engineering

**Decision:** Automated two-pass selection, not domain-expert feature curation.

**Why not hand-pick features per meter?**
- Manual curation for 42 meters × 60 candidate features is unsustainable.
- At 100 meters (near-term plan), manual curation breaks down completely.
- The two-pass selector is a data-driven requirements specification. It automatically figures out which features matter per meter.

**Trade-off:**
- **Pro:** Scales to 500+ meters. Objective (importance-based ranking). Transparent (ranked list of features).
- **Con:** Requires tuning the 55% threshold. Some domain knowledge is implicit (e.g., lag_48 is almost always selected, but a domain expert might know that a priori).

**Robustness:** The selector is robust to the 55% threshold. Even at 50% or 60%, results are similar. The key is "don't keep all 66 features."

### 8.4 Tier-Adaptive Regularization over Uniform Hyperparameters

**Decision:** Three sets of hyperparameters, one per tier.

**Why not use the same hyperparameters for all meters?**
- HT meters with 200 training days need different treatment than Medium meters with 350+ days.
- One-size-fits-all hyperparameters leave accuracy on the table. Adaptive tuning gains ~2–3 MAPE points fleet-wide.

**Trade-off:**
- **Pro:** Data-driven tier boundaries (based on mean demand). Simple tier assignment (just compute mean).
- **Con:** Requires manual tuning of 5 hyperparameters × 3 tiers = 15 values. Hyperparameter tuning is not automated (uses grid search + validation).

**Future:** Could use Bayesian optimization to auto-tune per tier, but manual tuning works well at 42 meters.

### 8.5 66 Candidates, Not 112

**Decision:** Distilled feature pipeline from 112 original candidates to 66.

**What we removed:**
- Redundant rolling windows: v1 had roll_mean at 6h, 12h, 18h, 24h, 48h, 72h, 168h. Reduced to 48h and 336h (24h and 7d). Intermediate windows were noise.
- Duplicate weather features: Open-Meteo provides both direct_radiation and shortwave_radiation (GHI). These are 95% correlated. Kept only GHI.
- Complex derived features that didn't rank high: e.g., "cumulative rain in last 7 days" (sparse, only relevant in monsoon). Kept only direct precipitation.

**Rationale:** 66 is the sweet spot. Enough diversity to capture all major signals (temporal, weather, voltage, tariff). Not so many that the selector struggles (diminishing returns). The 66 features are non-redundant and reasonably independent.

**Impact:** Reduced feature space → faster Pass 1 screening (150 rounds on 66 vs. 112), cleaner feature selection, lower overfitting risk.

---

## 9. Production Deployment

### 9.1 Versioning and Rollout

v4 is the **current production candidate**. Deployment procedure:

1. **Staging:** Train on 80% of data. Test holdout MAPE. Gate: Holdout MAPE < 12% on 40+ of 42 meters.
2. **Canary:** Deploy to 10 representative meters. Collect real-world predictions. Compare actual demand vs. forecast for 2 weeks.
3. **Ramp:** If canary metrics acceptable, roll out to all 42 meters in parallel.
4. **Monitoring:** Track actual vs. forecast MAPE weekly. If >15% on any meter, retrain with recent data.

### 9.2 Retraining Cadence

- **Batch retraining:** Full pipeline (data + Pass 1 + Pass 2) monthly or quarterly to incorporate new data, detect concept drift.
- **Online retraining:** Not currently implemented. Would require model update on streaming data (expensive).
- **Trigger retraining:** If 4-week rolling MAPE >15%, retrain immediately.

### 9.3 Cold Start (New Meter)

- **Minimum data:** 200 days of meter + weather history. Without this, statistical estimates are unstable.
- **Bootstrap:** If new meter added mid-month, use v4 ensemble (LightGBM + baseline Prophet) until 200 days accumulated.

---

## 10. Benchmarking and Roadmap

| Metric | v3 | v4 | v5 Target | Notes |
|--------|----|----|-----------|-------|
| Mean MAPE | 10.1% | 9.1% | 7.0% | v5 adds TFT/PatchTST. Estimated 2 MAPE point gain. |
| Median MAPE | 7.6% | 4.9% | 3.5% | Median is harder to move; requires focusing on weak meters. |
| Training time | 85s | 73s | 120s | v5 slower due to DL models; still acceptable. |
| Inference latency | 8ms | <10ms | 50–100ms | v5 uses GPU for inference (batch predictions acceptable). |
| Fleet size | 42 meters | 42 meters | 100+ meters | v5 will scale selection to 100+ meters automatically. |

---

## 11. Known Limitations and Future Work

1. **Holiday Effect:** is_holiday flag uses a static calendar. Real consumption on holidays depends on load type (e.g., residential drops 40%, industrial drops 10%). Could improve with meter-specific holiday coefficients.

2. **Weather Interpolation:** Open-Meteo provides hourly data; we interpolate to 30-min linearly. Non-linear interpolation (e.g., using solar geometry) could help.

3. **Cold-Start:** New meters need 200 days of history. Zero-shot foundation models (Chronos, TimesFM) could help. v5 will explore.

4. **Concept Drift:** If a meter gets a BESS installed or adds/removes major loads (e.g., EV charger), the model's coefficients become stale. Automatic concept drift detection would help.

5. **Uncertainty Quantification:** v4 outputs point forecasts only. Conformal prediction or quantile regression (LightGBM native) could provide prediction intervals. Important for BESS dispatch (need worst-case and best-case bounds).

---

## 12. Reproducibility and Code

**Training code location:** `/edgegrid-forecast-engine/src/train_v4.py`

**Key functions:**
- `compute_features()` — Generate all 66 candidate features
- `pass1_screening()` — LightGBM 150-round quick fit, select top 55%
- `pass2_train()` — LightGBM 800-round full training with tier-adaptive hyperparams
- `evaluate_meter()` — MAPE, MAE, R² on holdout

**Reproducible run:**
```bash
cd /edgegrid-forecast-engine
python src/train_v4.py --data-path data/meters.csv --weather-path data/weather.parquet --seed 42
```

Output: 42 model files, metrics CSV, feature importances JSON.

---

*Last updated April 2026. v4 production deployment in progress. v5 (with deep learning) in design phase.*
