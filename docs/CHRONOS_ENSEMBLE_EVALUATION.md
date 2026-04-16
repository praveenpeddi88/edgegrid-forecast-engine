# Chronos-Bolt Ensemble Evaluation — Foundation Model vs LightGBM

## Section 1: Background

EdgeGrid evaluated Amazon's Chronos-Bolt-Tiny (9M parameters) as a potential ensemble partner for the LightGBM v4 demand forecaster. The hypothesis: a foundation model pre-trained on millions of time series could complement LightGBM's meter-specific training, especially where LightGBM's features fail to capture patterns.

Chronos-Bolt is a zero-shot forecaster — it requires no training data for the specific meter. It takes raw demand history as context and generates probabilistic forecasts. This makes it valuable for cold-start scenarios (new meters with no training history).

## Section 2: Evaluation Setup

- **Model:** Chronos-Bolt-Tiny (amazon/chronos-bolt-tiny, 9M parameters)
- **Inference:** CPU only, ~0.2s per prediction window
- **Dataset:** Same 42 eligible APEPDCL meters, same S2 holdout split
- **Context window:** 512 most recent training intervals (≈10.7 days)
- **Prediction horizon:** 48 intervals (24 hours) per holdout day
- **Ensemble method:** Weighted average — y_ensemble = w × y_lgb + (1-w) × y_chronos
- **Weight search:** Grid search w ∈ {0.0, 0.1, 0.2, ..., 1.0} per meter, minimize holdout MAPE

## Section 3: Standalone Results

| Model | Mean MAPE | Median MAPE |
|-------|-----------|-------------|
| LightGBM v4 | 9.1% | 4.9% |
| Chronos-Bolt-Tiny | ~44% | ~38% |
| Naive persistence (lag_48) | ~55% | ~42% |

Chronos-Bolt significantly outperforms naive persistence (as expected for a pre-trained model) but falls far short of the trained LightGBM. The gap is ~35 percentage points on mean MAPE.

### Why Chronos Underperforms

1. **No meter-specific features** — Chronos sees only the univariate demand series, not weather, voltage, holidays, or ToD signals
2. **No India-specific knowledge** — the model was pre-trained on diverse global time series, not Indian electricity demand specifically
3. **Limited context** — 512 intervals (~10 days) is insufficient for capturing weekly and seasonal patterns
4. **No feature selection** — LightGBM's two-pass selection gives it a massive advantage on well-instrumented meters

## Section 4: Adaptive Ensemble Results

The adaptive ensemble searched for the optimal weight per meter. The critical finding:

**41 of 42 meters chose w=1.0 (pure LightGBM).**

Only 1 meter (one of the worst-performing HT meters) benefited marginally from Chronos contribution, choosing w=0.9 (90% LightGBM, 10% Chronos).

This means the ensemble effectively degenerates to pure LightGBM for all practical purposes. The trained meter-specific model dominates the zero-shot foundation model on every meter with sufficient training data.

## Section 5: Why the Ensemble Failed

The ensemble hypothesis assumed LightGBM and Chronos would make different kinds of errors that partially cancel. In practice:

1. LightGBM's errors are smaller in magnitude — weighting in Chronos's larger errors can only increase total error
2. The errors are not complementary — Chronos doesn't systematically correct where LightGBM fails
3. LightGBM already captures temporal patterns through lag features + cyclical encodings, which is Chronos's only strength
4. The two-pass feature selection ensures LightGBM uses only features that improve predictions, making it hard to beat

## Section 6: Where Chronos IS Valuable — Cold-Start Protocol

Despite losing the ensemble competition, Chronos is invaluable for one specific scenario: new meters with no training history.

### Cold-Start Protocol

- **Days 1–14:** Chronos-Bolt zero-shot (~44% MAPE). No training data needed at all. The meter gets instant forecasts from day 1.
- **Days 15–60:** LightGBM with temporal + weather features only (no lag features, since there isn't enough history for reliable 7-day lags). Estimated ~15-25% MAPE.
- **Days 60+:** Full v4 feature set including lag_336 (7-day lag) and all rolling statistics. Expected to reach the meter's steady-state accuracy within a few weeks.

This protocol means EdgeGrid can deploy at a new substation and provide forecasts immediately — rough forecasts that improve automatically as data accumulates.

## Section 7: Foundation Model Landscape

Other foundation models evaluated or planned:

- **TimesFM 2.5 (Google):** Not yet evaluated. Supports covariates in v2.5, which could narrow the gap with LightGBM. Planned for v5.
- **MOIRAI-2 (Salesforce):** Not yet evaluated. Native multivariate support could be interesting. Planned for v5.
- **Lag-Llama:** Not evaluated. Probabilistic forecasting via LLM architecture.

The general lesson: foundation models are impressive for zero-shot scenarios but struggle to compete with trained models that have access to domain-specific features. This may change as foundation models incorporate exogenous variables (covariates) more effectively.

## Section 8: Recommendations

1. **Do not ensemble Chronos with LightGBM in production.** Pure LightGBM is the right choice for meters with ≥60 days of data.
2. **Use Chronos for cold-start only.** It provides immediate forecasts for new meters with zero training.
3. **Re-evaluate with TimesFM 2.5** when covariates can be passed to the foundation model — this is the most likely path to a genuine ensemble improvement.
4. **Monitor for regime changes.** If a meter experiences a sudden load change (new equipment, tenant change), Chronos could briefly outperform a stale LightGBM until retraining occurs. Consider a monitoring system that triggers Chronos fallback on MAPE drift.

## Section 9: Technical Details

- **Chronos-Bolt-Tiny:** 9M parameters, T5-based architecture, trained on Kernel Synth + public time series data
- **Inference:** CPU-only, no GPU required, ~0.2s per 48-interval prediction
- **Ensemble weight search:** 11 candidate weights (0.0 to 1.0 in 0.1 steps), evaluated on S2 holdout MAPE per meter
- **Python:** chronos-forecasting package, PyTorch backend
