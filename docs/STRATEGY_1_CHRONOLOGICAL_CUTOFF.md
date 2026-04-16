# Strategy 1: Chronological Cutoff — Complete Analysis

> Train on first 75% of each meter's timeline. Predict the last 25%.
> The model never sees the future. This is the hardest honest test.

**Run date:** 2026-04-16
**Dataset:** 50 APEPDCL smart meters (5 SP + 45 TP), 30-min intervals
**Eligible meters:** 42 (filtered to >= 180 days of data)
**Model (v1):** LightGBM with early stopping (500 max rounds, patience=30)
**Features (v1):** 31 (temporal + cyclical + 9 lags + 16 rolling stats)
**Model (v3):** LightGBM with two-pass feature selection, 800 max rounds, patience=50
**Features (v3):** 43 candidates → ~25 selected per meter (added weather, holidays, derived)


---

## 0. Version Progression on Strategy 1

This document was originally written for the v1 baseline. The engine has since been iterated through v3 on Strategy 1:

| Version | Features | Selection | Mean MAPE | Median MAPE | Key Change |
|---------|----------|-----------|-----------|-------------|------------|
| **v1** | 18 temporal + lag | None | 55.0% | 42.2% | Baseline (this document's original results) |
| **v2** | 78 (added weather, voltage, derived) | None | 59.6% | 42.2% | Overfitting — 57% of meters regressed |
| **v3** | 43 → ~25 selected | Two-pass | 10.1% | 7.6% | Feature selection breakthrough |

**v1 → v3 represents an 82% reduction in mean MAPE.** The sections below document the v1 baseline results in full. The v3 improvement came from two-pass feature selection (see `docs/V4_ENGINE_ARCHITECTURE.md`) and the addition of weather, holiday, and derived features — but critically, with intelligent selection that prevented the overfitting disaster seen in v2.

**Note:** The current production candidate is v4, which uses Strategy 2 (Stratified Temporal) and achieves 4.9% median MAPE. See `docs/STRATEGY_2_STRATIFIED_TEMPORAL.md` for the latest results.

---

## 1. Why This Strategy

The chronological cutoff answers the most important deployment question: "If we install a meter, train on its history, and forecast forward into time the model has never seen — how accurate are we?"

This is the hardest test because the model cannot rely on having seen similar conditions in the test period. If demand patterns shift between the training and test windows (seasonality, load changes, new equipment), the model must extrapolate from what it learned.

For all 42 meters, the training period covers the earlier months and the test period covers roughly November 2025 through February 2026 (winter). The model trained on spring/summer/monsoon patterns must predict winter behavior it has never observed.


---

## 2. Headline Results

| Metric | Mean | Median | p10 (best) | p90 (worst) |
|--------|------|--------|------------|-------------|
| **MAPE** | 55.0% | 42.1% | 19.7% | 101.9% |
| **MAE** | 346 Wh | 254 Wh | 90 Wh | 721 Wh |
| **MBE** | +133 Wh | +70 Wh | +0 Wh | +372 Wh |
| **MBE %** | +10.5% | +8.7% | -0.4% | +18.7% |

The median MAPE of 42.1% on a pure forward-looking split is an honest but imperfect baseline. The top quartile of meters (p25 MAPE = 24.3%) shows the model works well on predictable loads. The bottom quartile (p75 MAPE = 57.3%) reveals where the model struggles.


---

## 3. Bias Analysis (MBE)

**The model systematically over-forecasts.**

| Direction | Meters | Share |
|-----------|--------|-------|
| Over-forecasting (MBE > 0) | 37 | 88% |
| Under-forecasting (MBE < 0) | 5 | 12% |

Mean absolute MBE as percentage of actual demand: **12.6%**.

This over-forecasting bias has a clear explanation: the training period includes summer months (May-September) when demand in Andhra Pradesh is higher due to cooling loads. The test period is winter (Nov-Feb) when demand drops. The model learned the higher demand level and hasn't adjusted downward.

By demand tier:

| Tier | Meters | Mean MBE % | Interpretation |
|------|--------|-----------|----------------|
| Small (<500 Wh) | 10 | +14.1% | Strongest bias — small loads shift more in percentage terms |
| Medium (500-1.5k Wh) | 27 | +9.7% | Moderate bias — most common tier |
| HT (>5 kWh) | 5 | +7.6% | Lowest bias — large industrial loads are more stable |

**Key insight for EdgeGrid:** MBE tracking reveals that a simple MAPE number hides the directional story. For BESS dispatch optimization, over-forecasting means we'd charge more battery than needed — not catastrophic, but it wastes energy and reduces savings. Adding a bias correction term (subtract the trailing MBE from predictions) could improve this significantly.


---

## 4. Results by Demand Tier

| Tier | Meters | MAPE (median) | MAE (median) | MBE (mean) |
|------|--------|---------------|-------------|-----------|
| **Small (<500 Wh)** | 10 | 27.6% | 96 Wh | +58 Wh |
| **Medium (500-1.5k Wh)** | 27 | 44.5% | 285 Wh | +77 Wh |
| **HT (>5 kWh)** | 5 | 20.4% | 935 Wh | +580 Wh |

Small and HT meters are the most predictable in percentage terms. HT meters have high absolute error (935 Wh MAE) but low MAPE (20.4%) because their base demand is large — the same 1,000 Wh error is 10% of a 10,000 Wh load but 200% of a 500 Wh load.

Medium meters are the hardest — they have enough demand to matter but enough variability to be unpredictable. This is the largest group (27 of 42 meters) and where improvement effort should focus.


---

## 5. Top 10 Performers

| Rank | MSN | Phase | Tier | MAPE | MAE | MBE | MBE% | Test Days |
|------|-----|-------|------|------|-----|-----|------|-----------|
| 1 | 67003694 | 3PH | HT | 15.5% | 840 | +273 | +4.4% | 53 |
| 2 | 67001151 | 3PH | HT | 15.6% | 935 | +512 | +5.4% | 81 |
| 3 | 65045250 | 3PH | Medium | 18.1% | 106 | -0 | -0.0% | 44 |
| 4 | 65021964 | 3PH | Medium | 19.1% | 102 | +9 | +1.6% | 106 |
| 5 | 65012662 | 3PH | Small | 19.7% | 80 | +14 | +3.3% | 108 |
| 6 | 50154700 | 1PH | Medium | 19.7% | 104 | +20 | +3.8% | 54 |
| 7 | 65003102 | 3PH | Small | 20.2% | 72 | +15 | +4.4% | 115 |
| 8 | 67001818 | 3PH | HT | 20.4% | 1269 | +1115 | +16.8% | 81 |
| 9 | 53407938 | 3PH | Medium | 20.6% | 132 | +29 | +3.6% | 59 |
| 10 | 65015739 | 3PH | Small | 20.8% | 67 | +34 | +8.7% | 97 |

Meter 65045250 deserves special attention: 18.1% MAPE with near-zero MBE (-0.0%). This is the model performing as intended — accurate and unbiased.


---

## 6. Bottom 5 Performers

| Rank | MSN | Phase | Tier | MAPE | MAE | MBE | Zero% | Root Cause |
|------|-----|-------|------|------|-----|-----|-------|------------|
| 42 | 65021169 | 3PH | Medium | 313.9% | 575 | +86 | 59.3% | Intermittent load — 59% zeros in test |
| 41 | 67003234 | 3PH | HT | 184.8% | 1762 | +624 | 1.6% | HT meter with volatile demand swings |
| 40 | 65021124 | 3PH | Medium | 136.8% | 285 | +53 | 5.9% | Partial zeros + demand pattern shift |
| 39 | 67003309 | 3PH | HT | 116.2% | 728 | +376 | 0.2% | Short training (171d), demand regime change |
| 38 | 53401842 | 3PH | Medium | 103.0% | 659 | +555 | 0.0% | Massive over-forecast (+74.7% MBE) |

Meter 65021169 is the clearest outlier: 59% zero-demand intervals in the test period. The model trained on active-demand patterns, but the load went mostly idle in winter. MAPE is mathematically inflated when actual values are near zero.

Meter 53401842 has the worst MBE at +74.7% — the model predicts nearly double the actual winter demand. This meter likely had a significant load reduction between the training and test periods.


---

## 7. The Zero-Demand Effect

| Group | Meters | Mean MAPE | Median MAPE |
|-------|--------|-----------|-------------|
| Zeros <= 5% in test | 38 | 46.1% | 40.8% |
| Zeros > 20% in test | 2 | 171.6% | 171.6% |

Intermittent loads inflate MAPE dramatically because any prediction against a zero actual produces infinite percentage error. For meters with high zero rates, MAE is the more meaningful metric.

**Recommendation:** Report MAPE only for meters with <10% zero demand. Use MAE + MBE for intermittent loads.


---

## 8. Feature Importance

| Rank | Feature | Description | Avg Gain |
|------|---------|-------------|----------|
| 1 | lag_48 | Same time yesterday (24h × 2 intervals) | 471B |
| 2 | lag_1 | 30 minutes ago | 320B |
| 3 | lag_24 | 12 hours ago | 123B |
| 4 | lag_96 | Same time 2 days ago | 49B |
| 5 | hour_cos | Cyclical hour encoding | 39B |
| 6 | lag_336 | Same time last week | 38B |
| 7 | hour | Raw hour of day | 15B |
| 8 | lag_2 | 1 hour ago | 8B |

The model is overwhelmingly lag-driven. `lag_48` (same time yesterday) dominates because demand at 2pm today is best predicted by demand at 2pm yesterday. This is expected behavior — but it also explains why the chronological cutoff hurts performance: when winter demand levels differ from summer, yesterday's lag points to the wrong baseline.

**Implication for Strategies 2 & 3:** The stratified approach (which includes winter training days) and rolling origin (which retrains as conditions change) should both outperform chronological precisely because the lags will be calibrated to the right season.


---

## 9. Model Training Details

| Parameter | Value |
|-----------|-------|
| Algorithm | LightGBM (gradient boosted trees) |
| Objective | Regression (L2) |
| Metric | MAE (L1) |
| Max boost rounds | 500 |
| Early stopping patience | 30 rounds |
| Learning rate | 0.05 |
| Num leaves | 63 |
| Feature fraction | 0.8 |
| Bagging fraction | 0.8 |
| Min child samples | 20 |
| L1 regularization (alpha) | 0.1 |
| L2 regularization (lambda) | 0.1 |

Convergence: models typically stop at iteration 104 (median), range 79-161. No model hit the 500-round ceiling, confirming early stopping is working correctly.

Validation: the last 15% of the training set was used as validation for early stopping. This means the effective training data is ~64% of the total timeline (75% × 85%).


---

## 10. Data Summary

| Property | Value |
|----------|-------|
| Source files | SP (t_blp_sp) + TP (t_blp_tp) + Vendor mapping (SCS_PMSGVENDOR) |
| Total meters | 50 (5 single-phase + 45 three-phase) |
| Eligible for benchmark | 42 (>= 180 days) |
| Interval | 30 minutes |
| Date range | Oct 2024 → Feb 2026 |
| Total rows (after dedup) | 717,262 |
| Features per model | 31 |

Phase distribution: 43 three-phase (3PH), 5 single-phase (1PH), 2 three-phase with 4CT.

Demand tiers: 6 HT (>5 kWh/interval), 2 Large (1.5-5k), 31 Medium (500-1.5k), 6 Small (<500).


---

## 11. What This Tells EdgeGrid

1. **The model works.** Median 42% MAPE on a pure forward-looking split across 42 diverse meters is a real baseline. The top 10 meters achieve sub-21% MAPE — proof that for stable loads, LightGBM with lag features is effective.

2. **Bias correction is low-hanging fruit.** 88% of meters over-forecast, with mean bias of +10.5%. A simple trailing-MBE correction (subtract the average recent bias from predictions) could improve accuracy significantly without retraining.

3. **Retraining matters.** The chronological split forces the model to extrapolate from summer into winter. In production, EdgeGrid would retrain monthly — which is exactly what Strategy 3 (Rolling Origin) will test.

4. **Different metrics for different meters.** MAPE works for active loads. MAE + MBE are better for intermittent loads. Report both.

5. **Medium-tier meters need the most attention.** They're the largest group (27/42) and have the highest median MAPE (44.5%). These are the meters where feature enrichment (weather, solar, price signals) would have the most impact.


---

## 12. Full Results Table

| MSN | Phase | Tier | MAPE% | MAE (Wh) | MBE (Wh) | MBE% | Mean Demand | Zero% | Train Days | Test Days | Best Iter |
|-----|-------|------|-------|---------|---------|------|-------------|-------|------------|-----------|-----------|
| 67003694 | 3PH | HT | 15.5 | 840 | +273 | +4.4 | 6156 | 0.0 | 159 | 53 | 138 |
| 67001151 | 3PH | HT | 15.6 | 935 | +512 | +5.4 | 9495 | 0.0 | 239 | 81 | 115 |
| 65045250 | 3PH | Medium | 18.1 | 106 | -0 | -0.0 | 562 | 0.0 | 131 | 44 | 87 |
| 65021964 | 3PH | Medium | 19.1 | 102 | +9 | +1.6 | 572 | 1.8 | 317 | 106 | 161 |
| 65012662 | 3PH | Small | 19.7 | 80 | +14 | +3.3 | 430 | 0.5 | 320 | 108 | 97 |
| 50154700 | 1PH | Medium | 19.7 | 104 | +20 | +3.8 | 520 | 0.6 | 160 | 54 | 101 |
| 65003102 | 3PH | Small | 20.2 | 72 | +15 | +4.4 | 352 | 0.3 | 344 | 115 | 97 |
| 67001818 | 3PH | HT | 20.4 | 1269 | +1115 | +16.8 | 6636 | 0.1 | 242 | 81 | 92 |
| 53407938 | 3PH | Medium | 20.6 | 132 | +29 | +3.6 | 817 | 0.1 | 172 | 59 | 97 |
| 65015739 | 3PH | Small | 20.8 | 67 | +34 | +8.7 | 398 | 0.0 | 288 | 97 | 95 |
| 65022501 | 3PH | Medium | 24.1 | 218 | -4 | -0.4 | 829 | 0.2 | 318 | 106 | 116 |
| 65030697 | 3PH | Small | 24.9 | 103 | +48 | +10.9 | 438 | 0.4 | 213 | 71 | 108 |
| 65024487 | 3PH | Small | 26.0 | 98 | +65 | +17.0 | 380 | 0.2 | 195 | 66 | 135 |
| 50143025 | 1PH | Small | 29.2 | 11 | +4 | +9.1 | 43 | 57.0 | 293 | 99 | 118 |
| 65044028 | 3PH | Medium | 33.5 | 364 | +77 | +5.6 | 1392 | 0.9 | 178 | 60 | 116 |
| 65011155 | 3PH | Medium | 33.6 | 347 | +17 | +1.6 | 1091 | 0.7 | 320 | 107 | 98 |
| 65024185 | 3PH | Small | 33.7 | 187 | +61 | +13.5 | 457 | 0.2 | 184 | 62 | 84 |
| 65041990 | 3PH | Medium | 37.1 | 97 | +11 | +1.7 | 626 | 3.2 | 142 | 48 | 82 |
| 65001891 | 3PH | Medium | 38.1 | 90 | +22 | +3.4 | 668 | 0.2 | 337 | 113 | 155 |
| 65007036 | 3PH | Medium | 38.2 | 361 | -54 | -4.5 | 1199 | 0.2 | 318 | 107 | 95 |
| 65003175 | 3PH | Medium | 41.4 | 338 | -14 | -1.6 | 885 | 0.3 | 344 | 115 | 121 |
| 65023781 | 3PH | Medium | 42.9 | 240 | +101 | +16.1 | 627 | 0.5 | 288 | 97 | 80 |
| 65015026 | 3PH | Small | 43.5 | 165 | +87 | +17.5 | 500 | 0.3 | 275 | 92 | 86 |
| 65004669 | 3PH | Medium | 44.3 | 243 | +111 | +15.2 | 727 | 0.3 | 300 | 101 | 98 |
| 53408407 | 3PH | Medium | 44.5 | 399 | +156 | +14.8 | 1056 | 0.1 | 176 | 59 | 79 |
| 65017058 | 3PH | Medium | 44.7 | 329 | +110 | +12.9 | 852 | 2.5 | 321 | 108 | 109 |
| 51057607 | 1PH | Small | 50.3 | 94 | +10 | +2.1 | 464 | 2.0 | 286 | 96 | 110 |
| 65000228 | 3PH | Medium | 51.7 | 233 | +56 | +9.0 | 616 | 0.3 | 354 | 118 | 114 |
| 65023784 | 3PH | Medium | 52.4 | 292 | +97 | +14.9 | 653 | 0.5 | 288 | 97 | 119 |
| 65002231 | 3PH | Medium | 53.5 | 410 | +102 | +12.3 | 824 | 0.3 | 344 | 115 | 104 |
| 53408951 | 3PH | Medium | 54.1 | 241 | +76 | +10.9 | 696 | 0.0 | 173 | 58 | 95 |
| 65022302 | 3PH | Medium | 58.3 | 174 | +75 | +14.6 | 511 | 0.4 | 264 | 88 | 103 |
| 65049719 | 3PH | Medium | 61.9 | 558 | +339 | +34.9 | 970 | 0.0 | 173 | 58 | 113 |
| 65025443 | 3PH | Medium | 66.1 | 357 | +147 | +18.7 | 783 | 0.0 | 242 | 82 | 109 |
| 50186364 | 1PH | Medium | 72.8 | 266 | -212 | -38.6 | 548 | 10.3 | 171 | 58 | 140 |
| 53401885 | 3PH | Medium | 73.4 | 312 | +125 | +21.3 | 586 | 2.4 | 244 | 82 | 95 |
| 53403416 | 3PH | Small | 91.8 | 303 | +240 | +54.1 | 444 | 0.8 | 259 | 87 | 86 |
| 53401842 | 3PH | Medium | 103.0 | 659 | +555 | +74.7 | 743 | 0.0 | 249 | 84 | 105 |
| 67003309 | 3PH | HT | 116.2 | 728 | +376 | +4.5 | 8260 | 0.2 | 171 | 57 | 111 |
| 65021124 | 3PH | Medium | 136.8 | 285 | +53 | +6.6 | 805 | 5.9 | 322 | 108 | 113 |
| 67003234 | 3PH | HT | 184.8 | 1762 | +624 | +6.9 | 9050 | 1.6 | 178 | 60 | 100 |
| 65021169 | 3PH | Medium | 313.9 | 575 | +86 | +8.3 | 1040 | 59.3 | 323 | 108 | 105 |
