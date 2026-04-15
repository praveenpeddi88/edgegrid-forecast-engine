"""
Demand forecast model training pipeline.

Trains LightGBM models in two configurations to measure the impact of
weather features on forecast accuracy:

1. Baseline: temporal + lag + rolling + price features only (~50 features)
2. Enriched: + weather + solar + air quality + interactions (~85 features)

Usage:
    python -m edgegrid_forecast.training.train_demand

Output:
    models/demand/     — saved model artifacts
    Prints comparison table showing MAPE improvement from weather features.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edgegrid_forecast.data.features import (
    add_temporal_features,
    add_lag_features,
    add_rolling_features,
    add_price_features,
    add_consumption_pattern_features,
    add_weather_features,
    add_solar_features,
    add_air_quality_features,
    add_weather_demand_interactions,
    build_forecast_features,
)
from edgegrid_forecast.data.synthetic import generate_all_demand
from edgegrid_forecast.models.demand import LightGBMDemandForecaster

# ─── Paths ──────────────────────────────────────────────────────────────────

DATA_DIR = PROJECT_ROOT / "data"
WEATHER_DIR = DATA_DIR / "external" / "weather"
AQ_DIR = DATA_DIR / "external" / "air_quality"
NASA_DIR = DATA_DIR / "external" / "nasa_power"
MODEL_DIR = PROJECT_ROOT / "models" / "demand"


def load_and_merge_data() -> pd.DataFrame:
    """
    Load weather, air quality, and demand data, merge on timestamp + location.

    Returns a single DataFrame with all columns ready for feature engineering.
    """
    logger.info("Loading external data...")

    # Weather + solar (from Open-Meteo)
    weather = pd.read_parquet(WEATHER_DIR / "all_locations_fy2425.parquet")
    logger.info(f"  Weather: {len(weather):,} rows, {len(weather.columns)} cols")

    # Air quality
    aq = pd.read_parquet(AQ_DIR / "all_locations_fy2425.parquet")
    logger.info(f"  Air quality: {len(aq):,} rows, {len(aq.columns)} cols")

    # Generate synthetic demand
    demand = generate_all_demand(str(WEATHER_DIR))
    logger.info(f"  Demand: {len(demand):,} rows")

    # Merge weather → demand (on timestamp + location)
    merged = demand.merge(
        weather,
        on=["timestamp", "location_id"],
        how="left",
        suffixes=("", "_weather"),
    )

    # Merge air quality → merged (on timestamp + location)
    aq_cols = ["timestamp", "location_id", "pm2_5", "pm10", "dust", "aerosol_optical_depth"]
    aq_subset = aq[aq_cols].copy()
    merged = merged.merge(
        aq_subset,
        on=["timestamp", "location_id"],
        how="left",
    )

    # Drop duplicate coordinate columns from merge
    for col in ["latitude_weather", "longitude_weather", "elevation_m_weather"]:
        if col in merged.columns:
            merged.drop(columns=[col], inplace=True)

    logger.info(f"  Merged dataset: {len(merged):,} rows, {len(merged.columns)} cols")
    return merged


def train_baseline_model(df: pd.DataFrame) -> dict:
    """
    Train model WITHOUT weather features — temporal + lag + rolling + price only.

    This establishes the baseline MAPE to beat.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING BASELINE MODEL (no weather features)")
    logger.info("=" * 60)

    # Build features WITHOUT weather
    featured = build_forecast_features(
        df.copy(),
        target_col="demand_kwh",
        include_weather=False,
        include_solar=False,
        include_air_quality=False,
    )

    model = LightGBMDemandForecaster(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
    )

    metrics = model.fit(featured, target_col="demand_kwh")

    # Cross-validate for robust estimate
    cv_metrics = model.cross_validate(featured, target_col="demand_kwh", n_splits=5)

    logger.info(f"\nBaseline CV Results:")
    logger.info(f"  MAPE: {cv_metrics['mape']:.2f}% ± {cv_metrics['mape_std']:.2f}%")
    logger.info(f"  MAE:  {cv_metrics['mae']:.2f} ± {cv_metrics['mae_std']:.2f}")
    logger.info(f"  RMSE: {cv_metrics['rmse']:.2f} ± {cv_metrics['rmse_std']:.2f}")

    return {
        "model": model,
        "val_metrics": metrics,
        "cv_metrics": cv_metrics,
        "n_features": len(model.feature_names),
    }


def train_enriched_model(df: pd.DataFrame) -> dict:
    """
    Train model WITH weather + solar + air quality features.

    This should show significant MAPE improvement from weather correlation.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING ENRICHED MODEL (with weather features)")
    logger.info("=" * 60)

    # Build features WITH weather
    featured = build_forecast_features(
        df.copy(),
        target_col="demand_kwh",
        include_weather=True,
        include_solar=True,
        include_air_quality=True,
    )

    model = LightGBMDemandForecaster(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
    )

    metrics = model.fit(featured, target_col="demand_kwh")

    # Cross-validate
    cv_metrics = model.cross_validate(featured, target_col="demand_kwh", n_splits=5)

    logger.info(f"\nEnriched CV Results:")
    logger.info(f"  MAPE: {cv_metrics['mape']:.2f}% ± {cv_metrics['mape_std']:.2f}%")
    logger.info(f"  MAE:  {cv_metrics['mae']:.2f} ± {cv_metrics['mae_std']:.2f}")
    logger.info(f"  RMSE: {cv_metrics['rmse']:.2f} ± {cv_metrics['rmse_std']:.2f}")

    # Top features
    logger.info(f"\nTop 20 Features by Importance:")
    for _, row in model.feature_importance.head(20).iterrows():
        logger.info(f"  {row['importance']:6.0f}  {row['feature']}")

    return {
        "model": model,
        "val_metrics": metrics,
        "cv_metrics": cv_metrics,
        "n_features": len(model.feature_names),
    }


def train_multistep_comparison(df: pd.DataFrame) -> dict:
    """
    Compare models in the REAL production scenario: 24h-ahead forecasting
    where demand lag features are NOT available (only weather forecasts are).

    This is where weather features truly shine — when you can't peek at
    recent demand but DO have weather forecast data from Open-Meteo.
    """
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
    import lightgbm as lgb

    logger.info("\n" + "=" * 60)
    logger.info("MULTI-STEP AHEAD (24h) — Production Scenario")
    logger.info("=" * 60)
    logger.info("  Simulating: no demand lags < 24h, only weather available")

    # Short lags excluded — only 24h, 48h, 168h available (yesterday+)
    MULTISTEP_LAGS = [24, 48, 168]
    MULTISTEP_WINDOWS = [24, 48, 168]

    # --- Model A: Temporal + limited lags only (no weather) ---
    logger.info("\n  Model A: Temporal + limited lags (no weather)")
    df_a = build_forecast_features(
        df.copy(),
        target_col="demand_kwh",
        lag_hours=MULTISTEP_LAGS,
        rolling_windows=MULTISTEP_WINDOWS,
        include_weather=False,
        include_solar=False,
        include_air_quality=False,
    )

    model_a = LightGBMDemandForecaster(n_estimators=800, learning_rate=0.05, max_depth=7)
    metrics_a = model_a.fit(df_a, target_col="demand_kwh")
    cv_a = model_a.cross_validate(df_a, target_col="demand_kwh", n_splits=5)

    # --- Model B: Temporal + limited lags + weather + solar + AQ ---
    logger.info("\n  Model B: Temporal + limited lags + WEATHER")
    df_b = build_forecast_features(
        df.copy(),
        target_col="demand_kwh",
        lag_hours=MULTISTEP_LAGS,
        rolling_windows=MULTISTEP_WINDOWS,
        include_weather=True,
        include_solar=True,
        include_air_quality=True,
    )

    model_b = LightGBMDemandForecaster(n_estimators=800, learning_rate=0.05, max_depth=7)
    metrics_b = model_b.fit(df_b, target_col="demand_kwh")
    cv_b = model_b.cross_validate(df_b, target_col="demand_kwh", n_splits=5)

    # --- Comparison ---
    mape_improvement = cv_a["mape"] - cv_b["mape"]
    mape_pct = (mape_improvement / cv_a["mape"]) * 100 if cv_a["mape"] > 0 else 0

    logger.info(f"\n{'='*60}")
    logger.info("24h-AHEAD COMPARISON (production-realistic)")
    logger.info(f"{'='*60}")
    logger.info(f"\n{'Metric':<20} {'No Weather':>12} {'+ Weather':>12} {'Improvement':>14}")
    logger.info("-" * 60)
    logger.info(f"{'Features':<20} {len(model_a.feature_names):>12d} {len(model_b.feature_names):>12d} {len(model_b.feature_names) - len(model_a.feature_names):>+13d}")
    logger.info(f"{'CV MAPE (%)':<20} {cv_a['mape']:>11.2f}% {cv_b['mape']:>11.2f}% {mape_improvement:>+12.2f}%")
    logger.info(f"{'CV MAE (kWh)':<20} {cv_a['mae']:>12.2f} {cv_b['mae']:>12.2f} {cv_a['mae'] - cv_b['mae']:>+13.2f}")
    logger.info(f"{'CV RMSE (kWh)':<20} {cv_a['rmse']:>12.2f} {cv_b['rmse']:>12.2f} {cv_a['rmse'] - cv_b['rmse']:>+13.2f}")
    logger.info(f"{'Val MAPE (%)':<20} {metrics_a['mape']:>11.2f}% {metrics_b['mape']:>11.2f}%")
    logger.info(f"{'Val R²':<20} {metrics_a['r2']:>12.4f} {metrics_b['r2']:>12.4f}")
    logger.info(f"\n  Weather features reduced 24h-ahead MAPE by {mape_pct:.1f}% relative")

    # Top weather features in Model B
    weather_features = model_b.feature_importance[
        model_b.feature_importance["feature"].str.contains(
            "cdh|hdh|heat_index|comfort|temp_|cloud|rain|wind|ghi|solar|dni|diffuse|"
            "pm2|aod|dust|soiling|pressure|rh_|humidity|extreme_heat|overcast",
            regex=True
        )
    ]
    if not weather_features.empty:
        logger.info(f"\n  Top Weather Features in 24h-ahead Model:")
        for _, row in weather_features.head(15).iterrows():
            logger.info(f"    {row['importance']:6.0f}  {row['feature']}")

    return {
        "no_weather": {"val": metrics_a, "cv": cv_a, "n_features": len(model_a.feature_names)},
        "with_weather": {"val": metrics_b, "cv": cv_b, "n_features": len(model_b.feature_names), "model": model_b},
        "mape_reduction_pct": mape_pct,
    }


def print_comparison(baseline: dict, enriched: dict):
    """Print side-by-side comparison of baseline vs enriched model."""
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON: Baseline vs Weather-Enriched")
    logger.info("=" * 60)

    b_cv = baseline["cv_metrics"]
    e_cv = enriched["cv_metrics"]

    mape_improvement = b_cv["mape"] - e_cv["mape"]
    mape_pct = (mape_improvement / b_cv["mape"]) * 100

    mae_improvement = b_cv["mae"] - e_cv["mae"]
    rmse_improvement = b_cv["rmse"] - e_cv["rmse"]

    logger.info(f"\n{'Metric':<20} {'Baseline':>12} {'Enriched':>12} {'Improvement':>14}")
    logger.info("-" * 60)
    logger.info(f"{'Features':<20} {baseline['n_features']:>12d} {enriched['n_features']:>12d} {enriched['n_features'] - baseline['n_features']:>+13d}")
    logger.info(f"{'MAPE (%)':<20} {b_cv['mape']:>11.2f}% {e_cv['mape']:>11.2f}% {mape_improvement:>+12.2f}%")
    logger.info(f"{'MAE (kWh)':<20} {b_cv['mae']:>12.2f} {e_cv['mae']:>12.2f} {mae_improvement:>+13.2f}")
    logger.info(f"{'RMSE (kWh)':<20} {b_cv['rmse']:>12.2f} {e_cv['rmse']:>12.2f} {rmse_improvement:>+13.2f}")
    logger.info(f"\n  Weather features reduced MAPE by {mape_pct:.1f}% relative")

    # Validation set metrics
    logger.info(f"\nValidation Set (last 20%):")
    logger.info(f"  Baseline: MAPE={baseline['val_metrics']['mape']:.2f}%, R²={baseline['val_metrics']['r2']:.4f}")
    logger.info(f"  Enriched: MAPE={enriched['val_metrics']['mape']:.2f}%, R²={enriched['val_metrics']['r2']:.4f}")


if __name__ == "__main__":
    logger.info("EdgeGrid Forecast Engine — Demand Model Training")
    logger.info(f"Project root: {PROJECT_ROOT}")

    # Load and merge all data
    df = load_and_merge_data()

    # Train baseline (no weather)
    baseline = train_baseline_model(df)

    # Train enriched (with weather)
    enriched = train_enriched_model(df)

    # Save enriched model (production model)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    enriched["model"].save(MODEL_DIR)
    logger.info(f"\nProduction model saved to {MODEL_DIR}")

    # Print comparison
    print_comparison(baseline, enriched)

    # The real test: 24h-ahead production scenario
    multistep = train_multistep_comparison(df)

    logger.info("\nDone.")
