"""
Solar generation forecasting — physics-informed ML hybrid.

Approach:
1. Physics layer (pvlib): Clear-sky irradiance model based on location + time
2. ML layer (LightGBM): Learns cloud/weather adjustments from historical data
3. Hybrid: Physics provides the baseline, ML learns the residual

This outperforms pure ML because:
- Solar generation has strong physics-based structure (sunrise/sunset, panel angle)
- India has distinct monsoon season that purely data-driven models struggle with
- Limited historical data (we may only have months) — physics fills the gap
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class SolarForecastResult:
    """Solar generation forecast output."""
    timestamp: pd.DatetimeIndex
    generation_kwh: np.ndarray
    clear_sky_kwh: np.ndarray       # What generation would be without clouds
    cloud_factor: np.ndarray        # Actual/clear_sky ratio
    capacity_factor: np.ndarray     # Actual/nameplate ratio


def clear_sky_irradiance(
    timestamps: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
) -> pd.Series:
    """
    Calculate clear-sky Global Horizontal Irradiance (GHI) using a simplified model.

    For production, use pvlib.irradiance with Ineichen or DISC models.
    This approximation works well for Indian latitudes (8-35°N).
    """
    hours = timestamps.hour + timestamps.minute / 60
    day_of_year = timestamps.dayofyear

    # Solar declination (simplified)
    declination = 23.45 * np.sin(np.radians((360 / 365) * (day_of_year - 81)))

    # Hour angle
    solar_noon = 12.0 - (longitude - 82.5) / 15  # IST meridian = 82.5°E
    hour_angle = 15 * (hours - solar_noon)

    # Solar elevation
    lat_rad = np.radians(latitude)
    dec_rad = np.radians(declination)
    ha_rad = np.radians(hour_angle)

    sin_elevation = (
        np.sin(lat_rad) * np.sin(dec_rad)
        + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
    )
    elevation = np.degrees(np.arcsin(np.clip(sin_elevation, -1, 1)))

    # Clear-sky GHI (simplified Ineichen model)
    # Peak GHI at sea level ~ 1000 W/m²
    air_mass = np.where(
        elevation > 0,
        1 / np.clip(np.sin(np.radians(elevation)), 0.01, 1),
        0,
    )
    # Atmospheric extinction
    ghi = np.where(
        elevation > 0,
        1000 * np.sin(np.radians(elevation)) * np.exp(-0.185 * air_mass),
        0,
    )

    return pd.Series(np.maximum(ghi, 0), index=timestamps, name="clear_sky_ghi_wm2")


def estimate_solar_generation(
    timestamps: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    capacity_kw: float,
    tilt: Optional[float] = None,
    azimuth: float = 180,  # South-facing
    efficiency: float = 0.20,
    performance_ratio: float = 0.80,
    cloud_cover_pct: Optional[pd.Series] = None,
) -> SolarForecastResult:
    """
    Estimate solar generation combining clear-sky physics with weather.

    Args:
        capacity_kw: Installed DC capacity in kW
        tilt: Panel tilt angle (defaults to latitude)
        azimuth: Panel azimuth (180 = south)
        efficiency: Module efficiency
        performance_ratio: System PR (inverter, wiring, soiling losses)
        cloud_cover_pct: 0-100 cloud cover percentage (from weather API)
    """
    if tilt is None:
        tilt = latitude  # Rule of thumb: tilt ≈ latitude

    # Clear-sky GHI
    clear_sky = clear_sky_irradiance(timestamps, latitude, longitude)

    # Cloud factor
    if cloud_cover_pct is not None:
        # Empirical: generation reduces roughly proportional to cloud cover
        # with some nonlinearity (thin clouds transmit more than thick)
        cloud_factor = 1 - 0.75 * (cloud_cover_pct / 100) ** 1.5
        cloud_factor = np.clip(cloud_factor, 0.1, 1.0)
    else:
        cloud_factor = pd.Series(1.0, index=timestamps)

    # Convert GHI to generation
    # capacity_kw at STC (1000 W/m²). At any given GHI:
    # generation_kw = capacity_kw × (GHI / 1000) × PR × cloud_factor
    generation_kw = capacity_kw * (clear_sky / 1000) * performance_ratio * cloud_factor
    generation_kwh = generation_kw  # Hourly kW = kWh for 1-hour intervals

    clear_sky_kwh = capacity_kw * (clear_sky / 1000) * performance_ratio
    capacity_factor_series = np.where(capacity_kw > 0, generation_kwh / capacity_kw, 0)

    return SolarForecastResult(
        timestamp=timestamps,
        generation_kwh=generation_kwh.values,
        clear_sky_kwh=clear_sky_kwh.values,
        cloud_factor=cloud_factor.values if isinstance(cloud_factor, pd.Series) else np.full(len(timestamps), cloud_factor),
        capacity_factor=capacity_factor_series,
    )


class SolarForecaster:
    """
    ML-enhanced solar forecaster.

    Uses clear-sky irradiance as a feature alongside weather data.
    The ML model learns location-specific corrections:
    - Local shading patterns
    - Soiling effects (dust accumulation)
    - Seasonal cloud patterns specific to the region
    """

    def __init__(self, capacity_kw: float, latitude: float, longitude: float):
        self.capacity_kw = capacity_kw
        self.latitude = latitude
        self.longitude = longitude
        self.model = None

    def fit(self, df: pd.DataFrame, actual_generation_col: str = "solar_kwh"):
        """Train ML correction model on historical generation data."""
        import lightgbm as lgb

        # Add clear-sky feature
        clear_sky = clear_sky_irradiance(
            pd.DatetimeIndex(df["timestamp"]), self.latitude, self.longitude
        )
        df = df.copy()
        df["clear_sky_ghi"] = clear_sky.values

        features = ["clear_sky_ghi", "hour", "month", "day_of_week"]
        weather_cols = ["temperature_c", "cloud_cover_pct", "humidity_pct", "wind_speed_ms"]
        features.extend([c for c in weather_cols if c in df.columns])

        X = df[features].values
        y = df[actual_generation_col].values

        self.feature_names = features
        self.model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6, verbose=-1,
        )
        self.model.fit(X, y)
        logger.info(f"Solar ML model trained on {len(X)} samples")

    def predict(self, df: pd.DataFrame) -> SolarForecastResult:
        """Generate solar forecast using physics + ML hybrid."""
        timestamps = pd.DatetimeIndex(df["timestamp"])

        if self.model is None:
            # Fallback to pure physics model
            return estimate_solar_generation(
                timestamps, self.latitude, self.longitude, self.capacity_kw,
                cloud_cover_pct=df.get("cloud_cover_pct"),
            )

        # Physics baseline
        physics_result = estimate_solar_generation(
            timestamps, self.latitude, self.longitude, self.capacity_kw,
            cloud_cover_pct=df.get("cloud_cover_pct"),
        )

        # ML correction
        clear_sky = clear_sky_irradiance(timestamps, self.latitude, self.longitude)
        df = df.copy()
        df["clear_sky_ghi"] = clear_sky.values

        X = df[self.feature_names].values
        ml_prediction = np.maximum(self.model.predict(X), 0)

        # Blend: 40% physics, 60% ML (ML has learned local corrections)
        blended = 0.4 * physics_result.generation_kwh + 0.6 * ml_prediction

        return SolarForecastResult(
            timestamp=timestamps,
            generation_kwh=blended,
            clear_sky_kwh=physics_result.clear_sky_kwh,
            cloud_factor=physics_result.cloud_factor,
            capacity_factor=np.where(self.capacity_kw > 0, blended / self.capacity_kw, 0),
        )
