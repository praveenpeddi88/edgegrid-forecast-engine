"""
Feature engineering for demand, solar, and price forecasting.

Creates 80+ features across 8 families that capture the patterns in Indian C&I
electricity consumption at APEPDCL substations:

Feature Families:
1. Temporal      — hour, dow, month, cyclical encoding, season, holiday, business hour
2. Lag           — 1,2,3,6,12,24,48,168h target lags
3. Rolling       — 3,6,12,24,48,168h mean/std/min/max windows
4. Price         — IEX landed cost, price-above-mean, ToD multiplier
5. Consumption   — daily range ratio, hourly share, deviation from typical
6. Weather       — temperature, humidity, wind, cloud, rain, pressure, CDD/HDD, comfort
7. Solar         — GHI, DNI, DHI, clear-sky ratio, solar availability index
8. Air Quality   — PM2.5, PM10, dust, AOD, soiling loss proxy

Weather features are the biggest lever for C&I demand — summer cooling load
in coastal AP adds 30-40% to base demand. Solar features feed the generation
forecast. Air quality affects PV soiling losses (1-3% in coastal AP).
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..utils.constants import get_tod_multiplier, get_landed_price


# ─── Indian Holiday Calendar ─────────────────────────────────────────────────
# Major holidays that affect C&I consumption in AP/Telangana

INDIAN_HOLIDAYS_2025_2026 = {
    # Format: (month, day): name
    (1, 14): "Makar Sankranti",
    (1, 26): "Republic Day",
    (3, 14): "Holi",
    (4, 6): "Ugadi",
    (4, 14): "Ambedkar Jayanti",
    (5, 1): "May Day",
    (8, 15): "Independence Day",
    (8, 27): "Janmashtami",
    (9, 5): "Vinayaka Chaturthi",
    (10, 2): "Gandhi Jayanti",
    (10, 12): "Dussehra",
    (10, 20): "Diwali",
    (11, 1): "AP Formation Day",
    (12, 25): "Christmas",
}


def is_holiday(dt: pd.Timestamp) -> bool:
    """Check if a date is an Indian public holiday."""
    return (dt.month, dt.day) in INDIAN_HOLIDAYS_2025_2026


def add_temporal_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Add time-based features that capture daily, weekly, and seasonal patterns.
    """
    ts = pd.to_datetime(df[timestamp_col])

    df = df.copy()

    # Basic temporal
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek  # 0=Monday
    df["day_of_month"] = ts.dt.day
    df["month"] = ts.dt.month
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    # Holiday features
    df["is_holiday"] = ts.apply(is_holiday).astype(int)
    df["is_working_day"] = ((~df["is_weekend"].astype(bool)) & (~df["is_holiday"].astype(bool))).astype(int)

    # Cyclical encoding (captures wrap-around: hour 23 is close to hour 0)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Time-of-day tariff period
    df["tod_multiplier"] = df["hour"].apply(get_tod_multiplier)
    df["is_peak_hour"] = df["tod_multiplier"].apply(lambda x: 1 if x > 1 else 0)

    # Business hour indicator (9am-6pm weekday)
    df["is_business_hour"] = (
        (df["hour"] >= 9) & (df["hour"] < 18) & (df["is_working_day"] == 1)
    ).astype(int)

    # Season (Indian power sector perspective)
    df["season"] = df["month"].map({
        1: "winter", 2: "winter", 3: "summer",
        4: "summer", 5: "summer", 6: "monsoon",
        7: "monsoon", 8: "monsoon", 9: "monsoon",
        10: "post_monsoon", 11: "post_monsoon", 12: "winter",
    })

    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lag_hours: List[int] = [1, 2, 3, 6, 12, 24, 48, 168],
    group_col: Optional[str] = "consumer_id",
) -> pd.DataFrame:
    """
    Add lagged features. Key lags for energy demand:
    - 1-3h: Short-term persistence
    - 6-12h: Half-day pattern
    - 24h: Same hour yesterday
    - 48h: Same hour 2 days ago
    - 168h: Same hour last week (strongest predictor for C&I)
    """
    df = df.copy()

    if group_col and group_col in df.columns:
        for lag in lag_hours:
            df[f"{target_col}_lag_{lag}h"] = df.groupby(group_col)[target_col].shift(lag)
    else:
        for lag in lag_hours:
            df[f"{target_col}_lag_{lag}h"] = df[target_col].shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: List[int] = [3, 6, 12, 24, 48, 168],
    group_col: Optional[str] = "consumer_id",
) -> pd.DataFrame:
    """
    Add rolling window statistics.
    Captures trend, volatility, and regime changes.
    """
    df = df.copy()

    for window in windows:
        if group_col and group_col in df.columns:
            grouped = df.groupby(group_col)[target_col]
            df[f"{target_col}_rmean_{window}h"] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f"{target_col}_rstd_{window}h"] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            df[f"{target_col}_rmin_{window}h"] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).min()
            )
            df[f"{target_col}_rmax_{window}h"] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )
        else:
            df[f"{target_col}_rmean_{window}h"] = (
                df[target_col].rolling(window, min_periods=1).mean()
            )
            df[f"{target_col}_rstd_{window}h"] = (
                df[target_col].rolling(window, min_periods=1).std()
            )

    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add IEX price and landed cost features for each timestep."""
    df = df.copy()

    if "month" in df.columns and "hour" in df.columns:
        df["iex_price"] = df.apply(
            lambda r: get_landed_price(int(r["month"]), int(r["hour"])),
            axis=1,
        )
        # Price momentum (useful for dispatch decisions)
        df["price_above_mean"] = (
            df["iex_price"] > df.groupby("month")["iex_price"].transform("mean")
        ).astype(int)

    return df


def add_consumption_pattern_features(
    df: pd.DataFrame,
    target_col: str = "demand_kwh",
    group_col: str = "consumer_id",
) -> pd.DataFrame:
    """
    Add features that capture consumer-specific consumption patterns.

    These are the features that differentiate a hospital (24/7 flat) from
    an office building (9-6 weekday peak) from a factory (shift-based).
    """
    df = df.copy()

    if group_col in df.columns:
        # Peak-to-trough ratio (load factor proxy)
        daily_max = df.groupby([group_col, df["timestamp"].dt.date])[target_col].transform("max")
        daily_min = df.groupby([group_col, df["timestamp"].dt.date])[target_col].transform("min")
        df["daily_range_ratio"] = np.where(
            daily_max > 0,
            (daily_max - daily_min) / daily_max,
            0,
        )

        # Hour's share of daily total (normalized load shape)
        daily_total = df.groupby([group_col, df["timestamp"].dt.date])[target_col].transform("sum")
        df["hourly_share"] = np.where(daily_total > 0, df[target_col] / daily_total, 0)

        # Deviation from consumer's typical hourly pattern
        hourly_mean = df.groupby([group_col, "hour"])[target_col].transform("mean")
        hourly_std = df.groupby([group_col, "hour"])[target_col].transform("std")
        df["deviation_from_typical"] = np.where(
            hourly_std > 0,
            (df[target_col] - hourly_mean) / hourly_std,
            0,
        )

    return df


# ─── Weather Feature Engineering ────────────────────────────────────────────

# Cooling/heating degree thresholds for coastal AP (°C)
# 26°C is the comfort threshold — above this, AC kicks in hard
CDD_BASE_TEMP = 26.0
HDD_BASE_TEMP = 18.0


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weather-derived features from Open-Meteo data.

    Key domain insight for coastal AP:
    - Temperature is THE dominant demand driver (cooling load above 26°C)
    - Humidity amplifies thermal discomfort (heat index)
    - Cloud cover suppresses solar but reduces cooling slightly
    - Wind speed affects both comfort and PV generation (cooling effect)
    - Rain signals monsoon (lower demand, lower solar, higher humidity)
    """
    df = df.copy()

    # ── Direct weather features (already in the dataframe if merged) ──────
    # These columns come from Open-Meteo weather data:
    # temperature_2m, relative_humidity_2m, wind_speed_10m,
    # cloud_cover, precipitation, surface_pressure

    if "temperature_2m" in df.columns:
        temp = df["temperature_2m"]

        # Cooling Degree Hours (CDH) — demand proxy above comfort threshold
        # This is the single most predictive weather feature for C&I in AP
        df["cdh"] = np.maximum(temp - CDD_BASE_TEMP, 0)
        df["cdh_squared"] = df["cdh"] ** 2  # Nonlinear cooling load (AC COP drops)

        # Heating Degree Hours — rare in AP but captures winter morning demand
        df["hdh"] = np.maximum(HDD_BASE_TEMP - temp, 0)

        # Temperature change rate — ramp-up/down signals HVAC startup
        df["temp_delta_1h"] = temp.diff(1)
        df["temp_delta_3h"] = temp.diff(3)

        # Rolling temperature features — captures heat accumulation
        df["temp_rmean_6h"] = temp.rolling(6, min_periods=1).mean()
        df["temp_rmean_24h"] = temp.rolling(24, min_periods=1).mean()
        df["temp_rmax_24h"] = temp.rolling(24, min_periods=1).max()

        # Is it hot? (above 35°C — extreme in AP, AC on full blast)
        df["is_extreme_heat"] = (temp > 35).astype(int)

    if "relative_humidity_2m" in df.columns:
        rh = df["relative_humidity_2m"]

        # Humidity affects perceived temperature — drives AC harder
        df["rh_above_70"] = (rh > 70).astype(int)  # Uncomfortable humidity

        # Heat index proxy (simplified Steadman formula)
        if "temperature_2m" in df.columns:
            t = df["temperature_2m"]
            # Simplified heat index — good enough for feature, not for medical advice
            df["heat_index"] = (
                -8.785 + 1.611 * t + 2.339 * rh - 0.146 * t * rh
                - 0.013 * t**2 - 0.016 * rh**2
                + 0.002 * t**2 * rh + 0.001 * t * rh**2
                - 3.58e-6 * t**2 * rh**2
            )
            # Comfort-adjusted CDH — heat index above threshold
            df["comfort_cdh"] = np.maximum(df["heat_index"] - CDD_BASE_TEMP, 0)

    if "wind_speed_10m" in df.columns:
        # Wind chill reduces effective temperature slightly
        df["wind_above_5ms"] = (df["wind_speed_10m"] > 5).astype(int)

    if "cloud_cover" in df.columns:
        # Cloud cover as fraction (0-1) for easier multiplication
        df["cloud_fraction"] = df["cloud_cover"] / 100.0
        # Overcast indicator (>80% cloud)
        df["is_overcast"] = (df["cloud_cover"] > 80).astype(int)

    if "precipitation" in df.columns:
        # Rain indicator — affects both demand and solar
        df["is_raining"] = (df["precipitation"] > 0.1).astype(int)
        # Heavy rain (>5mm/h) — significant solar impact
        df["is_heavy_rain"] = (df["precipitation"] > 5.0).astype(int)

    if "surface_pressure" in df.columns:
        # Pressure change as weather front proxy
        df["pressure_delta_3h"] = df["surface_pressure"].diff(3)

    return df


def add_solar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add solar radiation features from Open-Meteo + NASA POWER data.

    These feed directly into PV generation forecasting and affect the
    BESS charge/discharge strategy (solar surplus → charge battery).

    Key signals:
    - GHI: Total radiation on horizontal surface (primary PV driver)
    - DNI: Direct component (matters for tracking systems)
    - DHI: Diffuse component (overcast generation)
    - Clear-sky ratio: GHI vs theoretical clear-sky (cloud attenuation)
    """
    df = df.copy()

    if "shortwave_radiation" in df.columns:
        ghi = df["shortwave_radiation"]

        # Solar availability — is the sun producing?
        df["solar_producing"] = (ghi > 10).astype(int)  # >10 W/m² threshold
        df["solar_peak"] = (ghi > 500).astype(int)  # Strong solar

        # GHI rolling features — captures cloud transients
        df["ghi_rmean_3h"] = ghi.rolling(3, min_periods=1).mean()
        df["ghi_rstd_3h"] = ghi.rolling(3, min_periods=1).std().fillna(0)
        df["ghi_rmean_6h"] = ghi.rolling(6, min_periods=1).mean()

        # GHI variability index — high std/mean = intermittent clouds
        with np.errstate(divide="ignore", invalid="ignore"):
            df["ghi_variability"] = np.where(
                df["ghi_rmean_3h"] > 10,
                df["ghi_rstd_3h"] / df["ghi_rmean_3h"],
                0,
            )

        # GHI lag — persistence forecast baseline
        df["ghi_lag_1h"] = ghi.shift(1)
        df["ghi_lag_24h"] = ghi.shift(24)

    if "direct_normal_irradiance" in df.columns and "shortwave_radiation" in df.columns:
        # Direct/GHI ratio — indicates beam vs diffuse dominance
        ghi = df["shortwave_radiation"]
        dni = df["direct_normal_irradiance"]
        with np.errstate(divide="ignore", invalid="ignore"):
            df["dni_ghi_ratio"] = np.where(ghi > 10, dni / ghi, 0)

    if "diffuse_radiation" in df.columns and "shortwave_radiation" in df.columns:
        # Diffuse fraction — high = overcast, low = clear sky
        ghi = df["shortwave_radiation"]
        dhi = df["diffuse_radiation"]
        with np.errstate(divide="ignore", invalid="ignore"):
            df["diffuse_fraction"] = np.where(ghi > 10, dhi / ghi, 0)
        df["diffuse_fraction"] = df["diffuse_fraction"].clip(0, 1)

    # Clear-sky ratio from NASA POWER (if merged)
    if "clrsky_sfc_sw_dwn" in df.columns and "shortwave_radiation" in df.columns:
        clr = df["clrsky_sfc_sw_dwn"]
        ghi = df["shortwave_radiation"]
        with np.errstate(divide="ignore", invalid="ignore"):
            df["clearsky_ratio"] = np.where(clr > 10, ghi / clr, 0)
        df["clearsky_ratio"] = df["clearsky_ratio"].clip(0, 1.5)

    # Cloud × GHI interaction — captures the nonlinear cloud impact on solar
    if "cloud_fraction" in df.columns and "shortwave_radiation" in df.columns:
        df["cloud_ghi_interaction"] = df["cloud_fraction"] * df["shortwave_radiation"]

    return df


def add_air_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add air quality features that affect PV soiling and generation.

    In coastal AP, dust and PM2.5 from seasonal winds can reduce PV output
    by 1-5%. AOD (Aerosol Optical Depth) directly attenuates incoming solar.
    """
    df = df.copy()

    if "pm2_5" in df.columns:
        # Air quality categories (WHO thresholds)
        df["aqi_poor"] = (df["pm2_5"] > 35).astype(int)  # PM2.5 > 35 µg/m³
        df["aqi_very_poor"] = (df["pm2_5"] > 75).astype(int)

        # Rolling PM2.5 — cumulative soiling over days
        df["pm25_rmean_24h"] = df["pm2_5"].rolling(24, min_periods=1).mean()
        df["pm25_rmean_72h"] = df["pm2_5"].rolling(72, min_periods=1).mean()

    if "aerosol_optical_depth" in df.columns:
        # AOD directly reduces solar radiation
        # Soiling loss proxy: higher AOD = more scattering = less DNI
        df["aod_high"] = (df["aerosol_optical_depth"] > 0.4).astype(int)

        # AOD rolling — captures seasonal haze patterns
        df["aod_rmean_24h"] = df["aerosol_optical_depth"].rolling(24, min_periods=1).mean()

    if "dust" in df.columns:
        # Dust events — can cut solar by 10-20% in a few hours
        df["dust_event"] = (df["dust"] > 50).astype(int)

    # Combined soiling index — higher = more PV losses
    soiling_components = []
    if "pm25_rmean_72h" in df.columns:
        soiling_components.append(df["pm25_rmean_72h"] / 100)  # Normalize to ~0-1
    if "aod_rmean_24h" in df.columns:
        soiling_components.append(df["aod_rmean_24h"])
    if "dust" in df.columns:
        soiling_components.append(df["dust"] / 200)  # Normalize

    if soiling_components:
        df["soiling_index"] = sum(soiling_components) / len(soiling_components)

    return df


# ─── Weather × Demand Interaction Features ──────────────────────────────────

def add_weather_demand_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-feature interactions between weather, time, and consumption.

    These capture domain-specific patterns like:
    - Hot afternoons drive peak demand (temp × business_hour)
    - Weekend heat doesn't spike commercial demand (temp × is_weekend)
    - Monsoon clouds reduce both demand and solar (rain × cloud × GHI)
    """
    df = df.copy()

    # Temperature × time-of-day interactions
    if "cdh" in df.columns and "is_business_hour" in df.columns:
        # Commercial cooling load: hot + business hours = peak demand
        df["cdh_x_business"] = df["cdh"] * df["is_business_hour"]

    if "cdh" in df.columns and "is_peak_hour" in df.columns:
        # Peak hour cooling: highest tariff × highest load
        df["cdh_x_peak"] = df["cdh"] * df["is_peak_hour"]

    if "cdh" in df.columns and "is_weekend" in df.columns:
        # Weekend cooling: lower commercial, but residential up
        df["cdh_x_weekend"] = df["cdh"] * df["is_weekend"]

    # Solar × demand opportunity
    if "shortwave_radiation" in df.columns and "is_business_hour" in df.columns:
        # Solar generation during demand hours — self-consumption potential
        df["ghi_x_business"] = df["shortwave_radiation"] * df["is_business_hour"]

    return df


def build_forecast_features(
    df: pd.DataFrame,
    target_col: str = "demand_kwh",
    timestamp_col: str = "timestamp",
    group_col: str = "consumer_id",
    lag_hours: List[int] = [1, 2, 3, 6, 12, 24, 48, 168],
    rolling_windows: List[int] = [3, 6, 12, 24, 48, 168],
    include_weather: bool = True,
    include_solar: bool = True,
    include_air_quality: bool = True,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline. Combines all 8 feature families.

    This is the main entry point — call this to prepare data for model training.

    Args:
        df: DataFrame with at least timestamp + target columns.
              If weather/solar columns are present, those features are auto-added.
        target_col: Column to forecast (demand_kwh, solar_kwh, etc.)
        include_weather: Add weather features if weather columns are present
        include_solar: Add solar radiation features if radiation columns are present
        include_air_quality: Add air quality features if AQ columns are present
    """
    logger.info(f"Building forecast features for {target_col}")
    logger.info(f"  Input: {len(df)} rows, {len(df.columns)} columns")

    # Family 1: Temporal features
    df = add_temporal_features(df, timestamp_col)

    # Family 2: Lag features
    df = add_lag_features(df, target_col, lag_hours, group_col)

    # Family 3: Rolling features
    df = add_rolling_features(df, target_col, rolling_windows, group_col)

    # Family 4: Price features
    df = add_price_features(df)

    # Family 5: Consumption pattern features
    df = add_consumption_pattern_features(df, target_col, group_col)

    # Family 6: Weather features (if weather data is merged)
    if include_weather and "temperature_2m" in df.columns:
        df = add_weather_features(df)
        logger.info("  Added weather features (temp, humidity, wind, CDD/HDD, comfort)")

    # Family 7: Solar features (if radiation data is merged)
    if include_solar and "shortwave_radiation" in df.columns:
        df = add_solar_features(df)
        logger.info("  Added solar features (GHI, DNI, DHI, clear-sky ratio, variability)")

    # Family 8: Air quality features (if AQ data is merged)
    if include_air_quality and "pm2_5" in df.columns:
        df = add_air_quality_features(df)
        logger.info("  Added air quality features (PM2.5, AOD, soiling index)")

    # Cross-family interactions (weather × time × demand)
    if "cdh" in df.columns:
        df = add_weather_demand_interactions(df)
        logger.info("  Added weather×demand interaction features")

    # Drop rows where lag features create NaN (start of series)
    max_lag = max(lag_hours + rolling_windows)
    initial_rows = len(df)
    df = df.dropna(subset=[f"{target_col}_lag_{max(lag_hours)}h"])
    logger.info(f"  Dropped {initial_rows - len(df)} rows (lookback={max_lag}h)")

    logger.info(f"  Final feature set: {len(df)} rows, {len(df.columns)} columns")
    return df
