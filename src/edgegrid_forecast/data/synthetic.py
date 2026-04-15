"""
Synthetic demand generator for model development and validation.

Generates realistic C&I electricity demand correlated with actual weather data.
Used until real APEPDCL meter data is available.

The demand model captures:
1. Base load (24/7 lighting, IT, security) — varies by consumer type
2. Scheduled load (office hours, shift patterns) — time-of-day dependent
3. Cooling load (HVAC) — strongly correlated with temperature above 26°C
4. Random variation (process-specific, occupancy fluctuations)

Consumer profiles mirror the 6 Bajaj connections across 3 APEPDCL regions.
"""

import numpy as np
import pandas as pd
from loguru import logger


# ─── Consumer Profiles ──────────────────────────────────────────────────────
# Each profile defines realistic demand patterns for a consumer type.
# Based on typical HT-I industrial/commercial loads in coastal AP.

CONSUMER_PROFILES = {
    "RJY1197": {
        "type": "manufacturing",
        "base_kw": 180,          # 24/7 base load
        "peak_kw": 420,          # Daytime peak capacity
        "temp_sensitivity": 8.0,  # kW per degree above 26°C
        "shift_start": 8,
        "shift_end": 20,
        "weekend_factor": 0.45,   # Manufacturing runs partial weekend
    },
    "RJY1622": {
        "type": "commercial",
        "base_kw": 120,
        "peak_kw": 350,
        "temp_sensitivity": 12.0,  # Higher — office AC load
        "shift_start": 9,
        "shift_end": 18,
        "weekend_factor": 0.30,
    },
    "SKL724": {
        "type": "manufacturing",
        "base_kw": 200,
        "peak_kw": 500,
        "temp_sensitivity": 6.0,   # Less AC-dependent (factory floor)
        "shift_start": 6,
        "shift_end": 22,          # Long shift
        "weekend_factor": 0.55,
    },
    "VSP2315": {
        "type": "commercial",
        "base_kw": 150,
        "peak_kw": 380,
        "temp_sensitivity": 11.0,
        "shift_start": 9,
        "shift_end": 19,
        "weekend_factor": 0.25,
    },
    "VSP2432": {
        "type": "IT_park",
        "base_kw": 250,
        "peak_kw": 550,
        "temp_sensitivity": 14.0,  # IT parks have heavy cooling
        "shift_start": 8,
        "shift_end": 21,
        "weekend_factor": 0.40,   # Some ops on weekends
    },
    "VSP2439": {
        "type": "manufacturing",
        "base_kw": 170,
        "peak_kw": 400,
        "temp_sensitivity": 7.0,
        "shift_start": 7,
        "shift_end": 19,
        "weekend_factor": 0.50,
    },
}

# Location → Consumer mapping (must match pull_all.py)
LOCATION_CONSUMERS = {
    "rajahmundry": ["RJY1197", "RJY1622"],
    "srikakulam": ["SKL724"],
    "visakhapatnam": ["VSP2315", "VSP2432", "VSP2439"],
}


def generate_demand_for_consumer(
    weather_df: pd.DataFrame,
    consumer_id: str,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic hourly demand for one consumer, correlated with weather.

    The demand model:
        demand = base_load + scheduled_load + cooling_load + noise

    Where:
        base_load = constant 24/7 load
        scheduled_load = ramp up during working hours, down at night
        cooling_load = temp_sensitivity × max(0, temperature - 26°C)
        noise = ±5-10% random variation
    """
    profile = CONSUMER_PROFILES[consumer_id]
    rng = np.random.RandomState(seed + hash(consumer_id) % 10000)

    ts = pd.to_datetime(weather_df["timestamp"])
    temp = weather_df["temperature_2m"].values
    cloud = weather_df["cloud_cover"].values if "cloud_cover" in weather_df.columns else np.zeros(len(ts))
    rh = weather_df["relative_humidity_2m"].values if "relative_humidity_2m" in weather_df.columns else np.full(len(ts), 60)

    hours = ts.dt.hour.values
    dow = ts.dt.dayofweek.values  # 0=Monday
    month = ts.dt.month.values

    n = len(ts)

    # 1. Base load — always on, with slight diurnal variation
    base = np.full(n, float(profile["base_kw"]))
    # Slight dip at night (lighting reduction)
    night_mask = (hours >= 23) | (hours <= 5)
    base[night_mask] *= 0.85

    # 2. Scheduled load — ramps up during shift hours
    scheduled = np.zeros(n)
    shift_range = profile["shift_end"] - profile["shift_start"]
    peak_addition = profile["peak_kw"] - profile["base_kw"]

    for i in range(n):
        h = hours[i]
        if profile["shift_start"] <= h < profile["shift_end"]:
            # Ramp-up in first 2 hours, ramp-down in last hour
            hours_into_shift = h - profile["shift_start"]
            hours_left = profile["shift_end"] - h
            ramp = min(hours_into_shift / 2.0, 1.0) * min(hours_left / 1.0, 1.0)
            scheduled[i] = peak_addition * ramp

    # Weekend reduction
    is_weekend = dow >= 5
    scheduled[is_weekend] *= profile["weekend_factor"]

    # 3. Cooling load — the weather-correlated component
    cooling = np.maximum(temp - CDD_BASE, 0) * profile["temp_sensitivity"]

    # Humidity amplifier: above 70% RH, effective cooling need increases 15%
    humidity_factor = np.where(rh > 70, 1.15, 1.0)
    cooling *= humidity_factor

    # Cooling only during occupied hours (HVAC off at night for commercial)
    if profile["type"] in ("commercial", "IT_park"):
        unoccupied = (hours < profile["shift_start"] - 1) | (hours > profile["shift_end"] + 1)
        cooling[unoccupied] *= 0.15  # Minimal overnight cooling

    # 4. Seasonal adjustment — monsoon is milder, pre-monsoon is brutal
    seasonal = np.ones(n)
    seasonal[np.isin(month, [6, 7, 8, 9])] = 0.92    # Monsoon: milder temps + rain
    seasonal[np.isin(month, [4, 5])] = 1.08            # Pre-monsoon: peak heat
    seasonal[np.isin(month, [12, 1, 2])] = 0.88        # Winter: lower cooling

    # 5. Random noise — process variability, occupancy fluctuation
    noise_pct = 0.06 if profile["type"] == "manufacturing" else 0.08
    noise = rng.normal(1.0, noise_pct, n)

    # 6. Combine
    demand_kw = (base + scheduled + cooling) * seasonal * noise

    # Clamp to reasonable bounds
    demand_kw = np.clip(demand_kw, profile["base_kw"] * 0.5, profile["peak_kw"] * 1.3)

    # Convert kW to kWh (hourly data, so kW ≈ kWh for each hour)
    demand_kwh = demand_kw

    result = pd.DataFrame({
        "timestamp": ts.values,
        "consumer_id": consumer_id,
        "demand_kwh": demand_kwh,
        "location_id": weather_df["location_id"].values,
    })

    logger.info(
        f"  [{consumer_id}] Demand: mean={demand_kwh.mean():.0f} kWh, "
        f"peak={demand_kwh.max():.0f} kWh, min={demand_kwh.min():.0f} kWh"
    )
    return result


# Comfort threshold used in generation — matches features.py
CDD_BASE = 26.0


def generate_all_demand(
    weather_dir: str = "data/external/weather",
) -> pd.DataFrame:
    """
    Generate synthetic demand for all 6 consumers using actual weather data.

    Returns a single DataFrame ready for merging with weather features.
    """
    logger.info("Generating synthetic demand for all consumers")

    all_dfs = []

    for location_id, consumers in LOCATION_CONSUMERS.items():
        weather_path = f"{weather_dir}/{location_id}_fy2425.parquet"
        weather_df = pd.read_parquet(weather_path)
        logger.info(f"  Location: {location_id} ({len(weather_df)} hours, {len(consumers)} consumers)")

        for consumer_id in consumers:
            demand_df = generate_demand_for_consumer(weather_df, consumer_id)
            all_dfs.append(demand_df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.sort_values(["consumer_id", "timestamp"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    logger.info(
        f"Total synthetic demand: {len(combined):,} rows, "
        f"{combined['consumer_id'].nunique()} consumers"
    )
    return combined
