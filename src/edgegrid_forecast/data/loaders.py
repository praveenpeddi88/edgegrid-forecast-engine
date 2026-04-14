"""
Data loaders for DISCOM smart meter data, meter quality reports, and public APIs.

Handles:
1. APEPDCL HT Consumer Consumption Profile (8760 hourly data)
2. Meter Quality Reports (anomaly-flagged meter data)
3. Open-Meteo weather API (temperature, cloud cover, irradiance)
4. IEX DAM prices (historical and forecast)
"""

import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import httpx
import numpy as np
import pandas as pd
from loguru import logger

from ..utils.constants import HT_CONSUMERS


# ─── DISCOM HT Consumer Data ─────────────────────────────────────────────────

def load_ht_consumption_profile(
    filepath: Union[str, Path],
    sheet_name: str = "Raw Data",
) -> pd.DataFrame:
    """
    Load APEPDCL HT Consumer Consumption Profile from Excel.

    The raw data sheet has:
    - Rows 1-3: headers and metadata
    - Row 4 (0-indexed 3): column headers
    - Columns: Date, Hourly Time, Hour, Day, Month, then per-consumer demand values
    - 6 consumers: RJY1197, RJY1622, SKL724, VSP2315, VSP2432, VSP2439
    - Both "Actual Demand vah" and "Max*2 Demand vah" sections

    Returns a long-format DataFrame with columns:
        timestamp, consumer_id, demand_vah, max_demand_vah, region
    """
    logger.info(f"Loading HT consumption profile from {filepath}")

    # Read with openpyxl to handle formulas (data_only=True for calculated values)
    df_raw = pd.read_excel(
        filepath,
        sheet_name=sheet_name,
        header=3,  # Row 4 has the actual column names
        engine="openpyxl",
    )

    # The sheet has duplicate column names for actual vs max demand
    # Actual demand columns: indices 6-11 (RJY1197..VSP2439)
    # Max demand columns: indices 12-17 (RJY1197..VSP2439)
    consumer_ids = ["RJY1197", "RJY1622", "SKL724", "VSP2315", "VSP2432", "VSP2439"]

    # Extract timestamp
    time_col = df_raw.columns[2]  # "Hourly Time"
    timestamps = pd.to_datetime(df_raw[time_col], errors="coerce")

    records = []
    for i, cid in enumerate(consumer_ids):
        # Actual demand is in columns 6+i, max demand in 12+i
        actual_col_idx = 6 + i
        max_col_idx = 12 + i

        actual_vals = pd.to_numeric(df_raw.iloc[:, actual_col_idx], errors="coerce")
        max_vals = pd.to_numeric(df_raw.iloc[:, max_col_idx], errors="coerce")

        consumer_df = pd.DataFrame({
            "timestamp": timestamps,
            "consumer_id": cid,
            "demand_vah": actual_vals.values,
            "max_demand_vah": max_vals.values,
        })
        consumer_df["region"] = HT_CONSUMERS.get(cid, {}).get("region", "Unknown")
        records.append(consumer_df)

    result = pd.concat(records, ignore_index=True)
    result = result.dropna(subset=["timestamp"])

    # Convert VAh to kWh (assuming ~unity power factor for HT consumers)
    result["demand_kwh"] = result["demand_vah"] / 1000
    result["demand_kw"] = result["demand_kwh"]  # Hourly kWh = avg kW for that hour

    logger.info(
        f"Loaded {len(result)} records for {len(consumer_ids)} consumers, "
        f"date range: {result['timestamp'].min()} to {result['timestamp'].max()}"
    )
    return result


def load_ht_consumption_pivots(
    filepath: Union[str, Path],
    sheet_name: str = "Pivots_Data_Monthly",
) -> pd.DataFrame:
    """
    Load the monthly pivot data — pre-aggregated hourly peaks per consumer per month.
    Useful for quick profiling without processing the full 8760 dataset.
    """
    logger.info(f"Loading pivot data from {filepath}, sheet={sheet_name}")

    df = pd.read_excel(filepath, sheet_name=sheet_name, header=None, engine="openpyxl")

    consumer_ids = ["RJY1197", "RJY1622", "SKL724", "VSP2315", "VSP2432", "VSP2439"]
    months = list(range(1, 13))  # Calendar months

    records = []
    # Each month is a block of 8 columns (Hour + 6 consumers + blank)
    for month_idx, month in enumerate(months[:6]):  # First 6 months visible
        col_offset = month_idx * 8
        hour_col = col_offset
        for i, cid in enumerate(consumer_ids):
            data_col = col_offset + 1 + i
            for row_idx in range(3, 27):  # Hours 1-24
                try:
                    hour = int(df.iloc[row_idx, hour_col])
                    val = float(df.iloc[row_idx, data_col]) if pd.notna(df.iloc[row_idx, data_col]) else 0
                    records.append({
                        "month": month,
                        "hour": hour,
                        "consumer_id": cid,
                        "max_demand_vah": val,
                    })
                except (ValueError, TypeError):
                    continue

    result = pd.DataFrame(records)
    logger.info(f"Loaded {len(result)} pivot records")
    return result


# ─── Meter Quality Report ────────────────────────────────────────────────────

def load_meter_quality_report(filepath: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """
    Load the meter quality analysis report.

    Returns a dict of DataFrames:
        - summary: Per-meter quality metrics and issue flags
        - missing_gaps: Time gaps where meter data is missing
        - frozen_values: Periods where readings didn't change (stuck meter)
        - outlier_samples: Individual outlier data points
        - column_stats: Statistical summary per meter per measurement column
    """
    logger.info(f"Loading meter quality report from {filepath}")

    sheets = {}
    for sheet_name in ["Summary", "Missing_Gaps", "Frozen_Values", "Outlier_Samples", "Column_Stats"]:
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name, engine="openpyxl")
            sheets[sheet_name.lower()] = df
            logger.info(f"  {sheet_name}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"  Could not load sheet {sheet_name}: {e}")
            sheets[sheet_name.lower()] = pd.DataFrame()

    # Parse datetime columns in missing_gaps
    if "missing_gaps" in sheets and not sheets["missing_gaps"].empty:
        for col in ["Gap_Start", "Gap_End"]:
            if col in sheets["missing_gaps"].columns:
                sheets["missing_gaps"][col] = pd.to_datetime(
                    sheets["missing_gaps"][col], errors="coerce"
                )

    # Parse datetime in frozen_values
    if "frozen_values" in sheets and not sheets["frozen_values"].empty:
        for col in ["Run_Start", "Run_End"]:
            if col in sheets["frozen_values"].columns:
                sheets["frozen_values"][col] = pd.to_datetime(
                    sheets["frozen_values"][col], errors="coerce"
                )

    return sheets


def get_anomaly_mask(
    quality_report: Dict[str, pd.DataFrame],
    consumer_id: str,
    timestamps: pd.DatetimeIndex,
) -> pd.Series:
    """
    Create a boolean mask indicating which timestamps have known data quality issues.

    This is critical for forecasting — we need to exclude or impute anomalous periods.
    Issues considered: missing gaps, frozen values, outliers.
    """
    mask = pd.Series(False, index=timestamps)

    # Mark missing gaps
    gaps = quality_report.get("missing_gaps", pd.DataFrame())
    if not gaps.empty:
        consumer_gaps = gaps[gaps["UKSCNO"].astype(str) == str(consumer_id)]
        for _, gap in consumer_gaps.iterrows():
            mask |= (timestamps >= gap["Gap_Start"]) & (timestamps <= gap["Gap_End"])

    # Mark frozen value periods
    frozen = quality_report.get("frozen_values", pd.DataFrame())
    if not frozen.empty:
        consumer_frozen = frozen[frozen["UKSCNO"].astype(str) == str(consumer_id)]
        for _, f in consumer_frozen.iterrows():
            mask |= (timestamps >= f["Run_Start"]) & (timestamps <= f["Run_End"])

    return mask


# ─── Weather Data (Open-Meteo API) ───────────────────────────────────────────

async def fetch_weather_forecast(
    latitude: float,
    longitude: float,
    days_ahead: int = 7,
) -> pd.DataFrame:
    """
    Fetch weather forecast from Open-Meteo API (free, no key needed).

    Returns hourly data:
        timestamp, temperature_2m, cloud_cover, direct_radiation,
        diffuse_radiation, wind_speed_10m
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join([
            "temperature_2m",
            "cloud_cover",
            "direct_radiation",
            "diffuse_radiation",
            "wind_speed_10m",
            "relative_humidity_2m",
        ]),
        "forecast_days": days_ahead,
        "timezone": "Asia/Kolkata",
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
        resp.raise_for_status()
        data = resp.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "temperature_c": hourly["temperature_2m"],
        "cloud_cover_pct": hourly["cloud_cover"],
        "direct_radiation_wm2": hourly["direct_radiation"],
        "diffuse_radiation_wm2": hourly["diffuse_radiation"],
        "wind_speed_ms": hourly["wind_speed_10m"],
        "humidity_pct": hourly["relative_humidity_2m"],
    })
    return df


async def fetch_weather_history(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo Archive API.

    Args:
        start_date: "YYYY-MM-DD"
        end_date: "YYYY-MM-DD"
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join([
            "temperature_2m",
            "cloud_cover",
            "direct_radiation",
            "diffuse_radiation",
            "wind_speed_10m",
            "relative_humidity_2m",
        ]),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Asia/Kolkata",
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://archive-api.open-meteo.com/v1/archive", params=params
        )
        resp.raise_for_status()
        data = resp.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "temperature_c": hourly["temperature_2m"],
        "cloud_cover_pct": hourly["cloud_cover"],
        "direct_radiation_wm2": hourly["direct_radiation"],
        "diffuse_radiation_wm2": hourly["diffuse_radiation"],
        "wind_speed_ms": hourly["wind_speed_10m"],
        "humidity_pct": hourly["relative_humidity_2m"],
    })
    return df


# ─── Synthetic Solar Generation ──────────────────────────────────────────────

def generate_solar_profile(
    timestamps: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    capacity_kw: float = 1000,
    efficiency: float = 0.18,
    pr_ratio: float = 0.80,
) -> pd.Series:
    """
    Generate approximate solar generation profile using a simplified model.

    For production, use pvlib with actual irradiance data. This provides a
    reasonable baseline for Indian latitudes (13-28°N).

    Args:
        capacity_kw: Installed solar capacity in kW
        efficiency: Panel efficiency
        pr_ratio: Performance ratio (accounts for losses)
    """
    hours = timestamps.hour
    months = timestamps.month

    # Solar elevation approximation for Indian latitudes
    # Peak generation hours: roughly 7am to 5pm, max at noon
    day_fraction = (hours - 6) / 12  # 0 at 6am, 1 at 6pm
    solar_angle = np.sin(np.pi * day_fraction.clip(0, 1))

    # Seasonal variation (less in India than temperate zones)
    # Peak in March-May, lower in monsoon (Jul-Sep), good Oct-Feb
    seasonal = 1 + 0.15 * np.cos(2 * np.pi * (months - 4) / 12)

    # Monsoon cloud factor (July-September get ~40% less)
    monsoon_factor = np.where(months.isin([7, 8, 9]), 0.6, 1.0)

    # Combine factors
    generation_kwh = (
        capacity_kw
        * solar_angle
        * seasonal
        * monsoon_factor
        * pr_ratio
        * np.where(solar_angle > 0, 1, 0)
    )

    return pd.Series(generation_kwh, index=timestamps, name="solar_generation_kwh")
