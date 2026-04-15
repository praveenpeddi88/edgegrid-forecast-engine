"""
NASA POWER API collector for long-term solar and meteorological data.

Free, no API key. Global coverage. Hourly data from 2001 to near-real-time.
Used for:
- Long-term solar baseline (20+ year monthly/seasonal patterns)
- Cross-validation of Open-Meteo satellite radiation data
- Historical capacity factor estimation for BESS sizing

API docs: https://power.larc.nasa.gov/docs/services/api/temporal/hourly/
"""

import time
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests
from loguru import logger

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# Parameters relevant for solar + weather validation
SOLAR_PARAMS = [
    "ALLSKY_SFC_SW_DWN",    # All-sky surface shortwave downward irradiance (Wh/m²)
    "CLRSKY_SFC_SW_DWN",    # Clear-sky surface shortwave downward irradiance (Wh/m²)
]

WEATHER_PARAMS = [
    "T2M",                   # Temperature at 2m (°C)
    "RH2M",                  # Relative humidity at 2m (%)
    "WS10M",                 # Wind speed at 10m (m/s)
]

ALL_PARAMS = SOLAR_PARAMS + WEATHER_PARAMS

# NASA POWER limits: max 366 days per request for hourly
CHUNK_DAYS = 365
FILL_VALUE = -999.0


def fetch_nasa_power_history(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    location_id: str,
    parameters: List[str] = None,
) -> pd.DataFrame:
    """
    Fetch hourly data from NASA POWER API.

    Chunks into 365-day requests. Returns clean DataFrame with
    fill values (-999) replaced by NaN.
    """
    if parameters is None:
        parameters = ALL_PARAMS

    all_dfs = []
    current_start = start_date

    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=CHUNK_DAYS - 1), end_date)

        logger.info(
            f"[{location_id}] NASA POWER: {current_start} to {current_end} "
            f"({(current_end - current_start).days + 1} days)"
        )

        params = {
            "start": current_start.strftime("%Y%m%d"),
            "end": current_end.strftime("%Y%m%d"),
            "latitude": latitude,
            "longitude": longitude,
            "community": "RE",  # Renewable Energy community
            "parameters": ",".join(parameters),
            "format": "json",
            "time-standard": "lst",  # Local Solar Time
        }

        try:
            resp = requests.get(NASA_POWER_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"[{location_id}] NASA POWER request failed: {e}")
            current_start = current_end + timedelta(days=1)
            time.sleep(1)
            continue

        # Parse the nested JSON structure
        properties = data.get("properties", {})
        param_data = properties.get("parameter", {})

        if not param_data:
            logger.warning(f"[{location_id}] No parameter data in response")
            current_start = current_end + timedelta(days=1)
            continue

        # Build DataFrame from the parameter dictionaries
        # Keys are like "2025010100" (YYYYMMDDHH)
        first_param = list(param_data.keys())[0]
        timestamps = sorted(param_data[first_param].keys())

        rows = []
        for ts_key in timestamps:
            row = {"timestamp_key": ts_key}
            for param_name in parameters:
                if param_name in param_data:
                    val = param_data[param_name].get(ts_key, FILL_VALUE)
                    row[param_name.lower()] = val if val != FILL_VALUE else None
                else:
                    row[param_name.lower()] = None
            rows.append(row)

        df = pd.DataFrame(rows)

        # Parse timestamp: format is YYYYMMDDHH
        df["timestamp"] = pd.to_datetime(
            df["timestamp_key"], format="%Y%m%d%H", errors="coerce"
        )
        df.drop(columns=["timestamp_key"], inplace=True)
        df["location_id"] = location_id
        df["latitude"] = latitude
        df["longitude"] = longitude

        all_dfs.append(df)

        current_start = current_end + timedelta(days=1)
        # NASA POWER is slower — be more polite
        time.sleep(1)

    if not all_dfs:
        logger.warning(f"[{location_id}] No NASA POWER data retrieved")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result.sort_values("timestamp", inplace=True)
    result.reset_index(drop=True, inplace=True)

    # Drop rows where timestamp parsing failed
    result.dropna(subset=["timestamp"], inplace=True)

    logger.info(
        f"[{location_id}] NASA POWER: {len(result)} rows, "
        f"{result['timestamp'].min()} to {result['timestamp'].max()}"
    )
    return result
