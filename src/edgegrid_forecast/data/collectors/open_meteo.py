"""
Open-Meteo API collectors for weather, solar radiation, and air quality.

Three separate APIs, all free, no key required:
1. Weather: temperature, humidity, wind, cloud, rain, pressure
2. Solar Radiation: GHI, DNI, DHI (satellite-derived for India via Himawari-8/9)
3. Air Quality: PM2.5, PM10, dust, aerosol optical depth

Rate limit: 10,000 calls/day per API (free tier).
For 3 locations × hourly = ~216 calls/day. Well within limits.
"""

import time
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from loguru import logger

# ─── API Endpoints ──────────────────────────────────────────────────────────

WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# ─── Parameter Sets ─────────────────────────────────────────────────────────

WEATHER_PARAMS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
    "precipitation",
    "surface_pressure",
]

SOLAR_PARAMS = [
    "shortwave_radiation",        # GHI
    "direct_normal_irradiance",   # DNI
    "diffuse_radiation",          # DHI
    "direct_radiation",           # Direct on horizontal
]

AIR_QUALITY_PARAMS = [
    "pm2_5",
    "pm10",
    "dust",
    "aerosol_optical_depth",
]

# All weather + solar params combined (archive API serves both)
ALL_WEATHER_SOLAR_PARAMS = WEATHER_PARAMS + SOLAR_PARAMS

TIMEZONE = "Asia/Kolkata"

# ─── Retry Logic ────────────────────────────────────────────────────────────

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
# Open-Meteo archive API limits to ~3 months per request for hourly data
CHUNK_DAYS = 90


def _request_with_retry(url: str, params: dict, retries: int = MAX_RETRIES) -> dict:
    """Make API request with exponential backoff retry."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = RETRY_DELAY_SECONDS * (2 ** attempt)
                logger.warning(f"Rate limited. Waiting {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                logger.error(f"API error {resp.status_code}: {resp.text[:200]}")
                if attempt < retries - 1:
                    time.sleep(RETRY_DELAY_SECONDS)
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(f"Failed after {retries} attempts: {url}")


def _json_to_dataframe(data: dict, location_id: str) -> pd.DataFrame:
    """Convert Open-Meteo JSON response to DataFrame."""
    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise ValueError(f"No hourly data in response for {location_id}")

    df = pd.DataFrame(hourly)
    df["timestamp"] = pd.to_datetime(df["time"])
    df.drop(columns=["time"], inplace=True)
    df["location_id"] = location_id
    df["latitude"] = data.get("latitude")
    df["longitude"] = data.get("longitude")
    df["elevation_m"] = data.get("elevation")

    return df


# ─── Weather + Solar Collector ──────────────────────────────────────────────

def fetch_weather_solar_history(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    location_id: str,
) -> pd.DataFrame:
    """
    Fetch historical weather + solar radiation from Open-Meteo archive API.

    Chunks into 90-day requests to respect API limits.
    Returns a single DataFrame spanning the full date range.
    """
    all_dfs = []
    current_start = start_date

    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=CHUNK_DAYS - 1), end_date)

        logger.info(
            f"[{location_id}] Fetching weather+solar: "
            f"{current_start} to {current_end} ({(current_end - current_start).days + 1} days)"
        )

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": current_start.isoformat(),
            "end_date": current_end.isoformat(),
            "hourly": ",".join(ALL_WEATHER_SOLAR_PARAMS),
            "timezone": TIMEZONE,
        }

        data = _request_with_retry(WEATHER_ARCHIVE_URL, params)
        df = _json_to_dataframe(data, location_id)
        all_dfs.append(df)

        current_start = current_end + timedelta(days=1)
        # Be polite to the API
        time.sleep(0.5)

    result = pd.concat(all_dfs, ignore_index=True)
    result.sort_values("timestamp", inplace=True)
    result.reset_index(drop=True, inplace=True)

    logger.info(
        f"[{location_id}] Weather+solar: {len(result)} rows, "
        f"{result['timestamp'].min()} to {result['timestamp'].max()}"
    )
    return result


# ─── Air Quality Collector ──────────────────────────────────────────────────

def fetch_air_quality_history(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    location_id: str,
) -> pd.DataFrame:
    """
    Fetch historical air quality data from Open-Meteo.

    Note: Air quality history may have shorter availability than weather.
    The API provides forecast + a few past days. For deep history,
    we may get partial data — the collector handles this gracefully.
    """
    all_dfs = []
    current_start = start_date

    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=CHUNK_DAYS - 1), end_date)

        logger.info(
            f"[{location_id}] Fetching air quality: "
            f"{current_start} to {current_end}"
        )

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": current_start.isoformat(),
            "end_date": current_end.isoformat(),
            "hourly": ",".join(AIR_QUALITY_PARAMS),
            "timezone": TIMEZONE,
        }

        try:
            data = _request_with_retry(AIR_QUALITY_URL, params)
            df = _json_to_dataframe(data, location_id)
            all_dfs.append(df)
        except Exception as e:
            logger.warning(f"[{location_id}] Air quality chunk failed: {e}. Skipping.")

        current_start = current_end + timedelta(days=1)
        time.sleep(0.5)

    if not all_dfs:
        logger.warning(f"[{location_id}] No air quality data retrieved")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result.sort_values("timestamp", inplace=True)
    result.reset_index(drop=True, inplace=True)

    logger.info(
        f"[{location_id}] Air quality: {len(result)} rows, "
        f"{result['timestamp'].min()} to {result['timestamp'].max()}"
    )
    return result


# ─── Forecast Collector (for real-time inference) ───────────────────────────

def fetch_weather_solar_forecast(
    latitude: float,
    longitude: float,
    location_id: str,
    forecast_days: int = 7,
    past_days: int = 2,
) -> pd.DataFrame:
    """
    Fetch weather + solar forecast for upcoming days.
    Used in production for real-time dispatch optimization.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(ALL_WEATHER_SOLAR_PARAMS),
        "timezone": TIMEZONE,
        "forecast_days": forecast_days,
        "past_days": past_days,
    }

    data = _request_with_retry(WEATHER_FORECAST_URL, params)
    return _json_to_dataframe(data, location_id)
