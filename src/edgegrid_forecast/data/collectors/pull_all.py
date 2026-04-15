"""
Master data pull script — fetches 1 year of historical data from all free APIs
for all APEPDCL consumer locations.

Usage:
    python -m edgegrid_forecast.data.collectors.pull_all

Output:
    data/external/weather/       — Open-Meteo weather + solar radiation
    data/external/air_quality/   — Open-Meteo air quality (PM2.5, PM10, dust, AOD)
    data/external/nasa_power/    — NASA POWER 20-year solar baseline

All files saved as Parquet for efficient storage and fast reads.
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edgegrid_forecast.data.collectors.open_meteo import (
    fetch_weather_solar_history,
    fetch_air_quality_history,
)
from edgegrid_forecast.data.collectors.nasa_power import (
    fetch_nasa_power_history,
)

# ─── Locations ──────────────────────────────────────────────────────────────

LOCATIONS = {
    "rajahmundry": {"lat": 17.0005, "lon": 81.8040, "consumers": ["RJY1197", "RJY1622"]},
    "srikakulam":  {"lat": 18.2949, "lon": 83.8938, "consumers": ["SKL724"]},
    "visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "consumers": ["VSP2315", "VSP2432", "VSP2439"]},
}

# ─── Date Ranges ────────────────────────────────────────────────────────────

# 1 year of recent history for model training
HISTORY_START = date(2024, 4, 1)   # FY 2024-25 start
HISTORY_END = date(2025, 3, 31)     # FY 2024-25 end

# NASA POWER: pull 3 years for seasonal baseline
NASA_START = date(2022, 4, 1)
NASA_END = date(2025, 3, 31)

# ─── Output Paths ───────────────────────────────────────────────────────────

DATA_DIR = PROJECT_ROOT / "data" / "external"
WEATHER_DIR = DATA_DIR / "weather"
AIR_QUALITY_DIR = DATA_DIR / "air_quality"
NASA_DIR = DATA_DIR / "nasa_power"


def pull_weather_solar():
    """Pull 1 year weather + solar radiation for all locations."""
    logger.info("=" * 60)
    logger.info("PULLING WEATHER + SOLAR RADIATION (Open-Meteo Archive)")
    logger.info("=" * 60)

    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []

    for loc_id, loc in LOCATIONS.items():
        df = fetch_weather_solar_history(
            latitude=loc["lat"],
            longitude=loc["lon"],
            start_date=HISTORY_START,
            end_date=HISTORY_END,
            location_id=loc_id,
        )

        # Save per-location
        out_path = WEATHER_DIR / f"{loc_id}_fy2425.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"  Saved: {out_path} ({len(df)} rows)")

        all_dfs.append(df)

    # Save combined
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = WEATHER_DIR / "all_locations_fy2425.parquet"
    combined.to_parquet(combined_path, index=False)
    logger.info(f"  Combined: {combined_path} ({len(combined)} rows)")

    return combined


def pull_air_quality():
    """Pull air quality data for all locations."""
    logger.info("=" * 60)
    logger.info("PULLING AIR QUALITY (Open-Meteo)")
    logger.info("=" * 60)

    AIR_QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []

    for loc_id, loc in LOCATIONS.items():
        df = fetch_air_quality_history(
            latitude=loc["lat"],
            longitude=loc["lon"],
            start_date=HISTORY_START,
            end_date=HISTORY_END,
            location_id=loc_id,
        )

        if df.empty:
            logger.warning(f"  No air quality data for {loc_id}")
            continue

        out_path = AIR_QUALITY_DIR / f"{loc_id}_fy2425.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"  Saved: {out_path} ({len(df)} rows)")

        all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = AIR_QUALITY_DIR / "all_locations_fy2425.parquet"
        combined.to_parquet(combined_path, index=False)
        logger.info(f"  Combined: {combined_path} ({len(combined)} rows)")
        return combined

    return pd.DataFrame()


def pull_nasa_power():
    """Pull 3-year NASA POWER solar baseline for all locations."""
    logger.info("=" * 60)
    logger.info("PULLING NASA POWER (3-year solar baseline)")
    logger.info("=" * 60)

    NASA_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []

    for loc_id, loc in LOCATIONS.items():
        df = fetch_nasa_power_history(
            latitude=loc["lat"],
            longitude=loc["lon"],
            start_date=NASA_START,
            end_date=NASA_END,
            location_id=loc_id,
        )

        if df.empty:
            logger.warning(f"  No NASA POWER data for {loc_id}")
            continue

        out_path = NASA_DIR / f"{loc_id}_3yr.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"  Saved: {out_path} ({len(df)} rows)")

        all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = NASA_DIR / "all_locations_3yr.parquet"
        combined.to_parquet(combined_path, index=False)
        logger.info(f"  Combined: {combined_path} ({len(combined)} rows)")
        return combined

    return pd.DataFrame()


def print_summary(weather_df, aq_df, nasa_df):
    """Print a summary of all collected data."""
    logger.info("\n" + "=" * 60)
    logger.info("DATA COLLECTION SUMMARY")
    logger.info("=" * 60)

    for name, df in [("Weather+Solar", weather_df), ("Air Quality", aq_df), ("NASA POWER", nasa_df)]:
        if df is not None and not df.empty:
            logger.info(f"\n{name}:")
            logger.info(f"  Rows: {len(df):,}")
            logger.info(f"  Columns: {list(df.columns)}")
            logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            logger.info(f"  Locations: {df['location_id'].unique().tolist()}")

            # Check for nulls
            null_pct = df.isnull().mean() * 100
            high_nulls = null_pct[null_pct > 5]
            if not high_nulls.empty:
                logger.warning(f"  Columns with >5% nulls: {high_nulls.to_dict()}")
            else:
                logger.info(f"  Data completeness: >95% across all columns")
        else:
            logger.warning(f"\n{name}: NO DATA COLLECTED")


if __name__ == "__main__":
    logger.info("EdgeGrid Forecast Engine — External Data Pull")
    logger.info(f"History period: {HISTORY_START} to {HISTORY_END}")
    logger.info(f"NASA POWER period: {NASA_START} to {NASA_END}")
    logger.info(f"Locations: {list(LOCATIONS.keys())}")

    weather_df = pull_weather_solar()
    aq_df = pull_air_quality()
    nasa_df = pull_nasa_power()

    print_summary(weather_df, aq_df, nasa_df)

    logger.info("\nDone. All data saved to data/external/")
