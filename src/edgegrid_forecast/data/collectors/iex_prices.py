"""
IEX Day-Ahead Market price collector.

IEX India (iexindia.com) does NOT offer a public API for DAM prices.
Data must be obtained via:
1. Manual CSV export from iexindia.com/market-data/day-ahead-market/market-snapshot
2. Future: automated browser scraper (Playwright/Selenium)
3. Fallback: synthetic prices from the FY24-25 monthly average matrix

This module handles all three paths and produces a standardized DataFrame
with 15-minute or hourly MCP (Market Clearing Price) in INR/MWh.

Data flow:
    IEX website export (CSV) → parse_iex_csv() → standardized DataFrame
    No CSV available          → generate_synthetic_prices() → realistic synthetic
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from ...utils.constants import IEX_DAM_PRICES_FY2425, fy_month_index


def parse_iex_csv(
    csv_path: str,
    date_col: str = "Date",
    time_col: str = "Time Block",
    price_col: str = "MCP (Rs/MWh)",
) -> pd.DataFrame:
    """
    Parse a CSV exported from IEX DAM Market Snapshot.

    The IEX export typically has columns:
    - Date, Time Block (e.g., "00:00-00:15"), Purchase Bid (MW),
      Sell Bid (MW), MCV (MW), Final Scheduled Volume (MW), MCP (Rs/MWh)

    Returns standardized DataFrame with:
    - timestamp (datetime), mcp_inr_mwh (float), mcp_inr_kwh (float)
    """
    logger.info(f"Parsing IEX CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    # Handle various column name formats IEX uses
    col_map = {}
    for col in df.columns:
        col_lower = col.strip().lower()
        if "date" in col_lower:
            col_map[col] = "date_str"
        elif "time" in col_lower and "block" in col_lower:
            col_map[col] = "time_block"
        elif "mcp" in col_lower:
            col_map[col] = "mcp_inr_mwh"
        elif "mcv" in col_lower:
            col_map[col] = "mcv_mw"
        elif "purchase" in col_lower:
            col_map[col] = "purchase_bid_mw"
        elif "sell" in col_lower:
            col_map[col] = "sell_bid_mw"
        elif "scheduled" in col_lower or "final" in col_lower:
            col_map[col] = "scheduled_volume_mw"

    df.rename(columns=col_map, inplace=True)

    # Parse timestamp from date + time block
    if "time_block" in df.columns and "date_str" in df.columns:
        # Time block format: "00:00-00:15" → use start time
        df["time_start"] = df["time_block"].str.split("-").str[0].str.strip()
        df["timestamp"] = pd.to_datetime(
            df["date_str"].astype(str) + " " + df["time_start"],
            format="mixed",
            dayfirst=True,
        )
    elif "date_str" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date_str"], format="mixed", dayfirst=True)

    # Clean MCP column
    if "mcp_inr_mwh" in df.columns:
        df["mcp_inr_mwh"] = pd.to_numeric(df["mcp_inr_mwh"], errors="coerce")
        df["mcp_inr_kwh"] = df["mcp_inr_mwh"] / 1000  # Convert to INR/kWh

    # Sort and clean
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Determine resolution
    if len(df) > 1:
        delta = (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds()
        resolution = "15min" if delta <= 900 else "1h"
    else:
        resolution = "unknown"

    logger.info(
        f"  Parsed: {len(df)} rows, resolution={resolution}, "
        f"MCP range: {df['mcp_inr_mwh'].min():.0f}-{df['mcp_inr_mwh'].max():.0f} INR/MWh"
    )
    return df


def generate_synthetic_prices(
    start_date: date = date(2024, 4, 1),
    end_date: date = date(2025, 3, 31),
    resolution: str = "1h",
    add_noise: bool = True,
    noise_pct: float = 0.12,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic IEX DAM prices from the FY24-25 monthly average matrix.

    Adds realistic noise to the hourly averages to simulate actual market
    volatility. Real IEX prices have ~15-20% intraday volatility around
    the monthly average, with occasional spikes to 2-3× during peak hours.

    Args:
        resolution: "1h" for hourly, "15min" for 15-minute blocks
        add_noise: Add stochastic variation around averages
        noise_pct: Standard deviation of noise (12% = typical IEX volatility)
    """
    rng = np.random.RandomState(seed)

    # Generate timestamp range
    freq = "15min" if resolution == "15min" else "1h"
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
    # Exclude the very last timestamp if it's the end date at midnight next day
    timestamps = timestamps[timestamps <= pd.Timestamp(end_date) + pd.Timedelta(hours=23)]

    rows = []
    for ts in timestamps:
        cal_month = ts.month
        hour = ts.hour

        # Base price from monthly average matrix
        fy_idx = fy_month_index(cal_month)
        base_price = IEX_DAM_PRICES_FY2425[fy_idx][hour]

        if add_noise:
            # Log-normal noise preserves positivity and creates realistic spikes
            noise = rng.lognormal(mean=0, sigma=noise_pct)
            # Occasional price spikes (5% probability of 1.5-3× spike)
            if rng.random() < 0.05:
                noise *= rng.uniform(1.5, 3.0)
            price = base_price * noise
        else:
            price = base_price

        # Price floor (IEX has ₹0/MWh floor) and cap (₹20/kWh ceiling)
        price = max(price, 0.0)
        price = min(price, 20.0)

        rows.append({
            "timestamp": ts,
            "mcp_inr_kwh": round(price, 4),
            "mcp_inr_mwh": round(price * 1000, 2),
            "hour": hour,
            "month": cal_month,
        })

    df = pd.DataFrame(rows)

    logger.info(
        f"Synthetic IEX prices: {len(df)} rows, {resolution} resolution, "
        f"MCP range: {df['mcp_inr_kwh'].min():.2f}-{df['mcp_inr_kwh'].max():.2f} INR/kWh"
    )
    return df


def load_iex_prices(
    csv_dir: Optional[str] = None,
    start_date: date = date(2024, 4, 1),
    end_date: date = date(2025, 3, 31),
    resolution: str = "1h",
) -> pd.DataFrame:
    """
    Load IEX DAM prices — from CSV if available, synthetic otherwise.

    Priority:
    1. If csv_dir has IEX export files → parse and combine them
    2. Otherwise → generate synthetic prices from monthly matrix

    This is the main entry point for the rest of the pipeline.
    """
    if csv_dir:
        csv_path = Path(csv_dir)
        csv_files = sorted(csv_path.glob("*.csv"))

        if csv_files:
            logger.info(f"Found {len(csv_files)} IEX CSV files in {csv_dir}")
            all_dfs = []
            for f in csv_files:
                try:
                    df = parse_iex_csv(str(f))
                    all_dfs.append(df)
                except Exception as e:
                    logger.warning(f"Failed to parse {f}: {e}")

            if all_dfs:
                combined = pd.concat(all_dfs, ignore_index=True)
                combined.sort_values("timestamp", inplace=True)
                combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
                combined.reset_index(drop=True, inplace=True)
                logger.info(f"Loaded {len(combined)} IEX price rows from CSV")
                return combined

    # Fallback to synthetic
    logger.info("No IEX CSVs found — generating synthetic prices from FY24-25 matrix")
    return generate_synthetic_prices(start_date, end_date, resolution)
