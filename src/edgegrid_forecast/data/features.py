"""
Feature engineering for demand, solar, and price forecasting.

Creates temporal, lag, rolling, and calendar features that capture the patterns
in Indian C&I electricity consumption:
- Strong daily seasonality (office hours vs night)
- Weekly patterns (weekday vs weekend)
- Monthly/seasonal patterns (summer cooling load, monsoon)
- Holiday effects (Diwali, Sankranti, etc.)
- Temperature sensitivity (cooling degree days)
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


def build_forecast_features(
    df: pd.DataFrame,
    target_col: str = "demand_kwh",
    timestamp_col: str = "timestamp",
    group_col: str = "consumer_id",
    lag_hours: List[int] = [1, 2, 3, 6, 12, 24, 48, 168],
    rolling_windows: List[int] = [3, 6, 12, 24, 48, 168],
) -> pd.DataFrame:
    """
    Full feature engineering pipeline. Combines all feature families.

    This is the main entry point — call this to prepare data for model training.
    """
    logger.info(f"Building forecast features for {target_col}")

    df = add_temporal_features(df, timestamp_col)
    df = add_lag_features(df, target_col, lag_hours, group_col)
    df = add_rolling_features(df, target_col, rolling_windows, group_col)
    df = add_price_features(df)
    df = add_consumption_pattern_features(df, target_col, group_col)

    # Drop rows where lag features create NaN (start of series)
    max_lag = max(lag_hours + rolling_windows)
    initial_rows = len(df)
    df = df.dropna(subset=[f"{target_col}_lag_{max(lag_hours)}h"])
    logger.info(f"Dropped {initial_rows - len(df)} rows due to lag features (lookback={max_lag}h)")

    logger.info(f"Final feature set: {len(df)} rows, {len(df.columns)} columns")
    return df
