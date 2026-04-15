"""
M1-F1: Smart Meter AMI Ingestion.

Gap detection, duplicate handling, late arrival management, multi-channel sync,
physical range validation, consistency checks, and per-interval quality scoring.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ._constants import (
    AMI_CHANNELS,
    CHANNEL_RANGES,
    QUALITY_WEIGHTS,
    TIMELINESS_DEGRADE_END_MIN,
    TIMELINESS_DEGRADE_START_MIN,
    TIMELINESS_MIN_SCORE,
)


def detect_gaps(
    series: pd.Series,
    freq: str = "15min",
) -> pd.DataFrame:
    """
    Find missing intervals in a time-indexed series.

    Args:
        series: Time-indexed series (index must be DatetimeIndex)
        freq: Expected frequency of readings

    Returns:
        DataFrame with columns: gap_start, gap_end, gap_intervals, gap_duration_min
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex")

    empty = pd.DataFrame(columns=["gap_start", "gap_end", "gap_intervals", "gap_duration_min"])

    if len(series) < 2:
        return empty

    expected = pd.date_range(series.index.min(), series.index.max(), freq=freq)
    missing = expected.difference(series.dropna().index)

    if len(missing) == 0:
        return empty

    # Vectorized gap grouping: consecutive missing timestamps form a single gap
    freq_delta = pd.Timedelta(freq)
    diffs = pd.Series(missing).diff()
    # A new gap starts wherever the diff exceeds one frequency step
    gap_id = (diffs > freq_delta).cumsum()

    gap_starts = pd.Series(missing).groupby(gap_id).first()
    gap_ends = pd.Series(missing).groupby(gap_id).last()
    gap_counts = pd.Series(missing).groupby(gap_id).count()

    return pd.DataFrame({
        "gap_start": gap_starts.values,
        "gap_end": gap_ends.values,
        "gap_intervals": gap_counts.values,
        "gap_duration_min": gap_counts.values * (freq_delta.total_seconds() / 60),
    })


def handle_duplicates(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    meter_col: str = "meter_id",
    arrival_col: Optional[str] = "arrival_timestamp",
) -> pd.DataFrame:
    """
    Deduplicate AMI packets. Keep the latest arrival for each (meter, timestamp) pair.

    Always operates on a copy — never mutates the input DataFrame.
    """
    n_before = len(df)
    df = df.copy()

    if arrival_col and arrival_col in df.columns:
        df = df.sort_values(arrival_col, ascending=False)
        df = df.drop_duplicates(subset=[meter_col, timestamp_col], keep="first")
    else:
        df = df.drop_duplicates(subset=[meter_col, timestamp_col], keep="first")

    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info(f"Removed {n_removed} duplicate packets ({n_removed / n_before * 100:.1f}%)")

    return df.sort_values([meter_col, timestamp_col]).reset_index(drop=True)


def handle_late_arrivals(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    arrival_col: str = "arrival_timestamp",
    max_delay_intervals: int = 8,
    freq_minutes: int = 15,
) -> pd.DataFrame:
    """
    Flag late-arriving packets. Accept within window, reject beyond.

    A packet is 'late' if arrival_timestamp - timestamp > 2 × freq.
    Packets beyond max_delay_intervals × freq are dropped entirely.
    """
    if arrival_col not in df.columns:
        df = df.copy()
        df["is_late"] = False
        return df

    max_delay = pd.Timedelta(minutes=max_delay_intervals * freq_minutes)
    delay = pd.to_datetime(df[arrival_col]) - pd.to_datetime(df[timestamp_col])

    df = df.copy()
    df["packet_delay_min"] = delay.dt.total_seconds() / 60
    df["is_late"] = delay > pd.Timedelta(minutes=2 * freq_minutes)

    excessive = delay > max_delay
    n_dropped = excessive.sum()
    if n_dropped > 0:
        logger.warning(f"Dropping {n_dropped} packets with delay > {max_delay_intervals} intervals")
        df = df[~excessive]

    return df


def sync_channels(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    meter_col: str = "meter_id",
    required_channels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Ensure multi-channel alignment — all required channels present for each interval.

    If any required channel has NaN for an interval, that interval is flagged incomplete.
    If a required channel column doesn't exist at all, it's added as NaN.
    """
    if required_channels is None:
        required_channels = [ch for ch in AMI_CHANNELS if ch in df.columns]

    if not required_channels:
        df = df.copy()
        df["channels_complete"] = True
        return df

    df = df.copy()

    # Add missing channel columns
    for ch in required_channels:
        if ch not in df.columns:
            df[ch] = np.nan
            logger.warning(f"Required channel '{ch}' not in data — added as NaN")

    # Vectorized: all required channels must be non-NaN
    df["channels_complete"] = df[required_channels].notna().all(axis=1)

    incomplete_pct = (~df["channels_complete"]).mean() * 100
    if incomplete_pct > 0:
        logger.info(f"Channel sync: {incomplete_pct:.1f}% intervals incomplete")

    return df


def validate_physical_ranges(
    df: pd.DataFrame,
    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Check that each channel's values fall within physically plausible ranges.

    NaN values pass validation (no data ≠ bad data).
    Values outside range are flagged (not removed).
    """
    if ranges is None:
        ranges = CHANNEL_RANGES

    df = df.copy()
    for ch, (lo, hi) in ranges.items():
        if ch in df.columns:
            valid = ((df[ch] >= lo) & (df[ch] <= hi)) | df[ch].isna()
            df[f"{ch}_range_valid"] = valid
            invalid_pct = (~valid).mean() * 100
            if invalid_pct > 0:
                logger.warning(f"  {ch}: {invalid_pct:.1f}% out of range [{lo}, {hi}]")
        else:
            logger.debug(f"  Channel '{ch}' not in DataFrame — skipping range check")

    return df


def check_physical_consistency(
    df: pd.DataFrame,
    tolerance: float = 0.05,
) -> pd.DataFrame:
    """
    Check that channels are physically consistent with each other.

    Rules:
    - kVA ≈ √(kW² + kVAR²) within tolerance
    - PF ≈ kW / kVA within tolerance

    Division-safe: deviation is only computed where denominator > 0.
    """
    df = df.copy()
    df["physically_consistent"] = True

    # Rule 1: kVA ≈ √(kW² + kVAR²)
    if all(ch in df.columns for ch in ["kw", "kvar", "kva"]):
        computed_kva = np.sqrt(df["kw"] ** 2 + df["kvar"] ** 2)
        mask = df["kva"].notna() & (df["kva"] > 0)
        # Compute deviation only where safe (denominator > 0)
        deviation = pd.Series(np.nan, index=df.index)
        deviation[mask] = np.abs(computed_kva[mask] - df["kva"][mask]) / df["kva"][mask]
        df.loc[mask, "physically_consistent"] &= deviation[mask] <= tolerance

    # Rule 2: PF ≈ kW / kVA
    if all(ch in df.columns for ch in ["kw", "kva", "pf"]):
        mask = df["kva"].notna() & (df["kva"] > 0) & df["pf"].notna()
        computed_pf = pd.Series(np.nan, index=df.index)
        computed_pf[mask] = df["kw"][mask] / df["kva"][mask]
        deviation = pd.Series(np.nan, index=df.index)
        deviation[mask] = np.abs(computed_pf[mask] - df["pf"][mask])
        df.loc[mask, "physically_consistent"] &= deviation[mask] <= tolerance

    return df


def compute_interval_quality_score(
    df: pd.DataFrame,
    required_channels: Optional[List[str]] = None,
) -> pd.Series:
    """
    Compute a 0-1 quality score per interval.

    Score = weighted sum of:
    - Completeness (0.4): fraction of required channels present
    - Timeliness (0.3): 1.0 if on-time, degrades linearly for late
    - Validity (0.2): fraction of channels within physical range
    - Consistency (0.1): physical consistency check passes
    """
    if required_channels is None:
        required_channels = [ch for ch in AMI_CHANNELS if ch in df.columns]

    scores = pd.DataFrame(index=df.index)

    # Completeness
    if required_channels:
        present = sum(df[ch].notna().astype(float) for ch in required_channels if ch in df.columns)
        scores["completeness"] = present / len(required_channels)
    else:
        scores["completeness"] = 1.0

    # Timeliness
    if "is_late" in df.columns:
        scores["timeliness"] = (~df["is_late"]).astype(float)
        if "packet_delay_min" in df.columns:
            late_mask = df["is_late"]
            delay_min = df.loc[late_mask, "packet_delay_min"].clip(
                TIMELINESS_DEGRADE_START_MIN, TIMELINESS_DEGRADE_END_MIN
            )
            degrade_range = TIMELINESS_DEGRADE_END_MIN - TIMELINESS_DEGRADE_START_MIN
            scores.loc[late_mask, "timeliness"] = (
                1.0 - (1.0 - TIMELINESS_MIN_SCORE)
                * (delay_min - TIMELINESS_DEGRADE_START_MIN) / degrade_range
            )
    else:
        scores["timeliness"] = 1.0

    # Validity
    range_cols = [c for c in df.columns if c.endswith("_range_valid")]
    if range_cols:
        scores["validity"] = sum(df[c].astype(float) for c in range_cols) / len(range_cols)
    else:
        scores["validity"] = 1.0

    # Consistency
    if "physically_consistent" in df.columns:
        scores["consistency"] = df["physically_consistent"].astype(float)
    else:
        scores["consistency"] = 1.0

    # Weighted sum
    quality = (
        QUALITY_WEIGHTS["completeness"] * scores["completeness"]
        + QUALITY_WEIGHTS["timeliness"] * scores["timeliness"]
        + QUALITY_WEIGHTS["validity"] * scores["validity"]
        + QUALITY_WEIGHTS["consistency"] * scores["consistency"]
    )

    return quality
