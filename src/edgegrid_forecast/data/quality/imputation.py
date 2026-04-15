"""
Imputation strategies for missing and anomalous data.

Three modes:
- interpolate: Time-based interpolation (best for short gaps)
- seasonal: Same interval from previous week
- hybrid: Interpolate short gaps, seasonal for long gaps
"""

from typing import Optional

import numpy as np
import pandas as pd


def _infer_periods_per_week(index: pd.DatetimeIndex) -> int:
    """
    Compute number of periods in 7 days from the index frequency.

    Falls back to 672 (15-min intervals) if frequency cannot be inferred.
    """
    inferred = pd.infer_freq(index)
    if inferred:
        offset = pd.tseries.frequencies.to_offset(inferred)
        freq_ns = offset.nanos
        return int(pd.Timedelta(days=7).total_seconds() * 1e9 / freq_ns)
    return 672  # Default: 7 days × 24 hours × 4 (15-min intervals)


def impute_missing_and_anomalous(
    series: pd.Series,
    anomaly_mask: Optional[pd.Series] = None,
    method: str = "hybrid",
    short_gap_limit: int = 4,
) -> pd.Series:
    """
    Impute missing values and anomalous readings.

    Args:
        series: Original time series (must have DatetimeIndex for seasonal/hybrid)
        anomaly_mask: Boolean mask of values to replace with NaN before imputing
        method: 'interpolate', 'seasonal', or 'hybrid'
        short_gap_limit: Max gap length (in intervals) for interpolation in hybrid mode

    Returns:
        Series with NaN and anomalies filled. Any remaining NaN after all
        strategies are exhausted is left as NaN (never filled with 0).
    """
    if method not in ("interpolate", "seasonal", "hybrid"):
        raise ValueError(f"method must be 'interpolate', 'seasonal', or 'hybrid', got '{method}'")

    result = series.copy()

    if anomaly_mask is not None:
        result[anomaly_mask] = np.nan

    if method == "interpolate":
        result = result.interpolate(method="time", limit=6)
        result = result.ffill(limit=3).bfill(limit=3)

    elif method == "seasonal":
        periods_per_week = _infer_periods_per_week(result.index)
        week_shift = result.shift(periods=periods_per_week)
        still_nan = result.isna()
        result[still_nan] = week_shift[still_nan]
        result = result.interpolate(method="time", limit=6)

    elif method == "hybrid":
        nan_mask = result.isna()
        gap_groups = nan_mask.ne(nan_mask.shift()).cumsum()
        gap_lengths = nan_mask.groupby(gap_groups).transform("sum")

        # Short gaps: interpolate in-place
        short_gap = nan_mask & (gap_lengths < short_gap_limit)
        if short_gap.any():
            # Only interpolate through short gap positions; leave long gaps as NaN
            result_interp = result.copy()
            # Mark long-gap NaNs with a sentinel, interpolate, then restore
            long_gap_nan = nan_mask & ~short_gap
            result_interp[long_gap_nan] = -999_999.0  # Sentinel
            result_interp = result_interp.interpolate(method="time")
            # Copy only short-gap filled values
            result[short_gap] = result_interp[short_gap]
            # Restore long gaps as NaN (sentinel back to NaN)
            # (they're still NaN in `result` since we only copied short_gap positions)

        # Long gaps: seasonal fill (same interval, 7 days ago)
        long_gap = result.isna()
        if long_gap.any():
            periods_per_week = _infer_periods_per_week(result.index)
            week_shift = result.shift(periods=periods_per_week)
            result[long_gap] = week_shift[long_gap]

        # Final pass: interpolate any remaining short gaps from seasonal boundaries
        result = result.interpolate(method="time", limit=6)

    return result
