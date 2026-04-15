"""
M1-F2: Statistical Anomaly Detection.

Frozen readings, z-score, IQR, contextual (time-of-day), rolling baseline,
and multivariate Isolation Forest outlier detection.

All functions follow consistent NaN handling:
- NaN inputs are never flagged as anomalies (they're missing, not anomalous)
- Return type is always pd.Series(dtype=bool) with same index as input
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from ._constants import CONTEXTUAL_MIN_GROUP_SIZE, ROLLING_MIN_PERIODS


def detect_frozen_readings(
    series: pd.Series,
    min_run_length: int = 3,
) -> pd.Series:
    """
    Detect periods where meter readings are frozen (identical consecutive values).

    Args:
        series: Time series of meter readings
        min_run_length: Minimum number of identical consecutive readings to flag

    Returns:
        Boolean mask where True = frozen reading. NaN positions are False.
    """
    if len(series) < min_run_length:
        return pd.Series(False, index=series.index)

    diff = series.diff()
    is_same = (diff == 0) & series.notna()

    run_starts = is_same & ~is_same.shift(1, fill_value=False)
    run_id = run_starts.cumsum()

    # Vectorized: run lengths via groupby.transform, then threshold
    run_lengths = is_same.groupby(run_id).transform("sum")
    return is_same & (run_lengths >= min_run_length)


def detect_outliers_zscore(
    series: pd.Series,
    threshold: float = 3.0,
) -> pd.Series:
    """
    Detect outliers using global z-score method.

    NaN values are excluded from z-score computation and always return False.

    Args:
        series: Numeric series
        threshold: Z-score threshold (default 3.0 = ~0.27% of normal distribution)

    Returns:
        Boolean mask where True = outlier
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    mask = pd.Series(False, index=series.index)
    valid = series.dropna()

    if len(valid) < 3:
        return mask

    z_scores = np.abs(stats.zscore(valid.values))
    mask.loc[valid.index] = z_scores > threshold
    return mask


def detect_outliers_iqr(
    series: pd.Series,
    factor: float = 1.5,
) -> pd.Series:
    """
    Detect outliers using the IQR (Interquartile Range) method.

    NaN values always return False (consistent with other detectors).

    Args:
        series: Numeric series
        factor: IQR multiplier (1.5 = standard, 3.0 = far outliers)

    Returns:
        Boolean mask where True = outlier
    """
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")

    valid = series.dropna()
    if len(valid) < 4:
        return pd.Series(False, index=series.index)

    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    # Explicit NaN handling: NaN comparisons return False, which is correct,
    # but we make it explicit for clarity and consistency
    result = (series < lower) | (series > upper)
    result = result.fillna(False)
    return result


def detect_outliers_contextual(
    series: pd.Series,
    threshold: float = 3.0,
    group_by: str = "hour",
) -> pd.Series:
    """
    Contextual anomaly detection — z-score within time-of-day bands.

    A value that's normal at 2pm (peak hours) might be anomalous at 2am.

    Args:
        series: Time-indexed series (must have DatetimeIndex)
        threshold: Z-score threshold for flagging
        group_by: 'hour' (24 groups) or 'hour_dow' (168 groups: hour × day-of-week)

    Returns:
        Boolean mask where True = contextual outlier
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    if not isinstance(series.index, pd.DatetimeIndex):
        return pd.Series(False, index=series.index)

    if group_by == "hour_dow":
        groups = series.index.hour * 10 + series.index.dayofweek
    else:
        groups = series.index.hour

    # Vectorized: z-scores within each group via groupby.transform
    grouped = series.groupby(groups)
    group_mean = grouped.transform("mean")
    group_std = grouped.transform("std")
    group_count = grouped.transform("count")

    valid = (
        (group_count >= CONTEXTUAL_MIN_GROUP_SIZE)
        & (group_std > 0)
        & series.notna()
    )
    z_scores = ((series - group_mean) / group_std).abs()

    return valid & (z_scores > threshold)


def detect_outliers_rolling(
    series: pd.Series,
    window: str = "48h",
    threshold: float = 3.0,
) -> pd.Series:
    """
    Rolling baseline z-score — compares each point to its local context.

    Better than global z-score for data with trends, seasonality, or regime changes.
    Edge intervals (where the rolling window has < min_periods points) return False.

    Args:
        series: Time-indexed series
        window: Rolling window size (e.g., '48h')
        threshold: Z-score threshold

    Returns:
        Boolean mask where True = rolling outlier
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    rolling_median = series.rolling(
        window, center=True, min_periods=ROLLING_MIN_PERIODS
    ).median()
    rolling_std = series.rolling(
        window, center=True, min_periods=ROLLING_MIN_PERIODS
    ).std()

    # Safe division: where std is 0 or NaN, z-score is undefined → not an outlier
    safe_std = rolling_std.where(rolling_std > 0)
    z = ((series - rolling_median) / safe_std).abs()

    # Explicit: edges and NaN → False (not flagged)
    return (z > threshold).fillna(False)


def detect_outliers_isolation_forest(
    df: pd.DataFrame,
    columns: List[str],
    contamination: float = 0.05,
) -> pd.Series:
    """
    Multivariate outlier detection using Isolation Forest.

    Catches anomalies that look normal individually but are unusual in combination
    (e.g., high kW AND high kVAR simultaneously when they're normally anticorrelated).

    Args:
        df: DataFrame with feature columns
        columns: Column names to use for multivariate detection
        contamination: Expected fraction of outliers (0.0 to 0.5)

    Returns:
        Boolean mask where True = multivariate outlier
    """
    if not 0.0 < contamination < 0.5:
        raise ValueError(f"contamination must be in (0, 0.5), got {contamination}")

    features = df[columns].dropna()
    if len(features) < 10:
        return pd.Series(False, index=df.index)

    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    predictions = iso_forest.fit_predict(features)
    mask = pd.Series(False, index=df.index)
    mask.loc[features.index] = predictions == -1
    return mask
