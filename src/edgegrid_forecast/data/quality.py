"""
Data quality pipeline — clean, impute, and validate smart meter data.

Integrates findings from the meter quality report to handle:
- Missing gaps (interpolation)
- Frozen readings (detection + imputation)
- Outliers (z-score and IQR based)
- Non-stationarity (differencing, detrending)
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.ensemble import IsolationForest


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
        Boolean mask where True = frozen reading
    """
    mask = pd.Series(False, index=series.index)

    # Find consecutive identical values
    diff = series.diff()
    is_same = (diff == 0) & series.notna()

    # Find runs
    run_starts = is_same & ~is_same.shift(1, fill_value=False)
    run_id = run_starts.cumsum()

    # Only flag runs longer than min_run_length
    for rid in run_id[is_same].unique():
        run_mask = run_id == rid
        if run_mask.sum() >= min_run_length:
            mask |= run_mask

    return mask


def detect_outliers_zscore(
    series: pd.Series,
    threshold: float = 3.0,
) -> pd.Series:
    """Detect outliers using z-score method."""
    z_scores = np.abs(stats.zscore(series.dropna()))
    mask = pd.Series(False, index=series.index)
    mask.loc[series.dropna().index] = z_scores > threshold
    return mask


def detect_outliers_iqr(
    series: pd.Series,
    factor: float = 1.5,
) -> pd.Series:
    """Detect outliers using IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (series < lower) | (series > upper)


def detect_outliers_isolation_forest(
    df: pd.DataFrame,
    columns: list,
    contamination: float = 0.05,
) -> pd.Series:
    """
    Detect outliers using Isolation Forest on multivariate features.
    Good for catching anomalies that look normal in individual columns
    but are unusual in combination.
    """
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


def impute_missing_and_anomalous(
    series: pd.Series,
    anomaly_mask: Optional[pd.Series] = None,
    method: str = "interpolate",
) -> pd.Series:
    """
    Impute missing values and anomalous readings.

    Methods:
    - interpolate: Time-based interpolation (best for short gaps)
    - seasonal: Use same hour from previous week
    - hybrid: Interpolate short gaps, seasonal for long gaps

    Args:
        series: Original time series
        anomaly_mask: Boolean mask of values to replace
        method: Imputation strategy
    """
    result = series.copy()

    # Replace anomalous values with NaN for imputation
    if anomaly_mask is not None:
        result[anomaly_mask] = np.nan

    if method == "interpolate":
        result = result.interpolate(method="time", limit=6)
        # Fill remaining with forward/backward fill
        result = result.ffill(limit=3).bfill(limit=3)

    elif method == "seasonal":
        # Use same hour from previous week
        for i in result.index[result.isna()]:
            lookback = i - pd.Timedelta(hours=168)  # 7 days
            if lookback in result.index and pd.notna(result[lookback]):
                result[i] = result[lookback]
        # Fill remaining
        result = result.interpolate(method="time", limit=6)

    elif method == "hybrid":
        # Short gaps (< 3 hours): interpolate
        # Long gaps (>= 3 hours): seasonal
        nan_mask = result.isna()
        # Identify gap lengths
        gap_groups = nan_mask.ne(nan_mask.shift()).cumsum()
        gap_lengths = nan_mask.groupby(gap_groups).transform("sum")

        # Short gaps
        short_gap = nan_mask & (gap_lengths < 3)
        result_short = result.copy()
        result_short[~short_gap & nan_mask] = 0  # Temporarily fill long gaps
        result_short = result_short.interpolate(method="time")
        result[short_gap] = result_short[short_gap]

        # Long gaps — seasonal
        long_gap = nan_mask & (gap_lengths >= 3)
        for i in result.index[long_gap]:
            lookback = i - pd.Timedelta(hours=168)
            if lookback in result.index and pd.notna(result[lookback]):
                result[i] = result[lookback]

        result = result.interpolate(method="time", limit=6)

    return result


def run_quality_pipeline(
    df: pd.DataFrame,
    demand_col: str = "demand_kwh",
    timestamp_col: str = "timestamp",
    consumer_col: str = "consumer_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full quality pipeline: detect issues, impute, return clean data + quality report.

    Returns:
        Tuple of (cleaned DataFrame, quality metrics DataFrame)
    """
    logger.info("Running data quality pipeline")

    cleaned_dfs = []
    quality_records = []

    for cid, group in df.groupby(consumer_col):
        group = group.sort_values(timestamp_col).set_index(timestamp_col)
        series = group[demand_col]

        # Detect issues
        frozen = detect_frozen_readings(series)
        zscore_outliers = detect_outliers_zscore(series)
        iqr_outliers = detect_outliers_iqr(series)

        # Combined anomaly mask
        anomaly_mask = frozen | zscore_outliers | iqr_outliers

        # Impute
        clean_series = impute_missing_and_anomalous(
            series, anomaly_mask=anomaly_mask, method="hybrid"
        )

        # Record quality metrics
        quality_records.append({
            "consumer_id": cid,
            "total_records": len(series),
            "missing_pct": series.isna().mean() * 100,
            "frozen_pct": frozen.mean() * 100,
            "zscore_outlier_pct": zscore_outliers.mean() * 100,
            "iqr_outlier_pct": iqr_outliers.mean() * 100,
            "total_anomaly_pct": anomaly_mask.mean() * 100,
            "imputed_pct": (series.isna() | anomaly_mask).mean() * 100,
        })

        # Build cleaned dataframe
        clean_group = group.copy()
        clean_group[demand_col] = clean_series
        clean_group[f"{demand_col}_original"] = series
        clean_group["is_anomaly"] = anomaly_mask
        clean_group["is_imputed"] = anomaly_mask | series.isna()
        cleaned_dfs.append(clean_group.reset_index())

        logger.info(
            f"  {cid}: {anomaly_mask.mean()*100:.1f}% anomalous, "
            f"{frozen.mean()*100:.1f}% frozen, "
            f"{zscore_outliers.mean()*100:.1f}% z-score outliers"
        )

    cleaned = pd.concat(cleaned_dfs, ignore_index=True)
    quality_metrics = pd.DataFrame(quality_records)

    return cleaned, quality_metrics
