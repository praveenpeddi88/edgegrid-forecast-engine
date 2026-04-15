"""
M1 — Data Quality Engine for Indian distribution grid data.

Handles India-specific data quality problems that generic pipelines miss:
- AMI packet loss (15-40%), duplicates, late arrivals, multi-channel sync
- Voltage-compensated SOC correction (±10-15% grid voltage deviation)
- CT metering artefacts from 2-4 Hz frequency swings
- DG transition detection and exclusion from training data
- APFC switching events that mimic demand curtailment

PRD Reference: Module M1, Features M1-F1 through M1-F6
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# ─────────────────────────────────────────────────────────────────────────────
# M1-F1: Smart Meter AMI Ingestion
# ─────────────────────────────────────────────────────────────────────────────

# Standard AMI channels for HT consumers
AMI_CHANNELS = ["kw", "kvar", "kva", "voltage", "current", "pf"]

# Quality score weights for each dimension
QUALITY_WEIGHTS = {
    "completeness": 0.4,    # All channels present
    "timeliness": 0.3,      # Arrived within expected window
    "validity": 0.2,        # Passes range checks
    "consistency": 0.1,     # Channels are physically consistent
}

# Physical range limits for Indian HT consumers (11kV / 33kV)
CHANNEL_RANGES = {
    "kw": (0, 50_000),         # 0 to 50 MW (largest HT consumers)
    "kvar": (-20_000, 20_000), # Reactive power can be negative (leading PF)
    "kva": (0, 50_000),        # Apparent power always positive
    "voltage": (300, 500),     # 415V nominal ±20% (330-500V realistic)
    "current": (0, 5000),      # Amps
    "pf": (0.0, 1.0),         # Power factor 0 to 1
}


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

    if len(series) < 2:
        return pd.DataFrame(columns=["gap_start", "gap_end", "gap_intervals", "gap_duration_min"])

    expected = pd.date_range(series.index.min(), series.index.max(), freq=freq)
    missing = expected.difference(series.dropna().index)

    if len(missing) == 0:
        return pd.DataFrame(columns=["gap_start", "gap_end", "gap_intervals", "gap_duration_min"])

    # Group consecutive missing intervals into gaps
    freq_delta = pd.Timedelta(freq)
    gaps = []
    gap_start = missing[0]
    prev = missing[0]

    for ts in missing[1:]:
        if ts - prev > freq_delta:
            gap_intervals = int((prev - gap_start) / freq_delta) + 1
            gaps.append({
                "gap_start": gap_start,
                "gap_end": prev,
                "gap_intervals": gap_intervals,
                "gap_duration_min": gap_intervals * (freq_delta.total_seconds() / 60),
            })
            gap_start = ts
        prev = ts

    # Final gap
    gap_intervals = int((prev - gap_start) / freq_delta) + 1
    gaps.append({
        "gap_start": gap_start,
        "gap_end": prev,
        "gap_intervals": gap_intervals,
        "gap_duration_min": gap_intervals * (freq_delta.total_seconds() / 60),
    })

    return pd.DataFrame(gaps)


def handle_duplicates(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    meter_col: str = "meter_id",
    arrival_col: Optional[str] = "arrival_timestamp",
) -> pd.DataFrame:
    """
    Deduplicate AMI packets. Keep the latest arrival for each (meter, timestamp) pair.

    Args:
        df: Raw AMI data
        timestamp_col: Column with the measurement timestamp
        meter_col: Column with meter identifier
        arrival_col: Column with packet arrival time (if available)

    Returns:
        Deduplicated DataFrame
    """
    n_before = len(df)

    if arrival_col and arrival_col in df.columns:
        # Keep the latest arrival for each (meter, timestamp)
        df = df.sort_values(arrival_col, ascending=False)
        df = df.drop_duplicates(subset=[meter_col, timestamp_col], keep="first")
    else:
        # No arrival time — keep first occurrence
        df = df.drop_duplicates(subset=[meter_col, timestamp_col], keep="first")

    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info(f"Removed {n_removed} duplicate packets ({n_removed/n_before*100:.1f}%)")

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

    A packet is 'late' if arrival_timestamp - timestamp > max_delay_intervals × freq.
    Late packets within window are flagged but kept. Beyond window, they're dropped.

    Args:
        df: AMI data with both measurement and arrival timestamps
        max_delay_intervals: Maximum acceptable delay in number of intervals
        freq_minutes: Interval frequency in minutes

    Returns:
        DataFrame with 'is_late' flag column, excessively late packets removed
    """
    if arrival_col not in df.columns:
        df["is_late"] = False
        return df

    max_delay = pd.Timedelta(minutes=max_delay_intervals * freq_minutes)
    delay = pd.to_datetime(df[arrival_col]) - pd.to_datetime(df[timestamp_col])

    df = df.copy()
    df["packet_delay_min"] = delay.dt.total_seconds() / 60
    df["is_late"] = delay > pd.Timedelta(minutes=2 * freq_minutes)  # Late = >2 intervals

    # Drop packets that are excessively late (beyond max window)
    excessive = delay > max_delay
    n_dropped = excessive.sum()
    if n_dropped > 0:
        logger.warning(
            f"Dropping {n_dropped} packets with delay > {max_delay_intervals} intervals"
        )
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

    If any required channel is missing for an interval, the entire interval is flagged
    as incomplete. Missing channels are filled with NaN.

    Args:
        df: AMI data with channel columns
        required_channels: List of channel column names that must all be present

    Returns:
        DataFrame with 'channels_complete' boolean column
    """
    if required_channels is None:
        required_channels = [ch for ch in AMI_CHANNELS if ch in df.columns]

    if not required_channels:
        df["channels_complete"] = True
        return df

    df = df.copy()
    df["channels_complete"] = True

    for ch in required_channels:
        if ch in df.columns:
            df["channels_complete"] &= df[ch].notna()
        else:
            df[ch] = np.nan
            df["channels_complete"] = False

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

    Values outside range are flagged (not removed) — downstream logic decides
    whether to impute or exclude.

    Returns:
        DataFrame with '{channel}_range_valid' boolean columns
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
    - kVA ≈ √3 × V × I / 1000 within tolerance (for 3-phase)

    Returns:
        DataFrame with 'physically_consistent' boolean column
    """
    df = df.copy()
    df["physically_consistent"] = True

    # Rule 1: kVA ≈ √(kW² + kVAR²)
    if all(ch in df.columns for ch in ["kw", "kvar", "kva"]):
        computed_kva = np.sqrt(df["kw"] ** 2 + df["kvar"] ** 2)
        mask = df["kva"].notna() & (df["kva"] > 0)
        deviation = np.abs(computed_kva - df["kva"]) / df["kva"]
        df.loc[mask, "physically_consistent"] &= deviation[mask] <= tolerance

    # Rule 2: PF ≈ kW / kVA
    if all(ch in df.columns for ch in ["kw", "kva", "pf"]):
        mask = df["kva"].notna() & (df["kva"] > 0) & df["pf"].notna()
        computed_pf = df["kw"] / df["kva"]
        deviation = np.abs(computed_pf - df["pf"])
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
    - Timeliness (0.3): 1.0 if on-time, degraded for late
    - Validity (0.2): fraction of channels within physical range
    - Consistency (0.1): physical consistency check passes

    Returns:
        Series of quality scores (0.0 to 1.0)
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
        # Partial credit for late but within window
        if "packet_delay_min" in df.columns:
            late_mask = df["is_late"]
            # Linearly degrade from 1.0 at 30min delay to 0.5 at 120min delay
            delay_min = df.loc[late_mask, "packet_delay_min"].clip(30, 120)
            scores.loc[late_mask, "timeliness"] = 1.0 - 0.5 * (delay_min - 30) / 90
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


# ─────────────────────────────────────────────────────────────────────────────
# M1-F2: Statistical Anomaly Detection (enhanced from original)
# ─────────────────────────────────────────────────────────────────────────────

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
    diff = series.diff()
    is_same = (diff == 0) & series.notna()

    run_starts = is_same & ~is_same.shift(1, fill_value=False)
    run_id = run_starts.cumsum()

    # Vectorized: compute run lengths via groupby.transform, then threshold
    run_lengths = is_same.groupby(run_id).transform("sum")
    mask = is_same & (run_lengths >= min_run_length)

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


def detect_outliers_contextual(
    series: pd.Series,
    threshold: float = 3.0,
    group_by: str = "hour",
) -> pd.Series:
    """
    Contextual anomaly detection — z-score within time-of-day bands.

    A value that's normal at 2pm (peak hours) might be anomalous at 2am.
    This groups by hour-of-day and computes z-scores within each group.

    Args:
        series: Time-indexed series
        threshold: Z-score threshold for flagging
        group_by: Grouping method — 'hour' or 'hour_dow' (hour + day of week)

    Returns:
        Boolean mask where True = contextual outlier
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        return pd.Series(False, index=series.index)

    mask = pd.Series(False, index=series.index)

    if group_by == "hour":
        groups = series.index.hour
    elif group_by == "hour_dow":
        groups = series.index.hour * 10 + series.index.dayofweek
    else:
        groups = series.index.hour

    # Vectorized: compute z-scores within each group using groupby.transform
    grouped = series.groupby(groups)
    group_mean = grouped.transform("mean")
    group_std = grouped.transform("std")
    group_count = grouped.transform("count")

    # Only flag groups with enough data (≥5) and non-zero std
    valid = (group_count >= 5) & (group_std > 0) & series.notna()
    z_scores = ((series - group_mean) / group_std).abs()
    mask = valid & (z_scores > threshold)

    return mask


def detect_outliers_rolling(
    series: pd.Series,
    window: str = "48h",
    threshold: float = 3.0,
) -> pd.Series:
    """
    Rolling baseline z-score — compares each point to its local 48h context.

    Better than global z-score for data with trends, seasonality, or regime changes.

    Args:
        series: Time-indexed series
        window: Rolling window size
        threshold: Z-score threshold

    Returns:
        Boolean mask where True = rolling outlier
    """
    rolling_median = series.rolling(window, center=True, min_periods=10).median()
    rolling_std = series.rolling(window, center=True, min_periods=10).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    z = np.abs((series - rolling_median) / rolling_std)
    return z > threshold


def detect_outliers_isolation_forest(
    df: pd.DataFrame,
    columns: list,
    contamination: float = 0.05,
) -> pd.Series:
    """
    Detect outliers using Isolation Forest on multivariate features.
    Good for catching anomalies that look normal individually but are unusual in combination.
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


# ─────────────────────────────────────────────────────────────────────────────
# M1-F3: Voltage Compensation / SOC Correction
# ─────────────────────────────────────────────────────────────────────────────

class VoltageSOCCorrector:
    """
    Corrects BMS-reported SOC for Indian grid voltage deviation.

    Indian grid voltage deviates ±10-15% from nominal (380-440V on 415V nominal).
    BMS calculates SOC from terminal voltage using a curve calibrated at nominal.
    When grid voltage is high, BMS over-reports SOC. When low, it under-reports.

    PRD requirement: corrected SOC within ±2% of lab-calibrated reference.
    Calibration period: 30-60 days of operational data per site.

    Usage:
        corrector = VoltageSOCCorrector(nominal_voltage=415.0)
        corrector.calibrate(voltage_data, bms_soc_data, actual_soc_data)
        corrected = corrector.correct(new_voltage, new_bms_soc)
    """

    def __init__(
        self,
        nominal_voltage: float = 415.0,
        polynomial_degree: int = 2,
        min_calibration_points: int = 100,
    ):
        self.nominal_voltage = nominal_voltage
        self.polynomial_degree = polynomial_degree
        self.min_calibration_points = min_calibration_points
        self.model: Optional[LinearRegression] = None
        self.poly_features: Optional[PolynomialFeatures] = None
        self.is_calibrated = False
        self.calibration_stats: Dict = {}

    def detect_known_state_periods(
        self,
        soc_series: pd.Series,
        current_series: pd.Series,
        rest_duration_min: int = 30,
    ) -> pd.DataFrame:
        """
        Find periods where actual SOC can be reliably estimated:
        - Full charge: SOC > 98% AND current ≈ 0 (trickle/float)
        - Full discharge: SOC < 5% AND current ≈ 0
        - Rest: current ≈ 0 for > rest_duration_min (OCV = f(SOC) is reliable at rest)

        These periods provide ground truth for calibration.

        Returns:
            DataFrame with columns: start, end, state_type, duration_min
        """
        # Define "approximately zero" current (< 2% of max observed)
        current_threshold = current_series.abs().quantile(0.95) * 0.02
        is_rest = current_series.abs() < current_threshold

        periods = []
        freq_seconds = 900  # Default 15 min
        if isinstance(soc_series.index, pd.DatetimeIndex) and len(soc_series) > 1:
            freq_seconds = (soc_series.index[1] - soc_series.index[0]).total_seconds()

        # Find rest periods
        rest_starts = is_rest & ~is_rest.shift(1, fill_value=False)
        rest_id = rest_starts.cumsum()

        for rid in rest_id[is_rest].unique():
            run = is_rest.index[rest_id == rid]
            if len(run) == 0:
                continue
            duration_min = len(run) * freq_seconds / 60

            if duration_min >= rest_duration_min:
                mean_soc = soc_series.loc[run].mean()

                if mean_soc > 98:
                    state_type = "full_charge"
                elif mean_soc < 5:
                    state_type = "full_discharge"
                else:
                    state_type = "rest"

                periods.append({
                    "start": run[0],
                    "end": run[-1],
                    "state_type": state_type,
                    "duration_min": duration_min,
                    "mean_soc": mean_soc,
                })

        return pd.DataFrame(periods)

    def calibrate(
        self,
        voltage: pd.Series,
        bms_soc: pd.Series,
        actual_soc: pd.Series,
    ) -> Dict:
        """
        Build site-specific voltage→SOC_error correction model.

        Args:
            voltage: Grid voltage readings (V)
            bms_soc: BMS-reported SOC (%)
            actual_soc: Lab-calibrated or known-state SOC (%)

        Returns:
            Dict with calibration metrics
        """
        # Compute SOC error
        soc_error = bms_soc - actual_soc

        # Remove NaN
        valid = voltage.notna() & soc_error.notna()
        v = voltage[valid].values.reshape(-1, 1)
        e = soc_error[valid].values

        effective_degree = self.polynomial_degree
        if len(v) < self.min_calibration_points:
            logger.warning(
                f"Only {len(v)} calibration points (need {self.min_calibration_points}). "
                "Using linear model — collect more data for polynomial."
            )
            effective_degree = 1

        # Voltage deviation from nominal as feature
        v_deviation = v - self.nominal_voltage

        # Polynomial features
        self.poly_features = PolynomialFeatures(
            degree=effective_degree, include_bias=False
        )
        v_poly = self.poly_features.fit_transform(v_deviation)

        # Fit linear regression on polynomial features
        self.model = LinearRegression()
        self.model.fit(v_poly, e)

        # Compute calibration metrics
        predictions = self.model.predict(v_poly)
        residuals = e - predictions

        self.calibration_stats = {
            "n_points": len(v),
            "mean_error_before": float(np.mean(np.abs(e))),
            "mean_residual_after": float(np.mean(np.abs(residuals))),
            "max_residual": float(np.max(np.abs(residuals))),
            "voltage_range": (float(voltage[valid].min()), float(voltage[valid].max())),
            "r_squared": float(self.model.score(v_poly, e)),
        }

        self.is_calibrated = True

        logger.info(
            f"SOC correction calibrated: {self.calibration_stats['n_points']} points, "
            f"R²={self.calibration_stats['r_squared']:.3f}, "
            f"mean |residual|={self.calibration_stats['mean_residual_after']:.2f}%"
        )

        return self.calibration_stats

    def correct(
        self,
        voltage: pd.Series,
        bms_soc: pd.Series,
    ) -> pd.Series:
        """
        Apply voltage-SOC correction: corrected_SOC = BMS_SOC - predicted_error.

        Args:
            voltage: Current grid voltage readings
            bms_soc: Current BMS-reported SOC

        Returns:
            Corrected SOC series, clipped to [0, 100]
        """
        if not self.is_calibrated:
            logger.warning("SOC corrector not calibrated — returning BMS SOC unchanged")
            return bms_soc

        valid = voltage.notna() & bms_soc.notna()
        v_deviation = (voltage[valid].values - self.nominal_voltage).reshape(-1, 1)
        v_poly = self.poly_features.transform(v_deviation)

        predicted_error = self.model.predict(v_poly)

        corrected = bms_soc.copy()
        corrected[valid] = bms_soc[valid] - predicted_error

        # Clip to physical bounds
        corrected = corrected.clip(0, 100)

        return corrected

    def check_calibration_drift(
        self,
        recent_residuals: pd.Series,
        threshold: float = 2.0,
        consecutive_days: int = 7,
    ) -> bool:
        """
        Check if correction model has drifted and needs re-calibration.

        Returns True if re-calibration is needed: mean |residual| > threshold
        for consecutive_days in a row.
        """
        if len(recent_residuals) == 0:
            return False

        # Group by day, compute daily mean absolute residual
        daily = recent_residuals.abs().resample("D").mean()
        exceeds = daily > threshold

        # Check for consecutive days above threshold
        if len(exceeds) < consecutive_days:
            return exceeds.all()

        # Rolling sum of consecutive exceedances
        rolling = exceeds.rolling(consecutive_days).sum()
        return (rolling >= consecutive_days).any()


# ─────────────────────────────────────────────────────────────────────────────
# M1-F4: Demand Signal Noise Filter
# ─────────────────────────────────────────────────────────────────────────────

class DemandNoiseFilter:
    """
    Filters CT metering artefacts from Indian substation demand data.

    Indian substations experience 2-4 Hz frequency swings that corrupt CT readings,
    producing apparent kVA spikes that don't reflect real load changes. Additionally,
    high-impedance faults and capacitor switching on the 11kV feeder create transient
    distortions.

    PRD detection logic:
    1. Rolling median on 15-min kVA (window = 5 intervals = 75 min)
    2. Deviation > 3σ from 48h rolling baseline AND frequency outside 49.5-50.5 Hz
    3. Secondary: if kW stable but kVA spikes → PF artefact, not real load
    """

    def __init__(
        self,
        rolling_window: str = "75min",
        baseline_window: str = "48h",
        sigma_threshold: float = 3.0,
        freq_band: Tuple[float, float] = (49.5, 50.5),
    ):
        self.rolling_window = rolling_window
        self.baseline_window = baseline_window
        self.sigma_threshold = sigma_threshold
        self.freq_band = freq_band

    def compute_rolling_baseline(
        self,
        kva_series: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute rolling median and σ of kVA readings over the baseline window.

        Returns:
            Tuple of (rolling_median, rolling_sigma)
        """
        rolling_median = kva_series.rolling(
            self.baseline_window, center=False, min_periods=10
        ).median()
        rolling_sigma = kva_series.rolling(
            self.baseline_window, center=False, min_periods=10
        ).std()

        return rolling_median, rolling_sigma

    def detect_ct_artefacts(
        self,
        kva: pd.Series,
        kw: pd.Series,
        frequency: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Detect CT metering artefacts using rolling baseline + frequency correlation.

        An interval is flagged as a CT artefact if:
        1. kVA deviation > sigma_threshold × rolling_sigma from rolling_median
        2. AND grid frequency is outside the normal band (49.5-50.5 Hz)

        If frequency data is not available, only condition 1 is applied (higher
        false positive rate accepted).

        Returns:
            Boolean mask where True = CT artefact
        """
        rolling_median, rolling_sigma = self.compute_rolling_baseline(kva)

        # Avoid division by zero
        rolling_sigma = rolling_sigma.replace(0, np.nan)

        # Condition 1: kVA deviation exceeds threshold
        deviation = np.abs(kva - rolling_median) / rolling_sigma
        exceeds_sigma = deviation > self.sigma_threshold

        # Condition 2: frequency outside normal band
        if frequency is not None and len(frequency) > 0:
            freq_abnormal = (frequency < self.freq_band[0]) | (frequency > self.freq_band[1])
            ct_artefact = exceeds_sigma & freq_abnormal
        else:
            # Without frequency data, use a higher threshold to reduce false positives
            ct_artefact = deviation > (self.sigma_threshold + 1.0)
            logger.debug(
                "No frequency data — using elevated threshold for CT artefact detection"
            )

        n_flagged = ct_artefact.sum()
        if n_flagged > 0:
            logger.info(f"CT artefacts detected: {n_flagged} intervals ({n_flagged/len(kva)*100:.2f}%)")

        return ct_artefact

    def detect_pf_artefacts(
        self,
        kva: pd.Series,
        kw: pd.Series,
        pf: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Detect power factor artefacts: kVA spike with stable kW = PF transient.

        When APFC panels or grid-side capacitors switch, kVAR changes rapidly while
        kW stays constant. This causes kVA to spike/drop without real load change.

        Returns:
            Boolean mask where True = PF artefact
        """
        # kVA deviation from 5-interval rolling median
        kva_median = kva.rolling(self.rolling_window, center=True, min_periods=3).median()
        kva_sigma = kva.rolling(self.baseline_window, center=False, min_periods=10).std()
        kva_sigma = kva_sigma.replace(0, np.nan)

        kva_spike = np.abs(kva - kva_median) / kva_sigma > self.sigma_threshold

        # kW stability: deviation < 1σ
        kw_sigma = kw.rolling(self.baseline_window, center=False, min_periods=10).std()
        kw_sigma = kw_sigma.replace(0, np.nan)
        kw_median = kw.rolling(self.rolling_window, center=True, min_periods=3).median()
        kw_stable = np.abs(kw - kw_median) / kw_sigma < 1.0

        pf_artefact = kva_spike & kw_stable

        n_flagged = pf_artefact.sum()
        if n_flagged > 0:
            logger.info(f"PF artefacts detected: {n_flagged} intervals ({n_flagged/len(kva)*100:.2f}%)")

        return pf_artefact

    def clean_demand_signal(
        self,
        kva: pd.Series,
        kw: pd.Series,
        frequency: Optional[pd.Series] = None,
        pf: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Clean demand signal by replacing artefacts with rolling median.

        Returns:
            Tuple of (cleaned_kva, artefact_mask)
        """
        ct_mask = self.detect_ct_artefacts(kva, kw, frequency)
        pf_mask = self.detect_pf_artefacts(kva, kw, pf)

        artefact_mask = ct_mask | pf_mask

        # Replace artefacts with rolling median
        kva_median = kva.rolling(self.rolling_window, center=True, min_periods=3).median()
        cleaned = kva.copy()
        cleaned[artefact_mask] = kva_median[artefact_mask]

        # Fill any remaining NaN from edges
        cleaned = cleaned.ffill().bfill()

        total_cleaned = artefact_mask.sum()
        if total_cleaned > 0:
            logger.info(
                f"Demand signal cleaned: {total_cleaned} intervals replaced "
                f"({total_cleaned/len(kva)*100:.2f}%)"
            )

        return cleaned, artefact_mask


# ─────────────────────────────────────────────────────────────────────────────
# M1-F5: DG Transition Detection
# ─────────────────────────────────────────────────────────────────────────────

class DGEventType(Enum):
    """Types of DG-related events."""
    GRID_TO_DG = "grid_to_dg"
    DG_TO_GRID = "dg_to_grid"
    DG_RUNNING = "dg_running"


class DGTransitionDetector:
    """
    Detects diesel generator (DG) transitions in grid import data.

    Many C&I consumers in India have DG backup. When grid fails, grid import
    drops to near-zero but site load continues on DG. These periods MUST be
    excluded from demand forecasting training data and DR baselines — they
    represent supply-side events, not demand-side behavior.

    PRD detection logic:
    1. Grid import drops to <5% of rolling baseline within 1 interval
    2. Site load remains >50% of baseline (confirming DG is running)
    3. Voltage signature change: DG voltage is more variable (±5%) than grid
    4. All intervals from grid→DG to DG→grid are marked as DG period
    """

    def __init__(
        self,
        import_drop_threshold_pct: float = 5.0,
        load_continuation_threshold_pct: float = 50.0,
        baseline_window: str = "48h",
        voltage_variability_threshold: float = 0.05,
    ):
        self.import_drop_threshold_pct = import_drop_threshold_pct
        self.load_continuation_threshold_pct = load_continuation_threshold_pct
        self.baseline_window = baseline_window
        self.voltage_variability_threshold = voltage_variability_threshold

    def detect_grid_to_dg(
        self,
        grid_import: pd.Series,
        site_load: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Detect intervals where grid→DG switchover occurred.

        Primary signal: grid import drops to <5% of rolling baseline.
        Confirmation (if site_load available): site load remains >50% of baseline.

        Returns:
            Boolean mask where True = DG is running (grid import near zero)
        """
        # Rolling baseline of grid import
        baseline = grid_import.rolling(
            self.baseline_window, center=False, min_periods=10
        ).median()

        # Avoid false positives when baseline is near zero (e.g., nighttime for solar-only)
        baseline = baseline.clip(lower=grid_import.quantile(0.1))

        # Primary: grid import drops below threshold
        import_ratio = grid_import / baseline
        dg_on = import_ratio < (self.import_drop_threshold_pct / 100)

        # Confirmation: site load continues (if available)
        if site_load is not None:
            load_baseline = site_load.rolling(
                self.baseline_window, center=False, min_periods=10
            ).median()
            load_baseline = load_baseline.clip(lower=site_load.quantile(0.1))
            load_continues = (site_load / load_baseline) > (
                self.load_continuation_threshold_pct / 100
            )
            # Both conditions must hold
            dg_on = dg_on & load_continues

        return dg_on

    def detect_voltage_signature(
        self,
        voltage: pd.Series,
        window: str = "30min",
    ) -> pd.Series:
        """
        Detect DG voltage signature — more variable than grid voltage.

        DG voltage has higher coefficient of variation (CV) within short windows
        compared to grid supply. Typical grid CV < 2%, DG CV > 5%.

        Returns:
            Boolean mask where True = voltage pattern consistent with DG
        """
        rolling_std = voltage.rolling(window, center=True, min_periods=3).std()
        rolling_mean = voltage.rolling(window, center=True, min_periods=3).mean()

        cv = rolling_std / rolling_mean.replace(0, np.nan)

        return cv > self.voltage_variability_threshold

    def mark_dg_periods(
        self,
        grid_import: pd.Series,
        site_load: Optional[pd.Series] = None,
        voltage: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Produce complete DG period annotation with transition labels.

        Returns:
            DataFrame with columns:
            - is_dg: boolean, True during DG periods
            - dg_event_type: 'grid_to_dg', 'dg_running', 'dg_to_grid', or None
            - dg_period_id: integer grouping consecutive DG intervals
        """
        dg_on = self.detect_grid_to_dg(grid_import, site_load)

        # Optional voltage confirmation — soft signal, does not override primary
        dg_confidence = pd.Series("medium", index=grid_import.index)
        dg_confidence[~dg_on] = "none"

        if voltage is not None:
            dg_voltage = self.detect_voltage_signature(voltage)
            # Voltage confirms DG → high confidence
            dg_confidence[dg_on & dg_voltage] = "high"
            # Voltage doesn't confirm but import says DG → keep medium
            # (could be a short outage where DG voltage hasn't diverged yet)

        result = pd.DataFrame(index=grid_import.index)
        result["is_dg"] = dg_on
        result["dg_confidence"] = dg_confidence
        result["dg_event_type"] = None

        # Assign period IDs to consecutive DG blocks
        transitions = dg_on.astype(int).diff().fillna(0)
        result["dg_period_id"] = (transitions == 1).cumsum()
        result.loc[~dg_on, "dg_period_id"] = -1

        # Label transitions
        grid_to_dg = (transitions == 1)
        dg_to_grid = (transitions == -1)

        result.loc[grid_to_dg, "dg_event_type"] = DGEventType.GRID_TO_DG.value
        result.loc[dg_to_grid, "dg_event_type"] = DGEventType.DG_TO_GRID.value
        result.loc[dg_on & ~grid_to_dg, "dg_event_type"] = DGEventType.DG_RUNNING.value

        n_periods = result.loc[dg_on, "dg_period_id"].nunique()
        n_intervals = dg_on.sum()
        if n_periods > 0:
            logger.info(
                f"DG periods detected: {n_periods} events, "
                f"{n_intervals} intervals ({n_intervals/len(grid_import)*100:.2f}%)"
            )

        return result

    @staticmethod
    def exclude_dg_from_training(
        demand_df: pd.DataFrame,
        dg_mask: pd.Series,
    ) -> pd.DataFrame:
        """
        Remove DG periods from training data to prevent bias.

        DG periods show near-zero grid import, which would train the model
        to predict demand drops that are actually supply-side events.

        Returns:
            Filtered DataFrame with DG periods removed
        """
        n_before = len(demand_df)
        clean = demand_df[~dg_mask].copy()
        n_removed = n_before - len(clean)

        if n_removed > 0:
            logger.info(
                f"Excluded {n_removed} DG intervals from training data "
                f"({n_removed/n_before*100:.1f}%)"
            )

        return clean


# ─────────────────────────────────────────────────────────────────────────────
# M1-F6: APFC Switching Event Detection
# ─────────────────────────────────────────────────────────────────────────────

class APFCEventType(Enum):
    """Types of APFC switching events."""
    CAP_SWITCH_IN = "cap_switch_in"    # Capacitor bank engaged → PF improves, kVA drops
    CAP_SWITCH_OUT = "cap_switch_out"  # Capacitor bank disengaged → PF drops, kVA rises


class APFCSwitchingDetector:
    """
    Detects Automatic Power Factor Correction (APFC) switching events.

    APFC panels switch capacitor banks to maintain PF near 0.95-1.0. Each switch
    causes a step change in kVAR/kVA without changing kW. If included in DR
    baselines, these events can be mistaken for demand curtailment.

    PRD detection logic:
    1. kVA step change > 50 kVA within 1 interval
    2. kW change in same interval < 1σ of rolling baseline (kW is stable)
    3. PF jumps toward 0.95-1.0 (switch-in) or away from it (switch-out)
    4. All three conditions must coincide
    """

    def __init__(
        self,
        kva_step_threshold: float = 50.0,
        kw_sigma_threshold: float = 1.0,
        target_pf_range: Tuple[float, float] = (0.95, 1.0),
        baseline_window: str = "48h",
    ):
        self.kva_step_threshold = kva_step_threshold
        self.kw_sigma_threshold = kw_sigma_threshold
        self.target_pf_range = target_pf_range
        self.baseline_window = baseline_window

    def detect_kva_step(
        self,
        kva: pd.Series,
    ) -> pd.Series:
        """
        Detect intervals with sudden kVA step change.

        A step change is |kVA[t] - kVA[t-1]| > threshold.

        Returns:
            Boolean mask where True = sudden kVA step
        """
        kva_diff = kva.diff().abs()
        return kva_diff > self.kva_step_threshold

    def detect_stable_kw(
        self,
        kw: pd.Series,
        kva_step_mask: pd.Series,
    ) -> pd.Series:
        """
        Confirm kW is stable during kVA step events.

        kW change in the step interval must be < sigma_threshold × rolling σ.
        This distinguishes APFC events (kVA changes, kW doesn't) from real
        load changes (both kVA and kW change).

        Returns:
            Boolean mask where True = kW is stable (consistent with APFC)
        """
        kw_diff = kw.diff().abs()
        kw_sigma = kw.rolling(self.baseline_window, center=False, min_periods=10).std()
        kw_sigma = kw_sigma.replace(0, np.nan)

        kw_stable = kw_diff / kw_sigma < self.kw_sigma_threshold

        # Only evaluate at kVA step points
        return kw_stable & kva_step_mask

    def detect_pf_jump(
        self,
        pf: pd.Series,
        kva_step_mask: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Detect PF jumps toward or away from target range during kVA steps.

        Cap switch-in: PF moves toward 0.95-1.0 (kVAR decreases, PF improves)
        Cap switch-out: PF moves away from 0.95-1.0 (kVAR increases, PF drops)

        Returns:
            Tuple of (switch_in_mask, switch_out_mask)
        """
        pf_diff = pf.diff()
        pf_prev = pf.shift(1)

        target_mid = (self.target_pf_range[0] + self.target_pf_range[1]) / 2

        # Switch-in: PF moves closer to target (improves)
        was_below_target = pf_prev < self.target_pf_range[0]
        moved_toward = pf_diff > 0.02  # Minimum PF jump to detect

        switch_in = kva_step_mask & was_below_target & moved_toward

        # Switch-out: PF moves away from target (drops)
        was_in_target = (pf_prev >= self.target_pf_range[0]) & (
            pf_prev <= self.target_pf_range[1]
        )
        moved_away = pf_diff < -0.02

        switch_out = kva_step_mask & was_in_target & moved_away

        return switch_in, switch_out

    def classify_apfc_events(
        self,
        kva: pd.Series,
        kw: pd.Series,
        pf: pd.Series,
    ) -> pd.DataFrame:
        """
        Full APFC event classification pipeline.

        Requires all three conditions to coincide:
        1. kVA step change
        2. Stable kW
        3. PF jump

        Returns:
            DataFrame with columns:
            - is_apfc_event: boolean
            - apfc_event_type: 'cap_switch_in', 'cap_switch_out', or None
            - kva_step_magnitude: size of kVA change
        """
        # Step 1: kVA step detection
        kva_step = self.detect_kva_step(kva)

        # Step 2: kW stability confirmation
        kw_stable = self.detect_stable_kw(kw, kva_step)

        # Step 3: PF jump classification
        switch_in, switch_out = self.detect_pf_jump(pf, kw_stable)

        result = pd.DataFrame(index=kva.index)
        result["is_apfc_event"] = switch_in | switch_out
        result["apfc_event_type"] = None
        result.loc[switch_in, "apfc_event_type"] = APFCEventType.CAP_SWITCH_IN.value
        result.loc[switch_out, "apfc_event_type"] = APFCEventType.CAP_SWITCH_OUT.value
        result["kva_step_magnitude"] = kva.diff().abs()
        result.loc[~result["is_apfc_event"], "kva_step_magnitude"] = 0

        n_events = result["is_apfc_event"].sum()
        if n_events > 0:
            n_in = switch_in.sum()
            n_out = switch_out.sum()
            logger.info(
                f"APFC events detected: {n_events} total "
                f"({n_in} switch-in, {n_out} switch-out)"
            )

        return result

    @staticmethod
    def normalize_for_dr_baseline(
        kva: pd.Series,
        kw: pd.Series,
        apfc_events: pd.DataFrame,
    ) -> pd.Series:
        """
        Normalize kVA to remove APFC effects for DR baseline computation.

        At APFC switch-in events: the kVA drop is artificial (real power unchanged).
        We replace the post-switch kVA with a value computed from stable kW and
        the pre-switch PF, so the baseline reflects actual load not PF correction.

        Returns:
            kVA series with APFC effects removed
        """
        normalized = kva.copy()

        event_mask = apfc_events["is_apfc_event"]
        if not event_mask.any():
            return normalized

        # Vectorized: compute pre-event PF and apply correction at all event positions
        prev_kva = kva.shift(1)
        prev_kw = kw.shift(1)
        prev_pf = prev_kw / prev_kva

        # Mask: is APFC event AND has valid pre-event PF
        correctable = event_mask & (prev_kva > 0) & (prev_pf > 0) & prev_kva.notna()
        normalized[correctable] = kw[correctable] / prev_pf[correctable]

        return normalized


# ─────────────────────────────────────────────────────────────────────────────
# Imputation (enhanced from original)
# ─────────────────────────────────────────────────────────────────────────────

def impute_missing_and_anomalous(
    series: pd.Series,
    anomaly_mask: Optional[pd.Series] = None,
    method: str = "hybrid",
    short_gap_limit: int = 4,
) -> pd.Series:
    """
    Impute missing values and anomalous readings.

    Methods:
    - interpolate: Time-based interpolation (best for short gaps)
    - seasonal: Use same interval from previous week
    - hybrid: Interpolate short gaps (<short_gap_limit intervals), seasonal for long

    Args:
        series: Original time series
        anomaly_mask: Boolean mask of values to replace
        method: Imputation strategy
        short_gap_limit: Max gap length (in intervals) for interpolation
    """
    result = series.copy()

    if anomaly_mask is not None:
        result[anomaly_mask] = np.nan

    if method == "interpolate":
        result = result.interpolate(method="time", limit=6)
        result = result.ffill(limit=3).bfill(limit=3)

    elif method == "seasonal":
        # Vectorized: fill from same interval 7 days ago (672 periods at 15min)
        freq = pd.tseries.frequencies.to_offset(pd.infer_freq(result.index) or "15min")
        periods_per_week = int(pd.Timedelta(days=7) / pd.Timedelta(freq.nanos, unit="ns"))
        week_shift = result.shift(periods=periods_per_week)
        still_nan = result.isna()
        result[still_nan] = week_shift[still_nan]
        result = result.interpolate(method="time", limit=6)

    elif method == "hybrid":
        nan_mask = result.isna()
        gap_groups = nan_mask.ne(nan_mask.shift()).cumsum()
        gap_lengths = nan_mask.groupby(gap_groups).transform("sum")

        # Short gaps: interpolate
        short_gap = nan_mask & (gap_lengths < short_gap_limit)
        result_short = result.copy()
        result_short[~short_gap & nan_mask] = 0
        result_short = result_short.interpolate(method="time")
        result[short_gap] = result_short[short_gap]

        # Long gaps: vectorized seasonal (same interval, 7 days ago)
        long_gap = result.isna()  # Re-check after short gap fill
        if long_gap.any():
            freq = pd.tseries.frequencies.to_offset(
                pd.infer_freq(result.index) or "15min"
            )
            periods_per_week = int(pd.Timedelta(days=7) / pd.Timedelta(freq.nanos, unit="ns"))
            week_shift = result.shift(periods=periods_per_week)
            result[long_gap] = week_shift[long_gap]

        result = result.interpolate(method="time", limit=6)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Integrated Quality Pipeline
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QualityReport:
    """Summary of data quality for a single consumer."""
    consumer_id: str
    total_intervals: int
    missing_pct: float
    frozen_pct: float
    zscore_outlier_pct: float
    iqr_outlier_pct: float
    contextual_outlier_pct: float
    rolling_outlier_pct: float
    ct_artefact_pct: float
    pf_artefact_pct: float
    dg_period_pct: float
    apfc_event_count: int
    total_anomaly_pct: float
    imputed_pct: float
    mean_quality_score: float
    channels_complete_pct: float


def run_quality_pipeline(
    df: pd.DataFrame,
    demand_col: str = "demand_kwh",
    timestamp_col: str = "timestamp",
    consumer_col: str = "consumer_id",
    kva_col: Optional[str] = None,
    kw_col: Optional[str] = None,
    pf_col: Optional[str] = None,
    voltage_col: Optional[str] = None,
    frequency_col: Optional[str] = None,
    grid_import_col: Optional[str] = None,
    site_load_col: Optional[str] = None,
    freq: str = "15min",
) -> Tuple[pd.DataFrame, List[QualityReport]]:
    """
    Full M1 quality pipeline: detect issues, clean, impute, score.

    Orchestrates all M1 detectors:
    1. Basic anomaly detection (frozen, z-score, IQR, contextual, rolling)
    2. Demand noise filter (CT artefacts, PF artefacts) — if kVA/kW available
    3. DG transition detection — if grid_import available
    4. APFC event detection — if kVA/kW/PF available
    5. Imputation of flagged intervals
    6. Per-interval quality scoring

    Args:
        df: Raw meter data
        demand_col: Primary demand column (kWh or kVA)
        timestamp_col: Timestamp column name
        consumer_col: Consumer identifier column
        kva_col, kw_col, pf_col, voltage_col, frequency_col: Optional channel columns
        grid_import_col, site_load_col: For DG detection
        freq: Expected data frequency

    Returns:
        Tuple of (cleaned DataFrame, list of QualityReport per consumer)
    """
    logger.info("═══ M1 Data Quality Pipeline ═══")

    cleaned_dfs = []
    quality_reports = []

    for cid, group in df.groupby(consumer_col):
        logger.info(f"Processing consumer: {cid}")
        group = group.sort_values(timestamp_col).copy()

        # Ensure datetime index
        if timestamp_col in group.columns:
            group = group.set_index(timestamp_col)
        if not isinstance(group.index, pd.DatetimeIndex):
            group.index = pd.to_datetime(group.index)

        series = group[demand_col]

        # ── 1. Basic anomaly detection ──────────────────────────────────
        frozen = detect_frozen_readings(series)
        zscore_outliers = detect_outliers_zscore(series)
        iqr_outliers = detect_outliers_iqr(series)
        contextual_outliers = detect_outliers_contextual(series)
        rolling_outliers = detect_outliers_rolling(series)

        basic_anomaly = frozen | zscore_outliers | iqr_outliers | contextual_outliers | rolling_outliers

        # ── 2. Demand noise filter ──────────────────────────────────────
        ct_artefacts = pd.Series(False, index=group.index)
        pf_artefacts = pd.Series(False, index=group.index)

        has_kva_kw = kva_col and kw_col and kva_col in group.columns and kw_col in group.columns
        if has_kva_kw:
            noise_filter = DemandNoiseFilter()
            freq_series = group[frequency_col] if (frequency_col and frequency_col in group.columns) else None
            pf_series = group[pf_col] if (pf_col and pf_col in group.columns) else None

            ct_artefacts = noise_filter.detect_ct_artefacts(
                group[kva_col], group[kw_col], freq_series
            )
            pf_artefacts = noise_filter.detect_pf_artefacts(
                group[kva_col], group[kw_col], pf_series
            )

        # ── 3. DG transition detection ──────────────────────────────────
        dg_mask = pd.Series(False, index=group.index)

        has_grid_import = grid_import_col and grid_import_col in group.columns
        if has_grid_import:
            dg_detector = DGTransitionDetector()
            site_load = group[site_load_col] if (site_load_col and site_load_col in group.columns) else None
            voltage = group[voltage_col] if (voltage_col and voltage_col in group.columns) else None

            dg_result = dg_detector.mark_dg_periods(
                group[grid_import_col], site_load, voltage
            )
            dg_mask = dg_result["is_dg"]
            group["is_dg"] = dg_mask
            group["dg_event_type"] = dg_result["dg_event_type"]
            group["dg_period_id"] = dg_result["dg_period_id"]

        # ── 4. APFC detection ───────────────────────────────────────────
        apfc_count = 0

        has_pf = pf_col and pf_col in group.columns
        if has_kva_kw and has_pf:
            apfc_detector = APFCSwitchingDetector()
            apfc_result = apfc_detector.classify_apfc_events(
                group[kva_col], group[kw_col], group[pf_col]
            )
            group["is_apfc_event"] = apfc_result["is_apfc_event"]
            group["apfc_event_type"] = apfc_result["apfc_event_type"]
            apfc_count = apfc_result["is_apfc_event"].sum()

        # ── 5. Combined anomaly mask & imputation ───────────────────────
        combined_anomaly = basic_anomaly | ct_artefacts | pf_artefacts

        # DG periods are NOT imputed — they are excluded from training data entirely
        # We only impute point anomalies, not regime changes

        clean_series = impute_missing_and_anomalous(
            series, anomaly_mask=combined_anomaly, method="hybrid"
        )

        # ── 6. Build output ─────────────────────────────────────────────
        clean_group = group.copy()
        clean_group[demand_col] = clean_series
        clean_group[f"{demand_col}_original"] = series
        clean_group["is_anomaly"] = combined_anomaly
        clean_group["is_imputed"] = combined_anomaly | series.isna()
        clean_group["is_ct_artefact"] = ct_artefacts
        clean_group["is_pf_artefact"] = pf_artefacts

        # Quality score (needs validate + sync first for full scoring)
        # For basic pipeline without full AMI channels, score on completeness + anomaly rate
        quality_score = 1.0 - combined_anomaly.astype(float) * 0.5 - series.isna().astype(float) * 0.5
        quality_score = quality_score.clip(0, 1)
        clean_group["quality_score"] = quality_score

        cleaned_dfs.append(clean_group.reset_index())

        # ── 7. Quality report ───────────────────────────────────────────
        report = QualityReport(
            consumer_id=str(cid),
            total_intervals=len(series),
            missing_pct=float(series.isna().mean() * 100),
            frozen_pct=float(frozen.mean() * 100),
            zscore_outlier_pct=float(zscore_outliers.mean() * 100),
            iqr_outlier_pct=float(iqr_outliers.mean() * 100),
            contextual_outlier_pct=float(contextual_outliers.mean() * 100),
            rolling_outlier_pct=float(rolling_outliers.mean() * 100),
            ct_artefact_pct=float(ct_artefacts.mean() * 100),
            pf_artefact_pct=float(pf_artefacts.mean() * 100),
            dg_period_pct=float(dg_mask.mean() * 100),
            apfc_event_count=apfc_count,
            total_anomaly_pct=float(combined_anomaly.mean() * 100),
            imputed_pct=float((combined_anomaly | series.isna()).mean() * 100),
            mean_quality_score=float(quality_score.mean()),
            channels_complete_pct=100.0,  # Updated when full AMI channels are available
        )
        quality_reports.append(report)

        logger.info(
            f"  {cid}: {report.total_anomaly_pct:.1f}% anomalous, "
            f"{report.dg_period_pct:.1f}% DG, "
            f"{report.apfc_event_count} APFC events, "
            f"quality={report.mean_quality_score:.2f}"
        )

    cleaned = pd.concat(cleaned_dfs, ignore_index=True)

    logger.info(f"═══ Pipeline complete: {len(quality_reports)} consumers processed ═══")

    return cleaned, quality_reports
