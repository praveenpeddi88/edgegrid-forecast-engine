"""
M1-F3: Voltage Compensation / SOC Correction.

Corrects BMS-reported SOC for Indian grid voltage deviation (±10-15%).
Site-specific polynomial regression: voltage_deviation → SOC_error.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from ._constants import (
    DEFAULT_FREQ_SECONDS,
    FULL_CHARGE_SOC_THRESHOLD,
    FULL_DISCHARGE_SOC_THRESHOLD,
    NOMINAL_VOLTAGE_INDIA,
    REST_CURRENT_FRACTION,
    REST_CURRENT_PERCENTILE,
)


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
        nominal_voltage: float = NOMINAL_VOLTAGE_INDIA,
        polynomial_degree: int = 2,
        min_calibration_points: int = 100,
    ):
        """
        Args:
            nominal_voltage: Nominal grid voltage in volts (415V for Indian 3-phase LT).
            polynomial_degree: Degree of polynomial for voltage→SOC_error regression.
            min_calibration_points: Minimum data points for full polynomial fit; below
                this, falls back to linear regression with a warning.
        """
        if polynomial_degree < 1:
            raise ValueError(f"polynomial_degree must be >= 1, got {polynomial_degree}")
        if min_calibration_points < 10:
            raise ValueError(f"min_calibration_points must be >= 10, got {min_calibration_points}")

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
        - Full charge: SOC > 98% AND current ≈ 0
        - Full discharge: SOC < 5% AND current ≈ 0
        - Rest: current ≈ 0 for > rest_duration_min

        Returns:
            DataFrame with columns: start, end, state_type, duration_min, mean_soc
        """
        # Define "approximately zero" current
        current_threshold = current_series.abs().quantile(REST_CURRENT_PERCENTILE) * REST_CURRENT_FRACTION
        is_rest = current_series.abs() < current_threshold

        # Infer frequency from index
        freq_seconds = DEFAULT_FREQ_SECONDS
        if isinstance(soc_series.index, pd.DatetimeIndex) and len(soc_series) > 1:
            inferred = pd.infer_freq(soc_series.index)
            if inferred:
                freq_seconds = pd.Timedelta(pd.tseries.frequencies.to_offset(inferred).nanos, unit="ns").total_seconds()

        # Vectorized rest period detection
        rest_starts = is_rest & ~is_rest.shift(1, fill_value=False)
        rest_id = rest_starts.cumsum()

        # Filter to only rest intervals, then aggregate per period
        rest_data = pd.DataFrame({
            "is_rest": is_rest,
            "rest_id": rest_id,
            "soc": soc_series,
        })
        rest_only = rest_data[rest_data["is_rest"]]

        if rest_only.empty:
            return pd.DataFrame(columns=["start", "end", "state_type", "duration_min", "mean_soc"])

        agg = rest_only.groupby("rest_id").agg(
            start=("soc", lambda x: x.index[0]),
            end=("soc", lambda x: x.index[-1]),
            count=("soc", "count"),
            mean_soc=("soc", "mean"),
        )

        agg["duration_min"] = agg["count"] * freq_seconds / 60

        # Filter by minimum duration
        agg = agg[agg["duration_min"] >= rest_duration_min].copy()

        if agg.empty:
            return pd.DataFrame(columns=["start", "end", "state_type", "duration_min", "mean_soc"])

        # Classify state type vectorized
        agg["state_type"] = "rest"
        agg.loc[agg["mean_soc"] > FULL_CHARGE_SOC_THRESHOLD, "state_type"] = "full_charge"
        agg.loc[agg["mean_soc"] < FULL_DISCHARGE_SOC_THRESHOLD, "state_type"] = "full_discharge"

        return agg[["start", "end", "state_type", "duration_min", "mean_soc"]].reset_index(drop=True)

    def calibrate(
        self,
        voltage: pd.Series,
        bms_soc: pd.Series,
        actual_soc: pd.Series,
    ) -> Dict:
        """
        Build site-specific voltage→SOC_error correction model.

        Uses a local effective_degree (never mutates self.polynomial_degree).
        """
        soc_error = bms_soc - actual_soc

        valid = voltage.notna() & soc_error.notna()
        v = voltage[valid].values.reshape(-1, 1)
        e = soc_error[valid].values

        # Use local variable — never downgrade the configured degree for future calls
        effective_degree = self.polynomial_degree
        if len(v) < self.min_calibration_points:
            logger.warning(
                f"Only {len(v)} calibration points (need {self.min_calibration_points}). "
                "Using linear model — collect more data for polynomial."
            )
            effective_degree = 1

        v_deviation = v - self.nominal_voltage

        self.poly_features = PolynomialFeatures(degree=effective_degree, include_bias=False)
        v_poly = self.poly_features.fit_transform(v_deviation)

        self.model = LinearRegression()
        self.model.fit(v_poly, e)

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

        Returns corrected SOC clipped to [0, 100].
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

        return corrected.clip(0, 100)

    def check_calibration_drift(
        self,
        recent_residuals: pd.Series,
        threshold: float = 2.0,
        consecutive_days: int = 7,
    ) -> bool:
        """
        Check if correction model has drifted and needs re-calibration.

        Returns True if mean |residual| > threshold for consecutive_days in a row.
        """
        if len(recent_residuals) == 0:
            return False

        daily = recent_residuals.abs().resample("D").mean()
        exceeds = daily > threshold

        if len(exceeds) < consecutive_days:
            return bool(exceeds.all())

        rolling = exceeds.rolling(consecutive_days).sum()
        return bool((rolling >= consecutive_days).any())
