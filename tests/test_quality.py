"""Tests for data quality pipeline — frozen detection, outliers, imputation."""

import numpy as np
import pandas as pd
import pytest

from edgegrid_forecast.data.quality import (
    detect_frozen_readings,
    detect_outliers_zscore,
    detect_outliers_iqr,
    impute_missing_and_anomalous,
)


class TestFrozenReadings:

    def test_detects_frozen_run(self):
        """Consecutive identical values should be flagged."""
        data = pd.Series([100, 200, 200, 200, 200, 300])
        mask = detect_frozen_readings(data, min_run_length=3)
        # Indices 1-4 have value 200 repeated 4 times (3 consecutive diffs=0)
        assert mask.sum() >= 3

    def test_no_frozen_in_varying_data(self):
        """Varying data should have no frozen readings."""
        data = pd.Series([100, 200, 300, 400, 500, 600])
        mask = detect_frozen_readings(data, min_run_length=3)
        assert mask.sum() == 0

    def test_handles_nan(self):
        """NaN values should not be flagged as frozen."""
        data = pd.Series([100, np.nan, np.nan, np.nan, 200])
        mask = detect_frozen_readings(data, min_run_length=3)
        assert mask.sum() == 0


class TestOutlierDetection:

    def _make_series_with_outlier(self):
        np.random.seed(42)
        data = pd.Series(np.random.normal(100, 10, 200))
        data.iloc[50] = 500  # Clear outlier
        return data

    def test_zscore_detects_outlier(self):
        data = self._make_series_with_outlier()
        mask = detect_outliers_zscore(data, threshold=3.0)
        assert mask.iloc[50] == True

    def test_iqr_detects_outlier(self):
        data = self._make_series_with_outlier()
        mask = detect_outliers_iqr(data, factor=1.5)
        assert mask.iloc[50] == True

    def test_zscore_low_false_positives(self):
        """Normal data should have very few false positives."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(100, 10, 1000))
        mask = detect_outliers_zscore(data, threshold=3.0)
        # At z=3, expect ~0.3% false positives
        assert mask.sum() < 10


class TestImputation:

    def test_interpolate_fills_gaps(self):
        """Interpolation should fill small gaps."""
        idx = pd.date_range("2025-01-01", periods=24, freq="h")
        data = pd.Series(np.linspace(100, 200, 24), index=idx)
        anomaly_mask = pd.Series(False, index=idx)
        # Create a gap
        data.iloc[10:12] = np.nan
        anomaly_mask.iloc[10:12] = True

        result = impute_missing_and_anomalous(data, anomaly_mask, method="interpolate")
        assert result.isna().sum() == 0
        # Interpolated values should be between neighbors
        assert result.iloc[10] > result.iloc[9]
        assert result.iloc[11] < result.iloc[12]
