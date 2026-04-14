"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from edgegrid_forecast.data.features import (
    add_temporal_features,
    add_lag_features,
    add_rolling_features,
    add_price_features,
    build_forecast_features,
)


class TestTemporalFeatures:

    def _make_hourly_df(self, hours=168):
        """One week of hourly data."""
        timestamps = pd.date_range("2025-06-01", periods=hours, freq="h")
        return pd.DataFrame({
            "timestamp": timestamps,
            "consumption_kwh": np.random.uniform(100, 500, hours),
        })

    def test_adds_expected_columns(self):
        df = self._make_hourly_df()
        result = add_temporal_features(df)
        expected = ["hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos"]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_cyclical_encoding_bounded(self):
        """Sin/cos features should be in [-1, 1]."""
        df = self._make_hourly_df(8760)
        result = add_temporal_features(df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
            if col in result.columns:
                assert result[col].min() >= -1.01
                assert result[col].max() <= 1.01

    def test_is_weekend_correct(self):
        df = self._make_hourly_df()
        result = add_temporal_features(df)
        for _, row in result.iterrows():
            dow = row["timestamp"].dayofweek
            expected_weekend = 1 if dow >= 5 else 0
            assert row["is_weekend"] == expected_weekend


class TestLagFeatures:

    def test_lag_creates_columns(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=200, freq="h"),
            "consumer_id": "TEST",
            "consumption_kwh": np.random.uniform(100, 500, 200),
        })
        result = add_lag_features(df, target_col="consumption_kwh", lag_hours=[1, 24])
        assert "consumption_kwh_lag_1h" in result.columns
        assert "consumption_kwh_lag_24h" in result.columns

    def test_lag_values_correct(self):
        data = list(range(50))
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=50, freq="h"),
            "consumer_id": "TEST",
            "consumption_kwh": data,
        })
        result = add_lag_features(df, target_col="consumption_kwh", lag_hours=[1])
        # lag_1h at index 5 should equal value at index 4
        assert result["consumption_kwh_lag_1h"].iloc[5] == 4


class TestRollingFeatures:

    def test_rolling_creates_columns(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=200, freq="h"),
            "consumer_id": "TEST",
            "consumption_kwh": np.random.uniform(100, 500, 200),
        })
        result = add_rolling_features(df, target_col="consumption_kwh", windows=[3, 24])
        assert "consumption_kwh_rmean_3h" in result.columns
        assert "consumption_kwh_rmean_24h" in result.columns


class TestBuildFeatures:

    def test_full_pipeline_runs(self):
        """Full feature pipeline should not crash."""
        n = 500
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
            "consumer_id": "TEST",
            "consumption_kwh": np.random.uniform(100, 500, n),
        })
        result = build_forecast_features(df, target_col="consumption_kwh")
        assert len(result) > 0
        # Should have many more columns than input
        assert len(result.columns) > 10
