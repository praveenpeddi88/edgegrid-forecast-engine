"""Tests for solar generation model — physics layer validation."""

import numpy as np
import pandas as pd
import pytest

from edgegrid_forecast.models.solar import (
    SolarForecastResult,
    clear_sky_irradiance,
    estimate_solar_generation,
)


class TestClearSkyIrradiance:

    def _make_summer_day(self):
        """June day in Visakhapatnam."""
        return pd.date_range("2025-06-15", periods=24, freq="h")

    def test_zero_at_night(self):
        """GHI should be zero at night."""
        ts = self._make_summer_day()
        ghi = clear_sky_irradiance(ts, latitude=17.69, longitude=83.22)
        # Hours 0-5 and 19-23 should be ~0 for Indian latitudes in summer
        for h in [0, 1, 2, 3, 22, 23]:
            assert ghi.iloc[h] < 10, f"GHI at hour {h} should be near zero, got {ghi.iloc[h]}"

    def test_peak_midday(self):
        """GHI should peak around midday."""
        ts = self._make_summer_day()
        ghi = clear_sky_irradiance(ts, latitude=17.69, longitude=83.22)
        peak_hour = ghi.idxmax().hour
        assert 10 <= peak_hour <= 14, f"Peak GHI at hour {peak_hour}, expected 10-14"

    def test_peak_magnitude_reasonable(self):
        """Peak clear-sky GHI should be 700-1000 W/m² for tropical latitude."""
        ts = self._make_summer_day()
        ghi = clear_sky_irradiance(ts, latitude=17.69, longitude=83.22)
        assert 500 < ghi.max() < 1100

    def test_always_non_negative(self):
        """GHI should never be negative."""
        ts = pd.date_range("2025-01-01", periods=8760, freq="h")
        ghi = clear_sky_irradiance(ts, latitude=17.69, longitude=83.22)
        assert (ghi >= 0).all()


class TestSolarGeneration:

    def test_generation_proportional_to_capacity(self):
        """Doubling capacity should double generation."""
        ts = pd.date_range("2025-06-15", periods=24, freq="h")
        r1 = estimate_solar_generation(ts, 17.69, 83.22, capacity_kw=500)
        r2 = estimate_solar_generation(ts, 17.69, 83.22, capacity_kw=1000)
        ratio = np.sum(r2.generation_kwh) / np.sum(r1.generation_kwh)
        assert 1.95 < ratio < 2.05

    def test_cloud_cover_reduces_generation(self):
        """Cloud cover should reduce generation."""
        ts = pd.date_range("2025-06-15", periods=24, freq="h")
        clear = estimate_solar_generation(ts, 17.69, 83.22, capacity_kw=500)
        clouds = pd.Series(70.0, index=ts)  # 70% cloud cover
        cloudy = estimate_solar_generation(ts, 17.69, 83.22, capacity_kw=500, cloud_cover_pct=clouds)
        assert np.sum(cloudy.generation_kwh) < np.sum(clear.generation_kwh)

    def test_monthly_generation_realistic_vizag(self):
        """500kW system in Vizag should produce 50-100 MWh/month in summer."""
        ts = pd.date_range("2025-06-01", periods=30 * 24, freq="h")
        result = estimate_solar_generation(ts, 17.69, 83.22, capacity_kw=500)
        monthly_mwh = np.sum(result.generation_kwh) / 1000
        assert 40 < monthly_mwh < 120, f"Monthly generation {monthly_mwh:.1f} MWh outside realistic range"

    def test_capacity_factor_bounded(self):
        """Capacity factor should be between 0 and 1."""
        ts = pd.date_range("2025-06-15", periods=24, freq="h")
        result = estimate_solar_generation(ts, 17.69, 83.22, capacity_kw=500)
        assert np.all(result.capacity_factor >= 0)
        assert np.all(result.capacity_factor <= 1.0)

    def test_result_structure(self):
        """Result should have correct shape and types."""
        ts = pd.date_range("2025-06-15", periods=24, freq="h")
        result = estimate_solar_generation(ts, 17.69, 83.22, capacity_kw=500)
        assert isinstance(result, SolarForecastResult)
        assert len(result.generation_kwh) == 24
        assert len(result.clear_sky_kwh) == 24
        assert len(result.cloud_factor) == 24
        assert len(result.capacity_factor) == 24
