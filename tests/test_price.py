"""Tests for IEX price forecaster and landed cost logic."""

import numpy as np
import pandas as pd
import pytest

from edgegrid_forecast.models.price import IEXPriceForecaster
from edgegrid_forecast.utils.constants import landed_cost_from_iex


class TestIEXPriceForecaster:

    def test_baseline_price_in_range(self):
        """IEX DAM prices should be in ₹1-15/kWh range."""
        forecaster = IEXPriceForecaster()
        for month in range(1, 13):
            for hour in range(24):
                price = forecaster.get_baseline_price(month, hour)
                assert 0.5 < price < 20, f"Price ₹{price} at month={month} hour={hour} out of range"

    def test_evening_peak_higher_than_night(self):
        """Evening peak (18-21h) should be more expensive than early morning (2-5h)."""
        forecaster = IEXPriceForecaster()
        for month in [4, 6, 10]:  # Sample months
            night_avg = np.mean([forecaster.get_baseline_price(month, h) for h in [2, 3, 4]])
            peak_avg = np.mean([forecaster.get_baseline_price(month, h) for h in [18, 19, 20]])
            assert peak_avg > night_avg, f"Month {month}: peak ₹{peak_avg:.2f} should > night ₹{night_avg:.2f}"

    def test_cheapest_hours_returns_correct_count(self):
        """find_cheapest_hours should return requested number of hours."""
        forecaster = IEXPriceForecaster()
        hours = forecaster.find_cheapest_hours(month=6, n_hours=4)
        assert len(hours) == 4

    def test_expensive_hours_returns_correct_count(self):
        forecaster = IEXPriceForecaster()
        hours = forecaster.find_expensive_hours(month=6, n_hours=4)
        assert len(hours) == 4

    def test_forecast_prices_shape(self):
        """forecast_prices should return DataFrame with correct columns."""
        forecaster = IEXPriceForecaster()
        ts = pd.date_range("2025-06-15", periods=24, freq="h")
        df = forecaster.forecast_prices(ts)
        assert len(df) == 24
        assert "iex_price_inr_kwh" in df.columns
        assert "landed_cost_inr_kwh" in df.columns

    def test_spread_positive_for_arbitrage(self):
        """Spread between cheap and expensive hours should be positive."""
        forecaster = IEXPriceForecaster()
        spread = forecaster.compute_spread(month=6)
        assert spread["avg_charge_iex"] < spread["avg_discharge_iex"]
        assert spread["iex_spread"] > 0


class TestLandedCost:

    def test_landed_cost_higher_than_iex(self):
        """Landed cost should always exceed IEX price due to losses + charges."""
        for iex_price in [2.0, 4.0, 6.0, 8.0, 10.0]:
            landed = landed_cost_from_iex(iex_price)
            assert landed > iex_price, f"Landed ₹{landed:.2f} should > IEX ₹{iex_price:.2f}"

    def test_landed_cost_formula_known_value(self):
        """Verify against manually computed value."""
        # IEX = 5.0: landed = 5.0 / ((1-0.039)*(1-0.0275)*(1-0.0272)) + 0.41 + 0.31 + 0.47
        # denominator ≈ 0.961 * 0.9725 * 0.9728 ≈ 0.9093
        # 5.0 / 0.9093 ≈ 5.4985 + 1.19 ≈ 6.6885
        landed = landed_cost_from_iex(5.0)
        assert 6.4 < landed < 7.0, f"Landed cost ₹{landed:.2f} not in expected range"
