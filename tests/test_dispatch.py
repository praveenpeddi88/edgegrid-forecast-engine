"""Tests for dispatch optimizer — validates economic logic."""

import numpy as np
import pytest

from edgegrid_forecast.dispatch.optimizer import BESSConfig, DispatchOptimizer
from edgegrid_forecast.utils.constants import ChargingStrategy


class TestDispatchOptimizer:

    def _make_simple_scenario(self):
        """24h scenario: high demand during day, solar during midday, price peak at evening."""
        demand = np.array([
            100, 80, 70, 60, 60, 70,     # 0-5: Night (low)
            150, 300, 500, 600, 650, 700, # 6-11: Morning ramp
            700, 650, 600, 500, 400, 350, # 12-17: Afternoon
            500, 700, 800, 600, 300, 150, # 18-23: Evening peak
        ], dtype=float)

        solar = np.array([
            0, 0, 0, 0, 0, 0,            # 0-5: Night
            50, 200, 400, 600, 700, 750,  # 6-11: Morning solar
            750, 700, 600, 400, 200, 50,  # 12-17: Afternoon solar
            0, 0, 0, 0, 0, 0,            # 18-23: Night
        ], dtype=float)

        iex_prices = np.array([
            3.0, 2.8, 2.5, 2.3, 2.3, 2.7,
            3.5, 5.0, 6.5, 7.0, 7.5, 7.8,
            7.5, 7.0, 6.5, 6.0, 5.5, 5.8,
            7.5, 9.0, 8.5, 7.0, 5.5, 4.0,
        ], dtype=float)

        return demand, solar, iex_prices

    def test_no_bess_grid_only(self):
        """Without BESS, all non-solar demand comes from grid."""
        demand, solar, iex_prices = self._make_simple_scenario()

        optimizer = DispatchOptimizer(bess_config=None, fls_tariff_inr_kwh=6.50)
        schedule = optimizer.optimize_24h(demand, solar, iex_prices)

        # All demand should be met
        assert schedule.reliability_score > 0.99

        # No BESS activity
        assert np.sum(schedule.bess_charge_kwh) == 0
        assert np.sum(schedule.bess_discharge_kwh) == 0

    def test_bess_reduces_cost(self):
        """BESS should reduce total cost compared to no BESS."""
        demand, solar, iex_prices = self._make_simple_scenario()

        # Without BESS
        opt_no_bess = DispatchOptimizer(bess_config=None, fls_tariff_inr_kwh=6.50)
        sched_no_bess = opt_no_bess.optimize_24h(demand, solar, iex_prices)

        # With BESS
        bess = BESSConfig(capacity_kwh=2000, max_power_kw=500)
        opt_bess = DispatchOptimizer(
            bess_config=bess,
            fls_tariff_inr_kwh=6.50,
            strategy=ChargingStrategy.CHEAP_GRID,
        )
        sched_bess = opt_bess.optimize_24h(demand, solar, iex_prices)

        # BESS should save money
        assert sched_bess.total_cost_inr <= sched_no_bess.total_cost_inr

    def test_solar_used_before_grid(self):
        """Solar should be used directly before buying from grid."""
        demand, solar, iex_prices = self._make_simple_scenario()

        optimizer = DispatchOptimizer(bess_config=None, fls_tariff_inr_kwh=6.50)
        schedule = optimizer.optimize_24h(demand, solar, iex_prices)

        # During solar hours, direct solar use should equal min(solar, demand)
        for h in range(6, 18):
            expected = min(solar[h], demand[h])
            assert schedule.solar_direct_use_kwh[h] == pytest.approx(expected, abs=1)

    def test_bess_soc_within_bounds(self):
        """BESS SoC should never exceed physical limits."""
        demand, solar, iex_prices = self._make_simple_scenario()

        bess = BESSConfig(capacity_kwh=2000, max_power_kw=500)
        optimizer = DispatchOptimizer(
            bess_config=bess,
            fls_tariff_inr_kwh=6.50,
            strategy=ChargingStrategy.SOLAR_SURPLUS,
        )
        schedule = optimizer.optimize_24h(demand, solar, iex_prices)

        for soc in schedule.bess_soc_kwh:
            assert soc >= 0, f"SoC went negative: {soc}"
            # Allow small float tolerance above max
            assert soc <= bess.capacity_kwh * 1.01, f"SoC exceeded capacity: {soc}"

    def test_reliability_always_high(self):
        """Supply should always meet demand."""
        demand, solar, iex_prices = self._make_simple_scenario()

        bess = BESSConfig(capacity_kwh=2000, max_power_kw=500)
        optimizer = DispatchOptimizer(
            bess_config=bess,
            fls_tariff_inr_kwh=6.50,
            strategy=ChargingStrategy.CHEAP_GRID,
        )
        schedule = optimizer.optimize_24h(demand, solar, iex_prices)

        assert schedule.reliability_score > 0.95
