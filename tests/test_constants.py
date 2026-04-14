"""Tests for domain constants — these encode hard-won knowledge, protect them."""

import pytest

from edgegrid_forecast.utils.constants import (
    BESSConstraints,
    ChargingStrategy,
    fy_month_index,
    get_iex_price,
    get_landed_price,
    get_tod_multiplier,
    landed_cost_from_iex,
    mwh_inr_kwh_to_crores,
    mwh_inr_kwh_to_lakhs,
)


class TestLandedCost:
    """The landed cost formula is the most critical calculation. Guard it."""

    def test_basic_landed_cost(self):
        # IEX price of ₹5/kWh should result in ~₹6.7/kWh landed
        landed = landed_cost_from_iex(5.0)
        assert 6.0 < landed < 8.0, f"Landed cost {landed} out of expected range"

    def test_zero_iex_price(self):
        # Even at ₹0 IEX, there are fixed network charges
        landed = landed_cost_from_iex(0.0)
        assert landed == pytest.approx(0.41 + 0.31 + 0.47, abs=0.01)

    def test_network_charges_added(self):
        # Network charges should always be: SLDC + cross-subsidy + additional
        landed_zero = landed_cost_from_iex(0.0)
        landed_five = landed_cost_from_iex(5.0)
        assert landed_five > landed_zero + 4.5  # Loss-adjusted price > raw price


class TestFYMonthIndex:
    """Month indexing bugs are the #2 source of errors."""

    def test_april_is_zero(self):
        assert fy_month_index(4) == 0

    def test_march_is_eleven(self):
        assert fy_month_index(3) == 11

    def test_january(self):
        assert fy_month_index(1) == 9

    def test_all_months_unique(self):
        indices = [fy_month_index(m) for m in range(1, 13)]
        assert len(set(indices)) == 12


class TestUnitConversions:
    """Unit conversion bugs are the #1 source of errors."""

    def test_mwh_to_lakhs(self):
        assert mwh_inr_kwh_to_lakhs(100) == 1.0

    def test_mwh_to_crores(self):
        assert mwh_inr_kwh_to_crores(10000) == 1.0


class TestTODMultiplier:
    def test_peak_hours(self):
        # 10am-2pm and 6pm-10pm should be peak
        assert get_tod_multiplier(10) == 1.20
        assert get_tod_multiplier(19) == 1.20

    def test_off_peak_hours(self):
        assert get_tod_multiplier(2) == 0.90

    def test_normal_hours(self):
        assert get_tod_multiplier(8) == 1.00


class TestBESSConstraints:
    def test_defaults(self):
        bess = BESSConstraints()
        assert bess.min_soc == 0.10
        assert bess.max_soc == 0.90
        assert bess.round_trip_efficiency == 0.88

    def test_usable_capacity(self):
        bess = BESSConstraints()
        # Usable = 90% - 10% = 80% of nominal
        assert (bess.max_soc - bess.min_soc) == pytest.approx(0.80)
