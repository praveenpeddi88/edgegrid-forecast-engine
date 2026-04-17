"""
Tests for the commercial layer (IRR + FLS quote + brief).

Covers plan §A6 test #5: "IRR monotonicity — doubling BESS capacity at
fixed prices → IRR moves in the expected direction."
"""

from datetime import time

import pytest

from edgegrid_forecast.commercial.irr import (
    compute_project_irr,
    irr_heatmap,
    sensitivity_grid,
)
from edgegrid_forecast.commercial.quote import FLSQuoteGenerator


# ───────────────────── IRR ───────────────────────────────────────────────────


def test_compute_irr_positive_for_profitable_project():
    r = compute_project_irr(
        capex_inr=2_50_00_000,           # ₹2.5 Cr for 1 MWh
        annual_net_inr=60_00_000,        # ₹60 L/yr
        years=10,
    )
    # 10-yr project with ~25% undiscounted RoI/yr lands ~13-16% IRR after
    # the default 2.5%/yr degradation + 2%/yr opex escalation.
    assert r.irr_pct > 12.0
    assert r.payback_years < 6
    assert r.npv_inr > 0


def test_irr_heatmap_shape_matches_inputs():
    grid = irr_heatmap(
        capacities_mwh=[0.5, 1.0, 2.0],
        durations_h=[2, 4, 8],
        daily_net_inr_at_reference=15_000,
    )
    assert grid["capacities_mwh"] == [0.5, 1.0, 2.0]
    assert grid["durations_h"] == [2, 4, 8]
    assert len(grid["irr_pct"]) == 3
    assert all(len(row) == 3 for row in grid["irr_pct"])


def test_irr_monotone_in_annual_net():
    """Plan A6 test #5: more annual net → strictly higher IRR (at fixed CAPEX)."""
    low = compute_project_irr(capex_inr=1e7, annual_net_inr=1e6)
    high = compute_project_irr(capex_inr=1e7, annual_net_inr=2e6)
    assert high.irr_pct > low.irr_pct
    assert high.payback_years < low.payback_years


def test_sensitivity_grid_covers_spec_ranges():
    s = sensitivity_grid(base_capex_inr=1e7, base_annual_net_inr=1e6)
    assert {d["delta_pct"] for d in s["iex"]} == {-20.0, -10.0, 0.0, 10.0, 20.0}
    assert {d["delta_pct"] for d in s["capex"]} == {-15.0, -7.5, 0.0, 7.5, 15.0}


# ───────────────────── FLS quote ─────────────────────────────────────────────


@pytest.fixture
def qg():
    return FLSQuoteGenerator(
        landed_cost_inr_per_kwh=8.20,
        base_margin_inr_per_kwh=-1.30,
    )


def test_fls_quote_high_confidence_peak(qg):
    """HT peak median 1.61% MAPE → ~95% firmness cap."""
    q = qg.quote(
        contract_id="FLS-001", buyer_kw=500,
        window_start=time(18), window_end=time(22),
        peak_block_mape_pct=1.61,
    )
    assert q.firmness_pct == 95.0
    # Offered price should be below landed (that's the whole value prop)
    assert q.offered_price_inr_per_kwh < 8.20
    assert q.underlying_mape_pct == 1.61


def test_fls_quote_low_confidence_floored(qg):
    q = qg.quote(
        contract_id="FLS-002", buyer_kw=500,
        window_start=time(18), window_end=time(22),
        peak_block_mape_pct=40.0,
    )
    assert q.firmness_pct == 70.0  # floored
    assert "too high" in q.rationale


def test_fls_quote_rejects_negative_kw(qg):
    with pytest.raises(ValueError):
        qg.quote(
            contract_id="x", buyer_kw=-10,
            window_start=time(18), window_end=time(22),
            peak_block_mape_pct=5.0,
        )


def test_fls_quote_pricing_widens_with_risk(qg):
    """As firmness drops, offered price should rise."""
    q_firm = qg.quote(
        contract_id="a", buyer_kw=500,
        window_start=time(18), window_end=time(22),
        peak_block_mape_pct=2.0,
    )
    q_loose = qg.quote(
        contract_id="b", buyer_kw=500,
        window_start=time(18), window_end=time(22),
        peak_block_mape_pct=15.0,
    )
    assert q_loose.offered_price_inr_per_kwh > q_firm.offered_price_inr_per_kwh
    assert q_loose.firmness_pct < q_firm.firmness_pct
