"""
Tests for dispatch.audit — natural-language explanation strings.

Covers plan §A6 test #4: "Audit string coverage — every non-idle row has a
well-formed `[time] [action] [N kWh] because [reason] (confidence [P]%)`
string."
"""

from datetime import datetime

import pytest

from edgegrid_forecast.dispatch.audit import (
    AUDIT_PATTERN,
    AuditContext,
    build_audit_string,
    classify_action,
    validate_audit_format,
)


@pytest.fixture
def charge_ctx():
    return AuditContext(
        timestamp=datetime(2026, 4, 17, 2, 0),
        action="charge",
        kwh=120.5,
        iex_price_inr=2.80,
        landed_cost_inr=6.90,
        block_mape_pct=9.0,
    )


def test_charge_audit_matches_format(charge_ctx):
    s = build_audit_string(charge_ctx)
    assert validate_audit_format(s), f"string did not match pattern: {s}"
    m = AUDIT_PATTERN.match(s)
    assert m.group("time") == "02:00"
    assert m.group("action") == "charge"
    assert m.group("kwh") == "120.5"
    assert 80 <= int(m.group("pct")) <= 95  # MAPE 9% → confidence ~91%


def test_discharge_arbitrage_reason_mentions_iex_delta():
    ctx = AuditContext(
        timestamp=datetime(2026, 4, 17, 19, 15),
        action="discharge",
        kwh=240.0,
        iex_price_inr=9.50,
        landed_cost_inr=6.90,
        block_mape_pct=4.2,  # peak-block median in the spec
    )
    s = build_audit_string(ctx)
    assert "above landed cost" in s
    assert "arbitrage window" in s
    m = AUDIT_PATTERN.match(s)
    assert m is not None
    assert int(m.group("pct")) == 96  # 100 - 4


def test_hold_reason_flags_low_confidence():
    ctx = AuditContext(
        timestamp=datetime(2026, 4, 17, 11, 30),
        action="hold",
        kwh=0,
        iex_price_inr=4.1,
        landed_cost_inr=6.9,
        block_mape_pct=26.7,  # the 67003309 solar-block MAPE in manifest
        aggression_threshold_pct=15.0,
    )
    s = build_audit_string(ctx)
    assert "above" in s and "aggression threshold" in s
    assert validate_audit_format(s)


def test_peak_shave_reason_quotes_annual_savings():
    ctx = AuditContext(
        timestamp=datetime(2026, 4, 17, 18, 45),
        action="peak_shave",
        kwh=150.0,
        iex_price_inr=8.5,
        landed_cost_inr=6.9,
        block_mape_pct=1.61,  # HT peak median
        kva_peak_shaved=45.0,
        tariff_inr_per_kva=400.0,
    )
    s = build_audit_string(ctx)
    assert "45 kVA peak" in s
    # 45 × 400 × 12 = 216,000
    assert "216,000" in s
    assert validate_audit_format(s)


def test_curtail_solar_branch():
    ctx = AuditContext(
        timestamp=datetime(2026, 4, 17, 13, 0),
        action="curtail_solar",
        kwh=60.0,
        iex_price_inr=2.2,
        landed_cost_inr=6.9,
        block_mape_pct=6.0,
    )
    s = build_audit_string(ctx)
    assert "curtail solar" in s
    assert validate_audit_format(s)


def test_negative_kwh_rejected():
    with pytest.raises(ValueError):
        build_audit_string(AuditContext(
            timestamp=datetime(2026, 4, 17, 0, 0),
            action="charge", kwh=-5,
            iex_price_inr=3.0, landed_cost_inr=7.0, block_mape_pct=5.0,
        ))


@pytest.mark.parametrize("charge,discharge,curtail,kva,expected", [
    (0, 0, 0, 0, "hold"),
    (100, 0, 0, 0, "charge"),
    (0, 100, 0, 0, "discharge"),
    (0, 100, 0, 50, "peak_shave"),
    (0, 0, 50, 0, "curtail_solar"),
])
def test_classify_action(charge, discharge, curtail, kva, expected):
    got = classify_action(
        bess_charge_wh=charge, bess_discharge_wh=discharge,
        solar_curtail_wh=curtail, kva_shaved=kva,
    )
    assert got == expected
