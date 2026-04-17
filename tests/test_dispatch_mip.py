"""
Tests for the v2 PuLP MILP dispatch optimizer.

Covers plan §A6 tests #1-3:
    1. Determinism — same inputs → same schedule twice.
    2. Constraint sanity — SOC ∈ [10%, 90%]; no simultaneous charge+discharge.
    3. Confidence weighting — low-confidence block discharges ≤ high-confidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edgegrid_forecast.dispatch.audit import validate_audit_format
from edgegrid_forecast.dispatch.optimizer_v2 import (
    BESSSpec,
    HORIZON_INTERVALS,
    OptimizerConfig,
    TariffSpec,
    optimize_dispatch,
)


# ───────────────────── Fixtures ──────────────────────────────────────────────


def _make_forecast(as_of: pd.Timestamp, horizon_30min: int = 96) -> pd.DataFrame:
    """Synthetic 30-min forecast mimicking `predict()` output schema."""
    idx = pd.date_range(as_of, periods=horizon_30min, freq="30min")
    # Simple load profile: valley night, peak 18-22
    hours = idx.hour + idx.minute / 60
    base = 400 + 200 * np.sin((hours - 6) / 24 * 2 * np.pi)
    peak_kick = np.where((hours >= 18) & (hours < 22), 300, 0)
    forecast_wh = base + peak_kick
    # Block labels
    def blk(h):
        if 22 <= h or h < 6:
            return "night"
        if 6 <= h < 10:
            return "morning"
        if 10 <= h < 18:
            return "solar"
        return "peak"
    labels = [blk(h) for h in hours]
    block_mape = [9.0 if l != "peak" else 4.2 for l in labels]  # spec-aligned
    return pd.DataFrame(
        {
            "forecast_wh": forecast_wh,
            "confidence_low": forecast_wh * 0.85,
            "confidence_high": forecast_wh * 1.15,
            "block_label": labels,
            "historical_block_mape": block_mape,
        },
        index=idx,
    )


def _make_prices(as_of: pd.Timestamp, n: int = HORIZON_INTERVALS) -> pd.DataFrame:
    """Synthetic IEX DAM prices: cheap night ~2.5, peak evening ~9.5."""
    idx = pd.date_range(as_of, periods=n, freq="15min")
    hours = idx.hour + idx.minute / 60
    price = 5.5 + 3.5 * np.sin((hours - 14) / 24 * 2 * np.pi)
    return pd.DataFrame({"iex_price_inr": price}, index=idx)


@pytest.fixture
def base_inputs():
    as_of = pd.Timestamp("2026-04-17 00:00:00")
    return {
        "forecast_df": _make_forecast(as_of),
        "prices_df": _make_prices(as_of),
        "bess": BESSSpec(capacity_kwh=1000.0, duration_h=4.0),
        "tariff": TariffSpec(landed_cost_inr_per_kwh=8.2),
        "cfg": OptimizerConfig(solver_time_limit_seconds=20),
    }


# ───────────────────── Test cases ────────────────────────────────────────────


def test_determinism(base_inputs):
    """Same inputs → identical schedule, twice."""
    s1 = optimize_dispatch(**base_inputs)
    s2 = optimize_dispatch(**base_inputs)
    assert s1.solver_status in ("Optimal", "Not Solved", "Feasible")
    # Key numeric columns should match to 1e-3
    for col in ("bess_charge_kwh", "bess_discharge_kwh", "grid_import_kwh", "soc_kwh"):
        np.testing.assert_allclose(
            s1.df[col].to_numpy(), s2.df[col].to_numpy(), atol=1e-3,
            err_msg=f"determinism failed on {col}",
        )


def test_constraint_sanity(base_inputs):
    """SOC ∈ [10%, 90%]; never simultaneous charge + discharge."""
    s = optimize_dispatch(**base_inputs)
    bess = base_inputs["bess"]
    assert (s.df["soc_kwh"] >= bess.soc_min_kwh - 1e-6).all()
    assert (s.df["soc_kwh"] <= bess.soc_max_kwh + 1e-6).all()
    # Mutex: charge * discharge ≈ 0 in every interval
    simult = (s.df["bess_charge_kwh"] > 1e-3) & (s.df["bess_discharge_kwh"] > 1e-3)
    assert not simult.any(), f"simultaneous charge/discharge in {simult.sum()} intervals"


def test_confidence_weighting_reduces_aggression(base_inputs):
    """
    When peak-block MAPE is high (low confidence), the optimizer should
    discharge ≤ the high-confidence case. Tests that `historical_block_mape`
    actually scales aggression.
    """
    # High-confidence baseline (peak at 4.2% MAPE)
    s_high = optimize_dispatch(**base_inputs)

    # Force low confidence: inflate peak-block MAPE to 40%
    low_conf = base_inputs.copy()
    low_conf["forecast_df"] = base_inputs["forecast_df"].copy()
    low_conf["forecast_df"]["historical_block_mape"] = np.where(
        low_conf["forecast_df"]["block_label"] == "peak", 40.0,
        low_conf["forecast_df"]["historical_block_mape"],
    )
    s_low = optimize_dispatch(**low_conf)

    hi_discharge = s_high.df["bess_discharge_kwh"].sum()
    lo_discharge = s_low.df["bess_discharge_kwh"].sum()
    # Low-confidence → at most equal discharge (strictly less is typical)
    assert lo_discharge <= hi_discharge + 1e-3, (
        f"low-confidence case discharged more ({lo_discharge:.2f}) "
        f"than high-confidence ({hi_discharge:.2f})"
    )


def test_audit_string_coverage(base_inputs):
    """Every row must emit a well-formed audit string."""
    s = optimize_dispatch(**base_inputs)
    assert len(s.audit_strings) == HORIZON_INTERVALS
    bad = [a for a in s.audit_strings if not validate_audit_format(a)]
    assert not bad, f"malformed audit strings: {bad[:3]}"


def test_schedule_dataframe_schema(base_inputs):
    """DispatchScheduleV2.df must have all declared columns."""
    from edgegrid_forecast.dispatch.optimizer_v2 import DispatchScheduleV2
    s = optimize_dispatch(**base_inputs)
    for col in DispatchScheduleV2.EXPECTED_COLUMNS:
        assert col in s.df.columns, f"missing column: {col}"
    assert len(s.df) == HORIZON_INTERVALS
    assert s.df.index.name == "timestamp"
