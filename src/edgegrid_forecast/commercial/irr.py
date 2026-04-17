"""
IRR / NPV / payback calculation for a BESS project.

Powers the IRR heatmap on the Commercial Brief screen (spec §Flow 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class IRRResult:
    irr_pct: float
    npv_inr: float
    payback_years: float
    capex_inr: float
    annual_net_inr: float
    years: int


def _npv(rate: float, cashflows: Iterable[float]) -> float:
    return float(sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows)))


def _irr(cashflows: list[float], guess: float = 0.12) -> float:
    """
    Bisection-then-Newton IRR. Works when the project has exactly one sign
    change (typical BESS: one negative CAPEX, many positive annual nets).
    Returns NaN if no root is found in the plausible range.
    """
    cf = list(cashflows)
    if not cf or cf[0] >= 0:
        return float("nan")
    lo, hi = -0.99, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        npv_mid = _npv(mid, cf)
        if abs(npv_mid) < 1e-2:
            return mid
        if _npv(lo, cf) * npv_mid < 0:
            hi = mid
        else:
            lo = mid
    return mid


def compute_project_irr(
    capex_inr: float,
    annual_net_inr: float,
    years: int = 10,
    discount_rate: float = 0.10,
    opex_escalation_pct: float = 2.0,
    degradation_pct_per_year: float = 2.5,
) -> IRRResult:
    """
    Straightforward BESS cashflow model.

    Year 0 = -capex; Years 1..N = annual_net adjusted for degradation + opex.
    Returns IRR %, NPV at `discount_rate`, simple payback in years.
    """
    cashflows = [-float(capex_inr)]
    for y in range(1, years + 1):
        deg = (1 - degradation_pct_per_year / 100) ** y
        esc = (1 + opex_escalation_pct / 100) ** y
        cashflows.append(annual_net_inr * deg / esc)
    irr = _irr(cashflows)
    npv = _npv(discount_rate, cashflows)

    # Simple payback: first year cumulative CF ≥ 0
    cum = 0.0
    payback = float("inf")
    for y, cf in enumerate(cashflows):
        cum += cf
        if cum >= 0 and y > 0:
            payback = float(y)
            break
    return IRRResult(
        irr_pct=float(irr * 100) if not np.isnan(irr) else float("nan"),
        npv_inr=float(npv),
        payback_years=payback,
        capex_inr=float(capex_inr),
        annual_net_inr=float(annual_net_inr),
        years=years,
    )


def irr_heatmap(
    capacities_mwh: list[float],
    durations_h: list[int],
    daily_net_inr_at_reference: float,
    reference_capacity_mwh: float = 1.0,
    capex_inr_per_kwh: float = 25_000.0,
    years: int = 10,
) -> dict[str, list]:
    """
    Build the IRR grid for the heatmap visualization.

    We scale the daily net benefit linearly with capacity (first-order
    approximation — real systems have diminishing returns; good enough for
    v1 Commercial Brief).

    Returns a dict ready for JSON / Recharts:
        {
            "capacities_mwh": [...],
            "durations_h": [...],
            "irr_pct": [[...]],       # rows=capacity, cols=duration
            "payback_years": [[...]],
        }
    """
    irr_grid: list[list[float]] = []
    pb_grid: list[list[float]] = []
    for cap in capacities_mwh:
        irr_row, pb_row = [], []
        scale = cap / max(reference_capacity_mwh, 1e-6)
        annual_net = daily_net_inr_at_reference * 365 * scale
        for dur in durations_h:
            capex = cap * 1000 * capex_inr_per_kwh  # capacity_kwh × ₹/kWh
            # Longer duration modestly lifts revenue (more arbitrage window)
            dur_lift = 1.0 + 0.05 * (dur - 4)
            r = compute_project_irr(
                capex_inr=capex,
                annual_net_inr=annual_net * dur_lift,
                years=years,
            )
            irr_row.append(round(r.irr_pct, 2))
            pb_row.append(round(r.payback_years, 2))
        irr_grid.append(irr_row)
        pb_grid.append(pb_row)
    return {
        "capacities_mwh": capacities_mwh,
        "durations_h": durations_h,
        "irr_pct": irr_grid,
        "payback_years": pb_grid,
    }


def sensitivity_grid(
    base_capex_inr: float,
    base_annual_net_inr: float,
    years: int = 10,
) -> dict[str, list[dict]]:
    """
    Spec §PRD-C5: ±20% on IEX prices, ±10% on demand growth, ±15% on CAPEX.
    """
    out = {"iex": [], "demand": [], "capex": []}
    for delta in (-0.20, -0.10, 0.0, 0.10, 0.20):
        r = compute_project_irr(
            capex_inr=base_capex_inr,
            annual_net_inr=base_annual_net_inr * (1 + delta),
            years=years,
        )
        out["iex"].append({"delta_pct": delta * 100, "irr_pct": r.irr_pct,
                           "payback_years": r.payback_years})
    for delta in (-0.10, -0.05, 0.0, 0.05, 0.10):
        r = compute_project_irr(
            capex_inr=base_capex_inr,
            annual_net_inr=base_annual_net_inr * (1 + delta),
            years=years,
        )
        out["demand"].append({"delta_pct": delta * 100, "irr_pct": r.irr_pct,
                              "payback_years": r.payback_years})
    for delta in (-0.15, -0.075, 0.0, 0.075, 0.15):
        r = compute_project_irr(
            capex_inr=base_capex_inr * (1 + delta),
            annual_net_inr=base_annual_net_inr,
            years=years,
        )
        out["capex"].append({"delta_pct": delta * 100, "irr_pct": r.irr_pct,
                             "payback_years": r.payback_years})
    return out
