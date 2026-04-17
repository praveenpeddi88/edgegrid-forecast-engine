"""
Natural-language audit strings for every dispatch decision.

Spec hard-constraint #4: "Every dispatch decision must emit an audit string."
Format: "[HH:MM] [action] [N kWh] because [reason] (confidence [P]%)."

The audit layer is the regulatory/CFO interface — every charge/discharge must
be explainable in one sentence. The three aha moments in the spec all route
through this module:

    3. *"The audit column explains why the battery charged at 02:00
       — and a regulator can read it."*

Templates are intentionally short and human. Avoid jargon.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

Action = Literal["charge", "discharge", "hold", "curtail_solar", "peak_shave"]

# Regex used by tests to validate the audit-string format (spec criterion #5,
# test case 4).
AUDIT_PATTERN = re.compile(
    r"^\[(?P<time>\d{2}:\d{2})\] (?P<action>charge|discharge|hold|curtail solar|peak shave) "
    r"(?P<kwh>\d+(?:\.\d+)?) kWh because .+ \(confidence (?P<pct>\d{1,3})%\)\.$"
)


@dataclass
class AuditContext:
    """Minimal inputs to build a single audit string."""

    timestamp: datetime
    action: Action
    kwh: float
    iex_price_inr: float
    landed_cost_inr: float
    block_mape_pct: float
    kva_peak_shaved: float = 0.0
    tariff_inr_per_kva: float = 400.0
    aggression_threshold_pct: float = 15.0


def _confidence_from_mape(mape_pct: float) -> int:
    """Convert MAPE % to a 0–100 confidence integer. 1% MAPE → 99% confidence."""
    c = 100.0 - float(mape_pct)
    return max(0, min(100, int(round(c))))


def _fmt_price(v: float) -> str:
    return f"\u20B9{v:.2f}"  # ₹


def _reason_for(ctx: AuditContext) -> str:
    """Pick the reason template based on action and economics."""
    delta = ctx.landed_cost_inr - ctx.iex_price_inr
    if ctx.action == "charge":
        if delta > 0:
            return (
                f"IEX price {_fmt_price(ctx.iex_price_inr)}/kWh is "
                f"{_fmt_price(delta)} below landed grid tariff"
            )
        return (
            f"pre-positioning SOC ahead of peak "
            f"(IEX {_fmt_price(ctx.iex_price_inr)}/kWh)"
        )
    if ctx.action == "discharge":
        if delta < 0:
            return (
                f"IEX price {_fmt_price(ctx.iex_price_inr)}/kWh is "
                f"{_fmt_price(-delta)} above landed cost — arbitrage window"
            )
        return (
            f"discharging into peak demand "
            f"(IEX {_fmt_price(ctx.iex_price_inr)}/kWh)"
        )
    if ctx.action == "peak_shave":
        annual_savings = ctx.kva_peak_shaved * ctx.tariff_inr_per_kva * 12
        return (
            f"shaving {ctx.kva_peak_shaved:.0f} kVA peak "
            f"(\u20B9{annual_savings:,.0f}/yr at {_fmt_price(ctx.tariff_inr_per_kva)}/kVA)"
        )
    if ctx.action == "curtail_solar":
        return "curtailing solar — grid import cheaper this block"
    # hold
    if ctx.block_mape_pct > ctx.aggression_threshold_pct:
        return (
            f"holding SOC — block MAPE {ctx.block_mape_pct:.1f}% above "
            f"{ctx.aggression_threshold_pct:.0f}% aggression threshold"
        )
    return "holding SOC — no commercial edge in this block"


def build_audit_string(ctx: AuditContext) -> str:
    """
    Render a one-sentence audit string in the spec-mandated format.

    Examples
    --------
    >>> from datetime import datetime
    >>> s = build_audit_string(AuditContext(
    ...     timestamp=datetime(2026, 4, 17, 2, 0),
    ...     action="charge", kwh=120.5,
    ...     iex_price_inr=2.80, landed_cost_inr=6.90,
    ...     block_mape_pct=9.0,
    ... ))
    >>> s.startswith("[02:00] charge 120.5 kWh because")
    True
    """
    if math.isnan(ctx.kwh) or ctx.kwh < 0:
        raise ValueError(f"audit kwh must be non-negative, got {ctx.kwh}")
    confidence = _confidence_from_mape(ctx.block_mape_pct)
    action_phrase = {
        "charge": "charge",
        "discharge": "discharge",
        "hold": "hold",
        "curtail_solar": "curtail solar",
        "peak_shave": "peak shave",
    }[ctx.action]
    reason = _reason_for(ctx)
    kwh_str = f"{ctx.kwh:.1f}" if ctx.kwh % 1 else f"{int(ctx.kwh)}"
    return (
        f"[{ctx.timestamp:%H:%M}] {action_phrase} {kwh_str} kWh "
        f"because {reason} (confidence {confidence}%)."
    )


def validate_audit_format(s: str) -> bool:
    """Used by tests to assert every non-idle row has a well-formed string."""
    return bool(AUDIT_PATTERN.match(s))


def classify_action(
    *, bess_charge_wh: float, bess_discharge_wh: float, solar_curtail_wh: float,
    kva_shaved: float = 0.0, tolerance_wh: float = 1.0,
) -> Action:
    """
    Infer the Action label from raw dispatch decision variables.
    Peak-shave takes precedence over discharge if kVA was actively shaved.
    """
    if solar_curtail_wh > tolerance_wh:
        return "curtail_solar"
    if bess_charge_wh > tolerance_wh:
        return "charge"
    if bess_discharge_wh > tolerance_wh:
        if kva_shaved > 0:
            return "peak_shave"
        return "discharge"
    return "hold"
