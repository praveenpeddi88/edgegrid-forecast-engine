"""
FLS (Firm Load-following Supply) quote generator.

Given a buyer window + kW, produce a ₹/kWh offer + firmness guarantee %.
Firmness is derived from the peak-block MAPE of the relevant meters — this
is the aha moment #2 from the spec:

    "HT meters at 1.61% MAPE during 18-22h means I can sign a 95%-firm
     FLS contract."
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import time
from typing import Optional


@dataclass
class FLSQuote:
    contract_id: str
    buyer_kw: float
    window_start: str  # "18:00"
    window_end: str    # "22:00"
    weekdays_only: bool
    landed_cost_inr_per_kwh: float
    offered_price_inr_per_kwh: float
    firmness_pct: float       # e.g. 95.0 → "95%-firm"
    underlying_mape_pct: float  # peak-block MAPE that backs the firmness
    tenor_months: int
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


class FLSQuoteGenerator:
    """
    Converts peak-block MAPE + dispatch confidence → commercial quote.

    The firmness % is a simple transform:  firmness = max(70, 100 - peak_mape).
    Below 70% we refuse to quote; above 95% we cap (no credible "100% firm" in
    a real grid). The pricing adds a margin over landed cost that widens as
    firmness drops.
    """

    def __init__(
        self,
        landed_cost_inr_per_kwh: float = 8.20,
        base_margin_inr_per_kwh: float = -1.30,  # negative = we offer cheaper
        firmness_floor_pct: float = 70.0,
        firmness_cap_pct: float = 95.0,
    ):
        self.landed = landed_cost_inr_per_kwh
        self.base_margin = base_margin_inr_per_kwh
        self.floor = firmness_floor_pct
        self.cap = firmness_cap_pct

    def quote(
        self,
        contract_id: str,
        buyer_kw: float,
        window_start: time,
        window_end: time,
        peak_block_mape_pct: float,
        weekdays_only: bool = True,
        tenor_months: int = 12,
    ) -> FLSQuote:
        if buyer_kw <= 0:
            raise ValueError("buyer_kw must be positive")
        if peak_block_mape_pct < 0:
            raise ValueError("MAPE must be non-negative")

        firmness = max(
            self.floor, min(self.cap, 100.0 - peak_block_mape_pct)
        )
        # Every 5 pp of firmness shortfall adds ₹0.40 to offer price
        shortfall = self.cap - firmness
        risk_uplift = shortfall / 5.0 * 0.40
        offered = self.landed + self.base_margin + risk_uplift
        # Never offer below ₹4/kWh — a floor to avoid implausible quotes
        offered = max(offered, 4.00)

        if firmness <= self.floor:
            rationale = (
                f"Peak-block MAPE {peak_block_mape_pct:.2f}% is too high to "
                f"quote above {self.floor:.0f}% firmness. Recommend adding "
                "additional BESS capacity or lowering contracted kW."
            )
        else:
            rationale = (
                f"Peak-block MAPE {peak_block_mape_pct:.2f}% supports "
                f"{firmness:.0f}%-firm quote. Offered price "
                f"\u20B9{offered:.2f}/kWh is "
                f"\u20B9{self.landed - offered:.2f} below landed cost."
            )

        return FLSQuote(
            contract_id=contract_id,
            buyer_kw=float(buyer_kw),
            window_start=window_start.strftime("%H:%M"),
            window_end=window_end.strftime("%H:%M"),
            weekdays_only=weekdays_only,
            landed_cost_inr_per_kwh=self.landed,
            offered_price_inr_per_kwh=round(offered, 2),
            firmness_pct=round(firmness, 2),
            underlying_mape_pct=round(peak_block_mape_pct, 2),
            tenor_months=tenor_months,
            rationale=rationale,
        )
