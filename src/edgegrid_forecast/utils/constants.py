"""
Domain constants for Indian power distribution grid economics.

Sources:
- APEPDCL tariff orders (FY24-25)
- IEX Day-Ahead Market historical data
- BESS Explorer domain knowledge (EdgeGrid internal)
- CERC regulations for open access charges

These constants encode hard-won domain knowledge. Change with extreme care.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

# ─── Grid Loss Factors ───────────────────────────────────────────────────────
# These are cascaded: each applies to the remaining after the previous loss.
# Source: APEPDCL tariff order, applicable to HT consumers

CTU_LOSS = 0.039       # Central Transmission Utility loss (3.9%)
STU_LOSS = 0.0275      # State Transmission Utility loss (2.75%)
DIST_LOSS = 0.0272     # Distribution loss (2.72%)

# ─── Network Charges (INR/kWh) ───────────────────────────────────────────────
# Applicable for open access / IEX procurement at APEPDCL substations

SLDC_CHARGE = 0.41     # State Load Dispatch Centre charge
CROSS_SUBSIDY = 0.31   # Cross-subsidy surcharge
ADDITIONAL_SURCHARGE = 0.47  # Additional surcharge

DEMAND_CHARGE_INR_PER_KVA_MONTH = 475  # HT-I demand charge


def landed_cost_from_iex(iex_price_inr_kwh: float) -> float:
    """
    Convert IEX DAM clearing price to landed cost at substation busbar.

    The formula accounts for:
    1. Cascaded transmission & distribution losses
    2. SLDC operating charge
    3. Cross-subsidy surcharge
    4. Additional surcharge

    Args:
        iex_price_inr_kwh: IEX Day-Ahead Market price in INR/kWh

    Returns:
        Landed cost in INR/kWh at the consumer busbar
    """
    loss_adjusted = iex_price_inr_kwh / (
        (1 - CTU_LOSS) * (1 - STU_LOSS) * (1 - DIST_LOSS)
    )
    return loss_adjusted + SLDC_CHARGE + CROSS_SUBSIDY + ADDITIONAL_SURCHARGE


# ─── BESS Parameters ─────────────────────────────────────────────────────────

class ChargingStrategy(str, Enum):
    """Battery charging strategies for dispatch optimization."""
    SOLAR_SURPLUS = "solar_surplus"   # Charge only from excess solar (solar > demand)
    FULL_SOLAR = "full_solar"         # Charge from all solar generation
    CHEAP_GRID = "cheap_grid"         # Solar first, then grid when landed < FLS tariff


@dataclass(frozen=True)
class BESSConstraints:
    """Physical constraints for battery energy storage system."""
    min_soc: float = 0.10          # Minimum state of charge (10%)
    max_soc: float = 0.90          # Maximum state of charge (90%)
    round_trip_efficiency: float = 0.88  # Round-trip efficiency (88%)
    degradation_annual_pct: float = 2.5  # Annual capacity degradation
    calendar_life_years: int = 15       # Expected calendar life
    cycle_life: int = 6000             # Expected cycle life at 80% DoD


# ─── IEX Price Matrix (FY24-25) ──────────────────────────────────────────────
# 12 months (Apr=0 .. Mar=11) × 24 hours (0-23)
# Source: IEX DAM historical averages, FY2024-25
# Units: INR/kWh

IEX_DAM_PRICES_FY2425: List[List[float]] = [
    # Apr (month index 0)
    [3.50, 3.20, 3.00, 2.90, 2.85, 3.10, 3.80, 5.20, 6.50, 7.20, 7.80, 8.00,
     7.50, 7.00, 6.50, 6.00, 5.50, 5.80, 7.50, 9.00, 8.50, 7.00, 5.50, 4.20],
    # May
    [3.80, 3.50, 3.20, 3.10, 3.00, 3.30, 4.20, 5.80, 7.00, 7.80, 8.50, 9.00,
     8.50, 8.00, 7.50, 7.00, 6.50, 6.80, 8.50, 10.0, 9.50, 7.50, 5.80, 4.50],
    # Jun
    [4.00, 3.70, 3.50, 3.30, 3.20, 3.50, 4.50, 6.00, 7.50, 8.20, 9.00, 9.50,
     9.00, 8.50, 8.00, 7.50, 7.00, 7.20, 9.00, 10.5, 10.0, 8.00, 6.00, 4.80],
    # Jul
    [3.80, 3.50, 3.30, 3.20, 3.10, 3.40, 4.30, 5.50, 7.00, 7.50, 8.00, 8.50,
     8.00, 7.50, 7.00, 6.80, 6.50, 6.80, 8.50, 9.50, 9.00, 7.50, 5.80, 4.50],
    # Aug
    [3.50, 3.30, 3.10, 3.00, 2.90, 3.20, 4.00, 5.20, 6.50, 7.20, 7.80, 8.00,
     7.50, 7.00, 6.50, 6.20, 6.00, 6.30, 8.00, 9.00, 8.50, 7.00, 5.50, 4.20],
    # Sep
    [3.20, 3.00, 2.80, 2.70, 2.60, 2.90, 3.70, 5.00, 6.20, 7.00, 7.50, 7.80,
     7.20, 6.80, 6.20, 5.80, 5.50, 5.80, 7.50, 8.50, 8.00, 6.80, 5.20, 4.00],
    # Oct
    [3.00, 2.80, 2.60, 2.50, 2.50, 2.80, 3.50, 4.80, 6.00, 6.50, 7.00, 7.20,
     6.80, 6.50, 6.00, 5.50, 5.20, 5.50, 7.00, 8.00, 7.50, 6.50, 5.00, 3.80],
    # Nov
    [2.80, 2.60, 2.50, 2.40, 2.40, 2.70, 3.20, 4.50, 5.50, 6.00, 6.50, 6.80,
     6.50, 6.00, 5.50, 5.20, 5.00, 5.50, 7.00, 7.80, 7.20, 6.00, 4.50, 3.50],
    # Dec
    [2.50, 2.30, 2.20, 2.10, 2.10, 2.40, 3.00, 4.20, 5.00, 5.50, 6.00, 6.20,
     6.00, 5.50, 5.20, 5.00, 4.80, 5.20, 6.50, 7.50, 7.00, 5.80, 4.20, 3.20],
    # Jan
    [2.80, 2.60, 2.50, 2.40, 2.40, 2.70, 3.30, 4.50, 5.50, 6.00, 6.50, 6.80,
     6.50, 6.00, 5.50, 5.20, 5.00, 5.50, 7.00, 8.00, 7.50, 6.20, 4.80, 3.50],
    # Feb
    [3.00, 2.80, 2.60, 2.50, 2.50, 2.80, 3.50, 4.80, 6.00, 6.50, 7.00, 7.20,
     6.80, 6.50, 6.00, 5.50, 5.20, 5.50, 7.00, 8.00, 7.50, 6.50, 5.00, 3.80],
    # Mar
    [3.20, 3.00, 2.80, 2.70, 2.60, 2.90, 3.70, 5.00, 6.20, 7.00, 7.50, 7.80,
     7.20, 6.80, 6.20, 5.80, 5.50, 5.80, 7.50, 8.50, 8.00, 6.80, 5.20, 4.00],
]


def fy_month_index(calendar_month: int) -> int:
    """
    Convert calendar month (1-12) to financial year index (0-11).
    April=0, May=1, ..., March=11.

    This is a common source of bugs — the CSV data uses calendar months (integers),
    but the IEX price array is indexed by financial year.
    """
    return ((calendar_month - 4) + 12) % 12


def get_iex_price(calendar_month: int, hour: int) -> float:
    """Get IEX DAM price for a given calendar month and hour."""
    return IEX_DAM_PRICES_FY2425[fy_month_index(calendar_month)][hour]


def get_landed_price(calendar_month: int, hour: int) -> float:
    """Get landed cost at busbar for a given calendar month and hour."""
    return landed_cost_from_iex(get_iex_price(calendar_month, hour))


# ─── APEPDCL Substation Definitions ──────────────────────────────────────────

@dataclass(frozen=True)
class SubstationConfig:
    """Configuration for an APEPDCL substation."""
    code: str
    name: str
    fls_capacity_mw: float      # Firm & Long-term Supply capacity
    grid_cos_inr_kwh: float     # Grid Cost of Supply (FLS tariff)
    latitude: float
    longitude: float


# The 6 HT consumers from the uploaded DISCOM data
HT_CONSUMERS: Dict[str, Dict] = {
    "RJY1197": {"region": "Rajahmundry", "lat": 17.0005, "lon": 81.8040},
    "RJY1622": {"region": "Rajahmundry", "lat": 17.0005, "lon": 81.8040},
    "SKL724":  {"region": "Srikakulam",  "lat": 18.2949, "lon": 83.8938},
    "VSP2315": {"region": "Visakhapatnam", "lat": 17.6868, "lon": 83.2185},
    "VSP2432": {"region": "Visakhapatnam", "lat": 17.6868, "lon": 83.2185},
    "VSP2439": {"region": "Visakhapatnam", "lat": 17.6868, "lon": 83.2185},
}

# ─── Time-of-Day Tariff Slabs ────────────────────────────────────────────────
# APEPDCL HT consumers have time-of-day differentiated tariff

TOD_SLABS = {
    "off_peak":  {"hours": list(range(0, 6)) + [22, 23], "multiplier": 0.90},
    "normal":    {"hours": list(range(6, 10)) + list(range(14, 18)), "multiplier": 1.00},
    "peak":      {"hours": list(range(10, 14)) + list(range(18, 22)), "multiplier": 1.20},
}


def get_tod_multiplier(hour: int) -> float:
    """Get time-of-day tariff multiplier for a given hour."""
    for slab in TOD_SLABS.values():
        if hour in slab["hours"]:
            return slab["multiplier"]
    return 1.0


# ─── Unit Conversion Helpers ─────────────────────────────────────────────────
# These are the most common source of bugs in energy calculations.

def mwh_inr_kwh_to_lakhs(value: float) -> float:
    """Convert MWh × INR/kWh to INR Lakhs. Divide by 100."""
    return value / 100

def mwh_inr_kwh_to_crores(value: float) -> float:
    """Convert MWh × INR/kWh to INR Crores. Divide by 10,000."""
    return value / 10_000

def kwh_to_mwh(kwh: float) -> float:
    return kwh / 1000

def mw_to_kw(mw: float) -> float:
    return mw * 1000

def vah_to_kwh(vah: float) -> float:
    """Convert VAh (volt-ampere-hours) to kWh. Assumes unity power factor."""
    return vah / 1000
