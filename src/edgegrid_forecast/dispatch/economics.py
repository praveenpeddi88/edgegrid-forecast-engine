"""
Financial calculations for energy dispatch decisions.

Computes:
- Energy cost breakdown (grid vs IEX vs solar vs BESS)
- Demand charge savings from peak shaving
- ROI on BESS investment
- Carbon savings valuation
- Network value (what value does a consumer bring to the VNM cluster)
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from ..utils.constants import (
    DEMAND_CHARGE_INR_PER_KVA_MONTH,
    landed_cost_from_iex,
    mwh_inr_kwh_to_crores,
    mwh_inr_kwh_to_lakhs,
)


@dataclass
class MonthlyEconomics:
    """Monthly financial summary for a consumer."""
    month: int
    year: int
    total_consumption_kwh: float
    peak_demand_kw: float
    cmd_kva: float                   # Contracted Maximum Demand

    # Cost breakdown
    energy_charge_inr: float         # Volume-based energy cost
    demand_charge_inr: float         # Peak demand-based charge
    tod_adjustment_inr: float        # Time-of-day adjustment
    total_bill_inr: float

    # Savings from optimization
    solar_savings_inr: float
    bess_savings_inr: float
    demand_charge_savings_inr: float  # From peak shaving via BESS
    iex_arbitrage_savings_inr: float
    total_savings_inr: float

    # Carbon
    co2_avoided_kg: float            # From solar + BESS dispatch


def compute_demand_charge_savings(
    original_peak_kw: float,
    reduced_peak_kw: float,
    power_factor: float = 0.95,
) -> float:
    """
    Compute monthly demand charge savings from peak shaving.

    APEPDCL charges ₹475/kVA/month for contracted demand.
    If BESS can reduce the peak, the CMD can be renegotiated lower.

    Args:
        original_peak_kw: Peak demand without BESS
        reduced_peak_kw: Peak demand with BESS peak shaving
        power_factor: Average power factor (kW to kVA conversion)
    """
    original_kva = original_peak_kw / power_factor
    reduced_kva = reduced_peak_kw / power_factor

    savings = (original_kva - reduced_kva) * DEMAND_CHARGE_INR_PER_KVA_MONTH
    return max(savings, 0)


def compute_carbon_savings(
    solar_kwh: float,
    bess_shifted_kwh: float,
    grid_emission_factor: float = 0.82,  # India grid: ~0.82 kg CO2/kWh
    solar_emission_factor: float = 0.05,  # Lifecycle solar: ~0.05 kg CO2/kWh
) -> float:
    """
    Compute CO2 avoided from solar generation and BESS-enabled solar shifting.

    India's grid emission factor is ~0.82 kg CO2/kWh (CEA 2024).
    Solar lifecycle is ~0.05 kg CO2/kWh.
    """
    # Solar directly displaces grid
    solar_savings = solar_kwh * (grid_emission_factor - solar_emission_factor)
    # BESS shifts solar to peak hours (displaces higher-carbon peak generation)
    # Peak generation in India is often gas/diesel: ~0.95 kg CO2/kWh
    bess_savings = bess_shifted_kwh * (0.95 - solar_emission_factor)

    return solar_savings + bess_savings


def compute_network_value(
    consumer_demand_kwh: float,
    cluster_total_kwh: float,
    consumer_solar_kwh: float,
    cluster_solar_kwh: float,
    vnm_savings_inr: float,
) -> Dict[str, float]:
    """
    Compute the value a consumer brings to the VNM energy cluster.

    This is critical for EdgeGrid's marketplace model — each building's
    value is a function of:
    1. Its demand share (how much of cluster demand it represents)
    2. Its solar contribution (prosumer value)
    3. Its demand flexibility (can it shift load?)

    The network value determines:
    - Pricing within the cluster
    - Priority for VNM allocation
    - Revenue share in the marketplace
    """
    demand_share = consumer_demand_kwh / max(cluster_total_kwh, 1)
    solar_share = consumer_solar_kwh / max(cluster_solar_kwh, 1)

    # Net position: positive = net producer, negative = net consumer
    net_position_kwh = consumer_solar_kwh - consumer_demand_kwh

    # Value to cluster (prosumers are more valuable)
    prosumer_premium = 1.5 if net_position_kwh > 0 else 1.0
    network_value = vnm_savings_inr * demand_share * prosumer_premium

    return {
        "demand_share_pct": demand_share * 100,
        "solar_share_pct": solar_share * 100,
        "net_position_kwh": net_position_kwh,
        "is_prosumer": net_position_kwh > 0,
        "network_value_inr": network_value,
        "cluster_contribution_score": demand_share * 0.4 + solar_share * 0.6,
    }


def compute_monthly_economics(
    demand_kwh: np.ndarray,         # Hourly demand for the month
    solar_kwh: np.ndarray,          # Hourly solar generation
    grid_purchase_kwh: np.ndarray,  # From dispatch optimizer
    iex_purchase_kwh: np.ndarray,
    bess_discharge_kwh: np.ndarray,
    fls_tariff: float,
    iex_prices: np.ndarray,
    cmd_kva: float,
    month: int,
    year: int,
) -> MonthlyEconomics:
    """Compute full monthly economics from dispatch results."""

    total_consumption = np.sum(demand_kwh)
    peak_demand = np.max(demand_kwh)

    # Energy charges
    energy_charge = np.sum(grid_purchase_kwh) * fls_tariff
    iex_cost = np.sum(iex_purchase_kwh * np.array([
        landed_cost_from_iex(p) for p in iex_prices
    ]))

    # Demand charges
    actual_peak_kva = peak_demand / 0.95
    demand_charge = max(actual_peak_kva, cmd_kva) * DEMAND_CHARGE_INR_PER_KVA_MONTH

    # Peak with BESS shaving
    net_demand = demand_kwh - bess_discharge_kwh
    reduced_peak = np.max(np.maximum(net_demand, 0))
    demand_savings = compute_demand_charge_savings(peak_demand, reduced_peak)

    # Solar savings
    solar_direct = np.minimum(solar_kwh, demand_kwh)
    solar_savings = np.sum(solar_direct) * fls_tariff

    # BESS savings (arbitrage)
    bess_savings = np.sum(bess_discharge_kwh) * fls_tariff - np.sum(bess_discharge_kwh) * np.mean(iex_prices) * 0.5

    # Carbon
    co2_avoided = compute_carbon_savings(
        np.sum(solar_direct), np.sum(bess_discharge_kwh)
    )

    total_bill = energy_charge + iex_cost + demand_charge
    total_savings = solar_savings + bess_savings + demand_savings

    return MonthlyEconomics(
        month=month,
        year=year,
        total_consumption_kwh=total_consumption,
        peak_demand_kw=peak_demand,
        cmd_kva=cmd_kva,
        energy_charge_inr=energy_charge,
        demand_charge_inr=demand_charge,
        tod_adjustment_inr=0,  # Simplified
        total_bill_inr=total_bill,
        solar_savings_inr=solar_savings,
        bess_savings_inr=bess_savings,
        demand_charge_savings_inr=demand_savings,
        iex_arbitrage_savings_inr=0,  # TODO
        total_savings_inr=total_savings,
        co2_avoided_kg=co2_avoided,
    )
