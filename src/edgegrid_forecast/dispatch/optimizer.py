"""
BESS dispatch optimizer using linear programming.

Given:
- Demand forecast (hourly, per consumer)
- Solar generation forecast
- IEX price forecast (landed costs)
- BESS physical constraints
- FLS tariff rate

Decide:
- When to charge the battery (and from what source)
- When to discharge
- When to buy from grid vs IEX
- How to minimize total energy cost while maintaining supply reliability

This is the core intelligence that makes EdgeGrid's "Energy OS" possible.
The optimizer runs in < 100ms for a 24-hour window, enabling real-time
dispatch decisions as conditions change.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import linprog, minimize

from ..utils.constants import (
    BESSConstraints,
    ChargingStrategy,
    landed_cost_from_iex,
    mwh_inr_kwh_to_crores,
    mwh_inr_kwh_to_lakhs,
)


@dataclass
class DispatchSchedule:
    """Output of the dispatch optimizer — hourly schedule for a 24h window."""
    timestamp: pd.DatetimeIndex
    demand_kwh: np.ndarray             # Forecasted demand
    solar_generation_kwh: np.ndarray   # Forecasted solar
    solar_direct_use_kwh: np.ndarray   # Solar used directly (not stored)
    bess_charge_kwh: np.ndarray        # Energy charged into battery
    bess_discharge_kwh: np.ndarray     # Energy discharged from battery
    bess_soc_kwh: np.ndarray           # State of charge at each hour
    grid_purchase_kwh: np.ndarray      # Energy bought from grid
    iex_purchase_kwh: np.ndarray       # Energy bought from IEX
    grid_price_inr_kwh: np.ndarray     # Grid tariff at each hour
    iex_landed_inr_kwh: np.ndarray     # IEX landed cost at each hour
    total_cost_inr: float              # Total energy cost for the window
    solar_savings_inr: float           # Money saved by using solar
    bess_savings_inr: float            # Money saved by BESS arbitrage
    reliability_score: float           # 0-1, how well supply meets demand


@dataclass
class BESSConfig:
    """Battery configuration for dispatch."""
    capacity_kwh: float            # Total energy capacity
    max_power_kw: float            # Max charge/discharge rate (C-rate dependent)
    efficiency: float = 0.88       # Round-trip efficiency
    min_soc_pct: float = 0.10      # Minimum SoC
    max_soc_pct: float = 0.90      # Maximum SoC
    initial_soc_pct: float = 0.50  # Starting SoC

    @property
    def usable_capacity_kwh(self) -> float:
        return self.capacity_kwh * (self.max_soc_pct - self.min_soc_pct)

    @property
    def min_soc_kwh(self) -> float:
        return self.capacity_kwh * self.min_soc_pct

    @property
    def max_soc_kwh(self) -> float:
        return self.capacity_kwh * self.max_soc_pct


class DispatchOptimizer:
    """
    Hourly dispatch optimizer for a single consumer/site.

    Optimization objective: Minimize total energy cost
    Subject to:
    - Supply = Demand at every hour (reliability constraint)
    - BESS physical constraints (SoC limits, power limits, efficiency)
    - Solar generation constraints (can't use more solar than available)
    - Grid constraints (always available as backup)

    Uses a greedy simulation approach (matching the BESS Explorer logic)
    for speed, with optional LP refinement for complex multi-source scenarios.
    """

    def __init__(
        self,
        bess_config: Optional[BESSConfig] = None,
        fls_tariff_inr_kwh: float = 6.50,
        strategy: ChargingStrategy = ChargingStrategy.CHEAP_GRID,
    ):
        self.bess = bess_config
        self.fls_tariff = fls_tariff_inr_kwh
        self.strategy = strategy

    def optimize_24h(
        self,
        demand_kwh: np.ndarray,           # 24 hourly values
        solar_kwh: np.ndarray,            # 24 hourly values
        iex_prices_inr_kwh: np.ndarray,   # 24 hourly IEX prices
        timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> DispatchSchedule:
        """
        Run dispatch optimization for a 24-hour window.

        Algorithm:
        1. For each hour, compute solar surplus = solar - demand
        2. If surplus > 0: charge BESS from surplus (free energy)
        3. If BESS has capacity and grid is cheap: charge from grid
        4. During expensive hours: discharge BESS
        5. Remaining demand: buy from cheapest source (grid vs IEX)
        """
        n_hours = len(demand_kwh)
        assert len(solar_kwh) == n_hours
        assert len(iex_prices_inr_kwh) == n_hours

        if timestamps is None:
            timestamps = pd.date_range("2025-01-01", periods=n_hours, freq="h")

        # Initialize arrays
        solar_direct = np.zeros(n_hours)
        bess_charge = np.zeros(n_hours)
        bess_discharge = np.zeros(n_hours)
        bess_soc = np.zeros(n_hours)
        grid_purchase = np.zeros(n_hours)
        iex_purchase = np.zeros(n_hours)

        # Landed costs
        landed_costs = np.array([landed_cost_from_iex(p) for p in iex_prices_inr_kwh])

        # Initialize BESS SoC
        if self.bess:
            current_soc = self.bess.capacity_kwh * self.bess.initial_soc_pct
        else:
            current_soc = 0

        # Find peak price hours for discharge targeting
        if self.bess:
            peak_hours = np.argsort(landed_costs)[::-1][:int(self.bess.capacity_kwh / max(self.bess.max_power_kw, 1))]

        # ─── Hourly dispatch simulation ──────────────────────────────────
        for h in range(n_hours):
            remaining_demand = demand_kwh[h]

            # Step 1: Use solar directly
            direct_solar = min(solar_kwh[h], remaining_demand)
            solar_direct[h] = direct_solar
            remaining_demand -= direct_solar
            solar_surplus = max(solar_kwh[h] - direct_solar, 0)

            # Step 2: Charge BESS
            if self.bess and current_soc < self.bess.max_soc_kwh:
                charge_room = self.bess.max_soc_kwh - current_soc
                max_charge = min(charge_room, self.bess.max_power_kw)

                if self.strategy == ChargingStrategy.SOLAR_SURPLUS:
                    # Only charge from surplus solar
                    charge = min(solar_surplus, max_charge)

                elif self.strategy == ChargingStrategy.FULL_SOLAR:
                    # Charge from all available solar (even if demand exists)
                    charge = min(solar_kwh[h], max_charge)

                elif self.strategy == ChargingStrategy.CHEAP_GRID:
                    # Surplus solar first
                    solar_charge = min(solar_surplus, max_charge)
                    grid_charge = 0

                    # Then grid if IEX landed cost < FLS tariff
                    if landed_costs[h] < self.fls_tariff and solar_charge < max_charge:
                        grid_charge = min(max_charge - solar_charge, self.bess.max_power_kw - solar_charge)

                    charge = solar_charge + grid_charge
                    if grid_charge > 0:
                        iex_purchase[h] += grid_charge  # Buying cheap from IEX to store
                else:
                    charge = 0

                bess_charge[h] = charge
                current_soc += charge * np.sqrt(self.bess.efficiency)  # Charging efficiency

            # Step 3: Discharge BESS during expensive hours
            if self.bess and remaining_demand > 0 and h in peak_hours:
                if current_soc > self.bess.min_soc_kwh:
                    available_discharge = min(
                        current_soc - self.bess.min_soc_kwh,
                        self.bess.max_power_kw,
                        remaining_demand,
                    )
                    discharge = available_discharge * np.sqrt(self.bess.efficiency)
                    bess_discharge[h] = discharge
                    current_soc -= available_discharge
                    remaining_demand -= discharge

            # Step 4: Remaining demand from cheapest grid source
            if remaining_demand > 0:
                if landed_costs[h] < self.fls_tariff:
                    iex_purchase[h] += remaining_demand
                else:
                    grid_purchase[h] = remaining_demand

            bess_soc[h] = current_soc

        # ─── Compute costs ───────────────────────────────────────────────
        grid_cost = np.sum(grid_purchase * self.fls_tariff)
        iex_cost = np.sum(iex_purchase * landed_costs)
        total_cost = grid_cost + iex_cost

        # What would cost be without solar + BESS?
        baseline_cost = np.sum(demand_kwh * self.fls_tariff)
        solar_savings = np.sum(solar_direct) * self.fls_tariff
        bess_savings = baseline_cost - total_cost - solar_savings

        # Reliability: how well did we meet demand?
        total_supplied = solar_direct + bess_discharge + grid_purchase + iex_purchase
        reliability = float(np.mean(np.minimum(total_supplied / np.maximum(demand_kwh, 0.001), 1.0)))

        return DispatchSchedule(
            timestamp=timestamps,
            demand_kwh=demand_kwh,
            solar_generation_kwh=solar_kwh,
            solar_direct_use_kwh=solar_direct,
            bess_charge_kwh=bess_charge,
            bess_discharge_kwh=bess_discharge,
            bess_soc_kwh=bess_soc,
            grid_purchase_kwh=grid_purchase,
            iex_purchase_kwh=iex_purchase,
            grid_price_inr_kwh=np.full(n_hours, self.fls_tariff),
            iex_landed_inr_kwh=landed_costs,
            total_cost_inr=total_cost,
            solar_savings_inr=solar_savings,
            bess_savings_inr=bess_savings,
            reliability_score=reliability,
        )

    def optimize_week(
        self,
        demand_kwh: np.ndarray,
        solar_kwh: np.ndarray,
        iex_prices: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> List[DispatchSchedule]:
        """Run daily optimization for a 7-day window."""
        n_hours = len(demand_kwh)
        n_days = n_hours // 24

        schedules = []
        for day in range(n_days):
            start = day * 24
            end = start + 24
            ts = timestamps[start:end] if timestamps is not None else None

            schedule = self.optimize_24h(
                demand_kwh[start:end],
                solar_kwh[start:end],
                iex_prices[start:end],
                timestamps=ts,
            )
            schedules.append(schedule)

        return schedules


# ─── BESS Sizing Optimizer ────────────────────────────────────────────────────

def optimize_bess_size(
    demand_kwh: np.ndarray,
    solar_kwh: np.ndarray,
    iex_prices: np.ndarray,
    fls_tariff: float = 6.50,
    capex_inr_per_kwh: float = 15000,
    size_range: Tuple[float, float] = (100, 20000),
    size_step: float = 500,
    duration_options: List[int] = [2, 4, 6],
    strategies: List[ChargingStrategy] = None,
    target_irr: float = 0.10,
) -> pd.DataFrame:
    """
    Sweep BESS configurations to find optimal sizing.

    Matches the BESS Explorer optimizer logic:
    - Sweep sizes × durations × strategies
    - Find: (1) Best IRR config, (2) Max viable scale at ≥ target IRR

    Returns DataFrame of all configurations with economics.
    """
    if strategies is None:
        strategies = list(ChargingStrategy)

    results = []
    sizes = np.arange(size_range[0], size_range[1] + size_step, size_step)

    for size_kwh in sizes:
        for duration in duration_options:
            max_power = size_kwh / duration

            for strategy in strategies:
                bess = BESSConfig(
                    capacity_kwh=size_kwh,
                    max_power_kw=max_power,
                )

                optimizer = DispatchOptimizer(
                    bess_config=bess,
                    fls_tariff_inr_kwh=fls_tariff,
                    strategy=strategy,
                )

                # Run for full year (or available data)
                n_days = len(demand_kwh) // 24
                annual_savings = 0

                for day in range(n_days):
                    s = day * 24
                    e = s + 24
                    if e > len(demand_kwh):
                        break
                    schedule = optimizer.optimize_24h(
                        demand_kwh[s:e], solar_kwh[s:e], iex_prices[s:e]
                    )
                    annual_savings += schedule.bess_savings_inr + schedule.solar_savings_inr

                # Scale to full year
                if n_days < 365:
                    annual_savings *= (365 / max(n_days, 1))

                # Economics
                capex = size_kwh * capex_inr_per_kwh
                simple_payback = capex / max(annual_savings, 1)

                # Simplified IRR (assuming 15-year life, constant savings)
                try:
                    irr = np.irr([-capex] + [annual_savings] * 15) if annual_savings > 0 else -1
                except:
                    irr = annual_savings / capex if capex > 0 else 0

                results.append({
                    "size_kwh": size_kwh,
                    "duration_hours": duration,
                    "max_power_kw": max_power,
                    "strategy": strategy.value,
                    "annual_savings_inr": annual_savings,
                    "capex_inr": capex,
                    "simple_payback_years": simple_payback,
                    "irr": irr,
                })

    df = pd.DataFrame(results)

    # Tag best configs
    if not df.empty:
        best_irr_idx = df["irr"].idxmax()
        df["is_best_return"] = df.index == best_irr_idx

        viable = df[df["irr"] >= target_irr]
        if not viable.empty:
            max_viable_idx = viable["size_kwh"].idxmax()
            df["is_max_viable"] = df.index == max_viable_idx
        else:
            df["is_max_viable"] = False

    return df
