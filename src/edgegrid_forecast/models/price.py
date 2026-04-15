"""
IEX Day-Ahead Market price forecasting and landed cost computation.

The IEX DAM price determines whether it's cheaper to:
- Buy from the grid (at FLS tariff)
- Buy from IEX (at DAM price + network charges)
- Discharge BESS (stored energy, zero marginal cost)

Price forecasting enables:
1. Optimal BESS charging schedule (charge when cheap, discharge when expensive)
2. Open access procurement decisions
3. Dynamic pricing within energy clusters
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..utils.constants import (
    IEX_DAM_PRICES_FY2425,
    fy_month_index,
    get_iex_price,
    get_landed_price,
    landed_cost_from_iex,
)


class IEXPriceForecaster:
    """
    IEX DAM price forecaster using historical patterns + ML.

    IEX prices have strong structure:
    - Time-of-day pattern (peak 10am-2pm and 6pm-10pm)
    - Seasonal pattern (summer > monsoon > winter)
    - Day-of-week (lower on weekends)
    - Temperature correlation (cooling load drives demand)

    For v1, we use the historical monthly×hourly average matrix as the baseline
    and train an ML model to capture deviations.
    """

    def __init__(self):
        self.model = None
        self.historical_matrix = IEX_DAM_PRICES_FY2425

    def get_baseline_price(self, month: int, hour: int) -> float:
        """Get historical average IEX price for a month-hour combination."""
        return get_iex_price(month, hour)

    def get_baseline_landed(self, month: int, hour: int) -> float:
        """Get historical average landed cost for a month-hour combination."""
        return get_landed_price(month, hour)

    def forecast_prices(
        self,
        timestamps: pd.DatetimeIndex,
        temperature: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Forecast IEX prices and landed costs for given timestamps.

        Returns DataFrame with columns:
            timestamp, iex_price, landed_cost, price_percentile
        """
        months = timestamps.month
        hours = timestamps.hour

        iex_prices = np.array([
            self.get_baseline_price(int(m), int(h))
            for m, h in zip(months, hours)
        ])

        landed_costs = np.array([
            self.get_baseline_landed(int(m), int(h))
            for m, h in zip(months, hours)
        ])

        # Temperature adjustment (optional)
        if temperature is not None:
            # High temperatures drive up demand → higher prices
            # +1°C above 35°C ≈ +2% price increase
            temp_premium = np.where(
                temperature > 35,
                0.02 * (temperature - 35) * iex_prices,
                0,
            )
            iex_prices = iex_prices + temp_premium
            landed_costs = np.array([landed_cost_from_iex(p) for p in iex_prices])

        # Price percentile (within its month)
        price_pctile = np.zeros(len(iex_prices))
        for m in range(1, 13):
            month_mask = months == m
            if month_mask.any():
                month_prices = iex_prices[month_mask]
                for i, idx in enumerate(np.where(month_mask)[0]):
                    price_pctile[idx] = (month_prices < month_prices[i]).mean()

        return pd.DataFrame({
            "timestamp": timestamps,
            "iex_price_inr_kwh": iex_prices,
            "landed_cost_inr_kwh": landed_costs,
            "price_percentile": price_pctile,
        })

    def find_cheapest_hours(
        self,
        month: int,
        n_hours: int = 4,
    ) -> List[int]:
        """Find the cheapest N hours in a given month for BESS charging."""
        fy_idx = fy_month_index(month)
        hourly_prices = self.historical_matrix[fy_idx]
        sorted_hours = np.argsort(hourly_prices)
        return sorted_hours[:n_hours].tolist()

    def find_expensive_hours(
        self,
        month: int,
        n_hours: int = 4,
    ) -> List[int]:
        """Find the most expensive N hours for BESS discharge."""
        fy_idx = fy_month_index(month)
        hourly_prices = self.historical_matrix[fy_idx]
        sorted_hours = np.argsort(hourly_prices)[::-1]
        return sorted_hours[:n_hours].tolist()

    def compute_spread(
        self,
        month: int,
        charge_hours: int = 4,
        discharge_hours: int = 4,
    ) -> Dict[str, float]:
        """
        Compute the charge-discharge price spread for a month.
        This is the gross margin opportunity for BESS arbitrage.
        """
        cheapest = self.find_cheapest_hours(month, charge_hours)
        expensive = self.find_expensive_hours(month, discharge_hours)

        fy_idx = fy_month_index(month)
        avg_charge_price = np.mean([self.historical_matrix[fy_idx][h] for h in cheapest])
        avg_discharge_price = np.mean([self.historical_matrix[fy_idx][h] for h in expensive])

        avg_charge_landed = np.mean([landed_cost_from_iex(self.historical_matrix[fy_idx][h]) for h in cheapest])
        avg_discharge_landed = np.mean([landed_cost_from_iex(self.historical_matrix[fy_idx][h]) for h in expensive])

        return {
            "month": month,
            "avg_charge_iex": avg_charge_price,
            "avg_discharge_iex": avg_discharge_price,
            "iex_spread": avg_discharge_price - avg_charge_price,
            "avg_charge_landed": avg_charge_landed,
            "avg_discharge_landed": avg_discharge_landed,
            "landed_spread": avg_discharge_landed - avg_charge_landed,
            "cheapest_hours": cheapest,
            "expensive_hours": expensive,
        }

    def annual_spread_summary(self) -> pd.DataFrame:
        """Summary of monthly arbitrage opportunity."""
        records = [self.compute_spread(m) for m in range(1, 13)]
        df = pd.DataFrame(records)
        df["month_name"] = pd.to_datetime(df["month"], format="%m").dt.strftime("%b")
        return df
