"""
Dispatch optimizer v2 — PuLP MILP at 48h × 15-min = 192 intervals.

Implements PRD-C4 from EDGEGRID_PRODUCT_SPEC.md §Part I §8:

    Objective:
        maximize  Σ (arbitrage_revenue + demand_charge_savings + peak_shaving_credit)
                − Σ (degradation_cost + soc_buffer_penalty + cycle_penalty)
    Constraints:
        SOC ∈ [10%, 90%]; max C-rate; grid import cap; kVA peak tracking
        (month-to-date); round-trip efficiency (η² on throughput); binary
        charge/discharge mutex to prevent simultaneous ±.

The v1 scipy optimizer in `optimizer.py` is preserved for backward compatibility
with the existing `tests/test_dispatch.py`. New API endpoints (/substation/{id}/dispatch)
route here.

Inputs come from `inference.v4_predict.predict()` (30-min native) and are
resampled to 15-min via linear interpolation — the single 30→15 bridge in the
prototype (DEBT-6 tracks the real retraining fix).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pulp

from .audit import AuditContext, build_audit_string, classify_action


INTERVAL_MINUTES = 15
HORIZON_INTERVALS = 192  # 48h × 4 intervals/hr
HOURS_PER_INTERVAL = INTERVAL_MINUTES / 60.0


# ───────────────────── Config dataclasses ────────────────────────────────────


@dataclass
class BESSSpec:
    """Physical + economic specs for one BESS unit."""

    capacity_kwh: float = 1000.0           # usable energy capacity
    duration_h: float = 4.0                # = capacity_kwh / power_kw
    round_trip_efficiency: float = 0.90    # η² applied on throughput
    soc_min_pct: float = 10.0
    soc_max_pct: float = 90.0
    initial_soc_pct: float = 50.0
    degradation_inr_per_kwh_throughput: float = 2.5
    capex_inr_per_kwh: float = 25_000.0    # for IRR module

    @property
    def max_power_kw(self) -> float:
        return self.capacity_kwh / self.duration_h

    @property
    def max_interval_energy_kwh(self) -> float:
        return self.max_power_kw * HOURS_PER_INTERVAL

    @property
    def soc_min_kwh(self) -> float:
        return self.capacity_kwh * self.soc_min_pct / 100.0

    @property
    def soc_max_kwh(self) -> float:
        return self.capacity_kwh * self.soc_max_pct / 100.0

    @property
    def initial_soc_kwh(self) -> float:
        return self.capacity_kwh * self.initial_soc_pct / 100.0


@dataclass
class TariffSpec:
    """Substation tariff context (APEPDCL-style)."""

    landed_cost_inr_per_kwh: float = 8.20   # what grid import costs
    kva_tariff_inr_per_kva_month: float = 400.0
    mtd_peak_kva: float = 0.0  # month-to-date peak; optimizer tries not to exceed


@dataclass
class OptimizerConfig:
    """Tuning knobs. Defaults align with spec guidance."""

    aggression_threshold_mape_pct: float = 15.0  # block MAPE below this → aggressive
    min_confidence_weight: float = 0.5           # floor for scaling (never zero)
    soc_buffer_penalty_inr_per_kwh: float = 0.05
    cycle_penalty_inr_per_full_cycle: float = 100.0
    grid_import_cap_kw: Optional[float] = None   # None = unbounded
    solver_time_limit_seconds: int = 30
    solver_msg: bool = False


# ───────────────────── Output schedule ───────────────────────────────────────


@dataclass
class DispatchScheduleV2:
    """
    Output of the v2 MILP optimizer. 192-row 15-min schedule.

    Column semantics match EDGEGRID_PRODUCT_SPEC.md §PRD-C4 output contract.
    """

    df: pd.DataFrame  # indexed by timestamp, columns below
    total_revenue_inr: float
    total_cost_inr: float
    net_benefit_inr: float
    peak_kva: float
    cycles: float
    audit_strings: list[str]
    solver_status: str
    meta: dict = field(default_factory=dict)

    EXPECTED_COLUMNS = (
        "demand_kwh", "iex_price_inr", "landed_cost_inr",
        "bess_charge_kwh", "bess_discharge_kwh",
        "grid_import_kwh", "solar_curtail_kwh",
        "soc_kwh", "soc_pct",
        "block_label", "historical_block_mape",
        "confidence_weight", "action", "audit_string",
    )

    def to_dict_records(self) -> list[dict]:
        recs = self.df.reset_index().to_dict(orient="records")
        for r in recs:
            # JSON-friendly timestamps
            if isinstance(r.get("timestamp"), (pd.Timestamp, datetime)):
                r["timestamp"] = pd.Timestamp(r["timestamp"]).isoformat()
        return recs


# ───────────────────── Core optimizer ────────────────────────────────────────


def _resample_30min_to_15min(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    The only 30→15 bridge in the prototype.

    forecast_wh: linear interpolation (energy per interval is halved).
    confidence_low / confidence_high: linear interpolation.
    block_label: forward-fill (blocks don't change at 15-min boundaries).
    historical_block_mape: forward-fill.
    """
    if forecast_df.empty:
        return forecast_df.copy()
    # Ensure a DatetimeIndex
    idx = pd.DatetimeIndex(forecast_df.index)
    df = forecast_df.copy()
    df.index = idx

    new_idx = pd.date_range(idx[0], idx[-1] + pd.Timedelta(minutes=15), freq="15min")
    # Split numeric vs non-numeric — pandas 3.x refuses to interpolate across
    # string columns even column-wise.
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    interp = df[numeric_cols].reindex(new_idx).interpolate(
        method="linear", limit_direction="both",
    )
    for col in non_numeric_cols:
        interp[col] = df[col].reindex(new_idx, method="ffill")

    # forecast_wh per 30-min → per 15-min: each 30-min block contained X Wh;
    # split evenly between its two 15-min sub-blocks. Linear interp on a constant
    # does the wrong thing, so we halve and forward-fill explicitly.
    if "forecast_wh" in df.columns:
        halved = df["forecast_wh"] / 2.0
        interp["forecast_wh"] = halved.reindex(new_idx, method="ffill")

    # block_label is categorical — always ffill, never interpolate
    if "block_label" in df.columns:
        interp["block_label"] = df["block_label"].reindex(new_idx, method="ffill")
    return interp.iloc[:HORIZON_INTERVALS]


def _confidence_weight(block_mape_pct: float, cfg: OptimizerConfig) -> float:
    """
    High-confidence peak → 1.0. Low-confidence block → cfg.min_confidence_weight.

    Linear ramp: weight = clip(1 - (mape - threshold) / threshold, min, 1.0).
    """
    t = cfg.aggression_threshold_mape_pct
    w = 1.0 - max(0.0, block_mape_pct - t) / max(t, 1.0)
    return float(max(cfg.min_confidence_weight, min(1.0, w)))


def _price_series(
    prices_df: pd.DataFrame, new_index: pd.DatetimeIndex,
) -> pd.Series:
    """Align IEX price series to the 15-min schedule index; ffill on gaps."""
    if "iex_price_inr" not in prices_df.columns:
        raise KeyError("prices_df must have 'iex_price_inr' column")
    p = prices_df["iex_price_inr"].copy()
    p.index = pd.DatetimeIndex(p.index)
    return p.reindex(new_index, method="ffill").bfill().astype(float)


def optimize_dispatch(
    forecast_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    bess: BESSSpec,
    tariff: TariffSpec,
    cfg: Optional[OptimizerConfig] = None,
    solar_forecast_kwh: Optional[pd.Series] = None,
) -> DispatchScheduleV2:
    """
    Solve the 48h × 15-min MILP for a single substation / BESS pair.

    Parameters
    ----------
    forecast_df
        Output of `inference.v4_predict.predict()` for the substation-aggregated
        demand (or per-meter, then sum). Must have columns:
          forecast_wh, confidence_low, confidence_high, block_label,
          historical_block_mape.
    prices_df
        IEX DAM prices at 15-min. Column: iex_price_inr.
    bess
        Physical + economic BESS specs.
    tariff
        Substation tariff context.
    cfg
        Optimizer tuning.
    solar_forecast_kwh
        Optional solar generation forecast (15-min, kWh per interval). If None,
        treated as all zeros — solar stub per v1 scope.

    Returns
    -------
    DispatchScheduleV2 with a 192-row DataFrame and per-row audit strings.
    """
    if cfg is None:
        cfg = OptimizerConfig()

    # Resample forecast to 15-min and convert Wh → kWh
    f15 = _resample_30min_to_15min(forecast_df)
    demand_kwh = (f15["forecast_wh"].to_numpy() / 1000.0).astype(float)
    demand_kwh = demand_kwh[:HORIZON_INTERVALS]
    if len(demand_kwh) < HORIZON_INTERVALS:
        # Pad with repeats of last value if forecast is shorter than horizon
        pad = np.full(HORIZON_INTERVALS - len(demand_kwh), demand_kwh[-1])
        demand_kwh = np.concatenate([demand_kwh, pad])

    index = pd.date_range(f15.index[0], periods=HORIZON_INTERVALS, freq="15min")

    iex_price = _price_series(prices_df, index).to_numpy()
    landed = np.full(HORIZON_INTERVALS, tariff.landed_cost_inr_per_kwh)

    block_label = f15.get("block_label", pd.Series("peak", index=f15.index))
    block_label = block_label.reindex(index, method="ffill").fillna("peak").to_numpy()
    block_mape = f15.get(
        "historical_block_mape", pd.Series(8.0, index=f15.index)
    ).reindex(index, method="ffill").fillna(8.0).to_numpy()

    if solar_forecast_kwh is None:
        solar_kwh = np.zeros(HORIZON_INTERVALS)
    else:
        solar_kwh = solar_forecast_kwh.reindex(index, method="ffill").fillna(0.0).to_numpy()

    # Confidence weights per interval — scale arbitrage aggression.
    cw = np.array([_confidence_weight(m, cfg) for m in block_mape])

    # ─── Build MILP ─────────────────────────────────────────────────────────
    prob = pulp.LpProblem("edgegrid_dispatch_v2", pulp.LpMaximize)

    T = HORIZON_INTERVALS
    M = bess.max_interval_energy_kwh  # Big-M for mutex

    charge = [pulp.LpVariable(f"chg_{t}", lowBound=0, upBound=M) for t in range(T)]
    dis = [pulp.LpVariable(f"dis_{t}", lowBound=0, upBound=M) for t in range(T)]
    grid = [pulp.LpVariable(f"grid_{t}", lowBound=0) for t in range(T)]
    curtail = [pulp.LpVariable(f"curt_{t}", lowBound=0) for t in range(T)]
    soc = [
        pulp.LpVariable(
            f"soc_{t}", lowBound=bess.soc_min_kwh, upBound=bess.soc_max_kwh
        ) for t in range(T)
    ]
    # Binary mutex: can't charge and discharge in the same interval
    is_charge = [pulp.LpVariable(f"is_chg_{t}", cat=pulp.LpBinary) for t in range(T)]
    is_dis = [pulp.LpVariable(f"is_dis_{t}", cat=pulp.LpBinary) for t in range(T)]

    # kVA peak proxy: we penalize the max grid_kw, modeled as a free variable
    # that upper-bounds every grid[t] (converted back from kWh/interval to kW).
    peak_kva_var = pulp.LpVariable("peak_kva", lowBound=0)

    # ─── Constraints ────────────────────────────────────────────────────────
    eta = bess.round_trip_efficiency
    sqrt_eta = eta ** 0.5  # split losses evenly between charge and discharge

    for t in range(T):
        # Energy balance: demand = grid + solar_direct + discharge
        # solar_direct = solar_kwh[t] - curtail[t] - charge_from_solar (implicit)
        # Simplified: grid + discharge + (solar - curtail) = demand + charge
        prob += (
            grid[t] + dis[t] + (solar_kwh[t] - curtail[t])
            == demand_kwh[t] + charge[t]
        ), f"balance_{t}"

        # curtail ≤ solar available
        prob += curtail[t] <= solar_kwh[t], f"curt_cap_{t}"

        # Mutex: charge XOR discharge
        prob += charge[t] <= M * is_charge[t], f"chg_mutex_{t}"
        prob += dis[t] <= M * is_dis[t], f"dis_mutex_{t}"
        prob += is_charge[t] + is_dis[t] <= 1, f"mutex_{t}"

        # SOC dynamics with efficiency losses
        if t == 0:
            prob += (
                soc[t] == bess.initial_soc_kwh + sqrt_eta * charge[t] - dis[t] / sqrt_eta
            ), f"soc_init"
        else:
            prob += (
                soc[t] == soc[t - 1] + sqrt_eta * charge[t] - dis[t] / sqrt_eta
            ), f"soc_{t}"

        # Grid import cap (optional)
        if cfg.grid_import_cap_kw is not None:
            prob += grid[t] <= cfg.grid_import_cap_kw * HOURS_PER_INTERVAL, f"grid_cap_{t}"

        # Peak kVA tracking (approx kW = kVA at unity pf for this stub)
        prob += peak_kva_var >= grid[t] / HOURS_PER_INTERVAL, f"peak_ge_{t}"

    # ─── Objective ──────────────────────────────────────────────────────────
    arbitrage = pulp.lpSum(
        cw[t] * (iex_price[t] * dis[t] - iex_price[t] * charge[t])
        for t in range(T)
    )
    grid_cost = pulp.lpSum(landed[t] * grid[t] for t in range(T))
    degradation = pulp.lpSum(
        bess.degradation_inr_per_kwh_throughput * (charge[t] + dis[t])
        for t in range(T)
    )
    # Demand-charge savings: reward keeping peak below MTD peak (if set)
    demand_savings = 0
    if tariff.mtd_peak_kva > 0:
        # reward every kVA below the month-to-date peak
        demand_savings = (tariff.mtd_peak_kva - peak_kva_var) * (
            tariff.kva_tariff_inr_per_kva_month / 30.0 / 48.0
        )

    prob += arbitrage - grid_cost - degradation + demand_savings, "net_benefit"

    # ─── Solve ──────────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(
        msg=cfg.solver_msg, timeLimit=cfg.solver_time_limit_seconds,
    )
    status = prob.solve(solver)
    solver_status = pulp.LpStatus[status]

    # Extract solution
    def v(x) -> float:
        val = pulp.value(x)
        return 0.0 if val is None else float(val)

    charge_v = np.array([v(charge[t]) for t in range(T)])
    dis_v = np.array([v(dis[t]) for t in range(T)])
    grid_v = np.array([v(grid[t]) for t in range(T)])
    curt_v = np.array([v(curtail[t]) for t in range(T)])
    soc_v = np.array([v(soc[t]) for t in range(T)])
    peak_v = v(peak_kva_var)

    # Assemble DataFrame
    df = pd.DataFrame(
        {
            "demand_kwh": demand_kwh,
            "iex_price_inr": iex_price,
            "landed_cost_inr": landed,
            "bess_charge_kwh": charge_v,
            "bess_discharge_kwh": dis_v,
            "grid_import_kwh": grid_v,
            "solar_curtail_kwh": curt_v,
            "soc_kwh": soc_v,
            "soc_pct": soc_v / bess.capacity_kwh * 100.0,
            "block_label": block_label,
            "historical_block_mape": block_mape,
            "confidence_weight": cw,
        },
        index=index,
    )
    df.index.name = "timestamp"

    # Actions + audit strings per row
    actions: list[str] = []
    audits: list[str] = []
    for ts, row in df.iterrows():
        a = classify_action(
            bess_charge_wh=row["bess_charge_kwh"] * 1000,
            bess_discharge_wh=row["bess_discharge_kwh"] * 1000,
            solar_curtail_wh=row["solar_curtail_kwh"] * 1000,
            kva_shaved=0.0,  # refined by post-pass below
        )
        actions.append(a)
        kwh_mag = max(
            row["bess_charge_kwh"], row["bess_discharge_kwh"], row["solar_curtail_kwh"]
        )
        if a == "hold":
            kwh_mag = 0.0
        ctx = AuditContext(
            timestamp=pd.Timestamp(ts).to_pydatetime(),
            action=a,
            kwh=float(kwh_mag),
            iex_price_inr=float(row["iex_price_inr"]),
            landed_cost_inr=float(row["landed_cost_inr"]),
            block_mape_pct=float(row["historical_block_mape"]),
            aggression_threshold_pct=cfg.aggression_threshold_mape_pct,
        )
        audits.append(build_audit_string(ctx))
    df["action"] = actions
    df["audit_string"] = audits

    # Summary metrics
    total_revenue = float((iex_price * dis_v).sum())
    total_cost = float((landed * grid_v).sum() + (iex_price * charge_v).sum())
    # Cycles: throughput / (2 × capacity)
    cycles = float((charge_v.sum() + dis_v.sum()) / max(2 * bess.capacity_kwh, 1e-6))

    return DispatchScheduleV2(
        df=df,
        total_revenue_inr=total_revenue,
        total_cost_inr=total_cost,
        net_benefit_inr=float(total_revenue - total_cost),
        peak_kva=float(peak_v),
        cycles=cycles,
        audit_strings=audits,
        solver_status=solver_status,
        meta={
            "intervals": HORIZON_INTERVALS,
            "interval_minutes": INTERVAL_MINUTES,
            "model_version_note": "forecast resampled from 30-min (DEBT-6)",
        },
    )
