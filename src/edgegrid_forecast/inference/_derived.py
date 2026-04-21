"""
Derived CEA-aligned load metrics (W4).

The v5 engine predicts half-hourly **energy** in Wh. CEA and SERC filings,
and most APEPDCL internal planning decks, consume **demand** (kW) and
**load factor** (ratio). This module converts a half-hourly Wh forecast
into the three commercially salient metrics:

- peak_kw       : max instantaneous demand across the horizon
- average_kw    : mean demand across the horizon
- load_factor   : average_kw / peak_kw  (0–1; higher = flatter, more revenue-efficient)
- diversity_factor (fleet only):
                  sum(individual peaks) / coincident fleet peak
                  (always ≥ 1; higher = less coincident, lower feeder stress)

Conversions
-----------
A 30-min interval of E Wh corresponds to an average power of
    P_kW = E_Wh * 2 / 1000
(E Wh delivered in 0.5 h → P = E/0.5 Wh/h = 2E Wh/h = 2E/1000 kW.)

We treat that as the demand for the interval — matching how ABT and MDI
meters report half-hourly demand.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# 2 × Wh / 1000 = kW for a 30-min slot
WH_PER_SLOT_TO_KW = 2.0 / 1000.0


@dataclass
class DerivedLoadMetrics:
    peak_kw: float
    peak_ts: str
    average_kw: float
    load_factor: float
    total_energy_kwh: float
    horizon_hours: float

    def as_dict(self) -> dict:
        return {
            "peak_kw": round(self.peak_kw, 3),
            "peak_ts": self.peak_ts,
            "average_kw": round(self.average_kw, 3),
            "load_factor": round(self.load_factor, 4),
            "total_energy_kwh": round(self.total_energy_kwh, 3),
            "horizon_hours": round(self.horizon_hours, 2),
        }


def wh_to_kw(wh_series: pd.Series) -> pd.Series:
    """Convert half-hourly Wh to instantaneous kW (average over the slot).

    Assumes 30-min slots. If the caller gives a different cadence, scale
    the multiplier accordingly before calling.
    """
    return wh_series.astype(float) * WH_PER_SLOT_TO_KW


def peak_kw(wh_series: pd.Series, index: Optional[pd.DatetimeIndex] = None) -> tuple[float, str]:
    """Return (peak_kw, peak_ts_iso).

    Uses the max of the Wh→kW converted series. If `index` is provided (or
    `wh_series` has a DatetimeIndex), returns the ISO timestamp of the peak.
    Falls back to empty string when timestamps are unavailable.
    """
    if len(wh_series) == 0:
        return 0.0, ""
    kw = wh_to_kw(wh_series)
    peak = float(kw.max())
    ts_idx = index if index is not None else (
        kw.index if isinstance(kw.index, pd.DatetimeIndex) else None
    )
    if ts_idx is None:
        return peak, ""
    i = int(np.argmax(kw.values))
    return peak, pd.Timestamp(ts_idx[i]).isoformat()


def load_factor(wh_series: pd.Series) -> float:
    """Load factor = average_kw / peak_kw ∈ (0, 1].

    Returns 0.0 when input is empty or peak is zero (all-off meter) — so the
    caller never has to guard against divide-by-zero.
    """
    if len(wh_series) == 0:
        return 0.0
    kw = wh_to_kw(wh_series)
    peak = float(kw.max())
    if peak <= 0:
        return 0.0
    return float(kw.mean() / peak)


def derive(
    wh_series: pd.Series,
    index: Optional[pd.DatetimeIndex] = None,
) -> DerivedLoadMetrics:
    """Compute peak/average/LF/total-energy in one pass — the usual entry point."""
    if len(wh_series) == 0:
        return DerivedLoadMetrics(0.0, "", 0.0, 0.0, 0.0, 0.0)
    kw = wh_to_kw(wh_series)
    peak = float(kw.max())
    avg = float(kw.mean())
    lf = (avg / peak) if peak > 0 else 0.0
    total_kwh = float(wh_series.astype(float).sum() / 1000.0)
    hours = len(wh_series) * 0.5
    # peak_ts
    ts_idx = index if index is not None else (
        kw.index if isinstance(kw.index, pd.DatetimeIndex) else None
    )
    peak_ts = ""
    if ts_idx is not None and peak > 0:
        i = int(np.argmax(kw.values))
        peak_ts = pd.Timestamp(ts_idx[i]).isoformat()
    return DerivedLoadMetrics(
        peak_kw=peak,
        peak_ts=peak_ts,
        average_kw=avg,
        load_factor=lf,
        total_energy_kwh=total_kwh,
        horizon_hours=hours,
    )


def diversity_factor(per_meter_wh: dict[str, pd.Series]) -> float:
    """Diversity factor = Σ individual peaks / coincident fleet peak (≥ 1).

    A DF of 1.0 means all meters peak at the same moment (worst case for
    transformer loadability). A DF of 1.5 means the fleet's coincident peak
    is 33% below the non-coincident sum — the operational headroom APEPDCL
    can count on at the LT/HT feeder.
    """
    if not per_meter_wh:
        return 0.0
    # Align onto a common DateTimeIndex, pad zeros for gaps
    series = {m: s.astype(float) for m, s in per_meter_wh.items() if len(s) > 0}
    if not series:
        return 0.0
    idx = sorted(set().union(*(s.index for s in series.values())))
    kw_mat = pd.DataFrame({m: s.reindex(idx).fillna(0.0) * WH_PER_SLOT_TO_KW
                           for m, s in series.items()})
    sum_individual = float(kw_mat.max(axis=0).sum())
    coincident = float(kw_mat.sum(axis=1).max())
    if coincident <= 0:
        return 0.0
    return sum_individual / coincident
