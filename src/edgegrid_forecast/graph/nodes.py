"""
Typed node dataclasses for the EdgeGrid network.

Attributes align with EDGEGRID_PRODUCT_SPEC.md §Part II "The graph" node table.
Every node has a stable string id, a human-readable label, and a kind tag
(used by the frontend to route to the right drilldown screen).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional

NodeKind = Literal[
    "meter",
    "feeder",
    "substation",
    "bess",
    "solar",
    "ci_load",
    "weather",
    "iex",
]


@dataclass
class Node:
    """Base node. Concrete types extend this with typed attributes."""

    id: str
    label: str
    kind: NodeKind

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Meter(Node):
    """
    A single AMI meter. Sources of truth:
        - tier / phase / holdout_mape / historical_block_mape: models/v4/_manifest.json
        - geocoords / feeder_id: assigned in demo_data (synthetic for prototype).
    """

    msn: str = ""
    tier: str = "Medium"  # "HT (>5kWh)" | "Medium" | "Small (<500Wh)"
    phase: str = "3PH"
    mean_demand_wh: float = 0.0
    holdout_mape: float = 0.0
    historical_block_mape: dict[str, float] = field(default_factory=dict)
    n_features_selected: int = 0
    feeder_id: Optional[str] = None
    substation_id: Optional[str] = None
    lat: float = 17.7231  # Visakhapatnam default
    lon: float = 83.3010
    kind: NodeKind = "meter"

    @property
    def health(self) -> str:
        """Health bucket for graph coloring: green <8%, amber 8-12%, red >12%."""
        m = self.holdout_mape
        if m < 8:
            return "green"
        if m < 12:
            return "amber"
        return "red"


@dataclass
class Feeder(Node):
    """11kV LT feeder aggregating multiple meters."""

    feeder_code: str = ""
    capacity_kw: float = 500.0
    loss_factor: float = 0.04
    substation_id: Optional[str] = None
    kind: NodeKind = "feeder"


@dataclass
class Substation(Node):
    """33/11kV APEPDCL substation — the commercial dispatch unit."""

    substation_code: str = ""
    zone: str = "Visakhapatnam Zone 1"
    landed_cost_inr_per_kwh: float = 8.2
    tariff_regime: str = "APEPDCL-HT-2024"
    lat: float = 17.7231
    lon: float = 83.3010
    total_contracted_kva: float = 2000.0
    kind: NodeKind = "substation"


@dataclass
class BESSUnit(Node):
    """Battery energy storage system. Specs drive the MILP optimizer."""

    capacity_mwh: float = 1.0
    duration_h: int = 4
    c_rate: float = 0.5  # = 1 / duration_h, but explicit for overrides
    round_trip_efficiency: float = 0.90
    soc_pct: float = 50.0
    cycle_count: int = 0
    degradation_pct: float = 0.0
    capex_inr_per_kwh: float = 25_000.0
    substation_id: Optional[str] = None
    kind: NodeKind = "bess"

    @property
    def max_power_kw(self) -> float:
        return self.capacity_mwh * 1000 * self.c_rate


@dataclass
class SolarUnit(Node):
    """Rooftop or ground solar. Stub for v1 — pvlib integration in later pass."""

    kwp: float = 500.0
    tilt_deg: float = 17.0
    azimuth_deg: float = 180.0
    soiling_pct: float = 2.0
    substation_id: Optional[str] = None
    kind: NodeKind = "solar"


@dataclass
class CILoad(Node):
    """Commercial & Industrial offtaker. Used by FLS quote generator."""

    contracted_kva: float = 500.0
    tod_pattern: str = "office-hours"  # "office-hours" | "manufacturing" | "24x7"
    ppa_inr_per_kwh: Optional[float] = None
    substation_id: Optional[str] = None
    kind: NodeKind = "ci_load"


@dataclass
class WeatherStation(Node):
    """Open-Meteo grid cell pseudo-node. Edges connect it to local meters."""

    lat: float = 17.7231
    lon: float = 83.3010
    cell_resolution_km: float = 10.0
    kind: NodeKind = "weather"


@dataclass
class IEXPriceCurve(Node):
    """Pseudo-node for the IEX DAM price curve (single global node for v1)."""

    market: str = "IEX-DAM"
    block_minutes: int = 15
    kind: NodeKind = "iex"
