"""
Typed edges for the EdgeGrid graph.

Edge types follow EDGEGRID_PRODUCT_SPEC.md §Part II "The graph" edge table.
Edges are directional and carry lightweight metadata for UI rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

EdgeType = Literal[
    "consumes",    # meter  → feeder         (energy flow, measured)
    "feeds",       # feeder → substation     (aggregated load)
    "serves",      # substation → C&I load   (contracted supply)
    "dispatches",  # bess   → substation     (scheduled charge/discharge)
    "observes",    # weather → local nodes   (forecast)
    "prices",      # iex    → dispatch       (₹/kWh block)
    "forecasts",   # predict() → meter       (timestamped forecast_wh)
    "explains",    # audit  → dispatch edge  (natural-language reason)
]


@dataclass
class Edge:
    source: str
    target: str
    edge_type: EdgeType
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def id(self) -> str:
        return f"{self.source}->{self.target}:{self.edge_type}"
