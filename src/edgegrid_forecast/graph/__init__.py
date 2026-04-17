"""
EdgeGrid network graph — first-class abstraction for substations, meters, BESS.

See EDGEGRID_PRODUCT_SPEC.md §Part II "Network Effects as Architectural Moat".
The prototype treats the graph as the home screen, not a rendering afterthought.
"""

from .nodes import (
    Node,
    Meter,
    Feeder,
    Substation,
    BESSUnit,
    SolarUnit,
    CILoad,
    WeatherStation,
    IEXPriceCurve,
)
from .edges import Edge, EdgeType
from .network import EdgeGridNetwork
from .demo_data import build_demo_network, CANONICAL_SUBSTATION_ID

__all__ = [
    "Node",
    "Meter",
    "Feeder",
    "Substation",
    "BESSUnit",
    "SolarUnit",
    "CILoad",
    "WeatherStation",
    "IEXPriceCurve",
    "Edge",
    "EdgeType",
    "EdgeGridNetwork",
    "build_demo_network",
    "CANONICAL_SUBSTATION_ID",
]
