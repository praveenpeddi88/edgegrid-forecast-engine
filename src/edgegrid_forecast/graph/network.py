"""
EdgeGridNetwork — in-memory graph container with traversal + JSON export.

Graph-first, per EDGEGRID_PRODUCT_SPEC.md §Part II:
- Entity-centric data model (nodes + edges, not one-big-table).
- `neighborhood(node_id, radius)` for drilldowns.
- `to_json()` is the wire format consumed by `/network` and the frontend D3 graph.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from .edges import Edge, EdgeType
from .nodes import Node, Meter, Substation, BESSUnit


@dataclass
class EdgeGridNetwork:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    _adj: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    # ─── mutation ────────────────────────────────────────────────────────────

    def add_node(self, node: Node) -> None:
        if node.id in self.nodes:
            raise ValueError(f"duplicate node id: {node.id}")
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        if edge.source not in self.nodes:
            raise KeyError(f"edge source not in graph: {edge.source}")
        if edge.target not in self.nodes:
            raise KeyError(f"edge target not in graph: {edge.target}")
        self.edges.append(edge)
        self._adj[edge.source].append(edge.target)
        self._adj[edge.target].append(edge.source)  # undirected traversal

    # ─── query ───────────────────────────────────────────────────────────────

    def node(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def nodes_by_kind(self, kind: str) -> Iterator[Node]:
        return (n for n in self.nodes.values() if n.kind == kind)

    def edges_of(self, node_id: str) -> Iterator[Edge]:
        return (e for e in self.edges if e.source == node_id or e.target == node_id)

    def edges_by_type(self, edge_type: EdgeType) -> Iterator[Edge]:
        return (e for e in self.edges if e.edge_type == edge_type)

    def neighborhood(self, node_id: str, radius: int = 1) -> set[str]:
        """BFS from node_id up to `radius` hops. Returns the set of node ids."""
        if node_id not in self.nodes:
            raise KeyError(node_id)
        visited = {node_id}
        frontier = deque([(node_id, 0)])
        while frontier:
            nid, depth = frontier.popleft()
            if depth >= radius:
                continue
            for nb in self._adj[nid]:
                if nb not in visited:
                    visited.add(nb)
                    frontier.append((nb, depth + 1))
        return visited

    def meters_of_substation(self, substation_id: str) -> list[Meter]:
        """Return all meters whose `substation_id` == this substation."""
        return [
            n for n in self.nodes.values()
            if isinstance(n, Meter) and n.substation_id == substation_id
        ]

    def bess_of_substation(self, substation_id: str) -> list[BESSUnit]:
        return [
            n for n in self.nodes.values()
            if isinstance(n, BESSUnit) and n.substation_id == substation_id
        ]

    # ─── serialization ───────────────────────────────────────────────────────

    def to_json(self) -> dict[str, Any]:
        """Wire format for /network endpoint and frontend D3."""
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "meta": {
                "n_nodes": len(self.nodes),
                "n_edges": len(self.edges),
                "kinds": sorted({n.kind for n in self.nodes.values()}),
            },
        }

    def dump_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=2, default=str))

    # ─── constructors ────────────────────────────────────────────────────────

    @classmethod
    def from_nodes_edges(
        cls, nodes: Iterable[Node], edges: Iterable[Edge]
    ) -> EdgeGridNetwork:
        net = cls()
        for n in nodes:
            net.add_node(n)
        for e in edges:
            net.add_edge(e)
        return net
