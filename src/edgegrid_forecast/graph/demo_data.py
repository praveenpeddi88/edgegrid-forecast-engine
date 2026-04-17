"""
Deterministic demo network synthesis for the Dispatch Console prototype.

Loads the 42 real meters from `models/v4/_manifest.json` and distributes them
across 3 APEPDCL substations in Visakhapatnam Zone 1, each with:
- 5-15 meters (HT + Medium + Small mix for the canonical substation)
- 1-2 BESS stubs (500 kWh × 4h or 1 MWh × 4h)
- 1 solar stub
- 1 weather pseudo-node (shared across adjacent substations)
- Feeder layer grouping meters before rollup to the substation

Synthesis is deterministic — seeded by MSN position in the manifest — so the
graph renders the same every run. This lets the frontend demo be reproducible.

Meets spec acceptance criterion #1: ≥3 substations, each with 5-15 meters,
1-2 BESS stubs.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from .edges import Edge
from .network import EdgeGridNetwork
from .nodes import (
    BESSUnit,
    CILoad,
    Feeder,
    IEXPriceCurve,
    Meter,
    SolarUnit,
    Substation,
    WeatherStation,
)


# Canonical substation picked for the end-to-end demo story (spec criterion #7:
# "one canonical demo substation with HT + Medium + Small mix").
CANONICAL_SUBSTATION_ID = "ss-vskp-01"


# Three APEPDCL substations in and around Visakhapatnam. Real zone, synthetic
# substation codes — real deployments would pull these from APEPDCL GIS.
SUBSTATION_SEEDS = [
    {
        "id": "ss-vskp-01",
        "label": "Madhurawada 33/11kV",
        "code": "VSKP-MW-01",
        "zone": "Visakhapatnam Zone 1",
        "lat": 17.7908,
        "lon": 83.3768,
        "landed_cost": 8.2,
        "bess": [("bess-vskp-01a", 1.0, 4), ("bess-vskp-01b", 0.5, 4)],
        "solar_kwp": 750,
        "contracted_kva": 2400,
    },
    {
        "id": "ss-vskp-02",
        "label": "Gajuwaka 33/11kV",
        "code": "VSKP-GJ-02",
        "zone": "Visakhapatnam Zone 2",
        "lat": 17.6868,
        "lon": 83.2093,
        "landed_cost": 8.45,
        "bess": [("bess-vskp-02a", 1.0, 4)],
        "solar_kwp": 500,
        "contracted_kva": 1800,
    },
    {
        "id": "ss-vskp-03",
        "label": "Pendurthi 33/11kV",
        "code": "VSKP-PD-03",
        "zone": "Visakhapatnam Zone 3",
        "lat": 17.7923,
        "lon": 83.1870,
        "landed_cost": 8.05,
        "bess": [("bess-vskp-03a", 0.5, 4)],
        "solar_kwp": 400,
        "contracted_kva": 1500,
    },
]


def _load_manifest(repo_root: Path) -> list[dict]:
    manifest_path = repo_root / "models" / "v4" / "_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest not found at {manifest_path}. Run scripts/train_all_v4.py first."
        )
    data = json.loads(manifest_path.read_text())
    return data["models"]


def _distribute_meters_across_substations(
    meters: list[dict],
) -> dict[str, list[dict]]:
    """
    Deterministic round-robin distribution biased so the CANONICAL substation
    receives the richest tier mix (HT + Medium + Small), since the demo story
    rides on it. Other substations get the remainder.

    Returns {substation_id: [manifest_entry, ...]}.
    """
    # Split by tier so we can guarantee a mix for the canonical substation.
    by_tier: dict[str, list[dict]] = {"HT (>5kWh)": [], "Medium": [], "Small (<500Wh)": []}
    for m in meters:
        t = m["tier"]
        # Manifest sometimes emits "HT" without the qualifier; normalize.
        if t.startswith("HT"):
            by_tier["HT (>5kWh)"].append(m)
        elif t.startswith("Small"):
            by_tier["Small (<500Wh)"].append(m)
        else:
            by_tier["Medium"].append(m)

    rng = random.Random(42)  # deterministic
    for tier_list in by_tier.values():
        rng.shuffle(tier_list)

    out: dict[str, list[dict]] = {s["id"]: [] for s in SUBSTATION_SEEDS}
    canonical = CANONICAL_SUBSTATION_ID

    # Each substation caps at CAP meters (spec acceptance criterion #1:
    # 5-15 meters). Distribute the first `CAP_PER_SUB * n_subs` meters and
    # stop — any surplus beyond that stays unassigned (logged, not rendered).
    CAP_PER_SUB = 14

    # Canonical gets: 5 HT + 4 Medium + 2 Small = 11 (HT + Medium + Small mix
    # for the end-to-end demo story).
    targets = [
        (canonical, {"HT (>5kWh)": 5, "Medium": 4, "Small (<500Wh)": 2}),
        ("ss-vskp-02", {"HT (>5kWh)": 4, "Medium": 5, "Small (<500Wh)": 2}),
        ("ss-vskp-03", {"HT (>5kWh)": 3, "Medium": 4, "Small (<500Wh)": 2}),
    ]
    for sid, plan in targets:
        for tier, n in plan.items():
            take = by_tier[tier][:n]
            by_tier[tier] = by_tier[tier][n:]
            out[sid].extend(take)

    # Spill remaining meters round-robin across all three substations, but
    # refuse to push any substation past CAP_PER_SUB.
    remaining: list[dict] = []
    for tier_list in by_tier.values():
        remaining.extend(tier_list)
    subs = [s["id"] for s in SUBSTATION_SEEDS]
    for m in remaining:
        placed = False
        for sid in subs:
            if len(out[sid]) < CAP_PER_SUB:
                out[sid].append(m)
                placed = True
                break
        if not placed:
            break  # all full; the rest stay out of the demo network

    return out


def build_demo_network(
    repo_root: Optional[Path] = None,
    ci_loads_per_substation: int = 2,
) -> EdgeGridNetwork:
    """
    Build the full demo graph. Deterministic given the manifest and the seeds.

    Raises FileNotFoundError if the manifest is missing.
    """
    if repo_root is None:
        # Walk up from this file: src/edgegrid_forecast/graph/demo_data.py → repo root
        repo_root = Path(__file__).resolve().parents[3]
    manifest_models = _load_manifest(Path(repo_root))
    distribution = _distribute_meters_across_substations(manifest_models)

    net = EdgeGridNetwork()

    # Global pseudo-nodes: weather + IEX price curve.
    weather = WeatherStation(
        id="weather-vskp",
        label="Visakhapatnam weather cell",
        kind="weather",
        lat=17.72,
        lon=83.30,
        cell_resolution_km=10.0,
    )
    iex = IEXPriceCurve(id="iex-dam", label="IEX DAM (15-min)", kind="iex")
    net.add_node(weather)
    net.add_node(iex)

    rng = random.Random(7)
    for seed in SUBSTATION_SEEDS:
        sid = seed["id"]
        ss = Substation(
            id=sid,
            label=seed["label"],
            kind="substation",
            substation_code=seed["code"],
            zone=seed["zone"],
            landed_cost_inr_per_kwh=seed["landed_cost"],
            lat=seed["lat"],
            lon=seed["lon"],
            total_contracted_kva=seed["contracted_kva"],
        )
        net.add_node(ss)
        net.add_edge(Edge(source=iex.id, target=ss.id, edge_type="prices"))
        net.add_edge(Edge(source=weather.id, target=ss.id, edge_type="observes"))

        # BESS units
        for (bid, mwh, dur) in seed["bess"]:
            bess = BESSUnit(
                id=bid,
                label=f"BESS {mwh} MWh / {dur}h",
                kind="bess",
                capacity_mwh=mwh,
                duration_h=dur,
                c_rate=1.0 / dur,
                substation_id=sid,
                soc_pct=50.0,
            )
            net.add_node(bess)
            net.add_edge(Edge(source=bid, target=sid, edge_type="dispatches"))

        # Solar stub
        solar = SolarUnit(
            id=f"solar-{sid}",
            label=f"Solar {seed['solar_kwp']} kWp",
            kind="solar",
            kwp=float(seed["solar_kwp"]),
            substation_id=sid,
        )
        net.add_node(solar)
        net.add_edge(Edge(source=solar.id, target=sid, edge_type="feeds"))

        # C&I loads (synthetic, typical mix)
        for i in range(ci_loads_per_substation):
            ci = CILoad(
                id=f"ci-{sid}-{i+1}",
                label=f"C&I {seed['code']}-L{i+1}",
                kind="ci_load",
                contracted_kva=float(rng.choice([250, 500, 750, 1000])),
                tod_pattern=rng.choice(["office-hours", "manufacturing", "24x7"]),
                substation_id=sid,
            )
            net.add_node(ci)
            net.add_edge(Edge(source=sid, target=ci.id, edge_type="serves"))

        # Feeders (one feeder per ~5 meters)
        meter_entries = distribution[sid]
        n_feeders = max(1, (len(meter_entries) + 4) // 5)
        feeders: list[Feeder] = []
        for fi in range(n_feeders):
            f = Feeder(
                id=f"feeder-{sid}-{fi+1}",
                label=f"Feeder {seed['code']}-F{fi+1}",
                kind="feeder",
                feeder_code=f"{seed['code']}-F{fi+1}",
                substation_id=sid,
            )
            net.add_node(f)
            net.add_edge(Edge(source=f.id, target=sid, edge_type="feeds"))
            feeders.append(f)

        # Meters
        for idx, m in enumerate(meter_entries):
            feeder = feeders[idx % n_feeders]
            meter = Meter(
                id=f"meter-{m['msn']}",
                label=f"MSN {m['msn']}",
                kind="meter",
                msn=m["msn"],
                tier=m["tier"],
                phase=m.get("phase", "3PH"),
                holdout_mape=float(m["holdout_mape"]),
                historical_block_mape=dict(m.get("historical_block_mape", {})),
                n_features_selected=int(m.get("n_features_selected", 0)),
                feeder_id=feeder.id,
                substation_id=sid,
                # Jitter coords around substation for pleasing graph layout.
                lat=seed["lat"] + rng.uniform(-0.015, 0.015),
                lon=seed["lon"] + rng.uniform(-0.015, 0.015),
            )
            net.add_node(meter)
            net.add_edge(Edge(source=meter.id, target=feeder.id, edge_type="consumes"))
            net.add_edge(Edge(source=weather.id, target=meter.id, edge_type="observes"))

    return net
