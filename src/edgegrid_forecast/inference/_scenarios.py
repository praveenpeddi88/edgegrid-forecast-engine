"""
CEA three-scenario framework (W2).

The v5 forecast engine trains three LightGBM models per meter — a mean objective
and two quantile objectives (q10, q90). The CEA 19th Electric Power Survey (EPS)
guidelines, and most DISCOM planning decks APEPDCL submits to the state load
dispatch centre, speak in terms of **three commercial scenarios**:

- Pessimistic — conservative upside / downside-risk planning view
- BAU        — central / business-as-usual forecast
- Optimistic — upside-risk / tight-supply planning view

This module is the single source of truth for:
1. How our statistical quantiles map to those commercial labels.
2. The narrative drivers each scenario assumes (weather, industrial load,
   EV/solar trajectory) so commercial + regulatory users can *explain* the
   number, not just receive it.

The mapping is intentionally simple and explicit so downstream dashboards
(pricing desks, procurement, regulatory filings) can consume either label
without having to know the underlying ML construction.
"""
from __future__ import annotations

from typing import Literal

ScenarioKey = Literal["pessimistic", "bau", "optimistic"]

# Statistical → commercial label map.
# q10 => pessimistic energy draw (low load), q90 => optimistic (high load).
# NOTE: "optimistic" here means "high energy delivered" from a DISCOM revenue
# lens; from a consumer cost lens it would flip. We follow DISCOM convention.
SCENARIO_TO_QUANTILE: dict[ScenarioKey, str] = {
    "pessimistic": "q10",
    "bau": "mean",
    "optimistic": "q90",
}

QUANTILE_TO_SCENARIO: dict[str, ScenarioKey] = {
    v: k for k, v in SCENARIO_TO_QUANTILE.items()
}


# Narrative drivers — what the three scenarios *mean* in plain English.
# Kept tight (≤ 2 sentences each) so they fit in exec dashboards and
# CEA/SERC filing footnotes without editing.
SCENARIO_NARRATIVES: dict[ScenarioKey, dict[str, str]] = {
    "pessimistic": {
        "label": "Pessimistic (P10)",
        "summary": "Conservative load case — mild weather, subdued industrial "
                   "offtake, slower EV uptake.",
        "weather": "Weather-normal year; no extended heat-wave. HDD/CDD at "
                   "climatological median.",
        "industrial": "Industrial demand −10% vs trend (soft MSME demand, "
                      "commodity down-cycle).",
        "ev_solar": "EV uptake tracks lower bound of APTRANSCO roll-out plan; "
                    "behind-the-meter solar penetration +0.5pp y/y.",
        "use_case": "Procurement floor; PPA minimum take; reliability margin "
                    "on generation dispatch.",
    },
    "bau": {
        "label": "BAU (Central)",
        "summary": "Business-as-usual central forecast — median weather, "
                   "trend industrial growth, CEA-consistent EV uptake.",
        "weather": "CDD/HDD at 21 °C threshold as per CEA 19th EPS "
                   "Guidelines, long-run average.",
        "industrial": "Industrial demand at trend; APEPDCL feeder-wise growth "
                      "at 5-yr CAGR.",
        "ev_solar": "EV uptake matches NITI Aayog mid-case; behind-the-meter "
                    "solar +1pp y/y.",
        "use_case": "Day-ahead market bids; monthly revenue forecasting; "
                    "standard regulatory filings.",
    },
    "optimistic": {
        "label": "Optimistic (P90)",
        "summary": "High-load case — hot summer, strong industrial offtake, "
                   "accelerated EV uptake.",
        "weather": "Hot-summer stress year; CDD +30% vs normal; extended "
                   "May–Jun heat-wave episodes.",
        "industrial": "Industrial demand +15% vs trend (capex cycle up; MSME "
                      "export surge).",
        "ev_solar": "EV uptake at upper bound of NITI mid-case; "
                    "behind-the-meter solar +1.5pp y/y.",
        "use_case": "Peak reserve planning; BESS sizing; transformer "
                    "loadability headroom checks.",
    },
}


def scenario_metadata() -> dict[str, dict[str, str]]:
    """Return the full three-scenario metadata block suitable for embedding
    in an API response. Stable structure — pin this in integration tests."""
    return {k: dict(v) for k, v in SCENARIO_NARRATIVES.items()}


def resolve_scenario(scenario: str | None) -> ScenarioKey | None:
    """Normalise a user-supplied scenario label to the canonical key.

    Accepts: pessimistic/p10/low, bau/central/mean, optimistic/p90/high.
    Returns None if `scenario` is None/empty (meaning: return all three).
    Raises ValueError on an unknown label.
    """
    if scenario is None or scenario == "":
        return None
    s = scenario.strip().lower()
    aliases: dict[str, ScenarioKey] = {
        "pessimistic": "pessimistic", "p10": "pessimistic", "low": "pessimistic",
        "bau": "bau", "central": "bau", "mean": "bau", "baseline": "bau",
        "optimistic": "optimistic", "p90": "optimistic", "high": "optimistic",
    }
    if s not in aliases:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Valid: pessimistic/bau/optimistic (or p10/p90/central)."
        )
    return aliases[s]
