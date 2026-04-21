"""
Tests for the CEA Bridge layer (W1 features + W2 three scenarios + W4 derived).

The CEA Bridge is the commercial/regulatory wrapper on top of v5:
- W1: 21 °C HDD/CDD features in the canonical builder
- W2: Pessimistic/BAU/Optimistic scenario rebrand of q10/mean/q90
- W4: Peak-kW, load-factor, diversity-factor derivation from the Wh forecast

These tests lock the contract so downstream dashboards/filings don't break
silently on refactors.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edgegrid_forecast.inference._derived import (
    WH_PER_SLOT_TO_KW,
    derive,
    diversity_factor,
    load_factor,
    peak_kw,
    wh_to_kw,
)
from edgegrid_forecast.inference._scenarios import (
    QUANTILE_TO_SCENARIO,
    SCENARIO_NARRATIVES,
    SCENARIO_TO_QUANTILE,
    resolve_scenario,
    scenario_metadata,
)


# ════════════════════════════════════════════════════════════════════════════
# W1: HDD/CDD features land in the canonical builder
# ════════════════════════════════════════════════════════════════════════════
def test_w1_hdd_cdd_features_present():
    """CEA-canonical 21 °C HDD/CDD features must be exposed by the v4/v5
    shared feature builder. Losing these would silently un-CEA-align the
    engine, so pin them here. Uses the production data loader to avoid
    drift with the builder's weather-column expectations."""
    from edgegrid_forecast.inference._features import (
        build_features_v4_s1,
        compute_fleet_aggregate,
        fetch_weather_expanded,
        load_meter_data,
    )

    all_df, _ = load_meter_data()
    # Single meter, tail slice — keep this fast.
    msn = str(all_df["msn"].unique()[0])
    md = all_df[all_df["msn"] == msn].tail(800).copy()
    wx = fetch_weather_expanded()
    fleet = compute_fleet_aggregate(all_df)

    result = build_features_v4_s1(md, wx, fleet)
    # The builder returns a tuple (feature_df, …) in some versions or a bare
    # DataFrame in others — accept both.
    out = result[0] if isinstance(result, tuple) else result
    assert isinstance(out, pd.DataFrame), f"Unexpected return type: {type(out)}"

    for col in ["hdd_21", "cdd_21", "cdd_21_rmean_48", "cdd_21_rmean_336",
                "cdd_21_x_hour", "cdd_21_x_peak"]:
        assert col in out.columns, f"Missing CEA-canonical feature: {col}"

    # HDD/CDD non-negative definition
    assert (out["hdd_21"].dropna() >= 0).all()
    assert (out["cdd_21"].dropna() >= 0).all()
    # HDD and CDD can't both be strictly positive at the same moment (21 °C break)
    both_pos = ((out["hdd_21"] > 0) & (out["cdd_21"] > 0)).sum()
    assert both_pos == 0, f"HDD and CDD can't both be positive: {both_pos} rows do"


# ════════════════════════════════════════════════════════════════════════════
# W2: three-scenario rebrand
# ════════════════════════════════════════════════════════════════════════════
def test_w2_scenario_mapping_stable():
    """Lock the statistical→commercial label mapping — downstream dashboards
    depend on this contract."""
    assert SCENARIO_TO_QUANTILE == {
        "pessimistic": "q10",
        "bau": "mean",
        "optimistic": "q90",
    }
    assert QUANTILE_TO_SCENARIO == {
        "q10": "pessimistic", "mean": "bau", "q90": "optimistic",
    }


def test_w2_scenario_narratives_shape():
    """Each scenario must ship with the five narrative fields exec dashboards
    and CEA/SERC footnotes expect."""
    required = {"label", "summary", "weather", "industrial", "ev_solar", "use_case"}
    for key in ("pessimistic", "bau", "optimistic"):
        assert set(SCENARIO_NARRATIVES[key].keys()) == required


def test_w2_metadata_is_deep_copy():
    """scenario_metadata() must hand out an independent copy so callers
    mutating the payload can't corrupt the module singleton."""
    m = scenario_metadata()
    m["bau"]["label"] = "MUTATED"
    m2 = scenario_metadata()
    assert m2["bau"]["label"] != "MUTATED"


def test_w2_resolve_aliases():
    assert resolve_scenario(None) is None
    assert resolve_scenario("") is None
    assert resolve_scenario("Pessimistic") == "pessimistic"
    assert resolve_scenario("P10") == "pessimistic"
    assert resolve_scenario("bau") == "bau"
    assert resolve_scenario("central") == "bau"
    assert resolve_scenario("mean") == "bau"
    assert resolve_scenario("P90") == "optimistic"
    with pytest.raises(ValueError, match="Unknown scenario"):
        resolve_scenario("nonsense")


# ════════════════════════════════════════════════════════════════════════════
# W4: derived metrics — math
# ════════════════════════════════════════════════════════════════════════════
def test_w4_wh_slot_to_kw_constant():
    """30-min slot: 1000 Wh → 2 kW. Pin this or the unit conversion will silently drift."""
    assert WH_PER_SLOT_TO_KW == pytest.approx(2.0 / 1000.0)
    assert float(wh_to_kw(pd.Series([1000.0])).iloc[0]) == pytest.approx(2.0)


def test_w4_flat_load_gives_LF_1():
    idx = pd.date_range("2026-04-21", periods=48, freq="30min")
    s = pd.Series([1000.0] * 48, index=idx)
    d = derive(s, idx)
    assert d.peak_kw == pytest.approx(2.0)
    assert d.average_kw == pytest.approx(2.0)
    assert d.load_factor == pytest.approx(1.0)
    assert d.total_energy_kwh == pytest.approx(48.0)   # 48 × 1 kWh
    assert d.horizon_hours == pytest.approx(24.0)


def test_w4_peaky_load_gives_low_LF():
    idx = pd.date_range("2026-04-21", periods=4, freq="30min")
    s = pd.Series([500, 500, 5000, 500], index=idx, dtype=float)
    d = derive(s, idx)
    # peak = 5000 × 2/1000 = 10 kW; avg = 6500/4 × 2/1000 = 3.25 kW
    assert d.peak_kw == pytest.approx(10.0)
    assert d.load_factor == pytest.approx(0.325)
    assert d.peak_ts == "2026-04-21T01:00:00"


def test_w4_empty_input_is_safe():
    d = derive(pd.Series([], dtype=float))
    assert d.peak_kw == 0.0 and d.load_factor == 0.0
    assert load_factor(pd.Series([], dtype=float)) == 0.0
    peak, ts = peak_kw(pd.Series([], dtype=float))
    assert peak == 0.0 and ts == ""


def test_w4_diversity_factor_bounds():
    """DF = 1 when all peaks coincide, DF = N when peaks are fully staggered
    (N meters, each peaking in a disjoint slot)."""
    idx = pd.date_range("2026-04-21", periods=4, freq="30min")
    # Fully coincident: DF = 1
    s_same = {"a": pd.Series([1000, 0, 0, 0], index=idx),
              "b": pd.Series([1000, 0, 0, 0], index=idx)}
    assert diversity_factor(s_same) == pytest.approx(1.0)

    # Fully non-coincident: DF = 2
    s_staggered = {"a": pd.Series([1000, 0, 0, 0], index=idx),
                   "b": pd.Series([0, 0, 1000, 0], index=idx)}
    assert diversity_factor(s_staggered) == pytest.approx(2.0)

    # Always ≥ 1 (by construction, as long as at least one meter has load)
    assert diversity_factor(s_same) >= 1.0
    assert diversity_factor(s_staggered) >= 1.0

    # Empty input is safe
    assert diversity_factor({}) == 0.0


# ════════════════════════════════════════════════════════════════════════════
# API contract tests — W2 + W4 surfaced via FastAPI router
# ════════════════════════════════════════════════════════════════════════════
def _has_v5_bundles() -> bool:
    root = Path(__file__).resolve().parents[1]
    return (root / "models" / "v5" / "_manifest.json").exists()


@pytest.mark.skipif(not _has_v5_bundles(), reason="v5 bundles not present")
def test_api_scenarios_endpoint():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from edgegrid_forecast.api.v5_router import router

    app = FastAPI(); app.include_router(router)
    c = TestClient(app)
    r = c.get("/v5/scenarios")
    assert r.status_code == 200
    j = r.json()
    assert j["framework"].startswith("CEA")
    assert set(j["scenarios"].keys()) == {"pessimistic", "bau", "optimistic"}
    assert j["mapping"]["bau"].startswith("mean")


@pytest.mark.skipif(not _has_v5_bundles(), reason="v5 bundles not present")
def test_api_predict_carries_scenarios_and_derived():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from edgegrid_forecast.api.v5_router import router

    app = FastAPI(); app.include_router(router)
    c = TestClient(app)

    meters = c.get("/v5/meters").json()
    good = sorted(
        [m for m in meters if m.get("holdout_mape") is not None],
        key=lambda x: x["holdout_mape"],
    )
    assert good, "no v5 meters with holdout_mape"
    msn = good[0]["msn"]

    r = c.post("/v5/predict", json={"msn": msn, "horizon_steps": 6, "mode": "batch"})
    assert r.status_code == 200, r.text
    j = r.json()

    # W2: dual-labelled rows
    row = j["predictions"][0]
    assert row["bau_wh"] == pytest.approx(row["forecast_wh"])
    assert row["pessimistic_wh"] == pytest.approx(row["q10_wh"])
    assert row["optimistic_wh"] == pytest.approx(row["q90_wh"])
    # full scenario block on response
    assert set(j["scenarios"].keys()) == {"pessimistic", "bau", "optimistic"}

    # W4: derived metrics present and sensible
    d = j["derived"]
    for k in ("pessimistic", "bau", "optimistic"):
        blk = d[k]
        assert blk["peak_kw"] > 0
        assert 0 <= blk["load_factor"] <= 1
        assert blk["horizon_hours"] == pytest.approx(3.0)   # 6 × 30min

    # P10 peak ≤ mean peak ≤ P90 peak (quantile ordering)
    assert d["pessimistic"]["peak_kw"] <= d["bau"]["peak_kw"] + 1e-6
    assert d["bau"]["peak_kw"] <= d["optimistic"]["peak_kw"] + 1e-6


@pytest.mark.skipif(not _has_v5_bundles(), reason="v5 bundles not present")
def test_api_predict_scenario_filter():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from edgegrid_forecast.api.v5_router import router

    app = FastAPI(); app.include_router(router)
    c = TestClient(app)
    meters = c.get("/v5/meters").json()
    good = sorted(
        [m for m in meters if m.get("holdout_mape") is not None],
        key=lambda x: x["holdout_mape"],
    )
    msn = good[0]["msn"]

    r = c.post("/v5/predict",
               json={"msn": msn, "horizon_steps": 2, "mode": "batch",
                     "scenario": "Pessimistic"})
    assert r.status_code == 200
    j = r.json()
    assert list(j["scenarios"].keys()) == ["pessimistic"]

    # bogus
    r = c.post("/v5/predict",
               json={"msn": msn, "horizon_steps": 2, "mode": "batch",
                     "scenario": "moon-shot"})
    assert r.status_code == 400


@pytest.mark.skipif(not _has_v5_bundles(), reason="v5 bundles not present")
def test_api_fleet_peak_diversity_invariant():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from edgegrid_forecast.api.v5_router import router

    app = FastAPI(); app.include_router(router)
    c = TestClient(app)
    meters = c.get("/v5/meters").json()
    good = sorted(
        [m for m in meters if m.get("holdout_mape") is not None],
        key=lambda x: x["holdout_mape"],
    )[:3]
    msns = [m["msn"] for m in good]

    r = c.post("/v5/fleet/peak", json={"msns": msns, "horizon_steps": 12})
    assert r.status_code == 200
    j = r.json()
    assert j["n_meters"] == len(msns)
    # Core engineering invariant: Σ individual peaks ≥ coincident peak
    assert j["sum_individual_peaks_kw"] + 1e-6 >= j["coincident_peak_kw"]
    # DF ≥ 1 by construction
    assert j["diversity_factor"] + 1e-6 >= 1.0
