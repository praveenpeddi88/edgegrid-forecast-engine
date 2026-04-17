"""
Unit tests for the production inference wrapper (Session 11 — Gap A).

These tests exercise the full load → predict path against two meters that are
already persisted at models/v4/ (one with bias gate=gated_out, one with
gate=applied). The benchmark CSV (benchmarks/results/benchmark_strategy1_v4.csv)
provides the authoritative holdout metrics; we assert the persisted bundle
matches those numbers exactly — that's our guard against silent drift in the
replicated feature pipeline at src/edgegrid_forecast/inference/_features.py.

Run:
    pytest tests/test_v4_predict.py -v

Requires:
    * models/v4/67001151.joblib   (pre-trained by scripts/train_all_v4.py)
    * models/v4/67003234.joblib   (pre-trained — has gate=applied)
    * data/raw/*.parquet
    * data/external/weather/visakhapatnam_expanded_*.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from edgegrid_forecast.inference import (  # noqa: E402
    MODEL_VERSION,
    load_model,
    predict,
    predict_with_context,
)
from edgegrid_forecast.inference._features import (  # noqa: E402
    TIME_BLOCKS,
    block_label_for,
    calc_metrics,
    compute_fleet_aggregate,
    fetch_weather_expanded,
    load_meter_data,
)


SMOKE_MSN_GATED = "67001151"   # gate=gated_out, MAPE=6.34%
SMOKE_MSN_APPLIED = "67003234" # gate=applied,   MAPE=10.59%
MODELS_DIR = REPO / "models" / "v4"


def _has_bundle(msn: str) -> bool:
    return (MODELS_DIR / f"{msn}.joblib").exists()


# ════════════════════════════════════════════════════════════════════════════
# Bundle schema & load_model
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not _has_bundle(SMOKE_MSN_GATED), reason="bundle not trained yet")
def test_bundle_schema_complete():
    bundle = load_model(SMOKE_MSN_GATED)
    required = {
        "msn", "tier", "phase", "version", "strategy",
        "model_mean", "model_q10", "model_q90",
        "selected_features", "use_log1p", "trail_mbe_capped", "bias_gate",
        "best_iter", "train_cutoff", "trained_at", "train_mean_demand",
        "holdout_metrics", "historical_block_mape",
    }
    missing = required - set(bundle.keys())
    assert not missing, f"Bundle missing keys: {missing}"
    assert bundle["version"] == MODEL_VERSION
    assert bundle["strategy"] == "S1"
    assert bundle["msn"] == SMOKE_MSN_GATED


def test_load_model_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_model("nonexistent-msn-xxxx")


# ════════════════════════════════════════════════════════════════════════════
# MAPE parity vs benchmark CSV
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not _has_bundle(SMOKE_MSN_GATED), reason="bundle not trained yet")
def test_mape_parity_gated():
    """Persisted bundle's holdout MAPE must match benchmark CSV for 67001151."""
    bundle = load_model(SMOKE_MSN_GATED)
    benchmark = pd.read_csv(REPO / "benchmarks" / "results" / "benchmark_strategy1_v4.csv")
    benchmark["msn"] = benchmark["msn"].astype(str)
    row = benchmark[benchmark["msn"] == SMOKE_MSN_GATED].iloc[0]
    assert bundle["holdout_metrics"]["mape"] == pytest.approx(row["mape"], abs=0.01)
    assert bundle["bias_gate"] == row["bias_gate"]


@pytest.mark.skipif(not _has_bundle(SMOKE_MSN_APPLIED), reason="bundle not trained yet")
def test_mape_parity_applied():
    """Second meter exercises the Phase B bias-correction path (gate=applied)."""
    bundle = load_model(SMOKE_MSN_APPLIED)
    benchmark = pd.read_csv(REPO / "benchmarks" / "results" / "benchmark_strategy1_v4.csv")
    benchmark["msn"] = benchmark["msn"].astype(str)
    row = benchmark[benchmark["msn"] == SMOKE_MSN_APPLIED].iloc[0]
    assert bundle["holdout_metrics"]["mape"] == pytest.approx(row["mape"], abs=0.01)
    assert bundle["bias_gate"] == "applied"
    assert abs(bundle["trail_mbe_capped"]) > 0.5  # correction is actually applied


# ════════════════════════════════════════════════════════════════════════════
# predict() — Rich DataFrame shape and contract
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not _has_bundle(SMOKE_MSN_GATED), reason="bundle not trained yet")
def test_predict_returns_rich_dataframe():
    df = predict(SMOKE_MSN_GATED, horizon=48)
    assert len(df) == 48
    assert df.index.name == "ts"
    assert isinstance(df.index, pd.DatetimeIndex)
    required_cols = {
        "forecast_wh", "confidence_low", "confidence_high",
        "block_label", "historical_block_mape",
    }
    assert set(df.columns) == required_cols


@pytest.mark.skipif(not _has_bundle(SMOKE_MSN_GATED), reason="bundle not trained yet")
def test_predict_non_negative():
    df = predict(SMOKE_MSN_GATED, horizon=48)
    assert (df["forecast_wh"] >= 0).all()
    assert (df["confidence_low"] >= 0).all()
    assert (df["confidence_high"] >= 0).all()


@pytest.mark.skipif(not _has_bundle(SMOKE_MSN_GATED), reason="bundle not trained yet")
def test_predict_confidence_monotonic():
    """confidence_low ≤ forecast_wh ≤ confidence_high should always hold."""
    df = predict(SMOKE_MSN_GATED, horizon=48)
    assert (df["confidence_low"] <= df["forecast_wh"] + 1e-6).all()
    assert (df["forecast_wh"] <= df["confidence_high"] + 1e-6).all()


@pytest.mark.skipif(not _has_bundle(SMOKE_MSN_GATED), reason="bundle not trained yet")
def test_predict_block_labels_valid():
    df = predict(SMOKE_MSN_GATED, horizon=48)
    valid_blocks = {"night", "morning", "solar", "peak"}
    assert set(df["block_label"].unique()).issubset(valid_blocks)

    # Spot-check: 4am should label as 'night', 8am as 'morning',
    # 2pm as 'solar', 8pm as 'peak'
    assert block_label_for(pd.Timestamp("2026-01-01 04:00")) == "night"
    assert block_label_for(pd.Timestamp("2026-01-01 08:00")) == "morning"
    assert block_label_for(pd.Timestamp("2026-01-01 14:00")) == "solar"
    assert block_label_for(pd.Timestamp("2026-01-01 20:00")) == "peak"
    assert block_label_for(pd.Timestamp("2026-01-01 23:30")) == "night"


@pytest.mark.skipif(not _has_bundle(SMOKE_MSN_GATED), reason="bundle not trained yet")
def test_predict_historical_block_mape_matches_bundle():
    df = predict(SMOKE_MSN_GATED, horizon=48)
    bundle = load_model(SMOKE_MSN_GATED)
    for _, row in df.iterrows():
        expected = bundle["historical_block_mape"][row["block_label"]]
        assert row["historical_block_mape"] == pytest.approx(expected, abs=0.01)


@pytest.mark.skipif(not _has_bundle(SMOKE_MSN_GATED), reason="bundle not trained yet")
def test_predict_horizon_length_respected():
    for h in [4, 12, 24, 48]:
        df = predict(SMOKE_MSN_GATED, horizon=h)
        assert len(df) == h


# ════════════════════════════════════════════════════════════════════════════
# predict_with_context — caller supplies data
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not _has_bundle(SMOKE_MSN_GATED), reason="bundle not trained yet")
def test_predict_with_context_matches_predict():
    """predict() is sugar over predict_with_context(); equal results expected."""
    # predict() path
    df_wrap = predict(SMOKE_MSN_GATED, horizon=12)

    # Direct path
    all_df, _ = load_meter_data()
    mdata = all_df[all_df["msn"] == SMOKE_MSN_GATED].sort_values("ts").copy()
    as_of = mdata["ts"].max()
    context = mdata[mdata["ts"] <= as_of].tail(500).copy()
    weather = fetch_weather_expanded()
    fleet = compute_fleet_aggregate(all_df)
    df_direct = predict_with_context(
        SMOKE_MSN_GATED,
        context,
        weather_df=weather,
        fleet_df=fleet,
        horizon_ts=context["ts"].tail(12),
    )

    assert len(df_wrap) == len(df_direct) == 12
    pd.testing.assert_series_equal(
        df_wrap["forecast_wh"].reset_index(drop=True),
        df_direct["forecast_wh"].reset_index(drop=True),
        check_names=False,
        atol=0.01,
    )


# ════════════════════════════════════════════════════════════════════════════
# Metrics helper sanity checks
# ════════════════════════════════════════════════════════════════════════════
def test_calc_metrics_perfect_prediction():
    y = np.array([100.0, 200.0, 300.0, 400.0])
    m = calc_metrics(y, y.copy())
    assert m["mape"] == 0.0
    assert m["mae"] == 0.0
    assert m["mbe"] == 0.0


def test_calc_metrics_constant_bias():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = y_true + 50.0
    m = calc_metrics(y_true, y_pred)
    assert m["mbe"] == 50.0
    assert m["mae"] == 50.0


def test_time_blocks_cover_24h():
    """Every hour must belong to exactly one block."""
    for h in range(24):
        matches = [name for name, pred in TIME_BLOCKS if pred(np.array([h]))[0]]
        assert len(matches) == 1, f"hour {h} matched {matches}"
