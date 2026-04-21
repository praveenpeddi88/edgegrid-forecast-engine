"""
v5 recursive-prediction backtest on the 8-day verified window (Feb 5 – Feb 12 2026).

Goal: prove that change #1 from the v5 plan (recursive autoregressive
inference) closes the bundle-vs-backtest gap. Compare:
  * v3 backtest MAPE (DOW-primed, single batch predict) — baseline
  * v5 backtest MAPE (recursive, one-step-ahead with prediction feedback) — new

Output: a JSON results file + console table sorted by improvement.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from edgegrid_forecast.inference._features import (  # noqa: E402
    TIME_BLOCKS,
    block_label_for,
    compute_fleet_aggregate,
    fetch_weather_expanded,
    load_meter_data,
)
from edgegrid_forecast.inference.v5_predict import predict_recursive  # noqa: E402


# ── Configuration ──────────────────────────────────────────────────────────
CONTEXT_CUTOFF = pd.Timestamp("2026-02-04T23:30:00")
HORIZON_START  = pd.Timestamp("2026-02-05T00:00:00")
HORIZON_END    = pd.Timestamp("2026-02-12T15:00:00")  # last verified actual
WARMUP_DAYS    = 30  # need ~14d history but pad more for rolling features

OUT_PATH = REPO / "prototypes" / "forecast_engine_v3" / "v5_recursive_backtest.json"


def _meter_msns() -> list[str]:
    """Read the 42 modeled meters from the existing v3 fixture."""
    fx = json.load(open(REPO / "prototypes" / "forecast_engine_v3" / "forecasts_v3.json"))
    return [m["msn"] for m in fx["meters"]]


def _v3_baselines() -> dict[str, dict]:
    """Pull v3 backtest_mape per meter for side-by-side comparison."""
    fx = json.load(open(REPO / "prototypes" / "forecast_engine_v3" / "forecasts_v3.json"))
    out = {}
    for m in fx["meters"]:
        out[m["msn"]] = {
            "bundle_mape": m.get("bundle_holdout_mape"),
            "v3_backtest_mape": m.get("backtest_mape"),
            "tier": m.get("tier"),
            "phase": m.get("phase"),
        }
    return out


def _mape_real(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.001) -> float:
    """MAPE on rows where actual > threshold kWh."""
    mask = y_true > threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)


def _block_mape(ts, y_true, y_pred) -> dict[str, float]:
    h = pd.to_datetime(ts).hour
    out = {}
    for name, fn in TIME_BLOCKS:
        m = fn(h.values)
        if m.sum() < 3:
            out[name] = float("nan")
            continue
        out[name] = round(_mape_real(y_true[m], y_pred[m]), 2)
    return out


def run_one_meter(
    msn: str,
    all_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    fleet_df: pd.DataFrame,
) -> dict:
    """Run recursive backtest for one meter on the verified window."""
    md = all_df[all_df["msn"] == msn].sort_values("ts").reset_index(drop=True)
    if md.empty:
        return {"msn": msn, "error": "no_data"}

    # Convert demand_wh from Wh to kWh per 30-min block? v4 stores in Wh per block.
    # The fixture's "kwh" column was already kWh, so we'll keep Wh internally and
    # convert at the end for comparison.
    context = md[md["ts"] <= CONTEXT_CUTOFF].copy()
    horizon_actuals = md[(md["ts"] >= HORIZON_START) & (md["ts"] <= HORIZON_END)].copy()

    if len(context) < 400:
        return {"msn": msn, "error": "insufficient_context", "n_context": len(context)}
    if len(horizon_actuals) < 48:
        return {"msn": msn, "error": "insufficient_horizon", "n_horizon": len(horizon_actuals)}

    horizon_ts = pd.DatetimeIndex(horizon_actuals["ts"].values)
    t0 = time.time()
    try:
        result = predict_recursive(
            msn, context, horizon_ts,
            weather_df=weather_df, fleet_df=fleet_df,
            progress=False,
        )
    except Exception as e:
        return {"msn": msn, "error": f"predict_failed: {e}"}
    dt = time.time() - t0

    y_true = horizon_actuals["demand_wh"].values.astype(np.float64)
    y_pred = result["forecast_wh"].values.astype(np.float64)
    # Align (defensive)
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    mape_real = _mape_real(y_true, y_pred, threshold=0.5)  # Wh threshold matches v4
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    block = _block_mape(horizon_ts[:n], y_true, y_pred)

    return {
        "msn": msn,
        "n_eval": int(n),
        "v5_mape": round(mape_real, 2) if not np.isnan(mape_real) else None,
        "v5_mae_wh": round(mae, 1),
        "v5_rmse_wh": round(rmse, 1),
        "v5_block_mape": block,
        "wallclock_s": round(dt, 1),
    }


def main(max_meters: Optional[int] = None) -> None:
    print(f"[v5-backtest] loading data...", flush=True)
    all_df, _ = load_meter_data()
    print(f"  loaded {len(all_df)} rows for {all_df['msn'].nunique()} meters")

    print(f"[v5-backtest] fetching weather (cached)...", flush=True)
    weather = fetch_weather_expanded()

    print(f"[v5-backtest] computing fleet aggregate...", flush=True)
    fleet = compute_fleet_aggregate(all_df)

    msns = _meter_msns()
    if max_meters:
        msns = msns[:max_meters]
    baselines = _v3_baselines()

    results = []
    for i, msn in enumerate(msns, 1):
        print(f"  [{i:>2}/{len(msns)}] {msn} ", end="", flush=True)
        r = run_one_meter(msn, all_df, weather, fleet)
        b = baselines.get(msn, {})
        r["bundle_mape"] = b.get("bundle_mape")
        r["v3_backtest_mape"] = b.get("v3_backtest_mape")
        r["tier"] = b.get("tier")
        r["phase"] = b.get("phase")
        if "v5_mape" in r and r["v5_mape"] is not None:
            d = (r["v3_backtest_mape"] or 0) - r["v5_mape"]
            print(f"v3={r['v3_backtest_mape']:>6.2f}%  v5={r['v5_mape']:>6.2f}%  "
                  f"Δ={d:+.2f}pp  ({r['wallclock_s']:.0f}s)")
        else:
            print(f"ERROR: {r.get('error')}")
        results.append(r)

    # Summary
    valid = [r for r in results if r.get("v5_mape") is not None]
    v3_mapes = [r["v3_backtest_mape"] for r in valid if r["v3_backtest_mape"] is not None]
    v5_mapes = [r["v5_mape"] for r in valid]
    bundle_mapes = [r["bundle_mape"] for r in valid if r["bundle_mape"] is not None]

    print("\n" + "=" * 72)
    print(f"  v5 RECURSIVE BACKTEST · {len(valid)}/{len(results)} meters succeeded")
    print("=" * 72)
    print(f"  bundle MAPE     mean={np.mean(bundle_mapes):>6.2f}%  "
          f"median={np.median(bundle_mapes):>6.2f}%  "
          f"p75={np.percentile(bundle_mapes,75):>6.2f}%")
    print(f"  v3 backtest     mean={np.mean(v3_mapes):>6.2f}%  "
          f"median={np.median(v3_mapes):>6.2f}%  "
          f"p75={np.percentile(v3_mapes,75):>6.2f}%")
    print(f"  v5 recursive    mean={np.mean(v5_mapes):>6.2f}%  "
          f"median={np.median(v5_mapes):>6.2f}%  "
          f"p75={np.percentile(v5_mapes,75):>6.2f}%")
    deltas = [r["v3_backtest_mape"] - r["v5_mape"] for r in valid
              if r["v3_backtest_mape"] is not None]
    print(f"  Δ (v3-v5)       mean={np.mean(deltas):>+6.2f}pp  "
          f"median={np.median(deltas):>+6.2f}pp")
    n_under5 = sum(1 for r in valid if r["v5_mape"] < 5)
    n_under10 = sum(1 for r in valid if r["v5_mape"] < 10)
    print(f"  Under 5%:  {n_under5}/{len(valid)} meters")
    print(f"  Under 10%: {n_under10}/{len(valid)} meters")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "context_cutoff": CONTEXT_CUTOFF.isoformat(),
        "horizon_start":  HORIZON_START.isoformat(),
        "horizon_end":    HORIZON_END.isoformat(),
        "n_meters_run":   len(results),
        "n_meters_ok":    len(valid),
        "summary": {
            "bundle_mean":   round(float(np.mean(bundle_mapes)), 2) if bundle_mapes else None,
            "bundle_median": round(float(np.median(bundle_mapes)), 2) if bundle_mapes else None,
            "v3_mean":       round(float(np.mean(v3_mapes)), 2) if v3_mapes else None,
            "v3_median":     round(float(np.median(v3_mapes)), 2) if v3_mapes else None,
            "v5_mean":       round(float(np.mean(v5_mapes)), 2),
            "v5_median":     round(float(np.median(v5_mapes)), 2),
            "delta_mean_pp":   round(float(np.mean(deltas)), 2) if deltas else None,
            "delta_median_pp": round(float(np.median(deltas)), 2) if deltas else None,
            "n_under_5":  n_under5,
            "n_under_10": n_under10,
        },
        "per_meter": results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(max_meters=n)
