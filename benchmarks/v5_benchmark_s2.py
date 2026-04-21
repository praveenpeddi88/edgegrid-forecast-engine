"""
v5 S2 benchmark — stratified temporal holdout.

Every 4th complete day per meter is the holdout.  We use the Feb-12 v5 bundle
to predict those holdout days via `predict_with_context` (batch), and report
two metrics per meter:

  1) leaky_mape   — MAPE on ALL stratified holdout days (most fall inside the
                    v5 training window → v5 has effectively memorised them).
                    Reported for parity with the v4 S2 numbers documented in
                    docs/STRATEGY_2_STRATIFIED_TEMPORAL.md.
  2) clean_mape   — MAPE restricted to holdout days AFTER the v5 train_cutoff
                    (2026-02-12). No leakage. This is the honest S2 number.

A "complete day" = ≥44 of 48 half-hour slots present (≥91.7% completeness).

Outputs:
  benchmarks/results/v5_benchmark_s2.json
  benchmarks/results/v5_benchmark_s2.csv

Fleet summary printed to stdout.

Usage
-----
    PYTHONPATH=src python benchmarks/v5_benchmark_s2.py
    PYTHONPATH=src python benchmarks/v5_benchmark_s2.py --msns 50143025 67003234
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from edgegrid_forecast.inference._features import (
    compute_fleet_aggregate,
    fetch_weather_expanded,
    load_meter_data,
)
from edgegrid_forecast.inference.v4_predict import load_model, predict_with_context

V5_DIR = REPO / "models" / "v5"
OUT_DIR = REPO / "benchmarks" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def cohort(m: float) -> str:
    if m is None or (isinstance(m, float) and (np.isnan(m) or np.isinf(m))):
        return "X: invalid"
    if m < 4:   return "A: <4% (target)"
    if m < 5:   return "A': 4-5%"
    if m < 10:  return "B: 5-10%"
    if m < 15:  return "C: 10-15%"
    if m < 25:  return "D: 15-25%"
    return "E: >=25%"


def _holdout_days(mdf: pd.DataFrame, completeness: int = 44) -> list[pd.Timestamp]:
    """Every 4th complete day (≥44 of 48 slots). Returns list of day-floor ts."""
    mdf = mdf.copy()
    mdf["day"] = mdf["ts"].dt.floor("D")
    day_counts = mdf.groupby("day").size()
    complete_days = sorted(day_counts[day_counts >= completeness].index)
    return complete_days[3::4]  # every 4th, offset so we don't start on day 0


def _mape_safe(y, yhat) -> float:
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    mask = y > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y[mask] - yhat[mask]) / y[mask]) * 100)


def _eval_one_meter(
    msn: str,
    mdf: pd.DataFrame,
    weather_df: pd.DataFrame,
    fleet_df: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> dict:
    """Evaluate v5 bundle on every-4th-day stratified holdout for one meter."""
    bundle = load_model(msn, models_dir=V5_DIR)

    # Build full timeline context (all actuals for this meter).  For S2, lag
    # features for a holdout day legitimately look back into training days
    # (production sees yesterday's actuals before predicting today).
    ctx = mdf[["ts", "demand_wh", "voltage"]].copy().sort_values("ts").reset_index(drop=True)

    h_days = _holdout_days(mdf)
    if not h_days:
        return {"msn": msn, "n_holdout_days": 0, "leaky_mape": None, "clean_mape": None}

    # Run one batch predict over the whole timeline; filter afterwards.
    horizon_ts = pd.DatetimeIndex(mdf["ts"].sort_values().unique())
    result = predict_with_context(
        msn, ctx,
        weather_df=weather_df,
        fleet_df=fleet_df,
        models_dir=V5_DIR,
        horizon_ts=horizon_ts,
    )
    if result.empty:
        return {"msn": msn, "n_holdout_days": 0, "leaky_mape": None, "clean_mape": None}

    res = result.reset_index()  # has ts column now
    # Map actuals onto predictions
    actual_map = mdf.set_index("ts")["demand_wh"]
    res["actual"] = res["ts"].map(actual_map)
    res["day"]    = res["ts"].dt.floor("D")

    h_set = set(h_days)
    holdout = res[res["day"].isin(h_set)].dropna(subset=["actual"]).copy()

    if holdout.empty:
        return {"msn": msn, "n_holdout_days": 0, "leaky_mape": None, "clean_mape": None}

    leaky  = _mape_safe(holdout["actual"].values, holdout["forecast_wh"].values)
    leaky_mae = float(np.mean(np.abs(holdout["actual"].values - holdout["forecast_wh"].values)))
    leaky_mbe = float(np.mean(holdout["forecast_wh"].values - holdout["actual"].values))

    clean = holdout[holdout["ts"] > cutoff]
    if not clean.empty:
        clean_mape = _mape_safe(clean["actual"].values, clean["forecast_wh"].values)
        clean_mae  = float(np.mean(np.abs(clean["actual"].values - clean["forecast_wh"].values)))
        clean_mbe  = float(np.mean(clean["forecast_wh"].values - clean["actual"].values))
        n_clean = int(len(clean))
    else:
        clean_mape = None; clean_mae = None; clean_mbe = None; n_clean = 0

    return {
        "msn": msn,
        "tier": bundle["meta"].get("tier") if "meta" in bundle else None,
        "n_holdout_days": int(holdout["day"].nunique()),
        "n_holdout_slots": int(len(holdout)),
        "n_clean_slots": n_clean,
        "leaky_mape": round(leaky, 3),
        "leaky_mae": round(leaky_mae, 2),
        "leaky_mbe": round(leaky_mbe, 2),
        "clean_mape": round(clean_mape, 3) if clean_mape is not None else None,
        "clean_mae":  round(clean_mae, 2)  if clean_mae  is not None else None,
        "clean_mbe":  round(clean_mbe, 2)  if clean_mbe  is not None else None,
        "leaky_cohort": cohort(leaky),
        "clean_cohort": cohort(clean_mape) if clean_mape is not None else "X: no post-cutoff data",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msns", nargs="*", default=None,
                    help="Subset of MSNs (default: all in v5 manifest)")
    args = ap.parse_args()

    manifest_path = V5_DIR / "_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if isinstance(manifest, dict) and "models" in manifest:
        manifest = manifest["models"]
    msn_list = args.msns if args.msns else [r["msn"] for r in manifest]
    manifest_by_msn = {r["msn"]: r for r in manifest}

    # Detect v5 train cutoff from the manifest (falls back to 2026-02-12)
    cutoff = pd.Timestamp(
        manifest_by_msn[msn_list[0]].get("train_cutoff", "2026-02-12T15:00:00")
    )
    print(f"[s2-bench] v5 train cutoff: {cutoff}")

    print("[s2-bench] loading meter frames + weather + fleet…")
    all_df, _profile = load_meter_data()
    weather_df = fetch_weather_expanded()
    fleet_df = compute_fleet_aggregate(all_df)
    print(f"[s2-bench] meters={all_df['msn'].nunique()}  ts_range=[{all_df['ts'].min()}, {all_df['ts'].max()}]")

    rows: list[dict] = []
    t0 = time.time()
    for i, msn in enumerate(msn_list, 1):
        mdf = all_df[all_df["msn"] == msn].sort_values("ts").reset_index(drop=True)
        if mdf.empty:
            print(f"  [{i:2d}/{len(msn_list)}] {msn}: NO DATA  skip")
            continue
        t1 = time.time()
        try:
            row = _eval_one_meter(msn, mdf, weather_df, fleet_df, cutoff)
        except Exception as e:
            print(f"  [{i:2d}/{len(msn_list)}] {msn}: ERROR {type(e).__name__}: {e}")
            continue
        # Splice manifest tier (for meters where bundle meta lacks it)
        row["tier"] = row.get("tier") or manifest_by_msn.get(msn, {}).get("tier")
        row["s1_holdout_mape"] = manifest_by_msn.get(msn, {}).get("holdout_mape")
        rows.append(row)
        print(f"  [{i:2d}/{len(msn_list)}] {msn:10s} ({row.get('tier') or '?':20s})  "
              f"holdout_days={row['n_holdout_days']:3d}  "
              f"leaky={row['leaky_mape']!r:>7}%  "
              f"clean={row['clean_mape']!r:>7}% "
              f"({time.time()-t1:.1f}s)")

    if not rows:
        print("No meters successfully evaluated.", file=sys.stderr)
        return 1

    # ── Fleet summary ──────────────────────────────────────────────────
    leaky_list = [r["leaky_mape"] for r in rows if r.get("leaky_mape") is not None]
    clean_list = [r["clean_mape"] for r in rows if r.get("clean_mape") is not None]

    def _fleet(xs, label):
        if not xs:
            return {"n": 0}
        a = np.array(xs)
        return {
            "n": len(xs),
            "mean":   round(float(a.mean()), 2),
            "median": round(float(np.median(a)), 2),
            "p90":    round(float(np.percentile(a, 90)), 2),
            "max":    round(float(a.max()), 2),
            "under_4pct":  int((a < 4).sum()),
            "under_10pct": int((a < 10).sum()),
        }

    s_leaky = _fleet(leaky_list, "leaky")
    s_clean = _fleet(clean_list, "clean")

    print("\n═══════════════ S2 STRATIFIED (every-4th-day holdout) ═══════════════")
    print(f"  v5 leaky (all holdout days — pre+post cutoff mixed):  {s_leaky}")
    print(f"  v5 clean (post-cutoff days only, no leakage):         {s_clean}")

    # cohort histogram
    c_leaky = Counter(r["leaky_cohort"] for r in rows if r.get("leaky_mape") is not None)
    c_clean = Counter(r["clean_cohort"] for r in rows if r.get("clean_mape") is not None)
    print("\n  Cohort distribution")
    print(f"    {'cohort':20s}  {'leaky':>6s}   {'clean':>6s}")
    for k in sorted(set(c_leaky) | set(c_clean)):
        print(f"    {k:20s}  {c_leaky.get(k,0):6d}   {c_clean.get(k,0):6d}")

    # per-tier breakdown
    from collections import defaultdict
    by_tier: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("leaky_mape") is not None:
            by_tier[r.get("tier") or "Unknown"].append(r["leaky_mape"])
    print("\n  Per-tier leaky MAPE")
    for tier, vals in sorted(by_tier.items()):
        a = np.array(vals)
        print(f"    {tier:25s}  n={len(vals):2d}  mean={a.mean():6.2f}%  median={np.median(a):6.2f}%  max={a.max():6.2f}%")

    # Persist artifacts
    out_json = OUT_DIR / "v5_benchmark_s2.json"
    out_json.write_text(json.dumps({
        "benchmark": "S2 stratified (every 4th complete day)",
        "train_cutoff": str(cutoff),
        "v5_leaky_fleet": s_leaky,
        "v5_clean_fleet": s_clean,
        "cohort_leaky": dict(c_leaky),
        "cohort_clean": dict(c_clean),
        "per_meter": rows,
    }, indent=2, default=str))

    out_csv = OUT_DIR / "v5_benchmark_s2.csv"
    if rows:
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"\n  → JSON: {out_json.relative_to(REPO)}")
    print(f"  → CSV:  {out_csv.relative_to(REPO)}")
    print(f"  → total {time.time()-t0:.1f}s for {len(rows)} meters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
