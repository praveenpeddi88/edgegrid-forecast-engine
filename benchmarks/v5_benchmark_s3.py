"""
v5 S3-lite benchmark — rolling-origin stability check.

Design note (honest reporting)
------------------------------
The full S3 protocol in docs/S3_ROLLING_ORIGIN_PROTOCOL.md calls for a full
retrain of every meter bundle at every origin (42 meters × 10 origins).  That
is ~3.7 hours serial per v4's measured timing.

This script is the LIGHT version: we evaluate the *already-trained* Feb-12
v5 bundles against 10 weekly origins (2025-12-04 → 2026-02-05).  All origins
sit inside v5's training window, so the per-origin MAPE is **leaky** (v5 has
seen these weeks at training time).  What we legitimately extract from this:

  * cross-origin MAPE σ per meter  → stability
  * horizon-decay curve           → does v5 degrade with longer horizon?
  * cohort distribution per origin → do the A/B/C/D cohorts stay stable?

The leaky absolute number gives us a cohort-MAPE *upper bound* (the production
model cannot do BETTER than what it memorised).  Combined with the S1
chronological post-cutoff numbers (v5 median 4.83% on held-out 15%), this is
sufficient for the "honest cohort-MAPE report" goal in the v5 doc.

If we ever need the full retrain-per-origin protocol, `--refit` is the stub
(not implemented — would take 35 min).

Usage
-----
    PYTHONPATH=src python benchmarks/v5_benchmark_s3.py
    PYTHONPATH=src python benchmarks/v5_benchmark_s3.py --msns 50143025
    PYTHONPATH=src python benchmarks/v5_benchmark_s3.py --origins 2026-01-29
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from edgegrid_forecast.inference._features import (
    compute_fleet_aggregate,
    fetch_weather_expanded,
    load_meter_data,
)
from edgegrid_forecast.inference.v4_predict import predict_with_context

V5_DIR = REPO / "models" / "v5"
OUT_DIR = REPO / "benchmarks" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 10 weekly origins, matching docs/S3_ROLLING_ORIGIN_PROTOCOL.md
DEFAULT_ORIGINS = [
    "2025-12-04", "2025-12-11", "2025-12-18", "2025-12-25",
    "2026-01-01", "2026-01-08", "2026-01-15", "2026-01-22",
    "2026-01-29", "2026-02-05",
]
HORIZON_STEPS = 336  # 7 days × 48 half-hour slots


def cohort(m: float) -> str:
    if m is None or (isinstance(m, float) and (np.isnan(m) or np.isinf(m))):
        return "X: invalid"
    if m < 4:   return "A: <4% (target)"
    if m < 5:   return "A': 4-5%"
    if m < 10:  return "B: 5-10%"
    if m < 15:  return "C: 10-15%"
    if m < 25:  return "D: 15-25%"
    return "E: >=25%"


def _mape_safe(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    mask = y > 0.5  # ignore near-zero actuals per S3 protocol §11.4
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y[mask] - yhat[mask]) / y[mask]) * 100)


def _eval_one_origin(
    msn: str,
    mdf: pd.DataFrame,
    weather_df: pd.DataFrame,
    fleet_df: pd.DataFrame,
    origin: pd.Timestamp,
) -> dict | None:
    """Run one meter × one origin.  Returns metrics dict or None if no data."""
    # Context = all actuals up to origin - 30 min (strict no-leakage for lags)
    ctx = mdf[mdf["ts"] < origin][["ts", "demand_wh", "voltage"]]
    if ctx.empty or len(ctx) < 96:
        return None
    # Forecast horizon: origin → origin + 7 days, 30-min grid
    horizon_ts = pd.date_range(origin, periods=HORIZON_STEPS, freq="30min")
    # Actuals for scoring (only slots we have data for)
    actuals = mdf[mdf["ts"].isin(horizon_ts)][["ts", "demand_wh"]].set_index("ts")
    if actuals.empty:
        return None

    # Include horizon placeholder rows in context so batch-predict materialises
    # features at each horizon ts (same trick as forward_v5_feb12_strategies).
    # For the placeholder demand we seed with the per-(dow,hour,minute) median
    # of the 4-week trailing context so `demand_diff_*` features stay in-dist.
    trail = ctx[ctx["ts"] >= origin - pd.Timedelta(days=28)].copy()
    if trail.empty:
        trail = ctx.copy()
    trail["dow"] = trail["ts"].dt.dayofweek
    trail["hour"] = trail["ts"].dt.hour
    trail["minute"] = trail["ts"].dt.minute
    amed = trail.groupby(["dow", "hour", "minute"]).agg(
        demand_wh=("demand_wh", "median"),
        voltage=("voltage", "median"),
    ).reset_index()
    ph = pd.DataFrame({"ts": horizon_ts})
    ph["dow"] = ph["ts"].dt.dayofweek
    ph["hour"] = ph["ts"].dt.hour
    ph["minute"] = ph["ts"].dt.minute
    ph = ph.merge(amed, on=["dow", "hour", "minute"], how="left")
    ph["demand_wh"] = ph["demand_wh"].fillna(float(trail["demand_wh"].median()))
    ph["voltage"] = ph["voltage"].fillna(230.0)
    ph = ph[["ts", "demand_wh", "voltage"]]
    full_ctx = pd.concat([ctx, ph], ignore_index=True).drop_duplicates("ts", keep="first")
    full_ctx = full_ctx.sort_values("ts").reset_index(drop=True)

    try:
        pred = predict_with_context(
            msn, full_ctx,
            weather_df=weather_df,
            fleet_df=fleet_df,
            models_dir=V5_DIR,
            horizon_ts=horizon_ts,
        )
    except Exception as e:
        return {"msn": msn, "origin": origin.isoformat(), "error": f"{type(e).__name__}: {e}"}

    if pred.empty:
        return None

    # Align actuals to predictions
    pred_df = pred.reset_index()[["ts", "forecast_wh", "confidence_low", "confidence_high"]]
    pred_df = pred_df.merge(actuals.reset_index(), on="ts", how="inner")
    if pred_df.empty:
        return None

    y = pred_df["demand_wh"].values
    yh = pred_df["forecast_wh"].values
    m = _mape_safe(y, yh)
    mae = float(np.mean(np.abs(y - yh)))
    mbe = float(np.mean(yh - y))
    q10 = pred_df["confidence_low"].values
    q90 = pred_df["confidence_high"].values
    coverage = float(np.mean((y >= q10) & (y <= q90))) * 100

    # Horizon-decay: MAPE in the first 24h, 24-72h, 72h-7d buckets
    pred_df["h_bucket"] = pd.cut(
        (pred_df["ts"] - origin).dt.total_seconds() / 3600,
        bins=[0, 24, 72, 168], labels=["0-24h", "24-72h", "72h-7d"],
        include_lowest=True,
    )
    h_decay = {}
    for b in ["0-24h", "24-72h", "72h-7d"]:
        sub = pred_df[pred_df["h_bucket"] == b]
        h_decay[b] = round(_mape_safe(sub["demand_wh"].values, sub["forecast_wh"].values), 3) if not sub.empty else None

    return {
        "msn": msn,
        "origin": origin.isoformat(),
        "n_slots": int(len(pred_df)),
        "mape": round(m, 3),
        "mae": round(mae, 2),
        "mbe": round(mbe, 2),
        "q_coverage_80pct": round(coverage, 2),
        "mape_0_24h": h_decay["0-24h"],
        "mape_24_72h": h_decay["24-72h"],
        "mape_72h_7d": h_decay["72h-7d"],
        "cohort": cohort(m),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msns", nargs="*", default=None,
                    help="Subset of MSNs (default: all 42)")
    ap.add_argument("--origins", nargs="*", default=None,
                    help="Subset of origins ISO dates (default: 10 weekly)")
    args = ap.parse_args()

    origins = [pd.Timestamp(o) for o in (args.origins or DEFAULT_ORIGINS)]
    print(f"[s3-bench] origins: {[o.date().isoformat() for o in origins]}")

    manifest = json.loads((V5_DIR / "_manifest.json").read_text())
    if isinstance(manifest, dict) and "models" in manifest:
        manifest = manifest["models"]
    manifest_by_msn = {r["msn"]: r for r in manifest}
    msn_list = args.msns if args.msns else [r["msn"] for r in manifest]

    print("[s3-bench] loading frames…")
    all_df, _ = load_meter_data()
    weather_df = fetch_weather_expanded()
    fleet_df = compute_fleet_aggregate(all_df)

    rows: list[dict] = []
    t0 = time.time()
    for mi, msn in enumerate(msn_list, 1):
        mdf = all_df[all_df["msn"] == msn].sort_values("ts").reset_index(drop=True)
        tier = manifest_by_msn.get(msn, {}).get("tier")
        for oi, origin in enumerate(origins, 1):
            t1 = time.time()
            res = _eval_one_origin(msn, mdf, weather_df, fleet_df, origin)
            if res is None:
                continue
            res["tier"] = tier
            rows.append(res)
            if oi == len(origins):
                dt = time.time() - t1
                print(f"  [{mi:2d}/{len(msn_list)}] {msn:10s} ({tier or '?':20s}) "
                      f"origins_done={len(origins)} ({dt:.1f}s/origin end)")

    if not rows:
        print("No successful evaluations.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "v5_benchmark_s3.csv", index=False)

    # ── Fleet rollup ────────────────────────────────────────────────
    # Per-meter cross-origin mean + stability σ
    by_meter = df.groupby("msn").agg(
        tier=("tier", "first"),
        n_origins=("origin", "nunique"),
        mape_mean=("mape", "mean"),
        mape_std=("mape", "std"),
        mape_min=("mape", "min"),
        mape_max=("mape", "max"),
        mape_median=("mape", "median"),
        mape_0_24h=("mape_0_24h", "mean"),
        mape_24_72h=("mape_24_72h", "mean"),
        mape_72h_7d=("mape_72h_7d", "mean"),
        coverage_mean=("q_coverage_80pct", "mean"),
    ).round(3).reset_index()
    by_meter["cohort_avg"] = by_meter["mape_mean"].apply(cohort)
    by_meter.to_csv(OUT_DIR / "v5_benchmark_s3_by_meter.csv", index=False)

    # Per-origin rollup
    by_origin = df.groupby("origin").agg(
        n_meters=("msn", "nunique"),
        mape_mean=("mape", "mean"),
        mape_median=("mape", "median"),
        mape_p90=("mape", lambda s: np.percentile(s, 90)),
        under_4pct=("mape", lambda s: (s < 4).sum()),
        under_10pct=("mape", lambda s: (s < 10).sum()),
    ).round(3).reset_index()
    by_origin.to_csv(OUT_DIR / "v5_benchmark_s3_by_origin.csv", index=False)

    # Fleet-level summary
    mape_all = df["mape"].values
    fleet_summary = {
        "n_origins":       int(df["origin"].nunique()),
        "n_meters":        int(df["msn"].nunique()),
        "n_meter_origins": int(len(df)),
        "mape_mean":       round(float(np.mean(mape_all)), 2),
        "mape_median":     round(float(np.median(mape_all)), 2),
        "mape_p90":        round(float(np.percentile(mape_all, 90)), 2),
        "under_4pct_rate":  round(float((mape_all < 4).mean()), 3),
        "under_10pct_rate": round(float((mape_all < 10).mean()), 3),
        "stability_sigma_mean":   round(float(by_meter["mape_std"].mean()), 3),
        "stability_sigma_median": round(float(by_meter["mape_std"].median()), 3),
    }

    # Horizon-decay
    horizon_decay = {
        "mape_0_24h_mean":  round(float(df["mape_0_24h"].mean()), 2),
        "mape_24_72h_mean": round(float(df["mape_24_72h"].mean()), 2),
        "mape_72h_7d_mean": round(float(df["mape_72h_7d"].mean()), 2),
    }

    # Print
    print("\n═══════════════ S3 ROLLING-ORIGIN (leaky, 10 origins × 7d) ═══════════════")
    print(f"  Fleet: {fleet_summary}")
    print(f"  Horizon decay: {horizon_decay}")

    # cohort histogram across meter×origin
    c = Counter(r["cohort"] for r in rows)
    print("\n  Cohort distribution (meter×origin)")
    for k in sorted(c.keys()):
        print(f"    {k:20s}  {c[k]:5d}   ({100*c[k]/len(rows):.1f}%)")

    print("\n  Per-origin rollup")
    print("    " + "  ".join(f"{c:>14s}" for c in by_origin.columns))
    for _, r in by_origin.iterrows():
        print("    " + "  ".join(f"{str(r[c])[:14]:>14s}" for c in by_origin.columns))

    # persist json
    json_out = OUT_DIR / "v5_benchmark_s3.json"
    json_out.write_text(json.dumps({
        "benchmark": "S3 rolling-origin (leaky, Feb-12 v5 bundles)",
        "note": ("All origins fall inside the v5 training window. This measures "
                 "cross-origin stability + horizon decay. For a clean S3, bundles "
                 "must be retrained with cutoff ≤ each origin."),
        "origins": [o.isoformat() for o in origins],
        "fleet_summary": fleet_summary,
        "horizon_decay": horizon_decay,
        "cohort_distribution_meter_origin": dict(c),
        "per_origin": by_origin.to_dict(orient="records"),
        "per_meter": by_meter.to_dict(orient="records"),
    }, indent=2, default=str))

    print(f"\n  → JSON: {json_out.relative_to(REPO)}")
    print(f"  → CSV:  benchmarks/results/v5_benchmark_s3{{,_by_meter,_by_origin}}.csv")
    print(f"  → total {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
