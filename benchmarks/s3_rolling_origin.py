"""
S3 Rolling-Origin Validation — single-origin runnable.

Per protocol (docs/S3_ROLLING_ORIGIN_PROTOCOL.md), this script does ONE origin
end-to-end so we can validate wiring + collect honest timing before
committing to the 10-origin overnight sweep.

For each --origin t:
  1. Truncate training data to t - 30 min.
  2. Re-fit (full retrain — no leaf-only refit exists in the repo) the 42
     v4 quantile bundles on that truncated data. Save under
     models/v4_s3/<origin_iso>/  (production models/v4/ is NEVER touched).
  3. Forecast next 7 days (336 × 30-min slots) per meter using the v5
     recursive path.
  4. Load actuals for [t, t + 7d) from data/raw/{sp,tp}_data.parquet, join.
  5. Compute per-meter per-horizon-step MAPE/MBE/coverage.
  6. Compute cohort rollups at the 4 cadences (15-min DAM, 30-min native,
     hourly, ToD-4h) using block_accuracy.cohort_rollup.
  7. Persist parquet + csv outputs and print wall-clock timing.

Usage:
  python3 benchmarks/s3_rolling_origin.py --origin 2026-01-29 --mode full_retrain

  --mode refit accepted but maps to full_retrain (no LightGBM .refit() path
  is wired in this codebase yet). The protocol's "refit-only" assumption is
  flagged in the dry-run report.

Outputs (under outputs/s3/<origin_iso>/):
  forecast.parquet         per-meter 30-min forecast for the 7-day horizon
  metrics_long.parquet     per-(meter, ts) row with mape/mbe/coverage etc.
  cohort_summary.csv       cohort_rollup at the 4 cadences
  per_meter_summary.csv    per-meter 7-day rollups (sortable for offenders)
  timing.json              wall-clock for retrain / forecast / metrics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks"))

from edgegrid_forecast.accuracy.block_accuracy import (  # noqa: E402
    BLOCK_SPECS,
    cohort_rollup,
    load_oracle_cohort_map,
    per_meter_block_mape,
)
from edgegrid_forecast.inference._features import (  # noqa: E402
    compute_fleet_aggregate,
    fetch_weather_expanded,
    load_meter_data,
)
from edgegrid_forecast.inference.v5_predict import predict_recursive  # noqa: E402

# v5_retrain_one.retrain_meter is the closest-to-canonical "fit one bundle
# given a train cutoff" function in the repo. We import + reuse it, but
# point its output dir at models/v4_s3/<origin>/ via direct call.
import v5_retrain_one as v5_retrain  # noqa: E402

CADENCES = ["dam_15min", "native_30min", "hourly", "tod_4h"]
HORIZON_DAYS = 7
HORIZON_STEPS = HORIZON_DAYS * 48  # 336


def _retrain_one_bundle(msn, all_df, weather_df, fleet_df, train_cutoff, out_dir):
    """Wrap v5_retrain.retrain_meter so the bundle is saved into out_dir
    instead of the default models/v5/."""
    import joblib

    bundle, _ = v5_retrain.retrain_meter(
        msn, all_df, weather_df, fleet_df, train_cutoff=train_cutoff
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{msn}.joblib"
    joblib.dump(bundle, path, compress=3)
    return bundle, path


def _forecast_one(msn, all_df, weather_df, fleet_df, models_dir, origin, horizon_end):
    """Recursive 7-day forecast for one meter."""
    md = all_df[all_df["msn"] == msn].sort_values("ts").reset_index(drop=True)
    ctx = md[md["ts"] < origin].copy()
    if ctx.empty:
        return None
    horizon_ts = pd.date_range(origin, horizon_end, freq="30min", inclusive="left")
    res = predict_recursive(
        msn,
        ctx,
        horizon_ts,
        weather_df=weather_df,
        fleet_df=fleet_df,
        models_dir=models_dir,
    )
    out = res.reset_index().rename(
        columns={
            "forecast_wh": "predicted_wh",
            "confidence_low": "q10_wh",
            "confidence_high": "q90_wh",
        }
    )
    out["meter_id"] = str(msn)
    out["predicted_kwh"] = out["predicted_wh"].astype(float) / 1000.0
    out["q10_kwh"] = out["q10_wh"].astype(float) / 1000.0
    out["q90_kwh"] = out["q90_wh"].astype(float) / 1000.0
    out["horizon_step"] = np.arange(1, len(out) + 1, dtype=np.int32)
    return out[
        [
            "meter_id",
            "ts",
            "predicted_kwh",
            "q10_kwh",
            "q90_kwh",
            "horizon_step",
        ]
    ]


def _build_actuals(all_df, origin, horizon_end):
    a = all_df[(all_df["ts"] >= origin) & (all_df["ts"] < horizon_end)].copy()
    a = a.rename(columns={"msn": "meter_id"})
    a["meter_id"] = a["meter_id"].astype(str)
    a["actual_kwh"] = a["demand_wh"].astype(float) / 1000.0
    return a[["meter_id", "ts", "actual_kwh"]]


def _per_meter_horizon_metrics(forecasts, actuals, origin):
    """Long row per (meter, ts) with mape/mbe/coverage + horizon_step."""
    df = forecasts.merge(actuals, on=["meter_id", "ts"], how="inner")
    EPS = 0.0005
    pred = df["predicted_kwh"].to_numpy(dtype=float)
    act = df["actual_kwh"].to_numpy(dtype=float)
    err = pred - act
    df["mbe_kwh"] = err
    df["abs_err_kwh"] = np.abs(err)
    df["mape"] = 100.0 * np.where(act >= EPS, np.abs(err) / np.maximum(act, EPS), np.nan)
    df["in_q10_q90"] = (
        (df["actual_kwh"] >= df["q10_kwh"])
        & (df["actual_kwh"] <= df["q90_kwh"])
    ).astype(np.int8)
    df["origin"] = pd.Timestamp(origin)
    return df[
        [
            "meter_id",
            "ts",
            "origin",
            "horizon_step",
            "mape",
            "mbe_kwh",
            "in_q10_q90",
            "predicted_kwh",
            "actual_kwh",
            "q10_kwh",
            "q90_kwh",
        ]
    ]


def _per_meter_summary(metrics_long, cohort_map):
    g = (
        metrics_long.dropna(subset=["mape"])
        .groupby("meter_id")
        .agg(
            mape_7d=("mape", "mean"),
            mape_p95=("mape", lambda s: float(np.percentile(s, 95))),
            mbe_kwh_mean=("mbe_kwh", "mean"),
            cov_q10_q90=("in_q10_q90", "mean"),
            n=("mape", "size"),
        )
        .reset_index()
    )
    g["cohort"] = g["meter_id"].map(cohort_map).fillna("?")
    g = g.sort_values("mape_7d", ascending=False).reset_index(drop=True)
    return g


def run_origin(origin: pd.Timestamp, mode: str):
    origin_iso = origin.strftime("%Y-%m-%dT%H%M")
    out_dir = REPO / "outputs" / "s3" / origin_iso
    models_dir = REPO / "models" / "v4_s3" / origin_iso
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Translate "refit" -> full_retrain (no leaf-only path exists)
    effective_mode = "full_retrain"
    if mode == "refit":
        print(
            "[s3] WARNING: requested --mode refit but no LightGBM .refit() path "
            "wired in repo. Falling back to full_retrain.",
            flush=True,
        )

    train_cutoff = origin - pd.Timedelta(minutes=30)
    horizon_end = origin + pd.Timedelta(days=HORIZON_DAYS)
    print(f"[s3] origin={origin}  train_cutoff={train_cutoff}  horizon_end={horizon_end}")
    print(f"[s3] mode={effective_mode}")
    print(f"[s3] outputs   -> {out_dir}")
    print(f"[s3] s3 models -> {models_dir}")

    print("[s3] loading data + weather + fleet aggregate ...", flush=True)
    t0 = time.time()
    all_df, profile = load_meter_data()
    weather_df = fetch_weather_expanded()
    # Fleet aggregate must also be limited to <= train_cutoff to avoid leakage
    fleet_df = compute_fleet_aggregate(all_df[all_df["ts"] <= train_cutoff])
    t_load = time.time() - t0
    print(f"[s3]   data ready in {t_load:.1f}s  (n_rows={len(all_df):,})", flush=True)

    meters = sorted([p.stem for p in (REPO / "models" / "v4").glob("*.joblib")])
    print(f"[s3] meters to fit: {len(meters)}", flush=True)

    # ───────── Re-fit ─────────
    retrain_times = {}
    failed_train = []
    t0 = time.time()
    for i, msn in enumerate(meters, 1):
        ts0 = time.time()
        try:
            _retrain_one_bundle(
                msn, all_df, weather_df, fleet_df, train_cutoff, models_dir
            )
        except Exception as e:
            print(f"[s3]   [{i:>2}/{len(meters)}] {msn} TRAIN FAIL: {e}", flush=True)
            failed_train.append((msn, str(e)))
            continue
        dt = time.time() - ts0
        retrain_times[msn] = dt
        if i % 5 == 0 or i == len(meters):
            print(
                f"[s3]   [{i:>2}/{len(meters)}] {msn} fit in {dt:.1f}s "
                f"(cum={time.time() - t0:.1f}s)",
                flush=True,
            )
    t_retrain = time.time() - t0
    print(f"[s3] retrain total: {t_retrain:.1f}s ({len(meters) - len(failed_train)} ok)")

    # ───────── Forecast ─────────
    print("[s3] forecasting recursive 7d for each meter ...", flush=True)
    t0 = time.time()
    forecasts = []
    fc_times = {}
    failed_fc = []
    for i, msn in enumerate(meters, 1):
        if not (models_dir / f"{msn}.joblib").exists():
            failed_fc.append((msn, "no bundle"))
            continue
        ts0 = time.time()
        try:
            fc = _forecast_one(
                msn, all_df, weather_df, fleet_df, models_dir, origin, horizon_end
            )
        except Exception as e:
            print(f"[s3]   [{i:>2}/{len(meters)}] {msn} FC FAIL: {e}", flush=True)
            failed_fc.append((msn, str(e)))
            continue
        if fc is None or fc.empty:
            failed_fc.append((msn, "empty forecast"))
            continue
        forecasts.append(fc)
        dt = time.time() - ts0
        fc_times[msn] = dt
        if i % 5 == 0 or i == len(meters):
            print(
                f"[s3]   [{i:>2}/{len(meters)}] {msn} fc in {dt:.1f}s "
                f"(cum={time.time() - t0:.1f}s)",
                flush=True,
            )
    t_forecast = time.time() - t0
    print(f"[s3] forecast total: {t_forecast:.1f}s ({len(forecasts)} ok)")

    if not forecasts:
        raise RuntimeError("no forecasts produced; aborting metrics")

    forecast_df = pd.concat(forecasts, ignore_index=True)
    forecast_path = out_dir / "forecast.parquet"
    forecast_df.to_parquet(forecast_path, index=False)
    print(f"[s3] wrote {forecast_path}  ({len(forecast_df):,} rows)")

    # ───────── Metrics ─────────
    print("[s3] computing metrics ...", flush=True)
    t0 = time.time()
    actuals_df = _build_actuals(all_df, origin, horizon_end)
    metrics_long = _per_meter_horizon_metrics(forecast_df, actuals_df, origin)
    metrics_path = out_dir / "metrics_long.parquet"
    metrics_long.to_parquet(metrics_path, index=False)
    print(f"[s3] wrote {metrics_path}  ({len(metrics_long):,} rows)")

    # Cohort + cadence rollups via block_accuracy
    cohort_map = load_oracle_cohort_map()
    print(f"[s3] cohort_map size: {len(cohort_map)}", flush=True)

    rollups = []
    for cadence in CADENCES:
        bm = per_meter_block_mape(forecast_df, actuals_df, cadence)
        if bm.empty:
            continue
        roll = cohort_rollup(bm, cohort_map)
        if not roll.empty:
            roll["cadence"] = cadence
            rollups.append(roll)
    cohort_summary = (
        pd.concat(rollups, ignore_index=True) if rollups else pd.DataFrame()
    )
    cohort_path = out_dir / "cohort_summary.csv"
    cohort_summary.to_csv(cohort_path, index=False)
    print(f"[s3] wrote {cohort_path}")

    # Per-meter summary (handy for picking offenders)
    per_meter = _per_meter_summary(metrics_long, cohort_map)
    per_meter_path = out_dir / "per_meter_summary.csv"
    per_meter.to_csv(per_meter_path, index=False)
    print(f"[s3] wrote {per_meter_path}")
    t_metrics = time.time() - t0
    print(f"[s3] metrics total: {t_metrics:.1f}s")

    # ───────── Timing ─────────
    timing = {
        "origin": origin.isoformat(),
        "mode_requested": mode,
        "mode_effective": effective_mode,
        "n_meters_total": len(meters),
        "n_train_ok": len(meters) - len(failed_train),
        "n_forecast_ok": len(forecasts),
        "failed_train": failed_train,
        "failed_forecast": failed_fc,
        "wall_clock_s": {
            "data_load": round(t_load, 1),
            "retrain_total": round(t_retrain, 1),
            "retrain_per_meter_mean": round(
                sum(retrain_times.values()) / max(len(retrain_times), 1), 1
            ),
            "forecast_total": round(t_forecast, 1),
            "forecast_per_meter_mean": round(
                sum(fc_times.values()) / max(len(fc_times), 1), 1
            ),
            "metrics_total": round(t_metrics, 1),
            "grand_total": round(t_load + t_retrain + t_forecast + t_metrics, 1),
        },
    }
    timing_path = out_dir / "timing.json"
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2, default=str)
    print(f"[s3] wrote {timing_path}")

    # Console digest
    print("\n[s3] ─── Top-3 offenders by 7-day MAPE ───")
    for _, r in per_meter.head(3).iterrows():
        print(
            f"   meter={r['meter_id']}  cohort={r['cohort']}  "
            f"mape_7d={r['mape_7d']:.2f}%  p95={r['mape_p95']:.2f}%  "
            f"cov={r['cov_q10_q90']*100:.0f}%"
        )

    print("\n[s3] ─── Cohort summary (native_30min) ───")
    if not cohort_summary.empty:
        sub = cohort_summary[cohort_summary["cadence"] == "native_30min"]
        if not sub.empty:
            for _, r in sub.iterrows():
                print(
                    f"   cohort={r['cohort']}  median={r['median_mape']:.2f}%  "
                    f"p95={r['p95_mape']:.2f}%  meters={int(r['n_meters'])}"
                )
    return timing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--origin", required=True, help="YYYY-MM-DD (00:00 local).")
    p.add_argument(
        "--mode",
        choices=["refit", "full_retrain"],
        default="refit",
        help="refit falls back to full_retrain (no leaf-only path wired).",
    )
    args = p.parse_args()
    origin = pd.Timestamp(args.origin)
    run_origin(origin, args.mode)


if __name__ == "__main__":
    main()
