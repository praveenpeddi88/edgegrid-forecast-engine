"""Train + recursive-forecast ONE meter for the S3 dry-run origin.

Designed for the 45s sandbox-bash budget: each invocation handles a single
meter so the overall sweep can be chunked across many bash calls. State is
persisted to disk; nothing in memory carries between calls.

Inputs are loaded fresh each call (data loaders are cheap, ~1-2 s) but the
re-fitted bundle is written to models/v4_s3/<origin_iso>/<msn>.joblib and
the per-meter forecast to outputs/s3/<origin_iso>/forecasts/<msn>.parquet.

Usage:
  python3 benchmarks/_s3_one_meter.py <origin_iso> <msn>
"""
from __future__ import annotations
import sys, time, json, traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks"))

import pandas as pd
import joblib
from edgegrid_forecast.inference._features import (
    load_meter_data, fetch_weather_expanded, compute_fleet_aggregate
)
from edgegrid_forecast.inference.v5_predict import predict_recursive
import v5_retrain_one as vr


def main(origin_iso: str, msn: str):
    origin = pd.Timestamp(origin_iso)
    cutoff = origin - pd.Timedelta(minutes=30)
    horizon_end = origin + pd.Timedelta(days=7)

    out_root = REPO / "outputs" / "s3" / origin.strftime("%Y-%m-%dT%H%M")
    fc_dir = out_root / "forecasts"
    fc_dir.mkdir(parents=True, exist_ok=True)
    mdir = REPO / "models" / "v4_s3" / origin.strftime("%Y-%m-%dT%H%M")
    mdir.mkdir(parents=True, exist_ok=True)
    timing_dir = out_root / "_timing"
    timing_dir.mkdir(parents=True, exist_ok=True)

    t_total0 = time.time()
    t0 = time.time()
    all_df, _ = load_meter_data()
    wx = fetch_weather_expanded()
    fleet = compute_fleet_aggregate(all_df[all_df["ts"] <= cutoff])
    t_load = time.time() - t0

    # Train (full retrain — repo has no leaf-only refit path).
    t0 = time.time()
    bundle, _ = vr.retrain_meter(msn, all_df, wx, fleet, train_cutoff=cutoff)
    bundle_path = mdir / f"{msn}.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    t_train = time.time() - t0

    # Recursive forecast for 7 days (336 × 30-min slots).
    t0 = time.time()
    md = all_df[all_df["msn"] == msn].sort_values("ts").reset_index(drop=True)
    ctx = md[md["ts"] < origin]
    horizon_ts = pd.date_range(origin, horizon_end, freq="30min", inclusive="left")
    res = predict_recursive(
        msn, ctx, horizon_ts,
        weather_df=wx, fleet_df=fleet, models_dir=mdir,
    )
    t_fc = time.time() - t0

    fc = res.reset_index().rename(columns={
        "forecast_wh": "predicted_wh",
        "confidence_low": "q10_wh",
        "confidence_high": "q90_wh",
    })
    fc["meter_id"] = str(msn)
    fc["predicted_kwh"] = fc["predicted_wh"].astype(float) / 1000.0
    fc["q10_kwh"] = fc["q10_wh"].astype(float) / 1000.0
    fc["q90_kwh"] = fc["q90_wh"].astype(float) / 1000.0
    fc["horizon_step"] = list(range(1, len(fc) + 1))
    fc[["meter_id","ts","predicted_kwh","q10_kwh","q90_kwh","horizon_step"]] \
        .to_parquet(fc_dir / f"{msn}.parquet", index=False)

    rec = {
        "msn": msn,
        "origin": origin.isoformat(),
        "load_s": round(t_load, 2),
        "train_s": round(t_train, 2),
        "forecast_s": round(t_fc, 2),
        "total_s": round(time.time() - t_total0, 2),
        "bundle_holdout_mape": float(bundle["holdout_metrics"]["mape"]),
        "n_horizon": len(fc),
    }
    (timing_dir / f"{msn}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec))


if __name__ == "__main__":
    try:
        origin_iso, msn = sys.argv[1], sys.argv[2]
        main(origin_iso, msn)
    except Exception as e:
        sys.stderr.write("FAIL " + repr(e) + "\n")
        sys.stderr.write(traceback.format_exc())
        sys.exit(1)
