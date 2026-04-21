"""3-meter smoke for S3 wiring; writes /tmp/s3_smoke_done.json on success."""
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
from edgegrid_forecast.accuracy.block_accuracy import (
    per_meter_block_mape, cohort_rollup, load_oracle_cohort_map
)
import v5_retrain_one as vr

LOG = Path("/tmp/s3_smoke.log")
DONE = Path("/tmp/s3_smoke_done.json")

def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)

try:
    LOG.write_text("")
    if DONE.exists(): DONE.unlink()

    origin = pd.Timestamp("2026-01-29")
    cutoff = origin - pd.Timedelta(minutes=30)
    horizon_end = origin + pd.Timedelta(days=7)

    t0 = time.time()
    all_df, prof = load_meter_data()
    wx = fetch_weather_expanded()
    fleet = compute_fleet_aggregate(all_df[all_df["ts"] <= cutoff])
    t_load = time.time() - t0
    log(f"load {t_load:.1f}s n_rows={len(all_df):,}")

    meters = sorted([p.stem for p in (REPO / "models" / "v4").glob("*.joblib")])[:3]
    log(f"smoke meters: {meters}")
    mdir = REPO / "models" / "v4_s3_smoke"
    mdir.mkdir(parents=True, exist_ok=True)

    timings = {"train": {}, "forecast": {}}
    for m in meters:
        ts0 = time.time()
        bundle, _ = vr.retrain_meter(m, all_df, wx, fleet, train_cutoff=cutoff)
        joblib.dump(bundle, mdir/f"{m}.joblib", compress=3)
        timings["train"][m] = round(time.time() - ts0, 1)
        log(f"  {m} train {timings['train'][m]}s hold_mape={bundle['holdout_metrics']['mape']:.2f}")

        ts0 = time.time()
        md = all_df[all_df["msn"] == m].sort_values("ts").reset_index(drop=True)
        ctx = md[md["ts"] < origin]
        horizon_ts = pd.date_range(origin, horizon_end, freq="30min", inclusive="left")
        fc = predict_recursive(m, ctx, horizon_ts, weather_df=wx, fleet_df=fleet, models_dir=mdir)
        timings["forecast"][m] = round(time.time() - ts0, 1)
        log(f"  {m} fc {timings['forecast'][m]}s n={len(fc)} mean_pred_wh={float(fc['forecast_wh'].mean()):.1f}")

    DONE.write_text(json.dumps({"ok": True, "timings": timings, "load_s": round(t_load, 1)}, indent=2))
    log("DONE OK")
except Exception as e:
    log("FAIL: " + repr(e))
    log(traceback.format_exc())
    DONE.write_text(json.dumps({"ok": False, "err": repr(e)}, indent=2))
