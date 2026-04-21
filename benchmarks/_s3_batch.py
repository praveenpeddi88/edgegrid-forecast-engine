"""Process as many meters as will fit in a ~40s budget; resumable.

- Loads data + weather + fleet ONCE (~0.5 s).
- Iterates meters in the canonical order, SKIPS those already having a
  forecast parquet written. Trains + forecasts each. Stops when the wall
  clock exceeds the budget.

Usage:
  python3 benchmarks/_s3_batch.py <origin_iso> [budget_s=38]
"""
from __future__ import annotations
import sys, time, json
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


def main(origin_iso: str, budget_s: float = 38.0):
    origin = pd.Timestamp(origin_iso)
    cutoff = origin - pd.Timedelta(minutes=30)
    horizon_end = origin + pd.Timedelta(days=7)

    out_root = REPO / "outputs" / "s3" / origin.strftime("%Y-%m-%dT%H%M")
    fc_dir = out_root / "forecasts"; fc_dir.mkdir(parents=True, exist_ok=True)
    timing_dir = out_root / "_timing"; timing_dir.mkdir(parents=True, exist_ok=True)
    mdir = REPO / "models" / "v4_s3" / origin.strftime("%Y-%m-%dT%H%M")
    mdir.mkdir(parents=True, exist_ok=True)

    meters = sorted([p.stem for p in (REPO / "models" / "v4").glob("*.joblib")])
    todo = [m for m in meters if not (fc_dir / f"{m}.parquet").exists()]
    print(f"meters: total={len(meters)} done={len(meters)-len(todo)} todo={len(todo)}", flush=True)
    if not todo:
        print("ALL DONE"); return

    t_total0 = time.time()
    all_df, _ = load_meter_data()
    wx = fetch_weather_expanded()
    fleet = compute_fleet_aggregate(all_df[all_df["ts"] <= cutoff])
    print(f"load {time.time()-t_total0:.1f}s", flush=True)

    processed = 0
    for msn in todo:
        if time.time() - t_total0 > budget_s:
            print(f"BUDGET REACHED @ {time.time()-t_total0:.1f}s", flush=True)
            break
        ts0 = time.time()
        try:
            bundle, _ = vr.retrain_meter(msn, all_df, wx, fleet, train_cutoff=cutoff)
            joblib.dump(bundle, mdir / f"{msn}.joblib", compress=3)
            dt_tr = time.time() - ts0

            ts1 = time.time()
            md = all_df[all_df["msn"] == msn].sort_values("ts").reset_index(drop=True)
            ctx = md[md["ts"] < origin]
            horizon_ts = pd.date_range(origin, horizon_end, freq="30min", inclusive="left")
            res = predict_recursive(msn, ctx, horizon_ts,
                                    weather_df=wx, fleet_df=fleet, models_dir=mdir)
            dt_fc = time.time() - ts1

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

            rec = {"msn": msn, "origin": origin.isoformat(),
                   "train_s": round(dt_tr,2), "forecast_s": round(dt_fc,2),
                   "bundle_holdout_mape": float(bundle["holdout_metrics"]["mape"])}
            (timing_dir / f"{msn}.json").write_text(json.dumps(rec, indent=2))
            processed += 1
            print(f"  {msn} tr={dt_tr:.1f}s fc={dt_fc:.1f}s  cum={time.time()-t_total0:.1f}s", flush=True)
        except Exception as e:
            print(f"  {msn} FAIL: {e!r}", flush=True)
            (timing_dir / f"{msn}.json").write_text(json.dumps({"msn": msn, "error": repr(e)}))

    print(f"BATCH DONE processed={processed} elapsed={time.time()-t_total0:.1f}s", flush=True)


if __name__ == "__main__":
    origin_iso = sys.argv[1]
    budget_s = float(sys.argv[2]) if len(sys.argv) > 2 else 38.0
    main(origin_iso, budget_s)
