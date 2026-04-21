"""Aggregate per-meter forecasts under outputs/s3/<iso>/forecasts/ into:
  - forecast.parquet (concat of all per-meter forecasts)
  - metrics_long.parquet (per-(meter,ts) row with mape/mbe/coverage)
  - cohort_summary.csv (cohort_rollup at 4 cadences)
  - per_meter_summary.csv
  - timing.json (collated wall-clock per stage)
"""
from __future__ import annotations
import sys, time, json, glob
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"src"))
import numpy as np, pandas as pd
from edgegrid_forecast.accuracy.block_accuracy import (
    per_meter_block_mape, cohort_rollup, load_oracle_cohort_map
)

origin_iso = sys.argv[1]
origin = pd.Timestamp(origin_iso)
horizon_end = origin + pd.Timedelta(days=7)
out_root = REPO/"outputs"/"s3"/origin.strftime("%Y-%m-%dT%H%M")
fc_dir = out_root/"forecasts"
timing_dir = out_root/"_timing"

# Concat forecasts
parts = []
for p in sorted(fc_dir.glob("*.parquet")):
    parts.append(pd.read_parquet(p))
forecast_df = pd.concat(parts, ignore_index=True)
forecast_df["meter_id"] = forecast_df["meter_id"].astype(str)
forecast_df.to_parquet(out_root/"forecast.parquet", index=False)
print(f"forecast.parquet rows={len(forecast_df):,} meters={forecast_df['meter_id'].nunique()}")

# Build actuals
import importlib
sys.path.insert(0, str(REPO/"src"))
from edgegrid_forecast.inference._features import load_meter_data
all_df, _ = load_meter_data()
actuals_df = all_df[(all_df["ts"]>=origin)&(all_df["ts"]<horizon_end)].copy()
actuals_df = actuals_df.rename(columns={"msn":"meter_id"})
actuals_df["meter_id"] = actuals_df["meter_id"].astype(str)
actuals_df["actual_kwh"] = actuals_df["demand_wh"].astype(float)/1000.0
actuals_df = actuals_df[["meter_id","ts","actual_kwh"]]
print(f"actuals rows={len(actuals_df):,}")

# Per-(meter,ts) long metrics
EPS = 0.0005
m = forecast_df.merge(actuals_df, on=["meter_id","ts"], how="inner")
pred = m["predicted_kwh"].to_numpy(float)
act = m["actual_kwh"].to_numpy(float)
err = pred - act
m["mbe_kwh"] = err
m["abs_err_kwh"] = np.abs(err)
m["mape"] = 100.0 * np.where(act>=EPS, np.abs(err)/np.maximum(act, EPS), np.nan)
m["in_q10_q90"] = ((m["actual_kwh"]>=m["q10_kwh"]) & (m["actual_kwh"]<=m["q90_kwh"])).astype(np.int8)
m["origin"] = origin
metrics_cols = ["meter_id","ts","origin","horizon_step","mape","mbe_kwh","in_q10_q90","predicted_kwh","actual_kwh","q10_kwh","q90_kwh"]
m[metrics_cols].to_parquet(out_root/"metrics_long.parquet", index=False)
print(f"metrics_long.parquet rows={len(m):,}")

# Per-meter summary
cohort_map = load_oracle_cohort_map()
gg = (m.dropna(subset=["mape"])
        .groupby("meter_id")
        .agg(mape_7d=("mape","mean"),
             mape_p95=("mape", lambda s: float(np.percentile(s, 95))),
             mbe_kwh_mean=("mbe_kwh","mean"),
             cov_q10_q90=("in_q10_q90","mean"),
             n=("mape","size"))
        .reset_index())
gg["cohort"] = gg["meter_id"].map(cohort_map).fillna("?")
gg = gg.sort_values("mape_7d", ascending=False).reset_index(drop=True)
gg.to_csv(out_root/"per_meter_summary.csv", index=False)
print(f"per_meter_summary.csv rows={len(gg)}")

# Cohort cadence rollups
rollups = []
for cadence in ["dam_15min","native_30min","hourly","tod_4h"]:
    bm = per_meter_block_mape(forecast_df, actuals_df, cadence)
    if bm.empty: continue
    roll = cohort_rollup(bm, cohort_map)
    if not roll.empty:
        roll["cadence_label"] = cadence
        rollups.append(roll)
cohort_summary = pd.concat(rollups, ignore_index=True) if rollups else pd.DataFrame()
cohort_summary.to_csv(out_root/"cohort_summary.csv", index=False)
print(f"cohort_summary.csv rows={len(cohort_summary)}")

# Collate timing
timings = []
for p in sorted(timing_dir.glob("*.json")):
    timings.append(json.loads(p.read_text()))
train_total = sum(t.get("train_s", 0) for t in timings)
fc_total = sum(t.get("forecast_s", 0) for t in timings)
n_train = sum(1 for t in timings if "train_s" in t)
n_fc = sum(1 for t in timings if "forecast_s" in t)
total_summary = {
    "origin": origin.isoformat(),
    "n_meters_total": 42,
    "n_train_records": n_train,
    "n_forecast_records": n_fc,
    "wall_clock_s": {
        "retrain_total_summed": round(train_total, 1),
        "retrain_per_meter_mean": round(train_total/max(n_train,1), 2),
        "forecast_total_summed": round(fc_total, 1),
        "forecast_per_meter_mean": round(fc_total/max(n_fc,1), 2),
    },
    "extrapolation_10_origins_serial_s": round(10*(train_total + fc_total), 1),
    "extrapolation_10_origins_serial_hours": round(10*(train_total + fc_total)/3600.0, 2),
}
(out_root/"timing.json").write_text(json.dumps(total_summary, indent=2))
print(json.dumps(total_summary, indent=2))

# Print top-3 offenders
print("\n--- Top-3 offenders by 7-day MAPE ---")
print(gg.head(3).to_string(index=False))
print("\n--- Cohort summary (native_30min) ---")
sub = cohort_summary[cohort_summary["cadence_label"]=="native_30min"]
print(sub.to_string(index=False))
print("\n--- Cohort summary (dam_15min) ---")
sub = cohort_summary[cohort_summary["cadence_label"]=="dam_15min"]
print(sub.to_string(index=False))
