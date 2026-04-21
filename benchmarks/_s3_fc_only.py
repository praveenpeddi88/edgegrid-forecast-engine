"""Recursive 7-day forecast for ONE meter, against bundles in models/v4_s3/<iso>/."""
from __future__ import annotations
import sys, time, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"src")); sys.path.insert(0, str(REPO/"benchmarks"))
import pandas as pd
from edgegrid_forecast.inference._features import load_meter_data, fetch_weather_expanded, compute_fleet_aggregate
from edgegrid_forecast.inference.v5_predict import predict_recursive

origin_iso, msn = sys.argv[1], sys.argv[2]
origin = pd.Timestamp(origin_iso); cutoff = origin - pd.Timedelta(minutes=30)
horizon_end = origin + pd.Timedelta(days=7)
out_root = REPO/"outputs"/"s3"/origin.strftime("%Y-%m-%dT%H%M")
fc_dir = out_root/"forecasts"; fc_dir.mkdir(parents=True, exist_ok=True)
mdir = REPO/"models"/"v4_s3"/origin.strftime("%Y-%m-%dT%H%M")

t0=time.time()
all_df,_ = load_meter_data(); wx = fetch_weather_expanded()
fleet = compute_fleet_aggregate(all_df[all_df["ts"]<=cutoff])
t_load=time.time()-t0

t0=time.time()
md = all_df[all_df["msn"]==msn].sort_values("ts").reset_index(drop=True)
ctx = md[md["ts"] < origin]
horizon_ts = pd.date_range(origin, horizon_end, freq="30min", inclusive="left")
res = predict_recursive(msn, ctx, horizon_ts, weather_df=wx, fleet_df=fleet, models_dir=mdir)

fc = res.reset_index().rename(columns={"forecast_wh":"predicted_wh","confidence_low":"q10_wh","confidence_high":"q90_wh"})
fc["meter_id"] = str(msn)
fc["predicted_kwh"] = fc["predicted_wh"].astype(float)/1000.0
fc["q10_kwh"] = fc["q10_wh"].astype(float)/1000.0
fc["q90_kwh"] = fc["q90_wh"].astype(float)/1000.0
fc["horizon_step"] = list(range(1, len(fc)+1))
fc[["meter_id","ts","predicted_kwh","q10_kwh","q90_kwh","horizon_step"]].to_parquet(fc_dir/f"{msn}.parquet", index=False)
print(json.dumps({"msn":msn,"forecast_s":round(time.time()-t0,2),"load_s":round(t_load,2),"n":len(fc)}))
