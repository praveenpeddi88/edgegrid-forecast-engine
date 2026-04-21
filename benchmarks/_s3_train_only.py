"""Train ONE meter bundle for a given origin. Writes models/v4_s3/<iso>/<msn>.joblib."""
from __future__ import annotations
import sys, time, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"src")); sys.path.insert(0, str(REPO/"benchmarks"))
import pandas as pd, joblib
from edgegrid_forecast.inference._features import load_meter_data, fetch_weather_expanded, compute_fleet_aggregate
import v5_retrain_one as vr

origin_iso, msn = sys.argv[1], sys.argv[2]
origin = pd.Timestamp(origin_iso); cutoff = origin - pd.Timedelta(minutes=30)
mdir = REPO/"models"/"v4_s3"/origin.strftime("%Y-%m-%dT%H%M"); mdir.mkdir(parents=True, exist_ok=True)

t0=time.time()
all_df,_ = load_meter_data(); wx = fetch_weather_expanded()
fleet = compute_fleet_aggregate(all_df[all_df["ts"]<=cutoff])
t_load=time.time()-t0

t0=time.time()
bundle,_ = vr.retrain_meter(msn, all_df, wx, fleet, train_cutoff=cutoff)
joblib.dump(bundle, mdir/f"{msn}.joblib", compress=3)
print(json.dumps({"msn":msn,"train_s":round(time.time()-t0,2),"load_s":round(t_load,2),"hold_mape":float(bundle['holdout_metrics']['mape'])}))
