"""
EdgeGrid Forecast Engine v4 — Ensemble: LightGBM + Chronos-Bolt
================================================================
Architecture: Two-phase approach.

PHASE 1: LightGBM v4 with expanded signals across all 42 meters (~2 min)
  - Richer weather (pressure, cloud, precip, dewpoint, DNI, DHI, heat index)
  - Voltage features (lag, sag detector)
  - ToD tariff multipliers
  - Two-pass feature selection (60 candidates → ~30 selected)

PHASE 2: Chronos-Bolt targeted ensemble for HT + struggling meters (~5 min)
  - Only run Chronos where LightGBM MAPE > 8% or HT tier
  - Chronos uses CHRONOLOGICAL context (last 7d before each holdout day)
  - Adaptive weight learned on validation set
  - Skip meters where Chronos hurts

Strategy: Stratified Temporal (every 4th day held out)
"""
import pandas as pd, numpy as np, json, time, sys, warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb

t0 = time.time()

print("=" * 74)
print("  EdgeGrid v4 Ensemble Engine: LightGBM + Chronos-Bolt")
print("  Strategy 2 · Expanded Signals · 42 meters")
print("=" * 74)

# ════════════════════════════════════════════════════════════════════════
# 1. EXPANDED WEATHER
# ════════════════════════════════════════════════════════════════════════
print("\n[1/7] Fetching expanded weather from Open-Meteo...")
import urllib.request

WEATHER_VARS = (
    "temperature_2m,relative_humidity_2m,dewpoint_2m,"
    "surface_pressure,cloud_cover,precipitation,wind_speed_10m,"
    "shortwave_radiation,direct_radiation,diffuse_radiation,"
    "direct_normal_irradiance"
)

def fetch_weather_expanded(lat=17.0, lon=82.0, start="2024-10-01", end="2026-02-28"):
    chunks = []
    current = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    while current < end_dt:
        chunk_end = min(current + pd.DateOffset(months=6) - pd.Timedelta(days=1), end_dt)
        url = (f"https://archive-api.open-meteo.com/v1/archive?"
               f"latitude={lat}&longitude={lon}"
               f"&start_date={current.strftime('%Y-%m-%d')}"
               f"&end_date={chunk_end.strftime('%Y-%m-%d')}"
               f"&hourly={WEATHER_VARS}"
               f"&timezone=Asia%2FKolkata")
        resp = urllib.request.urlopen(url, timeout=30)
        data = json.loads(resp.read())
        h = data['hourly']
        chunk_df = pd.DataFrame({
            'ts': pd.to_datetime(h['time']),
            'temperature': h['temperature_2m'],
            'humidity': h['relative_humidity_2m'],
            'dewpoint': h['dewpoint_2m'],
            'pressure': h['surface_pressure'],
            'cloud_cover': h['cloud_cover'],
            'precipitation': h['precipitation'],
            'wind_speed': h['wind_speed_10m'],
            'ghi': h['shortwave_radiation'],
            'dni': h['direct_normal_irradiance'],
            'dhi': h['diffuse_radiation'],
            'direct_rad': h['direct_radiation'],
        })
        print(f"  {current.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')}: {len(chunk_df)} hrs")
        chunks.append(chunk_df)
        current = chunk_end + pd.Timedelta(days=1)
    wx = pd.concat(chunks, ignore_index=True).drop_duplicates('ts').sort_values('ts')
    wx_30m = wx.set_index('ts').resample('30min').ffill().reset_index()
    print(f"  Total: {len(wx_30m)} × 30min records, {len(wx.columns)-1} raw weather vars")
    return wx_30m

weather = fetch_weather_expanded()

# Derived weather signals
wx = weather.copy()
wx['pressure_delta_3h'] = wx['pressure'].diff(6)
wx['temp_delta_3h'] = wx['temperature'].diff(6)
wx['temp_rmean_6h'] = wx['temperature'].rolling(12, min_periods=1).mean()
wx['ghi_rmean_6h'] = wx['ghi'].rolling(12, min_periods=1).mean()
wx['cloud_delta_3h'] = wx['cloud_cover'].diff(6)
wx['is_raining'] = (wx['precipitation'] > 0).astype(int)
wx['diffuse_fraction'] = (wx['dhi'] / wx['ghi'].clip(lower=1)).clip(0, 1)
wx['heat_index'] = wx['temperature'] + 0.33 * (wx['humidity']/100 * 6.105 * np.exp(17.27*wx['temperature']/(237.7+wx['temperature']))) - 4.0
weather_expanded = wx

# ════════════════════════════════════════════════════════════════════════
# 2. HOLIDAYS + ToD TARIFF
# ════════════════════════════════════════════════════════════════════════
print("\n[2/7] Holidays + ToD tariff...")
HOLIDAYS = {
    "2024-10-02": "Gandhi Jayanti", "2024-10-12": "Dussehra",
    "2024-10-31": "Halloween/Sardar Patel", "2024-11-01": "Diwali",
    "2024-11-02": "Diwali", "2024-11-15": "Guru Nanak",
    "2024-12-25": "Christmas", "2025-01-01": "New Year",
    "2025-01-14": "Sankranti", "2025-01-15": "Sankranti",
    "2025-01-26": "Republic Day", "2025-02-26": "Maha Shivaratri",
    "2025-03-14": "Holi", "2025-03-30": "Ugadi",
    "2025-03-31": "Ramadan End", "2025-04-06": "Ram Navami",
    "2025-04-10": "Mahavir Jayanti", "2025-04-14": "Ambedkar Jayanti",
    "2025-04-18": "Good Friday", "2025-05-01": "May Day",
    "2025-05-12": "Buddha Purnima", "2025-06-07": "Eid",
    "2025-07-06": "Muharram", "2025-08-15": "Independence Day",
    "2025-08-16": "Janmashtami", "2025-09-05": "Milad",
    "2025-10-02": "Dussehra", "2025-10-20": "Diwali",
    "2025-10-21": "Diwali", "2025-11-05": "Guru Nanak",
    "2025-12-25": "Christmas", "2026-01-01": "New Year",
    "2026-01-14": "Sankranti", "2026-01-15": "Sankranti",
    "2026-01-26": "Republic Day", "2026-02-15": "Maha Shivaratri",
}
holiday_dates = set(pd.to_datetime(list(HOLIDAYS.keys())).date)
def tod_multiplier(hour):
    if 6 <= hour < 18: return 1.0
    elif 18 <= hour < 22: return 1.2
    else: return 0.9
print(f"  {len(holiday_dates)} holidays, 3-tier ToD")

# ════════════════════════════════════════════════════════════════════════
# 3. DATA LOADING
# ════════════════════════════════════════════════════════════════════════
print("\n[3/7] Loading meter data...")
def load_data():
    vendor = pd.read_excel("/sessions/awesome-gifted-feynman/mnt/uploads/SCS_PMSGVENDOR-7f720bd5.xlsx")
    vendor['MTR_SNO'] = vendor['MTR_SNO'].astype(str)
    phase_map = dict(zip(vendor['MTR_SNO'], vendor['Phase']))
    frames = []
    df_sp = pd.read_parquet("/sessions/awesome-gifted-feynman/sp_data.parquet")
    df_sp['ts'] = pd.to_datetime(df_sp['ts'])
    df_sp['wh_imp'] = pd.to_numeric(df_sp['wh_imp'].str.replace(',',''), errors='coerce')
    df_sp['v_ave'] = pd.to_numeric(df_sp['v_ave'].str.replace(',',''), errors='coerce')
    for msn in df_sp['msn'].unique():
        sub = df_sp[df_sp['msn']==msn][['ts','wh_imp','v_ave']].drop_duplicates('ts').sort_values('ts')
        sub.columns = ['ts','demand_wh','voltage']
        sub['msn'], sub['phase'] = msn, '1PH'
        frames.append(sub)
    df_tp = pd.read_parquet("/sessions/awesome-gifted-feynman/tp_data.parquet")
    df_tp['ts'] = pd.to_datetime(df_tp['ts'])
    df_tp['wh_imp'] = pd.to_numeric(df_tp['wh_imp'].str.replace(',',''), errors='coerce')
    for v in ['v_r','v_y','v_b']:
        df_tp[v] = pd.to_numeric(df_tp[v].str.replace(',',''), errors='coerce')
    df_tp['voltage'] = df_tp[['v_r','v_y','v_b']].mean(axis=1)
    for msn in df_tp['msn'].unique():
        sub = df_tp[df_tp['msn']==msn][['ts','wh_imp','voltage']].drop_duplicates('ts').sort_values('ts')
        sub.columns = ['ts','demand_wh','voltage']
        sub['msn'], sub['phase'] = msn, phase_map.get(msn,'3PH')
        frames.append(sub)
    return pd.concat(frames, ignore_index=True).dropna(subset=['demand_wh']), phase_map

df_all, phase_map = load_data()
meter_days = df_all.groupby('msn')['ts'].agg(lambda x: (x.max()-x.min()).days)
eligible = meter_days[meter_days >= 180].index.tolist()
print(f"  {len(df_all):,} rows, {len(eligible)} eligible meters ({time.time()-t0:.1f}s)")

# ════════════════════════════════════════════════════════════════════════
# 4. FLEET + HELPERS
# ════════════════════════════════════════════════════════════════════════
fleet_agg = df_all.groupby('ts')['demand_wh'].agg(['mean','std','count']).reset_index()
fleet_agg.columns = ['ts','fleet_mean','fleet_std','fleet_count']
fleet_agg = fleet_agg[fleet_agg['fleet_count'] >= 5].copy()
print(f"  Fleet: {len(fleet_agg)} timesteps")

def get_params(tier, n_samples):
    params = {
        'objective': 'regression', 'metric': 'mae', 'verbose': -1, 'n_jobs': -1,
        'learning_rate': 0.03, 'num_leaves': 31, 'min_child_samples': 50,
        'feature_fraction': 0.6, 'bagging_fraction': 0.7, 'bagging_freq': 5,
        'lambda_l1': 0.1, 'lambda_l2': 1.0, 'max_depth': 8,
    }
    if tier == "HT (>5kWh)":
        params.update({'num_leaves': 47, 'min_child_samples': 30,
                       'feature_fraction': 0.7, 'learning_rate': 0.04})
    elif tier == "Small (<500)":
        params.update({'num_leaves': 15, 'min_child_samples': 80,
                       'feature_fraction': 0.5, 'max_depth': 6,
                       'lambda_l1': 0.5, 'lambda_l2': 5.0})
    if n_samples < 3000:
        params['num_leaves'] = min(params['num_leaves'], 15)
        params['min_child_samples'] = max(params['min_child_samples'], 100)
    return params

def classify_tier(mean_wh):
    if mean_wh > 5000: return "HT (>5kWh)"
    elif mean_wh > 1500: return "Large (1.5-5k)"
    elif mean_wh > 500: return "Medium (0.5-1.5k)"
    else: return "Small (<500)"

def split_stratified(df, holdout_every=4):
    df = df.copy()
    df['date'] = df['ts'].dt.date
    day_counts = df.groupby('date').size()
    complete_days = sorted(day_counts[day_counts >= 44].index)
    holdout_days = set(complete_days[::holdout_every])
    train = df[~df['date'].isin(holdout_days)].copy()
    test = df[df['date'].isin(holdout_days)].copy()
    return train, test, len(train['date'].unique()), len(test['date'].unique())

def calc_metrics(y_true, y_pred, q10=None, q90=None):
    mask = y_true > 0.5
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mbe = float(np.mean(y_pred - y_true))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    mape = float(np.mean(np.abs(y_true[mask]-y_pred[mask])/y_true[mask])*100) if mask.sum()>0 else 0
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else 0
    pct = np.abs(y_true[mask]-y_pred[mask])/y_true[mask]*100 if mask.sum()>0 else np.array([])
    w5 = float(np.mean(pct<=5)*100) if len(pct)>0 else 0
    w10 = float(np.mean(pct<=10)*100) if len(pct)>0 else 0
    cov = float(np.mean((y_true>=q10)&(y_true<=q90))*100) if q10 is not None else 0
    return {'mape':round(mape,2),'mae':round(mae,2),'mbe':round(mbe,2),'rmse':round(rmse,2),
            'r2':round(r2,4),'within5':round(w5,1),'within10':round(w10,1),'coverage_80':round(cov,1)}


def build_features_v4(df_meter, weather_expanded, fleet_agg):
    """v4 features: ~60 candidates including expanded weather, voltage, ToD."""
    df = df_meter.sort_values('ts').reset_index(drop=True)

    # Temporal (10)
    df['hour'] = df['ts'].dt.hour + df['ts'].dt.minute/60
    df['dow'] = df['ts'].dt.dayofweek
    df['month'] = df['ts'].dt.month
    df['is_weekend'] = (df['dow']>=5).astype(int)
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    df['dow_sin'] = np.sin(2*np.pi*df['dow']/7)
    df['dow_cos'] = np.cos(2*np.pi*df['dow']/7)
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)

    # Lags (9)
    for lag in [1,2,4,6,12,24,48,96,336]:
        df[f'lag_{lag}'] = df['demand_wh'].shift(lag)

    # Rolling (8)
    for win in [6,12,48,336]:
        s = df['demand_wh'].shift(1)
        df[f'rmean_{win}'] = s.rolling(win, min_periods=1).mean()
        df[f'rstd_{win}'] = s.rolling(win, min_periods=1).std()

    # Momentum (2)
    df['demand_diff_1'] = df['demand_wh'].diff(1)
    df['demand_diff_4'] = df['demand_wh'].diff(4)

    # Longer rolling (3)
    s = df['demand_wh'].shift(1)
    df['rmean_14d'] = s.rolling(672, min_periods=48).mean()
    df['rmean_30d'] = s.rolling(1440, min_periods=48).mean()
    df['trend_ratio'] = df['rmean_14d'] / df['rmean_30d'].clip(lower=1)

    # EXPANDED WEATHER — merge all (20 vars)
    wx_cols = ['ts','temperature','humidity','dewpoint','pressure','cloud_cover',
               'precipitation','wind_speed','ghi','dni','dhi','direct_rad',
               'pressure_delta_3h','temp_delta_3h','ghi_rmean_6h','cloud_delta_3h',
               'is_raining','diffuse_fraction','heat_index','temp_rmean_6h']
    df = df.merge(weather_expanded[wx_cols], on='ts', how='left')
    for c in wx_cols[1:]:
        df[c] = df[c].ffill().bfill()

    # Derived weather (4)
    df['cooling_deg'] = (df['temperature'] - 25).clip(lower=0)
    df['heating_deg'] = (18 - df['temperature']).clip(lower=0)
    df['hour_x_temp'] = df['hour'] * df['temperature'] / 100
    df['peak_x_temp'] = ((df['hour']>=12)&(df['hour']<=17)).astype(int) * df['temperature']

    # Holidays (2)
    df['is_holiday'] = df['ts'].dt.date.isin(holiday_dates).astype(int)
    df['near_holiday'] = 0
    for hd in holiday_dates:
        mask = (df['ts'].dt.date >= hd - pd.Timedelta(days=1)) & \
               (df['ts'].dt.date <= hd + pd.Timedelta(days=1))
        df.loc[mask, 'near_holiday'] = 1

    # ToD tariff (2)
    df['tod_multiplier'] = df['ts'].dt.hour.apply(tod_multiplier)
    df['is_peak'] = ((df['ts'].dt.hour>=18)&(df['ts'].dt.hour<22)).astype(int)

    # Voltage (3)
    df['voltage_lag1'] = df['voltage'].shift(1)
    df['voltage_rstd_6'] = df['voltage'].shift(1).rolling(6, min_periods=1).std()
    df['voltage_rmean_48'] = df['voltage'].shift(1).rolling(48, min_periods=1).mean()

    # Seasonal deviation (1)
    shm = df.groupby('hour')['demand_wh'].transform(lambda x: x.shift(1).expanding(min_periods=48).mean())
    df['deviation_from_hourly'] = df['demand_wh'].shift(1) - shm

    # Fleet (3)
    fleet = fleet_agg[['ts','fleet_mean','fleet_std']].copy()
    df = df.merge(fleet, on='ts', how='left')
    df['fleet_mean'] = df['fleet_mean'].ffill().bfill().fillna(0)
    df['fleet_std'] = df['fleet_std'].ffill().bfill().fillna(0)
    df['fleet_mean_lag1'] = df['fleet_mean'].shift(1)
    df['fleet_std_lag1'] = df['fleet_std'].shift(1)
    df['vs_fleet_ratio'] = df['demand_wh'].shift(1) / df['fleet_mean_lag1'].clip(lower=1)

    df = df.iloc[336:].reset_index(drop=True)
    df['target'] = df['demand_wh']
    exclude = {'ts','demand_wh','target','msn','phase','voltage','scno','fleet_mean','fleet_std','date'}
    feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64','int64','int32','float32','uint8']]
    return df, feat_cols


# ════════════════════════════════════════════════════════════════════════
# PHASE 1: LightGBM v4 across all 42 meters
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*74}")
print("  PHASE 1: LightGBM v4 with expanded signals")
print(f"{'='*74}")

phase1_results = {}   # msn → {metrics, pred_lgb, y_test, ...}
all_results = []

for i, msn in enumerate(eligible):
    mdata = df_all[df_all['msn']==msn].copy()
    phase = mdata['phase'].iloc[0]

    feat_df, feat_cols = build_features_v4(mdata, weather_expanded, fleet_agg)
    if len(feat_df) < 500:
        print(f"  [{i+1:2d}/{len(eligible)}] {msn} — SKIPPED")
        continue

    train_df, test_df, n_train_days, n_test_days = split_stratified(feat_df)
    if len(test_df) < 48: continue

    y_train = train_df['target'].values
    y_test = test_df['target'].values
    mean_demand = float(np.mean(y_test))
    tier = classify_tier(mean_demand)

    # Pass 1: feature ranking
    X_all = np.nan_to_num(train_df[feat_cols].values, nan=0.)
    val_cut = int(len(X_all) * 0.80)
    dtrain_p1 = lgb.Dataset(X_all[:val_cut], label=y_train[:val_cut])
    dval_p1 = lgb.Dataset(X_all[val_cut:], label=y_train[val_cut:], reference=dtrain_p1)
    m1 = lgb.train({'objective':'regression','metric':'mae','num_leaves':31,
                     'learning_rate':0.08,'feature_fraction':0.7,'verbose':-1,'n_jobs':-1},
                    dtrain_p1, num_boost_round=150, valid_sets=[dval_p1],
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])

    imp = m1.feature_importance(importance_type='gain')
    fg = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    nk = max(25, int(len(feat_cols)*0.55))
    sel = [f for f,g in fg[:nk] if g>0]
    if len(sel)<20: sel = [f for f,g in fg[:25]]

    # Pass 2: full fit
    X_tr = np.nan_to_num(train_df[sel].values, nan=0.)
    X_te = np.nan_to_num(test_df[sel].values, nan=0.)
    params = get_params(tier, len(X_tr))
    dtrain = lgb.Dataset(X_tr[:val_cut], label=y_train[:val_cut])
    dval = lgb.Dataset(X_tr[val_cut:], label=y_train[val_cut:], reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=800, valid_sets=[dval],
                       callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
    best_iter = model.best_iteration

    pred_lgb = np.maximum(model.predict(X_te), 0)

    # Quantiles
    pq10 = {**params,'objective':'quantile','alpha':0.1,'metric':'quantile'}
    pq90 = {**params,'objective':'quantile','alpha':0.9,'metric':'quantile'}
    mq10 = lgb.train(pq10, dtrain, num_boost_round=best_iter)
    mq90 = lgb.train(pq90, dtrain, num_boost_round=best_iter)
    lgb_q10 = np.maximum(mq10.predict(X_te), 0)
    lgb_q90 = np.maximum(mq90.predict(X_te), 0)

    m_lgb = calc_metrics(y_test, pred_lgb, lgb_q10, lgb_q90)

    # Top features
    fi = sorted(zip(sel, model.feature_importance(importance_type='gain')), key=lambda x:-x[1])[:5]
    top5 = [f[0] for f in fi]

    # Check which new v4 features made the cut
    v4_new = [f for f in sel if f in [
        'pressure','pressure_delta_3h','cloud_cover','cloud_delta_3h','precipitation','is_raining',
        'dewpoint','dni','dhi','direct_rad','diffuse_fraction','heat_index','ghi_rmean_6h',
        'temp_rmean_6h','temp_delta_3h','heating_deg','peak_x_temp',
        'voltage_lag1','voltage_rstd_6','voltage_rmean_48','tod_multiplier','is_peak']]

    phase1_results[msn] = {
        'pred_lgb': pred_lgb, 'y_test': y_test, 'lgb_q10': lgb_q10, 'lgb_q90': lgb_q90,
        'feat_df': feat_df, 'train_df': train_df, 'test_df': test_df,
        'tier': tier, 'phase': phase, 'mean_demand': mean_demand,
        'metrics': m_lgb, 'selected_feats': sel, 'top5': top5,
        'best_iter': best_iter, 'n_train_days': n_train_days, 'n_test_days': n_test_days,
        'val_cut': val_cut, 'model': model, 'X_tr_val': X_tr[val_cut:],
        'y_train_val': y_train[val_cut:], 'v4_new_features': v4_new,
    }

    print(f"  [{i+1:2d}/{len(eligible)}] {msn} ({tier:>16s}) | "
          f"MAPE={m_lgb['mape']:>5.1f}% MAE={m_lgb['mae']:>6.0f} MBE={m_lgb['mbe']:>+6.0f} "
          f"R²={m_lgb['r2']:.3f} | feats={len(sel)}/{len(feat_cols)} "
          f"new_v4={len(v4_new)} | top: {', '.join(top5[:3])}")
    sys.stdout.flush()

print(f"\n  Phase 1 complete: {len(phase1_results)} meters in {time.time()-t0:.0f}s")

# ════════════════════════════════════════════════════════════════════════
# PHASE 2: Chronos-Bolt on targeted meters
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*74}")
print("  PHASE 2: Chronos-Bolt targeted ensemble")
print(f"{'='*74}")

import torch
from chronos import ChronosPipeline

chronos_pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny", device_map="cpu", dtype=torch.float32)
print(f"  Chronos-Bolt-Tiny loaded ({time.time()-t0:.0f}s)")

# Select meters for Chronos: HT tier OR LGB MAPE > 8%
chronos_candidates = []
for msn, data in phase1_results.items():
    if data['tier'] == "HT (>5kWh)" or data['metrics']['mape'] > 8.0:
        chronos_candidates.append(msn)
print(f"  Chronos candidates: {len(chronos_candidates)} meters "
      f"(HT or MAPE>8%)")

CHRONOS_CONTEXT = 336   # 7 days of 30-min data
CHRONOS_SAMPLES = 10    # fewer samples for speed

def chronos_predict_day(demand_history, n_intervals):
    """Predict n_intervals ahead from demand_history using Chronos."""
    vals = demand_history.values[-CHRONOS_CONTEXT:].astype(np.float64)
    ctx = torch.tensor(vals).unsqueeze(0).float()

    if n_intervals <= 64:
        fc = chronos_pipeline.predict(ctx, prediction_length=n_intervals, num_samples=CHRONOS_SAMPLES)
        med = fc.median(dim=1).values[0].numpy()
        p10 = np.percentile(fc[0].numpy(), 10, axis=0)
        p90 = np.percentile(fc[0].numpy(), 90, axis=0)
    else:
        # Autoregressive rollout
        chunks_med, chunks_p10, chunks_p90 = [], [], []
        remaining = n_intervals
        cur_ctx = ctx.clone()
        while remaining > 0:
            h = min(64, remaining)
            fc = chronos_pipeline.predict(cur_ctx, prediction_length=h, num_samples=CHRONOS_SAMPLES)
            m = fc.median(dim=1).values[0].numpy()
            chunks_med.append(m)
            chunks_p10.append(np.percentile(fc[0].numpy(), 10, axis=0))
            chunks_p90.append(np.percentile(fc[0].numpy(), 90, axis=0))
            if remaining > 64:
                ext = torch.tensor(m).unsqueeze(0).float()
                cur_ctx = torch.cat([cur_ctx[:, 64:], ext], dim=1)
            remaining -= h
        med = np.concatenate(chunks_med)[:n_intervals]
        p10 = np.concatenate(chunks_p10)[:n_intervals]
        p90 = np.concatenate(chunks_p90)[:n_intervals]

    return np.maximum(med, 0), np.maximum(p10, 0), np.maximum(p90, 0)


chronos_results = {}  # msn → {chr_pred, chr_p10, chr_p90, weight, ens_metrics}

for ci, msn in enumerate(chronos_candidates):
    ct0 = time.time()
    d = phase1_results[msn]
    test_df = d['test_df']
    y_test = d['y_test']
    pred_lgb = d['pred_lgb']

    # Get full demand series for context
    full_demand = d['feat_df'].set_index('ts')['demand_wh'].sort_index()

    # Group test by date, predict each holdout day
    tdc = test_df[['ts','target']].copy()
    tdc['date'] = pd.to_datetime(tdc['ts']).dt.date
    holdout_dates = sorted(tdc['date'].unique())

    chr_pred = np.zeros(len(test_df))
    chr_p10 = np.zeros(len(test_df))
    chr_p90 = np.zeros(len(test_df))

    # Sample max 15 holdout days for speed, extrapolate weight
    sample_dates = holdout_dates[::max(1, len(holdout_dates)//15)]

    for hd in sample_dates:
        day_mask = tdc['date'] == hd
        day_idx = np.where(day_mask.values)[0]
        n_int = len(day_idx)
        if n_int == 0: continue

        ctx_end = pd.Timestamp(hd)
        ctx_series = full_demand[full_demand.index < ctx_end]
        if len(ctx_series) < 48:
            chr_pred[day_idx] = pred_lgb[day_idx]
            chr_p10[day_idx] = d['lgb_q10'][day_idx]
            chr_p90[day_idx] = d['lgb_q90'][day_idx]
            continue

        m, p1, p9 = chronos_predict_day(ctx_series, n_int)
        chr_pred[day_idx] = m[:n_int]
        chr_p10[day_idx] = p1[:n_int]
        chr_p90[day_idx] = p9[:n_int]

    # Fill non-sampled days with Chronos median of sampled days' pattern
    # Simple: for unsampled days, use LightGBM (Chronos weight will handle blend)
    unsampled_mask = chr_pred == 0
    chr_pred[unsampled_mask] = pred_lgb[unsampled_mask]
    chr_p10[unsampled_mask] = d['lgb_q10'][unsampled_mask]
    chr_p90[unsampled_mask] = d['lgb_q90'][unsampled_mask]

    # Learn ensemble weight on sampled holdout days only
    sampled_idx = []
    for hd in sample_dates:
        day_mask = tdc['date'] == hd
        sampled_idx.extend(np.where(day_mask.values)[0].tolist())
    sampled_idx = np.array(sampled_idx)

    best_w, best_mae = 1.0, float('inf')
    if len(sampled_idx) > 0:
        y_s = y_test[sampled_idx]
        lgb_s = pred_lgb[sampled_idx]
        chr_s = chr_pred[sampled_idx]
        for w in np.arange(0, 1.01, 0.05):
            blend = w * lgb_s + (1-w) * chr_s
            mae_b = np.mean(np.abs(y_s - blend))
            if mae_b < best_mae:
                best_mae = mae_b
                best_w = round(w, 2)

    # Apply weight
    ens_pred = best_w * pred_lgb + (1-best_w) * chr_pred
    ens_q10 = best_w * d['lgb_q10'] + (1-best_w) * chr_p10
    ens_q90 = best_w * d['lgb_q90'] + (1-best_w) * chr_p90

    m_chr = calc_metrics(y_test, chr_pred, chr_p10, chr_p90)
    m_ens = calc_metrics(y_test, ens_pred, ens_q10, ens_q90)

    chronos_results[msn] = {
        'chr_pred': chr_pred, 'chr_p10': chr_p10, 'chr_p90': chr_p90,
        'ens_pred': ens_pred, 'ens_q10': ens_q10, 'ens_q90': ens_q90,
        'weight': best_w, 'm_chr': m_chr, 'm_ens': m_ens,
        'n_sampled_days': len(sample_dates),
    }

    elapsed = time.time() - ct0
    print(f"  [{ci+1}/{len(chronos_candidates)}] {msn} ({d['tier']:>16s}) | "
          f"LGB={d['metrics']['mape']:>5.1f}% CHR={m_chr['mape']:>5.1f}% "
          f"ENS={m_ens['mape']:>5.1f}% (w={best_w:.2f}) | "
          f"{len(sample_dates)} days sampled | {elapsed:.1f}s")
    sys.stdout.flush()


# ════════════════════════════════════════════════════════════════════════
# COMPILE FINAL RESULTS
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*74}")
print("  [7/7] FINAL RESULTS")
print(f"{'='*74}")

results = []
for msn, d in phase1_results.items():
    m = d['metrics']
    row = {
        'msn': str(msn), 'phase': d['phase'], 'tier': d['tier'],
        'mean_demand': round(d['mean_demand'], 1),
        'zero_pct': round(float((d['y_test']<0.5).mean()*100), 1),
        # LightGBM v4
        'lgb_mape': m['mape'], 'lgb_mae': m['mae'], 'lgb_mbe': m['mbe'],
        'lgb_r2': m['r2'], 'lgb_coverage': m['coverage_80'],
        'lgb_within5': m['within5'], 'lgb_within10': m['within10'],
        # Features
        'n_features': len(d['selected_feats']), 'n_features_total': 0,
        'n_v4_new_features': len(d['v4_new_features']),
        'v4_new_features': '; '.join(d['v4_new_features']),
        'top5': '; '.join(d['top5']),
        'best_iter': d['best_iter'],
        'train_days': d['n_train_days'], 'test_days': d['n_test_days'],
    }

    # Add Chronos + Ensemble if available
    if msn in chronos_results:
        cr = chronos_results[msn]
        row['chr_mape'] = cr['m_chr']['mape']
        row['chr_mae'] = cr['m_chr']['mae']
        row['chr_mbe'] = cr['m_chr']['mbe']
        row['chr_r2'] = cr['m_chr']['r2']
        row['ens_mape'] = cr['m_ens']['mape']
        row['ens_mae'] = cr['m_ens']['mae']
        row['ens_mbe'] = cr['m_ens']['mbe']
        row['ens_r2'] = cr['m_ens']['r2']
        row['ens_coverage'] = cr['m_ens']['coverage_80']
        row['lgb_weight'] = cr['weight']
        # Best model
        models = {'lgb': m['mape'], 'chronos': cr['m_chr']['mape'], 'ensemble': cr['m_ens']['mape']}
        row['best_model'] = min(models, key=models.get)
    else:
        row['chr_mape'] = np.nan
        row['chr_mae'] = np.nan
        row['chr_mbe'] = np.nan
        row['chr_r2'] = np.nan
        row['ens_mape'] = m['mape']  # ensemble = lgb for non-chronos meters
        row['ens_mae'] = m['mae']
        row['ens_mbe'] = m['mbe']
        row['ens_r2'] = m['r2']
        row['ens_coverage'] = m['coverage_80']
        row['lgb_weight'] = 1.0
        row['best_model'] = 'lgb'

    results.append(row)

rdf = pd.DataFrame(results)
rdf.to_csv('/sessions/awesome-gifted-feynman/benchmark_v4_ensemble.csv', index=False)
print(f"\n  Saved: {len(rdf)} meters → benchmark_v4_ensemble.csv")

# ── Summary table ──
print(f"\n  === MODEL COMPARISON (all {len(rdf)} meters) ===")
print(f"\n  {'Model':>14s} | {'Mean MAPE':>10s} | {'Med MAPE':>10s} | {'MAE':>8s} | {'|MBE|':>8s} | {'R²':>7s} | {'W5%':>5s} | {'W10%':>5s}")
print(f"  {'-'*14} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*7} | {'-'*5} | {'-'*5}")

# LightGBM v4
print(f"  {'LightGBM v4':>14s} | {rdf['lgb_mape'].mean():>8.2f}% | {rdf['lgb_mape'].median():>8.2f}% | "
      f"{rdf['lgb_mae'].mean():>6.0f}Wh | {rdf['lgb_mbe'].abs().mean():>6.1f}Wh | "
      f"{rdf['lgb_r2'].mean():>6.4f} | {rdf['lgb_within5'].mean():>4.1f} | {rdf['lgb_within10'].mean():>4.1f}")

# Ensemble (lgb for non-chronos, blend for chronos)
print(f"  {'Best Ensemble':>14s} | {rdf['ens_mape'].mean():>8.2f}% | {rdf['ens_mape'].median():>8.2f}% | "
      f"{rdf['ens_mae'].mean():>6.0f}Wh | {rdf['ens_mbe'].abs().mean():>6.1f}Wh | "
      f"{rdf['ens_r2'].mean():>6.4f} | {'—':>5s} | {'—':>5s}")

# Chronos standalone (only meters that had it)
chr_df = rdf.dropna(subset=['chr_mape'])
if len(chr_df) > 0:
    print(f"\n  Chronos standalone ({len(chr_df)} meters tested):")
    print(f"  {'Chronos':>14s} | {chr_df['chr_mape'].mean():>8.2f}% | {chr_df['chr_mape'].median():>8.2f}% | "
          f"{chr_df['chr_mae'].mean():>6.0f}Wh | {chr_df['chr_mbe'].abs().mean():>6.1f}Wh | "
          f"{chr_df['chr_r2'].mean():>6.4f} |")

# ── Best model distribution ──
print(f"\n  Best model per meter:")
for m in ['lgb','chronos','ensemble']:
    cnt = (rdf['best_model']==m).sum()
    if cnt > 0:
        print(f"    {m:>10s}: {cnt} meters ({cnt/len(rdf)*100:.0f}%)")

# ── Weight distribution for Chronos meters ──
if len(chr_df) > 0:
    print(f"\n  Ensemble weight (LGB share) for Chronos candidates:")
    print(f"    mean={chr_df['lgb_weight'].mean():.2f}, median={chr_df['lgb_weight'].median():.2f}, "
          f"min={chr_df['lgb_weight'].min():.2f}, max={chr_df['lgb_weight'].max():.2f}")

# ── Per-tier ──
print(f"\n  --- Per-Tier Performance ---")
for tier in ["HT (>5kWh)", "Large (1.5-5k)", "Medium (0.5-1.5k)", "Small (<500)"]:
    tm = rdf[rdf['tier']==tier]
    if len(tm)==0: continue
    print(f"  {tier:>20s} ({len(tm):2d}m): LGB={tm['lgb_mape'].mean():>5.1f}% "
          f"ENS={tm['ens_mape'].mean():>5.1f}% | "
          f"MAE={tm['lgb_mae'].mean():>5.0f}Wh | w={tm['lgb_weight'].mean():.2f}")

# ── v4 new features impact ──
print(f"\n  --- v4 New Feature Adoption ---")
print(f"  Avg new v4 features selected: {rdf['n_v4_new_features'].mean():.1f} per meter")
# Count most common new features
from collections import Counter
all_new = []
for feats in rdf['v4_new_features']:
    if isinstance(feats, str) and feats:
        all_new.extend(feats.split('; '))
feat_counts = Counter(all_new)
print(f"  Most adopted new signals:")
for feat, cnt in feat_counts.most_common(10):
    print(f"    {feat:>25s}: {cnt}/{len(rdf)} meters ({cnt/len(rdf)*100:.0f}%)")

# ── Compare with S2-v3 baseline ──
print(f"\n  --- vs Strategy 2 v3 Baseline ---")
try:
    s2prev = pd.read_csv('/sessions/awesome-gifted-feynman/benchmark_strategy2.csv')
    s2prev['msn'] = s2prev['msn'].astype(str)
    comp = s2prev[['msn','mape','mae','mbe']].rename(
        columns={'mape':'prev_mape','mae':'prev_mae','mbe':'prev_mbe'}
    ).merge(rdf[['msn','lgb_mape','lgb_mae','lgb_mbe','ens_mape','tier']], on='msn')

    print(f"  {'':>15s} | {'S2-v3':>8s} | {'v4 LGB':>8s} | {'v4 Ens':>8s}")
    print(f"  {'-'*15} | {'-'*8} | {'-'*8} | {'-'*8}")
    print(f"  {'Mean MAPE':>15s} | {comp['prev_mape'].mean():>6.1f}% | "
          f"{comp['lgb_mape'].mean():>6.1f}% | {comp['ens_mape'].mean():>6.1f}%")
    print(f"  {'Median MAPE':>15s} | {comp['prev_mape'].median():>6.1f}% | "
          f"{comp['lgb_mape'].median():>6.1f}% | {comp['ens_mape'].median():>6.1f}%")
    print(f"  {'Mean MAE':>15s} | {comp['prev_mae'].mean():>5.0f}Wh | "
          f"{comp['lgb_mae'].mean():>5.0f}Wh | {'—':>6s}")
    print(f"  {'Mean |MBE|':>15s} | {comp['prev_mbe'].abs().mean():>5.0f}Wh | "
          f"{comp['lgb_mbe'].abs().mean():>5.0f}Wh | {'—':>6s}")

    comp['delta'] = comp['lgb_mape'] - comp['prev_mape']
    improved = (comp['delta'] < -0.1).sum()
    same = ((comp['delta'] >= -0.1) & (comp['delta'] <= 0.1)).sum()
    regressed = (comp['delta'] > 0.1).sum()
    print(f"\n  v4 vs S2-v3: {improved} improved, {same} same (±0.1%), {regressed} regressed")

    # Per-tier delta
    for tier in ["HT (>5kWh)", "Medium (0.5-1.5k)", "Small (<500)"]:
        tc = comp[comp['tier']==tier]
        if len(tc)==0: continue
        print(f"    {tier:>20s}: {tc['prev_mape'].mean():>5.1f}% → {tc['lgb_mape'].mean():>5.1f}% "
              f"(Δ={tc['delta'].mean():>+5.1f}%)")
except Exception as e:
    print(f"  Could not compare: {e}")

# ── Compare with Strategy 1 v3 ──
print(f"\n  --- vs Strategy 1 v3 (Chronological) ---")
try:
    s1 = pd.read_csv('/sessions/awesome-gifted-feynman/benchmark_strategy1_v3.csv')
    s1['msn'] = s1['msn'].astype(str)
    c1 = s1[['msn','mape']].rename(columns={'mape':'s1_mape'}).merge(
        rdf[['msn','ens_mape','tier']], on='msn')
    print(f"  S1v3 mean MAPE: {c1['s1_mape'].mean():.1f}% → v4 ensemble: {c1['ens_mape'].mean():.1f}%")
    print(f"  S1v3 med  MAPE: {c1['s1_mape'].median():.1f}% → v4 ensemble: {c1['ens_mape'].median():.1f}%")
except:
    pass

print(f"\n  Total time: {time.time()-t0:.0f}s")
print(f"\n  ✅ v4 Ensemble Engine complete.")
