"""
Generate Strategy 2 v4 Dashboard Data
======================================
Trains v4 LightGBM per meter with Strategy 2 (stratified temporal) split,
captures interval-level predictions for the interactive dashboard.

Output format matches S1 v3 dashboard JSON for code reuse.
"""
import pandas as pd, numpy as np, json, time, sys, warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
import urllib.request

t0 = time.time()
print("=" * 70)
print("  Generating Strategy 2 v4 Dashboard Data")
print("=" * 70)

# ═══════ WEATHER ═══════
print("\n[1/5] Fetching expanded weather...")
WEATHER_VARS = (
    "temperature_2m,relative_humidity_2m,dewpoint_2m,"
    "surface_pressure,cloud_cover,precipitation,wind_speed_10m,"
    "shortwave_radiation,direct_radiation,diffuse_radiation,"
    "direct_normal_irradiance"
)
def fetch_weather(lat=17.0, lon=82.0, start="2024-10-01", end="2026-02-28"):
    chunks = []
    current = pd.Timestamp(start); end_dt = pd.Timestamp(end)
    while current < end_dt:
        chunk_end = min(current + pd.DateOffset(months=6) - pd.Timedelta(days=1), end_dt)
        url = (f"https://archive-api.open-meteo.com/v1/archive?"
               f"latitude={lat}&longitude={lon}"
               f"&start_date={current.strftime('%Y-%m-%d')}&end_date={chunk_end.strftime('%Y-%m-%d')}"
               f"&hourly={WEATHER_VARS}&timezone=Asia%2FKolkata")
        data = json.loads(urllib.request.urlopen(url, timeout=30).read())
        h = data['hourly']
        chunks.append(pd.DataFrame({
            'ts': pd.to_datetime(h['time']), 'temperature': h['temperature_2m'],
            'humidity': h['relative_humidity_2m'], 'dewpoint': h['dewpoint_2m'],
            'pressure': h['surface_pressure'], 'cloud_cover': h['cloud_cover'],
            'precipitation': h['precipitation'], 'wind_speed': h['wind_speed_10m'],
            'ghi': h['shortwave_radiation'], 'dni': h['direct_normal_irradiance'],
            'dhi': h['diffuse_radiation'], 'direct_rad': h['direct_radiation'],
        }))
        current = chunk_end + pd.Timedelta(days=1)
    wx = pd.concat(chunks, ignore_index=True).drop_duplicates('ts').sort_values('ts')
    wx_30m = wx.set_index('ts').resample('30min').ffill().reset_index()
    return wx_30m

weather = fetch_weather()
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
print(f"  {len(weather_expanded)} records, 20 vars")

# ═══════ HOLIDAYS + ToD ═══════
HOLIDAYS = {
    "2024-10-02":"G","2024-10-12":"D","2024-11-01":"Dw","2024-11-02":"Dw",
    "2024-11-15":"GN","2024-12-25":"X","2025-01-01":"NY","2025-01-14":"S",
    "2025-01-15":"S","2025-01-26":"R","2025-02-26":"MS","2025-03-14":"H",
    "2025-03-30":"U","2025-03-31":"RE","2025-04-06":"RN","2025-04-10":"MJ",
    "2025-04-14":"AJ","2025-04-18":"GF","2025-05-01":"MD","2025-05-12":"BP",
    "2025-06-07":"E","2025-07-06":"MH","2025-08-15":"ID","2025-08-16":"JM",
    "2025-09-05":"ML","2025-10-02":"DS","2025-10-20":"Dw","2025-10-21":"Dw",
    "2025-11-05":"GN","2025-12-25":"X","2026-01-01":"NY","2026-01-14":"S",
    "2026-01-15":"S","2026-01-26":"R","2026-02-15":"MS",
}
holiday_dates = set(pd.to_datetime(list(HOLIDAYS.keys())).date)
def tod_multiplier(h):
    if 6<=h<18: return 1.0
    elif 18<=h<22: return 1.2
    else: return 0.9

# ═══════ DATA ═══════
print("\n[2/5] Loading meter data...")
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
        sub.columns = ['ts','demand_wh','voltage']; sub['msn'],sub['phase'] = msn,'1PH'
        frames.append(sub)
    df_tp = pd.read_parquet("/sessions/awesome-gifted-feynman/tp_data.parquet")
    df_tp['ts'] = pd.to_datetime(df_tp['ts'])
    df_tp['wh_imp'] = pd.to_numeric(df_tp['wh_imp'].str.replace(',',''), errors='coerce')
    for v in ['v_r','v_y','v_b']:
        df_tp[v] = pd.to_numeric(df_tp[v].str.replace(',',''), errors='coerce')
    df_tp['voltage'] = df_tp[['v_r','v_y','v_b']].mean(axis=1)
    for msn in df_tp['msn'].unique():
        sub = df_tp[df_tp['msn']==msn][['ts','wh_imp','voltage']].drop_duplicates('ts').sort_values('ts')
        sub.columns = ['ts','demand_wh','voltage']; sub['msn'],sub['phase'] = msn,phase_map.get(msn,'3PH')
        frames.append(sub)
    return pd.concat(frames, ignore_index=True).dropna(subset=['demand_wh']), phase_map

df_all, phase_map = load_data()
meter_days = df_all.groupby('msn')['ts'].agg(lambda x: (x.max()-x.min()).days)
eligible = meter_days[meter_days>=180].index.tolist()
print(f"  {len(eligible)} eligible meters")

# Fleet
fleet_agg = df_all.groupby('ts')['demand_wh'].agg(['mean','std','count']).reset_index()
fleet_agg.columns = ['ts','fleet_mean','fleet_std','fleet_count']
fleet_agg = fleet_agg[fleet_agg['fleet_count']>=5].copy()

# ═══════ HELPERS ═══════
def get_params(tier, n):
    p = {'objective':'regression','metric':'mae','verbose':-1,'n_jobs':-1,
         'learning_rate':0.03,'num_leaves':31,'min_child_samples':50,
         'feature_fraction':0.6,'bagging_fraction':0.7,'bagging_freq':5,
         'lambda_l1':0.1,'lambda_l2':1.0,'max_depth':8}
    if tier=="HT (>5kWh)": p.update({'num_leaves':47,'min_child_samples':30,'feature_fraction':0.7,'learning_rate':0.04})
    elif tier=="Small (<500)": p.update({'num_leaves':15,'min_child_samples':80,'feature_fraction':0.5,'max_depth':6,'lambda_l1':0.5,'lambda_l2':5.0})
    if n<3000: p['num_leaves']=min(p['num_leaves'],15); p['min_child_samples']=max(p['min_child_samples'],100)
    return p

def classify_tier(m):
    if m>5000: return "HT (>5kWh)"
    elif m>1500: return "Large (1.5-5k)"
    elif m>500: return "Medium (0.5-1.5k)"
    else: return "Small (<500)"

def build_features_v4(df_meter, weather_expanded, fleet_agg):
    df = df_meter.sort_values('ts').reset_index(drop=True)
    df['hour'] = df['ts'].dt.hour + df['ts'].dt.minute/60
    df['dow'] = df['ts'].dt.dayofweek; df['month'] = df['ts'].dt.month
    df['is_weekend'] = (df['dow']>=5).astype(int)
    for name, period in [('hour',24),('dow',7),('month',12)]:
        df[f'{name}_sin'] = np.sin(2*np.pi*df[name]/period)
        df[f'{name}_cos'] = np.cos(2*np.pi*df[name]/period)
    for lag in [1,2,4,6,12,24,48,96,336]:
        df[f'lag_{lag}'] = df['demand_wh'].shift(lag)
    for win in [6,12,48,336]:
        s = df['demand_wh'].shift(1)
        df[f'rmean_{win}'] = s.rolling(win,min_periods=1).mean()
        df[f'rstd_{win}'] = s.rolling(win,min_periods=1).std()
    df['demand_diff_1'] = df['demand_wh'].diff(1)
    df['demand_diff_4'] = df['demand_wh'].diff(4)
    s = df['demand_wh'].shift(1)
    df['rmean_14d'] = s.rolling(672,min_periods=48).mean()
    df['rmean_30d'] = s.rolling(1440,min_periods=48).mean()
    df['trend_ratio'] = df['rmean_14d'] / df['rmean_30d'].clip(lower=1)
    wx_cols = ['ts','temperature','humidity','dewpoint','pressure','cloud_cover',
               'precipitation','wind_speed','ghi','dni','dhi','direct_rad',
               'pressure_delta_3h','temp_delta_3h','ghi_rmean_6h','cloud_delta_3h',
               'is_raining','diffuse_fraction','heat_index','temp_rmean_6h']
    df = df.merge(weather_expanded[wx_cols], on='ts', how='left')
    for c in wx_cols[1:]: df[c] = df[c].ffill().bfill()
    df['cooling_deg'] = (df['temperature']-25).clip(lower=0)
    df['heating_deg'] = (18-df['temperature']).clip(lower=0)
    df['hour_x_temp'] = df['hour']*df['temperature']/100
    df['peak_x_temp'] = ((df['hour']>=12)&(df['hour']<=17)).astype(int)*df['temperature']
    df['is_holiday'] = df['ts'].dt.date.isin(holiday_dates).astype(int)
    df['near_holiday'] = 0
    for hd in holiday_dates:
        mask = (df['ts'].dt.date>=hd-pd.Timedelta(days=1))&(df['ts'].dt.date<=hd+pd.Timedelta(days=1))
        df.loc[mask,'near_holiday'] = 1
    df['tod_multiplier'] = df['ts'].dt.hour.apply(tod_multiplier)
    df['is_peak'] = ((df['ts'].dt.hour>=18)&(df['ts'].dt.hour<22)).astype(int)
    df['voltage_lag1'] = df['voltage'].shift(1)
    df['voltage_rstd_6'] = df['voltage'].shift(1).rolling(6,min_periods=1).std()
    df['voltage_rmean_48'] = df['voltage'].shift(1).rolling(48,min_periods=1).mean()
    shm = df.groupby('hour')['demand_wh'].transform(lambda x: x.shift(1).expanding(min_periods=48).mean())
    df['deviation_from_hourly'] = df['demand_wh'].shift(1) - shm
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

# ═══════ GENERATE DASHBOARD DATA ═══════
print("\n[3/5] Training models and capturing interval-level predictions...")

MAX_HOLDOUT_DAYS = 25
MAX_TRAIN_DAYS = 5
consumers = []

for i, msn in enumerate(eligible):
    mdata = df_all[df_all['msn']==msn].copy()
    phase = mdata['phase'].iloc[0]
    feat_df, feat_cols = build_features_v4(mdata, weather_expanded, fleet_agg)
    if len(feat_df) < 500: continue

    # Strategy 2 split
    feat_df_c = feat_df.copy()
    feat_df_c['date'] = feat_df_c['ts'].dt.date
    day_counts = feat_df_c.groupby('date').size()
    complete_days = sorted(day_counts[day_counts>=44].index)
    holdout_days = set(complete_days[::4])
    train_df = feat_df_c[~feat_df_c['date'].isin(holdout_days)].copy()
    test_df = feat_df_c[feat_df_c['date'].isin(holdout_days)].copy()

    if len(test_df) < 48: continue

    y_train = train_df['target'].values
    y_test = test_df['target'].values
    mean_demand = float(np.mean(y_test))
    tier = classify_tier(mean_demand)

    # Two-pass training
    X_all = np.nan_to_num(train_df[feat_cols].values, nan=0.)
    val_cut = int(len(X_all)*0.80)
    d1 = lgb.Dataset(X_all[:val_cut], label=y_train[:val_cut])
    v1 = lgb.Dataset(X_all[val_cut:], label=y_train[val_cut:], reference=d1)
    m1 = lgb.train({'objective':'regression','metric':'mae','num_leaves':31,
                     'learning_rate':0.08,'feature_fraction':0.7,'verbose':-1,'n_jobs':-1},
                    d1, num_boost_round=150, valid_sets=[v1],
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
    imp = m1.feature_importance(importance_type='gain')
    fg = sorted(zip(feat_cols,imp), key=lambda x:-x[1])
    nk = max(25, int(len(feat_cols)*0.55))
    sel = [f for f,g in fg[:nk] if g>0]
    if len(sel)<20: sel = [f for f,g in fg[:25]]

    X_tr = np.nan_to_num(train_df[sel].values, nan=0.)
    X_te = np.nan_to_num(test_df[sel].values, nan=0.)
    params = get_params(tier, len(X_tr))
    d2 = lgb.Dataset(X_tr[:val_cut], label=y_train[:val_cut])
    v2 = lgb.Dataset(X_tr[val_cut:], label=y_train[val_cut:], reference=d2)
    model = lgb.train(params, d2, num_boost_round=800, valid_sets=[v2],
                       callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])

    pred = np.maximum(model.predict(X_te), 0)

    # Also predict training days for comparison
    train_pred = np.maximum(model.predict(X_tr), 0)

    # Top features
    fi = sorted(zip(sel, model.feature_importance(importance_type='gain')), key=lambda x:-x[1])
    top_feats = [{'name':f,'gain':round(float(g),1)} for f,g in fi[:8]]

    # Metrics
    mask = y_test > 0.5
    mape = float(np.mean(np.abs(y_test[mask]-pred[mask])/y_test[mask])*100) if mask.sum()>0 else 0
    mae = float(np.mean(np.abs(y_test-pred)))
    mbe = float(np.mean(pred-y_test))
    ss_res = np.sum((y_test-pred)**2); ss_tot = np.sum((y_test-np.mean(y_test))**2)
    r2 = float(1-ss_res/ss_tot) if ss_tot>0 else 0

    # Build day-level data
    days_data = []

    # Holdout days (up to MAX_HOLDOUT_DAYS, spread across timeline)
    holdout_list = sorted(holdout_days)
    if len(holdout_list) > MAX_HOLDOUT_DAYS:
        step = len(holdout_list) // MAX_HOLDOUT_DAYS
        holdout_sample = holdout_list[::step][:MAX_HOLDOUT_DAYS]
    else:
        holdout_sample = holdout_list

    for hd in holdout_sample:
        day_mask = test_df['date'] == hd
        day_rows = test_df[day_mask]
        day_pred = pred[day_mask.values]
        if len(day_rows) < 10: continue
        intervals = []
        for j, (_, row) in enumerate(day_rows.iterrows()):
            intervals.append({
                'ts': str(row['ts']),
                'hour': round(float(row['hour']), 1),
                'actual': round(float(row['target']), 1),
                'predicted': round(float(day_pred[j]), 1),
            })
        day_actual = day_rows['target'].values
        day_p = day_pred[:len(day_actual)]
        dm = day_actual > 0.5
        day_mape = float(np.mean(np.abs(day_actual[dm]-day_p[dm])/day_actual[dm])*100) if dm.sum()>0 else 0
        days_data.append({
            'date': str(hd), 'split': 'holdout', 'mape': round(day_mape, 1),
            'intervals': intervals, 'n': len(intervals),
            'peakActual': round(float(day_actual.max()), 1),
        })

    # Training days (sample MAX_TRAIN_DAYS spread across timeline)
    train_dates = sorted(train_df['date'].unique())
    if len(train_dates) > MAX_TRAIN_DAYS:
        step = len(train_dates) // MAX_TRAIN_DAYS
        train_sample = train_dates[::step][:MAX_TRAIN_DAYS]
    else:
        train_sample = train_dates[:MAX_TRAIN_DAYS]

    for td in train_sample:
        day_mask = train_df['date'] == td
        day_rows = train_df[day_mask]
        day_pred_t = train_pred[day_mask.values]
        if len(day_rows) < 10: continue
        intervals = []
        for j, (_, row) in enumerate(day_rows.iterrows()):
            intervals.append({
                'ts': str(row['ts']),
                'hour': round(float(row['hour']), 1),
                'actual': round(float(row['target']), 1),
                'predicted': round(float(day_pred_t[j]), 1),
            })
        day_actual = day_rows['target'].values
        day_p = day_pred_t[:len(day_actual)]
        dm = day_actual > 0.5
        day_mape = float(np.mean(np.abs(day_actual[dm]-day_p[dm])/day_actual[dm])*100) if dm.sum()>0 else 0
        days_data.append({
            'date': str(td), 'split': 'train', 'mape': round(day_mape, 1),
            'intervals': intervals, 'n': len(intervals),
            'peakActual': round(float(day_actual.max()), 1),
        })

    # Sort days chronologically
    days_data.sort(key=lambda x: x['date'])

    consumers.append({
        'id': f's2_{msn}',
        'name': f'Meter {msn[-6:]}',
        'type': tier,
        'region': 'APEPDCL',
        'msn': str(msn),
        'metrics': {
            'mape': round(mape, 2), 'mae': round(mae, 1),
            'mbe': round(mbe, 1), 'r2': round(r2, 4),
        },
        'quality': {
            'zeroPercent': round(float((y_test<0.5).mean()*100), 1),
            'meanDemand': round(mean_demand, 1),
            'nFeatures': len(sel),
        },
        'topFeatures': top_feats,
        'days': days_data,
        'demandStats': {
            'min': round(float(y_test.min()), 1),
            'max': round(float(y_test.max()), 1),
            'mean': round(float(y_test.mean()), 1),
            'std': round(float(y_test.std()), 1),
        },
    })

    n_holdout = sum(1 for d in days_data if d['split']=='holdout')
    n_train = sum(1 for d in days_data if d['split']=='train')
    print(f"  [{i+1:2d}/{len(eligible)}] {msn} ({tier:>16s}) | "
          f"MAPE={mape:>5.1f}% | {n_holdout}h+{n_train}t days")
    sys.stdout.flush()

# ═══════ ASSEMBLE JSON ═══════
print("\n[4/5] Assembling dashboard JSON...")

# Read v4 benchmark for summary
v4_df = pd.read_csv('/sessions/awesome-gifted-feynman/benchmark_v4_ensemble.csv')

dashboard = {
    'strategy': 'stratified_temporal_v4',
    'strategyLabel': 'Strategy 2: Stratified Temporal (v4)',
    'description': (
        'Every 4th complete day held out — train and test both span full timeline. '
        'v4 engine: expanded weather (pressure, cloud, DNI/DHI, heat index), '
        'voltage features, ToD tariff, two-pass feature selection (66→36), '
        'per-tier adaptive regularization.'
    ),
    'consumers': consumers,
    'generated': pd.Timestamp.now().isoformat(),
    'summary': {
        'totalMeters': len(consumers),
        'meanMape': round(v4_df['lgb_mape'].mean(), 2),
        'medianMape': round(v4_df['lgb_mape'].median(), 2),
        'meanMae': round(v4_df['lgb_mae'].mean(), 1),
        'meanAbsMbe': round(v4_df['lgb_mbe'].abs().mean(), 1),
        'meanR2': round(v4_df['lgb_r2'].mean(), 4),
        'tiers': {},
    }
}

for tier in ["HT (>5kWh)", "Large (1.5-5k)", "Medium (0.5-1.5k)", "Small (<500)"]:
    tm = v4_df[v4_df['tier']==tier]
    if len(tm)==0: continue
    dashboard['summary']['tiers'][tier] = {
        'count': len(tm),
        'meanMape': round(tm['lgb_mape'].mean(), 2),
        'medianMape': round(tm['lgb_mape'].median(), 2),
    }

out_path = '/sessions/awesome-gifted-feynman/s2_v4_dashboard_data.json'
with open(out_path, 'w') as f:
    json.dump(dashboard, f, default=str)

size_mb = len(json.dumps(dashboard, default=str)) / 1024 / 1024
total_days = sum(len(c['days']) for c in consumers)
print(f"\n[5/5] Saved: {out_path}")
print(f"  {len(consumers)} consumers, {total_days} days, {size_mb:.1f} MB")
print(f"  Total time: {time.time()-t0:.0f}s")
