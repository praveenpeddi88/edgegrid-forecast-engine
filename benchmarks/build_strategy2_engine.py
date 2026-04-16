"""
EdgeGrid Forecast Engine — Strategy 2: Stratified Temporal
==========================================================
Every 4th complete day held out → train and test both span full timeline.
Uses v3 innovations: weather, two-pass feature selection, adaptive regularization.

Key difference from Strategy 1:
  - Strategy 1: train first 75%, predict last 25% → season shift problem
  - Strategy 2: every 4th day held out → both splits see all seasons
  - This tests "steady-state deployment" accuracy

Leakage note:
  Lag features on holdout days reference training-period values.
  This is realistic: in production, yesterday's actuals are known before predicting today.
"""
import pandas as pd, numpy as np, json, time, sys, warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb

t0 = time.time()

print("=" * 70)
print("  EdgeGrid Strategy 2: Stratified Temporal Holdout")
print("  Every 4th complete day held out — full-timeline coverage")
print("=" * 70)

# ════════════════════════════════════════════════════════════════════
# WEATHER DATA
# ════════════════════════════════════════════════════════════════════
print("\n[1/6] Fetching weather data from Open-Meteo...")
import urllib.request

def fetch_weather(lat=17.0, lon=82.0, start="2024-10-01", end="2026-02-28"):
    chunks = []
    current = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    while current < end_dt:
        chunk_end = min(current + pd.DateOffset(months=6) - pd.Timedelta(days=1), end_dt)
        url = (f"https://archive-api.open-meteo.com/v1/archive?"
               f"latitude={lat}&longitude={lon}"
               f"&start_date={current.strftime('%Y-%m-%d')}"
               f"&end_date={chunk_end.strftime('%Y-%m-%d')}"
               f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation"
               f"&timezone=Asia%2FKolkata")
        resp = urllib.request.urlopen(url, timeout=30)
        data = json.loads(resp.read())
        h = data['hourly']
        chunk_df = pd.DataFrame({
            'ts': pd.to_datetime(h['time']),
            'temperature': h['temperature_2m'],
            'humidity': h['relative_humidity_2m'],
            'wind_speed': h['wind_speed_10m'],
            'solar_radiation': h['shortwave_radiation'],
        })
        print(f"  Fetched {current.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}: {len(chunk_df)} hours")
        chunks.append(chunk_df)
        current = chunk_end + pd.Timedelta(days=1)
    wx = pd.concat(chunks, ignore_index=True).drop_duplicates('ts').sort_values('ts')
    print(f"  Weather: {len(wx)} hourly records")
    wx_30m = wx.set_index('ts').resample('30min').ffill().reset_index()
    print(f"  Resampled to 30-min: {len(wx_30m)} records")
    return wx_30m

weather = fetch_weather()

# ════════════════════════════════════════════════════════════════════
# HOLIDAY CALENDAR
# ════════════════════════════════════════════════════════════════════
print("\n[2/6] Building Indian holiday calendar...")
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
    "2025-10-02": "Gandhi Jayanti", "2025-10-02": "Dussehra",
    "2025-10-20": "Diwali", "2025-10-21": "Diwali",
    "2025-11-05": "Guru Nanak", "2025-12-25": "Christmas",
    "2026-01-01": "New Year", "2026-01-14": "Sankranti",
    "2026-01-15": "Sankranti", "2026-01-26": "Republic Day",
    "2026-02-15": "Maha Shivaratri",
}
holiday_dates = set(pd.to_datetime(list(HOLIDAYS.keys())).date)
print(f"  {len(holiday_dates)} holiday dates loaded")

# ════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════
print("\n[3/6] Loading meter data...")

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

    df = pd.concat(frames, ignore_index=True).dropna(subset=['demand_wh'])
    return df, phase_map

df_all, phase_map = load_data()
print(f"  {len(df_all):,} rows, {df_all['msn'].nunique()} meters ({time.time()-t0:.1f}s)")

meter_days = df_all.groupby('msn')['ts'].agg(lambda x: (x.max()-x.min()).days)
eligible = meter_days[meter_days >= 180].index.tolist()
print(f"  Eligible (>=180 days): {len(eligible)} meters")

# ════════════════════════════════════════════════════════════════════
# FLEET AGGREGATES
# ════════════════════════════════════════════════════════════════════
print("\n[4/6] Pre-computing fleet aggregates...")
fleet_agg = df_all.groupby('ts')['demand_wh'].agg(['mean','std','count']).reset_index()
fleet_agg.columns = ['ts','fleet_mean','fleet_std','fleet_count']
fleet_agg = fleet_agg[fleet_agg['fleet_count'] >= 5].copy()
print(f"  Fleet: {len(fleet_agg)} timesteps with >= 5 meters")

# ════════════════════════════════════════════════════════════════════
# TIER-ADAPTIVE HYPERPARAMETERS (same as v3)
# ════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════
# FEATURE BUILDER (same as v3)
# ════════════════════════════════════════════════════════════════════

def build_features_v3(df_meter, weather, fleet_agg):
    df = df_meter.sort_values('ts').reset_index(drop=True)

    # Time features
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

    # Lag features
    for lag in [1, 2, 4, 6, 12, 24, 48, 96, 336]:
        df[f'lag_{lag}'] = df['demand_wh'].shift(lag)

    # Rolling stats
    for win in [6, 12, 48, 336]:
        s = df['demand_wh'].shift(1)
        df[f'rmean_{win}'] = s.rolling(win, min_periods=1).mean()
        df[f'rstd_{win}'] = s.rolling(win, min_periods=1).std()

    # Momentum
    df['demand_diff_1'] = df['demand_wh'].diff(1)
    df['demand_diff_4'] = df['demand_wh'].diff(4)

    # Longer rolling
    s = df['demand_wh'].shift(1)
    df['rmean_14d'] = s.rolling(672, min_periods=48).mean()
    df['rmean_30d'] = s.rolling(1440, min_periods=48).mean()
    df['trend_ratio'] = df['rmean_14d'] / df['rmean_30d'].clip(lower=1)

    # Weather
    wx = weather[['ts','temperature','humidity','solar_radiation']].copy()
    df = df.merge(wx, on='ts', how='left')
    df['temperature'] = df['temperature'].ffill().bfill()
    df['humidity'] = df['humidity'].ffill().bfill()
    df['solar_radiation'] = df['solar_radiation'].ffill().bfill()
    df['cooling_deg'] = (df['temperature'] - 25).clip(lower=0)
    df['hour_x_temp'] = df['hour'] * df['temperature'] / 100

    # Holidays
    df['is_holiday'] = df['ts'].dt.date.isin(holiday_dates).astype(int)
    df['near_holiday'] = 0
    for hd in holiday_dates:
        mask = (df['ts'].dt.date >= hd - pd.Timedelta(days=1)) & \
               (df['ts'].dt.date <= hd + pd.Timedelta(days=1))
        df.loc[mask, 'near_holiday'] = 1

    # Seasonal deviation
    same_hour_mean = df.groupby('hour')['demand_wh'].transform(
        lambda x: x.shift(1).expanding(min_periods=48).mean()
    )
    df['deviation_from_hourly'] = df['demand_wh'].shift(1) - same_hour_mean

    # Fleet features
    fleet = fleet_agg[['ts','fleet_mean','fleet_std']].copy()
    df = df.merge(fleet, on='ts', how='left')
    df['fleet_mean'] = df['fleet_mean'].ffill().bfill().fillna(0)
    df['fleet_std'] = df['fleet_std'].ffill().bfill().fillna(0)
    df['fleet_mean_lag1'] = df['fleet_mean'].shift(1)
    df['fleet_std_lag1'] = df['fleet_std'].shift(1)
    df['vs_fleet_ratio'] = df['demand_wh'].shift(1) / df['fleet_mean_lag1'].clip(lower=1)

    # Drop warmup
    df = df.iloc[336:].reset_index(drop=True)
    df['target'] = df['demand_wh']

    exclude = {'ts','demand_wh','target','msn','phase','voltage','scno',
               'fleet_mean','fleet_std','date'}
    feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64','int64','int32','float32']]
    return df, feat_cols


def classify_tier(mean_wh):
    if mean_wh > 5000: return "HT (>5kWh)"
    elif mean_wh > 1500: return "Large (1.5-5k)"
    elif mean_wh > 500: return "Medium (0.5-1.5k)"
    else: return "Small (<500)"


# ════════════════════════════════════════════════════════════════════
# STRATEGY 2: STRATIFIED TEMPORAL SPLIT
# ════════════════════════════════════════════════════════════════════

def split_stratified_temporal(df, holdout_every=4):
    """
    Every Nth complete day is held out. Both splits span full timeline.
    A day is "complete" if it has >= 44 intervals (out of 48 for 30-min data).
    """
    df = df.copy()
    df['date'] = df['ts'].dt.date
    day_counts = df.groupby('date').size()
    complete_days = sorted(day_counts[day_counts >= 44].index)

    # Every 4th day → ~25% holdout, ~75% train
    holdout_days = set(complete_days[::holdout_every])

    train = df[~df['date'].isin(holdout_days)].copy()
    test = df[df['date'].isin(holdout_days)].copy()

    n_train_days = train['date'].nunique()
    n_test_days = test['date'].nunique()

    return train, test, holdout_days, n_train_days, n_test_days


# ════════════════════════════════════════════════════════════════════
# BENCHMARK LOOP
# ════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  [5/6] RUNNING STRATEGY 2 BENCHMARK")
print(f"{'=' * 70}")

results_s2 = []

for i, msn in enumerate(eligible):
    mdata = df_all[df_all['msn'] == msn].copy()
    phase = mdata['phase'].iloc[0]

    feat_df, feat_cols = build_features_v3(mdata, weather, fleet_agg)
    if len(feat_df) < 500:
        print(f"  [{i+1:2d}/{len(eligible)}] {msn} — SKIPPED (too few rows)")
        continue

    # ── Strategy 2: Stratified temporal split ──
    train_df, test_df, holdout_days, n_train_days, n_test_days = split_stratified_temporal(feat_df)

    if len(test_df) < 48:
        print(f"  [{i+1:2d}/{len(eligible)}] {msn} — SKIPPED (insufficient test data)")
        continue

    y_train = train_df['target'].values
    y_test = test_df['target'].values
    mean_demand = float(np.mean(y_test))
    tier = classify_tier(mean_demand)

    # ── PASS 1: Quick fit to rank features ──
    X_train_all = np.nan_to_num(train_df[feat_cols].values, nan=0.)
    X_test_all = np.nan_to_num(test_df[feat_cols].values, nan=0.)

    # For validation in stratified: use last 20% of training rows chronologically
    val_cut = int(len(X_train_all) * 0.80)
    dtrain_p1 = lgb.Dataset(X_train_all[:val_cut], label=y_train[:val_cut])
    dval_p1 = lgb.Dataset(X_train_all[val_cut:], label=y_train[val_cut:], reference=dtrain_p1)

    quick_params = {
        'objective': 'regression', 'metric': 'mae', 'num_leaves': 31,
        'learning_rate': 0.08, 'feature_fraction': 0.7, 'verbose': -1, 'n_jobs': -1,
    }
    model_p1 = lgb.train(quick_params, dtrain_p1, num_boost_round=150,
                          valid_sets=[dval_p1],
                          callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])

    # Select top features by gain
    importance = model_p1.feature_importance(importance_type='gain')
    feat_gain = sorted(zip(feat_cols, importance), key=lambda x: -x[1])
    n_keep = max(25, int(len(feat_cols) * 0.6))
    selected_feats = [f for f, g in feat_gain[:n_keep] if g > 0]
    if len(selected_feats) < 20:
        selected_feats = [f for f, g in feat_gain[:25]]

    # ── PASS 2: Refit with selected features + adaptive params ──
    X_train = np.nan_to_num(train_df[selected_feats].values, nan=0.)
    X_test = np.nan_to_num(test_df[selected_feats].values, nan=0.)

    params = get_params(tier, len(X_train))
    dtrain = lgb.Dataset(X_train[:val_cut], label=y_train[:val_cut])
    dval = lgb.Dataset(X_train[val_cut:], label=y_train[val_cut:], reference=dtrain)

    model = lgb.train(params, dtrain, num_boost_round=800,
                       valid_sets=[dval],
                       callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
    best_iter = model.best_iteration

    # ── Predictions ──
    pred_test = np.maximum(model.predict(X_test), 0)

    # ── Quantile models for prediction intervals ──
    params_q10 = {**params, 'objective': 'quantile', 'alpha': 0.1}
    params_q90 = {**params, 'objective': 'quantile', 'alpha': 0.9}
    params_q10['metric'] = 'quantile'
    params_q90['metric'] = 'quantile'

    model_q10 = lgb.train(params_q10, dtrain, num_boost_round=best_iter)
    model_q90 = lgb.train(params_q90, dtrain, num_boost_round=best_iter)

    pred_q10 = np.maximum(model_q10.predict(X_test), 0)
    pred_q90 = np.maximum(model_q90.predict(X_test), 0)
    coverage_80 = float(np.mean((y_test >= pred_q10) & (y_test <= pred_q90)) * 100)

    # ── Metrics ──
    mask = y_test > 0.5
    mae = float(np.mean(np.abs(y_test - pred_test)))
    mbe = float(np.mean(pred_test - y_test))
    rmse = float(np.sqrt(np.mean((y_test - pred_test)**2)))
    mape = float(np.mean(np.abs(y_test[mask]-pred_test[mask])/y_test[mask])*100) if mask.sum()>0 else 0

    ss_res = np.sum((y_test - pred_test)**2)
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else 0

    pct_err = np.abs(y_test[mask]-pred_test[mask])/y_test[mask]*100 if mask.sum()>0 else np.array([])
    within5 = float(np.mean(pct_err <= 5)*100) if len(pct_err)>0 else 0
    within10 = float(np.mean(pct_err <= 10)*100) if len(pct_err)>0 else 0

    results_s2.append({
        'msn': str(msn), 'phase': phase, 'tier': tier,
        'mape': round(mape, 2), 'mae': round(mae, 2), 'mbe': round(mbe, 2),
        'rmse': round(rmse, 2), 'r2': round(r2, 4),
        'mbe_pct': round(mbe/mean_demand*100, 2) if mean_demand > 0 else 0,
        'within5': round(within5, 1), 'within10': round(within10, 1),
        'coverage_80': round(coverage_80, 1),
        'mean_demand': round(mean_demand, 1),
        'zero_pct': round(float((y_test < 0.5).mean()*100), 1),
        'best_iter': best_iter,
        'n_features_selected': len(selected_feats),
        'n_features_initial': len(feat_cols),
        'train_days': n_train_days,
        'test_days': n_test_days,
        'train_rows': len(train_df),
        'test_rows': len(test_df),
    })

    print(f"  [{i+1:2d}/{len(eligible)}] {msn} ({phase:>4s} | {tier:>16s}) | "
          f"MAPE={mape:>6.1f}% MAE={mae:>8.1f} MBE={mbe:>+8.1f} R²={r2:.3f} "
          f"80%CI={coverage_80:.0f}% | days: train={n_train_days} test={n_test_days} "
          f"feats={len(selected_feats)}/{len(feat_cols)} iter={best_iter}")
    sys.stdout.flush()


# ════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ════════════════════════════════════════════════════════════════════
s2_df = pd.DataFrame(results_s2)
s2_df['msn'] = s2_df['msn'].astype(str)
s2_df.to_csv('/sessions/awesome-gifted-feynman/benchmark_strategy2.csv', index=False)
print(f"\nSaved: {len(s2_df)} meters → benchmark_strategy2.csv")


# ════════════════════════════════════════════════════════════════════
# [6/6] SUMMARY & COMPARISON WITH STRATEGY 1 v3
# ════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  STRATEGY 2 SUMMARY")
print(f"{'=' * 70}")

print(f"\n  Meters evaluated: {len(s2_df)}")
print(f"  Mean MAPE:   {s2_df['mape'].mean():.2f}%")
print(f"  Median MAPE: {s2_df['mape'].median():.2f}%")
print(f"  Mean MAE:    {s2_df['mae'].mean():.1f} Wh")
print(f"  Mean |MBE|:  {s2_df['mbe'].abs().mean():.1f} Wh")
print(f"  Mean R²:     {s2_df['r2'].mean():.4f}")
print(f"  80% CI cov:  {s2_df['coverage_80'].mean():.1f}%")
print(f"  Within 5%:   {s2_df['within5'].mean():.1f}%")
print(f"  Within 10%:  {s2_df['within10'].mean():.1f}%")

# Per-tier breakdown
print(f"\n  --- Per-Tier Breakdown ---")
for tier in ["HT (>5kWh)", "Large (1.5-5k)", "Medium (0.5-1.5k)", "Small (<500)"]:
    tm = s2_df[s2_df['tier'] == tier]
    if len(tm) == 0: continue
    print(f"  {tier:>20s} ({len(tm):2d} meters): "
          f"MAPE={tm['mape'].mean():>6.1f}% (median {tm['mape'].median():>5.1f}%) | "
          f"MAE={tm['mae'].mean():>6.0f} Wh | MBE={tm['mbe'].mean():>+6.0f} Wh | "
          f"R²={tm['r2'].mean():.3f}")

# ── Compare with Strategy 1 v3 ──
print(f"\n{'=' * 70}")
print("  STRATEGY 1 v3 vs STRATEGY 2 COMPARISON")
print(f"{'=' * 70}")

try:
    s1_df = pd.read_csv('/sessions/awesome-gifted-feynman/benchmark_strategy1_v3.csv')
    s1_df['msn'] = s1_df['msn'].astype(str)

    merged = s1_df[['msn','mape','mae','mbe','r2','tier']].rename(
        columns={'mape':'s1_mape','mae':'s1_mae','mbe':'s1_mbe','r2':'s1_r2'}
    ).merge(
        s2_df[['msn','mape','mae','mbe','r2']].rename(
            columns={'mape':'s2_mape','mae':'s2_mae','mbe':'s2_mbe','r2':'s2_r2'}
        ), on='msn'
    )

    print(f"\n  {'Metric':>15s} | {'S1 Chrono':>12s} | {'S2 Stratified':>12s} | {'Delta':>10s}")
    print(f"  {'-'*15} | {'-'*12} | {'-'*12} | {'-'*10}")

    for label, col, unit in [
        ('Mean MAPE', 'mape', '%'), ('Median MAPE', 'mape', '%'),
        ('Mean MAE', 'mae', ' Wh'), ('Mean |MBE|', 'mbe', ' Wh'),
    ]:
        if 'Median' in label:
            s1v = merged[f's1_{col}'].median()
            s2v = merged[f's2_{col}'].median()
        elif '|MBE|' in label:
            s1v = merged[f's1_{col}'].abs().mean()
            s2v = merged[f's2_{col}'].abs().mean()
        else:
            s1v = merged[f's1_{col}'].mean()
            s2v = merged[f's2_{col}'].mean()

        delta = s2v - s1v
        print(f"  {label:>15s} | {s1v:>9.1f}{unit:>3s} | {s2v:>9.1f}{unit:>3s} | {delta:>+8.1f}{unit}")

    s1r2 = merged['s1_r2'].mean()
    s2r2 = merged['s2_r2'].mean()
    print(f"  {'Mean R²':>15s} | {s1r2:>12.4f} | {s2r2:>12.4f} | {s2r2-s1r2:>+10.4f}")

    # Per-meter comparison
    print(f"\n  --- Per-Meter S1 vs S2 MAPE (sorted by delta) ---")
    merged['delta'] = merged['s2_mape'] - merged['s1_mape']
    s2_better = (merged['delta'] < 0).sum()
    s1_better = (merged['delta'] > 0).sum()

    for _, row in merged.sort_values('delta').iterrows():
        winner = 'S2✓' if row['delta'] < 0 else 'S1✓'
        print(f"  {winner} {row['msn']:>10s} ({row['tier']:>16s}): "
              f"S1={row['s1_mape']:>6.1f}% S2={row['s2_mape']:>6.1f}% ({row['delta']:>+6.1f}%)")

    print(f"\n  S2 better: {s2_better}/{len(merged)} meters")
    print(f"  S1 better: {s1_better}/{len(merged)} meters")

    # Per-tier comparison
    print(f"\n  --- Per-Tier S1 vs S2 ---")
    for tier in ["HT (>5kWh)", "Large (1.5-5k)", "Medium (0.5-1.5k)", "Small (<500)"]:
        tm = merged[merged['tier'] == tier]
        if len(tm) == 0: continue
        s1m = tm['s1_mape'].mean()
        s2m = tm['s2_mape'].mean()
        print(f"  {tier:>20s}: S1={s1m:>6.1f}% → S2={s2m:>6.1f}% (Δ={s2m-s1m:>+5.1f}%)")

except Exception as e:
    print(f"  Could not compare with S1: {e}")

print(f"\nTotal time: {time.time()-t0:.1f}s")
