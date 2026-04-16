"""
EdgeGrid Forecast Engine — Multi-Strategy Holdout Benchmark
============================================================
Trains LightGBM per meter × per strategy, computes MAPE/MAE/MBE.

Strategies:
  1. Chronological Cutoff  — train first 75%, predict last 25%
  2. Stratified Temporal   — every 4th complete day held out
  3. Rolling Origin        — expanding window, retrain monthly, predict next 2 weeks

Metrics:
  - MAPE (Mean Absolute Percentage Error) — scale-independent accuracy
  - MAE  (Mean Absolute Error)            — absolute Wh error per interval
  - MBE  (Mean Bias Error)                — directional bias (+ = over-forecast)
"""

import pandas as pd
import numpy as np
import warnings, time, json, sys
from pathlib import Path

warnings.filterwarnings("ignore")

# ── 1. DATA LOADING & NORMALIZATION ──────────────────────────────

def load_and_normalize():
    """Load SP + TP parquets, dedup, normalize into common schema."""
    
    # Vendor mapping
    vendor = pd.read_excel(
        "/sessions/awesome-gifted-feynman/mnt/uploads/SCS_PMSGVENDOR-7f720bd5.xlsx"
    )
    vendor['MTR_SNO'] = vendor['MTR_SNO'].astype(str)
    phase_map = dict(zip(vendor['MTR_SNO'], vendor['Phase']))
    scno_map = dict(zip(vendor['MTR_SNO'], vendor['SCNO']))
    
    frames = []
    
    # ── SP data ──
    df_sp = pd.read_parquet("/sessions/awesome-gifted-feynman/sp_data.parquet")
    df_sp['ts'] = pd.to_datetime(df_sp['ts'])
    df_sp['wh_imp'] = pd.to_numeric(
        df_sp['wh_imp'].str.replace(',', ''), errors='coerce'
    )
    df_sp['v_ave'] = pd.to_numeric(
        df_sp['v_ave'].str.replace(',', ''), errors='coerce'
    )
    for msn in df_sp['msn'].unique():
        sub = df_sp[df_sp['msn'] == msn][['ts', 'wh_imp', 'v_ave']].copy()
        sub = sub.drop_duplicates(subset='ts', keep='first').sort_values('ts')
        sub.rename(columns={'wh_imp': 'demand_wh', 'v_ave': 'voltage'}, inplace=True)
        sub['msn'] = msn
        sub['phase_type'] = '1PH'
        frames.append(sub)
    
    # ── TP data ──
    df_tp = pd.read_parquet("/sessions/awesome-gifted-feynman/tp_data.parquet")
    df_tp['ts'] = pd.to_datetime(df_tp['ts'])
    df_tp['wh_imp'] = pd.to_numeric(
        df_tp['wh_imp'].str.replace(',', ''), errors='coerce'
    )
    # Average of 3 phase voltages
    for v in ['v_r', 'v_y', 'v_b']:
        df_tp[v] = pd.to_numeric(df_tp[v].str.replace(',', ''), errors='coerce')
    df_tp['voltage'] = df_tp[['v_r', 'v_y', 'v_b']].mean(axis=1)
    
    for msn in df_tp['msn'].unique():
        sub = df_tp[df_tp['msn'] == msn][['ts', 'wh_imp', 'voltage']].copy()
        sub = sub.drop_duplicates(subset='ts', keep='first').sort_values('ts')
        sub.rename(columns={'wh_imp': 'demand_wh'}, inplace=True)
        sub['msn'] = msn
        sub['phase_type'] = phase_map.get(msn, '3PH')
        frames.append(sub)
    
    df = pd.concat(frames, ignore_index=True)
    df['scno'] = df['msn'].map(scno_map)
    
    # Drop rows with null demand
    df = df.dropna(subset=['demand_wh'])
    
    return df


# ── 2. FEATURE ENGINEERING ───────────────────────────────────────

def build_features(df_meter):
    """
    Build features for a single meter's time series.
    Input: DataFrame with columns [ts, demand_wh, voltage]
    Returns: DataFrame with features + target, NaN rows from lags dropped.
    """
    df = df_meter.copy()
    df = df.sort_values('ts').reset_index(drop=True)
    
    # ── Temporal features ──
    df['hour'] = df['ts'].dt.hour + df['ts'].dt.minute / 60  # 0, 0.5, 1, 1.5, ...
    df['hour_int'] = df['ts'].dt.hour
    df['dow'] = df['ts'].dt.dayofweek
    df['month'] = df['ts'].dt.month
    df['day_of_year'] = df['ts'].dt.dayofyear
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # ── Lag features (in 30-min steps) ──
    for lag in [1, 2, 4, 6, 12, 24, 48, 96, 336]:
        # 1=30min, 2=1h, 4=2h, 6=3h, 12=6h, 24=12h, 48=24h, 96=48h, 336=7days
        df[f'lag_{lag}'] = df['demand_wh'].shift(lag)
    
    # ── Rolling features (windows in 30-min steps) ──
    for win in [6, 12, 24, 48, 96, 336]:
        # 6=3h, 12=6h, 24=12h, 48=24h, 96=48h, 336=7days
        df[f'rmean_{win}'] = df['demand_wh'].shift(1).rolling(win, min_periods=1).mean()
        df[f'rstd_{win}'] = df['demand_wh'].shift(1).rolling(win, min_periods=1).std()
        df[f'rmax_{win}'] = df['demand_wh'].shift(1).rolling(win, min_periods=1).max()
        df[f'rmin_{win}'] = df['demand_wh'].shift(1).rolling(win, min_periods=1).min()
    
    # ── Consumption pattern features ──
    daily_mean = df.groupby(df['ts'].dt.date)['demand_wh'].transform('mean')
    df['deviation_from_daily_mean'] = df['demand_wh'].shift(1) - daily_mean.shift(1)
    
    hourly_profile = df.groupby('hour_int')['demand_wh'].transform('mean')
    df['hourly_profile'] = hourly_profile
    
    # ── Voltage feature ──
    df['voltage_lag1'] = df['voltage'].shift(1)
    
    # Drop warm-up rows (need 7 days = 336 half-hours of history)
    df = df.iloc[336:].reset_index(drop=True)
    
    # Target
    df['target'] = df['demand_wh']
    
    # Feature columns
    feature_cols = [c for c in df.columns 
                    if c not in ['ts', 'demand_wh', 'target', 'msn', 'phase_type', 'scno', 'voltage']]
    
    return df, feature_cols


# ── 3. HOLDOUT STRATEGIES ────────────────────────────────────────

def split_chronological(df, train_frac=0.75):
    """
    Strategy 1: Chronological Cutoff
    Train on first 75% of timeline, test on last 25%.
    Returns list of (train_df, test_df) tuples (single split).
    """
    n = len(df)
    cut = int(n * train_frac)
    return [(df.iloc[:cut], df.iloc[cut:])]


def split_stratified_temporal(df, holdout_every=4):
    """
    Strategy 2: Stratified Temporal
    Every Nth complete day is held out. Both splits see all seasons.
    Returns list of (train_df, test_df) tuples (single split).
    """
    df = df.copy()
    df['date'] = df['ts'].dt.date
    
    # Get complete days (48 intervals for 30-min data)
    day_counts = df.groupby('date').size()
    complete_days = sorted(day_counts[day_counts >= 44].index)  # allow slight gaps
    
    holdout_days = set(complete_days[::holdout_every])
    
    train = df[~df['date'].isin(holdout_days)].drop(columns='date')
    test = df[df['date'].isin(holdout_days)].drop(columns='date')
    
    return [(train, test)]


def split_rolling_origin(df, initial_train_days=90, step_days=30, horizon_days=14):
    """
    Strategy 3: Rolling Origin (Walk-Forward)
    Start with initial_train_days, step forward by step_days, predict horizon_days.
    Returns list of (train_df, test_df) tuples (multiple folds).
    """
    df = df.copy()
    df['date'] = df['ts'].dt.date
    dates = sorted(df['date'].unique())
    
    splits = []
    start_idx = 0
    train_end_idx = initial_train_days
    
    while train_end_idx + horizon_days <= len(dates):
        train_dates = set(dates[start_idx:train_end_idx])
        test_dates = set(dates[train_end_idx:train_end_idx + horizon_days])
        
        train = df[df['date'].isin(train_dates)].drop(columns='date')
        test = df[df['date'].isin(test_dates)].drop(columns='date')
        
        if len(train) > 0 and len(test) > 0:
            splits.append((train, test))
        
        train_end_idx += step_days
    
    return splits


# ── 4. MODEL TRAINING & EVALUATION ──────────────────────────────

def compute_metrics(y_true, y_pred):
    """Compute MAPE, MAE, MBE for a prediction set."""
    mask = y_true > 0  # avoid division by zero for MAPE
    
    mae = np.mean(np.abs(y_true - y_pred))
    mbe = np.mean(y_pred - y_true)  # positive = over-forecast
    
    if mask.sum() > 0:
        mape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100
    else:
        mape = np.nan
    
    return {'mape': round(mape, 2), 'mae': round(mae, 2), 'mbe': round(mbe, 2)}


def train_and_evaluate(splits, feature_cols, strategy_name):
    """
    Train LightGBM on each split, return per-fold metrics.
    """
    import lightgbm as lgb
    
    fold_results = []
    
    for fold_idx, (train_df, test_df) in enumerate(splits):
        if len(train_df) < 200 or len(test_df) < 48:
            continue
        
        X_train = train_df[feature_cols].values
        y_train = train_df['target'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['target'].values
        
        # Handle any remaining NaN
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1,
        }
        
        model = lgb.train(
            params, dtrain,
            num_boost_round=300,
            valid_sets=[dtrain],
            callbacks=[lgb.log_evaluation(0)],
        )
        
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)  # demand can't be negative
        
        metrics = compute_metrics(y_test, y_pred)
        metrics['fold'] = fold_idx
        metrics['train_rows'] = len(train_df)
        metrics['test_rows'] = len(test_df)
        metrics['test_days'] = test_df['ts'].dt.date.nunique()
        
        fold_results.append(metrics)
    
    return fold_results


# ── 5. MAIN BENCHMARK LOOP ──────────────────────────────────────

def run_benchmark(min_days=180):
    """Run all 3 strategies across all eligible meters."""
    
    print("=" * 70)
    print("  EdgeGrid Multi-Strategy Holdout Benchmark")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading and normalizing SP + TP data...")
    df_all = load_and_normalize()
    print(f"  Total rows: {len(df_all):,}, Meters: {df_all['msn'].nunique()}")
    
    # Filter to meters with enough data
    meter_days = df_all.groupby('msn')['ts'].agg(
        lambda x: (x.max() - x.min()).days
    )
    eligible = meter_days[meter_days >= min_days].index.tolist()
    print(f"\n[2/5] Filtering to meters with >= {min_days} days of data...")
    print(f"  Eligible: {len(eligible)} of {df_all['msn'].nunique()} meters")
    
    strategies = {
        'chronological': split_chronological,
        'stratified': split_stratified_temporal,
        'rolling_origin': split_rolling_origin,
    }
    
    all_results = []
    total_meters = len(eligible)
    
    for strat_name, split_fn in strategies.items():
        print(f"\n[3/5] Running strategy: {strat_name.upper()}")
        print(f"  {'─' * 50}")
        
        for m_idx, msn in enumerate(eligible):
            meter_data = df_all[df_all['msn'] == msn].copy()
            phase = meter_data['phase_type'].iloc[0]
            
            # Build features
            feat_df, feat_cols = build_features(meter_data)
            
            if len(feat_df) < 500:
                continue
            
            # Split
            splits = split_fn(feat_df)
            
            # Train & evaluate
            fold_results = train_and_evaluate(splits, feat_cols, strat_name)
            
            for fr in fold_results:
                fr['strategy'] = strat_name
                fr['msn'] = msn
                fr['phase_type'] = phase
                fr['total_rows'] = len(feat_df)
                all_results.append(fr)
            
            # Aggregate for this meter
            if fold_results:
                avg_mape = np.mean([r['mape'] for r in fold_results])
                avg_mae = np.mean([r['mae'] for r in fold_results])
                avg_mbe = np.mean([r['mbe'] for r in fold_results])
                n_folds = len(fold_results)
                print(f"  {msn} ({phase:>7s}) | folds={n_folds:>2d} | "
                      f"MAPE={avg_mape:>6.2f}% | MAE={avg_mae:>8.1f} Wh | "
                      f"MBE={avg_mbe:>+8.1f} Wh")
            
            sys.stdout.flush()
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("/sessions/awesome-gifted-feynman/holdout_benchmark_results.csv", index=False)
    results_df.to_parquet("/sessions/awesome-gifted-feynman/holdout_benchmark_results.parquet", index=False)
    
    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("  BENCHMARK SUMMARY")
    print(f"{'=' * 70}")
    
    for strat in ['chronological', 'stratified', 'rolling_origin']:
        sub = results_df[results_df['strategy'] == strat]
        if len(sub) == 0:
            continue
        
        # Aggregate: mean of per-meter averages
        meter_avg = sub.groupby('msn').agg({
            'mape': 'mean', 'mae': 'mean', 'mbe': 'mean'
        })
        
        print(f"\n  {strat.upper()}")
        print(f"  Meters evaluated: {len(meter_avg)}")
        print(f"  MAPE  — mean: {meter_avg['mape'].mean():.2f}%, "
              f"median: {meter_avg['mape'].median():.2f}%, "
              f"p90: {meter_avg['mape'].quantile(0.9):.2f}%")
        print(f"  MAE   — mean: {meter_avg['mae'].mean():.1f} Wh, "
              f"median: {meter_avg['mae'].median():.1f} Wh")
        print(f"  MBE   — mean: {meter_avg['mbe'].mean():+.1f} Wh "
              f"({'over-forecast' if meter_avg['mbe'].mean() > 0 else 'under-forecast'})")
    
    return results_df


if __name__ == '__main__':
    results = run_benchmark(min_days=180)
    print(f"\nResults saved to holdout_benchmark_results.csv")
    print(f"Total result rows: {len(results)}")
