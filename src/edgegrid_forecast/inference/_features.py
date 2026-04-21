"""
Feature pipeline for Strategy 1 v4 inference (replicated from
benchmarks/build_strategy1_v4_engine.py so the production wrapper has no
runtime dependency on the benchmark script).

Building this as a standalone module keeps production inference decoupled from
the benchmark harness — the benchmark script remains the source of truth for
the headline numbers; this module is the source of truth for what goes into
models/v4/*.joblib at inference time.

If the feature recipe in the benchmark changes, update this file to match and
bump MODEL_VERSION in v4_predict.py. A unit-test in tests/test_v4_predict.py
asserts MAPE parity on a smoke-test meter to guard against drift.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════════════
# HOLIDAYS (Visakhapatnam / Andhra Pradesh — mirrors benchmark)
# ════════════════════════════════════════════════════════════════════════════
HOLIDAYS = {
    "2024-10-02": "Gandhi Jayanti", "2024-10-12": "Dussehra",
    "2024-10-31": "Sardar Patel", "2024-11-01": "Diwali",
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
HOLIDAY_DATES = set(pd.to_datetime(list(HOLIDAYS.keys())).date)


def tod_multiplier(hour: int | float) -> float:
    """APEPDCL ToD tariff multiplier (mirrors benchmark)."""
    if 6 <= hour < 18:
        return 1.0
    elif 18 <= hour < 22:
        return 1.2
    return 0.9


# ════════════════════════════════════════════════════════════════════════════
# TIME BLOCKS (BESS dispatch windows)
# ════════════════════════════════════════════════════════════════════════════
TIME_BLOCKS = [
    ("night",   lambda h: (h >= 22) | (h < 6)),
    ("morning", lambda h: (h >= 6) & (h < 10)),
    ("solar",   lambda h: (h >= 10) & (h < 18)),
    ("peak",    lambda h: (h >= 18) & (h < 22)),
]


def block_label_for(ts_like) -> str:
    """Return the BESS time-block label for a single timestamp."""
    h = pd.Timestamp(ts_like).hour
    for name, pred in TIME_BLOCKS:
        if pred(np.array([h]))[0]:
            return name
    return "unknown"


# ════════════════════════════════════════════════════════════════════════════
# WEATHER LOADER (cache-first, Open-Meteo archive fallback)
# ════════════════════════════════════════════════════════════════════════════
def fetch_weather_expanded(
    lat: float = 17.7,
    lon: float = 83.2,
    start: str = "2024-10-01",
    end: str = "2026-03-01",
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Cached 30-min weather DataFrame for Visakhapatnam.

    Tries cache first; if absent, pulls Open-Meteo archive in 6-month chunks
    and writes a parquet cache. Returns a DataFrame with derived weather
    signals (deltas, rolling means, heat index, diffuse fraction, etc.).
    """
    vars_csv = (
        "temperature_2m,relative_humidity_2m,dewpoint_2m,"
        "surface_pressure,cloud_cover,precipitation,wind_speed_10m,"
        "shortwave_radiation,direct_radiation,diffuse_radiation,"
        "direct_normal_irradiance"
    )
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parents[3] / "data" / "external" / "weather"
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"visakhapatnam_expanded_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    chunks = []
    current = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    while current < end_dt:
        ce = min(current + pd.DateOffset(months=6) - pd.Timedelta(days=1), end_dt)
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={current:%Y-%m-%d}&end_date={ce:%Y-%m-%d}"
            f"&hourly={vars_csv}&timezone=Asia%2FKolkata"
        )
        try:
            r = urllib.request.urlopen(url, timeout=45)
            d = json.loads(r.read())
            h = d["hourly"]
            chunks.append(pd.DataFrame({
                "ts": pd.to_datetime(h["time"]),
                "temperature": h["temperature_2m"],
                "humidity": h["relative_humidity_2m"],
                "dewpoint": h["dewpoint_2m"],
                "pressure": h["surface_pressure"],
                "cloud_cover": h["cloud_cover"],
                "precipitation": h["precipitation"],
                "wind_speed": h["wind_speed_10m"],
                "ghi": h["shortwave_radiation"],
                "dni": h["direct_normal_irradiance"],
                "dhi": h["diffuse_radiation"],
                "direct_rad": h["direct_radiation"],
            }))
        except Exception as e:
            print(f"  [weather] chunk fail ({current:%Y-%m-%d}): {e}")
        current = ce + pd.Timedelta(days=1)

    if not chunks:
        raise RuntimeError(
            f"Weather fetch failed and no cache at {cache_path}. "
            "Either run the benchmark to populate cache or supply --weather."
        )

    wx = pd.concat(chunks, ignore_index=True).drop_duplicates("ts").sort_values("ts")
    wx_30m = wx.set_index("ts").resample("30min").ffill().reset_index()
    wx_30m = _add_weather_derivatives(wx_30m)

    cache_dir.mkdir(parents=True, exist_ok=True)
    wx_30m.to_parquet(cache_path, index=False)
    return wx_30m


def _add_weather_derivatives(wx: pd.DataFrame) -> pd.DataFrame:
    wx = wx.copy()
    wx["pressure_delta_3h"] = wx["pressure"].diff(6)
    wx["temp_delta_3h"]     = wx["temperature"].diff(6)
    wx["temp_rmean_6h"]     = wx["temperature"].rolling(12, min_periods=1).mean()
    wx["ghi_rmean_6h"]      = wx["ghi"].rolling(12, min_periods=1).mean()
    wx["cloud_delta_3h"]    = wx["cloud_cover"].diff(6)
    wx["is_raining"]        = (wx["precipitation"] > 0).astype(int)
    wx["diffuse_fraction"]  = (wx["dhi"] / wx["ghi"].clip(lower=1)).clip(0, 1)
    wx["heat_index"] = (
        wx["temperature"]
        + 0.33 * (wx["humidity"] / 100 * 6.105
                  * np.exp(17.27 * wx["temperature"] / (237.7 + wx["temperature"])))
        - 4.0
    )
    return wx


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADER (SP + TP parquets, unified schema)
# ════════════════════════════════════════════════════════════════════════════
def load_meter_data(data_dir: Path | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load clean SP + TP parquets + meter profile.

    Returns (all_df, profile) with unified schema:
      all_df: ts, demand_wh, voltage, msn, phase, tier
      profile: meter metadata (msn, tier, eligible, mean_demand_wh, ...)
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[3] / "data" / "raw"
    data_dir = Path(data_dir)

    profile = pd.read_parquet(data_dir / "meter_profile.parquet")
    profile["msn"] = profile["msn"].astype(str)

    sp = pd.read_parquet(data_dir / "sp_data.parquet")
    sp["msn"] = sp["msn"].astype(str)
    sp["voltage"] = sp["v_ave"]
    sp_clean = sp[["ts", "demand_wh", "voltage", "msn"]].copy()
    sp_clean["phase"] = "1PH"

    tp = pd.read_parquet(data_dir / "tp_data.parquet")
    tp["msn"] = tp["msn"].astype(str)
    tp["voltage"] = tp["v_avg"]
    tp_clean = tp[["ts", "demand_wh", "voltage", "msn"]].copy()
    tp_clean["phase"] = "3PH"

    all_df = pd.concat([sp_clean, tp_clean], ignore_index=True)
    all_df = all_df.dropna(subset=["demand_wh"]).sort_values(["msn", "ts"]).reset_index(drop=True)

    tier_map = dict(zip(profile["msn"], profile["tier"]))
    all_df["tier"] = all_df["msn"].map(tier_map)
    return all_df, profile


def compute_fleet_aggregate(all_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-meter mean/std per timestamp (min 5 meters for stability)."""
    fleet = all_df.groupby("ts")["demand_wh"].agg(["mean", "std", "count"]).reset_index()
    fleet.columns = ["ts", "fleet_mean", "fleet_std", "fleet_count"]
    return fleet[fleet["fleet_count"] >= 5].copy()


# ════════════════════════════════════════════════════════════════════════════
# FEATURE BUILDER (v4 + S1-specific)
# ════════════════════════════════════════════════════════════════════════════
def build_features_v4_s1(
    df_meter: pd.DataFrame,
    weather_expanded: pd.DataFrame,
    fleet_agg: pd.DataFrame,
    *,
    drop_warmup: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build v4 feature set with S1-specific anomaly + similar-day features.

    Parameters
    ----------
    df_meter : DataFrame with ts, demand_wh, voltage (and optionally tier/phase/msn)
    weather_expanded : DataFrame from fetch_weather_expanded()
    fleet_agg : DataFrame from compute_fleet_aggregate()
    drop_warmup : If True (train-time), strip first 336 rows (7 days) due to
                  unreliable lags. Set False when building features for
                  continuation/prediction where the caller has provided
                  sufficient historical context.

    Returns
    -------
    feat_df : DataFrame with features + target (+ ts retained)
    feat_cols : list of feature column names (numeric, excludes metadata)
    """
    df = df_meter.sort_values("ts").reset_index(drop=True).copy()

    # ── Temporal (10) ──
    df["hour"]       = df["ts"].dt.hour + df["ts"].dt.minute / 60
    df["dow"]        = df["ts"].dt.dayofweek
    df["month"]      = df["ts"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)

    # ── Lags (9) ──
    for lag in [1, 2, 4, 6, 12, 24, 48, 96, 336]:
        df[f"lag_{lag}"] = df["demand_wh"].shift(lag)

    # ── Rolling (8) ──
    for win in [6, 12, 48, 336]:
        s = df["demand_wh"].shift(1)
        df[f"rmean_{win}"] = s.rolling(win, min_periods=1).mean()
        df[f"rstd_{win}"]  = s.rolling(win, min_periods=1).std()

    # ── Momentum + longer rolling (5) ──
    df["demand_diff_1"] = df["demand_wh"].diff(1)
    df["demand_diff_4"] = df["demand_wh"].diff(4)
    s1 = df["demand_wh"].shift(1)
    df["rmean_14d"]    = s1.rolling(672,  min_periods=48).mean()
    df["rmean_30d"]    = s1.rolling(1440, min_periods=48).mean()
    df["trend_ratio"]  = df["rmean_14d"] / df["rmean_30d"].clip(lower=1)

    # ── Weather merge (20) ──
    wx_cols = [
        "ts", "temperature", "humidity", "dewpoint", "pressure", "cloud_cover",
        "precipitation", "wind_speed", "ghi", "dni", "dhi", "direct_rad",
        "pressure_delta_3h", "temp_delta_3h", "ghi_rmean_6h", "cloud_delta_3h",
        "is_raining", "diffuse_fraction", "heat_index", "temp_rmean_6h",
    ]
    wx = weather_expanded[wx_cols].copy()
    df = df.merge(wx, on="ts", how="left")
    for c in wx_cols[1:]:
        df[c] = df[c].ffill().bfill()

    # ── Derived weather (4) ──
    df["cooling_deg"] = (df["temperature"] - 25).clip(lower=0)
    df["heating_deg"] = (18 - df["temperature"]).clip(lower=0)
    df["hour_x_temp"] = df["hour"] * df["temperature"] / 100
    df["peak_x_temp"] = ((df["hour"] >= 12) & (df["hour"] <= 17)).astype(int) * df["temperature"]

    # ── CEA-canonical HDD/CDD at 21°C threshold (W1) ──
    # Per CEA 19th EPS / MoP 2024 Guidelines, base temperature = 21°C.
    # HDD and CDD at this threshold are the statutory load-temperature sensitivity
    # features. We keep the existing 18°C/25°C legacy features so the screener can
    # compare; in practice the 21°C-based features tend to dominate India-wide.
    # See docs/V5_CEA_ALIGNMENT.md section W1 for the rationale.
    df["hdd_21"] = (21.0 - df["temperature"]).clip(lower=0)
    df["cdd_21"] = (df["temperature"] - 21.0).clip(lower=0)
    # Accumulated CDD captures multi-day heat-wave pressure on AC demand:
    # one 32°C day is load-linear, three in a row is super-linear because
    # buildings & inverter-AC thermal mass haven't recovered overnight.
    df["cdd_21_rmean_48"]  = df["cdd_21"].shift(1).rolling(48,  min_periods=6).mean()   # 24h
    df["cdd_21_rmean_336"] = df["cdd_21"].shift(1).rolling(336, min_periods=48).mean()  # 7d
    # Interactions: CDD × hour-of-day (catches evening AC ramp post-4pm) and
    # CDD during the 18–22h peak block (captures hot-evening peak flare).
    df["cdd_21_x_hour"] = df["cdd_21"] * df["hour"] / 24
    df["cdd_21_x_peak"] = ((df["hour"] >= 18) & (df["hour"] < 22)).astype(int) * df["cdd_21"]

    # ── S1 anomaly features vs trailing 24h baseline (4) ──
    df["temp_anom_24h"]  = df["temperature"] - df["temperature"].shift(1).rolling(48, min_periods=12).mean()
    df["hum_anom_24h"]   = df["humidity"]    - df["humidity"].shift(1).rolling(48, min_periods=12).mean()
    df["ghi_anom_24h"]   = df["ghi"]         - df["ghi"].shift(1).rolling(48, min_periods=12).mean()
    df["cloud_anom_24h"] = df["cloud_cover"] - df["cloud_cover"].shift(1).rolling(48, min_periods=12).mean()

    # ── Holidays (2) ──
    df["is_holiday"] = df["ts"].dt.date.isin(HOLIDAY_DATES).astype(int)
    df["near_holiday"] = 0
    for hd in HOLIDAY_DATES:
        m = (df["ts"].dt.date >= hd - pd.Timedelta(days=1)) & \
            (df["ts"].dt.date <= hd + pd.Timedelta(days=1))
        df.loc[m, "near_holiday"] = 1

    # ── ToD tariff (2) ──
    df["tod_multiplier"] = df["ts"].dt.hour.apply(tod_multiplier)
    df["is_peak"]        = ((df["ts"].dt.hour >= 18) & (df["ts"].dt.hour < 22)).astype(int)

    # ── Voltage (3) ──
    df["voltage_lag1"]     = df["voltage"].shift(1)
    df["voltage_rstd_6"]   = df["voltage"].shift(1).rolling(6, min_periods=1).std()
    df["voltage_rmean_48"] = df["voltage"].shift(1).rolling(48, min_periods=1).mean()

    # ── Seasonal deviation (1) ──
    shm = df.groupby("hour")["demand_wh"].transform(
        lambda x: x.shift(1).expanding(min_periods=48).mean()
    )
    df["deviation_from_hourly"] = df["demand_wh"].shift(1) - shm

    # ── Fleet (3) ──
    fleet = fleet_agg[["ts", "fleet_mean", "fleet_std"]].copy()
    df = df.merge(fleet, on="ts", how="left")
    df["fleet_mean"] = df["fleet_mean"].ffill().bfill().fillna(0)
    df["fleet_std"]  = df["fleet_std"].ffill().bfill().fillna(0)
    df["fleet_mean_lag1"] = df["fleet_mean"].shift(1)
    df["fleet_std_lag1"]  = df["fleet_std"].shift(1)
    df["vs_fleet_ratio"]  = df["demand_wh"].shift(1) / df["fleet_mean_lag1"].clip(lower=1)

    # ── Similar-day lookback (k=5, hour × dow matched) ──
    df["hour_int"]     = df["ts"].dt.hour.astype(int)
    df["dow_int"]      = df["ts"].dt.dayofweek.astype(int)
    df["hour_dow_key"] = df["hour_int"].astype(str) + "_" + df["dow_int"].astype(str)
    df["similar_day_k5"] = df.groupby("hour_dow_key")["demand_wh"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).mean()
    )
    df = df.drop(columns=["hour_int", "dow_int", "hour_dow_key"])

    if drop_warmup:
        df = df.iloc[336:].reset_index(drop=True)

    df["target"] = df["demand_wh"]

    exclude = {
        "ts", "demand_wh", "target", "msn", "phase", "voltage", "scno", "date",
        "tier", "fleet_mean", "fleet_std",
    }
    feat_cols = [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in ["float64", "int64", "int32", "float32", "uint8"]
    ]
    return df, feat_cols


# ════════════════════════════════════════════════════════════════════════════
# SPLIT + TIER PARAMS (mirrors benchmark)
# ════════════════════════════════════════════════════════════════════════════
def split_chronological(df: pd.DataFrame, train_frac: float = 0.75):
    df = df.sort_values("ts").reset_index(drop=True)
    t_min = df["ts"].min()
    t_max = df["ts"].max()
    span = (t_max - t_min).total_seconds()
    cutoff = t_min + pd.Timedelta(seconds=span * train_frac)
    train = df[df["ts"] < cutoff].copy()
    test = df[df["ts"] >= cutoff].copy()
    n_train_days = (train["ts"].max() - train["ts"].min()).days if len(train) else 0
    n_test_days = (test["ts"].max() - test["ts"].min()).days if len(test) else 0
    return train, test, n_train_days, n_test_days, cutoff


def get_params(tier: str, n_samples: int) -> dict:
    p = {
        "objective": "regression", "metric": "mae", "verbose": -1, "n_jobs": -1,
        "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 50,
        "feature_fraction": 0.6, "bagging_fraction": 0.7, "bagging_freq": 5,
        "lambda_l1": 0.1, "lambda_l2": 1.0, "max_depth": 8,
    }
    if tier == "HT (>5kWh)":
        p.update({"num_leaves": 47, "min_child_samples": 30,
                  "feature_fraction": 0.7, "learning_rate": 0.04})
    elif tier == "Small (<500)":
        p.update({"num_leaves": 15, "min_child_samples": 80,
                  "feature_fraction": 0.5, "max_depth": 6,
                  "lambda_l1": 0.5, "lambda_l2": 5.0})
    if n_samples < 3000:
        p["num_leaves"] = min(p["num_leaves"], 15)
        p["min_child_samples"] = max(p["min_child_samples"], 100)
    return p


# ════════════════════════════════════════════════════════════════════════════
# METRICS (mirrors benchmark)
# ════════════════════════════════════════════════════════════════════════════
def calc_metrics(y_true, y_pred, q10=None, q90=None) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = y_true > 0.5
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mbe = float(np.mean(y_pred - y_true))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = (
        float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)
        if mask.sum() > 0 else 0.0
    )
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    pct = (
        np.abs(y_true[mask] - y_pred[mask]) / y_true[mask] * 100
        if mask.sum() > 0 else np.array([])
    )
    w5 = float(np.mean(pct <= 5) * 100) if len(pct) > 0 else 0.0
    w10 = float(np.mean(pct <= 10) * 100) if len(pct) > 0 else 0.0
    cov = (
        float(np.mean((y_true >= q10) & (y_true <= q90)) * 100)
        if q10 is not None else 0.0
    )
    return {
        "mape": round(mape, 2), "mae": round(mae, 2), "mbe": round(mbe, 2),
        "rmse": round(rmse, 2), "r2": round(r2, 4),
        "within5": round(w5, 1), "within10": round(w10, 1),
        "coverage_80": round(cov, 1),
    }


def calc_block_metrics(ts, y_true, y_pred) -> dict:
    hours = pd.to_datetime(ts).hour.values
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    out = {}
    for name, pred_fn in TIME_BLOCKS:
        m = pred_fn(hours)
        if m.sum() < 5:
            out[f"{name}_mape"] = float("nan")
            out[f"{name}_mae"] = float("nan")
            out[f"{name}_mbe"] = float("nan")
            out[f"{name}_n"] = int(m.sum())
            out[f"{name}_mean_demand"] = float("nan")
            continue
        yt, yp = y_true[m], y_pred[m]
        mask = yt > 0.5
        mape = (
            float(np.mean(np.abs(yt[mask] - yp[mask]) / yt[mask]) * 100)
            if mask.sum() > 0 else 0.0
        )
        out[f"{name}_mape"] = round(mape, 2)
        out[f"{name}_mae"]  = round(float(np.mean(np.abs(yt - yp))), 2)
        out[f"{name}_mbe"]  = round(float(np.mean(yp - yt)), 2)
        out[f"{name}_n"]    = int(m.sum())
        out[f"{name}_mean_demand"] = round(float(np.mean(yt)), 1)
    return out
