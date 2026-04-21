"""
Feature pipeline for v5 — extends v4 with:
  * Hour-of-year seasonality (annual cycle)
  * Robust rolling medians (outlier-resistant)
  * Exponential moving averages (faster adaptation)
  * Persistence baseline features (strong priors)
  * Longer seasonality windows (60d / 90d same-DOW-hour)
  * Weather lag features (yesterday's conditions)
  * Variability ratio features (recent vs long-run)

Backward-compatible with v4 feature names — v5 is a strict superset so v4
bundles remain loadable for side-by-side benchmarking.
"""
from __future__ import annotations

from typing import Tuple, List

import numpy as np
import pandas as pd

from ._features import (
    HOLIDAY_DATES,
    TIME_BLOCKS,
    block_label_for,
    build_features_v4_s1,
    calc_block_metrics,
    compute_fleet_aggregate,
    fetch_weather_expanded,
    load_meter_data,
    split_chronological,
    tod_multiplier,
)

# Re-export v4 symbols so v5 callers don't need both modules.
__all__ = [
    "HOLIDAY_DATES", "TIME_BLOCKS", "block_label_for",
    "compute_fleet_aggregate", "fetch_weather_expanded", "load_meter_data",
    "split_chronological",
    "build_features_v5",
    "calc_metrics_v5",
    "calc_block_metrics",
    "get_params_v5",
    "detect_intermittency",
]


# ════════════════════════════════════════════════════════════════════════════
# INTERMITTENCY DETECTION
# ════════════════════════════════════════════════════════════════════════════
def detect_intermittency(y: np.ndarray, thresh_wh: float = 0.5) -> dict:
    """Classify a meter's intermittency profile.

    Returns a dict with:
      zero_pct : fraction of rows below `thresh_wh`
      regime   : "continuous" (<5% zeros) | "sparse" (5-15%) | "intermittent" (>15%)
      use_two_stage : True when a binary-then-regressor stack is warranted.
    """
    y = np.asarray(y, dtype=np.float64)
    zero_pct = float((y < thresh_wh).mean() * 100)
    if zero_pct >= 15:
        regime = "intermittent"
    elif zero_pct >= 5:
        regime = "sparse"
    else:
        regime = "continuous"
    return {
        "zero_pct": round(zero_pct, 2),
        "regime": regime,
        "use_two_stage": regime == "intermittent",
    }


# ════════════════════════════════════════════════════════════════════════════
# FEATURE BUILDER v5
# ════════════════════════════════════════════════════════════════════════════
def build_features_v5(
    df_meter: pd.DataFrame,
    weather_expanded: pd.DataFrame,
    fleet_agg: pd.DataFrame,
    *,
    drop_warmup: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """v5 feature builder: v4 + ~22 new signals.

    The extra features target the failure modes from the v4 teardown:
      * Drift between training and holdout periods → longer rolling windows
        and persistence baselines capture the level shift.
      * Outlier-sensitivity of rolling means → rolling medians in parallel.
      * Annual cycle → hour_of_year_sin/cos.
      * Weather regime changes → lagged weather signals.
    """
    # Start from v4 (keeps all 55+ base features).
    df, v4_cols = build_features_v4_s1(
        df_meter, weather_expanded, fleet_agg, drop_warmup=False
    )

    # ── Annual cycle (2) ──
    doy = df["ts"].dt.dayofyear + df["ts"].dt.hour / 24
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # ── Robust rolling medians (4) ──
    s1 = df["demand_wh"].shift(1)
    for win in [6, 12, 48, 336]:
        df[f"rmedian_{win}"] = s1.rolling(win, min_periods=1).median()

    # ── EMAs at multiple horizons (3) ──
    for span in [6, 24, 48]:
        df[f"ema_{span}"] = s1.ewm(span=span, adjust=False, min_periods=1).mean()

    # ── Persistence baselines (3) ──
    # Yesterday same-half-hour and last-week same-half-hour are already in lags,
    # but give the model explicit ratio + delta so the tree doesn't have to learn it.
    df["persistence_24h"] = df["demand_wh"].shift(48)
    df["persistence_168h"] = df["demand_wh"].shift(336)
    df["persist_ratio_24_168"] = (
        df["persistence_24h"] / df["persistence_168h"].clip(lower=1)
    )

    # ── Longer seasonal lookbacks (3) ──
    # Same DOW × hour mean over last 60 days and last 90 days.
    df["hour_int"] = df["ts"].dt.hour.astype(int)
    df["dow_int"] = df["ts"].dt.dayofweek.astype(int)
    df["hour_dow_key"] = df["hour_int"].astype(str) + "_" + df["dow_int"].astype(str)

    df["sdh_mean_60d"] = df.groupby("hour_dow_key")["demand_wh"].transform(
        lambda x: x.shift(1).rolling(9, min_periods=2).mean()  # 9 occurrences ≈ 63 days
    )
    df["sdh_mean_90d"] = df.groupby("hour_dow_key")["demand_wh"].transform(
        lambda x: x.shift(1).rolling(13, min_periods=3).mean()  # 13 occ ≈ 91 days
    )
    df["sdh_std_60d"] = df.groupby("hour_dow_key")["demand_wh"].transform(
        lambda x: x.shift(1).rolling(9, min_periods=2).std()
    )

    # ── Weather lag features (3) ──
    for col in ["temperature", "humidity", "cloud_cover"]:
        if col in df.columns:
            df[f"{col}_lag24"] = df[col].shift(48)

    # ── Variability ratios (3) ──
    # These catch regime changes: a load that was stable is now volatile.
    df["rstd_ratio_48_336"] = df["rstd_48"] / df["rstd_336"].clip(lower=1)
    df["rmean_ratio_12_336"] = df["rmean_12"] / df["rmean_336"].clip(lower=1)
    df["cv_48"] = df["rstd_48"] / df["rmean_48"].clip(lower=1)

    # ── Similar-day expanded window (2) ──
    df["similar_day_k10"] = df.groupby("hour_dow_key")["demand_wh"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )
    df["sdh_deviation"] = df["demand_wh"].shift(1) - df["sdh_mean_60d"]

    df = df.drop(columns=["hour_int", "dow_int", "hour_dow_key"])

    if drop_warmup:
        # v4 dropped 336 rows; v5 needs 672 (14 days) because some features use
        # 9-occurrence rolling windows of hour×dow (9 occ × 7 days).
        drop_n = min(672, len(df) // 3)
        df = df.iloc[drop_n:].reset_index(drop=True)

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
# v5 HYPERPARAMETERS
# ════════════════════════════════════════════════════════════════════════════
def get_params_v5(tier: str, n_samples: int, variant: str = "base") -> dict:
    """Hyperparameter variant for v5 random search.

    `variant` ∈ {"base", "conservative", "aggressive", "deep", "wide", "regularized"}.
    Per-meter random search picks the best by internal CV.
    """
    p = {
        "objective": "regression", "metric": "mae", "verbose": -1, "n_jobs": -1,
        "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 50,
        "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 5,
        "lambda_l1": 0.1, "lambda_l2": 1.0, "max_depth": 8,
    }
    # Tier-adaptive baseline
    if tier == "HT (>5kWh)":
        p.update({"num_leaves": 63, "min_child_samples": 20,
                  "feature_fraction": 0.8, "learning_rate": 0.035})
    elif tier == "Small (<500)":
        p.update({"num_leaves": 15, "min_child_samples": 80,
                  "feature_fraction": 0.5, "max_depth": 6,
                  "lambda_l1": 0.5, "lambda_l2": 5.0})
    # Sample-count caps
    if n_samples < 3000:
        p["num_leaves"] = min(p["num_leaves"], 15)
        p["min_child_samples"] = max(p["min_child_samples"], 100)

    # Variants for random search
    if variant == "conservative":
        p.update({"learning_rate": 0.02, "num_leaves": max(15, p["num_leaves"] // 2),
                  "min_child_samples": p["min_child_samples"] * 2,
                  "lambda_l2": p["lambda_l2"] * 3})
    elif variant == "aggressive":
        p.update({"learning_rate": 0.05, "num_leaves": min(127, p["num_leaves"] * 2),
                  "min_child_samples": max(10, p["min_child_samples"] // 2),
                  "feature_fraction": min(0.9, p["feature_fraction"] + 0.1)})
    elif variant == "deep":
        p.update({"max_depth": 12, "num_leaves": min(127, p["num_leaves"] + 32),
                  "min_child_samples": max(20, p["min_child_samples"] - 10)})
    elif variant == "wide":
        p.update({"feature_fraction": min(0.95, p["feature_fraction"] + 0.2),
                  "bagging_fraction": min(0.95, p["bagging_fraction"] + 0.1)})
    elif variant == "regularized":
        p.update({"lambda_l1": p["lambda_l1"] * 5, "lambda_l2": p["lambda_l2"] * 3,
                  "feature_fraction": max(0.4, p["feature_fraction"] - 0.15)})
    return p


# ════════════════════════════════════════════════════════════════════════════
# METRICS — v5 reports both MAPE and sMAPE (honest for intermittent)
# ════════════════════════════════════════════════════════════════════════════
def calc_metrics_v5(y_true, y_pred, q10=None, q90=None, zero_thresh: float = 0.5) -> dict:
    """Extended metrics: MAPE + sMAPE + WAPE + MAE + coverage.

    sMAPE (symmetric) handles zero/near-zero actuals without exploding; we
    quote MAPE as the primary number when zero_pct < 15% and sMAPE otherwise.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_true)
    mask = y_true > zero_thresh
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mbe = float(np.mean(y_pred - y_true))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    mape = (
        float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)
        if mask.sum() > 0 else 0.0
    )
    # sMAPE: symmetric MAPE, bounded in [0, 200].
    denom = (np.abs(y_true) + np.abs(y_pred)).clip(min=0.5)
    smape = float(np.mean(2 * np.abs(y_true - y_pred) / denom) * 100)
    # WAPE: weighted absolute % error (ratio of total error to total actual).
    wape = (
        float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100)
        if np.sum(np.abs(y_true)) > 0 else 0.0
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
        "mape": round(mape, 3),
        "smape": round(smape, 3),
        "wape": round(wape, 3),
        "mae": round(mae, 2),
        "mbe": round(mbe, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "within5": round(w5, 1),
        "within10": round(w10, 1),
        "coverage_80": round(cov, 1),
        "n": int(n),
        "n_nonzero": int(mask.sum()),
        "zero_pct": round(float((~mask).mean() * 100), 2),
    }
