"""
v4_predict — Production inference wrapper for EdgeGrid Strategy 1 v4 models.

Closes Session 11's Gap A: converts the monolithic benchmark script into a
reusable API so the dispatch optimizer (and any downstream service) can call
`predict(msn, as_of_datetime)` and get a time-indexed forecast with
confidence bands and per-block historical accuracy — without re-training or
importing the benchmark harness.

Public API
----------
train_and_persist(msn, ...) -> Path
    Full two-pass training pipeline (Pass 1 screen → Pass 2 fit with
    tier-adaptive params → trailing-MBE bias correction, val-gated → q10/q90
    quantile fits). Saves a single joblib bundle per meter at
    models/v4/{msn}.joblib and returns the path.

load_model(msn, ...) -> dict
    Rehydrate the persisted bundle into a dict of loaded LightGBM Boosters
    plus all inference metadata (selected features, bias, training window,
    historical per-block MAPE, etc.).

predict(msn, as_of_datetime=None, horizon=48, ...) -> pd.DataFrame
    Load-then-predict for the last `horizon` timesteps ending at
    `as_of_datetime` (defaults to the end of that meter's raw data). Returns a
    Rich DataFrame with forecast_wh + confidence_low/high + block_label +
    historical_block_mape, ready for BESS dispatch consumption.

predict_with_context(msn, context_df, weather_df=None, ...) -> pd.DataFrame
    Lower-level: caller supplies its own context timeseries (e.g. streaming
    MDMS output) and optionally a weather frame. Useful for rolling-origin
    evaluation, A/B inference, or mocking in tests.

Format
------
models/v4/{msn}.joblib is a dict bundle. See `_MODEL_BUNDLE_SCHEMA` below for
the canonical schema. `MODEL_VERSION` is bumped whenever the recipe in
_features.py or the training pipeline here diverges from v4 parity.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ._features import (
    TIME_BLOCKS,
    block_label_for,
    build_features_v4_s1,
    calc_block_metrics,
    calc_metrics,
    compute_fleet_aggregate,
    fetch_weather_expanded,
    get_params,
    load_meter_data,
    split_chronological,
)


MODEL_VERSION = "v4.s1.0"  # bump on any training-recipe change

_MODEL_BUNDLE_SCHEMA = {
    "msn":                    "str",
    "tier":                   "str",
    "phase":                  "str",
    "version":                "str (e.g. v4.s1.0)",
    "strategy":               "S1",
    "model_mean":             "lgb.Booster — main regressor",
    "model_q10":              "lgb.Booster — quantile low",
    "model_q90":              "lgb.Booster — quantile high",
    "selected_features":      "list[str] — Pass 2 feature set (order matters)",
    "use_log1p":              "bool — target transform toggle",
    "trail_mbe_capped":       "float — Phase B correction subtracted from predictions",
    "bias_gate":              "str — applied | gated_out | below_threshold | val_too_small",
    "best_iter":              "int — Pass 2 early stopping round",
    "train_cutoff":           "ISO8601 — chronological 75% boundary",
    "trained_at":             "ISO8601",
    "train_mean_demand":      "float — mean demand over holdout (Wh/30m)",
    "holdout_metrics":        "dict — full metrics on holdout (MAPE, MAE, etc.)",
    "historical_block_mape":  "dict[block -> mape] — from the holdout evaluation",
}


def _default_models_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "models" / "v4"


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "raw"


# ════════════════════════════════════════════════════════════════════════════
# TRAIN + PERSIST
# ════════════════════════════════════════════════════════════════════════════
def train_and_persist(
    msn: str,
    *,
    data_dir: Optional[Path | str] = None,
    models_dir: Optional[Path | str] = None,
    weather_df: Optional[pd.DataFrame] = None,
    fleet_df: Optional[pd.DataFrame] = None,
    all_df: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> Path:
    """Train the v4 S1 pipeline on a single meter and persist the bundle.

    Parameters
    ----------
    msn : Meter serial number.
    data_dir : Override default data/raw directory.
    models_dir : Override default models/v4 output directory.
    weather_df, fleet_df, all_df : Inject pre-loaded frames (batch training
        reuses these to avoid redundant disk + network).
    verbose : Print per-stage progress.

    Returns
    -------
    Path to the persisted joblib bundle.
    """
    msn = str(msn)
    data_dir = Path(data_dir) if data_dir else _default_data_dir()
    models_dir = Path(models_dir) if models_dir else _default_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Load (or reuse) raw data, weather, fleet ──
    if all_df is None:
        all_df, _profile = load_meter_data(data_dir)
    if weather_df is None:
        weather_df = fetch_weather_expanded()
    if fleet_df is None:
        fleet_df = compute_fleet_aggregate(all_df)

    mdata = all_df[all_df["msn"] == msn].copy()
    if len(mdata) < 500:
        raise ValueError(f"{msn}: too few rows ({len(mdata)}) — need ≥500")

    tier  = mdata["tier"].iloc[0]  if "tier"  in mdata.columns else "Unknown"
    phase = mdata["phase"].iloc[0] if "phase" in mdata.columns else "Unknown"

    # ── Features ──
    feat_df, feat_cols = build_features_v4_s1(mdata, weather_df, fleet_df)
    if len(feat_df) < 500:
        raise ValueError(f"{msn}: feat_df too small ({len(feat_df)})")

    # ── Chronological split (Phase A) ──
    train_df, test_df, n_tr_days, n_te_days, cutoff = split_chronological(feat_df)
    if len(test_df) < 48 or len(train_df) < 500:
        raise ValueError(
            f"{msn}: insufficient split (train={len(train_df)}, test={len(test_df)})"
        )

    y_train = train_df["target"].values.astype(np.float64)
    y_test  = test_df["target"].values.astype(np.float64)
    mean_demand = float(np.mean(y_test))
    zero_pct = float((y_test < 0.5).mean() * 100)
    use_log = zero_pct > 30.0
    y_train_t = np.log1p(y_train) if use_log else y_train

    if verbose:
        print(f"  [{msn}] tier={tier} phase={phase} "
              f"rows={len(feat_df)} train_days={n_tr_days} test_days={n_te_days} "
              f"zero%={zero_pct:.1f} log1p={use_log}")

    # ── Pass 1: feature screening (chronological val_cut = last 15% of train) ──
    X_all = np.nan_to_num(train_df[feat_cols].values.astype(np.float64), nan=0.0)
    val_cut = int(len(X_all) * 0.85)
    dtr1 = lgb.Dataset(X_all[:val_cut], label=y_train_t[:val_cut])
    dv1  = lgb.Dataset(X_all[val_cut:], label=y_train_t[val_cut:], reference=dtr1)
    p1 = {"objective": "regression", "metric": "mae", "num_leaves": 31,
          "learning_rate": 0.08, "feature_fraction": 0.7, "verbose": -1, "n_jobs": -1}
    m1 = lgb.train(p1, dtr1, num_boost_round=150, valid_sets=[dv1],
                   callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
    imp = m1.feature_importance(importance_type="gain")
    fg = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    nk = max(25, int(len(feat_cols) * 0.55))
    sel = [f for f, g in fg[:nk] if g > 0]
    if len(sel) < 20:
        sel = [f for f, g in fg[:25]]

    # ── Pass 2: tier-adaptive fit ──
    X_tr = np.nan_to_num(train_df[sel].values.astype(np.float64), nan=0.0)
    X_te = np.nan_to_num(test_df[sel].values.astype(np.float64), nan=0.0)
    params = get_params(tier, len(X_tr))
    dtr = lgb.Dataset(X_tr[:val_cut], label=y_train_t[:val_cut])
    dv  = lgb.Dataset(X_tr[val_cut:], label=y_train_t[val_cut:], reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=800, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
    best_iter = int(model.best_iteration or 0)

    pred_raw_t = model.predict(X_te)
    pred_raw   = np.expm1(pred_raw_t) if use_log else pred_raw_t
    pred_raw   = np.maximum(pred_raw, 0)

    # ── Phase B: trailing-21d MBE bias correction (val-gated) ──
    trail_cutoff = train_df["ts"].max() - pd.Timedelta(days=21)
    trail_mask = train_df["ts"] > trail_cutoff
    trail_mbe = 0.0
    if trail_mask.sum() >= 48:
        X_trail = np.nan_to_num(
            train_df.loc[trail_mask, sel].values.astype(np.float64), nan=0.0
        )
        y_trail = train_df.loc[trail_mask, "target"].values.astype(np.float64)
        pred_trail_t = model.predict(X_trail)
        pred_trail   = np.expm1(pred_trail_t) if use_log else pred_trail_t
        pred_trail   = np.maximum(pred_trail, 0)
        trail_mbe    = float(np.mean(pred_trail - y_trail))

    DAMPEN = 0.5
    bias_cap = 0.30 * mean_demand
    trail_mbe_candidate = float(np.clip(trail_mbe * DAMPEN, -bias_cap, bias_cap))

    X_val = X_tr[val_cut:]
    y_val = y_train[val_cut:]
    if len(y_val) >= 48 and abs(trail_mbe_candidate) > 0.5:
        pred_val_t = model.predict(X_val)
        pred_val   = np.expm1(pred_val_t) if use_log else pred_val_t
        pred_val   = np.maximum(pred_val, 0)
        mask_val = y_val > 0.5
        if mask_val.sum() >= 20:
            mape_val_raw = float(
                np.mean(np.abs(y_val[mask_val] - pred_val[mask_val]) / y_val[mask_val]) * 100
            )
            pred_val_corr = np.maximum(pred_val - trail_mbe_candidate, 0)
            mape_val_corr = float(
                np.mean(np.abs(y_val[mask_val] - pred_val_corr[mask_val]) / y_val[mask_val]) * 100
            )
            if mape_val_corr + 0.1 < mape_val_raw:
                trail_mbe_capped = trail_mbe_candidate
                bias_gate = "applied"
            else:
                trail_mbe_capped = 0.0
                bias_gate = "gated_out"
        else:
            trail_mbe_capped = 0.0
            bias_gate = "val_too_small"
    else:
        trail_mbe_capped = 0.0
        bias_gate = "below_threshold"

    pred = np.maximum(pred_raw - trail_mbe_capped, 0)

    # ── Quantiles (fit on raw target at best_iter) ──
    pq10 = {**params, "objective": "quantile", "alpha": 0.1, "metric": "quantile"}
    pq90 = {**params, "objective": "quantile", "alpha": 0.9, "metric": "quantile"}
    dtr_raw = lgb.Dataset(X_tr[:val_cut], label=y_train[:val_cut])
    mq10 = lgb.train(pq10, dtr_raw, num_boost_round=best_iter or 100)
    mq90 = lgb.train(pq90, dtr_raw, num_boost_round=best_iter or 100)
    lgb_q10 = np.maximum(mq10.predict(X_te) - trail_mbe_capped, 0)
    lgb_q90 = np.maximum(mq90.predict(X_te) - trail_mbe_capped, 0)

    # ── Evaluate on holdout (to persist for predict() to surface) ──
    holdout = calc_metrics(y_test, pred, lgb_q10, lgb_q90)
    block_m = calc_block_metrics(test_df["ts"].values, y_test, pred)
    historical_block_mape = {
        b: block_m.get(f"{b}_mape", float("nan")) for b, _ in TIME_BLOCKS
    }

    bundle = {
        "msn": msn,
        "tier": tier,
        "phase": phase,
        "version": MODEL_VERSION,
        "strategy": "S1",
        "model_mean": model,
        "model_q10": mq10,
        "model_q90": mq90,
        "selected_features": sel,
        "use_log1p": bool(use_log),
        "trail_mbe_capped": float(trail_mbe_capped),
        "trail_mbe_raw": float(trail_mbe),
        "bias_gate": bias_gate,
        "best_iter": best_iter,
        "train_cutoff": cutoff.isoformat(),
        "train_start": str(train_df["ts"].min()),
        "train_end":   str(train_df["ts"].max()),
        "trained_at":  datetime.utcnow().isoformat() + "Z",
        "train_mean_demand": mean_demand,
        "zero_pct": zero_pct,
        "holdout_metrics": holdout,
        "historical_block_mape": historical_block_mape,
        "n_features_initial": len(feat_cols),
        "n_features_selected": len(sel),
    }

    out_path = models_dir / f"{msn}.joblib"
    joblib.dump(bundle, out_path, compress=3)
    if verbose:
        print(f"  [{msn}] MAPE={holdout['mape']:.2f}% MAE={holdout['mae']:.0f}Wh "
              f"gate={bias_gate} → {out_path.name}")
    return out_path


# ════════════════════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════════════════════
def load_model(msn: str, *, models_dir: Optional[Path | str] = None) -> dict:
    """Load the persisted v4 bundle for a meter. Raises FileNotFoundError if absent."""
    msn = str(msn)
    models_dir = Path(models_dir) if models_dir else _default_models_dir()
    path = models_dir / f"{msn}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No v4 model at {path}. Run train_and_persist({msn!r}) first."
        )
    return joblib.load(path)


# ════════════════════════════════════════════════════════════════════════════
# PREDICT
# ════════════════════════════════════════════════════════════════════════════
def _materialise_features(
    bundle: dict,
    context_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    fleet_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the v4 feature frame from a full context window, retaining all rows
    (no warm-up drop) so the caller can index into any subset they need."""
    feat_df, _cols = build_features_v4_s1(
        context_df, weather_df, fleet_df, drop_warmup=False
    )
    feat_df = feat_df.sort_values("ts").reset_index(drop=True)
    # Ensure all persisted features are present; missing → 0 (NaN-safe).
    for f in bundle["selected_features"]:
        if f not in feat_df.columns:
            feat_df[f] = 0.0
    return feat_df


def predict_with_context(
    msn: str,
    context_df: pd.DataFrame,
    *,
    weather_df: Optional[pd.DataFrame] = None,
    fleet_df: Optional[pd.DataFrame] = None,
    models_dir: Optional[Path | str] = None,
    horizon_ts: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Run inference given caller-supplied context (history) and optional
    weather/fleet frames. Returns one row per timestamp in context_df beyond
    warm-up, or filtered to `horizon_ts` if provided.

    Rich DataFrame columns (ts-indexed):
      forecast_wh, confidence_low, confidence_high, block_label,
      historical_block_mape
    """
    bundle = load_model(msn, models_dir=models_dir)

    if weather_df is None:
        weather_df = fetch_weather_expanded()
    if fleet_df is None:
        # When only one meter's context is supplied, fleet features fall back
        # to that meter's own values (via the ffill/fillna(0) path in the
        # feature builder). For production multi-meter fleet features, inject
        # a precomputed fleet_df.
        fleet_df = pd.DataFrame(columns=["ts", "fleet_mean", "fleet_std"])

    feat_df = _materialise_features(bundle, context_df, weather_df, fleet_df)
    if horizon_ts is not None:
        hset = pd.DatetimeIndex(pd.to_datetime(horizon_ts))
        feat_df = feat_df[feat_df["ts"].isin(hset)].copy()

    if feat_df.empty:
        return pd.DataFrame(columns=[
            "forecast_wh", "confidence_low", "confidence_high",
            "block_label", "historical_block_mape",
        ])

    X = np.nan_to_num(
        feat_df[bundle["selected_features"]].values.astype(np.float64), nan=0.0
    )
    use_log = bundle["use_log1p"]
    mbe = bundle["trail_mbe_capped"]

    pred_t = bundle["model_mean"].predict(X)
    pred   = np.expm1(pred_t) if use_log else pred_t
    pred   = np.maximum(pred - mbe, 0)

    q10 = np.maximum(bundle["model_q10"].predict(X) - mbe, 0)
    q90 = np.maximum(bundle["model_q90"].predict(X) - mbe, 0)

    # Monotonicity clamp: LightGBM q10/q90 are independently-trained models and
    # can cross the mean prediction on individual rows. For downstream dispatch
    # safety we enforce  confidence_low ≤ forecast_wh ≤ confidence_high. This
    # matches the behaviour users expect from "80% confidence interval".
    q10 = np.minimum(q10, pred)
    q90 = np.maximum(q90, pred)

    block_labels = [block_label_for(t) for t in feat_df["ts"]]
    hist = bundle["historical_block_mape"]
    hist_mape = [hist.get(b, float("nan")) for b in block_labels]

    out = pd.DataFrame({
        "forecast_wh": np.round(pred, 2),
        "confidence_low": np.round(q10, 2),
        "confidence_high": np.round(q90, 2),
        "block_label": block_labels,
        "historical_block_mape": hist_mape,
    }, index=pd.DatetimeIndex(feat_df["ts"], name="ts"))
    return out


def predict(
    msn: str,
    *,
    as_of_datetime: Optional[str | pd.Timestamp] = None,
    horizon: int = 48,
    models_dir: Optional[Path | str] = None,
    data_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Convenience: load raw data for `msn`, build features, and return the
    forecast for the last `horizon` 30-minute steps ending at `as_of_datetime`
    (defaults to that meter's latest raw timestamp).

    For streaming / production use, pass your own frame to predict_with_context
    rather than hitting disk every call.
    """
    data_dir = Path(data_dir) if data_dir else _default_data_dir()

    all_df, _profile = load_meter_data(data_dir)
    mdata = all_df[all_df["msn"] == str(msn)].copy()
    if mdata.empty:
        raise ValueError(f"No rows for msn={msn} in {data_dir}")
    mdata = mdata.sort_values("ts").reset_index(drop=True)

    if as_of_datetime is None:
        as_of = mdata["ts"].max()
    else:
        as_of = pd.Timestamp(as_of_datetime)

    # Need enough history (336 for warm-up + lag_336) before the horizon window.
    start_cut = as_of - pd.Timedelta(minutes=30 * (horizon + 400))
    context = mdata[mdata["ts"] <= as_of].copy()
    context = context[context["ts"] >= start_cut].copy()
    if len(context) < 400:
        # fall back: take all available
        context = mdata[mdata["ts"] <= as_of].copy()

    weather_df = fetch_weather_expanded()
    fleet_df = compute_fleet_aggregate(all_df)

    horizon_ts = context["ts"].tail(horizon)
    return predict_with_context(
        msn,
        context,
        weather_df=weather_df,
        fleet_df=fleet_df,
        models_dir=models_dir,
        horizon_ts=horizon_ts,
    )


__all__ = [
    "train_and_persist",
    "load_model",
    "predict",
    "predict_with_context",
    "MODEL_VERSION",
]
