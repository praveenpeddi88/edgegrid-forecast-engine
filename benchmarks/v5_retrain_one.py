"""
Retrain ONE meter with train_cutoff advanced to 2026-02-04, then evaluate on
the Feb 5-12 verified window. This is the v5-plan new-#1 smoke test:
prove that extending training data by ~51 days does (or does not) close the
bundle-vs-backtest gap.

Usage:
    python3 benchmarks/v5_retrain_one.py <msn>
    (defaults to 53407938 — cluster A, bundle 3.19%, v3 backtest 22.03%)
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from edgegrid_forecast.inference._features import (  # noqa: E402
    TIME_BLOCKS,
    build_features_v4_s1,
    calc_block_metrics,
    calc_metrics,
    compute_fleet_aggregate,
    fetch_weather_expanded,
    get_params,
    load_meter_data,
)
from edgegrid_forecast.inference.v4_predict import predict_with_context  # noqa: E402


TRAIN_CUTOFF = pd.Timestamp("2026-02-04T23:30:00")
HORIZON_START = pd.Timestamp("2026-02-05T00:00:00")
HORIZON_END = pd.Timestamp("2026-02-12T15:00:00")


def retrain_meter(
    msn: str,
    all_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    fleet_df: pd.DataFrame,
    train_cutoff: pd.Timestamp = TRAIN_CUTOFF,
) -> tuple[dict, Path]:
    mdata = all_df[all_df["msn"] == msn].copy()
    tier = mdata["tier"].iloc[0] if "tier" in mdata.columns else "Medium (0.5-1.5k)"
    phase = mdata["phase"].iloc[0] if "phase" in mdata.columns else "Unknown"

    feat_df, feat_cols = build_features_v4_s1(mdata, weather_df, fleet_df)

    train_df = feat_df[feat_df["ts"] <= train_cutoff].copy()
    test_df = feat_df[feat_df["ts"] > train_cutoff].copy()
    if len(train_df) < 500:
        raise ValueError(f"{msn}: train too short ({len(train_df)})")

    y_train = train_df["target"].values.astype(np.float64)
    y_test = test_df["target"].values.astype(np.float64)
    mean_demand = float(np.mean(y_test)) if len(y_test) else float(np.mean(y_train))
    zero_pct = float((y_train < 0.5).mean() * 100)
    use_log = zero_pct > 30.0
    y_train_t = np.log1p(y_train) if use_log else y_train

    X_all = np.nan_to_num(train_df[feat_cols].values.astype(np.float64), nan=0.0)
    val_cut = int(len(X_all) * 0.85)
    dtr1 = lgb.Dataset(X_all[:val_cut], label=y_train_t[:val_cut])
    dv1 = lgb.Dataset(X_all[val_cut:], label=y_train_t[val_cut:], reference=dtr1)
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

    X_tr = np.nan_to_num(train_df[sel].values.astype(np.float64), nan=0.0)
    params = get_params(tier, len(X_tr))
    dtr = lgb.Dataset(X_tr[:val_cut], label=y_train_t[:val_cut])
    dv = lgb.Dataset(X_tr[val_cut:], label=y_train_t[val_cut:], reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=800, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
    best_iter = int(model.best_iteration or 0)

    # Phase B MBE bias correction
    trail_cutoff_bias = train_df["ts"].max() - pd.Timedelta(days=21)
    trail_mask = train_df["ts"] > trail_cutoff_bias
    trail_mbe = 0.0
    trail_mbe_capped = 0.0
    bias_gate = "below_threshold"
    if trail_mask.sum() >= 48:
        X_trail = np.nan_to_num(
            train_df.loc[trail_mask, sel].values.astype(np.float64), nan=0.0
        )
        y_trail = train_df.loc[trail_mask, "target"].values.astype(np.float64)
        pred_trail_t = model.predict(X_trail)
        pred_trail = np.expm1(pred_trail_t) if use_log else pred_trail_t
        pred_trail = np.maximum(pred_trail, 0)
        trail_mbe = float(np.mean(pred_trail - y_trail))
        DAMPEN = 0.5
        bias_cap = 0.30 * mean_demand
        trail_mbe_candidate = float(np.clip(trail_mbe * DAMPEN, -bias_cap, bias_cap))
        X_val = X_tr[val_cut:]
        y_val = y_train[val_cut:]
        if len(y_val) >= 48 and abs(trail_mbe_candidate) > 0.5:
            pred_val_t = model.predict(X_val)
            pred_val = np.expm1(pred_val_t) if use_log else pred_val_t
            pred_val = np.maximum(pred_val, 0)
            mask_val = y_val > 0.5
            if mask_val.sum() >= 20:
                mape_val_raw = float(np.mean(np.abs(y_val[mask_val] - pred_val[mask_val]) / y_val[mask_val]) * 100)
                pred_val_corr = np.maximum(pred_val - trail_mbe_candidate, 0)
                mape_val_corr = float(np.mean(np.abs(y_val[mask_val] - pred_val_corr[mask_val]) / y_val[mask_val]) * 100)
                if mape_val_corr + 0.1 < mape_val_raw:
                    trail_mbe_capped = trail_mbe_candidate
                    bias_gate = "applied"
                else:
                    bias_gate = "gated_out"

    # Quantiles
    pq10 = {**params, "objective": "quantile", "alpha": 0.1, "metric": "quantile"}
    pq90 = {**params, "objective": "quantile", "alpha": 0.9, "metric": "quantile"}
    dtr_raw = lgb.Dataset(X_tr[:val_cut], label=y_train[:val_cut])
    mq10 = lgb.train(pq10, dtr_raw, num_boost_round=best_iter or 100)
    mq90 = lgb.train(pq90, dtr_raw, num_boost_round=best_iter or 100)

    # Internal holdout MAPE (val_cut→end of train, which approximates recent behaviour)
    X_val = X_tr[val_cut:]
    y_val = y_train[val_cut:]
    pred_val_t = model.predict(X_val)
    pred_val = np.expm1(pred_val_t) if use_log else pred_val_t
    pred_val = np.maximum(pred_val - trail_mbe_capped, 0)
    q10_val = np.maximum(mq10.predict(X_val) - trail_mbe_capped, 0)
    q90_val = np.maximum(mq90.predict(X_val) - trail_mbe_capped, 0)
    holdout = calc_metrics(y_val, pred_val, q10_val, q90_val)
    block_m = calc_block_metrics(
        train_df["ts"].values[val_cut:], y_val, pred_val
    )
    historical_block_mape = {
        b: block_m.get(f"{b}_mape", float("nan")) for b, _ in TIME_BLOCKS
    }

    bundle = {
        "msn": msn, "tier": tier, "phase": phase, "version": "v5.s1.0",
        "strategy": "S1", "model_mean": model, "model_q10": mq10, "model_q90": mq90,
        "selected_features": sel, "use_log1p": bool(use_log),
        "trail_mbe_capped": float(trail_mbe_capped), "trail_mbe_raw": float(trail_mbe),
        "bias_gate": bias_gate, "best_iter": best_iter,
        "train_cutoff": train_cutoff.isoformat(),
        "train_start": str(train_df["ts"].min()),
        "train_end": str(train_df["ts"].max()),
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "train_mean_demand": mean_demand, "zero_pct": zero_pct,
        "holdout_metrics": holdout,
        "historical_block_mape": historical_block_mape,
        "n_features_initial": len(feat_cols),
        "n_features_selected": len(sel),
    }
    out_dir = REPO / "models" / "v5"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{msn}.joblib"
    joblib.dump(bundle, path, compress=3)
    return bundle, path


def evaluate_on_feb_window(msn: str, all_df, weather_df, fleet_df, models_dir: Path) -> dict:
    """Forecast the Feb 5-12 verified window with the retrained v5 bundle,
    using DOW priming (same as v3 fixture) for comparability."""
    md = all_df[all_df["msn"] == msn].sort_values("ts").reset_index(drop=True)
    ctx = md[md["ts"] <= TRAIN_CUTOFF].copy()
    hor = md[(md["ts"] >= HORIZON_START) & (md["ts"] <= HORIZON_END)].copy()
    if len(hor) < 48:
        return {"error": f"horizon too short ({len(hor)})"}

    # DOW priming: for each horizon ts, pull demand_wh from 7 days earlier
    ts_index = ctx.set_index("ts")["demand_wh"]
    primed = []
    for t in hor["ts"]:
        back = t - pd.Timedelta(days=7)
        v = float(ts_index.get(back, ctx["demand_wh"].iloc[-1]))
        primed.append(v)
    last_volt = float(ctx["voltage"].iloc[-1])
    ph = pd.DataFrame({"ts": hor["ts"].values, "demand_wh": primed,
                       "voltage": [last_volt] * len(hor)})
    ctx_with_horizon = pd.concat([ctx, ph], ignore_index=True)

    horizon_ts = pd.DatetimeIndex(hor["ts"].values)
    res = predict_with_context(
        msn, ctx_with_horizon, weather_df=weather_df, fleet_df=fleet_df,
        models_dir=models_dir, horizon_ts=horizon_ts,
    )
    y_true = hor["demand_wh"].values.astype(float)
    y_pred = res["forecast_wh"].values.astype(float)
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    mask = y_true > 0.5
    mape = float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100) if mask.sum() else float("nan")
    return {
        "n_eval": int(n),
        "v5_backtest_mape": round(mape, 2),
        "first_12_actual": y_true[:12].tolist(),
        "first_12_pred":   [round(p, 1) for p in y_pred[:12].tolist()],
    }


def main(msn: str = "53407938"):
    print(f"[v5-retrain] meter {msn}", flush=True)
    print("  loading...", flush=True)
    t0 = time.time()
    all_df, _ = load_meter_data()
    weather = fetch_weather_expanded()
    fleet = compute_fleet_aggregate(all_df)
    print(f"  data ready in {time.time()-t0:.1f}s", flush=True)

    print(f"  retraining with train_cutoff={TRAIN_CUTOFF}...", flush=True)
    t0 = time.time()
    bundle, path = retrain_meter(msn, all_df, weather, fleet)
    dt_train = time.time() - t0
    print(f"  retrained in {dt_train:.1f}s → {path.name}", flush=True)
    print(f"  bundle holdout MAPE = {bundle['holdout_metrics']['mape']:.2f}%", flush=True)
    print(f"  bundle train: {bundle['train_start']} → {bundle['train_end']}", flush=True)

    print(f"  evaluating on Feb 5-12...", flush=True)
    t0 = time.time()
    ev = evaluate_on_feb_window(msn, all_df, weather, fleet, path.parent)
    dt_eval = time.time() - t0
    print(f"  evaluated in {dt_eval:.1f}s", flush=True)
    print(f"  v5 backtest MAPE = {ev.get('v5_backtest_mape')}%  n_eval={ev.get('n_eval')}")
    print(f"  first 6 actual vs pred:")
    for a, p in zip(ev.get("first_12_actual", [])[:6], ev.get("first_12_pred", [])[:6]):
        print(f"    actual={a:>8.1f}  pred={p:>8.1f}")

    # Save summary
    summary = {
        "msn": msn,
        "train_cutoff": TRAIN_CUTOFF.isoformat(),
        "train_time_s": round(dt_train, 1),
        "bundle_holdout_mape_v5": bundle["holdout_metrics"]["mape"],
        **ev,
    }
    out = REPO / "prototypes" / "forecast_engine_v3" / f"v5_retrain_{msn}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved → {out}")


if __name__ == "__main__":
    msn = sys.argv[1] if len(sys.argv) > 1 else "53407938"
    main(msn)
