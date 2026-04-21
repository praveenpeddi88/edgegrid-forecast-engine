"""
v5_retrain — fleet-scale training orchestrator for the v5 demand engine.

This module is the canonical entry point for (re)training all 42 meters with
the v5 recipe. It produces:

    models/v5/{msn}.joblib    — per-meter bundle with LightGBM mean + q10/q90
                                heads, selected features, bias correction,
                                and rich metadata.
    models/v5/_manifest.json  — list of summary rows for the fleet (tier,
                                holdout MAPE/MAE/MBE, bias gate, feature
                                count, train window), suitable for dashboards
                                and the /healthz endpoint.

The v5 recipe (vs v4):
  • Train-cutoff advanced to the latest verified half-hour (configurable),
    which lets us use more than two additional months of production data
    and gives the fleet a much better shot at the <4% MAPE gate than v4
    (which cut off at 2025-11-28 and had 39/42 meters >4%).
  • Two-pass feature screening: pass-1 fits a quick LGBM to rank gain-importance,
    top-55% (min 25) features go to pass-2.
  • Quantile heads (q10, q90) trained from the same feature set.
  • Trailing-21d MBE bias estimator dampened 0.5×, capped at ±30% of mean
    demand, and VAL-GATED — only applied when validation MAPE actually
    improves by ≥0.1pp. Prevents the v2 regression where blind MBE
    correction hurt dominant-zero meters.
  • log1p target transform for meters with ≥30% zero half-hours (intermittent).
  • 85/15 chronological train/val split, LightGBM early-stopping on val MAE.

Public API
----------
retrain_meter(msn, all_df, weather_df, fleet_df, *, train_cutoff, out_dir)
    Train ONE meter, persist the bundle, return (bundle, path).

retrain_fleet(*, train_cutoff=None, out_dir=None, msns=None, skip_existing=False)
    Train EVERY meter (or the supplied subset). Writes `_manifest.json`
    alongside the bundles. Returns the list of manifest rows.

build_manifest(bundles_dir)
    Scan a directory of v5 bundles and (re)write `_manifest.json` without
    retraining. Useful after a partial retrain.

CLI
---
    # retrain one meter, default cutoff
    python -m edgegrid_forecast.training.v5_retrain --msn 53407938

    # retrain the whole fleet, cutoff at end of Feb 12 verified data
    python -m edgegrid_forecast.training.v5_retrain --all \
        --train-cutoff 2026-02-12T15:00:00

    # just rebuild the manifest (e.g. after manual bundle edits)
    python -m edgegrid_forecast.training.v5_retrain --manifest-only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ..inference._features import (
    TIME_BLOCKS,
    build_features_v4_s1,
    calc_block_metrics,
    calc_metrics,
    compute_fleet_aggregate,
    fetch_weather_expanded,
    get_params,
    load_meter_data,
)

VERSION = "v5.s1.0"
STRATEGY = "S1"
DEFAULT_TRAIN_CUTOFF = pd.Timestamp("2026-02-12T15:00:00")  # end of verified window
MODELS_ROOT = Path(__file__).resolve().parents[3] / "models" / "v5"


# ----------------------------------------------------------------------------
# Helper: feature screening pass-1
# ----------------------------------------------------------------------------
def _screen_features(
    X_all: np.ndarray,
    y_train_t: np.ndarray,
    feat_cols: list[str],
    val_cut: int,
    *,
    keep_frac: float = 0.55,
    keep_min: int = 25,
) -> list[str]:
    dtr1 = lgb.Dataset(X_all[:val_cut], label=y_train_t[:val_cut])
    dv1 = lgb.Dataset(X_all[val_cut:], label=y_train_t[val_cut:], reference=dtr1)
    p1 = {
        "objective": "regression", "metric": "mae", "num_leaves": 31,
        "learning_rate": 0.08, "feature_fraction": 0.7, "verbose": -1, "n_jobs": -1,
    }
    m1 = lgb.train(
        p1, dtr1, num_boost_round=150, valid_sets=[dv1],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
    )
    imp = m1.feature_importance(importance_type="gain")
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    nk = max(keep_min, int(len(feat_cols) * keep_frac))
    sel = [f for f, g in ranked[:nk] if g > 0]
    if len(sel) < 20:
        sel = [f for f, _ in ranked[:25]]
    return sel


# ----------------------------------------------------------------------------
# Helper: val-gated bias correction
# ----------------------------------------------------------------------------
def _compute_bias_correction(
    model: lgb.Booster,
    train_df: pd.DataFrame,
    selected: list[str],
    use_log: bool,
    X_tr: np.ndarray,
    y_train: np.ndarray,
    val_cut: int,
    mean_demand: float,
) -> tuple[float, float, str]:
    """Returns (trail_mbe_capped, trail_mbe_raw, bias_gate)."""
    trail_cutoff = train_df["ts"].max() - pd.Timedelta(days=21)
    trail_mask = train_df["ts"] > trail_cutoff
    trail_mbe = 0.0
    trail_mbe_capped = 0.0
    bias_gate = "below_threshold"
    if trail_mask.sum() < 48:
        return trail_mbe_capped, trail_mbe, bias_gate

    X_trail = np.nan_to_num(
        train_df.loc[trail_mask, selected].values.astype(np.float64), nan=0.0
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
    if len(y_val) < 48 or abs(trail_mbe_candidate) <= 0.5:
        return trail_mbe_capped, trail_mbe, bias_gate

    pred_val_t = model.predict(X_val)
    pred_val = np.expm1(pred_val_t) if use_log else pred_val_t
    pred_val = np.maximum(pred_val, 0)
    mask_val = y_val > 0.5
    if mask_val.sum() < 20:
        return trail_mbe_capped, trail_mbe, bias_gate

    mape_raw = float(
        np.mean(np.abs(y_val[mask_val] - pred_val[mask_val]) / y_val[mask_val]) * 100
    )
    pred_val_corr = np.maximum(pred_val - trail_mbe_candidate, 0)
    mape_corr = float(
        np.mean(
            np.abs(y_val[mask_val] - pred_val_corr[mask_val]) / y_val[mask_val]
        )
        * 100
    )
    if mape_corr + 0.1 < mape_raw:
        trail_mbe_capped = trail_mbe_candidate
        bias_gate = "applied"
    else:
        bias_gate = "gated_out"
    return trail_mbe_capped, trail_mbe, bias_gate


# ----------------------------------------------------------------------------
# Core training for ONE meter
# ----------------------------------------------------------------------------
def retrain_meter(
    msn: str,
    all_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    fleet_df: pd.DataFrame,
    *,
    train_cutoff: pd.Timestamp = DEFAULT_TRAIN_CUTOFF,
    out_dir: Path = MODELS_ROOT,
    val_frac: float = 0.15,
    max_rounds: int = 800,
    early_stop: int = 40,
) -> tuple[dict, Path]:
    mdata = all_df[all_df["msn"] == msn].copy()
    if mdata.empty:
        raise ValueError(f"{msn}: no rows in meter data")
    tier = mdata["tier"].iloc[0] if "tier" in mdata.columns else "Medium (0.5-1.5k)"
    phase = mdata["phase"].iloc[0] if "phase" in mdata.columns else "Unknown"

    feat_df, feat_cols = build_features_v4_s1(mdata, weather_df, fleet_df)
    train_df = feat_df[feat_df["ts"] <= train_cutoff].copy()
    if len(train_df) < 500:
        raise ValueError(f"{msn}: train too short ({len(train_df)})")

    y_train = train_df["target"].values.astype(np.float64)
    mean_demand = float(np.mean(y_train))
    zero_pct = float((y_train < 0.5).mean() * 100)
    use_log = zero_pct > 30.0
    y_train_t = np.log1p(y_train) if use_log else y_train

    X_all = np.nan_to_num(train_df[feat_cols].values.astype(np.float64), nan=0.0)
    val_cut = int(len(X_all) * (1 - val_frac))

    # Pass-1: screen features by gain importance
    selected = _screen_features(X_all, y_train_t, feat_cols, val_cut)

    # Pass-2: fit final mean model on screened features
    X_tr = np.nan_to_num(train_df[selected].values.astype(np.float64), nan=0.0)
    params = get_params(tier, len(X_tr))
    dtr = lgb.Dataset(X_tr[:val_cut], label=y_train_t[:val_cut])
    dv = lgb.Dataset(X_tr[val_cut:], label=y_train_t[val_cut:], reference=dtr)
    model = lgb.train(
        params, dtr, num_boost_round=max_rounds, valid_sets=[dv],
        callbacks=[lgb.early_stopping(early_stop), lgb.log_evaluation(0)],
    )
    best_iter = int(model.best_iteration or max_rounds)

    # Bias correction (val-gated)
    trail_mbe_capped, trail_mbe_raw, bias_gate = _compute_bias_correction(
        model, train_df, selected, use_log, X_tr, y_train, val_cut, mean_demand
    )

    # Quantile heads (trained on raw target; no log1p)
    pq10 = {**params, "objective": "quantile", "alpha": 0.1, "metric": "quantile"}
    pq90 = {**params, "objective": "quantile", "alpha": 0.9, "metric": "quantile"}
    dtr_raw = lgb.Dataset(X_tr[:val_cut], label=y_train[:val_cut])
    mq10 = lgb.train(pq10, dtr_raw, num_boost_round=best_iter or 100)
    mq90 = lgb.train(pq90, dtr_raw, num_boost_round=best_iter or 100)

    # Holdout evaluation on val slice
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
        "msn": msn, "tier": tier, "phase": phase, "version": VERSION,
        "strategy": STRATEGY, "model_mean": model, "model_q10": mq10, "model_q90": mq90,
        "selected_features": selected, "use_log1p": bool(use_log),
        "trail_mbe_capped": float(trail_mbe_capped),
        "trail_mbe_raw": float(trail_mbe_raw),
        "bias_gate": bias_gate, "best_iter": best_iter,
        "train_cutoff": train_cutoff.isoformat(),
        "train_start": str(train_df["ts"].min()),
        "train_end": str(train_df["ts"].max()),
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "train_mean_demand": mean_demand, "zero_pct": zero_pct,
        "holdout_metrics": holdout,
        "historical_block_mape": historical_block_mape,
        "n_features_initial": len(feat_cols),
        "n_features_selected": len(selected),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{msn}.joblib"
    joblib.dump(bundle, path, compress=3)
    return bundle, path


# ----------------------------------------------------------------------------
# Manifest helpers
# ----------------------------------------------------------------------------
def _manifest_row(bundle: dict, path: Path) -> dict:
    m = bundle.get("holdout_metrics", {}) or {}
    # Store path relative to repo root so the manifest is portable.
    try:
        repo_root = Path(__file__).resolve().parents[3]
        rel_path = path.resolve().relative_to(repo_root)
        path_str = rel_path.as_posix()
    except Exception:
        path_str = path.as_posix()
    return {
        "msn": bundle["msn"],
        "tier": bundle.get("tier"),
        "phase": bundle.get("phase"),
        "version": bundle.get("version", VERSION),
        "strategy": bundle.get("strategy", STRATEGY),
        "trained_at": bundle.get("trained_at"),
        "train_cutoff": bundle.get("train_cutoff"),
        "train_start": bundle.get("train_start"),
        "train_end": bundle.get("train_end"),
        "zero_pct": bundle.get("zero_pct"),
        "holdout_mape": m.get("mape"),
        "holdout_mae": m.get("mae"),
        "holdout_mbe": m.get("mbe"),
        "holdout_rmse": m.get("rmse"),
        "holdout_r2": m.get("r2"),
        "holdout_within5": m.get("within5"),
        "holdout_within10": m.get("within10"),
        "coverage_80": m.get("coverage_80"),
        "bias_gate": bundle.get("bias_gate"),
        "trail_mbe_raw": bundle.get("trail_mbe_raw"),
        "trail_mbe_capped": bundle.get("trail_mbe_capped"),
        "n_features_initial": bundle.get("n_features_initial"),
        "n_features_selected": bundle.get("n_features_selected"),
        "historical_block_mape": bundle.get("historical_block_mape", {}),
        "path": path_str,
    }


def build_manifest(bundles_dir: Path = MODELS_ROOT) -> list[dict]:
    """Scan bundles_dir for *.joblib v5 bundles and (re)write _manifest.json."""
    rows: list[dict] = []
    for p in sorted(bundles_dir.glob("*.joblib")):
        try:
            b = joblib.load(p)
        except Exception as exc:
            print(f"  WARN: could not load {p.name}: {exc}", flush=True)
            continue
        if not isinstance(b, dict) or "msn" not in b:
            continue
        rows.append(_manifest_row(b, p))
    manifest_path = bundles_dir / "_manifest.json"
    manifest_path.write_text(json.dumps(rows, indent=2, default=str))
    return rows


# ----------------------------------------------------------------------------
# Fleet retraining
# ----------------------------------------------------------------------------
def retrain_fleet(
    *,
    train_cutoff: Optional[pd.Timestamp] = None,
    out_dir: Optional[Path] = None,
    msns: Optional[Iterable[str]] = None,
    skip_existing: bool = False,
) -> list[dict]:
    """Train all (or the supplied subset of) meters with the v5 recipe.

    Parameters
    ----------
    train_cutoff : Timestamp, optional
        Training hard-cutoff. Rows with ts > train_cutoff are excluded from
        training. Defaults to end of verified Feb 12 window.
    out_dir : Path, optional
        Where to persist bundles. Defaults to <repo>/models/v5.
    msns : iterable of str, optional
        Subset of meter serial numbers to retrain. Defaults to all.
    skip_existing : bool
        If True, meters whose bundle already exists on disk are skipped.
        Useful for resuming interrupted runs.

    Returns
    -------
    List of manifest-row dicts (also written to _manifest.json).
    """
    cutoff = train_cutoff or DEFAULT_TRAIN_CUTOFF
    target_dir = out_dir or MODELS_ROOT
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v5-retrain] loading fleet data...", flush=True)
    t0 = time.time()
    all_df, _ = load_meter_data()
    weather = fetch_weather_expanded()
    fleet = compute_fleet_aggregate(all_df)
    print(f"  fleet data ready in {time.time() - t0:.1f}s", flush=True)

    meter_list = sorted(all_df["msn"].unique().tolist())
    if msns is not None:
        wanted = {str(m) for m in msns}
        meter_list = [m for m in meter_list if m in wanted]
    print(f"[v5-retrain] retraining {len(meter_list)} meters (cutoff={cutoff})", flush=True)

    manifest_rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    for i, msn in enumerate(meter_list, 1):
        dest = target_dir / f"{msn}.joblib"
        if skip_existing and dest.exists():
            try:
                bundle = joblib.load(dest)
                manifest_rows.append(_manifest_row(bundle, dest))
                print(f"  [{i:2d}/{len(meter_list)}] {msn} SKIP (exists)", flush=True)
                continue
            except Exception:
                pass  # fall through to retrain
        t0 = time.time()
        try:
            bundle, path = retrain_meter(
                msn, all_df, weather, fleet,
                train_cutoff=cutoff, out_dir=target_dir,
            )
            mape = bundle["holdout_metrics"].get("mape", float("nan"))
            print(
                f"  [{i:2d}/{len(meter_list)}] {msn} OK "
                f"mape={mape:6.2f}%  n_feat={bundle['n_features_selected']}  "
                f"{time.time() - t0:5.1f}s",
                flush=True,
            )
            manifest_rows.append(_manifest_row(bundle, path))
        except Exception as exc:
            failures.append((msn, str(exc)))
            print(f"  [{i:2d}/{len(meter_list)}] {msn} FAIL — {exc}", flush=True)
            traceback.print_exc()

    # Always write manifest at end (even if partial)
    manifest_path = target_dir / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest_rows, indent=2, default=str))
    print(f"[v5-retrain] wrote {len(manifest_rows)} rows → {manifest_path}", flush=True)
    if failures:
        print(f"[v5-retrain] {len(failures)} failures:", flush=True)
        for msn, err in failures:
            print(f"    {msn} — {err[:200]}", flush=True)

    # Print fleet-level summary
    mapes = [r["holdout_mape"] for r in manifest_rows if r["holdout_mape"] is not None]
    if mapes:
        arr = np.array(mapes, dtype=float)
        print(
            f"[v5-retrain] fleet holdout MAPE: "
            f"mean={arr.mean():.2f}%  median={np.median(arr):.2f}%  "
            f"p90={np.percentile(arr, 90):.2f}%  "
            f"<4%: {int((arr < 4).sum())}/{len(arr)}  "
            f"<10%: {int((arr < 10).sum())}/{len(arr)}",
            flush=True,
        )
    return manifest_rows


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="v5 demand-forecast training orchestrator")
    ap.add_argument("--msn", help="train a single meter (serial no)")
    ap.add_argument("--all", action="store_true", help="retrain every meter")
    ap.add_argument(
        "--msns", help="comma-separated meter subset (overrides --all)"
    )
    ap.add_argument(
        "--train-cutoff", default=DEFAULT_TRAIN_CUTOFF.isoformat(),
        help=f"ISO timestamp; default {DEFAULT_TRAIN_CUTOFF}",
    )
    ap.add_argument(
        "--out-dir", default=str(MODELS_ROOT),
        help=f"bundle output directory (default {MODELS_ROOT})",
    )
    ap.add_argument(
        "--skip-existing", action="store_true",
        help="skip meters whose bundle already exists",
    )
    ap.add_argument(
        "--manifest-only", action="store_true",
        help="skip retraining; only rebuild _manifest.json from existing bundles",
    )
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    out_dir = Path(args.out_dir).expanduser().resolve()

    if args.manifest_only:
        rows = build_manifest(out_dir)
        print(f"wrote manifest with {len(rows)} rows → {out_dir / '_manifest.json'}")
        return 0

    cutoff = pd.Timestamp(args.train_cutoff)

    if args.msn and not args.all and not args.msns:
        print(f"[v5-retrain] single meter {args.msn} cutoff={cutoff}", flush=True)
        all_df, _ = load_meter_data()
        weather = fetch_weather_expanded()
        fleet = compute_fleet_aggregate(all_df)
        bundle, path = retrain_meter(
            args.msn, all_df, weather, fleet,
            train_cutoff=cutoff, out_dir=out_dir,
        )
        print(
            f"  holdout MAPE = {bundle['holdout_metrics'].get('mape'):.2f}%  → {path}",
            flush=True,
        )
        build_manifest(out_dir)
        return 0

    msns = None
    if args.msns:
        msns = [m.strip() for m in args.msns.split(",") if m.strip()]
    elif not args.all:
        print("nothing to do — pass --msn, --msns, --all, or --manifest-only", flush=True)
        return 1

    retrain_fleet(
        train_cutoff=cutoff, out_dir=out_dir, msns=msns,
        skip_existing=args.skip_existing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
