"""
v5 Forecast Engine — FastAPI router.

Wraps the per-meter v5 LightGBM bundles and forward-forecast strategies behind
a small, deployable HTTP surface.

Endpoints
---------
GET  /v5/healthz              — service + model registry health
GET  /v5/manifest             — the fleet manifest (42 meters, Feb-12 cutoff)
GET  /v5/meters               — list of available MSNs with their tier + holdout MAPE
POST /v5/predict              — per-meter half-hourly forecast (batch or recursive)
POST /v5/predict/fleet        — fleet-wide short-horizon forecast
POST /v5/retrain              — retrain one meter's bundle (write to models/v5)
GET  /v5/forecast/apr-may     — served artifact: 4-strategy Apr21-May21 forecast

Design notes
------------
- All predict paths load bundles via `edgegrid_forecast.inference.v4_predict.load_model`
  with `models_dir=models/v5` so they use the Feb-12 bundles.
- Bundle cache is process-local (simple dict). In production, back with Redis or
  a proper model-serving framework.
- Weather + fleet aggregates are cached at module import time. They can be
  refreshed via POST /v5/refresh-cache.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v5", tags=["v5-forecast"])


# ── Paths & imports ────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[3]
V5_DIR = REPO / "models" / "v5"
V4_DIR = REPO / "models" / "v4"
FORECAST_ARTIFACT_DIR = REPO / "outputs" / "forward_v5_feb12"

# Late-import so the router can be imported without the inference stack
# being loaded (useful for OpenAPI schema generation under minimal envs).
def _imports():
    from edgegrid_forecast.inference.v4_predict import load_model, predict_with_context
    from edgegrid_forecast.inference.v5_predict import predict_recursive
    from edgegrid_forecast.inference._features import (
        compute_fleet_aggregate,
        fetch_weather_expanded,
        load_meter_data,
    )
    return {
        "load_model": load_model,
        "predict_with_context": predict_with_context,
        "predict_recursive": predict_recursive,
        "compute_fleet_aggregate": compute_fleet_aggregate,
        "fetch_weather_expanded": fetch_weather_expanded,
        "load_meter_data": load_meter_data,
    }


# Module-level cache: lazy, but once hydrated persists for the life of the process.
_cache: dict[str, Any] = {
    "manifest": None,
    "meter_index": None,      # msn -> manifest row
    "bundle_by_msn": {},
    "all_df": None,
    "weather_df": None,
    "fleet_df": None,
    "hydrated_at": None,
}


def _load_manifest() -> list[dict]:
    if _cache["manifest"] is None:
        path = V5_DIR / "_manifest.json"
        if not path.exists():
            raise HTTPException(500, f"v5 manifest not found at {path}")
        manifest = json.loads(path.read_text())
        if isinstance(manifest, dict) and "models" in manifest:
            manifest = manifest["models"]
        _cache["manifest"] = manifest
        _cache["meter_index"] = {r["msn"]: r for r in manifest}
    return _cache["manifest"]


def _get_bundle(msn: str):
    if msn in _cache["bundle_by_msn"]:
        return _cache["bundle_by_msn"][msn]
    fns = _imports()
    try:
        bundle = fns["load_model"](msn, models_dir=V5_DIR)
    except FileNotFoundError:
        raise HTTPException(404, f"v5 bundle for msn='{msn}' not found in {V5_DIR}")
    _cache["bundle_by_msn"][msn] = bundle
    return bundle


def _hydrate_frames() -> None:
    """Load shared frames (fleet data, weather, fleet aggregate) once."""
    if _cache["all_df"] is not None and _cache["weather_df"] is not None:
        return
    fns = _imports()
    all_df, _profile = fns["load_meter_data"]()
    wx = fns["fetch_weather_expanded"]()
    fleet = fns["compute_fleet_aggregate"](all_df)
    _cache["all_df"] = all_df
    _cache["weather_df"] = wx
    _cache["fleet_df"] = fleet
    _cache["hydrated_at"] = datetime.now(tz=timezone.utc).isoformat()


# ── Pydantic models ────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    msn: str = Field(..., description="Meter serial number")
    as_of: Optional[str] = Field(
        None,
        description="ISO8601 timestamp marking 't' — forecast starts at t. "
                    "Defaults to the meter's latest actual.",
    )
    horizon_steps: int = Field(
        default=48,
        ge=1,
        le=1440,
        description="Number of 30-min steps to forecast (48 = 24h, 1440 = 30d).",
    )
    mode: str = Field(
        default="batch",
        pattern="^(batch|recursive)$",
        description="batch = single-shot LightGBM predict (fast, ideal <=48 steps). "
                    "recursive = one-step-ahead autoregressive (accurate longer horizons).",
    )


class PredictRow(BaseModel):
    ts: str
    forecast_wh: float
    q10_wh: float
    q90_wh: float
    block_label: str


class PredictResponse(BaseModel):
    msn: str
    tier: Optional[str]
    model_version: str
    as_of: str
    horizon_steps: int
    mode: str
    holdout_mape: Optional[float]
    historical_block_mape: dict
    predictions: list[PredictRow]


class FleetPredictRequest(BaseModel):
    as_of: Optional[str] = None
    horizon_steps: int = Field(default=48, ge=1, le=96)
    msns: Optional[list[str]] = Field(default=None, description="Subset of meters")


class RetrainRequest(BaseModel):
    msn: str
    train_cutoff: Optional[str] = Field(
        default=None,
        description="ISO8601. Defaults to v5 retrain module's DEFAULT_TRAIN_CUTOFF.",
    )


# ── Endpoints ──────────────────────────────────────────────────────────────
@router.get("/healthz")
def healthz():
    try:
        manifest = _load_manifest()
    except HTTPException as e:
        return {
            "status": "degraded",
            "reason": e.detail,
            "n_bundles": 0,
            "hydrated_at": _cache["hydrated_at"],
        }
    mapes = [r.get("holdout_mape") for r in manifest if r.get("holdout_mape") is not None]
    return {
        "status": "healthy",
        "engine": "edgegrid v5",
        "n_bundles": len(manifest),
        "bundles_cached_in_memory": len(_cache["bundle_by_msn"]),
        "frames_hydrated": _cache["all_df"] is not None,
        "hydrated_at": _cache["hydrated_at"],
        "fleet_mape": {
            "mean": round(float(np.mean(mapes)), 3) if mapes else None,
            "median": round(float(np.median(mapes)), 3) if mapes else None,
            "p90": round(float(np.percentile(mapes, 90)), 3) if mapes else None,
            "under_4pct": int(sum(m < 4 for m in mapes)) if mapes else 0,
            "under_10pct": int(sum(m < 10 for m in mapes)) if mapes else 0,
        },
    }


@router.get("/manifest")
def manifest():
    return {"n_meters": len(_load_manifest()), "models": _load_manifest()}


@router.get("/meters")
def meters():
    m = _load_manifest()
    return [
        {
            "msn": r["msn"],
            "tier": r.get("tier"),
            "zero_pct": r.get("zero_pct"),
            "holdout_mape": r.get("holdout_mape"),
            "train_cutoff": r.get("train_cutoff"),
            "version": r.get("version"),
        }
        for r in m
    ]


def _format_predictions(result_df: pd.DataFrame, msn: str, row: dict,
                        as_of: str, horizon_steps: int, mode: str) -> PredictResponse:
    preds = []
    for ts, r in result_df.iterrows():
        preds.append(PredictRow(
            ts=ts.isoformat(),
            forecast_wh=float(r["forecast_wh"]),
            q10_wh=float(r["confidence_low"]),
            q90_wh=float(r["confidence_high"]),
            block_label=str(r.get("block_label", "")),
        ))
    return PredictResponse(
        msn=msn,
        tier=row.get("tier"),
        model_version=row.get("version", "v5.s1.0"),
        as_of=as_of,
        horizon_steps=horizon_steps,
        mode=mode,
        holdout_mape=row.get("holdout_mape"),
        historical_block_mape=row.get("historical_block_mape", {}),
        predictions=preds,
    )


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    manifest = _load_manifest()
    row = _cache["meter_index"].get(req.msn)
    if row is None:
        raise HTTPException(
            404,
            f"msn='{req.msn}' not in v5 manifest. "
            f"Try GET /v5/meters for the full list."
        )
    _hydrate_frames()
    all_df = _cache["all_df"]
    md = all_df[all_df["msn"] == req.msn].sort_values("ts").reset_index(drop=True)
    if md.empty:
        raise HTTPException(404, f"No actuals found for msn='{req.msn}'")

    if req.as_of is None:
        as_of_ts = md["ts"].max()
    else:
        as_of_ts = pd.Timestamp(req.as_of)
    horizon_start = as_of_ts + pd.Timedelta(minutes=30)
    horizon_end = horizon_start + pd.Timedelta(minutes=30 * (req.horizon_steps - 1))
    horizon_ts = pd.date_range(horizon_start, horizon_end, freq="30min")

    # Build priming context: actuals up to as_of + seasonal pseudo for gap
    # and horizon placeholder rows (needed for batch mode feature materialisation).
    ctx_actuals = md[md["ts"] <= as_of_ts][["ts", "demand_wh", "voltage"]].copy()
    if ctx_actuals.empty:
        raise HTTPException(400, f"as_of={as_of_ts} is before earliest actual for {req.msn}")

    # Seasonal template from last 4 weeks
    last_real = ctx_actuals["ts"].max()
    anchor_cut = last_real - pd.Timedelta(days=28)
    a = ctx_actuals[ctx_actuals["ts"] >= anchor_cut].copy()
    if len(a) < 96:
        a = ctx_actuals.copy()
    a["dow"] = a["ts"].dt.dayofweek; a["hour"] = a["ts"].dt.hour; a["minute"] = a["ts"].dt.minute
    amed = a.groupby(["dow","hour","minute"]).agg(
        demand_wh=("demand_wh","median"),
        voltage=("voltage","median"),
    ).reset_index()
    g_med = float(a["demand_wh"].median())

    def _profile_rows(ts_idx: pd.DatetimeIndex) -> pd.DataFrame:
        f = pd.DataFrame({"ts": ts_idx})
        f["dow"] = f["ts"].dt.dayofweek; f["hour"] = f["ts"].dt.hour; f["minute"] = f["ts"].dt.minute
        f = f.merge(amed, on=["dow","hour","minute"], how="left")
        f["demand_wh"] = f["demand_wh"].fillna(g_med)
        f["voltage"]   = f["voltage"].fillna(230.0)
        return f[["ts","demand_wh","voltage"]]

    # Fill gap between last actual and as_of+30min
    gap_needed = horizon_start > last_real + pd.Timedelta(minutes=30)
    gap_rows = pd.DataFrame()
    if gap_needed:
        gap_idx = pd.date_range(last_real + pd.Timedelta(minutes=30),
                                horizon_start - pd.Timedelta(minutes=30),
                                freq="30min")
        if len(gap_idx) > 0:
            gap_rows = _profile_rows(gap_idx)

    fns = _imports()
    if req.mode == "recursive":
        # Recursive wants context ending strictly before horizon_ts[0].
        ctx = pd.concat([ctx_actuals, gap_rows], ignore_index=True)
        ctx = ctx.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        result = fns["predict_recursive"](
            req.msn, ctx, horizon_ts,
            weather_df=_cache["weather_df"],
            fleet_df=_cache["fleet_df"],
            models_dir=V5_DIR,
        )
    else:
        # Batch mode: include horizon placeholder rows so feature materialisation
        # produces a row for each horizon timestamp.
        horizon_ph = _profile_rows(horizon_ts)
        ctx = pd.concat([ctx_actuals, gap_rows, horizon_ph], ignore_index=True)
        ctx = ctx.drop_duplicates("ts", keep="first").sort_values("ts").reset_index(drop=True)
        result = fns["predict_with_context"](
            req.msn, ctx,
            weather_df=_cache["weather_df"],
            fleet_df=_cache["fleet_df"],
            models_dir=V5_DIR,
            horizon_ts=horizon_ts,
        )

    if result.empty:
        raise HTTPException(500, "Predictor returned empty frame")

    return _format_predictions(result, req.msn, row, as_of_ts.isoformat(),
                               req.horizon_steps, req.mode)


@router.post("/predict/fleet")
def predict_fleet(req: FleetPredictRequest):
    manifest = _load_manifest()
    ms = req.msns or [r["msn"] for r in manifest]
    out = []
    for msn in ms:
        sub = PredictRequest(
            msn=msn,
            as_of=req.as_of,
            horizon_steps=req.horizon_steps,
            mode="batch",
        )
        try:
            p = predict(sub)
            out.append({"msn": msn, "status": "ok",
                        "predictions": [r.model_dump() for r in p.predictions]})
        except HTTPException as e:
            out.append({"msn": msn, "status": "error", "detail": e.detail})
    return {"as_of": req.as_of, "horizon_steps": req.horizon_steps, "fleet": out}


@router.post("/retrain")
def retrain(req: RetrainRequest):
    """Fire off a single-meter retrain as a subprocess. Blocks until complete
    (meters retrain in 2–8 seconds with the current feature set)."""
    if req.msn not in _cache["meter_index"] and _load_manifest() is not None:
        # Still allow retraining of MSNs not yet in the manifest (new meters).
        pass
    cmd = [
        sys.executable, "-m", "edgegrid_forecast.training.v5_retrain",
        "--msns", req.msn,
        "--out-dir", str(V5_DIR),
    ]
    if req.train_cutoff:
        cmd += ["--train-cutoff", req.train_cutoff]

    t0 = datetime.now(tz=timezone.utc)
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO),
            env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/usr/local/bin"},
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Retrain timed out after 120s")

    _cache["manifest"] = None
    _cache["meter_index"] = None
    if req.msn in _cache["bundle_by_msn"]:
        del _cache["bundle_by_msn"][req.msn]

    manifest = _load_manifest()
    row = _cache["meter_index"].get(req.msn, {})
    return {
        "msn": req.msn,
        "exit_code": proc.returncode,
        "duration_s": (datetime.now(tz=timezone.utc) - t0).total_seconds(),
        "stdout_tail": proc.stdout.splitlines()[-15:] if proc.stdout else [],
        "stderr_tail": proc.stderr.splitlines()[-15:] if proc.stderr else [],
        "manifest_row": row,
    }


@router.get("/forecast/apr-may")
def apr_may_forecast(
    strategy: str = Query("D", pattern="^[ABCD]$",
                          description="A=seasonal_anchor, B=v4_batch, "
                                      "C=v5_batch_feb12, D=ensemble_blend"),
    msn: Optional[str] = Query(None),
):
    """Return the pre-built 4-strategy forward forecast (Apr 21 → May 21 2026)."""
    strategy_map = {
        "A": FORECAST_ARTIFACT_DIR / "strategy_A.parquet",
        "B": FORECAST_ARTIFACT_DIR / "strategy_B.parquet",
        "C": FORECAST_ARTIFACT_DIR / "strategy_C.parquet",
        "D": FORECAST_ARTIFACT_DIR / "strategy_D.parquet",
    }
    path = strategy_map[strategy]
    if not path.exists():
        raise HTTPException(
            404,
            f"Forecast artifact {path.name} not found. Re-run "
            f"`benchmarks/forward_v5_feb12_strategies.py` to generate."
        )
    df = pd.read_parquet(path)
    if msn:
        df = df[df["meter_id"] == msn]
        if df.empty:
            raise HTTPException(404, f"msn={msn} not in strategy {strategy}")
    df = df.head(2880)  # cap to avoid massive payloads; 2 meters worth of 30-min slots
    return {
        "strategy": strategy,
        "n_rows": len(df),
        "rows": df.assign(ts=df["ts"].astype(str)).to_dict(orient="records"),
    }


@router.post("/refresh-cache")
def refresh_cache():
    _cache["manifest"] = None
    _cache["meter_index"] = None
    _cache["bundle_by_msn"].clear()
    _cache["all_df"] = None
    _cache["weather_df"] = None
    _cache["fleet_df"] = None
    _cache["hydrated_at"] = None
    return {"status": "cache cleared"}
