"""
Forecast showcase endpoints — "see the v4 engine forecasting accurately, live"
on 42 real APEPDCL meters.

Design principle (per user feedback 2026-04-17): this is the PRIMARY screen.
Dispatch/commercial are secondary. Every endpoint here must deliver a fast,
honest picture of forecast-vs-actual.

Endpoints
---------
- GET  /fleet/summary        → headline accuracy stats for the proof strip
- GET  /meters               → list of 42 meters (grid rows)
- POST /fleet/replay         → batch forecast-vs-actual for many meters at a
                                given historical `as_of` (1 round-trip, not 42)
- GET  /meter/{msn}/history  → full forecast + actual + block breakdown for
                                the detail view

The batch /fleet/replay endpoint is the heart: it runs `predict_with_context`
for each MSN at the requested `as_of` and joins with actuals from the raw
parquets, returning timestamp-aligned series the frontend renders as sparklines.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# ─── cached data loaders ─────────────────────────────────────────────────────


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def _load_manifest() -> dict:
    path = _repo_root() / "models" / "v4" / "_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"models/v4/_manifest.json missing at {path}")
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def _load_actuals() -> pd.DataFrame:
    """
    Concatenate sp_data + tp_data into one tidy frame indexed (msn, ts).
    Column: `actual_wh` = demand_wh. Cached for the process lifetime.
    """
    root = _repo_root() / "data" / "raw"
    frames = []
    for name in ("sp_data.parquet", "tp_data.parquet"):
        path = root / name
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "demand_wh" not in df.columns:
            continue
        frames.append(df[["msn", "ts", "demand_wh"]])
    if not frames:
        return pd.DataFrame(columns=["msn", "ts", "actual_wh"])
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"demand_wh": "actual_wh"})
    out["ts"] = pd.to_datetime(out["ts"])
    out = out.sort_values(["msn", "ts"]).reset_index(drop=True)
    return out


@lru_cache(maxsize=1)
def _meter_summary_df() -> pd.DataFrame:
    """Per-meter summary rows powering /meters endpoint."""
    m = _load_manifest()
    rows = []
    for model in m.get("models", []):
        hb = model.get("historical_block_mape", {}) or {}
        mape = float(model.get("holdout_mape", 0.0))
        if mape < 8:
            health = "green"
        elif mape < 12:
            health = "amber"
        else:
            health = "red"
        rows.append({
            "msn": model["msn"],
            "tier": model.get("tier", "Unknown"),
            "phase": model.get("phase", ""),
            "holdout_mape": mape,
            "night_mape": float(hb.get("night", 0.0)),
            "morning_mape": float(hb.get("morning", 0.0)),
            "solar_mape": float(hb.get("solar", 0.0)),
            "peak_mape": float(hb.get("peak", 0.0)),
            "bias_gate": model.get("bias_gate", ""),
            "health": health,
        })
    df = pd.DataFrame(rows)
    # Sort: HT first (business value), then by holdout MAPE ascending (best first)
    tier_rank = {"HT (>5kWh)": 0}
    df["tier_rank"] = df["tier"].map(tier_rank).fillna(1)
    df = df.sort_values(["tier_rank", "holdout_mape"]).reset_index(drop=True)
    df = df.drop(columns=["tier_rank"])
    return df


# ─── helpers ─────────────────────────────────────────────────────────────────


def _safe_predict(msn: str, as_of: pd.Timestamp, horizon: int) -> Optional[pd.DataFrame]:
    try:
        from ..inference import v4_predict
        return v4_predict.predict(msn, as_of_datetime=as_of, horizon=horizon)
    except Exception as e:  # noqa: BLE001 — never fail a batch over one meter
        return None


def _actuals_for(msn: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = _load_actuals()
    m = df[(df["msn"] == msn) & (df["ts"] >= start) & (df["ts"] <= end)]
    return m[["ts", "actual_wh"]].set_index("ts")


def _compute_mape(forecast: np.ndarray, actual: np.ndarray) -> Optional[float]:
    """MAPE only on points with actual > threshold to avoid divide-by-near-zero."""
    mask = np.isfinite(forecast) & np.isfinite(actual) & (actual > 10)
    if mask.sum() < 4:
        return None
    pct = np.abs(forecast[mask] - actual[mask]) / actual[mask] * 100
    return float(pct.mean())


# ─── request / response models ───────────────────────────────────────────────


class FleetReplayRequest(BaseModel):
    msns: list[str]
    as_of: Optional[str] = None  # ISO timestamp; default = latest actual
    horizon: int = 48            # 30-min intervals (48 = 24h)
    include_actuals: bool = True


class MeterPoint(BaseModel):
    ts: str
    forecast_wh: float
    confidence_low: float
    confidence_high: float
    actual_wh: Optional[float] = None


class MeterReplay(BaseModel):
    msn: str
    mape: Optional[float]
    status: str       # "ok" | "no_forecast" | "no_actuals"
    n_points: int
    forecast_mean_wh: float
    actual_mean_wh: Optional[float]
    tier: str
    peak_mape: float
    series: list[MeterPoint]


class FleetReplayResponse(BaseModel):
    as_of: str
    horizon: int
    fleet_mape: Optional[float]
    fleet_mean_wh: float
    meters: list[MeterReplay]


# ─── router ──────────────────────────────────────────────────────────────────

router = APIRouter(prefix="", tags=["forecast-showcase"])


@router.get("/fleet/summary")
def fleet_summary() -> dict:
    """
    Headline numbers for the proof strip at the top of the Forecast Showcase.
    These are training-time holdout metrics from the manifest — honest, not
    re-computed on the fly.
    """
    m = _load_manifest()
    df = _meter_summary_df()
    peak_mapes = df["peak_mape"].replace(0, np.nan).dropna()
    ht_peak = df[df["tier"].str.startswith("HT")]["peak_mape"].replace(0, np.nan).dropna()

    return {
        "n_meters": int(len(df)),
        "mean_mape_pct": round(float(df["holdout_mape"].mean()), 2),
        "median_mape_pct": round(float(df["holdout_mape"].median()), 2),
        "peak_block_mean_mape_pct": round(float(peak_mapes.mean()), 2) if len(peak_mapes) else None,
        "peak_block_median_mape_pct": round(float(peak_mapes.median()), 2) if len(peak_mapes) else None,
        "ht_peak_median_mape_pct": round(float(ht_peak.median()), 2) if len(ht_peak) else None,
        "model_version": m.get("version"),
        "built_at": m.get("built_at"),
        "health_counts": {
            "green": int((df["health"] == "green").sum()),
            "amber": int((df["health"] == "amber").sum()),
            "red": int((df["health"] == "red").sum()),
        },
        "tier_counts": df["tier"].value_counts().to_dict(),
    }


@router.get("/meters")
def list_meters() -> dict:
    df = _meter_summary_df()
    return {
        "n": int(len(df)),
        "meters": df.to_dict(orient="records"),
    }


@router.get("/fleet/actuals-range")
def actuals_range() -> dict:
    """Latest available actual timestamp per MSN — drives the time scrubber default."""
    a = _load_actuals()
    if a.empty:
        return {"min": None, "max": None, "n_meters": 0}
    return {
        "min": a["ts"].min().isoformat(),
        "max": a["ts"].max().isoformat(),
        "n_meters": int(a["msn"].nunique()),
    }


@router.post("/fleet/replay")
def fleet_replay(body: FleetReplayRequest) -> FleetReplayResponse:
    """
    Batch forecast-vs-actual for N meters at a given `as_of` timestamp.

    For each MSN:
      1. Call `predict()` with as_of_datetime = as_of.
      2. Join predictions with parquet actuals on 30-min timestamp.
      3. Compute per-meter MAPE on the overlap.

    Runs in a thread pool (LightGBM releases the GIL on predict).
    Returns compact arrays to keep the tile sparklines snappy.
    """
    # Default as_of: latest timestamp with real actuals, minus horizon so we
    # have BOTH forecast and actuals to overlay (walk-forward replay).
    # Strip timezone — raw parquets are tz-naive in local wall clock, so any
    # UTC `Z` suffix from a browser `.toISOString()` must be normalized or
    # the backend joins miss every actual.
    if body.as_of:
        as_of = pd.Timestamp(body.as_of)
        if as_of.tzinfo is not None:
            as_of = as_of.tz_convert(None) if hasattr(as_of, "tz_convert") else as_of.replace(tzinfo=None)
    else:
        a = _load_actuals()
        as_of = a["ts"].max() - pd.Timedelta(hours=48) if not a.empty else pd.Timestamp.now()

    results: list[MeterReplay] = []
    summary_df = _meter_summary_df().set_index("msn")

    def _one_meter(msn: str) -> MeterReplay:
        meta = summary_df.loc[msn] if msn in summary_df.index else None
        tier = str(meta["tier"]) if meta is not None else ""
        peak_mape_trained = float(meta["peak_mape"]) if meta is not None else 0.0

        fdf = _safe_predict(msn, as_of, body.horizon)
        if fdf is None or fdf.empty:
            return MeterReplay(
                msn=msn, mape=None, status="no_forecast", n_points=0,
                forecast_mean_wh=0, actual_mean_wh=None, tier=tier,
                peak_mape=peak_mape_trained, series=[],
            )

        start = fdf.index.min()
        end = fdf.index.max()
        actuals = _actuals_for(msn, start, end) if body.include_actuals else pd.DataFrame()
        joined = fdf.join(actuals, how="left")

        fmean = float(joined["forecast_wh"].mean())
        amean = float(joined["actual_wh"].mean()) if "actual_wh" in joined.columns and joined["actual_wh"].notna().any() else None
        mape = None
        if amean is not None:
            mape = _compute_mape(
                joined["forecast_wh"].to_numpy(),
                joined["actual_wh"].to_numpy(),
            )

        status = "ok" if amean is not None else "no_actuals"
        points: list[MeterPoint] = []
        for ts, row in joined.iterrows():
            points.append(MeterPoint(
                ts=pd.Timestamp(ts).isoformat(),
                forecast_wh=float(row["forecast_wh"]),
                confidence_low=float(row["confidence_low"]),
                confidence_high=float(row["confidence_high"]),
                actual_wh=float(row["actual_wh"]) if "actual_wh" in joined.columns and pd.notna(row.get("actual_wh")) else None,
            ))
        return MeterReplay(
            msn=msn, mape=mape, status=status, n_points=len(points),
            forecast_mean_wh=fmean, actual_mean_wh=amean, tier=tier,
            peak_mape=peak_mape_trained, series=points,
        )

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_one_meter, body.msns))

    # Fleet aggregates
    mapes = [r.mape for r in results if r.mape is not None]
    fleet_mape = float(np.mean(mapes)) if mapes else None
    fmeans = [r.forecast_mean_wh for r in results if r.forecast_mean_wh]
    fleet_mean = float(np.mean(fmeans)) if fmeans else 0.0

    return FleetReplayResponse(
        as_of=as_of.isoformat(),
        horizon=body.horizon,
        fleet_mape=fleet_mape,
        fleet_mean_wh=fleet_mean,
        meters=results,
    )


@router.get("/meter/{msn}/history")
def meter_history(msn: str, as_of: Optional[str] = None, horizon: int = 96) -> dict:
    """
    Detail-view payload: longer forecast+actual overlay + block-level breakdown.
    `horizon` defaults to 96 (48h at 30-min) for the big chart.
    """
    summary_df = _meter_summary_df().set_index("msn")
    if msn not in summary_df.index:
        raise HTTPException(404, f"meter {msn} not in manifest")
    meta = summary_df.loc[msn]

    if as_of:
        as_of_ts = pd.Timestamp(as_of)
        if as_of_ts.tzinfo is not None:
            as_of_ts = (
                as_of_ts.tz_convert(None) if hasattr(as_of_ts, "tz_convert")
                else as_of_ts.replace(tzinfo=None)
            )
    else:
        a = _load_actuals()
        as_of_ts = a["ts"].max() - pd.Timedelta(hours=48) if not a.empty else pd.Timestamp.now()

    fdf = _safe_predict(msn, as_of_ts, horizon)
    if fdf is None or fdf.empty:
        raise HTTPException(500, f"predict failed for {msn}")

    actuals = _actuals_for(msn, fdf.index.min(), fdf.index.max())
    joined = fdf.join(actuals, how="left")
    joined["ts"] = joined.index
    joined["error_wh"] = (joined["forecast_wh"] - joined["actual_wh"]).astype(float)

    # Block-level MAPE on the replay window
    block_mape = {}
    for block in ("night", "morning", "solar", "peak"):
        sub = joined[joined["block_label"] == block]
        mape = _compute_mape(sub["forecast_wh"].to_numpy(), sub["actual_wh"].to_numpy()) if len(sub) else None
        block_mape[block] = mape

    series = []
    for ts, row in joined.iterrows():
        series.append({
            "ts": pd.Timestamp(ts).isoformat(),
            "forecast_wh": float(row["forecast_wh"]),
            "confidence_low": float(row["confidence_low"]),
            "confidence_high": float(row["confidence_high"]),
            "actual_wh": float(row["actual_wh"]) if pd.notna(row.get("actual_wh")) else None,
            "block_label": str(row["block_label"]),
        })

    overall_mape = _compute_mape(
        joined["forecast_wh"].to_numpy(), joined["actual_wh"].to_numpy(),
    )

    return {
        "msn": msn,
        "tier": str(meta["tier"]),
        "phase": str(meta["phase"]),
        "as_of": as_of_ts.isoformat(),
        "horizon": horizon,
        "holdout_mape_trained": float(meta["holdout_mape"]),
        "replay_mape_actual": overall_mape,
        "block_mape": block_mape,
        "block_mape_trained": {
            "night": float(meta["night_mape"]),
            "morning": float(meta["morning_mape"]),
            "solar": float(meta["solar_mape"]),
            "peak": float(meta["peak_mape"]),
        },
        "series": series,
    }
