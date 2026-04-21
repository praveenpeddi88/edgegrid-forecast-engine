"""
Forward forecast — 4-strategy fleet-wide rollup for 2026-04-21 → 2026-05-21
using the v5 bundles retrained on actuals through 2026-02-12 15:00.

Strategies
----------
A) seasonal_anchor   — pure (dow, hour, minute) median of last 4 weeks of
                       actuals. Baseline, no model.
B) v4_batch          — v4 LightGBM batch predict with 4-week seasonal-anchor
                       pseudo-history priming the lags/rolls.
C) v5_batch_feb12    — v5 LightGBM (train_cutoff=2026-02-12) batch predict with
                       the same seasonal-anchor priming. This is the engine that
                       beats v4 on every benchmark cohort.
D) ensemble_blend    — per-(meter, ts) median of A, B, C. Robust to individual
                       strategy outliers — deployed as the "safe" forecast.

Output
------
    outputs/forward_v5_feb12/strategy_A.parquet
    outputs/forward_v5_feb12/strategy_B.parquet
    outputs/forward_v5_feb12/strategy_C.parquet
    outputs/forward_v5_feb12/strategy_D.parquet
    outputs/forward_v5_feb12/all_strategies.parquet   (long-form union)
    outputs/forward_v5_feb12/per_strategy_rollup.csv  (fleet totals per strat)

Schema (shared):
    meter_id, ts, predicted_kwh, q10_kwh, q90_kwh, strategy
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from edgegrid_forecast.inference._features import (  # noqa: E402
    compute_fleet_aggregate,
    fetch_weather_expanded,
    load_meter_data,
)
from edgegrid_forecast.inference.v4_predict import predict_with_context  # noqa: E402


# ── Config ─────────────────────────────────────────────────────────────────
FORECAST_START = pd.Timestamp("2026-04-21 00:00:00")
FORECAST_END   = pd.Timestamp("2026-05-21 00:00:00")     # exclusive
N_SLOTS        = 1440                                    # 30 days × 48
ANCHOR_WEEKS   = 4
PSEUDO_DAYS    = 35

OUT_DIR = REPO / "outputs" / "forward_v5_feb12"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_A = OUT_DIR / "strategy_A.parquet"
OUT_B = OUT_DIR / "strategy_B.parquet"
OUT_C = OUT_DIR / "strategy_C.parquet"
OUT_D = OUT_DIR / "strategy_D.parquet"
OUT_ALL = OUT_DIR / "all_strategies.parquet"
OUT_ROLLUP = OUT_DIR / "per_strategy_rollup.csv"
OUT_README = OUT_DIR / "README.md"

V4_DIR = REPO / "models" / "v4"
V5_DIR = REPO / "models" / "v5"


def _msns_v5() -> list[str]:
    """Use the v5 manifest (42 meters with Feb-12 cutoff)."""
    manifest = json.loads((V5_DIR / "_manifest.json").read_text())
    return sorted([m["msn"] for m in manifest])


def _extend_weather(wx: pd.DataFrame, target_end: pd.Timestamp) -> pd.DataFrame:
    wx = wx.sort_values("ts").reset_index(drop=True).copy()
    if wx["ts"].max() >= target_end:
        return wx
    need_start = wx["ts"].max() + pd.Timedelta(minutes=30)
    need_idx = pd.date_range(need_start, target_end, freq="30min")
    src = wx.set_index("ts")
    pad = pd.DataFrame({"ts": need_idx})
    shift_src = pad["ts"] - pd.Timedelta(days=365)
    for c in [c for c in wx.columns if c != "ts"]:
        pad[c] = src[c].reindex(shift_src).values
    out = pd.concat([wx, pad], ignore_index=True).drop_duplicates("ts")
    feats = [c for c in out.columns if c != "ts"]
    out = out.sort_values("ts").reset_index(drop=True)
    out[feats] = out[feats].ffill().bfill()
    return out


def _seasonal_anchor(mdata: pd.DataFrame,
                     horizon_ts: pd.DatetimeIndex,
                     anchor_weeks: int = ANCHOR_WEEKS) -> pd.DataFrame:
    """(dow, hour, minute) median of last `anchor_weeks` weeks of actuals."""
    md = mdata.sort_values("ts").reset_index(drop=True)
    last = md["ts"].max()
    anchor = md[md["ts"] >= last - pd.Timedelta(days=7 * anchor_weeks)].copy()
    if len(anchor) < 96:
        anchor = md.copy()
    anchor["dow"]    = anchor["ts"].dt.dayofweek
    anchor["hour"]   = anchor["ts"].dt.hour
    anchor["minute"] = anchor["ts"].dt.minute
    grp = anchor.groupby(["dow", "hour", "minute"])["demand_wh"]
    med = grp.median().rename("mean_wh").reset_index()
    q10 = grp.quantile(0.10).rename("q10_wh").reset_index()
    q90 = grp.quantile(0.90).rename("q90_wh").reset_index()

    h = pd.DataFrame({"ts": horizon_ts})
    h["dow"]    = h["ts"].dt.dayofweek
    h["hour"]   = h["ts"].dt.hour
    h["minute"] = h["ts"].dt.minute
    h = h.merge(med, on=["dow", "hour", "minute"], how="left") \
         .merge(q10, on=["dow", "hour", "minute"], how="left") \
         .merge(q90, on=["dow", "hour", "minute"], how="left")
    g_med = float(np.nanmedian(anchor["demand_wh"].values))
    for col, factor in (("mean_wh", 1.0), ("q10_wh", 0.7), ("q90_wh", 1.3)):
        h[col] = h[col].fillna(g_med * factor)
    return h[["ts", "mean_wh", "q10_wh", "q90_wh"]]


def _build_priming_context(mdata: pd.DataFrame,
                           horizon_ts: pd.DatetimeIndex,
                           anchor_weeks: int = ANCHOR_WEEKS,
                           pseudo_days: int = PSEUDO_DAYS) -> pd.DataFrame:
    """Build (real_pre + seasonal_pseudo + horizon_placeholder) history so the
    feature builder can materialise feature rows AT every horizon timestamp.
    The horizon placeholder uses the seasonal (dow, hour, minute) median for
    its demand_wh — the batch predictor reads lag_N / rmean_N from shift(N) of
    the whole context, so this priming stays in-distribution for the model.
    """
    md = mdata.sort_values("ts").reset_index(drop=True)
    pseudo_end = FORECAST_START - pd.Timedelta(minutes=30)
    pseudo_start = FORECAST_START - pd.Timedelta(days=pseudo_days)

    anchor_cut = md["ts"].max() - pd.Timedelta(days=7 * anchor_weeks)
    a = md[md["ts"] >= anchor_cut].copy()
    if len(a) < 96:
        a = md.copy()
    a["dow"]    = a["ts"].dt.dayofweek
    a["hour"]   = a["ts"].dt.hour
    a["minute"] = a["ts"].dt.minute
    amed = a.groupby(["dow", "hour", "minute"]).agg(
        demand_wh=("demand_wh", "median"),
        voltage=("voltage", "median"),
    ).reset_index()

    g_med = float(md["demand_wh"].median())

    def _profile(ts_idx: pd.DatetimeIndex) -> pd.DataFrame:
        f = pd.DataFrame({"ts": ts_idx})
        f["dow"]    = f["ts"].dt.dayofweek
        f["hour"]   = f["ts"].dt.hour
        f["minute"] = f["ts"].dt.minute
        f = f.merge(amed, on=["dow", "hour", "minute"], how="left")
        f["demand_wh"] = f["demand_wh"].fillna(g_med)
        f["voltage"]   = f["voltage"].fillna(230.0)
        return f[["ts", "demand_wh", "voltage"]]

    pseudo = _profile(pd.date_range(pseudo_start, pseudo_end, freq="30min"))
    horizon_ph = _profile(pd.DatetimeIndex(horizon_ts))

    real_pre = md[md["ts"] < pseudo_start][["ts", "demand_wh", "voltage"]]
    real_pre = real_pre[real_pre["ts"] >= pseudo_start - pd.Timedelta(days=60)]
    ctx = pd.concat([real_pre, pseudo, horizon_ph], ignore_index=True)
    ctx = ctx.drop_duplicates("ts", keep="first").sort_values("ts").reset_index(drop=True)
    return ctx


def _as_strategy_df(horizon_ts, pred_wh, q10_wh, q90_wh, msn, strategy) -> pd.DataFrame:
    pred_wh = np.asarray(pred_wh, dtype=np.float64)
    q10_wh  = np.asarray(q10_wh,  dtype=np.float64)
    q90_wh  = np.asarray(q90_wh,  dtype=np.float64)
    pred_kwh = np.maximum(np.nan_to_num(pred_wh, nan=0.0) / 1000.0, 0.0)
    q10_kwh  = np.maximum(np.nan_to_num(q10_wh,  nan=0.0) / 1000.0, 0.0)
    q90_kwh  = np.maximum(np.nan_to_num(q90_wh,  nan=0.0) / 1000.0, 0.0)
    q10_kwh  = np.minimum(q10_kwh, pred_kwh)
    q90_kwh  = np.maximum(q90_kwh, pred_kwh)
    return pd.DataFrame({
        "meter_id": msn,
        "ts": horizon_ts,
        "predicted_kwh": np.round(pred_kwh, 6),
        "q10_kwh":       np.round(q10_kwh, 6),
        "q90_kwh":       np.round(q90_kwh, 6),
        "strategy":      strategy,
    })


def forecast_one_meter(msn: str,
                       all_df: pd.DataFrame,
                       wx: pd.DataFrame,
                       fleet_df: pd.DataFrame,
                       horizon_ts: pd.DatetimeIndex) -> dict:
    md = all_df[all_df["msn"] == msn].sort_values("ts").reset_index(drop=True)
    out = {}
    if md.empty:
        zeros = np.zeros(len(horizon_ts))
        for s in ("A", "B", "C"):
            out[s] = _as_strategy_df(horizon_ts, zeros, zeros, zeros, msn,
                                     {"A": "seasonal_anchor",
                                      "B": "v4_batch",
                                      "C": "v5_batch_feb12"}[s])
        return out

    # A: seasonal anchor
    sa = _seasonal_anchor(md, horizon_ts)
    out["A"] = _as_strategy_df(horizon_ts, sa["mean_wh"].values,
                               sa["q10_wh"].values, sa["q90_wh"].values,
                               msn, "seasonal_anchor")

    # Priming context shared by B and C — includes horizon placeholder rows
    ctx = _build_priming_context(md, horizon_ts)

    # B: v4 batch
    try:
        r = predict_with_context(msn, ctx, weather_df=wx, fleet_df=fleet_df,
                                 models_dir=V4_DIR, horizon_ts=horizon_ts)
        pB = r["forecast_wh"].reindex(horizon_ts).values
        q10B = r["confidence_low"].reindex(horizon_ts).values
        q90B = r["confidence_high"].reindex(horizon_ts).values
    except Exception as e:
        print(f"    {msn} B failed: {e}; using seasonal anchor", flush=True)
        pB, q10B, q90B = sa["mean_wh"].values, sa["q10_wh"].values, sa["q90_wh"].values
    out["B"] = _as_strategy_df(horizon_ts, pB, q10B, q90B, msn, "v4_batch")

    # C: v5 batch (feb 12)
    try:
        r = predict_with_context(msn, ctx, weather_df=wx, fleet_df=fleet_df,
                                 models_dir=V5_DIR, horizon_ts=horizon_ts)
        pC = r["forecast_wh"].reindex(horizon_ts).values
        q10C = r["confidence_low"].reindex(horizon_ts).values
        q90C = r["confidence_high"].reindex(horizon_ts).values
    except Exception as e:
        print(f"    {msn} C failed: {e}; using seasonal anchor", flush=True)
        pC, q10C, q90C = sa["mean_wh"].values, sa["q10_wh"].values, sa["q90_wh"].values
    out["C"] = _as_strategy_df(horizon_ts, pC, q10C, q90C, msn, "v5_batch_feb12")

    # D: ensemble blend = per-row median(A, B, C) for mean, q10, q90
    pD = np.nanmedian(np.vstack([out["A"]["predicted_kwh"].values * 1000.0,
                                 out["B"]["predicted_kwh"].values * 1000.0,
                                 out["C"]["predicted_kwh"].values * 1000.0]), axis=0)
    q10D = np.nanmedian(np.vstack([out["A"]["q10_kwh"].values * 1000.0,
                                   out["B"]["q10_kwh"].values * 1000.0,
                                   out["C"]["q10_kwh"].values * 1000.0]), axis=0)
    q90D = np.nanmedian(np.vstack([out["A"]["q90_kwh"].values * 1000.0,
                                   out["B"]["q90_kwh"].values * 1000.0,
                                   out["C"]["q90_kwh"].values * 1000.0]), axis=0)
    out["D"] = _as_strategy_df(horizon_ts, pD, q10D, q90D, msn, "ensemble_blend")
    return out


def main(msns_filter: list[str] | None = None) -> None:
    t_all = time.time()
    print(f"[v5-feb12-forecast] loading data...", flush=True)
    all_df, _profile = load_meter_data()
    msns = _msns_v5()
    if msns_filter:
        msns = [m for m in msns if m in msns_filter]
    print(f"  {len(msns)} meters, {len(all_df):,} rows across fleet", flush=True)

    print("[v5-feb12-forecast] extending weather through forecast end...", flush=True)
    wx = _extend_weather(fetch_weather_expanded(), FORECAST_END)
    print(f"  weather ts: {wx['ts'].min()} -> {wx['ts'].max()}", flush=True)

    print("[v5-feb12-forecast] computing fleet aggregate...", flush=True)
    fleet_df = compute_fleet_aggregate(all_df)

    horizon = pd.date_range(FORECAST_START,
                            FORECAST_END - pd.Timedelta(minutes=30),
                            freq="30min")
    print(f"  horizon: {len(horizon)} steps {horizon[0]} -> {horizon[-1]}", flush=True)

    per_strategy: dict[str, list[pd.DataFrame]] = {k: [] for k in ("A", "B", "C", "D")}
    t_start = time.time()
    for i, msn in enumerate(msns, 1):
        t0 = time.time()
        out = forecast_one_meter(msn, all_df, wx, fleet_df, horizon)
        for k in ("A", "B", "C", "D"):
            per_strategy[k].append(out[k])
        totC = out["C"]["predicted_kwh"].sum()
        print(f"  [{i:>2}/{len(msns)}] {msn} "
              f"A={out['A']['predicted_kwh'].sum():>9.1f} "
              f"B={out['B']['predicted_kwh'].sum():>9.1f} "
              f"C={totC:>9.1f} "
              f"D={out['D']['predicted_kwh'].sum():>9.1f} kWh  "
              f"({time.time()-t0:4.1f}s)", flush=True)
    print(f"[v5-feb12-forecast] compute: {time.time()-t_start:.1f}s", flush=True)

    dfA = pd.concat(per_strategy["A"], ignore_index=True)
    dfB = pd.concat(per_strategy["B"], ignore_index=True)
    dfC = pd.concat(per_strategy["C"], ignore_index=True)
    dfD = pd.concat(per_strategy["D"], ignore_index=True)

    dfA.to_parquet(OUT_A, index=False); print(f"  wrote {len(dfA):,} rows → {OUT_A.name}")
    dfB.to_parquet(OUT_B, index=False); print(f"  wrote {len(dfB):,} rows → {OUT_B.name}")
    dfC.to_parquet(OUT_C, index=False); print(f"  wrote {len(dfC):,} rows → {OUT_C.name}")
    dfD.to_parquet(OUT_D, index=False); print(f"  wrote {len(dfD):,} rows → {OUT_D.name}")

    all_df_out = pd.concat([dfA, dfB, dfC, dfD], ignore_index=True)
    all_df_out.to_parquet(OUT_ALL, index=False)
    print(f"  wrote {len(all_df_out):,} rows → {OUT_ALL.name}")

    # Per-strategy fleet rollup
    rollup = (all_df_out.groupby("strategy")["predicted_kwh"]
              .agg(total_kwh="sum", mean_kwh_per_30min="mean")
              .reset_index())
    rollup["expected_per_30min"] = rollup["total_kwh"] / N_SLOTS / len(msns)
    rollup.to_csv(OUT_ROLLUP, index=False)
    print("\n" + "═" * 70)
    print("  4-STRATEGY FORWARD FORECAST (Apr 21 -> May 21, 2026)")
    print("═" * 70)
    print(rollup.to_string(index=False))
    print(f"\n  total compute: {time.time()-t_all:.1f}s")

    readme = f"""# Forward Forecast — Apr 21 → May 21 2026 (4 strategies)

- Generated: `{pd.Timestamp.now().isoformat()}`
- Meters: **{len(msns)}** (v5 fleet, Feb-12 train cutoff)
- Horizon: `{FORECAST_START}` → `{FORECAST_END}` (30 days × 48 half-hourly = 1440 slots)
- Rows per strategy: **{len(dfA):,}** (42 × 1440)

## Strategies

| id | name | engine | priming | notes |
|----|------|--------|---------|-------|
| A  | seasonal_anchor  | none (median lookup) | 4-week (dow,h,min) | no weather, no model — baseline |
| B  | v4_batch         | v4 LightGBM quantile | 4-week seasonal pseudo-history + real pre | pre-v5 reference engine |
| C  | v5_batch_feb12   | v5 LightGBM quantile | 4-week seasonal pseudo-history + real pre | v5 bundles retrained through 2026-02-12 15:00 |
| D  | ensemble_blend   | per-(meter,ts) median(A, B, C) | — | robust to single-strategy outliers |

## Why priming?
Last verified actuals are 2026-02-12 ~15:00 — that's 67 days before the
forecast-window start (2026-04-21 00:00). A recursive forecast across 67+30 days
of prediction-fed lags accumulates massive drift. Instead we lay down a 35-day
synthetic pseudo-history built from (dow, hour, minute) medians of the last 4
weeks of real actuals, so the lag/roll/similar-day features evaluate against
in-distribution values.

## Output files

All parquets share the schema `meter_id, ts, predicted_kwh, q10_kwh, q90_kwh, strategy`.

- `strategy_A.parquet` — seasonal-anchor forecasts (42 × 1440 = 60,480 rows)
- `strategy_B.parquet` — v4 batch forecasts
- `strategy_C.parquet` — v5 batch forecasts (Feb-12 bundles)
- `strategy_D.parquet` — per-row median ensemble
- `all_strategies.parquet` — long-form union (241,920 rows)
- `per_strategy_rollup.csv` — fleet totals per strategy

## Known caveats

- **Batch vs recursive.** Strategies B and C use single-shot batch prediction on
  the primed pseudo-history. This matches how the bundles are trained (lags are
  seen directly) and is the same regime the bundle-holdout MAPE measures, but
  does NOT re-feed predictions back into lags mid-horizon. For recursive
  inference (one-step-ahead), use `edgegrid_forecast.inference.v5_predict.predict_recursive`.
- **Weather extrapolation.** Weather beyond `2026-04-21` is padded from
  same-day-of-year 2025, then ffilled. This is climatological, not a real forecast.
- **No ground truth.** These are forecasts, not backtests. When Apr-May 2026
  actuals arrive, per-meter MAPE can be computed by joining on `meter_id, ts`.
"""
    OUT_README.write_text(readme)
    print(f"\n  → README: {OUT_README}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--msns", type=str, default=None,
                   help="comma-sep MSNs to forecast (default: all 42)")
    args = p.parse_args()
    ms = args.msns.split(",") if args.msns else None
    main(msns_filter=ms)
