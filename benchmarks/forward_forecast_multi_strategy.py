"""
Multi-strategy forward forecasts for the 42-meter EdgeGrid fleet, window
2026-04-21 00:00 -> 2026-05-21 00:00 at 30-min cadence.

Strategies
----------
A) "seasonal_anchor" : Seasonal-anchor + weather-adjusted (reuse of existing
   outputs/forward_forecast_30d.parquet; re-labelled into the common schema).
B) "v4_batch" : v4 LightGBM batch prediction. For each meter we build a
   "pseudo-history" ending 2026-04-20 23:30 whose demand_wh values are the
   meter's last-4-weeks (dow, hour, minute) median, so lag features (lag_1,
   lag_24, ... lag_336, rmean_*, etc.) evaluate against realistic in-season
   values. Then predict the 1440 horizon timestamps in a single batch.
C) "v5_recursive" : v5 recursive one-step-ahead LightGBM prediction with the
   same seasonal-anchor priming. Ran on a 6-meter representative sample to
   stay within compute budget (each meter takes ~30s). The rest of the fleet
   gets a "null" column in the viewer and we document that it wasn't
   generated.

Outputs
-------
    outputs/forward_forecast_strategy_A_seasonal.parquet
    outputs/forward_forecast_strategy_B_v4batch.parquet
    outputs/forward_forecast_strategy_C_v5recursive.parquet
    outputs/forward_forecast_all_strategies.parquet   (long-format union)

Schema (shared by all strategy parquets):
    meter_id (str), ts (datetime64[ns]), predicted_kwh, q10_kwh, q90_kwh,
    strategy (str)
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from edgegrid_forecast.inference._features import (  # noqa: E402
    compute_fleet_aggregate,
    fetch_weather_expanded,
    load_meter_data,
)
from edgegrid_forecast.inference.v4_predict import (  # noqa: E402
    predict_with_context,
)

# Re-use the seasonal-anchor helpers from the existing script.
from forward_forecast_apr_may import (  # noqa: E402
    FORECAST_START,
    FORECAST_END,
    N_SLOTS,
    ANCHOR_WEEKS,
    PSEUDO_DAYS,
    _build_seasonal_pseudo_history,
    _extend_weather_for_forecast,
    _weather_adjusted_seasonal,
)


OUT_DIR = REPO / "outputs"
OUT_A = OUT_DIR / "forward_forecast_strategy_A_seasonal.parquet"
OUT_B = OUT_DIR / "forward_forecast_strategy_B_v4batch.parquet"
OUT_C = OUT_DIR / "forward_forecast_strategy_C_v5recursive.parquet"
OUT_ALL = OUT_DIR / "forward_forecast_all_strategies.parquet"

# v5 recursive is ~2 minutes per meter (measured: 128s on cohort A). Sample
# strategy to stay under a 10-minute compute budget:
#  - 1 meter per cohort A/B/C + the lone D meter 50186364
V5_SAMPLE_METERS = [
    "65045250",  # Cohort A (low MAPE, clean profile)
    "67001818",  # Cohort B (representative bulk, HT)
    "53401842",  # Cohort C (higher variance)
    "50186364",  # Cohort D (the outlier)
]


def _meter_msns() -> list[str]:
    manifest_path = REPO / "models" / "v4" / "_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return [m["msn"] for m in manifest["models"]]


# ── Strategy A: rebuild from existing 30d parquet ────────────────────────────
def build_strategy_A() -> pd.DataFrame:
    src = OUT_DIR / "forward_forecast_30d.parquet"
    if not src.exists():
        raise FileNotFoundError(
            f"Strategy A source parquet missing: {src}. "
            f"Run benchmarks/forward_forecast_apr_may.py first."
        )
    df = pd.read_parquet(src)
    out = df[["meter_id", "ts", "predicted_kwh", "q10_kwh", "q90_kwh"]].copy()
    out["strategy"] = "seasonal_anchor"
    out = out.sort_values(["meter_id", "ts"]).reset_index(drop=True)
    return out


# ── Strategy B: v4 batch with seasonal-anchor lag priming ────────────────────
def _v4_batch_one_meter(
    msn: str,
    all_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    fleet_df: pd.DataFrame,
    horizon_ts: pd.DatetimeIndex,
    fallback_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build pseudo-history + real-trailing context, then single-shot predict.

    Returns DataFrame with cols: meter_id, ts, predicted_kwh, q10_kwh, q90_kwh.
    Falls back to Strategy A values on failure (if fallback_df given).
    """
    md = all_df[all_df["msn"] == msn].sort_values("ts").reset_index(drop=True)
    if md.empty:
        if fallback_df is not None:
            sub = fallback_df[fallback_df["meter_id"] == msn][
                ["meter_id", "ts", "predicted_kwh", "q10_kwh", "q90_kwh"]
            ].copy()
            return sub
        return pd.DataFrame()

    # Build a pseudo-history seasonally aligned with the horizon, then append
    # real trailing history before it for a rich lag buffer.
    pseudo_start = FORECAST_START - pd.Timedelta(days=PSEUDO_DAYS)
    pseudo_end = FORECAST_END - pd.Timedelta(minutes=30)
    pseudo = _build_seasonal_pseudo_history(md, pseudo_start, pseudo_end, ANCHOR_WEEKS)
    real_pre = md[md["ts"] < pseudo_start][["ts", "demand_wh", "voltage"]].copy()
    pre_cut = pseudo_start - pd.Timedelta(days=45)
    real_pre = real_pre[real_pre["ts"] >= pre_cut]
    context = pd.concat([real_pre, pseudo], ignore_index=True).drop_duplicates("ts")
    context = context.sort_values("ts").reset_index(drop=True)

    try:
        res = predict_with_context(
            msn, context,
            weather_df=weather_df, fleet_df=fleet_df,
            horizon_ts=horizon_ts,
        )
    except Exception as e:
        print(f"    [{msn}] v4 batch failed ({type(e).__name__}: {e}); "
              f"falling back to strategy A", flush=True)
        traceback.print_exc()
        if fallback_df is not None:
            sub = fallback_df[fallback_df["meter_id"] == msn][
                ["meter_id", "ts", "predicted_kwh", "q10_kwh", "q90_kwh"]
            ].copy()
            return sub
        return pd.DataFrame()

    if res.empty or len(res) != len(horizon_ts):
        print(f"    [{msn}] v4 batch returned {len(res)}/{len(horizon_ts)} rows; "
              f"falling back", flush=True)
        if fallback_df is not None:
            sub = fallback_df[fallback_df["meter_id"] == msn][
                ["meter_id", "ts", "predicted_kwh", "q10_kwh", "q90_kwh"]
            ].copy()
            return sub
        return pd.DataFrame()

    res = res.reset_index()
    # Ensure alignment with horizon_ts order
    res["ts"] = pd.to_datetime(res["ts"])
    res = res.set_index("ts").reindex(horizon_ts).reset_index()
    res = res.rename(columns={"index": "ts"})
    pred_wh = np.nan_to_num(res["forecast_wh"].to_numpy(dtype=np.float64),
                            nan=0.0, posinf=0.0, neginf=0.0)
    q10_wh = np.nan_to_num(res["confidence_low"].to_numpy(dtype=np.float64),
                           nan=0.0, posinf=0.0, neginf=0.0)
    q90_wh = np.nan_to_num(res["confidence_high"].to_numpy(dtype=np.float64),
                           nan=0.0, posinf=0.0, neginf=0.0)
    pred_wh = np.maximum(pred_wh, 0.0)
    q10_wh = np.minimum(np.maximum(q10_wh, 0.0), pred_wh)
    q90_wh = np.maximum(q90_wh, pred_wh)

    out = pd.DataFrame({
        "meter_id": msn,
        "ts": horizon_ts,
        "predicted_kwh": np.round(pred_wh / 1000.0, 6),
        "q10_kwh": np.round(q10_wh / 1000.0, 6),
        "q90_kwh": np.round(q90_wh / 1000.0, 6),
    })
    return out


def build_strategy_B(
    msns: list[str],
    all_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    fleet_df: pd.DataFrame,
    horizon_ts: pd.DatetimeIndex,
    strategy_a_df: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    t_start = time.time()
    for i, msn in enumerate(msns, 1):
        t0 = time.time()
        print(f"  [B {i:>2}/{len(msns)}] {msn} ", end="", flush=True)
        df = _v4_batch_one_meter(
            msn, all_df, weather_df, fleet_df, horizon_ts, fallback_df=strategy_a_df
        )
        if df.empty:
            print(f"[skip]", flush=True)
            continue
        total = float(df["predicted_kwh"].sum())
        print(f"total={total:>9.1f} kWh  ({time.time() - t0:.1f}s)", flush=True)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out["strategy"] = "v4_batch"
    print(f"  strategy B total wallclock: {time.time() - t_start:.1f}s", flush=True)
    return out


# ── Strategy C: v5 recursive (sample) ────────────────────────────────────────
def build_strategy_C(
    msns_sample: list[str],
    all_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    fleet_df: pd.DataFrame,
    horizon_ts: pd.DatetimeIndex,
    strategy_a_df: pd.DataFrame,
    max_seconds: float = 600.0,
) -> pd.DataFrame:
    from edgegrid_forecast.inference.v5_predict import predict_recursive

    frames = []
    t_start = time.time()
    for i, msn in enumerate(msns_sample, 1):
        elapsed = time.time() - t_start
        if elapsed > max_seconds:
            print(f"  [C] compute budget hit ({elapsed:.0f}s > {max_seconds}s); "
                  f"stopping after {i-1}/{len(msns_sample)}", flush=True)
            break
        t0 = time.time()
        print(f"  [C {i:>2}/{len(msns_sample)}] {msn} ", end="", flush=True)
        md = all_df[all_df["msn"] == msn].sort_values("ts").reset_index(drop=True)
        if md.empty:
            print(f"[no data]", flush=True)
            continue

        pseudo_start = FORECAST_START - pd.Timedelta(days=PSEUDO_DAYS)
        pseudo_end = FORECAST_START - pd.Timedelta(minutes=30)
        pseudo = _build_seasonal_pseudo_history(md, pseudo_start, pseudo_end, ANCHOR_WEEKS)
        real_pre = md[md["ts"] < pseudo_start][["ts", "demand_wh", "voltage"]].copy()
        pre_cut = pseudo_start - pd.Timedelta(days=45)
        real_pre = real_pre[real_pre["ts"] >= pre_cut]
        context = pd.concat([real_pre, pseudo], ignore_index=True).drop_duplicates("ts")
        context = context.sort_values("ts").reset_index(drop=True)

        try:
            res = predict_recursive(
                msn, context, horizon_ts,
                weather_df=weather_df, fleet_df=fleet_df, progress=False,
            )
        except Exception as e:
            print(f"[error] {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            continue

        pred_wh = res["forecast_wh"].to_numpy(dtype=np.float64)
        q10_wh = res["confidence_low"].to_numpy(dtype=np.float64)
        q90_wh = res["confidence_high"].to_numpy(dtype=np.float64)
        pred_wh = np.nan_to_num(pred_wh, nan=0.0, posinf=0.0, neginf=0.0)
        q10_wh = np.nan_to_num(q10_wh, nan=0.0, posinf=0.0, neginf=0.0)
        q90_wh = np.nan_to_num(q90_wh, nan=0.0, posinf=0.0, neginf=0.0)
        pred_wh = np.maximum(pred_wh, 0.0)
        q10_wh = np.minimum(np.maximum(q10_wh, 0.0), pred_wh)
        q90_wh = np.maximum(q90_wh, pred_wh)

        df = pd.DataFrame({
            "meter_id": msn,
            "ts": horizon_ts,
            "predicted_kwh": np.round(pred_wh / 1000.0, 6),
            "q10_kwh": np.round(q10_wh / 1000.0, 6),
            "q90_kwh": np.round(q90_wh / 1000.0, 6),
        })
        frames.append(df)
        total = float(df["predicted_kwh"].sum())
        print(f"total={total:>9.1f} kWh  ({time.time() - t0:.1f}s)", flush=True)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["meter_id", "ts", "predicted_kwh", "q10_kwh", "q90_kwh"]
    )
    if not out.empty:
        out["strategy"] = "v5_recursive"
    print(f"  strategy C total wallclock: {time.time() - t_start:.1f}s", flush=True)
    return out


# ── Main ─────────────────────────────────────────────────────────────────────
def main(
    skip_B: bool = False,
    skip_C: bool = False,
    c_max_seconds: float = 600.0,
) -> None:
    print(f"[multi-strategy] window: {FORECAST_START} -> {FORECAST_END}", flush=True)

    horizon_ts = pd.date_range(FORECAST_START, FORECAST_END, freq="30min",
                               inclusive="left")
    assert len(horizon_ts) == N_SLOTS

    # ── Strategy A ──
    print("[multi-strategy] Strategy A: seasonal anchor (reuse existing)", flush=True)
    dfA = build_strategy_A()
    dfA.to_parquet(OUT_A, index=False)
    print(f"  wrote {len(dfA):,} rows -> {OUT_A.name}", flush=True)

    msns = _meter_msns()
    print(f"[multi-strategy] {len(msns)} meters in manifest", flush=True)

    # ── Strategy B ──
    if skip_B and OUT_B.exists():
        print("[multi-strategy] Strategy B: reusing existing parquet", flush=True)
        dfB = pd.read_parquet(OUT_B)
    else:
        print("[multi-strategy] Strategy B: v4 batch (seasonal-anchor priming)", flush=True)
        print("  loading meter data + weather...", flush=True)
        all_df, _ = load_meter_data()
        wx_hist = fetch_weather_expanded()
        wx = _extend_weather_for_forecast(wx_hist, FORECAST_END)
        fleet_df = compute_fleet_aggregate(all_df)
        dfB = build_strategy_B(msns, all_df, wx, fleet_df, horizon_ts, dfA)
        dfB = dfB.sort_values(["meter_id", "ts"]).reset_index(drop=True)
        dfB.to_parquet(OUT_B, index=False)
        print(f"  wrote {len(dfB):,} rows -> {OUT_B.name}", flush=True)

    # ── Strategy C ──
    if skip_C:
        if OUT_C.exists():
            print("[multi-strategy] Strategy C: reusing existing parquet", flush=True)
            dfC = pd.read_parquet(OUT_C)
        else:
            print("[multi-strategy] Strategy C: skipped", flush=True)
            dfC = pd.DataFrame(columns=[
                "meter_id", "ts", "predicted_kwh", "q10_kwh", "q90_kwh", "strategy",
            ])
    else:
        print(f"[multi-strategy] Strategy C: v5 recursive sample "
              f"({len(V5_SAMPLE_METERS)} meters, budget {c_max_seconds:.0f}s)",
              flush=True)
        # Load data only if we haven't yet
        if "all_df" not in locals():
            all_df, _ = load_meter_data()
            wx_hist = fetch_weather_expanded()
            wx = _extend_weather_for_forecast(wx_hist, FORECAST_END)
            fleet_df = compute_fleet_aggregate(all_df)
        dfC = build_strategy_C(
            V5_SAMPLE_METERS, all_df, wx, fleet_df, horizon_ts, dfA,
            max_seconds=c_max_seconds,
        )
        dfC = dfC.sort_values(["meter_id", "ts"]).reset_index(drop=True)
        dfC.to_parquet(OUT_C, index=False)
        print(f"  wrote {len(dfC):,} rows -> {OUT_C.name}", flush=True)

    # ── Union ──
    parts = [dfA, dfB]
    if not dfC.empty:
        parts.append(dfC)
    all_df_out = pd.concat(parts, ignore_index=True)
    all_df_out = all_df_out.sort_values(["strategy", "meter_id", "ts"]).reset_index(drop=True)
    all_df_out.to_parquet(OUT_ALL, index=False)
    print(f"[multi-strategy] wrote combined {len(all_df_out):,} rows -> {OUT_ALL.name}",
          flush=True)

    print("\n" + "=" * 78)
    print("  MULTI-STRATEGY FORWARD FORECAST SUMMARY")
    print("=" * 78)
    for name, df in (("A seasonal_anchor", dfA),
                     ("B v4_batch      ", dfB),
                     ("C v5_recursive  ", dfC)):
        if df.empty:
            print(f"  {name} : (empty)")
            continue
        nm = df["meter_id"].nunique()
        tot = df["predicted_kwh"].sum()
        print(f"  {name} : {nm:>2} meters, total {tot:>12,.1f} kWh, rows {len(df):,}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--skip-B", action="store_true")
    p.add_argument("--skip-C", action="store_true")
    p.add_argument("--c-budget", type=float, default=600.0)
    ns = p.parse_args()
    main(skip_B=ns.skip_B, skip_C=ns.skip_C, c_max_seconds=ns.c_budget)
