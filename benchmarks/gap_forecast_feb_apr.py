"""
Gap-window forecasts for the 42-meter EdgeGrid fleet, Feb 13 -> Apr 20, 2026
(67 days x 48 half-hourly = 3216 slots per meter).

Bridges the 67-day forecast gap between the last available actual
(2026-02-12 15:00) and the start of the existing forward forecast
(2026-04-21 00:00). The four strategies mirror the existing
`forward_forecast_multi_strategy.py` output for the Apr 21 -> May 21 window so
the gap parquets can be concatenated meter-by-meter into a continuous 98-day
forecast.

Strategies
----------
A) "seasonal_anchor" : (dow, hour, minute) median over the last 4 weeks of
   real actuals, with a per-meter per-hour weather-sensitivity overlay
   computed against ACTUAL historical weather for Feb 13 -> Apr 20 (the cache
   has full coverage there).

B) "v4_batch" : v4 LightGBM batch prediction. Pseudo-history seeded from the
   seasonal anchor + 45-day real trailing window from before Feb 13 priming
   the lag features.

C) "v5_recursive" : v5 one-step-ahead recursive prediction with the same
   seasonal-anchor priming. Run on a 4-meter representative sample (median
   MAPE meter from each of cohorts A/B/C/D) per the budget guidance.

D) "hybrid" : first 7 days from Strategy C (or B for non-sampled meters),
   then Strategy A for the remaining 60 days. Concatenated by meter without
   blending (predictions are in kWh on the same energy basis).

Outputs (all in long format with shared schema):
    outputs/gap_strategy_A_seasonal.parquet
    outputs/gap_strategy_B_v4batch.parquet
    outputs/gap_strategy_C_v5recursive.parquet
    outputs/gap_strategy_D_hybrid.parquet
    outputs/gap_all_strategies.parquet                  (union of A/B/C/D)

Schema:
    meter_id (str), ts (datetime64[ns]), predicted_kwh, q10_kwh, q90_kwh,
    strategy (str), model_version (str), generated_at (str ISO8601)
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
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

# Reuse seasonal-anchor + pseudo-history helpers
from forward_forecast_apr_may import (  # noqa: E402
    ANCHOR_WEEKS,
    PSEUDO_DAYS,
    _build_seasonal_pseudo_history,
    _weather_adjusted_seasonal,
)


# ── Configuration ──────────────────────────────────────────────────────────
GAP_START = pd.Timestamp("2026-02-13 00:00:00")
GAP_END   = pd.Timestamp("2026-04-21 00:00:00")  # exclusive (last covered slot is 04-20 23:30)
N_SLOTS   = 67 * 48                              # 3216
HYBRID_BOUNDARY = GAP_START + pd.Timedelta(days=7)  # 2026-02-20 00:00
MODEL_VERSION_A = "v5.s1.0-seasonal-anchor"
MODEL_VERSION_B = "v4.s1.0-batch"
MODEL_VERSION_C = "v5.s1.0-recursive"
MODEL_VERSION_D = "hybrid-7d-recursive+60d-anchor"

OUT_DIR = REPO / "outputs"
OUT_A = OUT_DIR / "gap_strategy_A_seasonal.parquet"
OUT_B = OUT_DIR / "gap_strategy_B_v4batch.parquet"
OUT_C = OUT_DIR / "gap_strategy_C_v5recursive.parquet"
OUT_D = OUT_DIR / "gap_strategy_D_hybrid.parquet"
OUT_ALL = OUT_DIR / "gap_all_strategies.parquet"

# Median-MAPE meter from each cohort in outputs/block_accuracy_summary.csv
V5_SAMPLE_METERS = [
    "65002231",  # Cohort A median MAPE 4.35%
    "65015026",  # Cohort B median MAPE 7.83%
    "65021124",  # Cohort C median MAPE 21.91%
    "50186364",  # Cohort D (only meter)
]


# ── Helpers ────────────────────────────────────────────────────────────────
def _meter_msns() -> list[str]:
    manifest_path = REPO / "models" / "v4" / "_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return [m["msn"] for m in manifest["models"]]


def _add_schema_cols(df: pd.DataFrame, strategy: str, model_version: str,
                     generated_at: str) -> pd.DataFrame:
    df = df.copy()
    df["strategy"] = strategy
    df["model_version"] = model_version
    df["generated_at"] = generated_at
    return df


def _clean_arrays(pred: np.ndarray, q10: np.ndarray, q90: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    q10 = np.nan_to_num(q10,  nan=0.0, posinf=0.0, neginf=0.0)
    q90 = np.nan_to_num(q90,  nan=0.0, posinf=0.0, neginf=0.0)
    pred = np.maximum(pred, 0.0)
    q10 = np.minimum(np.maximum(q10, 0.0), pred)
    q90 = np.maximum(q90, pred)
    return pred, q10, q90


# ── Strategy A: seasonal anchor + REAL historical weather ──────────────────
def _strategy_A_one_meter(
    msn: str,
    all_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    horizon_ts: pd.DatetimeIndex,
) -> pd.DataFrame:
    md = all_df[all_df["msn"] == msn].sort_values("ts").reset_index(drop=True)
    if md.empty:
        return pd.DataFrame()
    pred_wh, q10_wh, q90_wh = _weather_adjusted_seasonal(
        md, weather_df, horizon_ts, ANCHOR_WEEKS
    )
    pred_wh, q10_wh, q90_wh = _clean_arrays(pred_wh, q10_wh, q90_wh)
    return pd.DataFrame({
        "meter_id": msn,
        "ts": horizon_ts,
        "predicted_kwh": np.round(pred_wh / 1000.0, 6),
        "q10_kwh": np.round(q10_wh / 1000.0, 6),
        "q90_kwh": np.round(q90_wh / 1000.0, 6),
    })


def build_strategy_A(
    msns: list[str],
    all_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    horizon_ts: pd.DatetimeIndex,
) -> pd.DataFrame:
    frames = []
    t_start = time.time()
    for i, msn in enumerate(msns, 1):
        t0 = time.time()
        df = _strategy_A_one_meter(msn, all_df, weather_df, horizon_ts)
        if df.empty:
            print(f"  [A {i:>2}/{len(msns)}] {msn} [no data]", flush=True)
            continue
        total = float(df["predicted_kwh"].sum())
        print(f"  [A {i:>2}/{len(msns)}] {msn} total={total:>10,.1f} kWh "
              f"({time.time()-t0:.1f}s)", flush=True)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"  strategy A wallclock: {time.time()-t_start:.1f}s", flush=True)
    return out


# ── Strategy B: v4 batch with seasonal-anchor priming ──────────────────────
def _strategy_B_one_meter(
    msn: str,
    all_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    fleet_df: pd.DataFrame,
    horizon_ts: pd.DatetimeIndex,
    fallback_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    md = all_df[all_df["msn"] == msn].sort_values("ts").reset_index(drop=True)
    if md.empty:
        if fallback_df is not None:
            return fallback_df[fallback_df["meter_id"] == msn][
                ["meter_id", "ts", "predicted_kwh", "q10_kwh", "q90_kwh"]
            ].copy()
        return pd.DataFrame()

    # Pseudo-history covers (gap_start - PSEUDO_DAYS) -> gap_end - 30min so the
    # feature builder has rich lag/rolling values for every horizon row.
    pseudo_start = GAP_START - pd.Timedelta(days=PSEUDO_DAYS)
    pseudo_end = GAP_END - pd.Timedelta(minutes=30)
    pseudo = _build_seasonal_pseudo_history(md, pseudo_start, pseudo_end, ANCHOR_WEEKS)
    real_pre = md[md["ts"] < pseudo_start][["ts", "demand_wh", "voltage"]].copy()
    pre_cut = pseudo_start - pd.Timedelta(days=45)
    real_pre = real_pre[real_pre["ts"] >= pre_cut]
    context = pd.concat([real_pre, pseudo], ignore_index=True).drop_duplicates("ts")
    context = context.sort_values("ts").reset_index(drop=True)

    try:
        res = predict_with_context(
            msn, context, weather_df=weather_df, fleet_df=fleet_df,
            horizon_ts=horizon_ts,
        )
    except Exception as e:
        print(f"    [{msn}] v4 batch failed ({type(e).__name__}: {e}); fallback to A",
              flush=True)
        if fallback_df is not None:
            return fallback_df[fallback_df["meter_id"] == msn][
                ["meter_id", "ts", "predicted_kwh", "q10_kwh", "q90_kwh"]
            ].copy()
        return pd.DataFrame()

    if res.empty or len(res) != len(horizon_ts):
        if fallback_df is not None:
            return fallback_df[fallback_df["meter_id"] == msn][
                ["meter_id", "ts", "predicted_kwh", "q10_kwh", "q90_kwh"]
            ].copy()
        return pd.DataFrame()

    res = res.reset_index()
    res["ts"] = pd.to_datetime(res["ts"])
    res = res.set_index("ts").reindex(horizon_ts).reset_index()
    res = res.rename(columns={"index": "ts"})
    pred_wh = res["forecast_wh"].to_numpy(dtype=np.float64)
    q10_wh = res["confidence_low"].to_numpy(dtype=np.float64)
    q90_wh = res["confidence_high"].to_numpy(dtype=np.float64)
    pred_wh, q10_wh, q90_wh = _clean_arrays(pred_wh, q10_wh, q90_wh)
    return pd.DataFrame({
        "meter_id": msn,
        "ts": horizon_ts,
        "predicted_kwh": np.round(pred_wh / 1000.0, 6),
        "q10_kwh": np.round(q10_wh / 1000.0, 6),
        "q90_kwh": np.round(q90_wh / 1000.0, 6),
    })


def build_strategy_B(
    msns: list[str],
    all_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    fleet_df: pd.DataFrame,
    horizon_ts: pd.DatetimeIndex,
    strategy_A_df: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    t_start = time.time()
    for i, msn in enumerate(msns, 1):
        t0 = time.time()
        print(f"  [B {i:>2}/{len(msns)}] {msn} ", end="", flush=True)
        df = _strategy_B_one_meter(
            msn, all_df, weather_df, fleet_df, horizon_ts,
            fallback_df=strategy_A_df,
        )
        if df.empty:
            print(f"[skip]", flush=True)
            continue
        total = float(df["predicted_kwh"].sum())
        print(f"total={total:>10,.1f} kWh ({time.time()-t0:.1f}s)", flush=True)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"  strategy B wallclock: {time.time()-t_start:.1f}s", flush=True)
    return out


# ── Strategy C: v5 recursive (sample) ──────────────────────────────────────
def _strategy_C_one_meter(
    msn: str,
    all_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    fleet_df: pd.DataFrame,
    horizon_ts: pd.DatetimeIndex,
) -> pd.DataFrame:
    from edgegrid_forecast.inference.v5_predict import predict_recursive

    md = all_df[all_df["msn"] == msn].sort_values("ts").reset_index(drop=True)
    if md.empty:
        return pd.DataFrame()

    pseudo_start = GAP_START - pd.Timedelta(days=PSEUDO_DAYS)
    pseudo_end = GAP_START - pd.Timedelta(minutes=30)
    pseudo = _build_seasonal_pseudo_history(md, pseudo_start, pseudo_end, ANCHOR_WEEKS)
    real_pre = md[md["ts"] < pseudo_start][["ts", "demand_wh", "voltage"]].copy()
    pre_cut = pseudo_start - pd.Timedelta(days=45)
    real_pre = real_pre[real_pre["ts"] >= pre_cut]
    context = pd.concat([real_pre, pseudo], ignore_index=True).drop_duplicates("ts")
    context = context.sort_values("ts").reset_index(drop=True)

    res = predict_recursive(
        msn, context, horizon_ts,
        weather_df=weather_df, fleet_df=fleet_df, progress=False,
    )
    pred_wh = res["forecast_wh"].to_numpy(dtype=np.float64)
    q10_wh = res["confidence_low"].to_numpy(dtype=np.float64)
    q90_wh = res["confidence_high"].to_numpy(dtype=np.float64)
    pred_wh, q10_wh, q90_wh = _clean_arrays(pred_wh, q10_wh, q90_wh)
    return pd.DataFrame({
        "meter_id": msn,
        "ts": horizon_ts,
        "predicted_kwh": np.round(pred_wh / 1000.0, 6),
        "q10_kwh": np.round(q10_wh / 1000.0, 6),
        "q90_kwh": np.round(q90_wh / 1000.0, 6),
    })


def build_strategy_C(
    msns_sample: list[str],
    all_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    fleet_df: pd.DataFrame,
    horizon_ts: pd.DatetimeIndex,
    max_seconds_per_meter: float = 600.0,
    total_budget_seconds: float = 2400.0,
) -> pd.DataFrame:
    frames = []
    t_start = time.time()
    for i, msn in enumerate(msns_sample, 1):
        elapsed = time.time() - t_start
        if elapsed > total_budget_seconds:
            print(f"  [C] total budget hit ({elapsed:.0f}s > {total_budget_seconds}s); "
                  f"stopping at {i-1}/{len(msns_sample)}", flush=True)
            break
        t0 = time.time()
        print(f"  [C {i}/{len(msns_sample)}] {msn} ", end="", flush=True)
        try:
            df = _strategy_C_one_meter(
                msn, all_df, weather_df, fleet_df, horizon_ts,
            )
        except Exception as e:
            print(f"[error] {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            continue
        wall = time.time() - t0
        if wall > max_seconds_per_meter:
            print(f"total={float(df['predicted_kwh'].sum()):>10,.1f} kWh ({wall:.1f}s) "
                  f"[over per-meter cap; finishing this meter, may stop next]",
                  flush=True)
        else:
            total = float(df['predicted_kwh'].sum())
            print(f"total={total:>10,.1f} kWh ({wall:.1f}s)", flush=True)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["meter_id", "ts", "predicted_kwh", "q10_kwh", "q90_kwh"]
    )
    print(f"  strategy C wallclock: {time.time()-t_start:.1f}s", flush=True)
    return out


# ── Strategy D: hybrid (7d C/B + 60d A) ────────────────────────────────────
def build_strategy_D(
    msns: list[str],
    df_A: pd.DataFrame,
    df_B: pd.DataFrame,
    df_C: pd.DataFrame,
    horizon_ts: pd.DatetimeIndex,
) -> pd.DataFrame:
    """First 7 days from C (sampled meters) or B (others); rest from A."""
    boundary = HYBRID_BOUNDARY
    # Indices for the two halves
    ts_idx = pd.DatetimeIndex(horizon_ts)
    early_mask_a = (df_A["ts"] >= boundary)  # use A for ts >= boundary
    a_late = df_A.loc[early_mask_a, ["meter_id", "ts", "predicted_kwh",
                                     "q10_kwh", "q90_kwh"]].copy()

    c_ids = set(df_C["meter_id"].unique()) if not df_C.empty else set()
    # Early portion (ts < boundary):
    if not df_C.empty:
        c_early = df_C[df_C["ts"] < boundary][["meter_id", "ts", "predicted_kwh",
                                               "q10_kwh", "q90_kwh"]].copy()
    else:
        c_early = pd.DataFrame(columns=["meter_id", "ts", "predicted_kwh",
                                        "q10_kwh", "q90_kwh"])
    # For meters not in C sample, use B for the early half
    b_early = df_B[(df_B["ts"] < boundary)
                   & (~df_B["meter_id"].isin(c_ids))][
        ["meter_id", "ts", "predicted_kwh", "q10_kwh", "q90_kwh"]
    ].copy()

    out = pd.concat([c_early, b_early, a_late], ignore_index=True)
    out = out.sort_values(["meter_id", "ts"]).reset_index(drop=True)
    return out


# ── Main ───────────────────────────────────────────────────────────────────
def main(
    skip_A: bool = False,
    skip_B: bool = False,
    skip_C: bool = False,
    skip_D: bool = False,
    c_budget: float = 2400.0,
) -> None:
    print(f"[gap-forecast] window: {GAP_START} -> {GAP_END} ({N_SLOTS} slots)",
          flush=True)

    horizon_ts = pd.date_range(GAP_START, GAP_END, freq="30min", inclusive="left")
    assert len(horizon_ts) == N_SLOTS, f"expected {N_SLOTS}, got {len(horizon_ts)}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(tz=timezone.utc).isoformat()

    print("[gap-forecast] loading meter + weather + fleet...", flush=True)
    all_df, _ = load_meter_data()
    wx = fetch_weather_expanded()
    # Ensure weather covers the gap (Feb 13 -> Apr 20, weather cache ends 2026-04-21)
    wx_max = wx["ts"].max()
    print(f"  weather coverage: {wx['ts'].min()} -> {wx_max} "
          f"(need {GAP_END - pd.Timedelta(minutes=30)})",
          flush=True)
    if wx_max < GAP_END - pd.Timedelta(minutes=30):
        # Extend by climatology, mirroring the existing forward-forecast behaviour
        from forward_forecast_apr_may import _extend_weather_for_forecast
        wx = _extend_weather_for_forecast(wx, GAP_END)
        print(f"  extended weather: {wx['ts'].min()} -> {wx['ts'].max()}", flush=True)
    fleet_df = compute_fleet_aggregate(all_df)

    msns = _meter_msns()
    print(f"[gap-forecast] {len(msns)} meters in v4 manifest", flush=True)

    # ── Strategy A ──
    if skip_A and OUT_A.exists():
        print("[gap-forecast] Strategy A: reusing existing parquet", flush=True)
        dfA = pd.read_parquet(OUT_A)
    else:
        print("[gap-forecast] Strategy A: seasonal anchor + real Feb-Apr weather",
              flush=True)
        dfA = build_strategy_A(msns, all_df, wx, horizon_ts)
        dfA = dfA.sort_values(["meter_id", "ts"]).reset_index(drop=True)
        dfA_w = _add_schema_cols(dfA, "seasonal_anchor", MODEL_VERSION_A,
                                 generated_at)
        dfA_w.to_parquet(OUT_A, index=False)
        print(f"  wrote {len(dfA_w):,} rows -> {OUT_A.name}", flush=True)
        dfA = dfA_w

    # ── Strategy B ──
    if skip_B and OUT_B.exists():
        print("[gap-forecast] Strategy B: reusing existing parquet", flush=True)
        dfB = pd.read_parquet(OUT_B)
    else:
        print("[gap-forecast] Strategy B: v4 LightGBM batch (anchor-seeded lags)",
              flush=True)
        dfB = build_strategy_B(msns, all_df, wx, fleet_df, horizon_ts, dfA)
        dfB = dfB.sort_values(["meter_id", "ts"]).reset_index(drop=True)
        dfB_w = _add_schema_cols(dfB, "v4_batch", MODEL_VERSION_B, generated_at)
        dfB_w.to_parquet(OUT_B, index=False)
        print(f"  wrote {len(dfB_w):,} rows -> {OUT_B.name}", flush=True)
        dfB = dfB_w

    # ── Strategy C ──
    if skip_C:
        if OUT_C.exists():
            print("[gap-forecast] Strategy C: reusing existing parquet", flush=True)
            dfC = pd.read_parquet(OUT_C)
        else:
            print("[gap-forecast] Strategy C: skipped, no parquet", flush=True)
            dfC = pd.DataFrame(columns=[
                "meter_id", "ts", "predicted_kwh", "q10_kwh", "q90_kwh",
                "strategy", "model_version", "generated_at",
            ])
    else:
        print(f"[gap-forecast] Strategy C: v5 recursive on "
              f"{len(V5_SAMPLE_METERS)} sample meters, total budget {c_budget:.0f}s",
              flush=True)
        dfC = build_strategy_C(
            V5_SAMPLE_METERS, all_df, wx, fleet_df, horizon_ts,
            total_budget_seconds=c_budget,
        )
        dfC = dfC.sort_values(["meter_id", "ts"]).reset_index(drop=True)
        if not dfC.empty:
            dfC = _add_schema_cols(dfC, "v5_recursive", MODEL_VERSION_C,
                                   generated_at)
        dfC.to_parquet(OUT_C, index=False)
        print(f"  wrote {len(dfC):,} rows -> {OUT_C.name}", flush=True)

    # ── Strategy D ──
    if skip_D and OUT_D.exists():
        print("[gap-forecast] Strategy D: reusing existing parquet", flush=True)
        dfD = pd.read_parquet(OUT_D)
    else:
        print("[gap-forecast] Strategy D: hybrid 7d-recursive + 60d-anchor",
              flush=True)
        dfD = build_strategy_D(msns, dfA, dfB, dfC, horizon_ts)
        dfD = _add_schema_cols(dfD, "hybrid", MODEL_VERSION_D, generated_at)
        dfD.to_parquet(OUT_D, index=False)
        print(f"  wrote {len(dfD):,} rows -> {OUT_D.name}", flush=True)

    # ── Combined ──
    print("[gap-forecast] combining all strategies...", flush=True)
    parts = [dfA, dfB]
    if not dfC.empty:
        parts.append(dfC)
    parts.append(dfD)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values(["strategy", "meter_id", "ts"]).reset_index(drop=True)
    combined.to_parquet(OUT_ALL, index=False)
    print(f"  wrote combined {len(combined):,} rows -> {OUT_ALL.name}", flush=True)

    # ── Summary ──
    print("\n" + "=" * 78)
    print("  GAP FORECAST SUMMARY (Feb 13 -> Apr 20, 2026)")
    print("=" * 78)
    for nm, df in (("A seasonal_anchor", dfA),
                   ("B v4_batch       ", dfB),
                   ("C v5_recursive   ", dfC),
                   ("D hybrid         ", dfD)):
        if df is None or df.empty:
            print(f"  {nm} : (empty)")
            continue
        n_meters = df["meter_id"].nunique()
        total = df["predicted_kwh"].sum()
        print(f"  {nm} : {n_meters:>2} meters, total {total:>14,.1f} kWh, "
              f"rows {len(df):,}")

    # Sanity: max disagreement vs Strategy A
    if not dfA.empty and not dfB.empty:
        a_per = dfA.groupby("meter_id")["predicted_kwh"].sum()
        b_per = dfB.groupby("meter_id")["predicted_kwh"].sum()
        diff = (b_per - a_per).abs().sort_values(ascending=False)
        if len(diff):
            top = diff.head(3)
            print("\n  Top-3 |B-A| disagreement (per meter total kWh):")
            for m, d in top.items():
                print(f"    {m}: |B-A| = {d:,.1f} kWh "
                      f"(A={a_per[m]:,.1f}, B={b_per[m]:,.1f})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--skip-A", action="store_true")
    p.add_argument("--skip-B", action="store_true")
    p.add_argument("--skip-C", action="store_true")
    p.add_argument("--skip-D", action="store_true")
    p.add_argument("--c-budget", type=float, default=2400.0,
                   help="Total wallclock budget for Strategy C (seconds)")
    ns = p.parse_args()
    main(skip_A=ns.skip_A, skip_B=ns.skip_B, skip_C=ns.skip_C,
         skip_D=ns.skip_D, c_budget=ns.c_budget)
