"""
block_accuracy_backfill
=======================

Backfill the block-wise forecast accuracy capture system on the only
out-of-distribution window with actuals: **Feb 5 -- Feb 12, 2026**
(8-day verified window, ~366 native 30-min slots per meter).

This script:

1. Loads all 42 meter actuals from ``data/raw/{sp,tp}_data.parquet``.
2. Builds a 30-min forecast for the Feb 5--12 window for each meter
   using the v4 inference path (``predict_with_context``). v4 batch
   inference with real history is fast (~1 s/meter) and the resulting
   per-step predictions reflect the model's "teacher-forced" accuracy --
   the direct match between bundle MAPE and the production-grade
   inference path. We do NOT run v5 recursive here because the wall
   budget is 5 minutes for 42 meters; v5 recursive is ~25-30 s/meter
   = ~20 minutes.
3. Pipes the (forecast, actual) frames through the new
   ``edgegrid_forecast.accuracy`` module to compute per-block MAPE at
   four cadences:
     * dam_15min   (96/day)
     * native_30min (48/day)
     * hourly       (24/day)
     * tod_4h       (6/day, APEPDCL bands)
4. Persists four artifacts to ``outputs/``:
     * ``block_accuracy_backfill.parquet`` -- long-format full table
     * ``block_accuracy_summary.csv``      -- per (meter, cadence)
     * ``block_accuracy_cohort.csv``       -- per (cohort, cadence)
     * ``block_accuracy_fleet.csv``        -- per cadence fleet aggregate
5. Prints to stdout the headline ``<5% meter count`` per cadence.
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
from edgegrid_forecast.inference.v4_predict import predict_with_context  # noqa: E402
from edgegrid_forecast.accuracy import (  # noqa: E402
    BLOCK_SPECS,
    EPSILON_KWH,
    cohort_rollup,
    fleet_mape,
    load_oracle_cohort_map,
    per_meter_block_mape,
)


# ── Configuration ──────────────────────────────────────────────────────────
HORIZON_START = pd.Timestamp("2026-02-05T00:00:00")
HORIZON_END   = pd.Timestamp("2026-02-12T15:00:00")  # last verified actual

# Wh per kWh
WH_PER_KWH = 1000.0

OUT_DIR = REPO / "outputs"
OUT_PARQUET = OUT_DIR / "block_accuracy_backfill.parquet"
OUT_SUMMARY = OUT_DIR / "block_accuracy_summary.csv"
OUT_COHORT  = OUT_DIR / "block_accuracy_cohort.csv"
OUT_FLEET   = OUT_DIR / "block_accuracy_fleet.csv"


def _meter_msns() -> list[str]:
    """Return the 42 modeled MSNs from the persisted v4 manifest."""
    manifest = json.loads((REPO / "models" / "v4" / "_manifest.json").read_text())
    return [m["msn"] for m in manifest["models"]]


def _forecast_one_meter(
    msn: str,
    all_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    fleet_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """Produce a 30-min forecast frame for the Feb 5-12 horizon for one meter.

    Uses ``predict_with_context`` (v4 batch inference). Returns a frame
    with ``ts``, ``meter_id``, ``predicted_kwh`` or None on error.
    """
    md = all_df[all_df["msn"] == msn].sort_values("ts").reset_index(drop=True)
    if md.empty:
        return None

    # Use the full meter history through the horizon end as context. v4 batch
    # inference reads real lag values at each step (teacher forcing within the
    # context window), which is what we want for accuracy capture against
    # actuals on the verified Feb 5-12 window.
    context = md[md["ts"] <= HORIZON_END].copy()
    if len(context) < 400:
        return None

    horizon_ts = pd.DatetimeIndex(
        pd.date_range(HORIZON_START, HORIZON_END, freq="30min")
    )
    horizon_ts = horizon_ts.intersection(pd.DatetimeIndex(context["ts"]))

    if len(horizon_ts) < 48:
        return None

    try:
        out = predict_with_context(
            msn,
            context,
            weather_df=weather_df,
            fleet_df=fleet_df,
            horizon_ts=horizon_ts,
        )
    except Exception as e:
        print(f"    ! predict failed for {msn}: {type(e).__name__}: {e}",
              file=sys.stderr)
        return None

    if out.empty:
        return None

    df = out.reset_index().rename(columns={"forecast_wh": "predicted_wh"})
    df = df[["ts", "predicted_wh"]].copy()
    df["meter_id"] = msn
    # Convert Wh -> kWh
    df["predicted_kwh"] = df["predicted_wh"].astype(float) / WH_PER_KWH
    return df[["ts", "meter_id", "predicted_kwh"]]


def _actuals_for_window(
    all_df: pd.DataFrame,
    msns: list[str],
) -> pd.DataFrame:
    """Slice ``all_df`` to the Feb 5-12 horizon, returning a long-format
    frame with ``ts``, ``meter_id``, ``actual_kwh``."""
    a = all_df[all_df["msn"].isin(msns)].copy()
    a = a[(a["ts"] >= HORIZON_START) & (a["ts"] <= HORIZON_END)]
    a = a[["ts", "msn", "demand_wh"]].rename(
        columns={"msn": "meter_id", "demand_wh": "actual_wh"}
    )
    a["actual_kwh"] = a["actual_wh"].astype(float) / WH_PER_KWH
    return a[["ts", "meter_id", "actual_kwh"]]


# ── Main ───────────────────────────────────────────────────────────────────
def main(max_meters: Optional[int] = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[block-accuracy] window {HORIZON_START} -> {HORIZON_END}")
    print(f"[block-accuracy] loading meter data...")
    t0 = time.time()
    all_df, _ = load_meter_data()
    print(f"  {len(all_df):,} rows; {all_df['msn'].nunique()} meters; "
          f"last_ts={all_df['ts'].max()}  ({time.time()-t0:.1f}s)")

    print(f"[block-accuracy] fetching weather (cached)...")
    t0 = time.time()
    weather_df = fetch_weather_expanded()
    print(f"  weather rows: {len(weather_df):,}  ({time.time()-t0:.1f}s)")

    print(f"[block-accuracy] computing fleet aggregate...")
    t0 = time.time()
    fleet_df = compute_fleet_aggregate(all_df)
    print(f"  fleet rows: {len(fleet_df):,}  ({time.time()-t0:.1f}s)")

    msns = _meter_msns()
    if max_meters:
        msns = msns[:max_meters]
    print(f"[block-accuracy] {len(msns)} meters")

    actual_df = _actuals_for_window(all_df, msns)
    print(f"[block-accuracy] actuals rows in window: {len(actual_df):,}")

    # ── Forecast each meter ──
    print(f"[block-accuracy] forecasting...")
    t0 = time.time()
    fc_frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for i, msn in enumerate(msns, 1):
        fc = _forecast_one_meter(msn, all_df, weather_df, fleet_df)
        if fc is None:
            failed.append(msn)
            print(f"  [{i:>2}/{len(msns)}] {msn} FAILED")
            continue
        fc_frames.append(fc)
        if i % 10 == 0 or i == len(msns):
            print(f"  [{i:>2}/{len(msns)}] forecasted "
                  f"({time.time()-t0:.1f}s elapsed)")
    forecast_df = pd.concat(fc_frames, ignore_index=True) if fc_frames else \
        pd.DataFrame(columns=["ts", "meter_id", "predicted_kwh"])
    print(f"  forecast rows: {len(forecast_df):,}  failed: {len(failed)}  "
          f"total: {time.time()-t0:.1f}s")
    if failed:
        print(f"  failed meters: {failed}")

    # ── Compute per-block accuracy at all four cadences ──
    print(f"[block-accuracy] computing per-block MAPE for all 4 cadences...")
    t0 = time.time()
    all_blocks: list[pd.DataFrame] = []
    for cadence in BLOCK_SPECS.keys():
        blocks = per_meter_block_mape(forecast_df, actual_df, cadence)
        all_blocks.append(blocks)
        print(f"  {cadence:>14s}: {len(blocks):,} blocks  "
              f"mean_mape={blocks['ape'].mean():.2f}%  "
              f"median={blocks['ape'].median():.2f}%")
    full = pd.concat(all_blocks, ignore_index=True)
    print(f"  total block rows: {len(full):,}  ({time.time()-t0:.1f}s)")

    # ── Persist long-format parquet ──
    full.to_parquet(OUT_PARQUET, index=False)
    print(f"[block-accuracy] wrote {OUT_PARQUET} ({len(full):,} rows)")

    # ── Per-(meter, cadence) summary ──
    summary = (
        full.groupby(["meter_id", "cadence"])
            .agg(
                mape=("ape", "mean"),
                median_mape=("ape", "median"),
                p95_ape=("ape", lambda s: float(np.percentile(s.to_numpy(), 95))),
                mbe_kwh=("mbe_kwh", "mean"),
                n_blocks=("ape", "size"),
                total_actual_kwh=("actual_kwh", "sum"),
                total_predicted_kwh=("predicted_kwh", "sum"),
            )
            .reset_index()
    )
    summary["under_5pct"] = summary["mape"] < 5.0
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"[block-accuracy] wrote {OUT_SUMMARY} ({len(summary)} rows)")

    # ── Cohort rollup ──
    cohort_map = load_oracle_cohort_map()
    coh = cohort_rollup(full, cohort_map)
    coh.to_csv(OUT_COHORT, index=False)
    print(f"[block-accuracy] wrote {OUT_COHORT} ({len(coh)} rows)")

    # ── Fleet rollup per cadence ──
    fleet_rows = []
    for cadence, _spec in BLOCK_SPECS.items():
        f = fleet_mape(full[full["cadence"] == cadence])
        f["cadence"] = cadence
        fleet_rows.append(f)
    fleet = pd.DataFrame(fleet_rows)[
        ["cadence", "mean_mape", "median_mape", "p25_mape", "p75_mape",
         "p95_mape", "max_mape", "mbe_kwh", "n_blocks", "n_meters"]
    ]
    fleet.to_csv(OUT_FLEET, index=False)
    print(f"[block-accuracy] wrote {OUT_FLEET} ({len(fleet)} rows)")

    # ── Headline: <5% meter count per cadence ──
    print()
    print("=" * 70)
    print("  BLOCK-ACCURACY BACKFILL · Feb 5-12, 2026 · 42 meters")
    print(f"  forecast method: v4 predict_with_context (batch teacher-forced)")
    print("=" * 70)
    print(f"{'cadence':<14s}  {'<5%':>5s}  {'mean':>7s}  {'median':>7s}  "
          f"{'p25':>7s}  {'p75':>7s}  {'p95':>7s}  {'max':>9s}  {'n_blocks':>10s}")
    for _, row in fleet.iterrows():
        cad = row["cadence"]
        n_under5 = int((summary[(summary["cadence"] == cad) &
                                (summary["mape"] < 5.0)]).shape[0])
        n_total = int((summary[summary["cadence"] == cad]).shape[0])
        print(f"{cad:<14s}  {n_under5:>2d}/{n_total:<2d}  "
              f"{row['mean_mape']:>6.2f}%  {row['median_mape']:>6.2f}%  "
              f"{row['p25_mape']:>6.2f}%  {row['p75_mape']:>6.2f}%  "
              f"{row['p95_mape']:>6.2f}%  {row['max_mape']:>8.2f}%  "
              f"{int(row['n_blocks']):>10,d}")

    # ── Worst 5 meters per cadence ──
    print()
    print("Worst 5 meters per cadence (mean MAPE):")
    for cad in BLOCK_SPECS.keys():
        sub = summary[summary["cadence"] == cad].sort_values(
            "mape", ascending=False
        ).head(5)
        worst = ", ".join(f"{r.meter_id}={r.mape:.1f}%" for r in sub.itertuples())
        print(f"  {cad:<14s}: {worst}")

    print()
    print("Cohort rollup:")
    if not coh.empty:
        print(coh.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(max_meters=n)
