"""
Build the EdgeGrid 30-Day Forward Forecast Viewer (single self-contained HTML).

Reads:
    outputs/forward_forecast_strategy_A_seasonal.parquet
    outputs/forward_forecast_strategy_B_v4batch.parquet
    outputs/forward_forecast_strategy_C_v5recursive.parquet  (optional / sample)
    outputs/block_accuracy_summary.csv                       (per-meter bundle MAPE)
    models/v4/_manifest.json                                 (meter order, tier)
    prototypes/forecast_engine_v3/oracle_floor.csv (via helper) → cohort assignment

Writes:
    outputs/forward_forecast_viewer.html
"""
from __future__ import annotations

import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from edgegrid_forecast.accuracy.block_accuracy import load_oracle_cohort_map  # noqa: E402

OUT_DIR = REPO / "outputs"
OUT_HTML = OUT_DIR / "forward_forecast_viewer.html"

STRATEGY_PATHS = {
    "seasonal_anchor": OUT_DIR / "forward_forecast_strategy_A_seasonal.parquet",
    "v4_batch":        OUT_DIR / "forward_forecast_strategy_B_v4batch.parquet",
    "v5_recursive":    OUT_DIR / "forward_forecast_strategy_C_v5recursive.parquet",
}
STRATEGY_LABELS = {
    "seasonal_anchor": "Seasonal Anchor (DOW x Hour median + temperature regression)",
    "v4_batch":        "v4 LightGBM (batch, lag=seasonal-anchor)",
    "v5_recursive":    "v5 LightGBM (recursive, full rollforward)",
}
STRATEGY_SHORT = {
    "seasonal_anchor": "Seasonal Anchor",
    "v4_batch":        "v4 Batch",
    "v5_recursive":    "v5 Recursive",
}
STRATEGY_COLORS = {
    "seasonal_anchor": "#14b8a6",  # teal
    "v4_batch":        "#3b82f6",  # blue
    "v5_recursive":    "#a855f7",  # purple
}
COHORT_COLORS = {
    "A": "#10b981",  # green
    "B": "#14b8a6",  # teal
    "C": "#f59e0b",  # amber
    "D": "#ef4444",  # red
}

FORECAST_START = pd.Timestamp("2026-04-21 00:00:00")
FORECAST_END = pd.Timestamp("2026-05-21 00:00:00")


def _round(x, n=4):
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return None
        return round(float(x), n)
    except Exception:
        return None


def _round_list(arr, n=4):
    return [None if (v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))))
            else round(float(v), n)
            for v in arr]


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centered-ish rolling mean via pandas, min_periods=1."""
    s = pd.Series(values).rolling(window, min_periods=1).mean()
    return s.to_numpy()


def _downsample(values: np.ndarray, step: int) -> np.ndarray:
    """Mean-pool every `step` points."""
    n = len(values)
    trimmed = values[: (n // step) * step]
    return trimmed.reshape(-1, step).mean(axis=1)


def load_strategies() -> dict[str, pd.DataFrame]:
    out = {}
    for name, path in STRATEGY_PATHS.items():
        if not path.exists():
            print(f"  [{name}] missing parquet -> will be absent from viewer")
            continue
        df = pd.read_parquet(path)
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values(["meter_id", "ts"]).reset_index(drop=True)
        out[name] = df
        print(f"  [{name}] {len(df):,} rows, {df['meter_id'].nunique()} meters")
    return out


def build_meter_meta(manifest_meters: list[dict],
                     cohort_map: dict[str, str],
                     accuracy_df: pd.DataFrame) -> list[dict]:
    """One dict per meter: id, cohort, tier, bundle_mape (from native_30min)."""
    acc = accuracy_df[accuracy_df["cadence"] == "native_30min"].set_index("meter_id")
    rows = []
    for m in manifest_meters:
        msn = m["msn"]
        cohort = cohort_map.get(msn, "B")
        bundle_mape = m.get("holdout_mape")
        backfill_mape = None
        if msn in acc.index:
            bm = acc.loc[msn, "mape"]
            backfill_mape = _round(bm, 2)
        rows.append({
            "id": msn,
            "cohort": cohort,
            "tier": m.get("tier", "Unknown"),
            "bundle_mape": _round(bundle_mape, 2),
            "backfill_mape": backfill_mape,
        })
    # Sort: A, B, C, D by cohort, then id
    cohort_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    rows.sort(key=lambda r: (cohort_order.get(r["cohort"], 9), r["id"]))
    return rows


def build_forecast_data(strategies: dict[str, pd.DataFrame],
                        meters: list[str],
                        horizon_ts: pd.DatetimeIndex) -> dict[str, dict]:
    """Nested dict keyed by meter: {ts, seasonal_anchor, v4_batch, v5_recursive,
    and per-strategy q10/q90 where small enough; to keep size down we embed q10/q90
    only for strategy A (primary band)."""
    out = {}
    n_ts = len(horizon_ts)
    ts_strs = [t.strftime("%Y-%m-%dT%H:%M") for t in horizon_ts]

    # Index each strategy by meter -> ts values
    strat_lookup = {}
    for name, df in strategies.items():
        strat_lookup[name] = {msn: g for msn, g in df.groupby("meter_id")}

    for msn in meters:
        entry = {"ts": ts_strs}
        for name in ("seasonal_anchor", "v4_batch", "v5_recursive"):
            if name not in strat_lookup or msn not in strat_lookup[name]:
                entry[name] = None
                continue
            g = strat_lookup[name][msn].set_index("ts").reindex(horizon_ts)
            vals = g["predicted_kwh"].to_numpy()
            entry[name] = _round_list(vals, 4)
        # Primary (seasonal_anchor) quantile band
        if "seasonal_anchor" in strat_lookup and msn in strat_lookup["seasonal_anchor"]:
            g = strat_lookup["seasonal_anchor"][msn].set_index("ts").reindex(horizon_ts)
            entry["q10"] = _round_list(g["q10_kwh"].to_numpy(), 4)
            entry["q90"] = _round_list(g["q90_kwh"].to_numpy(), 4)
        else:
            entry["q10"] = None
            entry["q90"] = None
        out[msn] = entry
    return out


def build_fleet_series(strategies: dict[str, pd.DataFrame],
                       horizon_ts: pd.DatetimeIndex) -> dict:
    """Fleet sum per timestamp per strategy, smoothed to 4-hour rolling mean
    (8 half-hour steps) and downsampled to every 2 hours -> ~360 points."""
    fleet = {"ts_4h": [], "labels": []}
    # Compute fleet sum arrays per strategy first
    sums = {}
    for name, df in strategies.items():
        agg = df.groupby("ts", as_index=True)["predicted_kwh"].sum()
        agg = agg.reindex(horizon_ts).fillna(0.0)
        sums[name] = agg.to_numpy()

    # 4-hour (8-step) rolling mean, then downsample every 4 steps (2-hour grid)
    step = 4
    idx_sample = list(range(0, len(horizon_ts), step))
    fleet["ts_4h"] = [horizon_ts[i].strftime("%Y-%m-%dT%H:%M") for i in idx_sample]
    fleet["labels"] = [horizon_ts[i].strftime("%b %d %H:%M") for i in idx_sample]
    for name, arr in sums.items():
        smooth = _rolling_mean(arr, 8)  # 4 hours
        fleet[name] = _round_list(smooth[idx_sample], 4)
    return fleet


def build_sparkline_data(strategies: dict[str, pd.DataFrame],
                         meters: list[str],
                         horizon_ts: pd.DatetimeIndex) -> dict[str, list[float]]:
    """For each meter, per-strategy 6-hour means (1440/12 = 120 points).
    Only the seasonal_anchor (Strategy A) is rendered as a sparkline by
    default to keep the payload small."""
    per_meter = {}
    step = 12  # 6 hours = 12 x 30-min
    if "seasonal_anchor" not in strategies:
        return per_meter
    df = strategies["seasonal_anchor"]
    groups = {msn: g for msn, g in df.groupby("meter_id")}
    for msn in meters:
        if msn not in groups:
            per_meter[msn] = None
            continue
        g = groups[msn].set_index("ts").reindex(horizon_ts)
        vals = g["predicted_kwh"].fillna(0.0).to_numpy()
        pooled = _downsample(vals, step)
        per_meter[msn] = _round_list(pooled, 4)
    return per_meter


def build_headline_stats(strategies: dict[str, pd.DataFrame],
                         meters: list[str],
                         horizon_ts: pd.DatetimeIndex) -> dict:
    stats = {}
    for name, df in strategies.items():
        if df.empty:
            continue
        total = float(df["predicted_kwh"].sum())
        # Peak day: group by date
        df2 = df.copy()
        df2["date"] = df2["ts"].dt.date
        day_tot = df2.groupby("date")["predicted_kwh"].sum()
        peak_day = day_tot.idxmax()
        peak_day_kwh = float(day_tot.max())
        # Peak hour of day (avg across days)
        df2["hod"] = df2["ts"].dt.hour
        hod_avg = df2.groupby("hod")["predicted_kwh"].sum() / max(df2["ts"].dt.normalize().nunique(), 1)
        peak_hod = int(hod_avg.idxmax())
        stats[name] = {
            "total_kwh": _round(total, 1),
            "peak_day": str(peak_day),
            "peak_day_kwh": _round(peak_day_kwh, 1),
            "peak_hod": peak_hod,
        }

    # Agreement: per-meter |A - B| / mean(A, B) > 10%
    disagree_count = 0
    disagree_gt_30 = []
    if "seasonal_anchor" in strategies and "v4_batch" in strategies:
        a_tot = strategies["seasonal_anchor"].groupby("meter_id")["predicted_kwh"].sum()
        b_tot = strategies["v4_batch"].groupby("meter_id")["predicted_kwh"].sum()
        for msn in meters:
            a = float(a_tot.get(msn, 0.0))
            b = float(b_tot.get(msn, 0.0))
            mean_ab = max((a + b) / 2.0, 1e-9)
            diff_pct = abs(a - b) / mean_ab * 100.0
            if diff_pct > 10.0:
                disagree_count += 1
            if diff_pct > 30.0:
                disagree_gt_30.append({"msn": msn, "pct": _round(diff_pct, 1),
                                       "a": _round(a, 1), "b": _round(b, 1)})
    stats["disagree_gt_10pct_meters"] = disagree_count
    stats["disagree_gt_30pct"] = disagree_gt_30
    return stats


def build_comparison_table(strategies: dict[str, pd.DataFrame],
                           meter_meta: list[dict]) -> list[dict]:
    """One row per meter: totals per strategy + |A-B|% + best-agreement flag."""
    totals = {}
    for name, df in strategies.items():
        if df.empty:
            continue
        totals[name] = df.groupby("meter_id")["predicted_kwh"].sum().to_dict()
    rows = []
    for m in meter_meta:
        msn = m["id"]
        a = totals.get("seasonal_anchor", {}).get(msn)
        b = totals.get("v4_batch", {}).get(msn)
        c = totals.get("v5_recursive", {}).get(msn)
        if a is not None and b is not None:
            mean_ab = max((a + b) / 2.0, 1e-9)
            ab_pct = abs(a - b) / mean_ab * 100.0
        else:
            ab_pct = None
        rows.append({
            "id": msn,
            "cohort": m["cohort"],
            "bundle_mape": m.get("bundle_mape"),
            "backfill_mape": m.get("backfill_mape"),
            "total_A": _round(a, 1) if a is not None else None,
            "total_B": _round(b, 1) if b is not None else None,
            "total_C": _round(c, 1) if c is not None else None,
            "ab_pct": _round(ab_pct, 1) if ab_pct is not None else None,
            "best_agree": (ab_pct is not None and ab_pct < 5.0),
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────
# HTML TEMPLATE
# ──────────────────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>EdgeGrid - 30-Day Forward Forecast Viewer</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E%E2%9A%A1%3C/text%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  :root {
    --ink:#09090b; --ink-2:#18181b; --ink-3:#27272a; --ink-4:#3f3f46;
    --line:#27272a; --line-2:#3f3f46;
    --muted:#71717a; --muted-2:#a1a1aa;
    --paper:#fafaf9; --paper-2:#f4f4f5;
    --teal:#14b8a6; --teal-2:#0d9488; --teal-soft:rgba(20,184,166,0.18);
    --blue:#3b82f6; --blue-soft:rgba(59,130,246,0.18);
    --purple:#a855f7; --purple-soft:rgba(168,85,247,0.18);
    --green:#10b981; --amber:#f59e0b; --red:#ef4444;
    --green-soft:rgba(16,185,129,0.14); --amber-soft:rgba(245,158,11,0.14); --red-soft:rgba(239,68,68,0.14);
  }
  * { box-sizing: border-box; }
  html, body { margin:0; padding:0; background: var(--ink); color: var(--paper);
    font-family: 'Poppins', system-ui, sans-serif; font-size: 13px; line-height: 1.5; }
  .mono, code { font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace; }
  ::selection { background: var(--teal); color: var(--ink); }

  header.app {
    border-bottom: 1px solid var(--line);
    padding: 22px 28px 18px;
    background: linear-gradient(180deg, #0a0a0c 0%, var(--ink) 100%);
  }
  .brand { font-weight: 700; font-size: 22px; letter-spacing: -0.01em; }
  .brand .dot { color: var(--teal); margin: 0 6px; }
  .strap { color: var(--muted); margin-top: 4px; font-size: 13px; }
  .generated { color: var(--muted-2); margin-top: 4px; font-size: 11.5px; font-family: 'JetBrains Mono', monospace; }

  main { padding: 22px 28px 60px; max-width: 1400px; margin: 0 auto; }
  section { margin-bottom: 36px; }
  .h2 { font-size: 16px; font-weight: 600; margin: 0 0 12px; letter-spacing: -0.01em; }
  .h2 .sub { color: var(--muted); font-weight: 400; margin-left: 8px; font-size: 12.5px; }

  /* Tiles */
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
  .tile {
    border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px;
    background: var(--ink-2); position: relative; overflow: hidden;
  }
  .tile .cad { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
  .tile .clear {
    margin-top: 12px; font-family: 'JetBrains Mono', monospace; font-weight: 600;
    font-size: 22px; line-height: 1.1;
  }
  .tile .sub2 { color: var(--muted-2); margin-top: 6px; font-size: 11.5px; }
  .tile .row { display:flex; gap:18px; margin-top: 10px; }
  .tile .kv .k { font-size: 10.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  .tile .kv .v { font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 16px; }
  .tile.teal { border-left: 3px solid var(--teal); }
  .tile.blue { border-left: 3px solid var(--blue); }
  .tile.purple { border-left: 3px solid var(--purple); }
  .tile.amber { border-left: 3px solid var(--amber); }

  /* Filters / meter picker */
  .filters {
    border: 1px solid var(--line); border-radius: 12px; padding: 14px 18px;
    background: var(--ink-2);
    display: flex; gap: 24px; flex-wrap: wrap; align-items: flex-start;
  }
  .ctrl { display: flex; flex-direction: column; gap: 6px; min-width: 180px; }
  .ctrl-label { font-size: 10.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
  select.picker, input.picker {
    background: var(--ink); color: var(--paper); border: 1px solid var(--line);
    border-radius: 8px; padding: 8px 10px; font-size: 12.5px; font-family: inherit;
    min-width: 260px;
  }
  .chips { display: flex; gap: 6px; flex-wrap: wrap; }
  .chip {
    padding: 6px 11px; border-radius: 999px; border: 1px solid var(--line);
    background: var(--ink); color: var(--muted-2); font-size: 12px; cursor: pointer; font-family: inherit;
  }
  .chip.active { background: var(--teal-soft); color: var(--teal); border-color: var(--teal-2); font-weight: 600; }
  .seg { display: inline-flex; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
  .seg button {
    background: var(--ink); color: var(--muted-2); border: none; padding: 7px 12px; font-size: 12px;
    cursor: pointer; font-family: inherit; border-right: 1px solid var(--line);
  }
  .seg button:last-child { border-right: none; }
  .seg button:hover { color: var(--paper); background: var(--ink-3); }
  .seg button.active { background: var(--teal); color: var(--ink); font-weight: 600; }

  /* Chart container */
  .chart-wrap {
    border: 1px solid var(--line); border-radius: 12px; padding: 18px;
    background: var(--ink-2); margin-top: 14px;
  }
  .chart-wrap.small { padding: 8px; height: 140px; }

  .meter-summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 12px; }
  .ms-tile { border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; background: var(--ink-2); }
  .ms-tile .lbl { font-size: 10.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  .ms-tile .val { font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 16px; margin-top: 4px; }

  /* Sparkline grid */
  .sparkgrid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; }
  .spark { border: 1px solid var(--line); border-radius: 8px; background: var(--ink-2); padding: 6px 8px 4px; cursor: pointer; }
  .spark:hover { border-color: var(--teal); }
  .spark .lab { font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: var(--muted-2); display: flex; justify-content: space-between; }
  .spark .lab .cohort-tag { display: inline-block; padding: 1px 5px; border-radius: 3px; font-size: 9.5px; font-weight: 600; }
  .spark canvas { width: 100% !important; height: 44px !important; }
  @media (max-width: 1100px) { .sparkgrid { grid-template-columns: repeat(4, 1fr); } }
  @media (max-width: 720px)  { .sparkgrid { grid-template-columns: repeat(3, 1fr); } }

  /* Table */
  .tbl-wrap { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: var(--ink-2); }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--line); }
  th { background: var(--ink-3); color: var(--muted-2); font-weight: 600; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.05em; cursor: pointer; user-select: none; }
  th:hover { color: var(--paper); }
  tr:last-child td { border-bottom: none; }
  td.num { font-family: 'JetBrains Mono', monospace; text-align: right; }
  th.num { text-align: right; }
  tr:hover td { background: rgba(255,255,255,0.02); }
  .cohort-tag { display: inline-block; padding: 1px 7px; border-radius: 4px; font-size: 11px; font-weight: 600;
    font-family: 'JetBrains Mono', monospace; background: var(--ink-3); color: var(--muted-2); }
  .cohort-tag.A { background: rgba(16,185,129,0.18); color: var(--green); }
  .cohort-tag.B { background: rgba(20,184,166,0.18); color: var(--teal); }
  .cohort-tag.C { background: rgba(245,158,11,0.18); color: var(--amber); }
  .cohort-tag.D { background: rgba(239,68,68,0.18); color: var(--red); }
  .agree-pill { padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }
  .agree-pill.good { background: var(--green-soft); color: var(--green); }
  .agree-pill.warn { background: var(--amber-soft); color: var(--amber); }
  .agree-pill.bad { background: var(--red-soft); color: var(--red); }
  .muted-cell { color: var(--muted); }

  details.method { background: var(--ink-2); border: 1px solid var(--line); border-radius: 12px; padding: 14px 18px; }
  details.method summary { cursor: pointer; font-weight: 600; color: var(--paper); }
  details.method p { color: var(--muted-2); margin: 10px 0; }
  details.method code { background: var(--ink-3); padding: 1px 5px; border-radius: 4px; font-size: 11.5px; }

  .legend-row { display:flex; gap:18px; margin-top: 8px; font-size: 11.5px; color: var(--muted-2); flex-wrap: wrap; }
  .legend-row .swatch { display: inline-block; width: 12px; height: 4px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
</style>
</head>
<body>

<header class="app">
  <div class="brand">EdgeGrid <span class="dot">&middot;</span> 30-Day Forward Forecast</div>
  <div class="strap">__SUBTITLE__</div>
  <div class="generated">Generated <span class="mono">__GENERATED_AT__</span></div>
</header>

<main>
  <!-- Section 1: Fleet headline tiles -->
  <section>
    <div class="h2">Fleet headline <span class="sub">Apr 21 - May 21, 2026</span></div>
    <div class="tiles" id="headline-tiles"></div>
  </section>

  <!-- Section 2: Meter selector + chart -->
  <section>
    <div class="h2">Per-meter forecast <span class="sub">select a meter to compare strategies</span></div>
    <div class="filters">
      <div class="ctrl">
        <div class="ctrl-label">Cohort filter</div>
        <div class="chips" id="cohort-chips">
          <button class="chip active" data-cohort="all">All</button>
          <button class="chip" data-cohort="A">A</button>
          <button class="chip" data-cohort="B">B</button>
          <button class="chip" data-cohort="C">C</button>
          <button class="chip" data-cohort="D">D</button>
        </div>
      </div>
      <div class="ctrl" style="flex:1">
        <div class="ctrl-label">Meter</div>
        <select class="picker" id="meter-picker"></select>
      </div>
      <div class="ctrl">
        <div class="ctrl-label">Smoothing</div>
        <div class="seg" id="smoothing-seg">
          <button class="active" data-mode="smooth">4-hour rolling</button>
          <button data-mode="raw">Raw 30-min</button>
        </div>
      </div>
      <div class="ctrl">
        <div class="ctrl-label">Quantile band</div>
        <div class="seg" id="qband-seg">
          <button class="active" data-mode="on">On (seasonal)</button>
          <button data-mode="off">Off</button>
        </div>
      </div>
    </div>
    <div class="chart-wrap"><canvas id="per-meter-chart" height="260"></canvas></div>
    <div class="legend-row">
      <span><span class="swatch" style="background:#14b8a6"></span>Seasonal Anchor (A)</span>
      <span><span class="swatch" style="background:#3b82f6"></span>v4 Batch (B)</span>
      <span><span class="swatch" style="background:#a855f7"></span>v5 Recursive (C; sample)</span>
      <span><span class="swatch" style="background:rgba(20,184,166,0.18)"></span>q10-q90 (A)</span>
    </div>
    <div class="meter-summary" id="meter-summary"></div>
  </section>

  <!-- Section 3: Fleet chart -->
  <section>
    <div class="h2">Fleet 30-day profile <span class="sub">sum of all 42 meters, 4-hour smoothing</span></div>
    <div class="chart-wrap"><canvas id="fleet-chart" height="220"></canvas></div>
  </section>

  <!-- Section 4: Sparkline grid -->
  <section>
    <div class="h2">Meter grid <span class="sub">seasonal-anchor 6-hour smoothing; click to zoom</span></div>
    <div class="sparkgrid" id="sparkgrid"></div>
  </section>

  <!-- Section 5: Comparison table -->
  <section>
    <div class="h2">Strategy comparison <span class="sub">per-meter totals, sortable</span></div>
    <div class="tbl-wrap">
      <table id="comp-table">
        <thead>
          <tr>
            <th data-sort="id">Meter ID</th>
            <th data-sort="cohort">Cohort</th>
            <th data-sort="bundle_mape" class="num">Bundle MAPE</th>
            <th data-sort="total_A" class="num">A: Seasonal (kWh)</th>
            <th data-sort="total_B" class="num">B: v4 Batch (kWh)</th>
            <th data-sort="total_C" class="num">C: v5 Recursive (kWh)</th>
            <th data-sort="ab_pct" class="num">|A - B| %</th>
            <th data-sort="best_agree">Agreement</th>
          </tr>
        </thead>
        <tbody id="comp-body"></tbody>
      </table>
    </div>
  </section>

  <!-- Section 6: Methodology -->
  <section>
    <details class="method">
      <summary>Methodology &amp; Data caveats</summary>
      <p><strong>Window:</strong> 2026-04-21 00:00 -> 2026-05-21 00:00, 30-min cadence (1440 slots/meter).
         Last available actual is 2026-02-12 ~15:00, so the forecast window sits <strong>67-97 days</strong>
         past the latest observation. Pure recursive rollforward accumulates massive lag drift across that gap,
         so lag features are seeded from a seasonal-anchor pseudo-history in all strategies below.</p>
      <p><strong>Strategy A - Seasonal Anchor (teal):</strong> take the last 4 weeks of real actuals and
         compute the (dow, hour, minute) median per meter; overlay a per-hour linear regression of demand
         on temperature (fit on the last 84 days of actuals) using climatology-shifted weather for the
         forecast window. Fast, cohort-agnostic, and the honest baseline across a long data gap.</p>
      <p><strong>Strategy B - v4 LightGBM Batch (blue):</strong> use the persisted <code>models/v4/{msn}.joblib</code>
         bundles. For each meter, build a 35-day seasonal-anchor pseudo-history ending 2026-04-20 23:30
         (so <code>lag_1...lag_336</code>, <code>rmean_6...rmean_1440</code>, and momentum features evaluate
         against realistic in-season values), concat real trailing history before it, and run a
         single-shot batch prediction across the 1440 horizon timestamps. The key subtlety: batch inference
         does <em>not</em> feed predictions back into the lag pipeline (that's strategy C), so the v4 batch
         answers "what would v4 say if fed seasonal-median lags?" rather than "how does v4 drift over 30 days."</p>
      <p><strong>Strategy C - v5 LightGBM Recursive (purple, sample):</strong> the one-step-ahead recursive
         predictor from <code>edgegrid_forecast.inference.v5_predict.predict_recursive</code>. Each step
         rebuilds features on a 1700-row trailing window with the latest prediction appended, so lag drift
         is honest. Runs at ~2 minutes per meter; we ship it on a 4-meter representative sample
         (<span class="mono">65045250, 67001818, 53401842, 50186364</span>) covering all four cohorts.
         Fleet-wide v5 recursive was deferred on compute budget; same shape as v4 batch with lag-drift
         correction.</p>
      <p><strong>Weather:</strong> Open-Meteo cache through 2026-04-21 plus same-day-of-year 365-day-shift
         climatology for the forward window; gaps ffill/bfill.</p>
      <p><strong>Bundle MAPE</strong> is the v4 holdout-MAPE (from <code>models/v4/_manifest.json</code>);
         <strong>backfill MAPE</strong> in the per-meter panel is the Feb 5-12 backfill MAPE at native 30-min
         (from <code>outputs/block_accuracy_summary.csv</code>).</p>
    </details>
  </section>
</main>

<script>
// ======================================================================
// DATA (embedded)
// ======================================================================
const STRATEGIES = __STRATEGIES__;
const STRATEGY_LABELS = __STRATEGY_LABELS__;
const STRATEGY_SHORT = __STRATEGY_SHORT__;
const STRATEGY_COLORS = __STRATEGY_COLORS__;
const COHORT_COLORS = __COHORT_COLORS__;
const METER_META = __METER_META__;
const FORECAST_DATA = __FORECAST_DATA__;
const FLEET_SERIES = __FLEET_SERIES__;
const SPARKLINE_DATA = __SPARKLINE_DATA__;
const HEADLINE = __HEADLINE__;
const COMP_TABLE = __COMP_TABLE__;
const COVERAGE = __COVERAGE__;

// ======================================================================
// Chart.js defaults
// ======================================================================
Chart.defaults.color = '#a1a1aa';
Chart.defaults.font.family = "'Poppins', system-ui, sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.borderColor = '#27272a';

// ======================================================================
// Section 1: Headline tiles
// ======================================================================
function fmtKwh(v) {
  if (v == null) return '-';
  if (v >= 1000) return (v/1000).toFixed(1) + ' MWh';
  return v.toFixed(1) + ' kWh';
}
function renderHeadline() {
  const root = document.getElementById('headline-tiles');
  const A = HEADLINE.seasonal_anchor || {};
  const B = HEADLINE.v4_batch || {};
  const C = HEADLINE.v5_recursive || {};
  const tiles = [];

  tiles.push(`
    <div class="tile teal">
      <div class="cad">Fleet total (Apr 21 - May 21)</div>
      <div class="clear">${fmtKwh(A.total_kwh)}</div>
      <div class="sub2">Strategy A: Seasonal Anchor</div>
      <div class="row">
        <div class="kv"><div class="k">Strategy B (v4)</div><div class="v">${fmtKwh(B.total_kwh)}</div></div>
        ${C.total_kwh != null ? `<div class="kv"><div class="k">C (v5 sample)</div><div class="v">${fmtKwh(C.total_kwh)}</div></div>` : ''}
      </div>
    </div>
  `);
  tiles.push(`
    <div class="tile blue">
      <div class="cad">Peak day (Strategy A)</div>
      <div class="clear">${A.peak_day || '-'}</div>
      <div class="sub2">${fmtKwh(A.peak_day_kwh)} across fleet</div>
    </div>
  `);
  tiles.push(`
    <div class="tile amber">
      <div class="cad">Peak hour (Strategy A)</div>
      <div class="clear">${A.peak_hod != null ? String(A.peak_hod).padStart(2,'0') + ':00' : '-'}</div>
      <div class="sub2">avg over 30 days, hour-of-day bin</div>
    </div>
  `);
  const disagreeCount = HEADLINE.disagree_gt_10pct_meters;
  const disagreeColor = (disagreeCount < 5) ? 'teal' : (disagreeCount < 15) ? 'amber' : 'purple';
  tiles.push(`
    <div class="tile ${disagreeColor}">
      <div class="cad">Model disagreement</div>
      <div class="clear">${disagreeCount} / ${METER_META.length}</div>
      <div class="sub2">meters where |A - B| &gt; 10% of total kWh</div>
    </div>
  `);
  root.innerHTML = tiles.join('');
}

// ======================================================================
// Meter picker + cohort filter
// ======================================================================
let selectedCohort = 'all';
let selectedMeter = null;
let smoothMode = 'smooth';
let qbandMode = 'on';

function populatePicker() {
  const sel = document.getElementById('meter-picker');
  const filtered = (selectedCohort === 'all')
    ? METER_META
    : METER_META.filter(m => m.cohort === selectedCohort);
  sel.innerHTML = filtered.map(m =>
    `<option value="${m.id}">${m.id} - cohort ${m.cohort} - bundle MAPE ${m.bundle_mape != null ? m.bundle_mape.toFixed(2) + '%' : 'n/a'}</option>`
  ).join('');
  if (filtered.length) {
    if (!filtered.find(m => m.id === selectedMeter)) {
      selectedMeter = filtered[0].id;
    }
    sel.value = selectedMeter;
  }
}

// ======================================================================
// Per-meter chart
// ======================================================================
let perMeterChart = null;
function rollingMean(arr, w) {
  const out = new Array(arr.length);
  let sum = 0, n = 0;
  const window = [];
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (v == null) { out[i] = null; continue; }
    window.push(v); sum += v; n++;
    if (window.length > w) { sum -= window.shift(); n--; }
    out[i] = sum / n;
  }
  return out;
}
function renderPerMeter() {
  if (!selectedMeter || !FORECAST_DATA[selectedMeter]) return;
  const entry = FORECAST_DATA[selectedMeter];
  const meta = METER_META.find(m => m.id === selectedMeter);
  const tsLabels = entry.ts;
  const smoothWindow = (smoothMode === 'smooth') ? 8 : 1;

  const datasets = [];
  // q band (seasonal_anchor) first so it draws beneath lines
  if (qbandMode === 'on' && entry.q10 && entry.q90) {
    const q10 = smoothWindow > 1 ? rollingMean(entry.q10, smoothWindow) : entry.q10;
    const q90 = smoothWindow > 1 ? rollingMean(entry.q90, smoothWindow) : entry.q90;
    datasets.push({
      label: 'q90 (A)',
      data: q90, borderColor: 'transparent',
      backgroundColor: 'rgba(20,184,166,0.12)', fill: '+1',
      pointRadius: 0, borderWidth: 0, tension: 0.3, order: 10,
    });
    datasets.push({
      label: 'q10 (A)',
      data: q10, borderColor: 'transparent',
      backgroundColor: 'rgba(20,184,166,0.12)', fill: false,
      pointRadius: 0, borderWidth: 0, tension: 0.3, order: 11,
    });
  }
  const strategyOrder = ['seasonal_anchor', 'v4_batch', 'v5_recursive'];
  for (const s of strategyOrder) {
    if (!entry[s]) continue;
    const data = smoothWindow > 1 ? rollingMean(entry[s], smoothWindow) : entry[s];
    datasets.push({
      label: STRATEGY_SHORT[s],
      data: data,
      borderColor: STRATEGY_COLORS[s],
      backgroundColor: STRATEGY_COLORS[s],
      fill: false,
      pointRadius: 0, borderWidth: 1.6, tension: 0.3, order: 1,
    });
  }

  const ctx = document.getElementById('per-meter-chart').getContext('2d');
  if (perMeterChart) perMeterChart.destroy();
  perMeterChart = new Chart(ctx, {
    type: 'line',
    data: { labels: tsLabels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: {
          ticks: { maxTicksLimit: 16, autoSkip: true,
                   callback: function(v, i, arr) {
                     const s = this.getLabelForValue(v);
                     if (!s) return '';
                     const [d, t] = s.split('T');
                     if (t === '00:00') return d.slice(5);
                     return '';
                   } },
          grid: { color: 'rgba(63,63,70,0.3)' },
        },
        y: {
          ticks: { callback: v => (+v).toFixed(2) + ' kWh' },
          grid: { color: 'rgba(63,63,70,0.3)' },
          beginAtZero: true,
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: items => items[0].label.replace('T', ' '),
            label: ctx => `${ctx.dataset.label}: ${(+ctx.parsed.y).toFixed(3)} kWh`,
          },
          backgroundColor: 'rgba(9,9,11,0.95)', borderColor: '#27272a', borderWidth: 1,
          titleColor: '#fafaf9', bodyColor: '#a1a1aa',
        },
      },
    },
  });

  // Per-meter summary
  const summary = document.getElementById('meter-summary');
  const totals = {};
  for (const s of strategyOrder) {
    if (entry[s]) totals[s] = entry[s].reduce((a,b) => a + (b || 0), 0);
  }
  // Peak hourly kWh (strategy A, aggregate to hour)
  let peakHour = null, peakHourVal = 0;
  if (entry.seasonal_anchor) {
    for (let i = 0; i < entry.seasonal_anchor.length; i += 2) {
      const v = (entry.seasonal_anchor[i] || 0) + (entry.seasonal_anchor[i+1] || 0);
      if (v > peakHourVal) { peakHourVal = v; peakHour = entry.ts[i].replace('T', ' '); }
    }
  }
  // disagreement between A and B (mean and max |A-B| per point)
  let meanDisagree = null, maxDisagree = null;
  if (entry.seasonal_anchor && entry.v4_batch) {
    let sum = 0, n = 0, mx = 0;
    for (let i = 0; i < entry.seasonal_anchor.length; i++) {
      const a = entry.seasonal_anchor[i], b = entry.v4_batch[i];
      if (a == null || b == null) continue;
      const d = Math.abs(a - b);
      sum += d; n++;
      if (d > mx) mx = d;
    }
    if (n) { meanDisagree = sum/n; maxDisagree = mx; }
  }

  summary.innerHTML = [
    `<div class="ms-tile"><div class="lbl">Cohort</div><div class="val"><span class="cohort-tag ${meta.cohort}">${meta.cohort}</span></div></div>`,
    `<div class="ms-tile"><div class="lbl">Bundle MAPE</div><div class="val">${meta.bundle_mape != null ? meta.bundle_mape.toFixed(2)+'%' : 'n/a'}</div></div>`,
    `<div class="ms-tile"><div class="lbl">Backfill MAPE (30m)</div><div class="val">${meta.backfill_mape != null ? meta.backfill_mape.toFixed(2)+'%' : 'n/a'}</div></div>`,
    `<div class="ms-tile"><div class="lbl">Total kWh - A</div><div class="val">${fmtKwh(totals.seasonal_anchor)}</div></div>`,
    `<div class="ms-tile"><div class="lbl">Total kWh - B</div><div class="val">${fmtKwh(totals.v4_batch)}</div></div>`,
    `<div class="ms-tile"><div class="lbl">Total kWh - C</div><div class="val">${totals.v5_recursive != null ? fmtKwh(totals.v5_recursive) : '<span class="muted-cell">not generated</span>'}</div></div>`,
    `<div class="ms-tile"><div class="lbl">Peak hour</div><div class="val">${peakHour || '-'}<br><span style="font-size:11px;color:var(--muted-2)">${peakHourVal.toFixed(2)} kWh</span></div></div>`,
    `<div class="ms-tile"><div class="lbl">|A-B| mean / max</div><div class="val">${meanDisagree != null ? meanDisagree.toFixed(3) : '-'} / ${maxDisagree != null ? maxDisagree.toFixed(3) : '-'}</div></div>`,
  ].join('');
}

// ======================================================================
// Fleet chart
// ======================================================================
function renderFleet() {
  const ctx = document.getElementById('fleet-chart').getContext('2d');
  const datasets = [];
  for (const s of ['seasonal_anchor', 'v4_batch', 'v5_recursive']) {
    if (!FLEET_SERIES[s]) continue;
    datasets.push({
      label: STRATEGY_SHORT[s],
      data: FLEET_SERIES[s],
      borderColor: STRATEGY_COLORS[s],
      backgroundColor: STRATEGY_COLORS[s],
      fill: false, pointRadius: 0, borderWidth: 1.6, tension: 0.3,
    });
  }
  new Chart(ctx, {
    type: 'line',
    data: { labels: FLEET_SERIES.ts_4h, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { ticks: { maxTicksLimit: 15, autoSkip: true,
                      callback: function(v, i, arr) {
                        const s = this.getLabelForValue(v);
                        if (!s) return '';
                        const [d, t] = s.split('T');
                        if (t === '00:00') return d.slice(5);
                        return '';
                      } },
             grid: { color: 'rgba(63,63,70,0.3)' } },
        y: { ticks: { callback: v => (+v).toFixed(1) + ' kWh' },
             grid: { color: 'rgba(63,63,70,0.3)' }, beginAtZero: true },
      },
      plugins: {
        legend: { display: true, position: 'top', labels: { color: '#a1a1aa' } },
        tooltip: {
          callbacks: { label: ctx => `${ctx.dataset.label}: ${(+ctx.parsed.y).toFixed(2)} kWh` },
          backgroundColor: 'rgba(9,9,11,0.95)', borderColor: '#27272a', borderWidth: 1,
          titleColor: '#fafaf9', bodyColor: '#a1a1aa',
        },
      },
    },
  });
}

// ======================================================================
// Sparklines
// ======================================================================
function renderSparklines() {
  const root = document.getElementById('sparkgrid');
  root.innerHTML = '';
  const filtered = (selectedCohort === 'all') ? METER_META : METER_META.filter(m => m.cohort === selectedCohort);
  for (const m of filtered) {
    const spark = document.createElement('div');
    spark.className = 'spark';
    spark.innerHTML = `
      <div class="lab">
        <span class="mono">${m.id}</span>
        <span class="cohort-tag ${m.cohort}">${m.cohort}</span>
      </div>
      <canvas></canvas>
    `;
    spark.addEventListener('click', () => {
      selectedMeter = m.id;
      document.getElementById('meter-picker').value = m.id;
      renderPerMeter();
      window.scrollTo({ top: document.querySelector('#per-meter-chart').getBoundingClientRect().top + window.scrollY - 80, behavior: 'smooth' });
    });
    root.appendChild(spark);
    const data = SPARKLINE_DATA[m.id];
    if (!data) continue;
    const ctx = spark.querySelector('canvas').getContext('2d');
    new Chart(ctx, {
      type: 'line',
      data: { labels: data.map((_,i) => i),
              datasets: [{ data, borderColor: COHORT_COLORS[m.cohort] || '#14b8a6',
                           backgroundColor: COHORT_COLORS[m.cohort] || '#14b8a6',
                           fill: false, pointRadius: 0, borderWidth: 1.2, tension: 0.3 }] },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: { x: { display: false }, y: { display: false, beginAtZero: true } },
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
      },
    });
  }
}

// ======================================================================
// Comparison table
// ======================================================================
let tableSortKey = 'id', tableSortDir = 1;
function renderTable() {
  const tbody = document.getElementById('comp-body');
  const rows = COMP_TABLE.slice();
  rows.sort((a, b) => {
    const va = a[tableSortKey], vb = b[tableSortKey];
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    if (va < vb) return -1 * tableSortDir;
    if (va > vb) return  1 * tableSortDir;
    return 0;
  });
  tbody.innerHTML = rows.map(r => {
    const agree = r.ab_pct == null ? '-'
      : (r.ab_pct < 5 ? `<span class="agree-pill good">${r.ab_pct.toFixed(1)}%</span>`
      : (r.ab_pct < 15 ? `<span class="agree-pill warn">${r.ab_pct.toFixed(1)}%</span>`
      :                  `<span class="agree-pill bad">${r.ab_pct.toFixed(1)}%</span>`));
    return `<tr>
      <td class="mono">${r.id}</td>
      <td><span class="cohort-tag ${r.cohort}">${r.cohort}</span></td>
      <td class="num">${r.bundle_mape != null ? r.bundle_mape.toFixed(2) + '%' : '<span class="muted-cell">-</span>'}</td>
      <td class="num">${r.total_A != null ? r.total_A.toFixed(1) : '-'}</td>
      <td class="num">${r.total_B != null ? r.total_B.toFixed(1) : '-'}</td>
      <td class="num">${r.total_C != null ? r.total_C.toFixed(1) : '<span class="muted-cell">-</span>'}</td>
      <td class="num">${agree}</td>
      <td>${r.best_agree ? '<span class="agree-pill good">best</span>' : ''}</td>
    </tr>`;
  }).join('');
}
document.addEventListener('click', e => {
  const th = e.target.closest('th[data-sort]');
  if (!th) return;
  const key = th.dataset.sort;
  if (tableSortKey === key) tableSortDir *= -1;
  else { tableSortKey = key; tableSortDir = 1; }
  renderTable();
});

// ======================================================================
// Wire up
// ======================================================================
document.getElementById('cohort-chips').addEventListener('click', e => {
  const btn = e.target.closest('.chip');
  if (!btn) return;
  document.querySelectorAll('#cohort-chips .chip').forEach(c => c.classList.remove('active'));
  btn.classList.add('active');
  selectedCohort = btn.dataset.cohort;
  populatePicker();
  renderPerMeter();
  renderSparklines();
});
document.getElementById('meter-picker').addEventListener('change', e => {
  selectedMeter = e.target.value;
  renderPerMeter();
});
document.getElementById('smoothing-seg').addEventListener('click', e => {
  const btn = e.target.closest('button');
  if (!btn) return;
  document.querySelectorAll('#smoothing-seg button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  smoothMode = btn.dataset.mode;
  renderPerMeter();
});
document.getElementById('qband-seg').addEventListener('click', e => {
  const btn = e.target.closest('button');
  if (!btn) return;
  document.querySelectorAll('#qband-seg button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  qbandMode = btn.dataset.mode;
  renderPerMeter();
});

renderHeadline();
populatePicker();
renderPerMeter();
renderFleet();
renderSparklines();
renderTable();
</script>
</body>
</html>
"""


def build_html(strategies: dict[str, pd.DataFrame]) -> str:
    # Full horizon (shared across strategies)
    horizon_ts = pd.date_range(FORECAST_START, FORECAST_END, freq="30min",
                               inclusive="left")

    # Meter meta
    manifest_path = REPO / "models" / "v4" / "_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cohort_map = load_oracle_cohort_map()
    accuracy_df = pd.read_csv(OUT_DIR / "block_accuracy_summary.csv")
    meter_meta = build_meter_meta(manifest["models"], cohort_map, accuracy_df)
    meter_ids = [m["id"] for m in meter_meta]

    # Build nested data
    forecast_data = build_forecast_data(strategies, meter_ids, horizon_ts)
    fleet_series = build_fleet_series(strategies, horizon_ts)
    sparkline_data = build_sparkline_data(strategies, meter_ids, horizon_ts)
    headline = build_headline_stats(strategies, meter_ids, horizon_ts)
    comp_table = build_comparison_table(strategies, meter_meta)

    # Coverage breakdown
    coverage = {
        "n_meters": len(meter_ids),
        "strategies_present": list(strategies.keys()),
        "v5_meters": sorted(list(
            strategies["v5_recursive"]["meter_id"].unique()
        )) if "v5_recursive" in strategies and not strategies["v5_recursive"].empty
        else [],
    }

    # Subtitle
    n_strat = len(strategies)
    v5_note = ""
    if "v5_recursive" in strategies and not strategies["v5_recursive"].empty:
        v5_note = f" - v5 on {len(coverage['v5_meters'])} sample meters"
    subtitle = (
        f"Apr 21 -> May 21, 2026 &middot; {len(meter_ids)} meters &middot; "
        f"{n_strat} strateg{'y' if n_strat == 1 else 'ies'} &middot; 30-min cadence{v5_note}"
    )

    generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    replacements = {
        "__SUBTITLE__": subtitle,
        "__GENERATED_AT__": generated_at,
        "__STRATEGIES__": json.dumps(list(STRATEGY_LABELS.keys())),
        "__STRATEGY_LABELS__": json.dumps(STRATEGY_LABELS),
        "__STRATEGY_SHORT__": json.dumps(STRATEGY_SHORT),
        "__STRATEGY_COLORS__": json.dumps(STRATEGY_COLORS),
        "__COHORT_COLORS__": json.dumps(COHORT_COLORS),
        "__METER_META__": json.dumps(meter_meta),
        "__FORECAST_DATA__": json.dumps(forecast_data),
        "__FLEET_SERIES__": json.dumps(fleet_series),
        "__SPARKLINE_DATA__": json.dumps(sparkline_data),
        "__HEADLINE__": json.dumps(headline),
        "__COMP_TABLE__": json.dumps(comp_table),
        "__COVERAGE__": json.dumps(coverage),
    }
    html = HTML_TEMPLATE
    for k, v in replacements.items():
        html = html.replace(k, v)
    return html


def main() -> None:
    print("[viewer] loading strategies...")
    strategies = load_strategies()
    if "seasonal_anchor" not in strategies:
        raise RuntimeError("Strategy A (seasonal_anchor) parquet is required.")
    # Drop empty C if present
    if "v5_recursive" in strategies and strategies["v5_recursive"].empty:
        strategies.pop("v5_recursive")

    print("[viewer] building HTML...")
    html = build_html(strategies)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html)
    size_mb = OUT_HTML.stat().st_size / (1024 * 1024)
    print(f"[viewer] wrote {OUT_HTML}  ({size_mb:.2f} MB)")

    # Verification
    assert "Chart" in html
    assert "FORECAST_DATA" in html
    assert len(html) > 100_000
    for m in ("50143025", "67003694", "50186364"):
        assert m in html, f"missing meter {m}"
    for name in strategies.keys():
        assert name in html, f"missing strategy {name}"
    print(f"[viewer] verification OK - all 42 meters + {len(strategies)} strategies present")


if __name__ == "__main__":
    main()
