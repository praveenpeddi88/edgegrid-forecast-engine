"""
Phase 4: Continuous-history viewer builder.

Produces outputs/continuous_history_viewer.html — a single self-contained file
showing for each of 42 meters a continuous timeline of:
    actuals -> gap forecast -> forward forecast
with the 4 model strategies overlaid.

Inputs:
  data/raw/sp_data.parquet
  data/raw/tp_data.parquet
  outputs/full_forecast_feb13_may21.parquet
  prototypes/forecast_engine_v3/oracle_floor.csv

Output:
  outputs/continuous_history_viewer.html
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SP_PATH = ROOT / "data" / "raw" / "sp_data.parquet"
TP_PATH = ROOT / "data" / "raw" / "tp_data.parquet"
FORECAST_PATH = ROOT / "outputs" / "full_forecast_feb13_may21.parquet"
ORACLE_PATH = ROOT / "prototypes" / "forecast_engine_v3" / "oracle_floor.csv"
OUT_PATH = ROOT / "outputs" / "continuous_history_viewer.html"

# Strategy mapping: parquet string -> JSON key
STRATEGY_KEYS = {
    "seasonal_anchor": "seasonal",
    "v4_batch": "v4",
    "v5_recursive": "v5",
    "hybrid": "hybrid",
}

# History window (final ~140 days)
HISTORY_START = pd.Timestamp("2026-01-01 00:00:00")
HISTORY_END = pd.Timestamp("2026-05-20 23:00:00")

# Cutoff timestamps (hour-resolution, treated as UTC-naive)
ACTUALS_END = pd.Timestamp("2026-02-12 23:00:00")
FORECAST_START = pd.Timestamp("2026-02-13 00:00:00")
GAP_END = pd.Timestamp("2026-04-20 23:00:00")
FORWARD_START = pd.Timestamp("2026-04-21 00:00:00")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def cohort_for_bundle(mape: float) -> str:
    if mape < 5:
        return "A"
    if mape < 15:
        return "B"
    if mape < 25:
        return "C"
    return "D"


def round_or_none(x: float) -> float | None:
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return round(float(x), 3)


def to_iso(ts: pd.Timestamp) -> str:
    # Hour resolution UTC-marked iso string
    return ts.strftime("%Y-%m-%dT%H:00:00Z")


# --------------------------------------------------------------------------------------
# Data loading & shaping
# --------------------------------------------------------------------------------------
def load_actuals(meter_universe: set[str]) -> pd.DataFrame:
    sp = pd.read_parquet(SP_PATH, columns=["msn", "ts", "wh_imp"])
    tp = pd.read_parquet(TP_PATH, columns=["msn", "ts", "wh_imp"])
    df = pd.concat([sp, tp], ignore_index=True)
    df = df[df["msn"].isin(meter_universe)].copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df["kwh"] = df["wh_imp"].astype(float) / 1000.0
    df = df[["msn", "ts", "kwh"]]
    return df


def load_forecasts() -> pd.DataFrame:
    fc = pd.read_parquet(
        FORECAST_PATH,
        columns=["meter_id", "ts", "predicted_kwh", "strategy"],
    )
    fc["ts"] = pd.to_datetime(fc["ts"])
    return fc


def load_meter_meta(meter_universe: set[str]) -> pd.DataFrame:
    orc = pd.read_csv(ORACLE_PATH, dtype={"msn": str})
    orc = orc[orc["msn"].isin(meter_universe)].copy()
    orc["cohort"] = orc["bundle_mape"].apply(cohort_for_bundle)
    return orc[["msn", "bundle_mape", "cohort"]]


def aggregate_to_hourly(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Sum two 30-min slots per hour to produce hourly kWh."""
    df = df.copy()
    df["hour"] = df["ts"].dt.floor("h")
    out = df.groupby([df.columns[0], "hour"], as_index=False)[value_col].sum(min_count=1)
    out = out.rename(columns={out.columns[0]: df.columns[0], "hour": "ts"})
    return out


def build_hour_index() -> pd.DatetimeIndex:
    return pd.date_range(start=HISTORY_START, end=HISTORY_END, freq="h")


# --------------------------------------------------------------------------------------
# Build per-meter series
# --------------------------------------------------------------------------------------
def build_payload() -> dict:
    fc = load_forecasts()
    meter_universe = set(fc["meter_id"].astype(str).unique())
    print(f"Forecast meter universe: {len(meter_universe)}")

    meta = load_meter_meta(meter_universe)
    print(f"Meta rows: {len(meta)}")

    # Phase: SP if msn starts with 5 prefix that's in sp_data, else TP. Quick lookup.
    sp_meters = set(pd.read_parquet(SP_PATH, columns=["msn"])["msn"].unique())

    actuals = load_actuals(meter_universe)
    actuals_h = aggregate_to_hourly(actuals.rename(columns={"msn": "meter_id"}), "kwh")

    # Hourly forecasts per strategy
    fc["meter_id"] = fc["meter_id"].astype(str)
    fc_h_parts = []
    for strat in fc["strategy"].unique():
        sub = fc[fc["strategy"] == strat][["meter_id", "ts", "predicted_kwh"]].copy()
        sub_h = aggregate_to_hourly(sub, "predicted_kwh")
        sub_h["strategy"] = strat
        fc_h_parts.append(sub_h)
    fc_h = pd.concat(fc_h_parts, ignore_index=True)

    hour_index = build_hour_index()
    hours_iso = [to_iso(t) for t in hour_index]

    series: dict[str, dict] = {}
    meters_list: list[dict] = []

    # Sort meters by cohort then meter_id
    meta_sorted = meta.sort_values(["cohort", "msn"]).reset_index(drop=True)

    for _, row in meta_sorted.iterrows():
        mid = row["msn"]
        cohort = row["cohort"]
        bundle = float(row["bundle_mape"])
        phase = "SP" if mid in sp_meters else "TP"
        meters_list.append(
            {
                "id": mid,
                "cohort": cohort,
                "bundle_mape": round(bundle, 2),
                "phase": phase,
            }
        )

        # Actual array on the hourly grid
        act_sub = actuals_h[actuals_h["meter_id"] == mid].set_index("ts")["kwh"]
        act_sub = act_sub.reindex(hour_index)
        actual_arr = [round_or_none(v) for v in act_sub.values]

        # Per-strategy arrays
        strat_arrays: dict[str, list] = {}
        for parquet_strat, json_key in STRATEGY_KEYS.items():
            ssub = fc_h[(fc_h["meter_id"] == mid) & (fc_h["strategy"] == parquet_strat)]
            if ssub.empty:
                strat_arrays[json_key] = [None] * len(hour_index)
                continue
            s = ssub.set_index("ts")["predicted_kwh"].reindex(hour_index)
            strat_arrays[json_key] = [round_or_none(v) for v in s.values]

        series[mid] = {
            "hours_count": len(hour_index),  # diagnostic
            "actual": actual_arr,
            "seasonal": strat_arrays["seasonal"],
            "v4": strat_arrays["v4"],
            "v5": strat_arrays["v5"],
            "hybrid": strat_arrays["hybrid"],
        }

    payload = {
        "meters": meters_list,
        "hours": hours_iso,  # shared grid (all meters identical)
        "series": series,
        "cutoffs": {
            "actuals_end": to_iso(ACTUALS_END),
            "forecast_start": to_iso(FORECAST_START),
            "gap_end": to_iso(GAP_END),
            "forward_start": to_iso(FORWARD_START),
        },
        "meta": {
            "history_start": to_iso(HISTORY_START),
            "history_end": to_iso(HISTORY_END),
        },
    }
    return payload


# --------------------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>EdgeGrid - Continuous History Viewer</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
:root {
  --bg: #ffffff;
  --ink: #1a1f2c;
  --ink-soft: #4a5468;
  --line: #e6e9ef;
  --accent: #2563eb;
  --A: #2563eb;
  --B: #ea7a1a;
  --C: #8b5cf6;
  --D: #14b8a6;
  --actual: #1a1f2c;
  --shade-actuals: #f5f5f7;
  --shade-gap: #fff8e1;
  --shade-forward: #e8f7ee;
}
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0; background: var(--bg); color: var(--ink);
  font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-weight: 400;
  -webkit-font-smoothing: antialiased;
}
.container { max-width: 1400px; margin: 0 auto; padding: 24px 32px 64px; }
header.top {
  display: flex; align-items: baseline; justify-content: space-between;
  margin-bottom: 16px; gap: 16px; flex-wrap: wrap;
}
h1 { font-size: 22px; font-weight: 600; margin: 0; letter-spacing: -0.01em; }
.subtitle { font-size: 13px; color: var(--ink-soft); }

.controls {
  position: sticky; top: 0; z-index: 10;
  background: rgba(255,255,255,0.97);
  backdrop-filter: blur(8px);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 16px 18px;
  margin-bottom: 20px;
  display: grid;
  grid-template-columns: 1.4fr 1fr;
  gap: 14px 24px;
}
.control-row { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
.control-row label.tag {
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em;
  color: var(--ink-soft); font-weight: 600; margin-right: 4px;
}
select {
  font-family: inherit; font-size: 13px; padding: 7px 10px;
  border: 1px solid var(--line); border-radius: 8px; background: white;
  min-width: 280px;
}
.pill {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 12px; padding: 5px 11px; border-radius: 999px;
  border: 1px solid var(--line); background: white; cursor: pointer;
  font-weight: 500; color: var(--ink-soft);
  transition: all 0.12s ease;
}
.pill:hover { border-color: #c5cad6; }
.pill.active {
  background: var(--ink); color: white; border-color: var(--ink);
}
.btn {
  font-family: inherit; font-size: 12px; padding: 6px 12px;
  border-radius: 8px; border: 1px solid var(--line); background: white;
  cursor: pointer; color: var(--ink); font-weight: 500;
  transition: all 0.12s ease;
}
.btn:hover { background: #f6f7fa; }
.btn.active { background: var(--ink); color: white; border-color: var(--ink); }

.strat-toggle {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 12px; padding: 5px 10px;
  border-radius: 8px; border: 1px solid var(--line); cursor: pointer;
  user-select: none; font-weight: 500;
}
.strat-toggle input { accent-color: var(--ink); }
.strat-toggle .swatch {
  width: 10px; height: 10px; border-radius: 2px;
}
.strat-toggle.A .swatch { background: var(--A); }
.strat-toggle.B .swatch { background: var(--B); }
.strat-toggle.C .swatch { background: var(--C); }
.strat-toggle.D .swatch { background: var(--D); }

.chart-card {
  border: 1px solid var(--line); border-radius: 14px;
  padding: 22px 24px 18px;
  background: white;
  box-shadow: 0 1px 2px rgba(20,24,40,0.03);
}
.chart-meta {
  display: flex; gap: 22px; margin-bottom: 12px; flex-wrap: wrap;
  font-size: 12px; color: var(--ink-soft);
}
.chart-meta strong { color: var(--ink); font-weight: 600; }
.chart-wrap {
  position: relative; height: 460px;
}

.coverage-note {
  margin-top: 12px; font-size: 12px; padding: 8px 12px;
  background: #fff8e1; border: 1px solid #f0d985; border-radius: 8px;
  color: #66510a;
  display: none;
}
.coverage-note.show { display: block; }

.stats {
  margin-top: 22px;
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
}
.stat-card {
  border: 1px solid var(--line); border-radius: 10px;
  padding: 14px 16px; background: white;
}
.stat-card .strat-label {
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
  font-weight: 600; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;
}
.stat-card .strat-label .swatch {
  width: 10px; height: 10px; border-radius: 2px;
}
.stat-card.A .swatch { background: var(--A); }
.stat-card.B .swatch { background: var(--B); }
.stat-card.C .swatch { background: var(--C); }
.stat-card.D .swatch { background: var(--D); }
.stat-card .row {
  display: flex; justify-content: space-between; align-items: baseline;
  font-size: 12px; color: var(--ink-soft); margin-top: 4px;
}
.stat-card .row .v { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }
.stat-card.muted { opacity: 0.55; }

.legend-region {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 11px; color: var(--ink-soft); padding: 3px 8px; border-radius: 6px;
}
.legend-region .sw { width: 10px; height: 10px; border-radius: 2px; }
.legend-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 6px; }

footer {
  margin-top: 32px; font-size: 11px; color: var(--ink-soft);
  text-align: center; letter-spacing: 0.02em;
}
</style>
</head>
<body>
<div class="container">

<header class="top">
  <div>
    <h1>EdgeGrid - Continuous history viewer</h1>
    <div class="subtitle">Actuals -> gap forecast -> forward forecast on a single timeline. 42 meters x 4 strategies.</div>
  </div>
  <div class="legend-row">
    <span class="legend-region"><span class="sw" style="background:var(--shade-actuals)"></span>Actuals</span>
    <span class="legend-region"><span class="sw" style="background:var(--shade-gap)"></span>Gap forecast (Feb 13 - Apr 20)</span>
    <span class="legend-region"><span class="sw" style="background:var(--shade-forward)"></span>Forward forecast (Apr 21 - May 20)</span>
  </div>
</header>

<div class="controls">
  <div class="control-row">
    <label class="tag">Cohort</label>
    <button class="pill active" data-cohort="ALL">ALL</button>
    <button class="pill" data-cohort="A">A</button>
    <button class="pill" data-cohort="B">B</button>
    <button class="pill" data-cohort="C">C</button>
    <button class="pill" data-cohort="D">D</button>
    <label class="tag" style="margin-left:14px">Meter</label>
    <select id="meterSelect"></select>
  </div>

  <div class="control-row">
    <label class="tag">Strategies</label>
    <label class="strat-toggle A"><input type="checkbox" data-strat="seasonal" checked /><span class="swatch"></span>A Seasonal</label>
    <label class="strat-toggle B"><input type="checkbox" data-strat="v4" /><span class="swatch"></span>B v4 batch</label>
    <label class="strat-toggle C"><input type="checkbox" data-strat="v5" /><span class="swatch"></span>C v5 recursive</label>
    <label class="strat-toggle D"><input type="checkbox" data-strat="hybrid" checked /><span class="swatch"></span>D Hybrid</label>
  </div>

  <div class="control-row">
    <label class="tag">Smoothing</label>
    <button class="btn active" data-smooth="0">None</button>
    <button class="btn" data-smooth="4">4-hour rolling</button>
    <button class="btn" data-smooth="24">24-hour rolling</button>
  </div>

  <div class="control-row">
    <label class="tag">Range</label>
    <button class="btn active" data-range="full">Full view (Jan -> May)</button>
    <button class="btn" data-range="gap">Gap only (Feb 13 - Apr 20)</button>
    <button class="btn" data-range="last14">Last 14 days</button>
    <button class="btn" data-range="forward">Forward only (Apr 21 - May 20)</button>
  </div>
</div>

<div class="chart-card">
  <div class="chart-meta" id="chartMeta"></div>
  <div class="chart-wrap"><canvas id="mainChart"></canvas></div>
  <div class="coverage-note" id="coverageNote"></div>
  <div class="stats" id="statsPanel"></div>
</div>

<footer>EdgeGrid forecast engine - Phase 4 viewer - generated __GENERATED_AT__</footer>
</div>

<script type="application/json" id="viewer-data">__JSON_PAYLOAD__</script>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById('viewer-data').textContent);
  const HOURS = DATA.hours;
  const HOUR_TS = HOURS.map(h => Date.parse(h));
  const CUTOFFS = {
    actuals_end: Date.parse(DATA.cutoffs.actuals_end),
    forecast_start: Date.parse(DATA.cutoffs.forecast_start),
    gap_end: Date.parse(DATA.cutoffs.gap_end),
    forward_start: Date.parse(DATA.cutoffs.forward_start),
  };
  const STRAT_DEFS = [
    { key: 'seasonal', label: 'A Seasonal', cohort: 'A', color: getCss('--A') },
    { key: 'v4',       label: 'B v4 batch', cohort: 'B', color: getCss('--B') },
    { key: 'v5',       label: 'C v5 recursive', cohort: 'C', color: getCss('--C') },
    { key: 'hybrid',   label: 'D Hybrid', cohort: 'D', color: getCss('--D') },
  ];
  function getCss(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';
  }

  // ---- State ----
  const state = {
    cohort: 'ALL',
    meterId: null,
    enabled: { seasonal: true, v4: false, v5: false, hybrid: true },
    smooth: 0,
    range: 'full',
  };

  // ---- DOM ----
  const meterSelect = document.getElementById('meterSelect');
  const chartMeta = document.getElementById('chartMeta');
  const coverageNote = document.getElementById('coverageNote');
  const statsPanel = document.getElementById('statsPanel');
  const ctx = document.getElementById('mainChart').getContext('2d');

  // ---- Background-shading plugin ----
  const shadingPlugin = {
    id: 'regionShading',
    beforeDraw(chart) {
      const { ctx, chartArea, scales } = chart;
      if (!chartArea || !scales.x) return;
      const x = scales.x;
      const left = chartArea.left, right = chartArea.right;
      const top = chartArea.top, bottom = chartArea.bottom;
      const xMin = x.min, xMax = x.max;
      const clamp = v => Math.max(xMin, Math.min(xMax, v));
      const px = v => x.getPixelForValue(clamp(v));

      // Actuals region: xMin .. forecast_start
      drawRect(ctx, px(xMin), top, px(CUTOFFS.forecast_start), bottom, 'rgba(245,245,247,0.7)');
      // Gap region: forecast_start .. forward_start
      drawRect(ctx, px(CUTOFFS.forecast_start), top, px(CUTOFFS.forward_start), bottom, 'rgba(255,248,225,0.55)');
      // Forward region: forward_start .. xMax
      drawRect(ctx, px(CUTOFFS.forward_start), top, px(xMax), bottom, 'rgba(232,247,238,0.55)');

      // Vertical separators
      drawVLine(ctx, px(CUTOFFS.forecast_start), top, bottom, 'rgba(0,0,0,0.18)');
      drawVLine(ctx, px(CUTOFFS.forward_start), top, bottom, 'rgba(0,0,0,0.18)');

      // Region labels (top-left of each band)
      ctx.save();
      ctx.font = '500 10.5px Poppins, system-ui, sans-serif';
      ctx.fillStyle = 'rgba(50,55,70,0.7)';
      ctx.textBaseline = 'top';
      drawLabel(ctx, 'Actuals', (px(xMin) + px(CUTOFFS.forecast_start)) / 2, top + 4);
      drawLabel(ctx, 'Gap forecast (67 days, no actuals)', (px(CUTOFFS.forecast_start) + px(CUTOFFS.forward_start)) / 2, top + 4);
      drawLabel(ctx, 'Forward forecast (30 days)', (px(CUTOFFS.forward_start) + px(xMax)) / 2, top + 4);
      ctx.restore();

      function drawRect(ctx, x1, y1, x2, y2, fill) {
        if (x2 <= x1) return;
        ctx.save(); ctx.fillStyle = fill; ctx.fillRect(x1, y1, x2 - x1, y2 - y1); ctx.restore();
      }
      function drawVLine(ctx, x, y1, y2, stroke) {
        ctx.save(); ctx.strokeStyle = stroke; ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, y1); ctx.lineTo(x, y2); ctx.stroke(); ctx.restore();
      }
      function drawLabel(ctx, text, cx, cy) {
        const w = ctx.measureText(text).width;
        ctx.save();
        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.fillRect(cx - w/2 - 4, cy - 1, w + 8, 14);
        ctx.fillStyle = 'rgba(40,46,60,0.85)';
        ctx.fillText(text, cx - w/2, cy);
        ctx.restore();
      }
    }
  };

  // ---- Chart instance ----
  const chart = new Chart(ctx, {
    type: 'line',
    data: { datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      parsing: false,
      normalized: true,
      spanGaps: false,
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth: 12, boxHeight: 12, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            title: (items) => items.length ? new Date(items[0].parsed.x).toUTCString().slice(0, 22) : '',
            label: (item) => `${item.dataset.label}: ${item.parsed.y == null ? '-' : item.parsed.y.toFixed(3) + ' kWh'}`
          }
        }
      },
      scales: {
        x: {
          type: 'time',
          time: { tooltipFormat: 'PP HH:mm', displayFormats: { hour: 'MMM d', day: 'MMM d', week: 'MMM d' } },
          grid: { color: 'rgba(0,0,0,0.04)' },
          ticks: { font: { size: 11 } },
        },
        y: {
          title: { display: true, text: 'kWh / hour', font: { size: 11, weight: '500' }, color: '#4a5468' },
          grid: { color: 'rgba(0,0,0,0.05)' },
          ticks: { font: { size: 11 } },
          beginAtZero: true,
        }
      }
    },
    plugins: [shadingPlugin]
  });

  // ---- Helpers ----
  function rolling(arr, win) {
    if (!win || win <= 1) return arr.slice();
    const out = new Array(arr.length).fill(null);
    let sum = 0, cnt = 0;
    const q = [];
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      q.push(v);
      if (v != null) { sum += v; cnt++; }
      if (q.length > win) {
        const r = q.shift();
        if (r != null) { sum -= r; cnt--; }
      }
      out[i] = (cnt > 0) ? sum / cnt : null;
    }
    return out;
  }

  function arrToPoints(values) {
    const pts = new Array(values.length);
    for (let i = 0; i < values.length; i++) {
      pts[i] = { x: HOUR_TS[i], y: values[i] };
    }
    return pts;
  }

  function strategyHasAny(arr) {
    for (let i = 0; i < arr.length; i++) if (arr[i] != null) return true;
    return false;
  }

  function rangeBounds() {
    switch (state.range) {
      case 'gap': return [CUTOFFS.forecast_start, CUTOFFS.forward_start - 1];
      case 'forward': return [CUTOFFS.forward_start, HOUR_TS[HOUR_TS.length - 1]];
      case 'last14': {
        const end = HOUR_TS[HOUR_TS.length - 1];
        return [end - 14 * 24 * 3600 * 1000, end];
      }
      default: return [HOUR_TS[0], HOUR_TS[HOUR_TS.length - 1]];
    }
  }

  // ---- Render ----
  function render() {
    const meter = DATA.meters.find(m => m.id === state.meterId);
    if (!meter) return;
    const ser = DATA.series[meter.id];

    // Datasets
    const datasets = [];
    // Actual
    const actualVals = rolling(ser.actual, state.smooth);
    datasets.push({
      label: 'Actual',
      data: arrToPoints(actualVals),
      borderColor: getCss('--actual'),
      backgroundColor: getCss('--actual'),
      borderWidth: 2.0,
      pointRadius: 0,
      pointHoverRadius: 3,
      tension: 0.15,
      order: 1,
      spanGaps: false,
    });

    let visibleStrats = [];
    STRAT_DEFS.forEach(def => {
      if (!state.enabled[def.key]) return;
      const raw = ser[def.key];
      const present = strategyHasAny(raw);
      if (!present) return;  // don't add empty line
      visibleStrats.push(def.key);
      const vals = rolling(raw, state.smooth);
      datasets.push({
        label: def.label,
        data: arrToPoints(vals),
        borderColor: def.color,
        backgroundColor: def.color,
        borderWidth: 1.5,
        pointRadius: 0,
        pointHoverRadius: 3,
        tension: 0.15,
        order: 2,
        spanGaps: false,
      });
    });

    chart.data.datasets = datasets;
    const [xmin, xmax] = rangeBounds();
    chart.options.scales.x.min = xmin;
    chart.options.scales.x.max = xmax;
    chart.update('none');

    // Meta line
    chartMeta.innerHTML = `
      <span><strong>Meter</strong> ${meter.id}</span>
      <span><strong>Cohort</strong> ${meter.cohort}</span>
      <span><strong>Bundle MAPE</strong> ${meter.bundle_mape}%</span>
      <span><strong>Phase</strong> ${meter.phase}</span>
      <span><strong>Hours embedded</strong> ${HOURS.length}</span>
    `;

    // Coverage note
    const v5Has = strategyHasAny(ser.v5);
    if (!v5Has && state.enabled.v5) {
      coverageNote.classList.add('show');
      coverageNote.textContent = `Note: v5 recursive is not available for meter ${meter.id} (insufficient compute). Only 4 meters carry a v5 backtest.`;
    } else if (v5Has && state.enabled.v5) {
      // partial coverage info
      const total = ser.v5.length;
      let cnt = 0; for (let i = 0; i < total; i++) if (ser.v5[i] != null) cnt++;
      const pct = ((cnt / total) * 100).toFixed(0);
      if (cnt < total) {
        coverageNote.classList.add('show');
        coverageNote.textContent = `v5 recursive coverage on this meter: ${cnt} / ${total} hours (${pct}%).`;
      } else {
        coverageNote.classList.remove('show');
        coverageNote.textContent = '';
      }
    } else {
      coverageNote.classList.remove('show');
      coverageNote.textContent = '';
    }

    // Stats
    renderStats(ser);
  }

  // 30-day totals: gap (Feb 13 - Mar 14, 30 days) and forward (Apr 21 - May 20)
  function renderStats(ser) {
    const gapStart = CUTOFFS.forecast_start;
    const gapEnd30 = gapStart + 30 * 24 * 3600 * 1000;  // first 30 days of gap
    const fwdStart = CUTOFFS.forward_start;
    const fwdEnd30 = fwdStart + 30 * 24 * 3600 * 1000;

    function sumWindow(arr, t1, t2) {
      let s = 0, has = false;
      for (let i = 0; i < HOUR_TS.length; i++) {
        const t = HOUR_TS[i];
        if (t < t1 || t >= t2) continue;
        const v = arr[i];
        if (v != null) { s += v; has = true; }
      }
      return has ? s : null;
    }

    // Fleet-relative deviation: each strategy's gap sum vs fleet mean of strategy gap sums
    const gapSums = {};
    const fwdSums = {};
    STRAT_DEFS.forEach(def => {
      gapSums[def.key] = sumWindow(ser[def.key], gapStart, gapEnd30);
      fwdSums[def.key] = sumWindow(ser[def.key], fwdStart, fwdEnd30);
    });

    const enabledKeys = STRAT_DEFS.filter(d => state.enabled[d.key]).map(d => d.key);
    const refVals = enabledKeys.map(k => gapSums[k]).filter(v => v != null);
    const ref = refVals.length ? refVals.reduce((a, b) => a + b, 0) / refVals.length : null;

    let html = '';
    STRAT_DEFS.forEach(def => {
      const gap = gapSums[def.key];
      const fwd = fwdSums[def.key];
      const enabled = state.enabled[def.key];
      const muted = (!enabled || gap == null) ? 'muted' : '';
      const dev = (gap != null && ref != null && ref > 0)
        ? ((gap - ref) / ref * 100).toFixed(1) + '%'
        : '-';
      const fmt = v => v == null ? '-' : v.toLocaleString(undefined, { maximumFractionDigits: 0 }) + ' kWh';
      html += `
        <div class="stat-card ${def.cohort} ${muted}">
          <div class="strat-label"><span class="swatch"></span>${def.label}</div>
          <div class="row"><span>30d gap sum</span><span class="v">${fmt(gap)}</span></div>
          <div class="row"><span>30d forward sum</span><span class="v">${fmt(fwd)}</span></div>
          <div class="row"><span>Fleet-rel. dev (gap)</span><span class="v">${dev}</span></div>
        </div>`;
    });
    statsPanel.innerHTML = html;
  }

  // ---- UI wiring ----
  function rebuildMeterDropdown() {
    const filtered = DATA.meters.filter(m => state.cohort === 'ALL' || m.cohort === state.cohort);
    meterSelect.innerHTML = '';
    filtered.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = `${m.id}  -  ${m.cohort}  -  ${m.bundle_mape.toFixed(2)}%`;
      meterSelect.appendChild(opt);
    });
    if (filtered.length) {
      const stillThere = filtered.find(m => m.id === state.meterId);
      state.meterId = stillThere ? stillThere.id : filtered[0].id;
      meterSelect.value = state.meterId;
    }
  }

  // cohort pills
  document.querySelectorAll('.pill[data-cohort]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.pill[data-cohort]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.cohort = btn.dataset.cohort;
      rebuildMeterDropdown();
      render();
    });
  });

  meterSelect.addEventListener('change', () => {
    state.meterId = meterSelect.value;
    render();
  });

  document.querySelectorAll('.strat-toggle input').forEach(inp => {
    inp.addEventListener('change', () => {
      state.enabled[inp.dataset.strat] = inp.checked;
      render();
    });
  });

  document.querySelectorAll('.btn[data-smooth]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.btn[data-smooth]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.smooth = parseInt(btn.dataset.smooth, 10);
      render();
    });
  });

  document.querySelectorAll('.btn[data-range]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.btn[data-range]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.range = btn.dataset.range;
      render();
    });
  });

  // init
  rebuildMeterDropdown();
  render();
})();
</script>
</body>
</html>
"""


def render_html(payload: dict) -> str:
    json_str = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    generated_at = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return HTML_TEMPLATE.replace("__JSON_PAYLOAD__", json_str).replace(
        "__GENERATED_AT__", generated_at
    )


# --------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------
def main() -> None:
    print("Building continuous history viewer ...")
    payload = build_payload()
    html = render_html(payload)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")

    # ---- Reporting ----
    file_size = OUT_PATH.stat().st_size
    print()
    print("================ BUILD REPORT ================")
    print(f"HTML file: {OUT_PATH}")
    print(f"HTML file size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    n_meters = len(payload["meters"])
    print(f"Meters embedded: {n_meters}")
    hours_per_meter = len(payload["hours"])
    print(f"Hours per meter: {hours_per_meter}")

    # Sample meter id per cohort
    print()
    print("Sample meter id per cohort:")
    seen = set()
    for m in payload["meters"]:
        if m["cohort"] in seen:
            continue
        print(f"  cohort {m['cohort']}: {m['id']} (bundle_mape={m['bundle_mape']}%)")
        seen.add(m["cohort"])

    # Sanity check: meter 65002231
    print()
    print("Sanity check on meter 65002231 ...")
    target_id = "65002231"
    series = payload["series"].get(target_id)
    if series is None:
        print(f"  WARNING: {target_id} not present in payload")
    else:
        meta = next(m for m in payload["meters"] if m["id"] == target_id)
        assert meta["cohort"] == "A", f"expected cohort A, got {meta['cohort']}"

        hours = payload["hours"]
        feb13_idx = hours.index("2026-02-13T00:00:00Z")
        # Check actual non-null before Feb 13, null after
        before = series["actual"][:feb13_idx]
        after = series["actual"][feb13_idx:]
        non_null_before = sum(1 for v in before if v is not None)
        non_null_after = sum(1 for v in after if v is not None)
        assert non_null_before > 0, "expected non-null actuals before Feb 13"
        assert non_null_after == 0, f"expected null actuals after Feb 13, got {non_null_after} non-null"
        print(f"  actuals: {non_null_before} non-null before Feb 13, {non_null_after} after  OK")

        # Gap region indexes
        gap_end_idx = hours.index("2026-04-20T23:00:00Z")
        for k in ("seasonal", "v4", "v5", "hybrid"):
            arr = series[k]
            gap_slice = arr[feb13_idx : gap_end_idx + 1]
            cnt = sum(1 for v in gap_slice if v is not None)
            assert cnt > 0, f"expected non-null {k} values in gap region for {target_id}, got 0"
            print(f"  {k}: {cnt} non-null hours in gap region  OK")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
