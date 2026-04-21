"""Build the live block-accuracy dashboard.

Reads:
  outputs/block_accuracy_summary.csv      (per meter + cadence)
  outputs/block_accuracy_cohort.csv       (per cohort + cadence)
  outputs/block_accuracy_fleet.csv        (per cadence fleet)
  outputs/forward_forecast_30d.parquet    (Apr 21 - May 21)
  prototypes/forecast_engine_v3/oracle_floor.csv  (cohort assignment via bundle_mape)

Writes:
  outputs/block_accuracy_dashboard.html   (single self-contained file)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"
ORACLE_CSV = REPO / "prototypes" / "forecast_engine_v3" / "oracle_floor.csv"

CADENCE_LABEL = {
    "dam_15min": "15-min DAM",
    "native_30min": "30-min Native",
    "hourly": "Hourly",
    "tod_4h": "ToD 4-hour",
}
CADENCE_ORDER = ["dam_15min", "native_30min", "hourly", "tod_4h"]
COHORTS = ["A", "B", "C", "D"]


def assign_cohort(b: float) -> str:
    if b < 5.0:
        return "A"
    if b < 15.0:
        return "B"
    if b < 25.0:
        return "C"
    return "D"


def main() -> None:
    summary = pd.read_csv(OUT / "block_accuracy_summary.csv")
    cohort_df = pd.read_csv(OUT / "block_accuracy_cohort.csv")
    fleet = pd.read_csv(OUT / "block_accuracy_fleet.csv")
    oracle = pd.read_csv(ORACLE_CSV)

    # Build cohort map from oracle bundle_mape
    oracle["msn"] = oracle["msn"].astype(str)
    cohort_map = {row.msn: assign_cohort(float(row.bundle_mape)) for row in oracle.itertuples()}

    # Attach cohort to summary
    summary["meter_id"] = summary["meter_id"].astype(str)
    summary["cohort"] = summary["meter_id"].map(cohort_map).fillna("D")

    # Per-cadence fleet tile data
    fleet_tiles = []
    for cad in CADENCE_ORDER:
        f = fleet[fleet["cadence"] == cad].iloc[0]
        sub = summary[summary["cadence"] == cad]
        clear_count = int((sub["mape"] < 5.0).sum())
        fleet_tiles.append({
            "cadence": cad,
            "label": CADENCE_LABEL[cad],
            "mean_mape": float(f["mean_mape"]),
            "median_mape": float(f["median_mape"]),
            "clear_count": clear_count,
            "total_meters": int(sub["meter_id"].nunique()),
        })

    # Per-meter rows (one per meter+cadence)
    meter_rows = []
    for r in summary.itertuples():
        meter_rows.append({
            "meter_id": r.meter_id,
            "cohort": r.cohort,
            "cadence": r.cadence,
            "mape": round(float(r.mape), 3),
            "median_mape": round(float(r.median_mape), 3),
            "p95_ape": round(float(r.p95_ape), 3),
            "mbe_kwh": round(float(r.mbe_kwh), 4),
            "total_actual_kwh": round(float(r.total_actual_kwh), 3),
            "total_predicted_kwh": round(float(r.total_predicted_kwh), 3),
            "under_5pct": bool(r.under_5pct),
        })

    # Cohort rollup rows for chart
    cohort_rows = []
    for r in cohort_df.itertuples():
        cohort_rows.append({
            "cohort": r.cohort,
            "cadence": r.cadence,
            "mean_mape": round(float(r.mean_mape), 3),
            "median_mape": round(float(r.median_mape), 3),
            "n_meters": int(r.n_meters),
        })

    # Forward forecast: per-meter total predicted kWh
    fwd = pd.read_parquet(OUT / "forward_forecast_30d.parquet")
    fwd["meter_id"] = fwd["meter_id"].astype(str)
    fwd_summary = (
        fwd.groupby("meter_id", as_index=False)["predicted_kwh"]
        .sum()
        .rename(columns={"predicted_kwh": "total_predicted_kwh"})
    )
    fwd_summary["cohort"] = fwd_summary["meter_id"].map(cohort_map).fillna("D")
    fwd_summary["total_predicted_kwh"] = fwd_summary["total_predicted_kwh"].round(2)
    forward_rows = fwd_summary.sort_values("meter_id").to_dict(orient="records")

    # Date range for forward forecast
    fwd_start = pd.to_datetime(fwd["ts"]).min().strftime("%Y-%m-%d")
    fwd_end = pd.to_datetime(fwd["ts"]).max().strftime("%Y-%m-%d")

    js_data = {
        "fleet": fleet_tiles,
        "meters": meter_rows,
        "cohorts": cohort_rows,
        "forward": forward_rows,
        "forward_start": fwd_start,
        "forward_end": fwd_end,
        "cadence_label": CADENCE_LABEL,
        "cadence_order": CADENCE_ORDER,
    }

    html = build_html(js_data)
    out_path = OUT / "block_accuracy_dashboard.html"
    out_path.write_text(html, encoding="utf-8")

    size = out_path.stat().st_size
    print(f"Wrote: {out_path}")
    print(f"Size: {size:,} bytes ({size/1024:.1f} KB)")
    print(
        f"Embedded rows: fleet={len(fleet_tiles)} meters={len(meter_rows)} "
        f"cohorts={len(cohort_rows)} forward={len(forward_rows)}"
    )


def build_html(d: dict) -> str:
    payload = json.dumps(d, separators=(",", ":"))
    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
<title>EdgeGrid - Block-Wise Forecast Accuracy</title>
<link rel=\"icon\" href=\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E%F0%9F%93%8A%3C/text%3E%3C/svg%3E\">
<meta property=\"og:title\" content=\"EdgeGrid - Block-Wise Forecast Accuracy\">
<meta property=\"og:description\" content=\"Per-meter per-cadence MAPE, cohort rollup, and 30-day forward forecast for the EdgeGrid 42-meter fleet.\">
<meta property=\"og:type\" content=\"website\">
<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
<link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
<link href=\"https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap\" rel=\"stylesheet\">
<script src=\"https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js\"></script>
<style>
  :root {{
    --ink:#09090b; --ink-2:#18181b; --ink-3:#27272a; --ink-4:#3f3f46;
    --line:#27272a; --line-2:#3f3f46;
    --muted:#71717a; --muted-2:#a1a1aa;
    --paper:#fafaf9; --paper-2:#f4f4f5;
    --teal:#14b8a6; --teal-2:#0d9488; --teal-soft:rgba(20,184,166,0.18);
    --green:#10b981; --amber:#f59e0b; --red:#ef4444;
    --green-soft:rgba(16,185,129,0.14); --amber-soft:rgba(245,158,11,0.14); --red-soft:rgba(239,68,68,0.14);
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin:0; padding:0; background: var(--ink); color: var(--paper);
    font-family: 'Poppins', system-ui, sans-serif; font-size: 13px; line-height: 1.5; }}
  .mono, code {{ font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace; }}
  ::selection {{ background: var(--teal); color: var(--ink); }}

  header.app {{
    border-bottom: 1px solid var(--line);
    padding: 22px 28px 18px;
    background: linear-gradient(180deg, #0a0a0c 0%, var(--ink) 100%);
  }}
  .brand {{ font-weight: 700; font-size: 22px; letter-spacing: -0.01em; }}
  .brand .dot {{ color: var(--teal); margin: 0 6px; }}
  .strap {{ color: var(--muted); margin-top: 4px; font-size: 13px; }}
  .legend {{ margin-top: 10px; display: flex; gap: 14px; flex-wrap: wrap; font-size: 11.5px; color: var(--muted-2); }}
  .legend .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }}

  main {{ padding: 22px 28px 60px; max-width: 1400px; margin: 0 auto; }}
  section {{ margin-bottom: 36px; }}
  .h2 {{ font-size: 16px; font-weight: 600; margin: 0 0 12px; letter-spacing: -0.01em; }}
  .h2 .sub {{ color: var(--muted); font-weight: 400; margin-left: 8px; font-size: 12.5px; }}

  /* Tiles */
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; }}
  .tile {{
    border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px;
    background: var(--ink-2); position: relative; overflow: hidden;
  }}
  .tile .cad {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
  .tile .stat-row {{ display: flex; gap: 16px; margin-top: 10px; align-items: baseline; }}
  .tile .stat {{ display: flex; flex-direction: column; }}
  .tile .stat .lbl {{ font-size: 10.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .tile .stat .val {{ font-size: 18px; font-weight: 600; font-family: 'JetBrains Mono', monospace; margin-top: 2px; }}
  .tile .clear {{
    margin-top: 14px; font-family: 'JetBrains Mono', monospace; font-weight: 600;
    font-size: 28px; line-height: 1.1;
  }}
  .tile .clear .denom {{ color: var(--muted); font-size: 18px; }}
  .tile .clear-lbl {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px; }}
  .badge {{ position: absolute; top: 14px; right: 14px; padding: 3px 9px; border-radius: 999px; font-size: 11px; font-weight: 600; }}
  .badge.green {{ background: var(--green-soft); color: var(--green); }}
  .badge.amber {{ background: var(--amber-soft); color: var(--amber); }}
  .badge.red {{ background: var(--red-soft); color: var(--red); }}
  .tile.green {{ border-color: rgba(16,185,129,0.4); }}
  .tile.amber {{ border-color: rgba(245,158,11,0.4); }}
  .tile.red {{ border-color: rgba(239,68,68,0.4); }}

  /* Filters */
  .filters {{
    border: 1px solid var(--line); border-radius: 12px; padding: 14px 18px;
    background: var(--ink-2); position: sticky; top: 0; z-index: 30;
    display: flex; gap: 28px; flex-wrap: wrap; align-items: flex-start;
  }}
  .ctrl {{ display: flex; flex-direction: column; gap: 6px; }}
  .ctrl-label {{ font-size: 10.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }}
  .seg {{ display: inline-flex; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
  .seg button {{
    background: var(--ink); color: var(--muted-2); border: none; padding: 7px 12px; font-size: 12px;
    cursor: pointer; font-family: inherit; border-right: 1px solid var(--line);
  }}
  .seg button:last-child {{ border-right: none; }}
  .seg button:hover {{ color: var(--paper); background: var(--ink-3); }}
  .seg button.active {{ background: var(--teal); color: var(--ink); font-weight: 600; }}
  .chips {{ display: flex; gap: 6px; flex-wrap: wrap; }}
  .chip {{
    padding: 6px 11px; border-radius: 999px; border: 1px solid var(--line);
    background: var(--ink); color: var(--muted-2); font-size: 12px; cursor: pointer; font-family: inherit;
  }}
  .chip.active {{ background: var(--teal-soft); color: var(--teal); border-color: var(--teal-2); font-weight: 600; }}
  select.sort {{
    background: var(--ink); color: var(--paper); border: 1px solid var(--line); border-radius: 8px;
    padding: 7px 10px; font-size: 12px; font-family: inherit;
  }}

  /* Table */
  .tbl-wrap {{ border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: var(--ink-2); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  th, td {{ padding: 9px 12px; text-align: left; border-bottom: 1px solid var(--line); }}
  th {{ background: var(--ink-3); color: var(--muted-2); font-weight: 600; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.05em; }}
  tr:last-child td {{ border-bottom: none; }}
  td.num {{ font-family: 'JetBrains Mono', monospace; text-align: right; }}
  th.num {{ text-align: right; }}
  tr:hover td {{ background: rgba(255,255,255,0.02); }}
  .status-chip {{ display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 11px; font-weight: 600; }}
  .status-chip.green {{ background: var(--green-soft); color: var(--green); }}
  .status-chip.amber {{ background: var(--amber-soft); color: var(--amber); }}
  .status-chip.red {{ background: var(--red-soft); color: var(--red); }}
  .cohort-tag {{ display: inline-block; padding: 1px 7px; border-radius: 4px; font-size: 11px; font-weight: 600;
    font-family: 'JetBrains Mono', monospace; background: var(--ink-3); color: var(--muted-2); }}
  .cohort-tag.A {{ background: rgba(16,185,129,0.18); color: var(--green); }}
  .cohort-tag.B {{ background: rgba(20,184,166,0.18); color: var(--teal); }}
  .cohort-tag.C {{ background: rgba(245,158,11,0.18); color: var(--amber); }}
  .cohort-tag.D {{ background: rgba(239,68,68,0.18); color: var(--red); }}

  /* Chart */
  .chart-wrap {{ border: 1px solid var(--line); border-radius: 12px; padding: 16px; background: var(--ink-2); }}
  .chart-canvas-wrap {{ position: relative; height: 360px; }}

  /* Truth */
  .truth {{ border-left: 3px solid var(--teal); background: var(--ink-2); padding: 18px 22px;
    border-radius: 0 12px 12px 0; }}
  .truth p {{ margin: 0 0 12px; color: var(--paper-2); }}
  .truth p:last-child {{ margin-bottom: 0; }}
  .truth strong {{ color: var(--paper); }}

  footer {{ text-align: center; color: var(--muted); font-size: 11.5px; padding: 24px;
    border-top: 1px solid var(--line); margin-top: 30px; }}
</style>
</head>
<body>

<header class=\"app\">
  <div class=\"brand\">EdgeGrid <span class=\"dot\">.</span> Block-Wise Forecast Accuracy</div>
  <div class=\"strap\">Feb 5-12, 2026 backfill <span class=\"dot\" style=\"color:var(--muted)\">.</span> 42 meters <span class=\"dot\" style=\"color:var(--muted)\">.</span> 4 cadences</div>
  <div class=\"legend\">
    <span><span class=\"swatch\" style=\"background:var(--green)\"></span>green &lt;5% MAPE</span>
    <span><span class=\"swatch\" style=\"background:var(--amber)\"></span>amber 5-10%</span>
    <span><span class=\"swatch\" style=\"background:var(--red)\"></span>red &gt;10%</span>
  </div>
</header>

<main>

  <section>
    <div class=\"h2\">Fleet headline <span class=\"sub\">mean / median MAPE and meters clearing the 5% target</span></div>
    <div class=\"tiles\" id=\"fleet-tiles\"></div>
  </section>

  <section>
    <div class=\"h2\">Filters</div>
    <div class=\"filters\">
      <div class=\"ctrl\">
        <div class=\"ctrl-label\">Cadence</div>
        <div class=\"seg\" id=\"cadence-seg\"></div>
      </div>
      <div class=\"ctrl\">
        <div class=\"ctrl-label\">Cohort</div>
        <div class=\"chips\" id=\"cohort-chips\"></div>
      </div>
      <div class=\"ctrl\">
        <div class=\"ctrl-label\">Sort</div>
        <select class=\"sort\" id=\"sort-sel\">
          <option value=\"mape_asc\">MAPE asc</option>
          <option value=\"mape_desc\">MAPE desc</option>
          <option value=\"kwh_desc\">Total kWh desc</option>
        </select>
      </div>
    </div>
  </section>

  <section>
    <div class=\"h2\">Per-meter table <span class=\"sub\" id=\"meter-count\"></span></div>
    <div class=\"tbl-wrap\">
      <table>
        <thead>
          <tr>
            <th>Meter ID</th>
            <th>Cohort</th>
            <th class=\"num\">MAPE (%)</th>
            <th class=\"num\">Median APE (%)</th>
            <th class=\"num\">p95 APE (%)</th>
            <th class=\"num\">MBE (kWh)</th>
            <th class=\"num\">Total actual kWh</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody id=\"meter-tbody\"></tbody>
      </table>
    </div>
  </section>

  <section>
    <div class=\"h2\">Cohort rollup <span class=\"sub\">mean MAPE per (cohort, cadence). Dashed line = 5% target.</span></div>
    <div class=\"chart-wrap\">
      <div class=\"chart-canvas-wrap\"><canvas id=\"cohortChart\"></canvas></div>
    </div>
  </section>

  <section>
    <div class=\"h2\">Forward forecast preview <span class=\"sub\" id=\"fwd-range\"></span></div>
    <div class=\"tbl-wrap\">
      <table>
        <thead>
          <tr>
            <th>Meter ID</th>
            <th>Cohort</th>
            <th class=\"num\">Total predicted kWh</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody id=\"fwd-tbody\"></tbody>
      </table>
    </div>
    <div style=\"color:var(--muted); font-size:12px; margin-top:10px;\">
      MAPE pending - actuals will flow in T+1 day per block.
    </div>
  </section>

  <section>
    <div class=\"h2\">Truth statement</div>
    <div class=\"truth\">
      <p><strong>The 5% MAPE target is not physically reachable at 15-min DAM cadence for all 42 meters on current data.</strong></p>
      <p>The persistence oracle (median same-DOW same-hour from last 4 weeks - no model) hits 32% median MAPE at 30-min on this window. Broadcasting to 15-min doesn't change that floor. What would unlock it: EV telematics, AC cycle telemetry, per-appliance sub-metering - observability the model currently cannot see.</p>
      <p>At ToD 4-hour cadence, 23/42 meters clear 5%. At Hourly, 17/42. At 30-min / 15-min DAM, 10/42. The honest framing is: dispatch decisions that care about 4-hour windows are solvable today; real-time block-level forecasting for spike-y residential meters needs new data sources, not a better model.</p>
    </div>
  </section>

</main>

<footer>EdgeGrid Forecast Engine - generated from block_accuracy_summary.csv, block_accuracy_cohort.csv, block_accuracy_fleet.csv, forward_forecast_30d.parquet</footer>

<script id=\"payload\" type=\"application/json\">{payload}</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const CADENCE_ORDER = DATA.cadence_order;
const CADENCE_LABEL = DATA.cadence_label;
const COHORTS = ['A','B','C','D'];

const STATE = {{
  cadence: 'dam_15min',
  cohorts: new Set(COHORTS),
  sort: 'mape_asc',
}};

function classify(mape) {{
  if (mape < 5) return 'green';
  if (mape < 10) return 'amber';
  return 'red';
}}
function fmt(v, d=2) {{ return Number(v).toFixed(d); }}

function renderTiles() {{
  const wrap = document.getElementById('fleet-tiles');
  wrap.innerHTML = '';
  DATA.fleet.forEach(t => {{
    let badgeClass, badgeText;
    if (t.clear_count >= 34) {{ badgeClass = 'green'; badgeText = 'on target'; }}
    else if (t.clear_count >= 17) {{ badgeClass = 'amber'; badgeText = 'partial'; }}
    else {{ badgeClass = 'red'; badgeText = 'gap'; }}
    const div = document.createElement('div');
    div.className = 'tile ' + badgeClass;
    div.innerHTML = `
      <span class=\"badge ${{badgeClass}}\">${{badgeText}}</span>
      <div class=\"cad\">${{t.label}}</div>
      <div class=\"stat-row\">
        <div class=\"stat\"><span class=\"lbl\">Mean MAPE</span><span class=\"val\">${{fmt(t.mean_mape, 2)}}%</span></div>
        <div class=\"stat\"><span class=\"lbl\">Median</span><span class=\"val\">${{fmt(t.median_mape, 2)}}%</span></div>
      </div>
      <div class=\"clear\">${{t.clear_count}}<span class=\"denom\">/${{t.total_meters}}</span></div>
      <div class=\"clear-lbl\">meters &lt;5% MAPE</div>
    `;
    wrap.appendChild(div);
  }});
}}

function renderControls() {{
  const seg = document.getElementById('cadence-seg');
  seg.innerHTML = '';
  CADENCE_ORDER.forEach(c => {{
    const b = document.createElement('button');
    b.textContent = CADENCE_LABEL[c];
    if (c === STATE.cadence) b.classList.add('active');
    b.onclick = () => {{ STATE.cadence = c; renderControls(); renderTable(); }};
    seg.appendChild(b);
  }});
  const chips = document.getElementById('cohort-chips');
  chips.innerHTML = '';
  COHORTS.forEach(co => {{
    const b = document.createElement('button');
    b.className = 'chip' + (STATE.cohorts.has(co) ? ' active' : '');
    b.textContent = 'Cohort ' + co;
    b.onclick = () => {{
      if (STATE.cohorts.has(co)) STATE.cohorts.delete(co); else STATE.cohorts.add(co);
      if (STATE.cohorts.size === 0) STATE.cohorts = new Set(COHORTS);
      renderControls(); renderTable();
    }};
    chips.appendChild(b);
  }});
  document.getElementById('sort-sel').value = STATE.sort;
  document.getElementById('sort-sel').onchange = (e) => {{ STATE.sort = e.target.value; renderTable(); }};
}}

function renderTable() {{
  const tbody = document.getElementById('meter-tbody');
  tbody.innerHTML = '';
  let rows = DATA.meters.filter(r => r.cadence === STATE.cadence && STATE.cohorts.has(r.cohort));
  if (STATE.sort === 'mape_asc') rows.sort((a,b) => a.mape - b.mape);
  else if (STATE.sort === 'mape_desc') rows.sort((a,b) => b.mape - a.mape);
  else rows.sort((a,b) => b.total_actual_kwh - a.total_actual_kwh);
  rows.forEach(r => {{
    const cls = classify(r.mape);
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class=\"mono\">${{r.meter_id}}</td>
      <td><span class=\"cohort-tag ${{r.cohort}}\">${{r.cohort}}</span></td>
      <td class=\"num\">${{fmt(r.mape, 2)}}</td>
      <td class=\"num\">${{fmt(r.median_mape, 2)}}</td>
      <td class=\"num\">${{fmt(r.p95_ape, 2)}}</td>
      <td class=\"num\">${{fmt(r.mbe_kwh, 4)}}</td>
      <td class=\"num\">${{fmt(r.total_actual_kwh, 2)}}</td>
      <td><span class=\"status-chip ${{cls}}\">${{cls === 'green' ? 'on target' : cls === 'amber' ? '5-10%' : '>10%'}}</span></td>
    `;
    tbody.appendChild(tr);
  }});
  document.getElementById('meter-count').textContent =
    `${{rows.length}} meter${{rows.length === 1 ? '' : 's'}} - ${{CADENCE_LABEL[STATE.cadence]}}`;
}}

function renderForward() {{
  const tbody = document.getElementById('fwd-tbody');
  tbody.innerHTML = '';
  const sorted = [...DATA.forward].sort((a,b) => b.total_predicted_kwh - a.total_predicted_kwh);
  sorted.forEach(r => {{
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class=\"mono\">${{r.meter_id}}</td>
      <td><span class=\"cohort-tag ${{r.cohort}}\">${{r.cohort}}</span></td>
      <td class=\"num\">${{fmt(r.total_predicted_kwh, 2)}}</td>
      <td><span class=\"status-chip amber\">pending T+1</span></td>
    `;
    tbody.appendChild(tr);
  }});
  document.getElementById('fwd-range').textContent =
    `${{DATA.forward_start}} -> ${{DATA.forward_end}} (per-meter total predicted kWh)`;
}}

function renderChart() {{
  const colors = {{
    dam_15min: '#ef4444',
    native_30min: '#f59e0b',
    hourly: '#14b8a6',
    tod_4h: '#10b981',
  }};
  const datasets = CADENCE_ORDER.map(cad => ({{
    label: CADENCE_LABEL[cad],
    backgroundColor: colors[cad],
    borderColor: colors[cad],
    borderWidth: 1,
    data: COHORTS.map(co => {{
      const m = DATA.cohorts.find(r => r.cohort === co && r.cadence === cad);
      return m ? m.mean_mape : 0;
    }}),
  }}));
  const ctx = document.getElementById('cohortChart').getContext('2d');
  new Chart(ctx, {{
    type: 'bar',
    data: {{ labels: COHORTS.map(c => 'Cohort ' + c), datasets: datasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      scales: {{
        y: {{
          beginAtZero: true,
          title: {{ display: true, text: 'Mean MAPE (%)', color: '#a1a1aa' }},
          ticks: {{ color: '#a1a1aa' }},
          grid: {{ color: 'rgba(255,255,255,0.05)' }},
        }},
        x: {{ ticks: {{ color: '#a1a1aa' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
      }},
      plugins: {{
        legend: {{ labels: {{ color: '#fafaf9' }} }},
        tooltip: {{
          callbacks: {{
            label: (ctx) => `${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(2)}}%`
          }}
        }},
        annotation: {{}}
      }}
    }},
    plugins: [{{
      id: 'targetLine',
      afterDraw(chart) {{
        const {{ ctx, chartArea: {{ left, right }}, scales: {{ y }} }} = chart;
        const yPos = y.getPixelForValue(5);
        ctx.save();
        ctx.strokeStyle = '#14b8a6';
        ctx.setLineDash([6, 4]);
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(left, yPos); ctx.lineTo(right, yPos); ctx.stroke();
        ctx.fillStyle = '#14b8a6';
        ctx.font = '11px JetBrains Mono';
        ctx.fillText('5% target', right - 70, yPos - 5);
        ctx.restore();
      }}
    }}]
  }});
}}

renderTiles();
renderControls();
renderTable();
renderForward();
renderChart();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
