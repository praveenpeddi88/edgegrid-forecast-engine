"""
Per-substation commercial brief (one-page PDF for the DISCOM GM).

The prototype renders the brief to HTML first (self-contained, printable).
PDF conversion is delegated to the `anthropic-skills:pdf` skill — the
backend endpoint can either return HTML (let the browser Print-to-PDF) or
invoke the skill out-of-process to produce a true PDF.

Design system: Poppins + EdgeGrid teal, per `anthropic-skills:courier-pack`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


EDGEGRID_TEAL = "#0E7C7B"
EDGEGRID_INK = "#0F172A"  # zinc-900
EDGEGRID_PAPER = "#FAFAFA"


@dataclass
class BriefContent:
    substation_id: str
    substation_label: str
    as_of: datetime
    rupees_saved_today: float
    projected_annual_inr: float
    irr_pct: float
    payback_years: float
    top_audit_highlights: list[str] = field(default_factory=list)
    mape_health: str = "green"  # green | amber | red
    peak_block_mape_pct: float = 0.0
    fls_quote_inr_per_kwh: Optional[float] = None
    fls_firmness_pct: Optional[float] = None

    def to_html(self) -> str:
        """
        Self-contained HTML. Print with `html2pdf` or browser Print dialog.
        """
        highlights_html = "\n".join(
            f"<li>{h}</li>" for h in self.top_audit_highlights[:3]
        ) or "<li><em>No dispatch actions yet today.</em></li>"
        health_color = {
            "green": "#10B981", "amber": "#F59E0B", "red": "#EF4444"
        }.get(self.mape_health, "#10B981")
        fls_block = ""
        if self.fls_quote_inr_per_kwh is not None:
            fls_block = f"""
      <div class="kv">
        <div class="kv-label">FLS Quote</div>
        <div class="kv-value">
          \u20B9{self.fls_quote_inr_per_kwh:.2f}/kWh
          <span class="badge">{self.fls_firmness_pct:.0f}%-firm</span>
        </div>
      </div>"""
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>EdgeGrid Commercial Brief — {self.substation_label}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
  :root {{
    --teal: {EDGEGRID_TEAL};
    --ink: {EDGEGRID_INK};
    --paper: {EDGEGRID_PAPER};
  }}
  body {{
    font-family: 'Poppins', system-ui, sans-serif;
    color: var(--ink);
    background: var(--paper);
    margin: 0;
    padding: 40px;
  }}
  .brief {{
    max-width: 780px;
    margin: 0 auto;
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    overflow: hidden;
  }}
  header {{
    background: var(--teal);
    color: white;
    padding: 24px 32px;
  }}
  header h1 {{ margin: 0; font-weight: 600; font-size: 20px; letter-spacing: -0.01em; }}
  header p {{ margin: 4px 0 0; opacity: 0.85; font-size: 13px; font-weight: 400; }}
  .body {{ padding: 28px 32px; }}
  .kv-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin-bottom: 28px;
  }}
  .kv {{ border-left: 3px solid var(--teal); padding-left: 14px; }}
  .kv-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: #6B7280; font-weight: 500; }}
  .kv-value {{ font-size: 20px; font-weight: 600; margin-top: 4px; }}
  .badge {{
    display: inline-block; margin-left: 6px; padding: 2px 8px;
    background: #ECFDF5; color: #065F46; border-radius: 6px;
    font-size: 11px; font-weight: 500;
  }}
  .health-chip {{
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: {health_color}; margin-right: 6px;
    vertical-align: middle;
  }}
  h2 {{ font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; color: #374151; margin-top: 28px; }}
  ul.audit {{ padding-left: 18px; line-height: 1.7; font-size: 13px; }}
  ul.audit li {{ margin-bottom: 4px; }}
  footer {{
    padding: 16px 32px;
    background: #F9FAFB;
    font-size: 11px;
    color: #6B7280;
    display: flex; justify-content: space-between;
    border-top: 1px solid #E5E7EB;
  }}
</style>
</head>
<body>
<div class="brief">
  <header>
    <h1>{self.substation_label}</h1>
    <p>Commercial brief · {self.as_of:%Y-%m-%d %H:%M} IST</p>
  </header>
  <div class="body">
    <div class="kv-grid">
      <div class="kv">
        <div class="kv-label">Saved today</div>
        <div class="kv-value">\u20B9{self.rupees_saved_today:,.0f}</div>
      </div>
      <div class="kv">
        <div class="kv-label">Projected annual</div>
        <div class="kv-value">\u20B9{self.projected_annual_inr / 1e5:.1f} L</div>
      </div>
      <div class="kv">
        <div class="kv-label">Project IRR</div>
        <div class="kv-value">{self.irr_pct:.1f}%<span class="badge">{self.payback_years:.1f}y payback</span></div>
      </div>
      <div class="kv">
        <div class="kv-label">Forecast health</div>
        <div class="kv-value"><span class="health-chip"></span>{self.mape_health.upper()}</div>
      </div>
      <div class="kv">
        <div class="kv-label">Peak-block MAPE</div>
        <div class="kv-value">{self.peak_block_mape_pct:.2f}%</div>
      </div>
      {fls_block}
    </div>
    <h2>Today's dispatch highlights</h2>
    <ul class="audit">
      {highlights_html}
    </ul>
  </div>
  <footer>
    <span>EdgeGrid · Substation {self.substation_id}</span>
    <span>Clean, commercially-optimal energy delivery at the last mile.</span>
  </footer>
</div>
</body>
</html>"""

    def save_html(self, path: Path) -> Path:
        path.write_text(self.to_html(), encoding="utf-8")
        return path


def build_substation_brief(
    substation_id: str,
    substation_label: str,
    dispatch_schedule,  # DispatchScheduleV2 (avoid circular import)
    irr_result,
    peak_block_mape_pct: float,
    fls_quote=None,
) -> BriefContent:
    """
    Assemble the brief from a dispatch schedule + IRR result.

    Top-3 audit highlights: pick the 3 non-hold rows with the largest kWh
    magnitude — these are the commercially most interesting decisions.
    """
    df = dispatch_schedule.df
    # Net benefit today = revenue − cost for rows in "today" (first 96 intervals = 24h)
    today = df.head(96)
    today_net = (
        (today["bess_discharge_kwh"] * today["iex_price_inr"]).sum()
        - (today["grid_import_kwh"] * today["landed_cost_inr"]).sum()
        - (today["bess_charge_kwh"] * today["iex_price_inr"]).sum()
    )
    rupees_saved = max(0.0, float(today_net))

    # Rank dispatch rows by kWh magnitude, skip 'hold'
    df_nonhold = df[df["action"] != "hold"].copy()
    df_nonhold["kwh_abs"] = (
        df_nonhold[["bess_charge_kwh", "bess_discharge_kwh", "solar_curtail_kwh"]].max(axis=1)
    )
    top = df_nonhold.nlargest(3, "kwh_abs")
    highlights = [row["audit_string"] for _, row in top.iterrows()]

    health = "green" if peak_block_mape_pct < 4 else "amber" if peak_block_mape_pct < 8 else "red"

    fls_price = fls_quote.offered_price_inr_per_kwh if fls_quote else None
    fls_firm = fls_quote.firmness_pct if fls_quote else None

    return BriefContent(
        substation_id=substation_id,
        substation_label=substation_label,
        as_of=datetime.now(),
        rupees_saved_today=rupees_saved,
        projected_annual_inr=rupees_saved * 365,
        irr_pct=irr_result.irr_pct,
        payback_years=irr_result.payback_years,
        top_audit_highlights=highlights,
        mape_health=health,
        peak_block_mape_pct=peak_block_mape_pct,
        fls_quote_inr_per_kwh=fls_price,
        fls_firmness_pct=fls_firm,
    )
