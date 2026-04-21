# v5 vs CEA Demand-Forecast Guidelines — Alignment & Gap Analysis

**Source document:** *Guidelines for Medium and Long Term Power Demand
Forecast*, Central Electricity Authority, Ministry of Power, Government of
India (25 pages; sections A–H + Annexures I–V).

**Purpose:** map every CEA clause to what v5 does today, flag what's
missing, and lay out a pragmatic roadmap to make EdgeGrid's forecast
engine defensible under Indian regulatory review — without losing the
operational edge that makes it useful to a dispatcher.

**Headline:** CEA's guidelines describe a **statutory, category-wise,
annual-to-decadal** forecasting discipline. v5 is an **operational,
per-meter, half-hour-to-week** forecasting engine. The two are
**complementary layers**, not competing methods. A CEA-aligned EdgeGrid
product ships v5 as-is for dispatch and wraps it in a roll-up +
scenario layer for regulatory submission.

---

## 1. What CEA actually asks for

The CEA document is structured in eight parts plus five annexures.
The obligations are clustered into four themes, summarised below.

### 1.1 Horizon, granularity, scenarios  (Part A)

Medium term is >1 and ≤5 years; long term is ≥10 years (A.1, A.2).
Year-wise at Discom/State is mandatory; finer spatial (zone/circle/
district/sub-station/feeder/transformer) and finer time (month/day/
hour/time-block) are **encouraged if data permits** (A.7, A.8).
Three scenarios are required — Optimistic, Business-As-Usual,
Pessimistic — with more permutations possible for weather extremes
(A.9). Forecasts must be under the **unrestricted** scenario — the
demand that would have been served if outages and load-shedding did
not exist (A.10). Validation must be done by **at least one different
method**; Econometric (GDP/GSDP-based) is preferred as the second
method (A.14).

### 1.2 Inputs & method — Partial End Use Method  (Parts B, C)

Input is **category-wise consumption** for ≥10 past years, along the
tariff categories Domestic, Commercial, Public Lighting, Public Water
Works, Irrigation, LT Industries, HT Industries, Railways, Bulk
Supply, Open Access, Others (B.2, B.3). Weather (temperature,
rainfall) is a collected input (B.6).

The method is **PEUM — Partial End Use Method** — where each tariff
category's historical growth is extrapolated independently using
**Least Squares** and **Weighted Average** (with heavier weight on
recent years) and a hybrid selection rule (Annexure II: LSM when
growth rates diverge due to a definite policy/tech shift, Weighted
Average otherwise). Base year is T-2 so that T-1 can be used for
out-of-sample validation (A.5, A.6, C.5). T&D losses are projected
separately — Distribution, Intra-State Transmission, Inter-State
Transmission (C.2). Energy requirement = consumption + T&D losses;
peak demand is derived from energy via **Load Factor** (C.6, G.1).

### 1.3 Emerging aspects  (Part D)

Impacts of EV penetration, Green Hydrogen, solar rooftop, government
schemes and major tech shifts are quantified **on top of** the PEUM
baseline, apportioned to the matching tariff category (D.3). Annexure V
walks through the 20th-EPS EV methodology end-to-end — CAGR of vehicle
sales, 30% BEV share by 2029-30, 70/30 domestic-to-commercial split,
charging-time-based peak calculation.

### 1.4 Roll-up & sanity checks  (Parts E–H)

Discom energy = consumption + distribution losses + intra-state
transmission losses (E.1). State energy = sum of Discoms + inter-state
losses (F.1, F.3). Inter-state losses come from GRID-INDIA and are
applied to net-import (F.4). Peak demand via Load Factor (G.2).
Diversity factor (sum of Discom peaks / State peak) must be >1; the
CEA publishes regional ranges (1.05 Southern — 1.13 Northern) and
Load Factor sanity range is 40–80% (H.i–H.iii).

### 1.5 Weather — the degree-day method  (Annexure IV)

Rather than raw temperature, CEA prescribes **Heating Degree Days** and
**Cooling Degree Days** around a 21 °C threshold (per 19th EPS review),
summed over a year. The "extreme year" is identified by max/min
HDD_Y + CDD_Y, and the % deviation of actual vs. notional demand in
that year becomes the Optimistic / Pessimistic shift applied to BAU.

---

## 2. What v5 does today that aligns with CEA

| CEA clause | v5 equivalent | Status |
|---|---|---|
| A.8 — finer time granularity if data permits | Per-meter half-hour forecast over 42 smart meters | **Aligned, exceeds** |
| A.7 — finer spatial granularity if data permits | Feeder/transformer-level forecast at the smart-meter | **Aligned, exceeds** |
| A.5 / C.5 — out-of-sample validation, T-1 vs model | S1/S2/S3 rolling validation, Feb-12 train-cutoff, 10 weekly origins | **Aligned** |
| A.10 — unrestricted demand | `outage_windows.parquet` excludes outage hours from training loss so the model doesn't learn zeros that are really power cuts | **Partial** (excluded from training; not yet back-filled into output) |
| A.14 — validate through a different method | S1 (pure regression) vs S3 (forecast) is a *protocol* diversity; no second algorithmic family yet | **Partial** |
| B.6 — collect weather parameters | `fetch_weather_expanded()` pulls Visakhapatnam temperature, humidity, wind, etc. | **Aligned** |
| C.1 — analyse historical growth per category | LightGBM implicitly learns per-meter trend; no explicit per-category LSM/WA | **Partial** (captured implicitly in tier-adaptive regularisation) |
| A.9 — three scenarios | Four forward strategies (A seasonal anchor, B v4 batch, C v5 batch, D ensemble median) | **Partial** (inference variance, not Optimistic/BAU/Pessimistic) |

v5 **exceeds** CEA on granularity — CEA's guidelines are apologetic
about yearly-Discom scope and repeatedly nudge readers to go finer "if
adequate granular data is available." We already are.

---

## 3. Where v5 is silent on CEA and needs a companion layer

The gaps below are not v5 *bugs* — they're responsibilities of a
regulatory-reporting layer that wraps v5, not of the operational
engine. I've scored each by effort (S/M/L) and commercial leverage
(how much it helps when EdgeGrid is pitching APEPDCL or a neighbouring
DISCOM).

| Gap | CEA clause | Effort | Commercial leverage |
|---|---|---|---|
| **Category-wise (PEUM) roll-up** — aggregate the 42-meter forecast into Domestic / Commercial / HT Industry / Irrigation / LT Industry buckets using meter-to-category metadata | B.2, C.1 | M | **High** — this is the primary DISCOM-facing deliverable CEA expects. Without it, our forecast is "technically correct but not in the format the regulator reads." |
| **Optimistic / BAU / Pessimistic scenarios** — re-label Strategies A/C/D and layer HDD-extreme and HDD-mild scenarios on top | A.9, Annexure III | S | **High** — three scenarios is the statutory ask. Already most of the plumbing is there. |
| **HDD / CDD features** — replace raw temperature with Cooling Degree Days at 21 °C threshold (Annexure IV); add HDD for winter/hilly-meter coverage | Annexure IV | S | **Medium** — expected marginal MAPE gain on temperature-sensitive meters; gives us a CEA-named feature to cite in a technical appendix. |
| **Econometric validation cross-check** — fit a simple per-category GSDP-elasticity model, compare its forecast against v5's aggregate | A.14, Annexure III | M | **Medium** — satisfies the "validate through a different method" clause; trivial to produce from AP GSDP series. |
| **T&D loss decomposition** — split losses into Distribution / Intra-State / Inter-State, project separately, add on top of the Discom-level sum | C.2, E.1, F.1 | M | **High** — every DISCOM loss report uses this taxonomy. Without it our "energy requirement" number does not reconcile to theirs. |
| **Peak demand via Load Factor** — compute Peak (MW) from our half-hour forecast using G.2's formula, emit monthly/yearly Load Factor | G.1, G.2 | S | **High** — Peak Demand is the number that drives capacity-planning CapEx. Load Factor sanity bounds (40–80 %) give the regulator a quick credibility signal. |
| **Diversity-factor check** — if/when we roll up to APEPDCL substation totals, emit the diversity factor and flag if <1 | H.iii | S | **Low** (unit-test-grade), but free |
| **Emerging-aspects layer** — post-processing add-ons for EV penetration, solar rooftop, new industrial loads; each quantified on top of the baseline | Part D, Annexure V | L | **Very High** — this is the 2026–2030 growth story. EV penetration in AP is a line-item every DISCOM has to model for the next five years. |
| **Medium / long-term horizon** — explicitly not v5's job. Offer a companion module that takes v5's 30-day forward as Year-1 anchor and does CEA's LSM / Weighted-Average extrapolation out to Year-5 and Year-10 | A.1, A.2, A.13 | L | **Very High** — opens the door to being embedded in the statutory EPS (Electric Power Survey) cycle, not just operational dispatch. |
| **Unrestricted-demand back-fill** — when reporting, replace outage-hour zeros with the model's own counterfactual demand so the output reflects A.10's "as if no load-shedding" contract | A.10 | M | **Medium** — makes the output directly CEA-compliant at the reporting boundary. |

---

## 4. Reframing v5 for a CEA-aware commercial pitch

CEA's document implicitly splits forecasting into two worlds. We
should name them explicitly in our collateral.

**World 1 — Statutory forecasting (CEA's world).**
Annual, Discom-aggregated, category-wise, scenario-bounded. Used for
Electric Power Survey (EPS) submissions, tariff determination, and
capacity-planning CapEx. Accuracy is secondary to *traceability* — a
regulator must be able to replay the method and arrive at the same
number.

**World 2 — Operational forecasting (EdgeGrid's world).**
Half-hour, feeder / substation / meter, weather-aware, uncertainty-
banded. Used for BESS dispatch, solar-plus-storage sizing, IEX DAM bid
preparation, and real-time network planning. Accuracy matters because
money moves on it.

v5 is a **World 2** product today. The gap analysis in §3 is the
recipe for extending v5 into **World 1** without losing its World 2
edge. The single highest-leverage move is the **PEUM roll-up + three
scenarios + Load Factor / Peak Demand derivation** — about 6–8 weeks
of engineering — because it converts v5's half-hour output into the
annual-category-scenario format CEA expects, while keeping the engine
unchanged.

Positioning line for the EdgeGrid narrative:

> EdgeGrid's forecast engine operates at the half-hour / feeder scale
> that modern dispatch requires, and rolls up cleanly into the
> Discom-category-scenario format CEA's EPS process expects. One engine,
> two deliverables — dispatch-ready today, EPS-ready by quarter-end.

---

## 5. Recommended near-term work (roadmap slice)

Four workstreams, each a clean unit of scope:

**W1 — HDD/CDD feature swap.** Replace raw-temperature columns with
Cooling Degree Days (21 °C threshold, daily sum rolled into half-hour
encoding via dow×hour cross-terms). Retrain v5 bundles. Expected MAPE
impact: 0.2–0.5 pp on temperature-sensitive meters. Deliverable: new
manifest, updated S1 report.

**W2 — Three-scenario rebrand.** Relabel forward strategies:
- *BAU* = v5 batch with Feb-12 bundle (today's Strategy C)
- *Optimistic* = BAU × (1 + δ_hot) where δ_hot comes from the hottest
  CDD-year in the past 5 years of AP weather
- *Pessimistic* = BAU × (1 + δ_cool), symmetric
- (Strategy D ensemble stays as the deployment default)

Deliverable: `outputs/forward_v5_feb12/scenarios.parquet`, a new
endpoint `GET /v5/forecast/scenarios`, three matching rows in the
V5_RELEASE §2.4 table.

**W3 — PEUM roll-up layer.** New module `edgegrid_forecast/reporting/
peum_rollup.py`. Takes per-meter forecasts, joins to a meter-to-
category map (Domestic / Commercial / HT / LT / Irrigation from
APEPDCL tariff codes), aggregates to monthly / yearly, emits a CSV in
Annexure-I format. Deliverable: `outputs/peum/peum_rollup_FY2026.csv`
and a new endpoint `GET /v5/forecast/peum?year=2026`.

**W4 — Peak + Load Factor derivation.** One file, one screenful.
Read the half-hour series, compute monthly + yearly Peak Demand (max
MW), Energy Requirement (sum MU), Load Factor (%). Emit. Fail loudly
if Load Factor falls outside 40–80 %. Deliverable: a small report card
that a DISCOM compliance officer can sign off on in five minutes.

**W5 (stretch) — EV impact module.** Replicate the Annexure V
methodology for AP-specific vehicle registration data. This is the
2030 growth story. Worth doing last because it compounds on top of W3.

All five workstreams reuse the existing engine. No retraining is
required beyond W1.

---

## 6. What to tell a CEA-aware reviewer about v5 today

If an APEPDCL or CEA reviewer asked "is this CEA-compliant?" today,
the honest answer is:

> v5 is a half-hour, per-meter operational forecast engine that
> satisfies the spirit of CEA A.7, A.8, A.10, and A.14 on granularity,
> unrestricted-demand handling, and out-of-sample validation. It does
> not yet emit the category-wise PEUM roll-up, the Optimistic / BAU /
> Pessimistic scenario stack, or the T&D-loss-decomposed Energy
> Requirement that the statutory EPS template expects. Those are a
> reporting layer on top of the engine, not a different engine.
> Targeted 6–8 week delivery.

That's a better answer than dressing up a non-compliance, and it
matches the tone of honest reporting we've already used in
V5_RELEASE.md §2.

---

*Next update: after W1 (HDD/CDD swap) lands. Re-run S1 and fold the
delta into V5_RELEASE §2.1.*

---

## 7. CEA Bridge — delivered (W1 + W2 + W4)

Shipped on 2026-04-21. v5 version pinned at **v5.s1.1**. Three workstreams
land together; W3 (PEUM roll-up) and W5 (EV module) remain on the backlog
per original sequencing.

### 7.1 What shipped

**W1 — CEA-canonical HDD/CDD at 21 °C** (`inference/_features.py`)

Six new features land in the canonical builder, consumed by both v4 and v5
through `build_features_v4_s1`:

- `hdd_21`, `cdd_21` — non-negative degree-hour at the CEA 19th EPS threshold
- `cdd_21_rmean_48`, `cdd_21_rmean_336` — 24 h and 7 d trailing CDD (heat-wave pressure)
- `cdd_21_x_hour`, `cdd_21_x_peak` — interaction terms for evening AC ramp

The two-pass LightGBM screen picks these up per-meter where they help;
industrial / inverter-dominated meters appropriately ignore them.

**W2 — Three-scenario API rebrand** (`api/v5_router.py`, `inference/_scenarios.py`)

q10 / mean / q90 heads are dual-labelled with CEA commercial names:
- `pessimistic` ≡ q10 — conservative, weather-normal, industrial −10%
- `bau` ≡ mean — central, CEA-consistent EV uptake, trend industrial
- `optimistic` ≡ q90 — hot-summer stress, industrial +15%, accelerated EV

Every `PredictRow` carries both label families. `GET /v5/scenarios` exposes
the narrative drivers in a stable shape suited for exec dashboards and
SERC filing footnotes. `POST /v5/predict` accepts `scenario=` as a filter.

**W4 — Peak Demand + Load Factor derivation** (`inference/_derived.py`)

Commercial/regulatory filings consume demand (kW) and load factor, not
half-hour energy (Wh). The new module derives per-scenario:
- `peak_kw`, `peak_ts`
- `average_kw`, `load_factor` (0–1)
- `total_energy_kwh`, `horizon_hours`

`POST /v5/fleet/peak` adds coincident-peak + **diversity factor** across a
meter cohort — the invariant needed for LT/HT feeder loadability studies.

### 7.2 MAPE impact of W1 (honest)

Apples-to-apples on the 42 common meters, 85/15 chronological holdout:

| metric            | pre-W1 | post-W1 | delta   |
|-------------------|--------|---------|---------|
| mean MAPE         | 6.34%  | 6.14%   | ↓ 0.20pp |
| median MAPE       | 4.83%  | 4.75%   | ↓ 0.08pp |
| p90 MAPE          | 12.41% | 12.06%  | ↓ 0.34pp |
| meters < 4% MAPE  | 13     | 15      | +2       |
| meters < 5% MAPE  | 23     | 24      | +1       |
| meters < 10% MAPE | 35     | 35      | 0        |

Biggest wins: 65021124 (−3.42pp), 65023784 (−1.20pp), 65041990 (−1.01pp) —
residential / commercial cohorts with clear AC signatures.
Biggest regressions: 51057607 (+1.26pp), 67003234 (+1.13pp), 65003102
(+0.82pp) — mostly industrial + inverter-dominated meters where CDD
features compete with the feature budget.

**Headline:** CEA-canonical features are a net positive, particularly on
the thermally driven meters that matter for summer peak. They don't
break the <4% floor we care about, but they also don't single-handedly
push us through it — the last mile needs W3 (PEUM roll-up) and
meter-specific bias work.

### 7.3 Contract tests

`tests/test_cea_bridge.py` pins the contract:
- W1: CEA feature columns + non-overlap invariant on HDD/CDD
- W2: scenario→quantile mapping, alias resolution, narrative shape
- W4: Wh→kW conversion constant, LF bounds, DF ≥ 1 invariant
- API: `/v5/scenarios`, `/v5/predict` (full block + scenario filter + 400),
  `/v5/fleet/peak` (Σpeaks ≥ coincident peak)

14 new tests. Full suite: 177 pass, 2 pre-existing skips.

### 7.4 What's explicitly NOT shipped

- **W3 — PEUM roll-up** (category-wise Domestic/Commercial/Industrial/
  Agricultural aggregation + T&D-loss decomposition). Requires DTR/feeder
  mapping from APEPDCL and a category attribute on each MSN; not in the
  current data. Unblocks CEA 19th EPS template submission.
- **W5 — EV module** (AP-specific vehicle registration roll-up, per NITI
  mid-case). Deferred until W3 lands.

Re-estimate: W3 is 3 weeks once APEPDCL provides the MSN→category map;
W5 is 1 week on top of W3.
