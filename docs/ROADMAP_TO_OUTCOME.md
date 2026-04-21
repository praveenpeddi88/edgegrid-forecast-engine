# Roadmap — from v5.s1.1 to the stated outcome

**Outcome:** a production-grade demand forecasting engine that predicts/
forecasts demand with low MAPE, and low MAE, per smart meter, deployable
as a service — and specifically, MAPE < 4% across all 42 APEPDCL smart
meters on chronological holdout and on forward dates.

**Where we are today (v5.s1.1):** 15/42 meters under 4% MAPE; mean 6.14%,
median 4.75%, p90 12.06%. Service deployable end-to-end via FastAPI with
Docker + systemd + CI. 177 tests passing. CEA Bridge (W1+W2+W4) shipped.

**The gap is real and splits into three tracks.** They run partially in
parallel; Track A closes the MAPE gap, Track B upgrades the service
wrapper into a platform, Track C completes the CEA commercial layer
(blocked on APEPDCL data).

---

## Track A — MAPE quality (the honest last mile)

Sequenced. A1 is diagnosis and unblocks A2/A3/A4, which run in parallel.
A5 is a judgement call: only if A2-A4 leave ≥ 5 meters above 4%. A6 is
the credibility gate — we don't claim <4% without rolling-origin proof.

| # | Task | Eff | Expected impact | Depends on |
|---|---|---|---|---|
| A1 | Residual forensics — classify 27 above-4% meters by failure mode (bias-gate victim / industrial / outage-corrupted / data-quality / AC-missing). Output: `docs/MAPE_LAST_MILE.md` | 2 d | diagnosis; no direct MAPE lift | — |
| A2 | Per-tier bias-gate re-tuning (global → per-tier threshold) | 2–3 d | +1-3 meters under 4% | A1 |
| A3 | Industrial/inverter cohort separate recipe (no HDD/CDD, heavier lags, fleet-ratio features) | 4–5 d | +3-5 meters under 4% | A1 |
| A4 | Outage-corrupted meter treatment (imputation + row-weight + recent_outage flag) | 2–3 d | +1-2 meters under 4% | A1 |
| A5 | Second-estimator ensemble for the worst 5-7 meters (Chronos or LSTM blended per-meter) | 5–7 d | +2-5 meters under 4% | A2 + A3 + A4 |
| A6 | S3 rolling-origin re-benchmark on post-fix bundles (10 weekly origins) | 1 d | credibility gate | A5 |

**Track A total, elapsed time:** 3–4 weeks with focused effort.
**Best-case outcome:** 38-42 meters under 4%. The 2-3 meter tail may
simply have data-quality issues no model fixes — that's why A6 is the
"publish or hold" gate.

## Track B — Production platform

Can run in parallel with Track A. B1 is the unlock — nothing downstream
works without telemetry.

| # | Task | Eff | Expected impact | Depends on |
|---|---|---|---|---|
| B1 | Prometheus/OTel observability — `/v5/metrics` + Grafana dashboards | 3 d | foundational | — |
| B2 | Drift detection + automated retrain triggers + Slack pager | 4 d | alerting | B1 |
| B3 | Real data pipeline — replace CSV with APEPDCL MDM feed (SFTP/API ingest + schema validation + anomaly detection) | 4–5 d | live data | BLOCKED on APEPDCL feed spec |
| B4 | Model registry migration — filesystem → MLflow | 3 d | rollback-safe ops | — |
| B5 | Blue/green deployment + SLOs (p99 <500ms, 99.9% avail, MAPE SLO) | 3 d | safe rollouts | B1 |
| B6 | Structured logging + request tracing (request_id, msn, latency) | 2 d | production debuggability | — |

**Track B total, elapsed time:** 3–4 weeks with focused effort.
**Parallelism note:** B1/B4/B6 can start immediately. B2/B5 need B1
telemetry. B3 is blocked on an APEPDCL business conversation.

## Track C — CEA Bridge completion

The W1+W2+W4 work that shipped is the ~60% of the CEA Bridge we could do
without APEPDCL data. The remaining 40% needs APEPDCL to deliver the
MSN→category mapping and feeder hierarchy before we can start.

| # | Task | Eff | Expected impact | Depends on |
|---|---|---|---|---|
| C1 | W3 — PEUM category-wise roll-up (Domestic/Commercial/Industrial/Agri) + T&D-loss decomposition. Emits category-level Energy Requirement on CEA 19th EPS template. | 3 w | regulatory submission-ready | BLOCKED on APEPDCL MSN→category mapping |
| C2 | W5 — EV charging module (AP RTO feed + charger load shapes) | 1 w | 2030 growth narrative | C1 + AP RTO feed |

**Track C total, elapsed time:** 4 weeks once APEPDCL data lands.
**Strategic note:** C1 is what regulators actually consume. Category-level
MAPE is typically <3% even when individual meters are at 10%. Landing C1
gives us a parallel path to "MAPE <4%" that doesn't require fixing every
single meter — it's what CEA's own methodology assumes.

---

## Outcome gate

The stated outcome has three parts. We track against all three:

1. **< 4% MAPE per smart meter, all 42** — gated by A6. Publish only
   after rolling-origin proof. If we land at 40/42 or 41/42, publish
   honestly and scope the residual as a data-quality problem, not a
   modelling one.

2. **Low MAE** — not the headline KPI but track it in the manifest
   and report it in the v6 release doc. Should fall naturally as MAPE
   improves.

3. **Production-grade service** — Track B items B1+B2+B4+B5+B6 are
   table stakes. B3 is blocked on APEPDCL but we can ship a production
   service against CSV snapshots in the interim and add live ingest
   as a follow-on.

## Critical path

The fastest honest path to the outcome is:

- **Weeks 1-2:** A1 → (A2, A3, A4 in parallel); in parallel start B1, B4, B6
- **Weeks 3-4:** A5 if needed; B2, B5 land on top of B1
- **Week 5:** A6 rolling-origin validation; v6 release doc + tag
- **Weeks 6-9:** C1 starts (once APEPDCL data arrives; negotiate this
  in parallel starting week 1)
- **Week 10:** C2

Total elapsed: ~10 weeks to full outcome, with a shippable
"production-grade v5.s1.1 service with 35-40 meters under 4%" milestone
at week 5.

## What we will not do

- Rebuild on a different framework (PyTorch, XGBoost, DeepAR) unless A5
  fails — the framework isn't the bottleneck, the per-meter shape is.
- Add more weather features — diminishing returns after HDD/CDD.
- Promise <4% across all 42 before A6 clears it. The outcome is "honest
  MAPE we can defend to APEPDCL and CEA," not "a cherry-picked number on
  one holdout."

---

*Tracked as tasks #19–#33 in the task system. Last updated: 2026-04-21.*
