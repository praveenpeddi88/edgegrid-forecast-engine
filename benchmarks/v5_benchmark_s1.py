"""
v5 S1 benchmark — compare per-meter holdout MAPE of v4 vs v5 using each
bundle's chronological 15% validation slice.

Produces:
  prototypes/forecast_engine_v3/v5_benchmark_s1.json   (structured results)
  prototypes/forecast_engine_v3/v5_benchmark_s1.csv    (per-meter table)

Fleet summary printed to stdout:
    - mean/median/p90 MAPE by version
    - delta (v5 − v4) with cohort breakdown
    - per-tier improvement
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean, median

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def load_manifest(path: Path) -> dict[str, dict]:
    doc = json.loads(path.read_text())
    # v4 manifest: {version, models: [...]}, v5 manifest: [...]
    if isinstance(doc, dict) and "models" in doc:
        rows = doc["models"]
    elif isinstance(doc, list):
        rows = doc
    else:
        raise ValueError(f"Unexpected manifest shape at {path}")
    return {r["msn"]: r for r in rows}


def cohort(m: float) -> str:
    if m is None or np.isnan(m):
        return "X: invalid"
    if m < 4: return "A: <4% (target)"
    if m < 5: return "A': 4-5%"
    if m < 10: return "B: 5-10%"
    if m < 15: return "C: 10-15%"
    if m < 25: return "D: 15-25%"
    return "E: >=25%"


def main() -> int:
    v4_path = REPO / "models" / "v4" / "_manifest.json"
    v5_path = REPO / "models" / "v5" / "_manifest.json"
    if not v4_path.exists() or not v5_path.exists():
        print(f"ERR: missing manifest v4={v4_path.exists()} v5={v5_path.exists()}", file=sys.stderr)
        return 1

    v4 = load_manifest(v4_path)
    v5 = load_manifest(v5_path)
    common = sorted(set(v4) & set(v5))
    print(f"[s1-bench] v4 meters={len(v4)}  v5 meters={len(v5)}  common={len(common)}")

    rows = []
    for msn in common:
        r4, r5 = v4[msn], v5[msn]
        m4 = r4.get("holdout_mape")
        m5 = r5.get("holdout_mape")
        rows.append({
            "msn": msn,
            "tier": r5.get("tier"),
            "zero_pct": r5.get("zero_pct"),
            "v4_mape": m4,
            "v5_mape": m5,
            "delta_mape": (m5 - m4) if (m4 is not None and m5 is not None) else None,
            "v4_mae": r4.get("holdout_mae"),
            "v5_mae": r5.get("holdout_mae"),
            "v4_mbe": r4.get("holdout_mbe"),
            "v5_mbe": r5.get("holdout_mbe"),
            "v4_cohort": cohort(m4),
            "v5_cohort": cohort(m5),
            "v4_train_end": r4.get("train_end"),
            "v5_train_end": r5.get("train_end"),
        })

    # Fleet summary
    v4_mapes = [r["v4_mape"] for r in rows if r["v4_mape"] is not None]
    v5_mapes = [r["v5_mape"] for r in rows if r["v5_mape"] is not None]

    def stats(xs):
        a = np.array(xs, dtype=float)
        return {
            "n": len(a),
            "mean": round(float(a.mean()), 2),
            "median": round(float(np.median(a)), 2),
            "p90": round(float(np.percentile(a, 90)), 2),
            "max": round(float(a.max()), 2),
            "under_4pct": int((a < 4).sum()),
            "under_10pct": int((a < 10).sum()),
        }

    s4 = stats(v4_mapes)
    s5 = stats(v5_mapes)
    improved = sum(1 for r in rows if r["delta_mape"] is not None and r["delta_mape"] < -0.1)
    worsened = sum(1 for r in rows if r["delta_mape"] is not None and r["delta_mape"] > 0.1)

    print("\n═══════════════ S1 CHRONOLOGICAL (per-meter holdout) ═══════════════")
    print(f"  v4 stats: {s4}")
    print(f"  v5 stats: {s5}")
    print(f"  improved (v5 better by >0.1pp): {improved}/{len(rows)}")
    print(f"  worsened (v5 worse by >0.1pp):   {worsened}/{len(rows)}")

    from collections import Counter
    c4 = Counter(r["v4_cohort"] for r in rows)
    c5 = Counter(r["v5_cohort"] for r in rows)
    print("\n  Cohort distribution")
    print(f"    {'cohort':20s}  {'v4':>4s}   {'v5':>4s}   Δ")
    for k in sorted(set(c4) | set(c5)):
        print(f"    {k:20s}  {c4.get(k,0):4d}   {c5.get(k,0):4d}   {c5.get(k,0)-c4.get(k,0):+d}")

    # Per-tier breakdown
    tier_groups: dict[str, list[float]] = {}
    for r in rows:
        tier = r["tier"] or "Unknown"
        if r["delta_mape"] is not None:
            tier_groups.setdefault(tier, []).append(r["delta_mape"])
    print("\n  Per-tier improvement (mean Δ v5-v4, lower=better)")
    for tier, dvals in sorted(tier_groups.items()):
        arr = np.array(dvals)
        print(f"    {tier:25s}  n={len(dvals):2d}  Δ_mean={arr.mean():+6.2f}pp  Δ_median={np.median(arr):+6.2f}pp")

    # Top 10 biggest wins
    wins = sorted(
        [r for r in rows if r["delta_mape"] is not None],
        key=lambda r: r["delta_mape"],
    )[:10]
    print("\n  Top 10 biggest wins (v5 - v4):")
    for r in wins:
        print(f"    {r['msn']:10s} {r['tier']:20s}  v4={r['v4_mape']:6.2f}% → v5={r['v5_mape']:6.2f}% (Δ={r['delta_mape']:+.2f}pp)")

    regressions = sorted(
        [r for r in rows if r["delta_mape"] is not None and r["delta_mape"] > 0],
        key=lambda r: -r["delta_mape"],
    )[:10]
    if regressions:
        print("\n  Regressions (v5 worse than v4):")
        for r in regressions:
            print(f"    {r['msn']:10s} {r['tier']:20s}  v4={r['v4_mape']:6.2f}% → v5={r['v5_mape']:6.2f}% (Δ={r['delta_mape']:+.2f}pp)")
    else:
        print("\n  No regressions — v5 matches or beats v4 on every meter.")

    # Persist
    out_dir = REPO / "prototypes" / "forecast_engine_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "benchmark": "S1 chronological (15% holdout per meter)",
        "v4_fleet": s4,
        "v5_fleet": s5,
        "improved_count": improved,
        "worsened_count": worsened,
        "n_meters": len(rows),
        "cohort_v4": dict(c4),
        "cohort_v5": dict(c5),
        "per_meter": rows,
    }
    (out_dir / "v5_benchmark_s1.json").write_text(json.dumps(summary, indent=2, default=str))

    csv_path = out_dir / "v5_benchmark_s1.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n  → JSON: {(out_dir / 'v5_benchmark_s1.json').relative_to(REPO)}")
    print(f"  → CSV:  {csv_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
