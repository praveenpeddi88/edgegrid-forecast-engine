"""
block_accuracy
==============

Block-wise forecast accuracy capture for the EdgeGrid v4/v5 forecasting
engine.

Public API
----------
* :func:`broadcast_to_dam_15min` -- split each 30-min row into two 15-min
  rows on energy basis (kWh / 2 each).
* :func:`assign_blocks` -- attach block-id columns for all four cadences.
* :func:`per_meter_block_mape` -- compute per-(meter, block) MAPE / MBE /
  signed-pct-error / abs-error in long format.
* :func:`cohort_rollup` -- aggregate block-MAPE by cohort (A/B/C/D).
* :func:`fleet_mape` -- aggregate block-MAPE across the fleet.
* :func:`assign_cohorts` and :func:`load_oracle_cohort_map` -- derive the
  A/B/C/D cohort map from the v3 oracle floor (bundle_mape thresholds).

Conventions
-----------
* All values are in **kWh** at the per-block level. The v4 engine stores
  raw demand in **Wh per 30-min block**; callers must divide by 1000
  before passing forecasts/actuals into this module.
* APE uses ``100 * |pred - act| / max(|act|, EPSILON_KWH)``.
* Blocks where ``actual_kwh < EPSILON_KWH`` are dropped entirely
  (zero-load blocks, where MAPE is undefined).
* The ``EPSILON_KWH = 0.0005`` value mirrors ``0.5 Wh`` used by the v4
  ``inference/_features.calc_metrics`` mask (`y_true > 0.5` Wh).

ToD encoding (APEPDCL)
----------------------
Per spec, the four APEPDCL bands are:

  * ``0`` Night          ``22:00 - 06:00``
  * ``1`` Morning peak   ``06:00 - 10:00``
  * ``2`` Day            ``10:00 - 18:00``
  * ``3`` Evening peak   ``18:00 - 22:00``

For the 6-blocks-per-day requirement, the night band is split into a
"late-night" (00:00-06:00) and "post-evening" (22:00-24:00) half so each
date has exactly six contiguous, non-overlapping ToD blocks. The day
band is split into "morning-day" (10:00-14:00) and "afternoon-day"
(14:00-18:00). The encoded integer values 0..5 are emitted as
``block_id_tod`` and span the day in clock order:

  * ``0`` Late-night       00:00 - 06:00
  * ``1`` Morning peak     06:00 - 10:00
  * ``2`` Morning-day      10:00 - 14:00
  * ``3`` Afternoon-day    14:00 - 18:00
  * ``4`` Evening peak     18:00 - 22:00
  * ``5`` Post-evening     22:00 - 24:00
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Mapping, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Epsilon (in kWh) below which an actual value is treated as zero-load and
#: the corresponding block is dropped from MAPE evaluation. Mirrors the
#: ``y_true > 0.5`` Wh convention used in v4 inference and training.
EPSILON_KWH: float = 0.0005


BlockCadence = Literal["dam_15min", "native_30min", "hourly", "tod_4h"]


@dataclass(frozen=True)
class BlockSpec:
    """Cadence specification for a block."""

    cadence: BlockCadence
    minutes_per_block: int
    blocks_per_day: int
    label: str


BLOCK_SPECS: Dict[BlockCadence, BlockSpec] = {
    "dam_15min":    BlockSpec("dam_15min", 15, 96, "DAM 15-min"),
    "native_30min": BlockSpec("native_30min", 30, 48, "Native 30-min"),
    "hourly":       BlockSpec("hourly", 60, 24, "Hourly"),
    "tod_4h":       BlockSpec("tod_4h", 240, 6, "ToD 4h"),
}


# Cohort thresholds on the oracle bundle_mape (from prototypes/forecast_engine_v3/oracle_floor.csv).
COHORT_THRESHOLDS = [
    ("A", -np.inf,  5.0),
    ("B",  5.0,    15.0),
    ("C", 15.0,    25.0),
    ("D", 25.0,    np.inf),
]


# ---------------------------------------------------------------------------
# Broadcast: 30-min -> 15-min DAM
# ---------------------------------------------------------------------------

def broadcast_to_dam_15min(
    df_30min: pd.DataFrame,
    value_col: str = "predicted_kwh",
) -> pd.DataFrame:
    """Take a 30-min forecast and produce a 15-min DAM-block table.

    For each 30-min row at timestamp ``T``, emit two 15-min rows at ``T``
    and ``T + 15min``, each holding ``value_col / 2`` (energy basis
    preserved -- the two 15-min blocks sum to the original 30-min value).

    Any column whose name ends in ``_kwh`` is split in half automatically.
    Other columns (meter_id, model_version, ...) are duplicated as-is.

    Parameters
    ----------
    df_30min
        Long-format forecast frame at native 30-min cadence with at least
        ``ts`` and ``value_col`` columns.
    value_col
        The primary energy column (default ``predicted_kwh``). Used only
        to validate presence; all ``*_kwh`` columns are halved.

    Returns
    -------
    DataFrame
        Long-format frame at 15-min cadence with twice as many rows.
    """
    if df_30min.empty:
        return df_30min.copy()
    if value_col not in df_30min.columns:
        raise KeyError(
            f"broadcast_to_dam_15min: missing required column '{value_col}'"
        )
    if "ts" not in df_30min.columns:
        raise KeyError("broadcast_to_dam_15min: missing required column 'ts'")

    d = df_30min.copy()
    d["ts"] = pd.to_datetime(d["ts"])

    a = d.copy()                                  # first 15-min sub-block
    b = d.copy()                                  # second 15-min sub-block
    b["ts"] = b["ts"] + pd.Timedelta(minutes=15)

    out = pd.concat([a, b], ignore_index=True)

    energy_cols = [c for c in out.columns if c.endswith("_kwh")]
    for col in energy_cols:
        out[col] = out[col].astype(float) / 2.0

    sort_keys = ["meter_id", "ts"] if "meter_id" in out.columns else ["ts"]
    out = out.sort_values(sort_keys).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Block ID assignment
# ---------------------------------------------------------------------------

def _tod_block(hour: int) -> int:
    """Return the 6-bucket ToD block id for a given hour-of-day (0..23)."""
    if hour < 6:
        return 0  # late-night
    if hour < 10:
        return 1  # morning peak
    if hour < 14:
        return 2  # morning-day
    if hour < 18:
        return 3  # afternoon-day
    if hour < 22:
        return 4  # evening peak
    return 5      # post-evening (22:00-24:00)


def assign_blocks(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    """Add ``block_id_15min``, ``block_id_30min``, ``block_id_hour`` and
    ``block_id_tod`` integer columns spanning ``0..blocks_per_day-1``.

    Each ``block_id_*`` is the slot index within the day (so the first
    15-min block of the day is 0, the last is 95).

    Parameters
    ----------
    df
        A frame with a timestamp column.
    ts_col
        Name of the timestamp column (default ``"ts"``).

    Returns
    -------
    DataFrame
        Copy of ``df`` with four extra integer columns appended.
    """
    out = df.copy()
    ts = pd.to_datetime(out[ts_col])
    h = ts.dt.hour
    m = ts.dt.minute

    # 15-min: 4 per hour, 96 per day
    out["block_id_15min"] = (h * 4 + m // 15).astype("int16")
    # 30-min: 2 per hour, 48 per day
    out["block_id_30min"] = (h * 2 + m // 30).astype("int16")
    # 60-min: hour-of-day
    out["block_id_hour"]  = h.astype("int16")
    # ToD: 6 per day per the encoding above
    out["block_id_tod"]   = h.map(_tod_block).astype("int16")
    return out


# ---------------------------------------------------------------------------
# Block-start timestamp helper
# ---------------------------------------------------------------------------

def _block_start_ts(ts: pd.Series, cadence: BlockCadence) -> pd.Series:
    """Floor ``ts`` to the start of the containing block for ``cadence``."""
    spec = BLOCK_SPECS[cadence]
    if cadence == "tod_4h":
        # ToD bands are not equal-length within day boundaries; floor to the
        # block start hour from the bucket id.
        ts = pd.to_datetime(ts)
        h = ts.dt.hour
        bucket = h.map(_tod_block)
        # Start hour for each bucket
        start_hour = bucket.map({0: 0, 1: 6, 2: 10, 3: 14, 4: 18, 5: 22})
        date = ts.dt.normalize()
        return date + pd.to_timedelta(start_hour, unit="h")
    freq = f"{spec.minutes_per_block}min"
    return pd.to_datetime(ts).dt.floor(freq)


# ---------------------------------------------------------------------------
# Per-meter block MAPE
# ---------------------------------------------------------------------------

def _aggregate_to_block(
    df: pd.DataFrame,
    cadence: BlockCadence,
    pred_col: str = "predicted_kwh",
    act_col: str = "actual_kwh",
) -> pd.DataFrame:
    """Aggregate a fine-cadence (forecast, actual) table to ``cadence`` by
    summing energy within each block."""
    d = df.copy()
    d["block_start_ts"] = _block_start_ts(d["ts"], cadence)
    grouped = (
        d.groupby(["meter_id", "block_start_ts"], as_index=False)
         .agg(predicted_kwh=(pred_col, "sum"),
              actual_kwh=(act_col, "sum"))
    )
    return grouped


def per_meter_block_mape(
    forecast_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    cadence: BlockCadence,
) -> pd.DataFrame:
    """Compute per-(meter, block) accuracy at the requested ``cadence``.

    Parameters
    ----------
    forecast_df
        Long-format forecast frame with columns ``ts``, ``meter_id``,
        ``predicted_kwh``. Cadence is implied: callers provide either a
        15-min frame (use as-is) or a 30-min frame (which is broadcast
        to 15-min internally when ``cadence == "dam_15min"``).
    actual_df
        Long-format actuals frame with columns ``ts``, ``meter_id``,
        ``actual_kwh``. Should be at the **same native cadence** as
        ``forecast_df`` for the join to be lossless. Native data here
        is 30-min (matching ``data/raw/{sp,tp}_data.parquet``).
    cadence
        Target block cadence. The function aggregates both frames into
        blocks of this cadence (summing energy within each block) and
        then computes MAPE per (meter, block).

    Returns
    -------
    DataFrame
        One row per (meter_id, block_id, block_start_ts) with columns:
        ``predicted_kwh``, ``actual_kwh``, ``abs_error_kwh``,
        ``ape``, ``mbe_kwh``, ``signed_pct_error``, ``cadence``,
        ``block_id``.

        Blocks where ``actual_kwh < EPSILON_KWH`` are dropped.
    """
    spec = BLOCK_SPECS[cadence]

    # ---- normalise inputs ----
    if forecast_df.empty or actual_df.empty:
        return pd.DataFrame(
            columns=[
                "meter_id", "block_id", "block_start_ts",
                "predicted_kwh", "actual_kwh", "abs_error_kwh",
                "ape", "mbe_kwh", "signed_pct_error", "cadence",
            ]
        )

    fc = forecast_df[["ts", "meter_id", "predicted_kwh"]].copy()
    ac = actual_df[["ts", "meter_id", "actual_kwh"]].copy()
    fc["ts"] = pd.to_datetime(fc["ts"])
    ac["ts"] = pd.to_datetime(ac["ts"])
    fc["meter_id"] = fc["meter_id"].astype(str)
    ac["meter_id"] = ac["meter_id"].astype(str)

    # ---- detect forecast cadence to decide on broadcast ----
    # If forecast is 30-min and target is dam_15min, broadcast first so each
    # 15-min block gets a halved-kWh value.
    if cadence == "dam_15min":
        # Heuristic: if median delta between consecutive ts within a meter is
        # 30 minutes, broadcast.
        sample = fc.sort_values(["meter_id", "ts"])
        deltas = sample.groupby("meter_id")["ts"].diff().dropna()
        if not deltas.empty:
            median_min = float(deltas.median().total_seconds() / 60.0)
            if abs(median_min - 30.0) < 1.0:
                fc = broadcast_to_dam_15min(fc, value_col="predicted_kwh")
            elif abs(median_min - 15.0) < 1.0:
                pass  # already 15-min, use as-is
        # Actuals: native is 30-min. To produce 15-min ground truth,
        # we have to split actual energy in half too (same energy basis).
        a_deltas = ac.sort_values(["meter_id", "ts"]).groupby("meter_id")["ts"].diff().dropna()
        if not a_deltas.empty and abs(float(a_deltas.median().total_seconds() / 60.0) - 30.0) < 1.0:
            tmp = ac.rename(columns={"actual_kwh": "actual_kwh"}).copy()
            tmp["actual_kwh"] = tmp["actual_kwh"]
            # broadcast helper expects predicted_kwh; do it inline.
            a = tmp.copy()
            b = tmp.copy()
            b["ts"] = b["ts"] + pd.Timedelta(minutes=15)
            ac = pd.concat([a, b], ignore_index=True)
            ac["actual_kwh"] = ac["actual_kwh"].astype(float) / 2.0

    # ---- merge to a (meter, ts) table at the finest available cadence ----
    merged = pd.merge(
        fc, ac,
        on=["meter_id", "ts"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "meter_id", "block_id", "block_start_ts",
                "predicted_kwh", "actual_kwh", "abs_error_kwh",
                "ape", "mbe_kwh", "signed_pct_error", "cadence",
            ]
        )

    # ---- aggregate to target block cadence (sum kWh within each block) ----
    blk = _aggregate_to_block(merged, cadence)

    # ---- attach block_id (slot-of-day) ----
    blk = assign_blocks(blk, ts_col="block_start_ts")
    block_id_col = {
        "dam_15min":    "block_id_15min",
        "native_30min": "block_id_30min",
        "hourly":       "block_id_hour",
        "tod_4h":       "block_id_tod",
    }[cadence]
    blk["block_id"] = blk[block_id_col]

    # ---- compute errors ----
    pred = blk["predicted_kwh"].to_numpy(dtype=np.float64)
    act = blk["actual_kwh"].to_numpy(dtype=np.float64)
    err = pred - act
    blk["abs_error_kwh"] = np.abs(err)
    blk["mbe_kwh"] = err  # signed bias per block

    # ---- drop zero-load blocks ----
    keep = act >= EPSILON_KWH
    blk = blk.loc[keep].copy()
    if blk.empty:
        return pd.DataFrame(
            columns=[
                "meter_id", "block_id", "block_start_ts",
                "predicted_kwh", "actual_kwh", "abs_error_kwh",
                "ape", "mbe_kwh", "signed_pct_error", "cadence",
            ]
        )

    abs_act = np.maximum(np.abs(blk["actual_kwh"].to_numpy(dtype=np.float64)),
                         EPSILON_KWH)
    blk["ape"] = 100.0 * blk["abs_error_kwh"].to_numpy(dtype=np.float64) / abs_act
    blk["signed_pct_error"] = 100.0 * blk["mbe_kwh"].to_numpy(dtype=np.float64) / abs_act
    blk["cadence"] = cadence

    out_cols = [
        "meter_id", "block_id", "block_start_ts",
        "predicted_kwh", "actual_kwh", "abs_error_kwh",
        "ape", "mbe_kwh", "signed_pct_error", "cadence",
    ]
    return blk[out_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cohort assignment + rollup
# ---------------------------------------------------------------------------

def assign_cohorts(bundle_mape_by_meter: Mapping[str, float]) -> Dict[str, str]:
    """Map each meter to a cohort A/B/C/D from its v3 oracle ``bundle_mape``.

    Cohort thresholds:

      * A: bundle_mape  < 5%
      * B: 5  <= bundle_mape < 15%
      * C: 15 <= bundle_mape < 25%
      * D: bundle_mape >= 25%
    """
    cmap: Dict[str, str] = {}
    for msn, val in bundle_mape_by_meter.items():
        v = float(val)
        for name, lo, hi in COHORT_THRESHOLDS:
            if lo <= v < hi:
                cmap[str(msn)] = name
                break
        else:
            cmap[str(msn)] = "D"
    return cmap


def load_oracle_cohort_map(
    oracle_csv_path: Path | str | None = None,
) -> Dict[str, str]:
    """Read the v3 oracle floor CSV and return ``{msn -> cohort}``.

    If ``oracle_csv_path`` is None, defaults to
    ``prototypes/forecast_engine_v3/oracle_floor.csv`` relative to the
    repo root (resolved via this file's location).
    """
    if oracle_csv_path is None:
        repo_root = Path(__file__).resolve().parents[3]
        oracle_csv_path = repo_root / "prototypes" / "forecast_engine_v3" / "oracle_floor.csv"
    df = pd.read_csv(oracle_csv_path)
    df["msn"] = df["msn"].astype(str)
    return assign_cohorts(dict(zip(df["msn"], df["bundle_mape"].astype(float))))


def cohort_rollup(
    block_mape_df: pd.DataFrame,
    cohort_map: Mapping[str, str],
) -> pd.DataFrame:
    """Aggregate block MAPE by cohort.

    Parameters
    ----------
    block_mape_df
        Output of :func:`per_meter_block_mape` (single cadence) or its
        concatenation across cadences.
    cohort_map
        ``{meter_id -> 'A'|'B'|'C'|'D'}``. Meters not present in the map
        are dropped from the rollup.

    Returns
    -------
    DataFrame
        One row per (cohort, cadence) with columns: ``mean_mape``,
        ``median_mape``, ``p95_mape``, ``mbe_kwh``, ``n_blocks``,
        ``n_meters``.
    """
    if block_mape_df.empty:
        return pd.DataFrame(
            columns=["cohort", "cadence", "mean_mape", "median_mape",
                     "p95_mape", "mbe_kwh", "n_blocks", "n_meters"]
        )

    df = block_mape_df.copy()
    df["meter_id"] = df["meter_id"].astype(str)
    df["cohort"] = df["meter_id"].map(cohort_map)
    df = df.dropna(subset=["cohort"])
    if df.empty:
        return pd.DataFrame(
            columns=["cohort", "cadence", "mean_mape", "median_mape",
                     "p95_mape", "mbe_kwh", "n_blocks", "n_meters"]
        )

    rows = []
    group_cols = ["cohort", "cadence"] if "cadence" in df.columns else ["cohort"]
    for keys, g in df.groupby(group_cols):
        if isinstance(keys, tuple):
            cohort, cadence = keys[0], keys[1]
        else:
            cohort = keys
            cadence = g["cadence"].iloc[0] if "cadence" in g.columns else None
        rows.append({
            "cohort": cohort,
            "cadence": cadence,
            "mean_mape":   float(g["ape"].mean()),
            "median_mape": float(g["ape"].median()),
            "p95_mape":    float(np.percentile(g["ape"].to_numpy(), 95)),
            "mbe_kwh":     float(g["mbe_kwh"].mean()),
            "n_blocks":    int(len(g)),
            "n_meters":    int(g["meter_id"].nunique()),
        })
    return pd.DataFrame(rows).sort_values(["cadence", "cohort"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fleet aggregate
# ---------------------------------------------------------------------------

def fleet_mape(block_mape_df: pd.DataFrame) -> Dict[str, float | int]:
    """Fleet-level aggregate MAPE across all (meter, block) rows.

    Returns a dict with: ``mean_mape``, ``median_mape``, ``p25_mape``,
    ``p75_mape``, ``p95_mape``, ``max_mape``, ``mbe_kwh``, ``n_blocks``,
    ``n_meters``. If the input frame is empty, all numeric values are NaN.
    """
    if block_mape_df.empty:
        return {
            "mean_mape": float("nan"),
            "median_mape": float("nan"),
            "p25_mape": float("nan"),
            "p75_mape": float("nan"),
            "p95_mape": float("nan"),
            "max_mape": float("nan"),
            "mbe_kwh": float("nan"),
            "n_blocks": 0,
            "n_meters": 0,
        }
    ape = block_mape_df["ape"].to_numpy(dtype=np.float64)
    return {
        "mean_mape":   float(np.mean(ape)),
        "median_mape": float(np.median(ape)),
        "p25_mape":    float(np.percentile(ape, 25)),
        "p75_mape":    float(np.percentile(ape, 75)),
        "p95_mape":    float(np.percentile(ape, 95)),
        "max_mape":    float(np.max(ape)),
        "mbe_kwh":     float(np.mean(block_mape_df["mbe_kwh"].to_numpy())),
        "n_blocks":    int(len(block_mape_df)),
        "n_meters":    int(block_mape_df["meter_id"].nunique()),
    }
