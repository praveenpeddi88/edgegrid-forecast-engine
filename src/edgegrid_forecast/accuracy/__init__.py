"""
edgegrid_forecast.accuracy
==========================

Block-wise forecast accuracy capture for the EdgeGrid v4/v5 forecasting
engine.

Cadences supported:
  * ``dam_15min``   -- 15-min IEX DAM blocks (96 blocks/day)
  * ``native_30min``-- 30-min model output (48 blocks/day)
  * ``hourly``      -- 60-min rollup (24 blocks/day)
  * ``tod_4h``      -- APEPDCL ToD windows (6 blocks/day; night/morning
                        peak/day/evening peak split into day and night
                        halves)

This module is pure pandas + numpy -- no LightGBM or model dependency --
so the accuracy capture system can be run on any (forecast, actual)
pair, including forecasts produced by v4_predict, v5_predict, the seasonal
anchor fallback, or an external benchmark.

The epsilon conventions here match the v4 engine:

    ``EPSILON_KWH = 0.0005`` (equivalent to ``0.5 Wh``, the threshold used by
    ``inference/_features.calc_metrics`` and ``training/holdout_benchmark``).

"""

from .block_accuracy import (
    BlockCadence,
    BlockSpec,
    BLOCK_SPECS,
    EPSILON_KWH,
    assign_blocks,
    assign_cohorts,
    broadcast_to_dam_15min,
    cohort_rollup,
    fleet_mape,
    per_meter_block_mape,
    load_oracle_cohort_map,
)

__all__ = [
    "BlockCadence",
    "BlockSpec",
    "BLOCK_SPECS",
    "EPSILON_KWH",
    "assign_blocks",
    "assign_cohorts",
    "broadcast_to_dam_15min",
    "cohort_rollup",
    "fleet_mape",
    "per_meter_block_mape",
    "load_oracle_cohort_map",
]
