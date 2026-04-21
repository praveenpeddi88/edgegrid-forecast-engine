"""
v5_predict — Recursive prediction wrapper for v4 LightGBM bundles.

The v4 inference path (`v4_predict.predict_with_context`) builds features ONCE
on the input context and predicts the entire horizon in a single batch call.
For any t past the cutoff, the lag features (lag_1, lag_24, etc.), rolling
features (rmean_6, rmean_336, ...), momentum features (demand_diff_*), and the
seasonal-deviation feature are all stale — they read from whatever was in
`demand_wh` at that position in the priming context (DOW-back values or zeros).

This is the dominant source of the gap between bundle-holdout MAPE (~7.15%
median across 42 meters) and v3 backtest MAPE (~46% median): the model is
being asked to predict t with t's ACTUAL prior values during training, but at
inference time it's seeing stale priming.

`predict_recursive` fixes this by stepping through the horizon one timestep at
a time:
  for each t in horizon:
      build features ending at t-1 (using actual context + already-emitted
        predictions for any t' < t that fall after the cutoff)
      predict y_hat[t]
      append y_hat[t] to the working context as if it were the actual

This is the *standard* autoregressive inference pattern for tree-based models
on time series. v4 trained on real lags; v5 inference now feeds it real (or
model-emitted) lags one step at a time so the input distribution matches the
training distribution.

Public API
----------
predict_recursive(msn, context_df, horizon_ts, *, weather_df=None,
                  fleet_df=None, models_dir=None) -> pd.DataFrame
    Same return shape as v4 predict_with_context, but built via recursive
    one-step-ahead inference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from ._features import (
    TIME_BLOCKS,
    block_label_for,
    build_features_v4_s1,
    compute_fleet_aggregate,
    fetch_weather_expanded,
)
from .v4_predict import load_model


# Max lookback in build_features_v4_s1 is rmean_30d = 1440 rows.
# We keep a generous 1700-row trailing window so expanding/groupby features
# (deviation_from_hourly, similar_day_k5) have enough history to be stable.
_TRAILING_WINDOW = 1700


def _build_one_row_features(
    bundle: dict,
    working_context: pd.DataFrame,
    weather_df: pd.DataFrame,
    fleet_df: pd.DataFrame,
    target_ts: pd.Timestamp,
) -> np.ndarray:
    """Build the v4 feature vector for ONE timestamp (target_ts), using
    working_context as the history. Returns a 1×F numpy array ready for
    bundle["model_mean"].predict().

    Optimization: only the last _TRAILING_WINDOW rows of working_context are
    needed because the deepest lag/rolling feature is rmean_30d (1440 rows).
    Building features on a 1700-row slice is ~20× faster than rebuilding on
    the full ~18k-row context for every horizon step.
    """
    # Slice to trailing window ending at the placeholder for target_ts.
    if len(working_context) > _TRAILING_WINDOW:
        wc = working_context.iloc[-_TRAILING_WINDOW:].reset_index(drop=True)
    else:
        wc = working_context

    feat_df, _ = build_features_v4_s1(
        wc, weather_df, fleet_df, drop_warmup=False
    )
    feat_df = feat_df.sort_values("ts").reset_index(drop=True)
    # Pick the row for target_ts
    row = feat_df[feat_df["ts"] == target_ts]
    if row.empty:
        # If target_ts not in working_context (shouldn't happen if caller
        # appends a placeholder), return zeros — caller will get a zero pred.
        return np.zeros((1, len(bundle["selected_features"])), dtype=np.float64)
    # Ensure all selected features are present
    for f in bundle["selected_features"]:
        if f not in row.columns:
            row = row.assign(**{f: 0.0})
    X = np.nan_to_num(
        row[bundle["selected_features"]].values.astype(np.float64), nan=0.0
    )
    return X  # shape (1, F)


def predict_recursive(
    msn: str,
    context_df: pd.DataFrame,
    horizon_ts: pd.DatetimeIndex,
    *,
    weather_df: Optional[pd.DataFrame] = None,
    fleet_df: Optional[pd.DataFrame] = None,
    models_dir: Optional[Path | str] = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Recursive (one-step-ahead) inference for a v4 bundle.

    Parameters
    ----------
    msn : meter serial number
    context_df : history through the cutoff. Must have columns ts, demand_wh,
        voltage. The last row's ts must be strictly less than horizon_ts[0].
    horizon_ts : the timestamps to forecast (strictly increasing 30-min grid
        starting after context_df['ts'].max()).
    weather_df : pre-built weather frame (covers context + horizon range).
        If None, fetch_weather_expanded() is called.
    fleet_df : pre-built fleet aggregate frame. If None, falls back to empty.
    models_dir : override default models/v4 location.
    progress : print a dot per 96 steps (every 2 days).

    Returns
    -------
    DataFrame indexed by ts with columns:
        forecast_wh, confidence_low, confidence_high,
        block_label, historical_block_mape
    """
    bundle = load_model(msn, models_dir=models_dir)
    if weather_df is None:
        weather_df = fetch_weather_expanded()
    if fleet_df is None:
        fleet_df = pd.DataFrame(columns=["ts", "fleet_mean", "fleet_std"])

    # Working context: real history + we'll grow it one row at a time
    work = context_df[["ts", "demand_wh", "voltage"]].copy()
    work = work.sort_values("ts").reset_index(drop=True)

    horizon_ts = pd.DatetimeIndex(pd.to_datetime(horizon_ts)).sort_values()

    use_log = bundle["use_log1p"]
    mbe = bundle["trail_mbe_capped"]
    last_voltage = float(work["voltage"].iloc[-1]) if len(work) else 230.0

    # Build a (date, hour, demand) lookup so we can DOW-prime the placeholder.
    # The placeholder's demand_wh value matters because `demand_diff_1` and
    # `demand_diff_4` use df["demand_wh"].diff(N) which reads the current
    # row. With placeholder=0, those features become huge negative numbers
    # that the model never saw at training time and predictions collapse.
    # Priming with the same-DOW-same-hour value from a week back keeps the
    # diff features small and within the model's training distribution.
    work_ts = work.set_index("ts")["demand_wh"]

    def _dow_prime(t: pd.Timestamp) -> float:
        # Walk back 7 days at a time until we find a value (handles short context)
        for k in (1, 2, 3, 4):
            cand = t - pd.Timedelta(days=7 * k)
            if cand in work_ts.index:
                return float(work_ts.loc[cand])
        # Fallback: use the most recent value
        return float(work_ts.iloc[-1])

    preds: list[float] = []
    q10s: list[float] = []
    q90s: list[float] = []

    for i, t in enumerate(horizon_ts):
        # Append a placeholder row for t so feature builder can compute the
        # feature vector AT t. demand_wh is set to the DOW-back value so that
        # current-row features (demand_diff_1/4) stay realistic; lag/rolling
        # features for this row read shift(N) of working context, which is
        # real-or-recursive — that's where the recursion lives.
        ph_demand = _dow_prime(t)
        placeholder = pd.DataFrame({
            "ts": [t],
            "demand_wh": [ph_demand],
            "voltage": [last_voltage],
        })
        wctx = pd.concat([work, placeholder], ignore_index=True)

        X = _build_one_row_features(bundle, wctx, weather_df, fleet_df, t)

        pt = bundle["model_mean"].predict(X)
        p = float(np.expm1(pt[0]) if use_log else pt[0])
        p = max(p - mbe, 0.0)
        q10 = float(max(bundle["model_q10"].predict(X)[0] - mbe, 0.0))
        q90 = float(max(bundle["model_q90"].predict(X)[0] - mbe, 0.0))
        # Monotonicity clamp matching v4_predict
        q10 = min(q10, p)
        q90 = max(q90, p)

        preds.append(p)
        q10s.append(q10)
        q90s.append(q90)

        # Append the prediction back as if it were actual; this is the recursion
        new_row = pd.DataFrame({
            "ts": [t], "demand_wh": [p], "voltage": [last_voltage],
        })
        work = pd.concat([work, new_row], ignore_index=True)
        # Keep the DOW-prime lookup up to date so long-horizon steps that
        # look back 7 days can pull from recent predictions (not stale zeros).
        work_ts.loc[t] = p

        if progress and (i + 1) % 96 == 0:
            print(".", end="", flush=True)

    block_labels = [block_label_for(t) for t in horizon_ts]
    hist = bundle["historical_block_mape"]
    hist_mape = [hist.get(b, float("nan")) for b in block_labels]

    out = pd.DataFrame({
        "forecast_wh": np.round(preds, 2),
        "confidence_low": np.round(q10s, 2),
        "confidence_high": np.round(q90s, 2),
        "block_label": block_labels,
        "historical_block_mape": hist_mape,
    }, index=pd.DatetimeIndex(horizon_ts, name="ts"))
    return out


__all__ = ["predict_recursive"]
