"""EdgeGrid inference package — production wrappers around trained forecasting models.

v4_predict: Strategy 1 v4 (Chronological Cutoff) — per-meter LightGBM with
two-pass feature selection, tier-adaptive params, trailing-MBE bias correction,
and q10/q90 quantile confidence intervals.
"""

from .v4_predict import (
    train_and_persist,
    load_model,
    predict,
    predict_with_context,
    MODEL_VERSION,
)

__all__ = [
    "train_and_persist",
    "load_model",
    "predict",
    "predict_with_context",
    "MODEL_VERSION",
]
