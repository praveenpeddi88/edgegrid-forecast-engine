"""
Foundation model forecasters for zero-shot time series prediction.

These models are pretrained on millions of time series and can forecast
WITHOUT any training data. They serve as:
1. Instant baselines — deploy day one, no meter data needed
2. Ensemble members — combine with LightGBM for robust forecasts
3. Cold-start solution — new consumers get forecasts immediately

Models:
- Chronos-Bolt-Tiny (Amazon, 9M params): Fast, 2048-token context, CPU-friendly
- Chronos-Bolt-Small (Amazon, 48M params): Better accuracy, needs more RAM
- TimesFM 2.5 (Google): Coming soon — requires different install

Benchmark (synthetic data, 168h ahead, zero-shot):
  Chronos-Bolt-Tiny: 8.9% MAPE avg across 6 consumers
  Naive persistence:  10.9% MAPE avg
  → Chronos beats naive by 2.0pp with zero training
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class FoundationForecastResult:
    """Container for foundation model forecast output."""
    timestamp: pd.DatetimeIndex
    point_forecast: np.ndarray       # Median prediction
    lower_bound: np.ndarray          # P10
    upper_bound: np.ndarray          # P90
    model_name: str
    consumer_id: str
    context_length: int
    prediction_length: int
    metrics: Dict[str, float] = field(default_factory=dict)


class ChronosBoltForecaster:
    """
    Amazon Chronos-Bolt for zero-shot time series forecasting.

    Key properties:
    - Context window: 2048 tokens (= hours for hourly data)
    - Native prediction: up to 64 steps (quality degrades beyond)
    - For longer horizons: autoregressive rollout (feed predictions back)
    - Quantile outputs: P10, P50, P90 for uncertainty
    - CPU inference: ~0.2s per consumer per 168h forecast
    """

    AVAILABLE_MODELS = {
        "tiny": "amazon/chronos-bolt-tiny",      # 9M params, fastest
        "mini": "amazon/chronos-bolt-mini",       # 21M params
        "small": "amazon/chronos-bolt-small",     # 48M params, best accuracy
    }

    def __init__(self, model_size: str = "tiny"):
        if model_size not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model size must be one of {list(self.AVAILABLE_MODELS.keys())}")

        self.model_size = model_size
        self.model_name = self.AVAILABLE_MODELS[model_size]
        self.pipeline = None
        self.max_context = None
        self.native_horizon = None

    def load(self):
        """Load the pretrained model. Call once, reuse for all consumers."""
        import torch
        from chronos import BaseChronosPipeline

        logger.info(f"Loading Chronos-Bolt-{self.model_size} ({self.model_name})...")
        self.pipeline = BaseChronosPipeline.from_pretrained(
            self.model_name,
            device_map="cpu",
            dtype=torch.float32,
        )
        self.max_context = self.pipeline.model_context_length  # 2048
        self.native_horizon = self.pipeline.model_prediction_length  # 64
        logger.info(
            f"  Loaded. Context: {self.max_context}, "
            f"Native horizon: {self.native_horizon}"
        )

    def predict(
        self,
        series: np.ndarray,
        prediction_length: int = 168,
        quantile_levels: List[float] = [0.1, 0.5, 0.9],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate zero-shot forecast with uncertainty bounds.

        For horizons > native (64), uses autoregressive rollout:
        predict 64 steps, append to context, predict next 64, etc.

        Args:
            series: Historical time series (1D array)
            prediction_length: How many steps ahead to forecast
            quantile_levels: Quantiles for uncertainty bounds

        Returns:
            (median, lower, upper) — each shape (prediction_length,)
        """
        import torch

        if self.pipeline is None:
            self.load()

        # Truncate context to max window
        context = series[-self.max_context:]
        ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

        if prediction_length <= self.native_horizon:
            # Single-shot prediction
            quantiles, _ = self.pipeline.predict_quantiles(
                inputs=ctx_tensor,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
            )
            q = quantiles[0].numpy()  # (horizon, n_quantiles)
            return q[:, 1], q[:, 0], q[:, 2]  # median, lower, upper

        # Autoregressive rollout for longer horizons
        all_medians = []
        all_lowers = []
        all_uppers = []
        remaining = prediction_length
        current_context = context.copy()

        while remaining > 0:
            chunk = min(remaining, self.native_horizon)
            ctx_tensor = torch.tensor(
                current_context[-self.max_context:], dtype=torch.float32
            ).unsqueeze(0)

            quantiles, _ = self.pipeline.predict_quantiles(
                inputs=ctx_tensor,
                prediction_length=chunk,
                quantile_levels=quantile_levels,
            )
            q = quantiles[0].numpy()

            all_medians.append(q[:, 1])
            all_lowers.append(q[:, 0])
            all_uppers.append(q[:, 2])

            # Feed median predictions back as context
            current_context = np.concatenate([current_context, q[:, 1]])
            remaining -= chunk

        return (
            np.concatenate(all_medians)[:prediction_length],
            np.concatenate(all_lowers)[:prediction_length],
            np.concatenate(all_uppers)[:prediction_length],
        )

    def forecast_consumer(
        self,
        demand_series: pd.DataFrame,
        consumer_id: str,
        prediction_length: int = 168,
        value_col: str = "demand_kwh",
        timestamp_col: str = "timestamp",
    ) -> FoundationForecastResult:
        """
        Forecast for a single consumer.

        Args:
            demand_series: DataFrame with timestamp + value columns
            consumer_id: Consumer identifier
            prediction_length: Hours ahead to forecast
        """
        series = demand_series[value_col].values
        timestamps = pd.to_datetime(demand_series[timestamp_col])

        median, lower, upper = self.predict(series, prediction_length)

        # Generate future timestamps
        last_ts = timestamps.iloc[-1]
        future_ts = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1),
            periods=prediction_length,
            freq="h",
        )

        return FoundationForecastResult(
            timestamp=future_ts,
            point_forecast=median,
            lower_bound=lower,
            upper_bound=upper,
            model_name=f"chronos-bolt-{self.model_size}",
            consumer_id=consumer_id,
            context_length=min(len(series), self.max_context),
            prediction_length=prediction_length,
        )

    def benchmark(
        self,
        demand_df: pd.DataFrame,
        prediction_length: int = 168,
        value_col: str = "demand_kwh",
        consumer_col: str = "consumer_id",
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Benchmark zero-shot performance across all consumers.

        Uses last `prediction_length` hours as test set,
        everything before as context.

        Returns DataFrame with per-consumer metrics.
        """
        from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

        if self.pipeline is None:
            self.load()

        results = []

        for cid in sorted(demand_df[consumer_col].unique()):
            cdf = demand_df[demand_df[consumer_col] == cid].sort_values(timestamp_col)
            values = cdf[value_col].values

            # Split: context | test
            context = values[:-prediction_length]
            actual = values[-prediction_length:]

            # Naive baseline: repeat last week
            naive = values[-(prediction_length * 2):-prediction_length]

            # Chronos prediction
            median, lower, upper = self.predict(context, prediction_length)

            # Metrics
            mask = actual > 0
            c_mape = np.mean(np.abs((actual[mask] - median[mask]) / actual[mask])) * 100
            c_mae = np.mean(np.abs(actual[mask] - median[mask]))
            n_mape = np.mean(np.abs((actual[mask] - naive[mask]) / actual[mask])) * 100
            n_mae = np.mean(np.abs(actual[mask] - naive[mask]))

            # Coverage: % of actual values within P10-P90
            coverage = np.mean((actual >= lower) & (actual <= upper)) * 100

            results.append({
                "consumer_id": cid,
                "chronos_mape": round(c_mape, 2),
                "chronos_mae": round(c_mae, 1),
                "naive_mape": round(n_mape, 2),
                "naive_mae": round(n_mae, 1),
                "mape_improvement_pp": round(n_mape - c_mape, 2),
                "p10_p90_coverage": round(coverage, 1),
            })

        return pd.DataFrame(results)
