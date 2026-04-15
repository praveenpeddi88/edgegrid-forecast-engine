"""
M1-F4: Demand Signal Noise Filter.

Filters CT metering artefacts and PF transients from Indian substation demand data.
Indian substations experience 2-4 Hz frequency swings that corrupt CT readings.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ._constants import CT_ARTEFACT_SIGMA_ELEVATION, GRID_FREQ_BAND


class DemandNoiseFilter:
    """
    Filters CT metering artefacts from Indian substation demand data.

    PRD detection logic:
    1. Rolling median on 15-min kVA (window = 5 intervals = 75 min)
    2. Deviation > 3σ from 48h rolling baseline AND frequency outside 49.5-50.5 Hz
    3. Secondary: if kW stable but kVA spikes → PF artefact, not real load
    """

    def __init__(
        self,
        rolling_window: str = "75min",
        baseline_window: str = "48h",
        sigma_threshold: float = 3.0,
        freq_band: Tuple[float, float] = GRID_FREQ_BAND,
    ):
        """
        Args:
            rolling_window: Short rolling window for local median (5 intervals = 75min).
            baseline_window: Long rolling window for baseline σ computation (48h).
            sigma_threshold: Number of σ above baseline to flag as artefact.
            freq_band: Normal grid frequency band (Hz). Outside = frequency instability.
        """
        if sigma_threshold <= 0:
            raise ValueError(f"sigma_threshold must be positive, got {sigma_threshold}")

        self.rolling_window = rolling_window
        self.baseline_window = baseline_window
        self.sigma_threshold = sigma_threshold
        self.freq_band = freq_band

        # Cached rolling computations (populated on first use)
        self._cached_kva_median: Optional[pd.Series] = None
        self._cached_kva_sigma: Optional[pd.Series] = None
        self._cache_key: Optional[int] = None  # id() of the kva Series

    def _get_rolling_baseline(
        self,
        kva_series: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute (or return cached) rolling median and σ of kVA readings.
        """
        cache_key = id(kva_series)
        if self._cache_key == cache_key and self._cached_kva_median is not None:
            return self._cached_kva_median, self._cached_kva_sigma

        rolling_median = kva_series.rolling(
            self.baseline_window, center=False, min_periods=10
        ).median()
        rolling_sigma = kva_series.rolling(
            self.baseline_window, center=False, min_periods=10
        ).std()

        self._cached_kva_median = rolling_median
        self._cached_kva_sigma = rolling_sigma
        self._cache_key = cache_key

        return rolling_median, rolling_sigma

    def compute_rolling_baseline(
        self,
        kva_series: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute rolling median and σ of kVA readings over the baseline window.

        Public API — delegates to cached internal method.
        """
        return self._get_rolling_baseline(kva_series)

    def detect_ct_artefacts(
        self,
        kva: pd.Series,
        kw: pd.Series,
        frequency: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Detect CT metering artefacts using rolling baseline + frequency correlation.

        Flagged if: kVA deviation > sigma_threshold × σ AND frequency outside normal band.
        Without frequency data, uses elevated threshold to reduce false positives.
        """
        rolling_median, rolling_sigma = self._get_rolling_baseline(kva)

        safe_sigma = rolling_sigma.where(rolling_sigma > 0)
        deviation = (kva - rolling_median).abs() / safe_sigma
        exceeds_sigma = deviation > self.sigma_threshold

        if frequency is not None and len(frequency) > 0:
            freq_abnormal = (frequency < self.freq_band[0]) | (frequency > self.freq_band[1])
            ct_artefact = exceeds_sigma & freq_abnormal
        else:
            ct_artefact = deviation > (self.sigma_threshold + CT_ARTEFACT_SIGMA_ELEVATION)
            logger.debug("No frequency data — using elevated threshold for CT artefact detection")

        ct_artefact = ct_artefact.fillna(False)
        n_flagged = ct_artefact.sum()
        if n_flagged > 0:
            logger.info(f"CT artefacts detected: {n_flagged} intervals ({n_flagged / len(kva) * 100:.2f}%)")

        return ct_artefact

    def detect_pf_artefacts(
        self,
        kva: pd.Series,
        kw: pd.Series,
        pf: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Detect power factor artefacts: kVA spike with stable kW = PF transient.
        """
        kva_median = kva.rolling(self.rolling_window, center=True, min_periods=3).median()
        _, kva_sigma = self._get_rolling_baseline(kva)

        safe_kva_sigma = kva_sigma.where(kva_sigma > 0)
        kva_spike = (kva - kva_median).abs() / safe_kva_sigma > self.sigma_threshold

        kw_sigma = kw.rolling(self.baseline_window, center=False, min_periods=10).std()
        safe_kw_sigma = kw_sigma.where(kw_sigma > 0)
        kw_median = kw.rolling(self.rolling_window, center=True, min_periods=3).median()
        kw_stable = (kw - kw_median).abs() / safe_kw_sigma < 1.0

        pf_artefact = (kva_spike & kw_stable).fillna(False)

        n_flagged = pf_artefact.sum()
        if n_flagged > 0:
            logger.info(f"PF artefacts detected: {n_flagged} intervals ({n_flagged / len(kva) * 100:.2f}%)")

        return pf_artefact

    def clean_demand_signal(
        self,
        kva: pd.Series,
        kw: pd.Series,
        frequency: Optional[pd.Series] = None,
        pf: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Clean demand signal by replacing artefacts with rolling median.

        Returns:
            Tuple of (cleaned_kva, artefact_mask)
        """
        ct_mask = self.detect_ct_artefacts(kva, kw, frequency)
        pf_mask = self.detect_pf_artefacts(kva, kw, pf)

        artefact_mask = ct_mask | pf_mask

        # Reuse the cached short-window rolling median for replacement values
        kva_median = kva.rolling(self.rolling_window, center=True, min_periods=3).median()
        cleaned = kva.copy()
        cleaned[artefact_mask] = kva_median[artefact_mask]
        cleaned = cleaned.ffill().bfill()

        total_cleaned = artefact_mask.sum()
        if total_cleaned > 0:
            logger.info(
                f"Demand signal cleaned: {total_cleaned} intervals replaced "
                f"({total_cleaned / len(kva) * 100:.2f}%)"
            )

        return cleaned, artefact_mask
