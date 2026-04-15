"""
M1-F6: APFC Switching Event Detection.

Detects Automatic Power Factor Correction (APFC) switching events that
mimic demand curtailment. Must be excluded from DR baselines.
"""

from typing import Tuple

import pandas as pd
from loguru import logger

from ._constants import MIN_PF_JUMP


#: APFC event type string constants
APFC_EVENT_SWITCH_IN = "cap_switch_in"    # Capacitor bank engaged → PF improves, kVA drops
APFC_EVENT_SWITCH_OUT = "cap_switch_out"  # Capacitor bank disengaged → PF drops, kVA rises


class APFCSwitchingDetector:
    """
    Detects Automatic Power Factor Correction (APFC) switching events.

    APFC panels switch capacitor banks to maintain PF near 0.95-1.0.
    Each switch causes a step change in kVAR/kVA without changing kW.

    PRD detection logic:
    1. kVA step change > threshold within 1 interval
    2. kW change < σ_threshold × rolling σ (kW is stable)
    3. PF jumps toward 0.95-1.0 (switch-in) or away (switch-out)
    4. All three conditions must coincide
    """

    def __init__(
        self,
        kva_step_threshold: float = 50.0,
        kw_sigma_threshold: float = 1.0,
        target_pf_range: Tuple[float, float] = (0.95, 1.0),
        baseline_window: str = "48h",
    ):
        """
        Args:
            kva_step_threshold: Minimum kVA change (absolute) to trigger step detection.
            kw_sigma_threshold: Maximum kW change (in σ units) for "stable kW" confirmation.
            target_pf_range: PF range that APFC panels target (switch-in moves PF toward this).
            baseline_window: Rolling window for kW σ computation.
        """
        if kva_step_threshold <= 0:
            raise ValueError(f"kva_step_threshold must be positive, got {kva_step_threshold}")
        if kw_sigma_threshold <= 0:
            raise ValueError(f"kw_sigma_threshold must be positive, got {kw_sigma_threshold}")
        if not (0 <= target_pf_range[0] < target_pf_range[1] <= 1.0):
            raise ValueError(f"target_pf_range must be valid PF range, got {target_pf_range}")

        self.kva_step_threshold = kva_step_threshold
        self.kw_sigma_threshold = kw_sigma_threshold
        self.target_pf_range = target_pf_range
        self.baseline_window = baseline_window

    def detect_kva_step(self, kva: pd.Series) -> pd.Series:
        """Detect intervals with sudden kVA step change (|Δ| > threshold)."""
        kva_diff = kva.diff().abs()
        return kva_diff > self.kva_step_threshold

    def detect_stable_kw(self, kw: pd.Series, kva_step_mask: pd.Series) -> pd.Series:
        """
        Confirm kW is stable during kVA step events.

        Distinguishes APFC events (kVA changes, kW doesn't) from real
        load changes (both kVA and kW change).
        """
        kw_diff = kw.diff().abs()
        kw_sigma = kw.rolling(self.baseline_window, center=False, min_periods=10).std()
        safe_sigma = kw_sigma.where(kw_sigma > 0)

        kw_stable = (kw_diff / safe_sigma < self.kw_sigma_threshold).fillna(False)
        return kw_stable & kva_step_mask

    def detect_pf_jump(
        self,
        pf: pd.Series,
        kva_step_mask: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Detect PF jumps toward or away from target range during kVA steps.

        Returns (switch_in_mask, switch_out_mask).
        """
        pf_diff = pf.diff()
        pf_prev = pf.shift(1)

        # Switch-in: PF moves closer to target (improves)
        was_below_target = pf_prev < self.target_pf_range[0]
        moved_toward = pf_diff > MIN_PF_JUMP
        switch_in = kva_step_mask & was_below_target & moved_toward

        # Switch-out: PF moves away from target (drops)
        was_in_target = (pf_prev >= self.target_pf_range[0]) & (pf_prev <= self.target_pf_range[1])
        moved_away = pf_diff < -MIN_PF_JUMP
        switch_out = kva_step_mask & was_in_target & moved_away

        return switch_in, switch_out

    def classify_apfc_events(
        self,
        kva: pd.Series,
        kw: pd.Series,
        pf: pd.Series,
    ) -> pd.DataFrame:
        """
        Full APFC event classification pipeline.

        Returns DataFrame with: is_apfc_event, apfc_event_type, kva_step_magnitude.
        """
        kva_step = self.detect_kva_step(kva)
        kw_stable = self.detect_stable_kw(kw, kva_step)
        switch_in, switch_out = self.detect_pf_jump(pf, kw_stable)

        result = pd.DataFrame(index=kva.index)
        result["is_apfc_event"] = switch_in | switch_out
        result["apfc_event_type"] = None
        result.loc[switch_in, "apfc_event_type"] = APFC_EVENT_SWITCH_IN
        result.loc[switch_out, "apfc_event_type"] = APFC_EVENT_SWITCH_OUT
        result["kva_step_magnitude"] = kva.diff().abs()
        result.loc[~result["is_apfc_event"], "kva_step_magnitude"] = 0

        n_events = result["is_apfc_event"].sum()
        if n_events > 0:
            logger.info(
                f"APFC events detected: {n_events} total "
                f"({switch_in.sum()} switch-in, {switch_out.sum()} switch-out)"
            )

        return result

    @staticmethod
    def normalize_for_dr_baseline(
        kva: pd.Series,
        kw: pd.Series,
        apfc_events: pd.DataFrame,
    ) -> pd.Series:
        """
        Normalize kVA to remove APFC effects for DR baseline computation.

        At APFC switch-in: kVA drop is artificial (real power unchanged).
        Replace post-switch kVA with kW / pre-switch PF.
        """
        normalized = kva.copy()

        event_mask = apfc_events["is_apfc_event"]
        if not event_mask.any():
            return normalized

        # Vectorized: pre-event PF → correction
        prev_kva = kva.shift(1)
        prev_kw = kw.shift(1)
        prev_pf = prev_kw / prev_kva

        correctable = event_mask & (prev_kva > 0) & (prev_pf > 0) & prev_kva.notna()
        normalized[correctable] = kw[correctable] / prev_pf[correctable]

        return normalized
