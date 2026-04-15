"""
M1-F5: DG Transition Detection.

Detects diesel generator (DG) transitions in grid import data.
DG periods must be excluded from demand forecasting training data and DR baselines.
"""

from typing import Optional

import pandas as pd
from loguru import logger


#: DG event type string constants (simpler than Enum for serialization)
DG_EVENT_GRID_TO_DG = "grid_to_dg"
DG_EVENT_DG_TO_GRID = "dg_to_grid"
DG_EVENT_DG_RUNNING = "dg_running"


class DGTransitionDetector:
    """
    Detects diesel generator (DG) transitions in grid import data.

    Many C&I consumers in India have DG backup. When grid fails, grid import
    drops to near-zero but site load continues on DG. These periods MUST be
    excluded from demand forecasting training data and DR baselines.

    PRD detection logic:
    1. Grid import drops to <5% of rolling baseline within 1 interval
    2. Site load remains >50% of baseline (confirming DG is running)
    3. Voltage signature change: DG voltage is more variable (CV > 5%) than grid
    """

    def __init__(
        self,
        import_drop_threshold_pct: float = 5.0,
        load_continuation_threshold_pct: float = 50.0,
        baseline_window: str = "48h",
        voltage_variability_threshold: float = 0.05,
    ):
        """
        Args:
            import_drop_threshold_pct: Grid import must drop below this % of baseline to flag DG.
            load_continuation_threshold_pct: Site load must remain above this % to confirm DG.
            baseline_window: Rolling window for computing baseline grid import.
            voltage_variability_threshold: CV threshold above which voltage looks like DG.
        """
        if import_drop_threshold_pct <= 0 or import_drop_threshold_pct >= 100:
            raise ValueError(f"import_drop_threshold_pct must be in (0, 100), got {import_drop_threshold_pct}")

        self.import_drop_threshold_pct = import_drop_threshold_pct
        self.load_continuation_threshold_pct = load_continuation_threshold_pct
        self.baseline_window = baseline_window
        self.voltage_variability_threshold = voltage_variability_threshold

    def detect_grid_to_dg(
        self,
        grid_import: pd.Series,
        site_load: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Detect intervals where grid→DG switchover occurred.

        Primary: grid import drops below threshold % of rolling baseline.
        Confirmation (if site_load available): site load remains above threshold %.
        """
        baseline = grid_import.rolling(
            self.baseline_window, center=False, min_periods=10
        ).median()
        baseline = baseline.clip(lower=grid_import.quantile(0.1))

        import_ratio = grid_import / baseline
        dg_on = import_ratio < (self.import_drop_threshold_pct / 100)

        if site_load is not None:
            load_baseline = site_load.rolling(
                self.baseline_window, center=False, min_periods=10
            ).median()
            load_baseline = load_baseline.clip(lower=site_load.quantile(0.1))
            load_continues = (site_load / load_baseline) > (
                self.load_continuation_threshold_pct / 100
            )
            dg_on = dg_on & load_continues

        return dg_on

    def detect_voltage_signature(
        self,
        voltage: pd.Series,
        window: str = "30min",
    ) -> pd.Series:
        """
        Detect DG voltage signature — higher coefficient of variation than grid.

        Typical grid CV < 2%, DG CV > 5%.
        """
        rolling_std = voltage.rolling(window, center=True, min_periods=3).std()
        rolling_mean = voltage.rolling(window, center=True, min_periods=3).mean()

        safe_mean = rolling_mean.where(rolling_mean > 0)
        cv = rolling_std / safe_mean

        return (cv > self.voltage_variability_threshold).fillna(False)

    def mark_dg_periods(
        self,
        grid_import: pd.Series,
        site_load: Optional[pd.Series] = None,
        voltage: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Produce complete DG period annotation with transition labels and confidence.

        Returns:
            DataFrame with columns:
            - is_dg: boolean
            - dg_confidence: 'none', 'medium', or 'high'
            - dg_event_type: event label or None
            - dg_period_id: integer grouping consecutive DG intervals (-1 for non-DG)
        """
        dg_on = self.detect_grid_to_dg(grid_import, site_load)

        # Confidence: voltage confirmation upgrades medium → high
        dg_confidence = pd.Series("medium", index=grid_import.index)
        dg_confidence[~dg_on] = "none"

        if voltage is not None:
            dg_voltage = self.detect_voltage_signature(voltage)
            dg_confidence[dg_on & dg_voltage] = "high"

        result = pd.DataFrame(index=grid_import.index)
        result["is_dg"] = dg_on
        result["dg_confidence"] = dg_confidence
        result["dg_event_type"] = None

        # Period IDs for consecutive DG blocks
        transitions = dg_on.astype(int).diff().fillna(0)
        result["dg_period_id"] = (transitions == 1).cumsum()
        result.loc[~dg_on, "dg_period_id"] = -1

        # Label transitions
        grid_to_dg = transitions == 1
        dg_to_grid = transitions == -1

        result.loc[grid_to_dg, "dg_event_type"] = DG_EVENT_GRID_TO_DG
        result.loc[dg_to_grid, "dg_event_type"] = DG_EVENT_DG_TO_GRID
        result.loc[dg_on & ~grid_to_dg, "dg_event_type"] = DG_EVENT_DG_RUNNING

        n_periods = result.loc[dg_on, "dg_period_id"].nunique()
        n_intervals = dg_on.sum()
        if n_periods > 0:
            logger.info(
                f"DG periods detected: {n_periods} events, "
                f"{n_intervals} intervals ({n_intervals / len(grid_import) * 100:.2f}%)"
            )

        return result

    @staticmethod
    def exclude_dg_from_training(
        demand_df: pd.DataFrame,
        dg_mask: pd.Series,
    ) -> pd.DataFrame:
        """
        Remove DG periods from training data.

        DG periods show near-zero grid import — including them would train the
        model to predict demand drops that are actually supply-side events.
        """
        n_before = len(demand_df)
        clean = demand_df[~dg_mask].copy()
        n_removed = n_before - len(clean)

        if n_removed > 0:
            logger.info(
                f"Excluded {n_removed} DG intervals from training data "
                f"({n_removed / n_before * 100:.1f}%)"
            )

        return clean
