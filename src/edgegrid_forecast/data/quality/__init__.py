"""
M1 — Data Quality Engine for Indian distribution grid data.

This is a subpackage with focused modules per M1 feature:
- ami: Smart meter ingestion (gaps, duplicates, sync, quality scoring)
- anomaly: Statistical outlier detection (frozen, z-score, IQR, contextual, rolling, IF)
- voltage: Voltage-compensated SOC correction
- noise: Demand signal noise filter (CT/PF artefacts)
- dg: DG transition detection
- apfc: APFC switching event detection
- imputation: Missing/anomalous value imputation
- pipeline: Integrated quality pipeline + QualityReport

All public symbols are re-exported here for backward compatibility:
    from edgegrid_forecast.data.quality import detect_gaps  # works
"""

# ── M1-F1: AMI Ingestion ─────────────────────────────────────────────────────
from .ami import (
    check_physical_consistency,
    compute_interval_quality_score,
    detect_gaps,
    handle_duplicates,
    handle_late_arrivals,
    sync_channels,
    validate_physical_ranges,
)

# ── M1-F2: Anomaly Detection ─────────────────────────────────────────────────
from .anomaly import (
    detect_frozen_readings,
    detect_outliers_contextual,
    detect_outliers_isolation_forest,
    detect_outliers_iqr,
    detect_outliers_rolling,
    detect_outliers_zscore,
)

# ── M1-F3: Voltage SOC Correction ────────────────────────────────────────────
from .voltage import VoltageSOCCorrector

# ── M1-F4: Demand Noise Filter ───────────────────────────────────────────────
from .noise import DemandNoiseFilter

# ── M1-F5: DG Transition Detection ───────────────────────────────────────────
from .dg import DGTransitionDetector

# ── M1-F6: APFC Switching Detection ──────────────────────────────────────────
from .apfc import APFCSwitchingDetector

# ── Imputation ────────────────────────────────────────────────────────────────
from .imputation import impute_missing_and_anomalous

# ── Pipeline ──────────────────────────────────────────────────────────────────
from .pipeline import QualityReport, run_quality_pipeline

# ── Constants (for advanced users) ────────────────────────────────────────────
from ._constants import AMI_CHANNELS, CHANNEL_RANGES, QUALITY_WEIGHTS

__all__ = [
    # AMI
    "detect_gaps",
    "handle_duplicates",
    "handle_late_arrivals",
    "sync_channels",
    "validate_physical_ranges",
    "check_physical_consistency",
    "compute_interval_quality_score",
    # Anomaly
    "detect_frozen_readings",
    "detect_outliers_zscore",
    "detect_outliers_iqr",
    "detect_outliers_contextual",
    "detect_outliers_rolling",
    "detect_outliers_isolation_forest",
    # Classes
    "VoltageSOCCorrector",
    "DemandNoiseFilter",
    "DGTransitionDetector",
    "APFCSwitchingDetector",
    # Imputation
    "impute_missing_and_anomalous",
    # Pipeline
    "QualityReport",
    "run_quality_pipeline",
    # Constants
    "AMI_CHANNELS",
    "CHANNEL_RANGES",
    "QUALITY_WEIGHTS",
]
