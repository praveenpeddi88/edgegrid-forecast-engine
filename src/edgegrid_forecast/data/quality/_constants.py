"""
Named constants for M1 Data Quality Engine.

All magic numbers and configurable defaults are defined here.
Import from this module rather than scattering literals across the codebase.
"""

from typing import Dict, Tuple

# ── AMI Channel Configuration ────────────────────────────────────────────────

#: Standard AMI channels for Indian HT consumers (11kV / 33kV)
AMI_CHANNELS: list = ["kw", "kvar", "kva", "voltage", "current", "pf"]

#: Physical range limits per channel — values outside are flagged, not removed
CHANNEL_RANGES: Dict[str, Tuple[float, float]] = {
    "kw": (0, 50_000),         # 0 to 50 MW (largest HT consumers)
    "kvar": (-20_000, 20_000), # Reactive power can be negative (leading PF)
    "kva": (0, 50_000),        # Apparent power always positive
    "voltage": (300, 500),     # 415V nominal ±20% (330-500V realistic)
    "current": (0, 5000),      # Amps
    "pf": (0.0, 1.0),         # Power factor 0 to 1
}

# ── Quality Score Weights ─────────────────────────────────────────────────────

#: Per-interval quality score dimension weights (must sum to 1.0)
QUALITY_WEIGHTS: Dict[str, float] = {
    "completeness": 0.4,    # All channels present
    "timeliness": 0.3,      # Arrived within expected window
    "validity": 0.2,        # Passes range checks
    "consistency": 0.1,     # Channels are physically consistent
}

# ── Timeliness Scoring Curve ──────────────────────────────────────────────────

#: Delay (in minutes) at which timeliness score starts degrading
TIMELINESS_DEGRADE_START_MIN: float = 30.0

#: Delay (in minutes) at which timeliness score hits minimum
TIMELINESS_DEGRADE_END_MIN: float = 120.0

#: Minimum timeliness score for late-but-within-window packets
TIMELINESS_MIN_SCORE: float = 0.5

# ── Anomaly Detection Defaults ────────────────────────────────────────────────

#: Minimum group size for contextual z-score computation
CONTEXTUAL_MIN_GROUP_SIZE: int = 5

#: Rolling window minimum observations for rolling outlier detection
ROLLING_MIN_PERIODS: int = 10

#: When frequency data is unavailable, elevate CT artefact sigma threshold by this amount
CT_ARTEFACT_SIGMA_ELEVATION: float = 1.0

#: Minimum PF change (absolute) to register as a switch event
MIN_PF_JUMP: float = 0.02

# ── Voltage SOC Correction ────────────────────────────────────────────────────

#: Fraction of peak current below which the battery is considered "at rest"
REST_CURRENT_FRACTION: float = 0.02

#: Percentile used to define "peak current" for rest threshold
REST_CURRENT_PERCENTILE: float = 0.95

#: SOC threshold for "full charge" classification (%)
FULL_CHARGE_SOC_THRESHOLD: float = 98.0

#: SOC threshold for "full discharge" classification (%)
FULL_DISCHARGE_SOC_THRESHOLD: float = 5.0

#: Default measurement frequency in seconds (15 min)
DEFAULT_FREQ_SECONDS: int = 900

# ── Indian Grid Constants ─────────────────────────────────────────────────────

#: Nominal voltage for 3-phase LT supply in India (V)
NOMINAL_VOLTAGE_INDIA: float = 415.0

#: Normal grid frequency band (Hz) — outside this indicates instability
GRID_FREQ_BAND: Tuple[float, float] = (49.5, 50.5)
