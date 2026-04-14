"""
Central configuration for the EdgeGrid Forecast Engine.

Environment variables override defaults. Use a .env file for local development.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ForecastConfig:
    """Configuration for the forecasting pipeline."""

    # ─── Data paths ───────────────────────────────────────────────────────
    data_dir: Path = Path(os.getenv("EDGEGRID_DATA_DIR", "./data"))
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    model_dir: Path = Path(os.getenv("EDGEGRID_MODEL_DIR", "./models"))

    # ─── Forecast horizons ────────────────────────────────────────────────
    demand_horizon_hours: int = 48          # How far ahead to forecast demand
    solar_horizon_hours: int = 48           # Solar generation forecast horizon
    price_horizon_hours: int = 24           # IEX price forecast (day-ahead)
    dispatch_horizon_hours: int = 24        # Dispatch optimization window

    # ─── Model parameters ─────────────────────────────────────────────────
    demand_model_type: str = "lightgbm"     # lightgbm | prophet | ensemble
    lookback_hours: int = 168               # 7 days of history for features
    train_test_split_ratio: float = 0.8
    cv_folds: int = 5
    early_stopping_rounds: int = 50

    # ─── Feature engineering ──────────────────────────────────────────────
    lag_hours: list = field(default_factory=lambda: [1, 2, 3, 6, 12, 24, 48, 168])
    rolling_windows: list = field(default_factory=lambda: [3, 6, 12, 24, 48, 168])

    # ─── BESS optimization ────────────────────────────────────────────────
    bess_size_range_mwh: tuple = (0.5, 20.0)
    bess_size_step_mwh: float = 0.5
    bess_duration_options: list = field(default_factory=lambda: [2, 4, 6])
    bess_capex_lakhs_per_mwh: float = 150   # Current market rate
    target_irr_pct: float = 10.0            # Minimum acceptable IRR

    # ─── API configuration ────────────────────────────────────────────────
    api_host: str = os.getenv("EDGEGRID_API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("EDGEGRID_API_PORT", "8000"))

    # ─── External APIs ────────────────────────────────────────────────────
    # Open-Meteo for weather (free, no API key needed)
    weather_api_url: str = "https://api.open-meteo.com/v1/forecast"
    # NSRDB for solar irradiance (NREL)
    nsrdb_api_key: Optional[str] = os.getenv("NSRDB_API_KEY")
    # IEX API (if available)
    iex_api_key: Optional[str] = os.getenv("IEX_API_KEY")

    def __post_init__(self):
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        # Ensure directories exist
        for d in [self.raw_data_dir, self.processed_data_dir, self.model_dir]:
            d.mkdir(parents=True, exist_ok=True)


# Singleton config instance
config = ForecastConfig()
