"""
FastAPI service for EdgeGrid Forecast Engine.

Endpoints:
- POST /forecast/demand      — Demand forecast for a consumer
- POST /forecast/solar       — Solar generation forecast
- POST /forecast/price       — IEX price forecast
- POST /dispatch/optimize    — Dispatch optimization for 24h window
- POST /dispatch/bess-sizing — BESS sizing optimization
- GET  /health               — Health check
- GET  /consumers            — List available consumers
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from ..config import config
from ..dispatch.optimizer import BESSConfig, DispatchOptimizer, DispatchSchedule
from ..models.price import IEXPriceForecaster
from ..models.solar import SolarForecastResult, estimate_solar_generation
from ..utils.constants import ChargingStrategy, HT_CONSUMERS

app = FastAPI(
    title="EdgeGrid Forecast Engine",
    description=(
        "Predictive dispatch engine for India's distribution grid. "
        "Demand forecasting, solar generation prediction, BESS optimization, "
        "and energy market intelligence."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request/Response Models ─────────────────────────────────────────────────

class DemandForecastRequest(BaseModel):
    consumer_id: str
    horizon_hours: int = Field(default=48, ge=1, le=168)
    include_features: bool = False


class DemandForecastResponse(BaseModel):
    consumer_id: str
    timestamps: List[str]
    point_forecast_kwh: List[float]
    lower_bound_kwh: List[float]
    upper_bound_kwh: List[float]
    model_name: str
    metrics: Dict[str, float] = {}


class SolarForecastRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    capacity_kw: float = Field(gt=0)
    horizon_hours: int = Field(default=48, ge=1, le=168)


class SolarForecastResponse(BaseModel):
    timestamps: List[str]
    generation_kwh: List[float]
    clear_sky_kwh: List[float]
    capacity_factor: List[float]


class DispatchRequest(BaseModel):
    demand_kwh: List[float] = Field(..., min_length=24, max_length=24)
    solar_kwh: List[float] = Field(..., min_length=24, max_length=24)
    iex_prices: List[float] = Field(..., min_length=24, max_length=24)
    bess_capacity_kwh: Optional[float] = None
    bess_duration_hours: Optional[int] = 4
    fls_tariff: float = 6.50
    strategy: str = "cheap_grid"


class DispatchResponse(BaseModel):
    solar_direct_use_kwh: List[float]
    bess_charge_kwh: List[float]
    bess_discharge_kwh: List[float]
    bess_soc_kwh: List[float]
    grid_purchase_kwh: List[float]
    iex_purchase_kwh: List[float]
    total_cost_inr: float
    solar_savings_inr: float
    bess_savings_inr: float
    reliability_score: float


class PriceForecastResponse(BaseModel):
    timestamps: List[str]
    iex_price_inr_kwh: List[float]
    landed_cost_inr_kwh: List[float]
    cheapest_hours: List[int]
    expensive_hours: List[int]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "0.1.0",
        "engine": "EdgeGrid Forecast Engine",
        "available_consumers": list(HT_CONSUMERS.keys()),
    }


@app.get("/consumers")
async def list_consumers():
    """List available consumers with metadata."""
    return {
        cid: {
            "region": info["region"],
            "latitude": info["lat"],
            "longitude": info["lon"],
        }
        for cid, info in HT_CONSUMERS.items()
    }


@app.post("/forecast/price", response_model=PriceForecastResponse)
async def forecast_price(
    month: int = Field(default=4, ge=1, le=12),
    hours: int = Field(default=24, ge=1, le=168),
):
    """Forecast IEX DAM prices and landed costs."""
    forecaster = IEXPriceForecaster()

    timestamps = pd.date_range(
        datetime.now().replace(hour=0, minute=0, second=0),
        periods=hours,
        freq="h",
    )

    df = forecaster.forecast_prices(timestamps)

    return PriceForecastResponse(
        timestamps=[t.isoformat() for t in df["timestamp"]],
        iex_price_inr_kwh=df["iex_price_inr_kwh"].tolist(),
        landed_cost_inr_kwh=df["landed_cost_inr_kwh"].tolist(),
        cheapest_hours=forecaster.find_cheapest_hours(month),
        expensive_hours=forecaster.find_expensive_hours(month),
    )


@app.post("/forecast/solar", response_model=SolarForecastResponse)
async def forecast_solar(request: SolarForecastRequest):
    """Forecast solar generation for a given location and capacity."""
    timestamps = pd.date_range(
        datetime.now().replace(minute=0, second=0),
        periods=request.horizon_hours,
        freq="h",
    )

    result = estimate_solar_generation(
        timestamps=timestamps,
        latitude=request.latitude,
        longitude=request.longitude,
        capacity_kw=request.capacity_kw,
    )

    return SolarForecastResponse(
        timestamps=[t.isoformat() for t in timestamps],
        generation_kwh=result.generation_kwh.tolist(),
        clear_sky_kwh=result.clear_sky_kwh.tolist(),
        capacity_factor=result.capacity_factor.tolist(),
    )


@app.post("/dispatch/optimize", response_model=DispatchResponse)
async def optimize_dispatch(request: DispatchRequest):
    """Optimize energy dispatch for a 24-hour window."""
    bess = None
    if request.bess_capacity_kwh and request.bess_capacity_kwh > 0:
        bess = BESSConfig(
            capacity_kwh=request.bess_capacity_kwh,
            max_power_kw=request.bess_capacity_kwh / request.bess_duration_hours,
        )

    try:
        strategy = ChargingStrategy(request.strategy)
    except ValueError:
        strategy = ChargingStrategy.CHEAP_GRID

    optimizer = DispatchOptimizer(
        bess_config=bess,
        fls_tariff_inr_kwh=request.fls_tariff,
        strategy=strategy,
    )

    schedule = optimizer.optimize_24h(
        demand_kwh=np.array(request.demand_kwh),
        solar_kwh=np.array(request.solar_kwh),
        iex_prices_inr_kwh=np.array(request.iex_prices),
    )

    return DispatchResponse(
        solar_direct_use_kwh=schedule.solar_direct_use_kwh.tolist(),
        bess_charge_kwh=schedule.bess_charge_kwh.tolist(),
        bess_discharge_kwh=schedule.bess_discharge_kwh.tolist(),
        bess_soc_kwh=schedule.bess_soc_kwh.tolist(),
        grid_purchase_kwh=schedule.grid_purchase_kwh.tolist(),
        iex_purchase_kwh=schedule.iex_purchase_kwh.tolist(),
        total_cost_inr=schedule.total_cost_inr,
        solar_savings_inr=schedule.solar_savings_inr,
        bess_savings_inr=schedule.bess_savings_inr,
        reliability_score=schedule.reliability_score,
    )


# Need pandas for some endpoints
import pandas as pd


def run():
    """Entry point for the CLI."""
    uvicorn.run(
        "edgegrid_forecast.api.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=True,
    )


if __name__ == "__main__":
    run()
