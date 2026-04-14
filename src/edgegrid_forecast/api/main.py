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
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from ..config import config
from ..data.features import build_forecast_features
from ..dispatch.optimizer import BESSConfig, DispatchOptimizer, DispatchSchedule
from ..models.demand import LightGBMDemandForecaster, ForecastResult
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
def health_check():
    return {
        "status": "healthy",
        "version": "0.1.0",
        "engine": "EdgeGrid Forecast Engine",
        "available_consumers": list(HT_CONSUMERS.keys()),
    }


@app.get("/consumers")
def list_consumers():
    """List available consumers with metadata."""
    return {
        cid: {
            "region": info["region"],
            "latitude": info["lat"],
            "longitude": info["lon"],
        }
        for cid, info in HT_CONSUMERS.items()
    }


# ─── Model Registry ──────────────────────────────────────────────────────────
# In-memory cache of loaded models. In production, back this with Redis or
# a model-serving framework like MLflow / BentoML.

_model_registry: Dict[str, LightGBMDemandForecaster] = {}


def _get_demand_model(consumer_id: str) -> LightGBMDemandForecaster:
    """Load a trained demand model for a consumer, with caching."""
    if consumer_id in _model_registry:
        return _model_registry[consumer_id]

    model_path = config.model_dir / consumer_id
    if not (model_path / "lightgbm_demand.joblib").exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"No trained model found for consumer '{consumer_id}'. "
                f"Train a model first via the pipeline, then save it to {model_path}."
            ),
        )

    model = LightGBMDemandForecaster()
    model.load(model_path)
    _model_registry[consumer_id] = model
    logger.info(f"Loaded demand model for {consumer_id} ({len(model.feature_names)} features)")
    return model


@app.post("/forecast/demand", response_model=DemandForecastResponse)
def forecast_demand(request: DemandForecastRequest):
    """
    Generate demand forecast for an HT consumer.

    Requires a pre-trained LightGBM model saved under
    ``config.model_dir/<consumer_id>/lightgbm_demand.joblib``.

    The endpoint builds forecast features from the consumer's recent history,
    runs the model, and returns point predictions with uncertainty bounds.
    """
    # Validate consumer exists
    if request.consumer_id not in HT_CONSUMERS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown consumer '{request.consumer_id}'. Available: {list(HT_CONSUMERS.keys())}",
        )

    # Load model (from cache or disk)
    model = _get_demand_model(request.consumer_id)

    # Build feature DataFrame for the forecast horizon.
    # In production, this reads recent actuals from the data store.
    # For now, create a timestamp scaffold and generate temporal features.
    start = pd.Timestamp.now().floor("h") + pd.Timedelta(hours=1)
    timestamps = pd.date_range(start, periods=request.horizon_hours, freq="h")

    forecast_df = pd.DataFrame({
        "timestamp": timestamps,
        "consumer_id": request.consumer_id,
        # Placeholder: in production, fill from latest actuals for lag/rolling features.
        # When lag features are unavailable (no recent history), the model
        # degrades gracefully — LightGBM treats missing values as NaN natively.
        "consumption_kwh": np.nan,
    })

    forecast_df = build_forecast_features(forecast_df, target_col="consumption_kwh")

    # Ensure all expected feature columns exist; fill missing with NaN
    for col in model.feature_names:
        if col not in forecast_df.columns:
            forecast_df[col] = np.nan

    result: ForecastResult = model.predict(forecast_df)

    response = DemandForecastResponse(
        consumer_id=request.consumer_id,
        timestamps=[t.isoformat() for t in result.timestamp],
        point_forecast_kwh=result.point_forecast.tolist(),
        lower_bound_kwh=result.lower_bound.tolist(),
        upper_bound_kwh=result.upper_bound.tolist(),
        model_name=result.model_name,
        metrics=result.metrics,
    )

    if request.include_features:
        response.metrics["n_features"] = len(model.feature_names)
        response.metrics["top_features"] = {
            row["feature"]: float(row["importance"])
            for _, row in model.feature_importance.head(10).iterrows()
        } if model.feature_importance is not None else {}

    return response


@app.post("/forecast/price", response_model=PriceForecastResponse)
def forecast_price(
    month: int = Query(default=4, ge=1, le=12),
    hours: int = Query(default=24, ge=1, le=168),
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
def forecast_solar(request: SolarForecastRequest):
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
def optimize_dispatch(request: DispatchRequest):
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
