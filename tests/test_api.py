"""Tests for the FastAPI endpoints."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from edgegrid_forecast.api.main import app


client = TestClient(app)


class TestHealthEndpoint:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_contains_version(self):
        data = client.get("/health").json()
        assert "version" in data
        assert data["version"] == "0.1.0"

    def test_health_lists_consumers(self):
        data = client.get("/health").json()
        assert "available_consumers" in data
        assert len(data["available_consumers"]) > 0


class TestConsumersEndpoint:

    def test_consumers_returns_200(self):
        response = client.get("/consumers")
        assert response.status_code == 200

    def test_consumers_have_coordinates(self):
        data = client.get("/consumers").json()
        for cid, info in data.items():
            assert "latitude" in info
            assert "longitude" in info


class TestPriceEndpoint:

    def test_price_forecast_returns_200(self):
        response = client.post("/forecast/price?month=6&hours=24")
        assert response.status_code == 200

    def test_price_forecast_has_24_hours(self):
        data = client.post("/forecast/price?month=6&hours=24").json()
        assert len(data["timestamps"]) == 24
        assert len(data["iex_price_inr_kwh"]) == 24
        assert len(data["landed_cost_inr_kwh"]) == 24

    def test_landed_cost_exceeds_iex(self):
        data = client.post("/forecast/price?month=6&hours=24").json()
        for iex, landed in zip(data["iex_price_inr_kwh"], data["landed_cost_inr_kwh"]):
            assert landed > iex, "Landed cost must exceed raw IEX price"


class TestSolarEndpoint:

    def test_solar_forecast_returns_200(self):
        response = client.post("/forecast/solar", json={
            "latitude": 17.68,
            "longitude": 83.22,
            "capacity_kw": 500,
            "horizon_hours": 24,
        })
        assert response.status_code == 200

    def test_solar_generation_nonnegative(self):
        data = client.post("/forecast/solar", json={
            "latitude": 17.68,
            "longitude": 83.22,
            "capacity_kw": 500,
            "horizon_hours": 24,
        }).json()
        assert all(g >= 0 for g in data["generation_kwh"])


class TestDispatchEndpoint:

    def _make_request(self):
        return {
            "demand_kwh": [100 + 50 * np.sin(h * np.pi / 12) for h in range(24)],
            "solar_kwh": [max(0, 80 * np.sin((h - 6) * np.pi / 12)) for h in range(24)],
            "iex_prices": [3.5 + 2 * np.sin((h - 4) * np.pi / 12) for h in range(24)],
            "bess_capacity_kwh": 500,
            "bess_duration_hours": 4,
            "fls_tariff": 6.50,
            "strategy": "cheap_grid",
        }

    def test_dispatch_returns_200(self):
        response = client.post("/dispatch/optimize", json=self._make_request())
        assert response.status_code == 200

    def test_dispatch_has_24_hour_arrays(self):
        data = client.post("/dispatch/optimize", json=self._make_request()).json()
        for key in ["solar_direct_use_kwh", "bess_charge_kwh", "bess_discharge_kwh", "bess_soc_kwh"]:
            assert len(data[key]) == 24


class TestDemandEndpoint:

    def test_demand_unknown_consumer_404(self):
        response = client.post("/forecast/demand", json={
            "consumer_id": "NONEXISTENT",
            "horizon_hours": 24,
        })
        assert response.status_code == 404

    def test_demand_no_model_404(self):
        """Known consumer but no saved model should return 404."""
        response = client.post("/forecast/demand", json={
            "consumer_id": "RJY1197",
            "horizon_hours": 24,
        })
        # Should be 404 since we haven't saved a model in the test environment
        assert response.status_code == 404
        assert "No trained model" in response.json()["detail"]
