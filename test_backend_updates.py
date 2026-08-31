import sys
import os
from pathlib import Path
import pytest
import pandas as pd

# Add core-backend to python path
sys.path.append(str(Path(__file__).parent / "core-backend"))

from main import app, get_weather_features
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    return TestClient(app)

def test_normal_query(client):
    """Test normal queries against /api/v1/heatwave-risk with valid historical date and ward."""
    response = client.get("/api/v1/heatwave-risk?date=2020-01-02%2000:00:00&ward_id=80")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "metadata" in data
    assert len(data["data"]) == 5
    assert "date" in data["data"][0]
    assert "results" in data["data"][0]
    assert len(data["data"][0]["results"]) == 1
    assert data["data"][0]["results"][0]["ward_id"] == "80"
    assert "wbgt" in data["data"][0]["results"][0]
    assert "risk_level" in data["data"][0]["results"][0]

def test_temperature_offsets(client):
    """Test temperature offsets (temp_offset) in climate scenarios."""
    # Query without offset
    response_normal = client.get("/api/v1/heatwave-risk?ward_id=80")
    assert response_normal.status_code == 200
    normal_wbgt = response_normal.json()["data"][0]["results"][0]["wbgt"]

    # Query with +4.0 offset
    response_offset = client.get("/api/v1/heatwave-risk?ward_id=80&temp_offset=4.0")
    assert response_offset.status_code == 200
    offset_wbgt = response_offset.json()["data"][0]["results"][0]["wbgt"]

    # WBGT should be higher with a positive temperature offset
    assert offset_wbgt > normal_wbgt

@pytest.mark.parametrize(
    "test_date,description",
    [
        ("1900-01-01", "Past out-of-bounds date fallback"),
        ("2050-01-01", "Future out-of-bounds date fallback"),
        ("not-a-valid-date", "Invalid date string fallback"),
    ],
)
def test_out_of_bounds_date_fallback(client, test_date, description):
    """Test robust date fallback handling for out-of-bounds or invalid dates."""
    response = client.get(f"/api/v1/heatwave-risk?date={test_date}")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) == 5
    assert "metadata" in data

def test_ward_filtering(client):
    """Test ward filtering with both numeric and string ward IDs."""
    # Numeric ward ID (e.g. 80)
    response_num = client.get("/api/v1/heatwave-risk?ward_id=80")
    assert response_num.status_code == 200
    data_num = response_num.json()
    assert len(data_num["data"]) == 5
    assert len(data_num["data"][0]["results"]) == 1
    assert data_num["data"][0]["results"][0]["ward_id"] == "80"

    # String ward ID (e.g. CANT_1)
    response_str = client.get("/api/v1/heatwave-risk?ward_id=CANT_1")
    assert response_str.status_code == 200
    data_str = response_str.json()
    assert len(data_str["data"]) == 5
    assert len(data_str["data"][0]["results"]) == 1
    assert data_str["data"][0]["results"][0]["ward_id"] == "CANT_1"


def test_forecast_heatwave(client):
    """Test POST /api/v1/forecast-heatwave endpoint with 5-day forecast inputs."""
    payload = [
        {
            "ward_id": "80",
            "date": "2026-06-01",
            "temperature_2m": 38.5,
            "relative_humidity_2m": 45.0
        },
        {
            "ward_id": "80",
            "date": "2026-06-02",
            "temperature_2m": 40.0,
            "relative_humidity_2m": 40.0
        },
        {
            "ward_id": "80",
            "date": "2026-06-03",
            "temperature_2m": 37.0,
            "relative_humidity_2m": 50.0
        },
        {
            "ward_id": "80",
            "date": "2026-06-04",
            "temperature_2m": 39.2,
            "relative_humidity_2m": 42.0
        },
        {
            "ward_id": "80",
            "date": "2026-06-05",
            "temperature_2m": 41.5,
            "relative_humidity_2m": 38.0
        }
    ]
    response = client.post("/api/v1/forecast-heatwave", json=payload)
    assert response.status_code == 200
    res_json = response.json()
    assert "data" in res_json
    assert "metadata" in res_json
    assert len(res_json["data"]) == 5
    assert "models_used" in res_json["metadata"]
    assert "thermal_stress_model.json" in res_json["metadata"]["models_used"]
    assert "wbgt_risk_model.joblib" in res_json["metadata"]["models_used"]

    for day_res in res_json["data"]:
        assert "day" in day_res
        assert "ward_id" in day_res
        assert "ward_name" in day_res
        assert "wbgt" in day_res
        assert "thermal_stress_score" in day_res
        assert "risk_level" in day_res
        assert isinstance(day_res["thermal_stress_score"], (int, float))
        assert isinstance(day_res["risk_level"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

