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
    assert len(data["data"]) == 1
    assert data["data"][0]["ward_id"] == "80"
    assert "wbgt" in data["data"][0]
    assert "risk_level" in data["data"][0]

def test_temperature_offsets(client):
    """Test temperature offsets (temp_offset) in climate scenarios."""
    # Query without offset
    response_normal = client.get("/api/v1/heatwave-risk?ward_id=80")
    assert response_normal.status_code == 200
    normal_wbgt = response_normal.json()["data"][0]["wbgt"]

    # Query with +4.0 offset
    response_offset = client.get("/api/v1/heatwave-risk?ward_id=80&temp_offset=4.0")
    assert response_offset.status_code == 200
    offset_wbgt = response_offset.json()["data"][0]["wbgt"]

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
    assert len(data["data"]) > 0
    assert "metadata" in data

def test_ward_filtering(client):
    """Test ward filtering with both numeric and string ward IDs."""
    # Numeric ward ID (e.g. 80)
    response_num = client.get("/api/v1/heatwave-risk?ward_id=80")
    assert response_num.status_code == 200
    data_num = response_num.json()
    assert len(data_num["data"]) == 1
    assert data_num["data"][0]["ward_id"] == "80"

    # String ward ID (e.g. CANT_1)
    response_str = client.get("/api/v1/heatwave-risk?ward_id=CANT_1")
    assert response_str.status_code == 200
    data_str = response_str.json()
    assert len(data_str["data"]) == 1
    assert data_str["data"][0]["ward_id"] == "CANT_1"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

