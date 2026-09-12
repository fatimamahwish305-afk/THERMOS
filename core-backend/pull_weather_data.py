import sys
import argparse
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HISTORICAL_API_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

LATITUDE = 28.6139
LONGITUDE = 77.2090

# Historical params
HIST_START_DATE = "2020-01-01"
HIST_END_DATE = "2025-12-31"

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
]

HIST_OUTPUT_FILE = Path(__file__).resolve().parent / "delhi_historical_meteorology.csv"
FORECAST_OUTPUT_FILE = Path(__file__).resolve().parent / "delhi_forecast_meteorology.csv"

def fetch_data(url: str, params: dict) -> dict:
    """Generic function to call Open-Meteo API."""
    print(f"Requesting weather data from {url}...")
    
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    return response.json()

def fetch_historical_data():
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": HIST_START_DATE,
        "end_date": HIST_END_DATE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "Asia/Kolkata",
    }
    payload = fetch_data(HISTORICAL_API_URL, params)
    return payload

def fetch_forecast_data():
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "Asia/Kolkata",
        "forecast_days": 7,
    }
    payload = fetch_data(FORECAST_API_URL, params)
    return payload

def build_dataframe(payload: dict) -> pd.DataFrame:
    """Convert the API's 'hourly' block into a pandas DataFrame."""
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        print("ERROR: Unexpected API response — no 'hourly' data found.", file=sys.stderr)
        print(payload, file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(hourly)
    df.rename(columns={"time": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Add hour, month, day_of_year for compatibility with ML model
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Fetch weather data for Delhi.")
    parser.add_argument("--type", choices=["historical", "forecast"], default="historical", help="Data type to fetch.")
    args = parser.parse_args()

    if args.type == "historical":
        payload = fetch_historical_data()
        output_file = HIST_OUTPUT_FILE
    else:
        payload = fetch_forecast_data()
        output_file = FORECAST_OUTPUT_FILE

    df = build_dataframe(payload)
    df.to_csv(output_file, index=False)

    print(f"\nSuccess! Saved {len(df):,} hourly records to: {output_file}")
    print(f"Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print("\nPreview:")
    print(df.head())

if __name__ == "__main__":
    main()
