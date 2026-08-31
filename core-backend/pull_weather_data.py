"""
pull_weather_data.py

Fetches 5 years (2020-01-01 to 2025-12-31) of historical hourly weather data
for New Delhi from the Open-Meteo Historical Weather (Archive) API and saves
it to a CSV file for use in the Urban Heat Engine ML pipeline.

Variables pulled:
    - temperature_2m
    - relative_humidity_2m
    - wind_speed_10m
    - shortwave_radiation

Usage:
    python pull_weather_data.py
"""

import sys
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = "https://archive-api.open-meteo.com/v1/archive"

LATITUDE = 28.6139
LONGITUDE = 77.2090

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
]

OUTPUT_FILE = Path(__file__).resolve().parent / "delhi_historical_meteorology.csv"

REQUEST_PARAMS = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": ",".join(HOURLY_VARIABLES),
    "timezone": "Asia/Kolkata",
}


def fetch_weather_data(url: str, params: dict) -> dict:
    """Call the Open-Meteo Archive API and return the parsed JSON response."""
    print(f"Requesting historical weather data from Open-Meteo API...")
    print(f"  Location : New Delhi (lat={params['latitude']}, lon={params['longitude']})")
    print(f"  Period   : {params['start_date']} to {params['end_date']}")
    print(f"  Variables: {params['hourly']}")

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"ERROR: Failed to fetch data from Open-Meteo API: {exc}", file=sys.stderr)
        sys.exit(1)

    return response.json()


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
    return df


def main():
    payload = fetch_weather_data(API_URL, REQUEST_PARAMS)
    df = build_dataframe(payload)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSuccess! Saved {len(df):,} hourly records to: {OUTPUT_FILE}")
    print(f"Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()
