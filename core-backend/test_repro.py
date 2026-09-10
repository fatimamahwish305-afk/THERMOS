import pandas as pd
from pathlib import Path
import datetime
import random

# Mocking the historical_df loading for the test
DATA_PATH = Path("delhi_processed_wbgt.csv")
historical_df = pd.read_csv(DATA_PATH)
historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])

def get_weather_features(target_date, temp_offset):
    # Using the same logic as in main.py
    min_ts = historical_df['timestamp'].min()
    max_ts = historical_df['timestamp'].max()

    if target_date < min_ts:
        features = historical_df.iloc[0].copy()
    elif target_date > max_ts:
        features = historical_df.iloc[-1].copy()
    elif target_date in historical_df['timestamp'].values:
        features = historical_df[historical_df['timestamp'] == target_date].iloc[0].copy()
    else:
        idx = (historical_df['timestamp'] - target_date).abs().argsort().iloc[0]
        features = historical_df.iloc[idx].copy()
    
    # ... (omitted temp offset application for brevity)
    return features

# Test for May 18, 2026
target_date = pd.Timestamp("2026-05-18 12:00:00")
features = get_weather_features(target_date, 0.0)
print(f"Features for {target_date}:")
print(features[['timestamp', 'calculated_wbgt', 'temperature_2m']])
