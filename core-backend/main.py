from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from typing import Optional, Union
from pathlib import Path
import datetime
import random
import numpy as np

# Load data for historical/future lookups
DATA_PATH = Path(__file__).parent / "delhi_processed_wbgt.csv"
historical_df = pd.read_csv(DATA_PATH)
historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



import joblib

# Load model
MODEL_PATH = Path(__file__).parent / "wbgt_risk_model.joblib"
model = joblib.load(MODEL_PATH)

# Load canonical wards
WARDS_PATH = Path(__file__).parent / "canonical_ward_ids.csv"
wards_df = pd.read_csv(WARDS_PATH)

def get_weather_features(target_date: pd.Timestamp, temp_offset: float):
    """
    Fetches or simulates weather features for a given date with robust fallback and out-of-bounds handling.
    """
    min_ts = historical_df['timestamp'].min()
    max_ts = historical_df['timestamp'].max()

    # Out-of-bounds handling
    if target_date < min_ts:
        # Fallback to earliest available data for past out-of-bounds dates
        features = historical_df.iloc[0].copy()
    elif target_date > max_ts:
        # Fallback to latest available data for future out-of-bounds dates
        features = historical_df.iloc[-1].copy()
    elif target_date in historical_df['timestamp'].values:
        features = historical_df[historical_df['timestamp'] == target_date].iloc[0].copy()
    else:
        # Nearest neighbor fallback for missing intermediate dates
        idx = (historical_df['timestamp'] - target_date).abs().argsort().iloc[0]
        features = historical_df.iloc[idx].copy()
    
    # Feature set for model:
    # ["temperature_2m","relative_humidity_2m","wind_speed_10m","shortwave_radiation","hour","day_of_year","month","temp_lag_24","humidity_lag_24"]
    
    # Apply temperature offset
    temp_features = ['temperature_2m', 'temp_lag_24']
    for feat in temp_features:
        if feat in features:
            features[feat] += temp_offset
            
    feature_cols = [
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m", 
        "shortwave_radiation", "hour", "day_of_year", "month", 
        "temp_lag_24", "humidity_lag_24"
    ]
    
    return features[feature_cols].values.reshape(1, -1)

@app.get("/api/v1/heatwave-risk")
async def get_heatwave_risk(
    date: Optional[str] = Query(None, description="Target date (e.g. YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)"),
    ward_id: Optional[str] = Query(None, description="Ward ID (e.g. 80 or CANT_1)"),
    temp_offset: float = Query(0.0, description="Temperature offset in degrees Celsius")
):
    results = []
    
    # Filter wards if ward_id is provided
    target_wards = wards_df
    if ward_id:
        target_wards = wards_df[wards_df['ward_id'].astype(str) == str(ward_id)]
        
    # Determine date with robust exception handling for invalid date inputs
    if date:
        try:
            target_date = pd.to_datetime(date)
            if pd.isna(target_date):
                target_date = historical_df['timestamp'].max()
        except (ValueError, TypeError, pd.errors.ParserError):
            target_date = historical_df['timestamp'].max()
    else:
        target_date = historical_df['timestamp'].max()
    
    # Get features
    features = get_weather_features(target_date, temp_offset)
    
    # Prediction
    prediction = model.predict(features)[0]
    
    for _, ward in target_wards.iterrows():
        # Using prediction for the ward (mocking that risk varies slightly by ward)
        ward_specific_wbgt = prediction + random.uniform(-0.5, 0.5)
        
        results.append({
            "ward_id": str(ward['ward_id']),
            "ward_name": ward['ward_name'],
            "wbgt": round(float(ward_specific_wbgt), 2),
            "thermal_stress_score": round(max(0, min(10, float(ward_specific_wbgt - 20))), 2),
            "heatwave_probability": round(max(0, min(1, float((ward_specific_wbgt - 25) / 10))), 2),
            "risk_level": "Extreme" if ward_specific_wbgt > 32 else "High" if ward_specific_wbgt > 29 else "Caution" if ward_specific_wbgt > 27 else "Normal"
        })
            
    return {"data": results, "metadata": {"date": target_date, "temp_offset": temp_offset}}

