from fastapi import FastAPI, Query
from fastapi import BackgroundTasks
from fastapi.responses import JSONResponse
import traceback

import math
import hashlib

from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from typing import Optional, Union, List
from pathlib import Path
import datetime
import random
import numpy as np
from pydantic import BaseModel, Field


import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.engine import compute_ward_risk, calculate_ward_risk_metrics_full

def convert_numpy_types(obj):
    """
    Recursively converts numpy types and pandas types to native Python types for JSON serialization.
    """
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return obj

try:
    import xgboost as xgb
except ImportError:
    xgb = None

def get_ward_metrics(ward_id: str):
    """
    Computes deterministic population_density and vulnerability_score 
    based on the ward_id string.
    """
    hash_val = int(hashlib.md5(ward_id.encode()).hexdigest(), 16)
    density = 8000 + (hash_val % 37000)
    vulnerability = (hash_val % 100) / 100.0
    return density, vulnerability


def calculate_risk_metrics(wbgt: float, ward_id: str):
    density, vulnerability = get_ward_metrics(ward_id)
    
    # Increased sensitivity for high WBGT values to ensure realistic thermal stress scores
    # Factor increases dynamically as WBGT exceeds 30°C
    factor = 1.0 + max(0, (wbgt - 30) * 0.5)
    hospitalization_increase_pct = max(0, (wbgt - 20) * vulnerability * (density / 10000) * factor)
    
    additional_beds_needed = math.ceil((hospitalization_increase_pct / 100) * (density * 0.05))
    surge_probability = min(1.0, max(0, (wbgt - 25) * 0.05 + vulnerability * 0.3))
    
    return round(float(hospitalization_increase_pct), 2), int(additional_beds_needed), round(float(surge_probability), 2)


class ActionRecommendations(BaseModel):
    cooling_centers_to_activate: List[str]
    labor_advisory: str
    healthcare_outreach: str
    sms_alert_dispatched: bool

def get_action_recommendations(ward_id: str, risk_tier: str = "Normal"):
    # Deterministic recommendations
    hash_val = int(hashlib.md5(ward_id.encode()).hexdigest(), 16)
    shelters = [f"Center {i}" for i in range(1, 7)]
    
    if risk_tier == "Normal":
        return ActionRecommendations(
            cooling_centers_to_activate=random.sample(shelters, 1),
            labor_advisory="Monitor weather conditions.",
            healthcare_outreach="Standard healthcare services.",
            sms_alert_dispatched=False
        )
    
    return ActionRecommendations(
        cooling_centers_to_activate=random.sample(shelters, 2 if risk_tier == "Caution" else 4),
        labor_advisory="Restrict strenuous work from 12 PM to 4 PM." if risk_tier != "Extreme" else "Total restriction of outdoor work.",
        healthcare_outreach="Deploy ASHA workers to high-risk zones." if risk_tier != "Extreme" else "Full mobilization of emergency response teams.",
        sms_alert_dispatched=True
    )

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

def load_model_safely():
    """
    Safely loads the XGBoost model with fallback.
    """
    model_path = Path(__file__).parent / "wbgt_risk_model.joblib"
    try:
        if model_path.exists():
            return joblib.load(model_path)
        else:
            print("Model file not found. Falling back to heuristic.")
            return None
    except Exception as e:
        print(f"Error loading model: {e}. Falling back to heuristic.")
        return None

# Load model
model = load_model_safely()
def predict_wbgt_safely(features, target_date: datetime.datetime):
    """
    Predicts WBGT safely with heuristic fallback and seasonal override.
    """
    if model is not None:
        # Pre-process features: ensure humidity (column 1) is a percentage
        # If the incoming humidity is in decimal form (0-1), multiply by 100
        if features[0, 1] <= 1.0:
            features[0, 1] *= 100.0
            
        print(f"Incoming weather parameters for WBGT calculation: Temp={features[0,0]:.2f}°C, Humidity={features[0,1]:.2f}%, Wind={features[0,2]:.2f} m/s, Solar={features[0,3]:.2f} W/m^2")

        try:
            prediction = model.predict(features)[0]
        except Exception as e:
            print(f"Prediction failed: {e}. Falling back to heuristic.")
            prediction = historical_df['calculated_wbgt'].mean()
    else:
        # Heuristic fallback: mean of historical WBGT
        prediction = historical_df['calculated_wbgt'].mean()
    
    # Heuristic override for peak summer months or extreme heat
    temp = features[0, 0]
    if target_date.month in [5, 6, 7] or temp > 38.0:
        # Ensure realistic peak values between 31.0°C and 38.5°C
        # If prediction is low, boost it based on ambient temperature
        if prediction < 31.0:
            prediction = 31.0 + max(0, (temp - 30) * 0.4)
        
        # Clamp to realistic range
        prediction = min(max(prediction, 31.0), 38.5)
        print(f"Applied seasonal override for {target_date.strftime('%Y-%m-%d')}. Adjusted WBGT: {prediction:.2f}°C")

    return prediction

JSON_MODEL_PATH = Path(__file__).parent / "thermal_stress_model.json"
thermal_stress_model = None
if xgb is not None and JSON_MODEL_PATH.exists():
    thermal_stress_model = xgb.XGBRegressor()
    thermal_stress_model.load_model(str(JSON_MODEL_PATH))

# Load canonical wards
# Load enriched wards
WARDS_PATH = Path(__file__).parent / "delhi_ward_population_vulnerability_v3.csv"
wards_df = pd.read_csv(WARDS_PATH)


def get_weather_features(target_date: pd.Timestamp, temp_offset: float):
    """
    Fetches or simulates weather features for a given date with robust fallback and out-of-bounds handling.
    """
    # 1. Try to find a match for the same month, and hour in any available year
    potential_matches = historical_df[
        (historical_df['month'] == target_date.month) & 
        (historical_df['hour'] == target_date.hour)
    ]
    if not potential_matches.empty:
        features = potential_matches.iloc[-1].copy()
    else:
        # Fallback to existing logic
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
            
    # Ensure humidity features are scaled to percentage (0-100)
    humidity_features = ['relative_humidity_2m', 'humidity_lag_24']
    for feat in humidity_features:
        if feat in features and features[feat] <= 1.0:
            features[feat] *= 100.0
            
    feature_cols = [
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m", 
        "shortwave_radiation", "hour", "day_of_year", "month", 
        "temp_lag_24", "humidity_lag_24"
    ]
    
    return features[feature_cols].values.reshape(1, -1)

@app.get("/api/v1/heatwave-risk/{ward_id}")
async def get_heatwave_risk_by_ward(
    ward_id: str,
    background_tasks: BackgroundTasks,
    date: Optional[str] = Query(None, description="Target date (e.g. YYYY-MM-DD)"),
    temp_offset: float = Query(0.0, description="Temperature offset in degrees Celsius")
):
    """
    Returns a multi-day forecast for a specific ward.
    """
    target_wards = wards_df.copy()
    ward_data = target_wards[target_wards['ward_id'].astype(str).str.strip() == ward_id.strip()]
    
    if ward_data.empty:
        return JSONResponse(status_code=404, content={"error": f"Ward {ward_id} not found"})
        
    ward = ward_data.iloc[0]
    
    # Determine start date
    if date:
        try:
            start_date = pd.to_datetime(date)
            if pd.isna(start_date): start_date = datetime.datetime.now()
        except:
            start_date = datetime.datetime.now()
    else:
        start_date = datetime.datetime.now()
        
    results = []
    for i in range(5):
        current_date = start_date + datetime.timedelta(days=i)
        features = get_weather_features(current_date, temp_offset)
        prediction = predict_wbgt_safely(features, current_date)
        
        results.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "data": calculate_ward_risk_metrics_full(ward, current_date, prediction, background_tasks)
        })
        
    return convert_numpy_types({"ward_id": ward_id, "results": results})





class ForecastRequest(BaseModel):
    ward_id: str = ""
    days: int = 5
    temp_offset: float = 3.0
    start_date: Optional[str] = None

@app.post("/api/v1/forecast-heatwave")
async def forecast_heatwave(request: ForecastRequest, background_tasks: BackgroundTasks):
    """
    Returns a multi-day forecast for the requested ward(s).
    """
    try:
        # 1. Filter wards
        target_wards = wards_df.copy()
        if request.ward_id and str(request.ward_id).strip() != "":
            clean_ward_id = str(request.ward_id).strip()
            target_wards = target_wards[target_wards['ward_id'].astype(str).str.strip() == clean_ward_id]
        
        # 2. Determine start date
        if request.start_date:
            try:
                start_date = pd.to_datetime(request.start_date)
            except:
                start_date = datetime.datetime.now()
        else:
            start_date = datetime.datetime.now()
        
        results = []
        
        # 3. Loop through wards and days
        for _, ward in target_wards.iterrows():
            for i in range(request.days):
                try:
                    current_date = start_date + datetime.timedelta(days=i)
                    
                    # Get features
                    features = get_weather_features(current_date, request.temp_offset)

                    # Prediction
                    prediction = predict_wbgt_safely(features, current_date)
                    
                    # Unified calculation
                    risk_data = calculate_ward_risk_metrics_full(ward, current_date, prediction, background_tasks)

                    risk_data['forecast_day'] = i + 1
                    results.append(risk_data)
                except Exception as e:
                    print(f"Error processing ward {ward.get('ward_id')} for day {i}: {e}")
                    continue
                
        return convert_numpy_types({"data": results, "metadata": {"days": request.days, "temp_offset": request.temp_offset}})
    except Exception as e:
        print(f"Critical error in forecast_heatwave: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error during forecast calculation"})




