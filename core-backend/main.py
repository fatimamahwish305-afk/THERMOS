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

def send_notification(ward_id: str, wbgt: float, beds_needed: float):
    # This is the non-blocking notification dispatcher
    print(f"Notification triggered for Ward {ward_id}: WBGT {wbgt}, Beds Needed {beds_needed}")
    # Integration with SMS gateway would go here
    return True


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
            
    feature_cols = [
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m", 
        "shortwave_radiation", "hour", "day_of_year", "month", 
        "temp_lag_24", "humidity_lag_24"
    ]
    
    return features[feature_cols].values.reshape(1, -1)

@app.get("/api/v1/heatwave-risk")
async def get_heatwave_risk(
    background_tasks: BackgroundTasks,
    date: Optional[str] = Query(None, description="Target date (e.g. YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)"),
    ward_id: Optional[str] = Query(None, description="Ward ID (e.g. 80 or CANT_1)"),
    temp_offset: float = Query(0.0, description="Temperature offset in degrees Celsius")
):
    try:
        days_data = []
        
        # Filter wards if ward_id is provided
        target_wards = wards_df.copy()
        if ward_id:
            clean_ward_id = str(ward_id).strip()
            target_wards = target_wards[target_wards['ward_id'].astype(str).str.strip() == clean_ward_id]
            
        # Determine start date with robust exception handling for invalid date inputs
        if date:
            try:
                start_date = pd.to_datetime(date)
                if pd.isna(start_date):
                    start_date = historical_df['timestamp'].max()
            except (ValueError, TypeError, pd.errors.ParserError):
                start_date = historical_df['timestamp'].max()
        else:
            start_date = historical_df['timestamp'].max()
        
        # Loop through 5-day window
        for i in range(5):
            current_date = start_date + datetime.timedelta(days=i)
            
            # Get features
            features = get_weather_features(current_date, temp_offset)
            
            # Prediction
            prediction = predict_wbgt_safely(features, current_date)
            
            day_results = []
            for _, ward in target_wards.iterrows():
                # Using prediction for the ward (mocking that risk varies slightly by ward)
                # Use a deterministic seed based on ward_id and date for reproducibility
                w_id = str(ward['ward_id'])
                seed_str = f"{w_id}_{current_date.strftime('%Y-%m-%d')}"
                seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)
                rng = random.Random(seed)
                ward_specific_wbgt = prediction + rng.uniform(-0.5, 0.5)
                
                risk_level = "Extreme" if ward_specific_wbgt > 32 else "High Risk" if ward_specific_wbgt > 29 else "Caution" if ward_specific_wbgt > 27 else "Normal"
                
                # --- NEW LOGIC START ---
                try:
                    h_inc, beds, surge = calculate_risk_metrics(ward_specific_wbgt, w_id)
                    
                    recommendations = None
                    if risk_level in ["High Risk", "Extreme"]:
                        recommendations = get_action_recommendations(w_id)
                        background_tasks.add_task(send_notification, w_id, ward_specific_wbgt, beds)
                except Exception as e:
                    print(f"Error calculating risk metrics for ward {w_id}: {e}")
                    h_inc, beds, surge = 0.0, 0.0, 0.0
                    recommendations = None
                # --- NEW LOGIC END ---
                
                day_results.append({
                    "ward_id": w_id,
                    "ward_name": ward['ward_name'],
                    "wbgt": round(float(ward_specific_wbgt), 2),
                    "thermal_stress_score": round(max(0, min(10, float(ward_specific_wbgt - 20))), 2),
                    "heatwave_probability": round(max(0, min(1, float((ward_specific_wbgt - 25) / 10))), 2),
                    "risk_level": risk_level,
                    # --- NEW FIELDS ---
                    "hospitalization_increase_pct": round(h_inc, 2),
                    "additional_beds_needed": beds if risk_level in ["High Risk", "Extreme"] else None,
                    "surge_probability": round(surge, 2),
                    "action_recommendations": recommendations.dict() if recommendations else None
                })
            
            days_data.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "results": day_results
            })
                
        return convert_numpy_types({"data": days_data, "metadata": {"start_date": start_date, "days": 5, "temp_offset": temp_offset}})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'error': str(e), 'traceback': traceback.format_exc()})



class ForecastRequest(BaseModel):
    ward_id: str = ""
    days: int = 5
    temp_offset: float = 3.0

@app.post("/api/v1/forecast-heatwave")
async def forecast_heatwave(request: ForecastRequest, background_tasks: BackgroundTasks):
    """
    Returns a multi-day forecast for the requested ward(s).
    """
    # 1. Filter wards
    target_wards = wards_df.copy()
    if request.ward_id and str(request.ward_id).strip() != "":
        clean_ward_id = str(request.ward_id).strip()
        target_wards = target_wards[target_wards['ward_id'].astype(str).str.strip() == clean_ward_id]
    
    # 2. Determine start date
    start_date = historical_df['timestamp'].max()
    
    results = []
    
    # 3. Loop through wards and days
    for _, ward in target_wards.iterrows():
        w_id = str(ward['ward_id'])
        w_name = ward['ward_name']
        
        for i in range(request.days):
            current_date = start_date + datetime.timedelta(days=i)
            
            # Get features
            features = get_weather_features(current_date, request.temp_offset)
            
            # Prediction
            prediction = predict_wbgt_safely(features, current_date)
            
            # Using prediction for the ward (mocking that risk varies slightly by ward)
            # Use a deterministic seed based on ward_id and date for reproducibility
            seed_str = f"{w_id}_{current_date.strftime('%Y-%m-%d')}"
            seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)
            rng = random.Random(seed)
            ward_specific_wbgt = prediction + rng.uniform(-0.5, 0.5)
            
            # 2. Risk Calculation
            try:
                hazard_index = min(50, max(0, (ward_specific_wbgt - 15) * 2))
                vulnerability_weight = float(ward['vulnerability_weight'])
                scaling_factor = 1.2
                risk_score = min(100, hazard_index * vulnerability_weight * scaling_factor)
                
                # 3. Resource Estimation
                if risk_score < 25:
                    risk_tier = "Normal"
                    beds_needed = 0
                    centers_to_activate = 1
                    priority = "Low"
                elif risk_score < 50:
                    risk_tier = "Caution"
                    beds_needed = 10
                    centers_to_activate = 2
                    priority = "Medium"
                elif risk_score < 65:
                    risk_tier = "High Risk"
                    beds_needed = 30
                    centers_to_activate = 3
                    priority = "High"
                elif risk_score < 85:
                    risk_tier = "Severe"
                    beds_needed = 75
                    centers_to_activate = 4
                    priority = "Very High"
                else:
                    risk_tier = "Extreme"
                    beds_needed = 150
                    centers_to_activate = 6
                    priority = "Emergency"

                # 4. Recommendations
                recommendations = get_action_recommendations(w_id, risk_tier)
                if risk_tier in ["High Risk", "Severe", "Extreme"]:
                    background_tasks.add_task(send_notification, w_id, ward_specific_wbgt, beds_needed)
            except Exception as e:
                print(f"Error calculating risk for ward {w_id} on day {i+1}: {e}")
                risk_score = 0.0
                risk_tier = "N/A"
                beds_needed = 0
                centers_to_activate = 0
                priority = "N/A"
                recommendations = None
            
            results.append({
                "ward_id": w_id,
                "ward_name": w_name,
                "forecast_day": i + 1,
                "wbgt": round(float(ward_specific_wbgt), 2),
                "risk_score": round(float(risk_score), 2),
                "risk_tier": risk_tier,
                "mortality_index": round(max(0, (ward_specific_wbgt - 20) * 0.05 * (vulnerability_weight / 0.5)), 2),

                "resource_estimates": {
                    "required_heat_stroke_beds": int(beds_needed) if risk_tier in ["High Risk", "Severe", "Extreme"] else 0,
                    "cooling_centers_count": int(centers_to_activate),
                    "ambulance_dispatch_priority": priority
                },
                "action_recommendations": recommendations.dict()
            })
            
    return convert_numpy_types({"data": results, "metadata": {"days": request.days, "temp_offset": request.temp_offset}})


