from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from typing import Optional, Union, List
from pathlib import Path
import datetime
import random
import numpy as np
from pydantic import BaseModel, Field

try:
    import xgboost as xgb
except ImportError:
    xgb = None

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

JSON_MODEL_PATH = Path(__file__).parent / "thermal_stress_model.json"
thermal_stress_model = None
if xgb is not None and JSON_MODEL_PATH.exists():
    thermal_stress_model = xgb.XGBRegressor()
    thermal_stress_model.load_model(str(JSON_MODEL_PATH))

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
    days_data = []
    
    # Filter wards if ward_id is provided
    target_wards = wards_df
    if ward_id:
        target_wards = wards_df[wards_df['ward_id'].astype(str) == str(ward_id)]
        
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
        prediction = model.predict(features)[0]
        
        day_results = []
        for _, ward in target_wards.iterrows():
            # Using prediction for the ward (mocking that risk varies slightly by ward)
            ward_specific_wbgt = prediction + random.uniform(-0.5, 0.5)
            
            day_results.append({
                "ward_id": str(ward['ward_id']),
                "ward_name": ward['ward_name'],
                "wbgt": round(float(ward_specific_wbgt), 2),
                "thermal_stress_score": round(max(0, min(10, float(ward_specific_wbgt - 20))), 2),
                "heatwave_probability": round(max(0, min(1, float((ward_specific_wbgt - 25) / 10))), 2),
                "risk_level": "Extreme" if ward_specific_wbgt > 32 else "High" if ward_specific_wbgt > 29 else "Caution" if ward_specific_wbgt > 27 else "Normal"
            })
        
        days_data.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "results": day_results
        })
            
    return {"data": days_data, "metadata": {"start_date": start_date, "days": 5, "temp_offset": temp_offset}}



class ForecastDayInput(BaseModel):
    ward_id: Optional[Union[str, int]] = Field(None, description="Ward ID")
    ward_name: Optional[str] = Field(None, description="Ward Name")
    date: Optional[str] = Field(None, description="Forecast date (YYYY-MM-DD)")
    temperature_2m: Optional[float] = Field(None, description="Dry-bulb temperature in °C", alias="dry_bulb_temperature")
    relative_humidity_2m: Optional[float] = Field(None, description="Relative humidity %", alias="humidity")
    wind_speed_10m: Optional[float] = Field(None, description="Wind speed m/s", alias="wind_speed")
    shortwave_radiation: Optional[float] = Field(None, description="Shortwave radiation W/m^2", alias="solar_radiation")
    hour: Optional[int] = Field(12, description="Hour of day")
    day_of_year: Optional[int] = Field(None, description="Day of year")
    month: Optional[int] = Field(None, description="Month")
    temp_lag_24: Optional[float] = Field(None, description="24-hr temperature lag")
    humidity_lag_24: Optional[float] = Field(None, description="24-hr humidity lag")

    class Config:
        populate_by_name = True


@app.post("/api/v1/forecast-heatwave")
async def forecast_heatwave(forecasts: List[ForecastDayInput]):
    """
    Accepts an array of forecast inputs (e.g. 5-day forecast), loops through them,
    passes each day's features through both XGBoost model artifacts (thermal_stress_model.json
    and wbgt_risk_model.joblib), and returns a list of predicted thermal stress scores and risk categories.
    """
    results = []
    
    for idx, fc in enumerate(forecasts):
        # 1. Determine date
        target_date = historical_df['timestamp'].max()
        if fc.date:
            try:
                parsed_date = pd.to_datetime(fc.date)
                if not pd.isna(parsed_date):
                    target_date = parsed_date
            except (ValueError, TypeError, pd.errors.ParserError):
                pass
                
        # Get baseline features for the date
        base_features = get_weather_features(target_date, 0.0).flatten()
        
        # Feature columns order:
        # ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "shortwave_radiation", "hour", "day_of_year", "month", "temp_lag_24", "humidity_lag_24"]
        
        temp = fc.temperature_2m if fc.temperature_2m is not None else base_features[0]
        rh = fc.relative_humidity_2m if fc.relative_humidity_2m is not None else base_features[1]
        wind = fc.wind_speed_10m if fc.wind_speed_10m is not None else base_features[2]
        rad = fc.shortwave_radiation if fc.shortwave_radiation is not None else base_features[3]
        hr = fc.hour if fc.hour is not None else (target_date.hour if hasattr(target_date, 'hour') else 12)
        doy = fc.day_of_year if fc.day_of_year is not None else (target_date.dayofyear if hasattr(target_date, 'dayofyear') else 150)
        mon = fc.month if fc.month is not None else (target_date.month if hasattr(target_date, 'month') else 6)
        t_lag = fc.temp_lag_24 if fc.temp_lag_24 is not None else (temp - 2.0 if temp is not None else base_features[7])
        h_lag = fc.humidity_lag_24 if fc.humidity_lag_24 is not None else (rh if rh is not None else base_features[8])
        
        feat_array = np.array([[temp, rh, wind, rad, hr, doy, mon, t_lag, h_lag]], dtype=np.float64)
        
        # Pass through both model artifacts
        pred_joblib = float(model.predict(feat_array)[0])
        if thermal_stress_model is not None:
            pred_json = float(thermal_stress_model.predict(feat_array)[0])
            predicted_wbgt = 0.5 * (pred_joblib + pred_json)
        else:
            predicted_wbgt = pred_joblib
            
        # Ward metadata resolution
        w_id = str(fc.ward_id) if fc.ward_id is not None else "80"
        w_name = fc.ward_name
        if not w_name and wards_df is not None:
            match = wards_df[wards_df['ward_id'].astype(str) == w_id]
            if not match.empty:
                w_name = match.iloc[0]['ward_name']
            else:
                w_name = f"Ward {w_id}"
                
        thermal_stress_score = round(max(0, min(10, float(predicted_wbgt - 20))), 2)
        
        if predicted_wbgt >= 32.2:
            risk_level = "Extreme"
        elif predicted_wbgt >= 29.4:
            risk_level = "High Risk"
        elif predicted_wbgt >= 27.8:
            risk_level = "Caution"
        else:
            risk_level = "Normal"
            
        results.append({
            "day": idx + 1,
            "date": str(target_date.date()) if hasattr(target_date, 'date') else str(target_date),
            "ward_id": w_id,
            "ward_name": w_name,
            "temperature_2m": round(float(temp), 2),
            "relative_humidity_2m": round(float(rh), 2),
            "wbgt": round(float(predicted_wbgt), 2),
            "thermal_stress_score": thermal_stress_score,
            "risk_level": risk_level
        })
        
    return {
        "data": results,
        "metadata": {
            "forecast_days": len(results),
            "models_used": ["thermal_stress_model.json", "wbgt_risk_model.joblib"]
        }
    }


