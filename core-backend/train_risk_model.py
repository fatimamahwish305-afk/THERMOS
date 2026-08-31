"""
train_risk_model.py

1. Loads historical weather data (delhi_historical_meteorology.csv).
2. Cleans missing or null values cleanly.
3. Implements Stull's formula for Wet-Bulb Temperature (Tw).
4. Estimates Globe Temperature (Tg) using wind speed and solar radiation.
5. Computes target human thermal stress column:
     calculated_wbgt = 0.7 * Tw + 0.2 * Tg + 0.1 * T_air
6. Extracts temporal features (hour, day_of_year, month) and 24-hr lag features
   (temp_lag_24, humidity_lag_24), dropping initial NaNs.
7. Performs chronological train-test split (Train: 2020-2024, Test: 2025).
8. Trains and evaluates an ML regression model, saving the trained model artifact.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib 
try:
    import xgboost as xgb  # type: ignore
except ImportError:
    raise ImportError("xgboost is required. Install it with: pip install xgboost")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "delhi_historical_meteorology.csv"
MODEL_PATH = BASE_DIR / "wbgt_risk_model.joblib"
PROCESSED_DATA_PATH = BASE_DIR / "delhi_processed_wbgt.csv"


def calculate_wet_bulb_stull(temp_c: np.ndarray | pd.Series, rh_pct: np.ndarray | pd.Series) -> np.ndarray:
    """
    Computes Wet-Bulb Temperature (Tw in °C) using Stull's empirical formula (2011).
    Formula:
      Tw = T * atan(0.151977 * (RH + 8.313659)^0.5)
         + atan(T + RH)
         - atan(RH - 1.676331)
         + 0.00391838 * (RH^1.5) * atan(0.023101 * RH)
         - 4.686035
    """
    t = np.asarray(temp_c, dtype=np.float64)
    rh = np.clip(np.asarray(rh_pct, dtype=np.float64), 1.0, 100.0)

    term1 = t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
    term2 = np.arctan(t + rh)
    term3 = np.arctan(rh - 1.676331)
    term4 = 0.00391838 * (rh ** 1.5) * np.arctan(0.023101 * rh)
    term5 = 4.686035

    return term1 + term2 - term3 + term4 - term5


def estimate_globe_temperature(
    temp_c: np.ndarray | pd.Series,
    wind_speed: np.ndarray | pd.Series,
    solar_rad: np.ndarray | pd.Series,
) -> np.ndarray:
    """
    Estimates Globe Temperature (Tg in °C) using wind speed and solar radiation.
    Tg = T_air + 0.017 * solar_radiation - 0.208 * (wind_speed / 3.6)
    """
    t = np.asarray(temp_c, dtype=np.float64)
    wind = np.clip(np.asarray(wind_speed, dtype=np.float64), 0.0, None)
    rad = np.clip(np.asarray(solar_rad, dtype=np.float64), 0.0, None)

    tg = t + 0.017 * rad - 0.208 * (wind / 3.6)
    return np.maximum(tg, t - 2.0)


def compute_outdoor_wbgt(
    tw: np.ndarray | pd.Series,
    tg: np.ndarray | pd.Series,
    t_air: np.ndarray | pd.Series,
) -> np.ndarray:
    """
    Computes Outdoor Wet-Bulb Globe Temperature (WBGT) index in °C:
      calculated_wbgt = 0.7 * Tw + 0.2 * Tg + 0.1 * T_air
    """
    return 0.7 * tw + 0.2 * tg + 0.1 * t_air


def map_wbgt_to_category(wbgt_value: float) -> str:
    if wbgt_value < 27.8:
        return 'Normal'
    elif wbgt_value < 29.4:
        return 'Caution'
    elif wbgt_value < 32.2:
        return 'High Risk'
    else:
        return 'Extreme'




def load_and_preprocess_data(csv_path: Path) -> pd.DataFrame:
    """Loads CSV, handles missing values, and builds biometeorological & temporal features."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Meteorological CSV not found at: {csv_path}")

    print(f"[1/4] Loading meteorological dataset from: {csv_path.name}")
    df = pd.read_csv(csv_path)

    # Convert timestamp to datetime and sort
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Clean missing / null values cleanly using linear interpolation & ffill/bfill
    print("[2/4] Handling missing values cleanly...")
    numeric_cols = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "shortwave_radiation"]
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear").bfill().ffill()

    # Calculate biometeorology columns
    print("[3/4] Calculating Tw (Stull), Tg (Radiation/Wind), and calculated_wbgt...")
    df["Tw"] = calculate_wet_bulb_stull(df["temperature_2m"], df["relative_humidity_2m"])
    df["Tg"] = estimate_globe_temperature(
        df["temperature_2m"], df["wind_speed_10m"], df["shortwave_radiation"]
    )
    df["calculated_wbgt"] = compute_outdoor_wbgt(df["Tw"], df["Tg"], df["temperature_2m"])

    # Extract temporal features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["month"] = df["timestamp"].dt.month

    # Create 24-hour lagged features
    df["temp_lag_24"] = df["temperature_2m"].shift(24)
    df["humidity_lag_24"] = df["relative_humidity_2m"].shift(24)

    # Drop initial NaNs created by lag
    initial_rows = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"      Dropped initial {initial_rows - len(df)} NaN lag rows. Total valid rows: {len(df):,}")

    return df


def train_and_evaluate(df: pd.DataFrame):
    """Performs chronological train/test split, trains regression model, and evaluates on 2025."""
    feature_cols = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "shortwave_radiation",
        "hour",
        "day_of_year",
        "month",
        "temp_lag_24",
        "humidity_lag_24",
    ]
    target_col = "calculated_wbgt"

    # Chronological train-test split: 2020-2024 for train, 2025 for test
    train_mask = (df["timestamp"].dt.year >= 2020) & (df["timestamp"].dt.year <= 2024)
    test_mask = df["timestamp"].dt.year == 2025

    train_df = df[train_mask]
    test_df = df[test_mask]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    print("\n[4/4] Chronological Train-Test Split:")
    print(f"      Train Set (2020-2024): {len(train_df):,} records")
    print(f"      Test Set  (2025)     : {len(test_df):,} records")

    # Fit XGBRegressor model
    print("\nTraining XGBRegressor model on thermal risk data...")
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate on unseen 2025 data
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print("      MODEL EVALUATION ON TEST SET (2025)      ")
    print("=" * 55)
    print(f"  Mean Squared Error (MSE)    : {mse:.4f} °C^2")
    print(f"  Mean Absolute Error (MAE)   : {mae:.4f} °C")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f} °C")
    print(f"  R-squared (R^2) Score       : {r2:.4f}")
    print("=" * 55)

    # Classification metrics
    y_test_cats = y_test.apply(map_wbgt_to_category)
    y_pred_cats = pd.Series(y_pred, index=y_test.index).apply(map_wbgt_to_category)
    
    accuracy = accuracy_score(y_test_cats, y_pred_cats)
    report = classification_report(y_test_cats, y_pred_cats)
    
    print("\n      CLASSIFICATION REPORT (WBGT Categories)      ")
    print("=" * 55)
    print(f"  Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(report)
    print("=" * 55)
    
    # Side-by-side comparison
    print("\n      PREVIEW: ACTUAL VS PREDICTED (First 10)      ")
    print("=" * 70)
    comparison_df = pd.DataFrame({
        "Actual (°C)": y_test.values[:10],
        "Actual Cat": y_test_cats.values[:10],
        "Pred (°C)": y_pred[:10],
        "Pred Cat": y_pred_cats.values[:10]
    })
    print(comparison_df.to_string(index=False))
    print("=" * 70)
    
    # Categorical Distribution Comparison
    print("\n      CATEGORICAL DISTRIBUTION (Actual vs Predicted)      ")
    print("=" * 60)
    
    # Calculate counts
    actual_counts = y_test_cats.value_counts()
    pred_counts = y_pred_cats.value_counts()
    
    # Create DataFrame, fill missing with 0
    dist_df = pd.DataFrame({
        "Actual Count": actual_counts,
        "Pred Count": pred_counts
    })
    
    # Ensure correct category order
    categories = ['Normal', 'Caution', 'High Risk', 'Extreme']
    
    dist_df = dist_df.reindex(categories).fillna(0).astype(int)
    
    print(dist_df.to_string())
    print("=" * 60)

    # Save artifacts
    model.save_model("core-backend/thermal_stress_model.json")
    joblib.dump(model,"core-backend/wbgt_risk_model.joblib")
    print(f"\nTrained risk model weights saved to: core-backend/thermal_stress_model.json")
    
    # Generate and save plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual WBGT (°C)')
    plt.ylabel('Predicted WBGT (°C)')
    plt.title('Actual vs Predicted WBGT')
    plt.savefig("core-backend/model_evaluation_visualisation.png")
    plt.close()
    print(f"Evaluation plot saved to: core-backend/model_evaluation_visualisation.png")

    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed dataset with calculated_wbgt saved to: {PROCESSED_DATA_PATH}")

    return model


def main():
    df = load_and_preprocess_data(DATA_PATH)
    train_and_evaluate(df)


if __name__ == "__main__":
    main()
