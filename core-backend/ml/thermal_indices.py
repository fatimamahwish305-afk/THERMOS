import numpy as np
import pandas as pd

def calculate_wet_bulb_stull(temp_c: np.ndarray | pd.Series, rh_pct: np.ndarray | pd.Series) -> np.ndarray:
    """
    Computes Wet-Bulb Temperature (Tw in °C) using Stull's empirical formula (2011).
    """
    rh_input = np.asarray(rh_pct, dtype=np.float64)
    # Ensure RH is treated as a percentage (0-100)
    # If values are <= 1, assume they are fractions and multiply by 100
    rh = np.where(rh_input <= 1.0, rh_input * 100.0, rh_input)
    rh = np.clip(rh, 1.0, 100.0)
    t = np.asarray(temp_c, dtype=np.float64)

    term1 = t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
    term2 = np.arctan(t + rh)
    term3 = np.arctan(rh - 1.676331)
    term4 = 0.00391838 * (rh ** 1.5) * np.arctan(0.023101 * rh)
    term5 = 4.686035

    return term1 + term2 - term3 + term4 - term5

def estimate_globe_temperature(temp_c, wind_speed, solar_rad):
    t = np.asarray(temp_c, dtype=np.float64)
    wind = np.clip(np.asarray(wind_speed, dtype=np.float64), 0.0, None)
    
    # Handle missing or zeroed solar radiation defensively
    rad_input = np.asarray(solar_rad, dtype=np.float64)
    rad = np.where((rad_input <= 0.0) | np.isnan(rad_input), 500.0, rad_input) # Default daytime proxy if missing
    rad = np.clip(rad, 0.0, 1200.0)
    
    tg = t + 0.025 * rad - 0.208 * wind
    return np.maximum(tg, t - 2.0)

def compute_outdoor_wbgt(t_air, rh, wind_speed, solar_rad):
    """
    Unified outdoor WBGT computation under direct sunlight (ISO 7243): 
    Calculates Tw, Tg, then applies 0.7/0.2/0.1 weighting.
    """
    # Debug statements
    try:
        t_val = float(t_air[0]) if hasattr(t_air, '__getitem__') else float(t_air)
        rh_val = float(rh[0]) if hasattr(rh, '__getitem__') else float(rh)
        w_val = float(wind_speed[0]) if hasattr(wind_speed, '__getitem__') else float(wind_speed)
        s_val = float(solar_rad[0]) if hasattr(solar_rad, '__getitem__') else float(solar_rad)
        print(f"DEBUG compute_outdoor_wbgt: t_air={t_val:.2f}, rh={rh_val:.2f}, wind={w_val:.2f}, solar={s_val:.2f}")
    except Exception as e:
        print(f"DEBUG compute_outdoor_wbgt: Could not print inputs: {e}")

    tw = calculate_wet_bulb_stull(t_air, rh)
    tg = estimate_globe_temperature(t_air, wind_speed, solar_rad)
    return 0.7 * tw + 0.2 * tg + 0.1 * t_air

