import numpy as np
import pandas as pd
from train_risk_model import calculate_wet_bulb_stull, estimate_globe_temperature, compute_outdoor_wbgt

# Delhi May conditions
t_air = 40.0
rh = 40.0
solar_rad = 900.0
wind_speed = 3.0

tw = calculate_wet_bulb_stull(t_air, rh)
tg = estimate_globe_temperature(t_air, wind_speed, solar_rad)
wbgt = compute_outdoor_wbgt(tw, tg, t_air)

print(f"Tw: {tw}")
print(f"Tg: {tg}")
print(f"WBGT: {wbgt}")
