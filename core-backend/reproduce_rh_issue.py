import numpy as np
from train_risk_model import calculate_wet_bulb_stull

# Test case: Temp=30C, Humidity=0.5 (50%)
t = 30.0
rh_fraction = 0.5
rh_percent = 50.0

tw_from_fraction = calculate_wet_bulb_stull(t, rh_fraction)
tw_from_percent = calculate_wet_bulb_stull(t, rh_percent)

print(f"Tw from fraction (0.5): {tw_from_fraction}")
print(f"Tw from percent (50.0): {tw_from_percent}")
