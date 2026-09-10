import pandas as pd
import numpy as np
import os

# Path to the source CSV
source_path = r"C:\Users\dell\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\sessions\2CE3CCF0D70F4B10C28C9CFC73765619E29225C2\transfers\2026-37\delhi_ward_population_vulnerability_v2.csv"
target_path = "delhi_ward_population_vulnerability_v3.csv"

# Load the data
df = pd.read_csv(source_path)

# Generate realistic distributions
np.random.seed(42)
n = len(df)

# Influencing factors based on vulnerability_weight (normalized to 0-1)
# Some wards have weights up to 2.0 (like RAJ NAGAR), others are 1.0.
# We'll normalize based on min/max of the existing column.
v_min = df['vulnerability_weight'].min()
v_max = df['vulnerability_weight'].max()
v_norm = (df['vulnerability_weight'] - v_min) / (v_max - v_min)

# Generate correlated data
# Higher vulnerability -> Higher elderly ratio, higher outdoor worker density, lower green cover, higher lst, higher heat admissions
df['elderly_ratio'] = (5 + 15 * v_norm + np.random.normal(0, 2, n)).clip(5, 25).round(1)
df['outdoor_worker_density'] = (10 + 30 * v_norm + np.random.normal(0, 5, n)).clip(10, 60).round(1)
df['green_cover_pct'] = (40 - 20 * v_norm + np.random.normal(0, 5, n)).clip(5, 40).round(1)
df['lst_baseline'] = (35 + 5 * v_norm + np.random.normal(0, 1, n)).clip(35, 45).round(1)
df['historical_heat_admissions'] = (100 + 500 * v_norm + np.random.normal(0, 50, n)).clip(50, 1000).astype(int)

# Save to the local directory
df.to_csv(target_path, index=False)
print(f"Enriched CSV saved as {target_path}")
