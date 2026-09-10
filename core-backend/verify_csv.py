import pandas as pd
from pathlib import Path
import os

# Define the path to the CSV as main.py does
csv_path = Path(__file__).parent / "delhi_ward_population_vulnerability_v3.csv"

# Attempt to load the dataframe
try:
    wards_df = pd.read_csv(csv_path)
    print(f"Successfully loaded {len(wards_df)} wards.")
    print("Columns:", wards_df.columns.tolist())
    print("Sample data (first row):")
    print(wards_df.iloc[0])
except Exception as e:
    print(f"Error loading CSV: {e}")
