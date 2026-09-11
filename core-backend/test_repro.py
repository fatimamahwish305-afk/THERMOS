import pandas as pd
from pathlib import Path
import datetime
import random
import os

# Define the absolute path to the data
BASE_DIR = Path(r"c:\Users\dell\urban-heat-engine-mvp\core-backend")
DATA_PATH = BASE_DIR / "delhi_processed_wbgt.csv"

# Handle potential missing file for the test
if not DATA_PATH.exists():
    print(f"File {DATA_PATH} not found.")
    print(f"Current working directory: {os.getcwd()}")
    # List files in BASE_DIR to see if it's there
    if BASE_DIR.exists():
        print(f"Files in {BASE_DIR}: {os.listdir(BASE_DIR)}")
    exit(1)

historical_df = pd.read_csv(DATA_PATH)
historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])

print("Temperature Statistics:")
print(historical_df['temperature_2m'].describe())

# Check May data
may_data = historical_df[historical_df['timestamp'].dt.month == 5]
print("\nMay Temperature Statistics:")
print(may_data['temperature_2m'].describe())

print("\nSample May Data (first 5 rows):")
print(may_data.head())
