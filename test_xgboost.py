try:
    import xgboost as xgb
    # Access it to satisfy Pylance
    print(f"xgboost is installed, version: {xgb.__version__}")
except ImportError:
    print("xgboost is NOT installed")
