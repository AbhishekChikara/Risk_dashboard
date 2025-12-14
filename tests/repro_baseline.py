import sys
import os
import pandas as pd
import numpy as np
import json

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from main_file import load_data, process_data
except ImportError:
    print("Could not import from main_file. Check path.")
    sys.exit(1)

def serialize_df_info(df):
    if df is None: return None
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {k: str(v) for k,v in df.dtypes.items()},
        "head_hash": str(pd.util.hash_pandas_object(df.head()).sum())
    }

def create_baseline():
    print("Creating Baseline from main_file.py...")
    ticker = "NVDA"
    try:
        # Load Data
        returns_df, loadings_df, earnings_df = load_data(ticker)
        
        # Process Data
        full_df, factor_cols = process_data(returns_df, loadings_df, ticker)
        
        baseline = {
            "returns_df": serialize_df_info(returns_df),
            "loadings_df": serialize_df_info(loadings_df),
            "earnings_df": serialize_df_info(earnings_df),
            "full_df": serialize_df_info(full_df),
            "factor_cols": factor_cols
        }
        
        with open("tests/baseline_output.json", "w") as f:
            json.dump(baseline, f, indent=2)
            
        print("Baseline created: tests/baseline_output.json")
        return True
        
    except Exception as e:
        print(f"Error creating baseline: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    create_baseline()
