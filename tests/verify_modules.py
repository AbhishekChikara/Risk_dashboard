import sys
import os
import pandas as pd
import numpy as np
import json
import streamlit as st

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def serialize_df_info(df):
    if df is None: return None
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {k: str(v) for k,v in df.dtypes.items()},
        "head_hash": str(pd.util.hash_pandas_object(df.head()).sum())
    }

def verify_loader():
    print("Verifying utils.data_loader...")
    try:
        from utils.data_loader import load_data, process_data
        
        # Load Baseline
        with open("tests/baseline_output.json", "r") as f:
            baseline = json.load(f)
            
        ticker = "NVDA"
        returns_df, loadings_df, earnings_df = load_data(ticker)
        full_df, factor_cols = process_data(returns_df, loadings_df, ticker)
        
        current = {
            "returns_df": serialize_df_info(returns_df),
            "loadings_df": serialize_df_info(loadings_df),
            "earnings_df": serialize_df_info(earnings_df),
            "full_df": serialize_df_info(full_df),
            "factor_cols": factor_cols
        }
        
        # Deep Comparison
        diffs = []
        for key in baseline:
            # Hash comparison is the strongest check for identical data
            if baseline[key] != current[key]:
                diffs.append(f"Mismatch in {key}")
                print(f"--- {key} Mismatch ---")
                print(f"Baseline: {baseline[key]}")
                print(f"Current : {current[key]}")
        
        if not diffs:
            print("SUCCESS: utils.data_loader matches baseline perfectly.")
            return True
        else:
            print(f"FAILURE: Found differences: {diffs}")
            return False
            
    except Exception as e:
        print(f"Error verifying loader: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    verify_loader()
