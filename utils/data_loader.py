import os
import pandas as pd
import streamlit as st
import numpy as np

# --- Dynamic Stock Folder Logic ---
def get_stock_list(path='res'):
    """Scans for subdirectories which are assumed to be stock tickers."""
    if not os.path.exists(path):
        os.makedirs(path)
        return []
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and not d.startswith('.')]

# --- Data Loading Functions ---
@st.cache_data
def load_data(stock_folder):
    """Loads data for a specific stock from its folder."""
    base_path = os.path.join('res', stock_folder)

    # Helper to robustly load CSVs
    def robust_read(filename):
        file_path = os.path.join(base_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{filename}")
        try:
            # Try skipping header info (common in financial exports)
            return pd.read_csv(file_path, skiprows=2)
        except:
            # Fallback to standard read
            return pd.read_csv(file_path)

    # Load Returns
    returns_df = robust_read("01_case_study_returns.csv")
    returns_df['Date'] = pd.to_datetime(returns_df['Date'], format='%m/%d/%y', errors='coerce')
    returns_df = returns_df.dropna(subset=['Date'])
    
    # Load Loadings
    loadings_df = robust_read("02_case_study_factor_loadings.csv")
    loadings_df['Date'] = pd.to_datetime(loadings_df['Date'], format='mixed', errors='coerce')
    loadings_df = loadings_df.dropna(subset=['Date'])

    # Load Earnings Dates
    earnings_df = robust_read("03_case_study_earnings_dates.csv")
    # Extract the first column regardless of name and clean it
    earnings_df = earnings_df.iloc[:, 0].to_frame(name='Earnings Date')
    earnings_df['Earnings Date'] = pd.to_datetime(earnings_df['Earnings Date'], errors='coerce')
    earnings_df = earnings_df.dropna()
    
    return returns_df, loadings_df, earnings_df

# --- Data Processing ---
def process_data(returns_df, loadings_df, stock_ticker):
    # Ensure column names are clean
    returns_df.columns = returns_df.columns.str.strip()
    loadings_df.columns = loadings_df.columns.str.strip()

    # Merge Returns and Loadings
    df = pd.merge(returns_df, loadings_df, on='Date', suffixes=('_Ret', '_Load'))
    
    # Identify Factor Columns
    base_cols = [c.replace('_Ret', '') for c in df.columns if '_Ret' in c and stock_ticker not in c and 'Date' not in c]
    
    # Calculate Factor Contribution
    df['Factor_Return'] = 0.0
    for col in base_cols:
        ret_col = f"{col}_Ret"
        load_col = f"{col}_Load"
        if ret_col in df.columns and load_col in df.columns:
            df[f'Contrib_{col}'] = df[ret_col] * df[load_col]
            df['Factor_Return'] += df[f'Contrib_{col}']
            
    # Calculate Idiosyncratic Return (Alpha)
    df['Idiosyncratic_Return'] = df[stock_ticker] - df['Factor_Return']
    
    return df, base_cols

# --- Event Study Logic ---
def get_event_window(df, earnings_date, window_days=5):
    try:
        df = df.sort_values('Date').reset_index(drop=True)
        idx_list = df.index[df['Date'] == earnings_date].tolist()
        
        if not idx_list:
            return None
            
        t0_idx = idx_list[0]
        
        # Check bounds
        start_idx = max(0, t0_idx - window_days)
        end_idx = min(len(df) - 1, t0_idx + window_days + 1)
        
        window_df = df.iloc[start_idx:end_idx].copy()
        window_df['Rel_Day'] = window_df.index - t0_idx
        
        return window_df
    except Exception as e:
        return None
