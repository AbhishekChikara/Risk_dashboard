
import pandas as pd
import numpy as np
import os

# Re-use logic from main_file.py to ensure consistency
def load_data(stock_folder):
    base_path = os.path.join('res', stock_folder)
    def robust_read(filename):
        file_path = os.path.join(base_path, filename)
        try:
            return pd.read_csv(file_path, skiprows=2)
        except:
            return pd.read_csv(file_path)
    
    returns_df = robust_read("01_case_study_returns.csv")
    returns_df['Date'] = pd.to_datetime(returns_df['Date'], format='%m/%d/%y', errors='coerce')
    returns_df = returns_df.dropna(subset=['Date'])
    
    loadings_df = robust_read("02_case_study_factor_loadings.csv")
    loadings_df['Date'] = pd.to_datetime(loadings_df['Date'], format='mixed', errors='coerce')
    loadings_df = loadings_df.dropna(subset=['Date'])

    earnings_df = robust_read("03_case_study_earnings_dates.csv")
    earnings_df = earnings_df.iloc[:, 0].to_frame(name='Earnings Date')
    earnings_df['Earnings Date'] = pd.to_datetime(earnings_df['Earnings Date'], errors='coerce')
    earnings_df = earnings_df.dropna()
    
    return returns_df, loadings_df, earnings_df

def process_data(returns_df, loadings_df, stock_ticker):
    returns_df.columns = returns_df.columns.str.strip()
    loadings_df.columns = loadings_df.columns.str.strip()
    df = pd.merge(returns_df, loadings_df, on='Date', suffixes=('_Ret', '_Load'))
    base_cols = [c.replace('_Ret', '') for c in df.columns if '_Ret' in c and stock_ticker not in c and 'Date' not in c]
    df['Factor_Return'] = 0.0
    for col in base_cols:
        if f"{col}_Ret" in df.columns and f"{col}_Load" in df.columns:
            df[f'Contrib_{col}'] = df[f"{col}_Ret"] * df[f"{col}_Load"]
            df['Factor_Return'] += df[f'Contrib_{col}']
    df['Idiosyncratic_Return'] = df[stock_ticker] - df['Factor_Return']
    return df, base_cols

def get_event_window(df, earnings_date, window_days=5):
    try:
        df = df.sort_values('Date').reset_index(drop=True)
        idx_list = df.index[df['Date'] == earnings_date].tolist()
        if not idx_list: return None
        t0_idx = idx_list[0]
        start_idx = max(0, t0_idx - window_days)
        end_idx = min(len(df) - 1, t0_idx + window_days + 1)
        window_df = df.iloc[start_idx:end_idx].copy()
        window_df['Rel_Day'] = window_df.index - t0_idx
        return window_df
    except: return None

# --- Analysis ---
stock = "NVDA"
print(f"--- ANALYZING {stock} ---")
ret, load, earn = load_data(stock)
full_df, factors = process_data(ret, load, stock)

# 1. Window Sensitivity
print("\n--- 1. Window Sensitivity (Avg Abs Move) ---")
for w in [2, 5, 10]:
    moves = []
    for date in earn['Earnings Date']:
        win = get_event_window(full_df, date, w)
        if win is not None:
            # Cumulative return at end of window
            cum_ret = win[stock].sum() # Simple sum approximation for volatility magnitude check
            # Actually, let's look at Drift (T+2 to End)
            if len(win[win['Rel_Day'] > 1]) > 0:
                drift = win[(win['Rel_Day'] > 1) & (win['Rel_Day'] <= w)][stock].sum()
                moves.append(drift)
    
    avg_drift = np.mean(np.abs(moves))
    print(f"Window +/- {w} Days: Avg Post-Earnings Drift (Abs) = {avg_drift:.2%}")

# 2. Event Comparison (Latest 2)
print("\n--- 2. Event Comparison (Latest 2 Events) ---")
dates = earn['Earnings Date'].sort_values(ascending=False).head(2).values
for d in dates:
    d = pd.to_datetime(d)
    row = full_df[full_df['Date'] == d + pd.Timedelta(days=1)].iloc[0]
    total = row[stock]
    idio = row['Idiosyncratic_Return']
    factor = row['Factor_Return']
    print(f"Event {d.strftime('%Y-%m-%d')}: Total={total:.2%}, Idio={idio:.2%}, Factor={factor:.2%} ({factor/total:.0%} of move)")

# 3. Seasonality
print("\n--- 3. Seasonality ---")
reaction_days = []
for date in earn['Earnings Date']:
    win = get_event_window(full_df, date, 5)
    if win is not None:
        t1 = win[win['Rel_Day'] == 1]
        if not t1.empty:
            t1 = t1.copy()
            t1['Quarter'] = date.quarter
            reaction_days.append(t1)

if reaction_days:
    rd = pd.concat(reaction_days)
    season = rd.groupby('Quarter')[stock].apply(lambda x: x.abs().mean())
    print(season)
