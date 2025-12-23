import pandas as pd
import numpy as np
import streamlit as st
from arch import arch_model
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm

# --- GARCH Volatility Forecast Function ---
@st.cache_data
def get_garch_forecast(returns_series):
    """
    Fits a GARCH(1,1) model and returns the one-step-ahead daily volatility forecast.
    Returns forecast and a status message.
    """
    try:
        # GARCH models often converge better when returns are scaled (e.g., by 100)
        garch_model = arch_model(returns_series * 100, vol='Garch', p=1, q=1, dist='Normal')
        res = garch_model.fit(disp='off') # disp='off' suppresses convergence output
        
        # Forecast one step ahead
        forecast = res.forecast(horizon=1)
        
        # The output is variance, so we need the square root for volatility. Also scale back from %.
        predicted_vol = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
        
        return predicted_vol, "GARCH(1,1) Forecast"
    except Exception as e:
        # Fallback to simple 30-day rolling standard deviation if GARCH fails
        fallback_vol = returns_series.rolling(30).std().iloc[-1]
        return fallback_vol, "GARCH Failed, Fallback to 30D StdDev"

# --- Rolling Beta & Correlation Regime Logic ---
@st.cache_data
def calculate_rolling_stats(returns_df, stock_ticker, factor_cols, window=60):
    """
    Calculates:
    1. Rolling Beta of Stock vs Market (requires 'Market' in factor_cols).
    2. Rolling Correlation Regime (Average pairwise correlation of all factors).
    """
    # Create a clean dataframe with Date as index for rolling ops
    df = returns_df.copy().sort_values('Date')
    df = df.set_index('Date') 
    
    results = pd.DataFrame(index=df.index)
    # We want 'Date' as a column in results for merging later, or we just rely on index
    results['Date'] = df.index
    
    # 1. Rolling Beta (Stock vs Market)
    if 'Market' in factor_cols and stock_ticker in df.columns and 'Market' in df.columns:
        try:
            exog = sm.add_constant(df['Market'])
            endog = df[stock_ticker]
            rols = RollingOLS(endog, exog, window=window)
            rres = rols.fit()
            # Extract beta (coefficient for Market)
            params = rres.params
            if 'Market' in params.columns:
                results['Rolling_Beta'] = params['Market']
        except Exception as e:
            results['Rolling_Beta'] = np.nan
    
    # 2. Rolling Correlation Regime
    try:
        factor_returns = df[factor_cols]
        # Rolling correlation returns a MultiIndex: (Date, Factor)
        rolling_corr = factor_returns.rolling(window=window).corr()
        
        avg_corrs = []
        # Iterate over the dates in the index
        for date in df.index:
            if date in rolling_corr.index:
                # .loc[date] on a MultiIndex (Date, Col) returns the DataFrame for that Date
                matrix = rolling_corr.loc[date]
                
                # Get off-diagonal elements
                mask = np.ones(matrix.shape, dtype=bool)
                np.fill_diagonal(mask, 0)
                
                off_diag = matrix.where(mask)
                avg_abs_corr = off_diag.abs().mean().mean()
                avg_corrs.append(avg_abs_corr)
            else:
                avg_corrs.append(np.nan)
        
        results['Correlation_Regime'] = avg_corrs
            
    except Exception as e:
        results['Correlation_Regime'] = np.nan

    return results.reset_index(drop=True)

@st.cache_data
def calculate_volatility_profiles(full_df, stock_ticker, earnings_dates, window_days=30):
    """
    Calculates average realized volatility profiles (Total, Idio, Factor) around earnings events.
    Returns a DataFrame with columns: ['Rel_Day', 'Total_Vol', 'Idio_Vol', 'Factor_Vol']
    """
    # Work on a copy and ensure sort
    df = full_df.copy().sort_values('Date')
    
    # 1. Calculate Realized Volatility (21-day rolling, annualized)
    # We use 21 days (~1 trading month) to be sensitive enough to moves but smooth enough.
    # Scale by sqrt(252) to annualize.
    
    # Ensure cols exist
    if 'Idiosyncratic_Return' not in df.columns or 'Factor_Return' not in df.columns:
        return None

    # Rolling Std Dev of Returns
    df['Total_Vol'] = df[stock_ticker].rolling(21).std() * np.sqrt(252)
    df['Idio_Vol'] = df['Idiosyncratic_Return'].rolling(21).std() * np.sqrt(252)
    df['Factor_Vol'] = df['Factor_Return'].rolling(21).std() * np.sqrt(252)
    
    # Reset index for safe slicing
    df = df.reset_index(drop=True)
    
    # 2. Extract Event Windows
    profiles = []
    
    # Identify indices of earnings dates
    # earnings_dates is likely a Series or list of Timestamps.
    # df['Date'] is datetime64[ns]
    event_indices = df[df['Date'].isin(earnings_dates)].index
    
    for idx in event_indices:
        start = idx - window_days
        end = idx + window_days + 1
        
        # Check bounds
        if start < 0 or end > len(df):
            continue
            
        slice_df = df.iloc[start:end].copy()
        slice_df['Rel_Day'] = np.arange(-window_days, window_days + 1)
        
        profiles.append(slice_df[['Rel_Day', 'Total_Vol', 'Idio_Vol', 'Factor_Vol']])
        
    if not profiles:
        return None
        
    # 3. Aggregate across all events
    combined = pd.concat(profiles)
    avg_profile = combined.groupby('Rel_Day')[['Total_Vol', 'Idio_Vol', 'Factor_Vol']].mean().reset_index()
    
    return avg_profile

@st.cache_data
def calculate_fade_strategy(full_df, earnings_dates, stock_ticker, hold_days=5):
    """
    Backtests a "Fade the Move" strategy using User's Final Logic:
    - Logic: Identify Day 0 (Reaction Day).
    - Signal: If Day 0 > 0 -> Short (-1). If Day 0 < 0 -> Long (+1).
    - Return: Geometric compounding of next 'hold_days' returns.
    """
    df = full_df.copy().sort_values('Date').reset_index(drop=True)
    trades = []
    
    # User Logic: Iterate dates and find Day 0
    # earnings_dates is a Series of timestamps
    for ed in earnings_dates:
        # Find Day 0 (First trading day > earnings date)
        # We need the index in 'df'
        mask = df['Date'] > ed
        if not mask.any():
            continue
            
        reaction_idx = df[mask].index[0]
        day_0_date = df.loc[reaction_idx, 'Date']
        
        # Day 0 Move (Reaction)
        day_0_return = df.loc[reaction_idx, stock_ticker]
        
        # Determine Direction (FADE Strategy)
        # Snippet was Momentum (1 if > 0), we invert for Fade (-1 if > 0)
        direction = -1 if day_0_return > 0 else 1
        trade_type = "SHORT" if direction == -1 else "LONG"
        
        # Define Holding Window (T+1 to T+hold_days) relative to Day 0
        # Start: reaction_idx + 1
        start_pos = reaction_idx + 1
        end_pos = start_pos + hold_days
        
        if end_pos > len(df):
            continue
            
        # GEOMETRIC COMPONENT: Calculate actual P&L
        window_rets = df.iloc[start_pos:end_pos][stock_ticker]
        if window_rets.empty:
            continue
            
        cumulative_drift = (1 + window_rets).prod() - 1
        
        # Strategy P&L
        strategy_pnl = direction * cumulative_drift
        
        trades.append({
            'Date': df.loc[start_pos, 'Date'] if start_pos < len(df) else day_0_date, # Trade Entry Date (T+1)
            'Event_Date': ed,
            'Reaction': day_0_return,
            'Direction': trade_type,
            'Hold_Return': strategy_pnl, # This corresponds to '5_Day_PL' in snippet
            'Cumulative_Return': 0.0 # Placeholder, calc later
        })
        
    if not trades:
        return None, None
        
    trades_df = pd.DataFrame(trades)
    trades_df['Cumulative_Return'] = trades_df['Hold_Return'].cumsum()
    
    # Stats
    wins = (trades_df['Hold_Return'] > 0).sum()
    total = len(trades_df)
    win_rate = wins / total if total > 0 else 0
    avg_return = trades_df['Hold_Return'].mean()
    total_return = trades_df['Hold_Return'].sum()
    
    stats = {
        'Win Rate': win_rate,
        'Avg Return': avg_return,
        'Total Return': total_return,
        'Trade Count': total
    }
    
    return stats, trades_df
    return stats, trades_df

@st.cache_data
def calculate_beta_trends(rolling_stats):
    """
    Calculates the slope of the Rolling Beta over 10, 30, and 60 days.
    Returns a list of formatted strings.
    """
    beta_trends = []
    if 'Rolling_Beta' in rolling_stats.columns:
        for w in [10, 30, 60]:
            if len(rolling_stats) >= w:
                try:
                    y = rolling_stats['Rolling_Beta'].iloc[-w:].values
                    x = np.arange(len(y))
                    # Check for NaNs
                    if np.isnan(y).any():
                        continue
                    slope = np.polyfit(x, y, 1)[0]
                    
                    trend_desc = "Stable"
                    if slope > 0.005: trend_desc = "Rising"
                    elif slope < -0.005: trend_desc = "Falling"
                    
                    beta_trends.append(f"{w}d: {trend_desc} (Slope {slope:.4f})")
                except:
                    continue
    return beta_trends

@st.cache_data
def interpret_volatility_ramp(vol_profile):
    """
    Interprets the volatility profile (ramp up / crush down) and returns a summary string.
    """
    vol_ramp_str = "No Volatility Profile Data Available"
    if vol_profile is not None and not vol_profile.empty:
        try:
            days = vol_profile['Rel_Day'].values
            
            # T-Start (Closest to -30, or min)
            start_day = -30
            if -30 not in days:
                neg_days = days[days < 0]
                if len(neg_days) > 0:
                    start_day = neg_days.min()
                else:
                    start_day = days.min()
            
            # T-End (Closest to +10, or max)
            end_day = 10
            if 10 not in days:
                pos_days = days[days > 0]
                if len(pos_days) > 0:
                    end_day = pos_days.max() 
                else:
                    end_day = days.max()
            
            # Get Values
            t_start_val = vol_profile[vol_profile['Rel_Day'] == start_day]['Total_Vol'].iloc[0]
            
            # Get T=0
            t_0_val = 0
            t_0_idio = 0
            if 0 in days:
                 t_0_val = vol_profile[vol_profile['Rel_Day'] == 0]['Total_Vol'].iloc[0]
                 t_0_idio = vol_profile[vol_profile['Rel_Day'] == 0]['Idio_Vol'].iloc[0]
            else:
                idx_0 = np.abs(days).argmin()
                day_0 = days[idx_0]
                t_0_val = vol_profile[vol_profile['Rel_Day'] == day_0]['Total_Vol'].iloc[0]
                t_0_idio = vol_profile[vol_profile['Rel_Day'] == day_0]['Idio_Vol'].iloc[0]

            t_end_val = vol_profile[vol_profile['Rel_Day'] == end_day]['Total_Vol'].iloc[0]
            
            t_0_idio_ratio = t_0_idio / t_0_val if t_0_val > 0 else 0
            
            # Calc Ramps
            ramp_pct = (t_0_val - t_start_val) / t_start_val if t_start_val > 0 else 0
            crush_pct = (t_end_val - t_0_val) / t_0_val if t_0_val > 0 else 0
            
            vol_ramp_str = f"- Pre-Event Ramp (T{start_day} to T0): {ramp_pct:+.1%}\\n- Post-Event Crush (T0 to T+{end_day}): {crush_pct:+.1%}\\n- Risk Composition at Event: {t_0_idio_ratio:.0%} Idiosyncratic / {1-t_0_idio_ratio:.0%} Systematic"
        except Exception as e:
            vol_ramp_str = f"Error interpreting Volatility Profile: {e}"
    else:
         vol_ramp_str = "Volatility Profile Data is Empty"
         
    return vol_ramp_str

@st.cache_data
def analyze_post_event_drift(combined_events, stock_ticker):
    """
    Analyzes post-event price drift to identify Momentum vs Reversion.
    Returns formatted string and drift correlation.
    """
    drift_str = "N/A"
    drift_corr = 0.0
    
    if not combined_events.empty:
        drifts = []
        signs_match = []
        t1_moves = []
        drift_moves = []
        
        for eid, grp in combined_events.groupby('Event_ID'):
            try:
                # Dynamic Drift Calculation
                if 1 not in grp['Rel_Day'].values:
                    continue
                    
                max_day = grp['Rel_Day'].max()
                if max_day <= 1:
                    continue
                    
                p1 = grp[grp['Rel_Day'] == 1][stock_ticker].values[0]
                p_end = grp[grp['Rel_Day'] == max_day][stock_ticker].values[0]
                
                # For Prompt Text
                drift_val = p_end - p1
                drifts.append(drift_val)
                
                if (p1 > 0 and drift_val > 0) or (p1 < 0 and drift_val < 0):
                    signs_match.append(1) # Amplification
                else:
                    signs_match.append(0) # Reversal
                
                # For Correlation Metric (T+5 fixed if possible, else max)
                # Try to get exactly T+5
                if 5 in grp['Rel_Day'].values:
                    p5 = grp[grp['Rel_Day'] == 5][stock_ticker].values[0]
                    t1_moves.append(p1)
                    drift_moves.append(p5 - p1)
                else:
                    # Fallback to max day
                    t1_moves.append(p1)
                    drift_moves.append(drift_val)
                    
            except:
                continue
        
        if drifts:
            avg_drift = np.mean(drifts)
            match_pct = np.mean(signs_match)
            tendency = "MOMENTUM / AMPLIFICATION" if match_pct > 0.5 else "MEAN REVERSION / DAMPING"
            drift_str = f"{avg_drift:+.1%} (Avg T+1 to End of Window). Tendency: {tendency} ({match_pct:.0%} of time trajectory continues)."
            
            if len(t1_moves) > 2:
                drift_corr = np.corrcoef(t1_moves, drift_moves)[0, 1]
                if np.isnan(drift_corr): drift_corr = 0.0
                
    return drift_str, drift_corr

@st.cache_data
def calculate_factor_contribution(full_df, loadings_df, factor_cols):
    """
    Calculates the percentage contribution of each factor to the total SYSTEMATIC variance.
    Returns a dictionary: {Factor: Contribution_Pct}
    """
    contrib_dict = {}
    try:
        # Merge Returns and Loadings on Date
        # full_df has returns (e.g. 'Momentum'), loadings_df has 'Momentum' or 'Momentum_Load'
        # We need to align them carefully.
        
        # 1. Clean Dataframes
        # Ensure Date index alignment
        returns = full_df.copy().set_index('Date').sort_index()
        loadings = loadings_df.copy().set_index('Date').sort_index()
        
        # Intersect dates
        common_dates = returns.index.intersection(loadings.index)
        if len(common_dates) < 30: # Need some history
            return {}
            
        returns = returns.loc[common_dates]
        loadings = loadings.loc[common_dates]
        
        var_contribs = {}
        
        for f in factor_cols:
            # Find Return Col (Try base, then _Ret)
            ret_col = f
            if f not in returns.columns:
                 if f"{f}_Ret" in returns.columns:
                     ret_col = f"{f}_Ret"
            
            # Find Loading Col
            load_col = f
            if f not in loadings.columns:
                load_col = f"{f}_Load"
                
            if ret_col in returns.columns and load_col in loadings.columns:
                # Calculate Component Return: Beta(t) * FactorRet(t)
                component_ret = loadings[load_col] * returns[ret_col]
                # Variance of this component
                var_contribs[f] = component_ret.var()
                
        total_sys_var = sum(var_contribs.values())
        
        if total_sys_var > 0:
            for f, v in var_contribs.items():
                contrib_dict[f] = v / total_sys_var
                
    except Exception as e:
        return {}
        
    return contrib_dict

@st.cache_data
def calculate_factor_crowding(loadings_df, factor_cols):
    """
    Calculates the Z-Score of the current factor loadings vs their historical average (Crowding).
    Returns a dictionary: {Factor: Z_Score}
    """
    crowding_dict = {}
    try:
        df = loadings_df.copy().sort_values('Date').set_index('Date')
        
        for f in factor_cols:
            col = f
            if f not in df.columns:
                col = f"{f}_Load"
            
            if col in df.columns:
                series = df[col]
                if len(series) > 30:
                    # Use rolling window (e.g. 1 year / 252 days) or expanding if shorter
                    window = 252
                    
                    # Current Value
                    current_val = series.iloc[-1]
                    
                    # Historical Stats (excluding current if possible, or usually just rolling includes current)
                    mean = series.rolling(window, min_periods=30).mean().iloc[-1]
                    std = series.rolling(window, min_periods=30).std().iloc[-1]
                    
                    if std > 0:
                        z_score = (current_val - mean) / std
                        crowding_dict[f] = z_score
    except Exception as e:
        return {}
        
    return crowding_dict
