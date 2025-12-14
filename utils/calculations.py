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
