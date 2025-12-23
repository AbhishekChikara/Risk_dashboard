import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.ai_summary import generate_ai_summary
from utils.calculations import calculate_beta_trends

def test_generation():
    print("Testing imports and function calls...")
    
    # Mock Data
    dates = pd.date_range(start='2023-01-01', periods=100)
    full_df = pd.DataFrame({
        'Date': dates,
        'NVDA': np.random.randn(100) * 0.02,
        'Market': np.random.randn(100) * 0.01,
        'Idiosyncratic_Return': np.random.randn(100) * 0.01,
        'Factor_Return': np.random.randn(100) * 0.01,
        'Rolling_Beta': np.random.randn(100) + 1.0,
        'Correlation_Regime': np.random.rand(100)
    })
    
    reaction_days = pd.DataFrame({
        'Date': dates[:5],
        'NVDA': [0.05, -0.02, 0.03, -0.05, 0.01],
        'Idiosyncratic_Return': [0.04, -0.01, 0.02, -0.04, 0.00],
        'Factor_Return': [0.01, -0.01, 0.01, -0.01, 0.01]
    })
    
    combined_events = pd.DataFrame({
        'Event_ID': ['E1']*10 + ['E2']*10,
        'Rel_Day': list(range(1, 11))*2,
        'NVDA': np.random.randn(20) * 0.01
    })
    
    loadings_df = pd.DataFrame({
        'Date': dates[-1:],
        'Market_Load': [1.2],
        'Value_Load': [-0.5]
    })
    
    # Test new calculations function directly
    print("Testing calculate_beta_trends...")
    trends = calculate_beta_trends(full_df)
    print(f"Beta Trends: {trends}")
    
    # Test main function (will fail at API call but should pass data prep)
    print("Testing generate_ai_summary (expecting API error, but successful data prep)...")
    try:
        report, usage = generate_ai_summary(
            api_key="TEST_KEY",
            stock_ticker="NVDA",
            full_df=full_df,
            reaction_days=reaction_days,
            latest_reaction=reaction_days.iloc[-1:],
            second_latest_reaction=reaction_days.iloc[-2:-1],
            factor_cols=['Market', 'Value'],
            combined_events=combined_events,
            rolling_stats=full_df,
            loadings_df=loadings_df
        )
        print("Function returned:", report[:50]) # Should be error string
    except Exception as e:
        print(f"Crashed: {e}")

if __name__ == "__main__":
    test_generation()
