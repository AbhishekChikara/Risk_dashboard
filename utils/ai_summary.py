import google.generativeai as genai
import streamlit as st
import pandas as pd
import numpy as np

# --- AI Model Helper ---
@st.cache_data
def get_available_models(api_key):
    """Fetches available Gemini models from the API."""
    try:
        genai.configure(api_key=api_key)
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                models.append(m.name)
        # Sort to put likely best models first
        models.sort(key=lambda x: ('gemini' not in x, x))
        return models
    except Exception as e:
        return []

def generate_ai_summary(api_key, stock_ticker, full_df, reaction_days, latest_reaction, second_latest_reaction, factor_cols, combined_events, rolling_stats, loadings_df, model_name='models/gemini-1.5-flash'):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        return f"Error configuring AI model: {e}"

    # --- DATA & METRICS PREPARATION ---
    
    # 1. Stock Overview
    latest_date = full_df['Date'].max()
    # Calculate returns over last 60 days (Reporting Period)
    period_ret = full_df.iloc[-60:][stock_ticker].sum() # approx log return sum
    
    # 2. Factor Exposure Summary & Trends
    # Get latest loadings
    curr_loadings = loadings_df.sort_values('Date').iloc[-1]
    
    factor_vals = {}
    for f in factor_cols:
        # Try finding the column in loadings_df
        if f in loadings_df.columns:
            factor_vals[f] = curr_loadings[f]
        elif f"{f}_Load" in loadings_df.columns:
            factor_vals[f] = curr_loadings[f"{f}_Load"]

    sorted_factors = sorted(factor_vals.items(), key=lambda x: x[1], reverse=True)
    top_3_pos = [f"{k} ({v:.2f})" for k, v in sorted_factors if v > 0][:3]
    top_3_neg = [f"{k} ({v:.2f})" for k, v in sorted_factors if v < 0][-3:]
    
    # Beta Trends (Slope over 10, 30, 60 days)
    beta_trends = []
    if 'Rolling_Beta' in rolling_stats.columns:
        for w in [10, 30, 60]:
            if len(rolling_stats) >= w:
                y = rolling_stats['Rolling_Beta'].iloc[-w:].values
                x = np.arange(len(y))
                slope = np.polyfit(x, y, 1)[0]
                trend_desc = "Stable"
                if slope > 0.005: trend_desc = "Rising"
                elif slope < -0.005: trend_desc = "Falling"
                beta_trends.append(f"{w}d: {trend_desc} (Slope {slope:.4f})")
    
    # 3. Risk Decomposition
    window_60 = full_df.iloc[-60:].copy()
    var_factor = window_60['Factor_Return'].var() * 252
    var_idio = window_60['Idiosyncratic_Return'].var() * 252
    total_vol = np.sqrt(var_factor + var_idio)
    pct_idio = var_idio / (var_factor + var_idio) if total_vol > 0 else 0
    
    # 5. Earnings Diagnostics (Latest Event)
    # latest_reaction has the T+1 move
    if not latest_reaction.empty:
        row = latest_reaction.iloc[0]
        evt_ret = row[stock_ticker]
        evt_alpha = row['Idiosyncratic_Return']
        # Sigma Calc
        pre_vol = full_df[full_df['Date'] < row['Date']][stock_ticker].rolling(30).std().iloc[-1]
        sigma = abs(evt_ret / pre_vol) if pre_vol > 0 else 0
    else:
        evt_ret = 0
        evt_alpha = 0
        sigma = 0

    # 6. Scenario Analysis (Standardized)
    scenarios = [
        ("Market Crash (-5%)", {"Market": -0.05}),
        ("Tech Rally (+5%)", {"Semiconductors": 0.05, "Growth": 0.02}),
        ("Value Rotation (Val+3%, Gro-3%)", {"Value": 0.03, "Growth": -0.03})
    ]
    scenario_outputs = []
    for name, shocked_factors in scenarios:
        impact = 0
        for f, shock in shocked_factors.items():
            beta = factor_vals.get(f, 0)
            impact += beta * shock
        scenario_outputs.append(f"{name}: {impact:+.2%}")
    
    scenario_str = "\\n".join(scenario_outputs)

    # 7. Alerts
    alerts = []
    if pct_idio > 0.5: alerts.append(f"High Specific Risk ({pct_idio:.0%})")
    if sigma > 2.0: alerts.append(f"Unexpected Earnings Move ({sigma:.1f} Sigma)")
    if 'Rolling_Beta' in rolling_stats.columns:
        curr_b = rolling_stats['Rolling_Beta'].iloc[-1]
        mean_b = rolling_stats['Rolling_Beta'].rolling(60).mean().iloc[-1]
        if abs(curr_b - mean_b) > 0.5: alerts.append(f"Major Beta Drift ({curr_b:.2f} vs Avg {mean_b:.2f})")

    alerts_str = "\\n".join(alerts) if alerts else "None"

    # --- PROMPT CONSTRUCTION ---
    prompt = f"""
    You are a Risk Office reporting assistant. Your task is to generate a comprehensive, professional single stock risk report using the data provided from the dashboard.

    Follow these rules:
    Be concise but detailed.
    Highlight key risks, changes, and insights.
    Tell a story in the text, not just in the graphs.
    Always interpret the numbers, not just report them.
    Tone is professional and objective. Audience is Portfolio Managers and Risk Officers.
    Surface new insights: drift, regime change, anomalies, pre-earnings build-up, scenario asymmetry.
    Suggest actionable next steps/hedges.

    Here is the Data for {stock_ticker} (As of {latest_date.strftime('%Y-%m-%d')}):

    1. Stock Overview
    - Stock: {stock_ticker}
    - 60-Day Return: {period_ret:+.1%} (Approx).
    - Data Vintage: {latest_date.strftime('%Y-%m-%d')}

    2. Factor Exposure Summary
    - Top Positive Loadings: {', '.join(top_3_pos)}
    - Top Negative Loadings: {', '.join(top_3_neg)}
    - Beta Trends: {', '.join(beta_trends)}

    3. Risk Decomposition
    - Annualized Volatility: {total_vol:.1%}
    - Idiosyncratic Risk Ratio (Specific Risk): {pct_idio:.0%}
    - Factor Risk Ratio: {1-pct_idio:.0%}

    4. Performance Attribution (Latest Earnings Event)
    - Total Move: {evt_ret:+.1%}
    - Alpha Component: {evt_alpha:+.1%} (Remainder explained by factors)
    
    5. Event Risk & Earnings Diagnostics
    - Implied Move vs Realized: Realized move was {sigma:.1f} Standard Deviations (Sigma).
    
    6. Scenario Analysis (Hypothetical P&L Impact)
    {scenario_str}

    7. Alerts & Risk Flags Triggered
    {alerts_str}

    ---
    
    Generate the report following this structure EXACTLY:

    1. Stock Overview
    (Name, return, context)

    2. Factor Exposure Summary
    (Top loadings, trends, drift notes)

    3. Risk Decomposition
    (Vol breakdown, concentration check)

    4. Performance Attribution
    (Latest event drivers, alpha vs beta)

    5. Event Risk & Earnings Diagnostics
    (Sigma moves, surprise magnitude)

    6. Scenario Analysis
    (Discuss the specific scenario impacts provided above. Highlight vulnerabilities like "Vulnerable to Value Rotation")

    7. Alerts & Risk Flags
    (List the triggered alerts and explain implications)

    8. Actionable Insights Section
    (Provide 3-5 key takeaways: Beta drift prediction, style transition, hedging ideas)

    9. Summary Paragraph
    (4-5 lines summarizing the main risk narrative. Professional tone.)

    Use proper Markdown formatting with headers.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Generation Error: {e}"
