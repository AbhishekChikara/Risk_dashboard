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

# --- AGENT PERSONAS ---
AGENT_PROMPTS = {
    "Risk Manager 🛡️": """
    You are a conservative Chief Risk Officer (CRO). Your job is to protect capital.
    
    PRIMARY OBJECTIVE: Analyze the stock's behavior around Earnings Announcements.
    Investigate:
    1. Returns: Does it typically sell off or rally? Is the move asymmetric?
    2. Volatility: Analyze the "Fear Cycle" (Ramp & Crush). Does risk entitle (rise) leading into the event?
    
    Focus on: Downside tails, historic drawdown patterns around earnings, stress test failures, and "what can go wrong".
    Tone: Critical, defensive, cautious.
    
    Structure:
    1. Executive Risk Summary (The "Kill" Criteria)
    2. Earnings Behavior Analysis (Deep Dive)
        - Intraday Reaction: Win Rate & Asymmetry (Upside vs Downside Skew)
        - Volatility Dynamics: The "Fear Cycle" (Ramp up vs Crush down)
        - Drift Profile: Do moves sustain or reverse in days T+2 to T+5?
    3. Factor Exposure Risks (Concentration, crowding)
    4. Hedging Recommendations (Specific puts/collars based on the vol profile)
    5. Final Verdict: APPROVE / REJECT / HEDGE REQUIRED
    """,
    
    "Portfolio Manager 💼": """
    You are an aggressive Portfolio Manager (PM). Your job is to generate Alpha.
    Focus on: Upside potential, timing, sizing, and "is this a trade?".
    Tone: Decisive, opportunity-seeking, concise.
    Structure:
    1. Investment Thesis (The "Edge")
    2. Timing & Catalysts (Why now?)
    3. Sizing Recommendation (Max, Standard, or Trim)
    4. Upside/Downside Ratio
    5. Final Verdict: BUY / SELL / HOLD
    """,
    
    "Quantitative Analyst 🔢": """
    You are a PhD Quantitative Researcher. Your job is statistical validation.
    Focus on: Significance, robust anomalies, regimes, and "is it signal or noise?".
    Tone: Objective, dry, data-driven, academic.
    Structure:
    1. Statistical Significance of Move (Sigma checks)
    2. Regime Classification (Low vs High Vol)
    3. Anomaly Detection (Drift, Skew)
    4. Model Confidence Score (0-100%)
    5. Final Verdict: STATISTICALLY SIGNIFICANT / NOISE
    """
}

def generate_ai_summary(api_key, stock_ticker, full_df, reaction_days, latest_reaction, second_latest_reaction, factor_cols, combined_events, rolling_stats, loadings_df, vol_profile=None, model_name='models/gemini-1.5-flash', agent_persona="Risk Manager 🛡️"):
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
    
    # 4. Volatility Ramp (Fear Cycle) - Derived from output data
    vol_ramp_str = "No Volatility Profile Data Available"
    if vol_profile is not None:
        # Expecting cols: Rel_Day, Total_Vol, Idio_Vol, Factor_Vol
        # T-30, T=0, T+30
        try:
            t_minus_30 = vol_profile[vol_profile['Rel_Day'] == -30]['Total_Vol'].iloc[0]
            t_0 = vol_profile[vol_profile['Rel_Day'] == 0]['Total_Vol'].iloc[0]
            t_plus_10 = vol_profile[vol_profile['Rel_Day'] == 10]['Total_Vol'].iloc[0] if 10 in vol_profile['Rel_Day'].values else t_0
            
            t_0_idio_ratio = vol_profile[vol_profile['Rel_Day'] == 0]['Idio_Vol'].iloc[0] / t_0 if t_0 > 0 else 0
            
            ramp_pct = (t_0 - t_minus_30) / t_minus_30
            crush_pct = (t_plus_10 - t_0) / t_0
            
            vol_ramp_str = f"""
            - T-30 to T=0 Ramp: {ramp_pct:+.1%} (Rise in fear)
            - T=0 to T+10 Crush: {crush_pct:+.1%} (Resolution)
            - Risk Composition at Event: {t_0_idio_ratio:.0%} Idiosyncratic / {1-t_0_idio_ratio:.0%} Systematic
            """
        except:
            vol_ramp_str = "Error parsing Volatility Profile"
            
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

    # --- 8. ADVANCED CONTEXT (New) ---
    # A. Historical Win Rate
    win_rate_str = "N/A"
    if not reaction_days.empty:
        wins = (reaction_days[stock_ticker] > 0).sum()
        total_evts = len(reaction_days)
        win_rate = wins / total_evts
        win_rate_str = f"{win_rate:.0%} ({wins}/{total_evts} Positive Reactions)"

    # B. Pre-Earnings Setup (Current 30d Trend)
    # Using last 30 days of full_df to see entering momentum
    setup_30d = full_df.iloc[-30:][stock_ticker].sum()
    setup_str = f"{setup_30d:+.1%} (30-Day Run-up)"
    setup_desc = "Overbought" if setup_30d > 0.10 else ("Oversold" if setup_30d < -0.10 else "Neutral")

    # C. Market Regime
    regime_str = "Unknown"
    if 'Correlation_Regime' in rolling_stats.columns:
        curr_corr = rolling_stats['Correlation_Regime'].iloc[-1]
        regime_desc = "Crisis/High Correlation" if curr_corr > 0.5 else "Stock-Picker/Low Correlation"
        regime_str = f"{curr_corr:.2f} ({regime_desc})"

    # D. Seasonality
    season_str = "N/A"
    if not reaction_days.empty:
        # Check if 'Quarter' exists, if not derive it
        if 'Quarter' not in reaction_days.columns:
             reaction_days = reaction_days.copy()
             reaction_days['Quarter'] = reaction_days['Date'].dt.quarter
        
        q_stats = reaction_days.groupby('Quarter')[stock_ticker].mean()
        best_q = q_stats.idxmax()
        worst_q = q_stats.idxmin()
        season_str = f"Best: Q{best_q} ({q_stats[best_q]:+.1%}), Worst: Q{worst_q} ({q_stats[worst_q]:+.1%})"

    # E. Asymmetry (Risk/Reward Skew)
    asymmetry_str = "N/A"
    if not reaction_days.empty:
        avg_up = reaction_days[reaction_days[stock_ticker] > 0][stock_ticker].mean()
        avg_down = reaction_days[reaction_days[stock_ticker] < 0][stock_ticker].mean()
        # Handle nan if no up or no down moves
        avg_up = avg_up if not np.isnan(avg_up) else 0.0
        avg_down = avg_down if not np.isnan(avg_down) else 0.0
        
        asymmetry_str = f"Avg Upside: {avg_up:+.1%}, Avg Downside: {avg_down:+.1%}"

    # F. Post-Earnings Drift (T+2 to T+10)
    drift_str = "N/A"
    if not combined_events.empty:
        # Filter for T+10 cumulative - T+1 cumulative
        # Simplified: just average T+2 to T+5 return
        # We need to act on 'Event_ID' group
        drifts = []
        for eid, grp in combined_events.groupby('Event_ID'):
            # Check if we have day 1 and day 5 or similar
            try:
                p1 = grp[grp['Rel_Day'] == 1][stock_ticker].values[0]
                p_end = grp[grp['Rel_Day'] == 5][stock_ticker].values[0]
                drifts.append(p_end - p1)
            except:
                continue
        
        if drifts:
            avg_drift = np.mean(drifts)
            drift_desc = "Continuation" if avg_drift * (1 if wins/total_evts > 0.5 else -1) > 0 else "Reversal"
            drift_str = f"{avg_drift:+.1%} (Avg T+1 to T+5 move). Tendency: {drift_desc}"

    # --- PROMPT CONSTRUCTION ---
    
    # 1. Get Agent Instructions
    agent_instructions = AGENT_PROMPTS.get(agent_persona, AGENT_PROMPTS["Risk Manager 🛡️"])

    # 2. Build Full Prompt
    prompt = f"""
    {agent_instructions}

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
    {vol_ramp_str}

    4. Performance Attribution (Latest Earnings Event)
    - Total Move: {evt_ret:+.1%}
    - Alpha Component: {evt_alpha:+.1%} (Remainder explained by factors)
    
    5. Event Risk & Earnings Diagnostics
    - Implied Move vs Realized: Realized move was {sigma:.1f} Standard Deviations (Sigma).
    
    6. Scenario Analysis (Hypothetical P&L Impact)
    {scenario_str}

    7. Alerts & Risk Flags Triggered
    {alerts_str}
    
    8. ADVANCED CONTEXT & PROBABILITIES
    - Historical Win Rate (T+1): {win_rate_str}
    - Asymmetry Profile: {asymmetry_str} (Skew check).
    - Current Setup (30d Pre-Event): {setup_str} -> Market is potentially {setup_desc}.
    - Market Regime: {regime_str}
    - Seasonality Patterns: {season_str}
    - Post-Earnings Drift: {drift_str}
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Generation Error: {e}"
