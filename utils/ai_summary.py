import google.generativeai as genai
import streamlit as st
import pandas as pd
import numpy as np
from utils.calculations import calculate_beta_trends, interpret_volatility_ramp, analyze_post_event_drift, calculate_fade_strategy, calculate_factor_contribution, calculate_factor_crowding

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
    You are a conservative Chief Risk QUANT (CRO). Your job is to provide insights on how a stock performs and behaves around earnings announcements, both in terms of returns and volatility.
    
    **CRITICAL INSTRUCTION FOR VISUALS:**
    You must include specific PLACEHOLDER TAGS (e.g., [[IMAGE:vol_ramp]]) in your output. These tags are used by the reporting engine to insert charts. 
    - Do NOT describe the chart vaguely ("As seen in the chart...").
    - DO insert the exact tag [[IMAGE:key]] on its own line where the visual proof should go.
    - Follow the structure below EXACTLY.

    **OBJECTIVE:**
    Analyze the provided data (Earnings Returns, Factor Loadings, Volatility). Identify systematic vs. idiosyncratic risks.
    stock performs and behaves around earnings announcements, both in terms of Returns and Volatility.
    
    CRITICAL INTERPRETATION GUIDELINES:
    1. Win Rate vs Magnitude: A ~60% win rate is a coin flip. Focus on MAGNITUDE risk. If the stock drops 8-10%, that wipes out weeks of alpha. That is the risk.
    2. Drift Profile: If drift is "Momentum/Amplification", it means a bad earnings reaction gets worse over the week.
    3. Idiosyncratic Risk: If the moves are >60% Idiosyncratic, BETA HEDGING (shorting sector) WILL FAIL.
    
    Investigate:
    1. Returns: Does it typically sell off or rally? Is the move asymmetric?
    2. Volatility: Analyze the "Fear Cycle" (Ramp & Crush). Does risk entitle (rise) leading into the event?
    3. Regime Sensitivity: Does the stock behavior change when the broader market is in "Crisis Mode" (High Correlation)?
    
    STRATEGIC CONSISTENCY RULE:
    - IF the "Tactical Trading Implications" section says "**MOMENTUM EXTENSION**" (Failed Mean Reversion), you MUST characterize the stock's profile as MOMENTUM/TRENDING.
    - Do NOT call it "Mean Reversion" in your summary if the strategy fail rate confirms momentum.
    - Instead, frame it as a "Momentum Trap": Fading fails because the trend extends.
    
    Tone: Critical, defensive, cautious.
    
    Structure:
    1. Executive Risk Summary (Synthesize Earnings Analysis based on Return and Volatility)
    2. Earnings Behavior Analysis (Deep Dive)
        - Intraday Reaction: Win Rate & Asymmetry (Upside vs Downside Skew)
        - Volatility Dynamics: The "Fear Cycle" (Ramp up vs Crush down)
        [[IMAGE:vol_ramp]]
        - Drift Profile: Do moves sustain or reverse in days T+2 to T+5?

    2.1 Factor Drivers & Crowding Risks (CRITICAL NEW SECTION)
        - Dominant Factor Drivers: Use "Systematic Risk Breakdown" to explain the "Why". (e.g. "Semiconductors explain 70% of variance").
        - Factor Crowding Risk: Use "Factor Crowding / Z-Scores". Warn if > 2.0 Sigma. (e.g. "Growth loading is 2.5 sigma above average").
        - Hedging Dislocation ("Factor Betrayal"): EXPLICITLY reference the "Factor Betrayal Table". Compare Normal vs Earnings Correlation. (e.g. "Correlation drops from 0.85 to 0.12, rendering hedges ineffective").

    3. Factor Ecosystem Deep Dive (Legacy/Broader context)
        - Current Exposures: Top drivers (Positive/Negative).
        - Factor Rotation: Are we fighting macro headwinds? (Factor Drag).
        - Beta Trends: Is the stock becoming "More Systematic" or "More Idiosyncratic"?
    4. Regime & Stress Test Analysis (Vulnerability to market-wide shocks)
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

def generate_ai_summary(api_key, stock_ticker, full_df, reaction_days, latest_reaction, second_latest_reaction, factor_cols, combined_events, rolling_stats, loadings_df, vol_profile=None, model_name='models/gemini-1.5-flash', agent_persona="Risk Manager 🛡️", images=None, backtest_hold_days=5):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        return f"Error configuring AI model: {e}"

    # ... (Rest of data prep code stays same) ...
    # [Lines 86-474 are unchanged data prep]
    # ... (Prompt Construction code stays same) ...
    # [Lines 513-549 are prompt construction]

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
    # Beta Trends (REFACTORED)
    beta_trends = calculate_beta_trends(rolling_stats)

    # --- NEW: Deep Factor Analysis ---
    # 1. Factor Contribution (Variance Breakdown)
    contrib_dict = calculate_factor_contribution(full_df, loadings_df, factor_cols)
    if contrib_dict:
        sorted_contrib = sorted(contrib_dict.items(), key=lambda x: x[1], reverse=True)
        factor_contrib_str = ", ".join([f"{k} {v:.0%}" for k, v in sorted_contrib])
    else:
        factor_contrib_str = "Data Not Available"

    # 2. Factor Crowding (Z-Scores)
    crowding_dict = calculate_factor_crowding(loadings_df, factor_cols)
    crowding_alerts = []
    if crowding_dict:
        for f, z in crowding_dict.items():
            if abs(z) > 1.5:
                status = "CROWDED/STRETCHED" if z > 0 else "OVERSOLD/UNLOVED"
                crowding_alerts.append(f"{f}: {z:.2f} Sigma ({status})")
    
    crowding_str = "; ".join(crowding_alerts) if crowding_alerts else "No extreme crowding detected (All factors within 1.5 sigma)."
    
    # 3. Risk Decomposition
    window_60 = full_df.iloc[-60:].copy()
    var_factor = window_60['Factor_Return'].var() * 252
    var_idio = window_60['Idiosyncratic_Return'].var() * 252
    total_vol = np.sqrt(var_factor + var_idio)
    pct_idio = var_idio / (var_factor + var_idio) if total_vol > 0 else 0
    
    # 4. Volatility Ramp (Fear Cycle) (REFACTORED)
    vol_ramp_str = interpret_volatility_ramp(vol_profile)
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
        
    # A2. Tail Risk Analysis (Magnitude Probabilities)
    prob_5pct = "N/A"
    prob_10pct = "N/A"
    tail_risk_str = "N/A"
    if not reaction_days.empty:
        total = len(reaction_days)
        moves = reaction_days[stock_ticker].abs()
        n_gt_5 = (moves > 0.05).sum()
        n_gt_10 = (moves > 0.10).sum()
        prob_5pct = n_gt_5 / total
        prob_10pct = n_gt_10 / total
        tail_risk_str = f"Prob >5% Move: {prob_5pct:.0%}, Prob >10% Move: {prob_10pct:.0%}"

    # B. Pre-Earnings Setup & Correlation
    # Using last 30 days of full_df to see entering momentum
    setup_30d = full_df.iloc[-30:][stock_ticker].sum()
    setup_str = f"{setup_30d:+.1%} (30-Day Run-up)"
    setup_desc = "Overbought" if setup_30d > 0.10 else ("Oversold" if setup_30d < -0.10 else "Neutral")
    
    # Calc Correlation: Pre-Event Run-up (30d) vs Event Reaction
    runup_corr_str = "N/A"
    if not reaction_days.empty and len(reaction_days) > 4:
        runups = []
        reactions = []
        # Need to be careful with indexing. For loop is safest.
        sorted_evts = reaction_days.sort_values('Date')
        for idx, row in sorted_evts.iterrows():
            evt_date = row['Date']
            # Find 30d window before this date
            mask = (full_df['Date'] < evt_date) & (full_df['Date'] >= evt_date - pd.Timedelta(days=45)) 
            # 45d buffer to capture ~30 trading days
            window = full_df[mask].tail(20) # Approx 20 trading days = 1 month
            if not window.empty:
                r_up = window[stock_ticker].sum()
                runups.append(r_up)
                reactions.append(row[stock_ticker])
        
        if len(runups) > 4:
            corr = np.corrcoef(runups, reactions)[0,1]
            if np.isnan(corr): corr = 0
            sig = "Reversal" if corr < -0.2 else ("Continuation" if corr > 0.2 else "No Signal")
            runup_corr_str = f"{corr:.2f} ({sig} Tendency)"

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

    # G. Event-Specific Idiosyncratic Ratio (New User Req)
    # The 60-day window understates event risk. We need Avg(|Idio|) / Avg(|Total|) on T+1
    event_idio_ratio_str = "N/A"
    if not reaction_days.empty:
        avg_abs_idio = reaction_days['Idiosyncratic_Return'].abs().mean()
        avg_abs_total = reaction_days[stock_ticker].abs().mean()
        if avg_abs_total > 0:
            e_ratio = avg_abs_idio / avg_abs_total
            event_idio_ratio_str = f"{e_ratio:.0%} (Event-Specific)"

    # F. Post-Earnings Drift (T+2 to T+10)
    # F. Post-Earnings Drift (T+2 to T+10) (REFACTORED)
    drift_str, drift_corr = analyze_post_event_drift(combined_events, stock_ticker)

    # H. Historical Earnings Ledger (Contextual Memory)
    history_str = "N/A"
    records_str = "N/A"
    if not reaction_days.empty:
        # 1. Ledger of Last 5 Events
        last_5 = reaction_days.sort_values('Date', ascending=False).head(5)
        ledger_lines = []
        for idx, row in last_5.iterrows():
             d_str = row['Date'].strftime('%Y-%m-%d')
             move = row[stock_ticker]
             alpha = row['Idiosyncratic_Return']
             fac_ret = row['Factor_Return']
             
             # Find Top Factor Driver
             driver = "N/A"
             try:
                 # Check contrib cols based on factor_cols arg
                 contribs = {f: row.get(f"Contrib_{f}", 0) for f in factor_cols}
                 if contribs:
                     # Get max key by absolute value
                     top_f = max(contribs, key=lambda k: abs(contribs[k]))
                     driver = f"{top_f} {contribs[top_f]:+.1%}"
             except:
                 driver = "Unknown"

             ledger_lines.append(f"- {d_str}: Total {move:+.1%}, Alpha {alpha:+.1%}, Factor {fac_ret:+.1%} (Driver: {driver})")
        
        history_str = "\\n".join(ledger_lines)
        
        # 2. Records
        best_day = reaction_days.loc[reaction_days[stock_ticker].idxmax()]
        worst_day = reaction_days.loc[reaction_days[stock_ticker].idxmin()]
        records_str = f"""
        - Best Event: {best_day['Date'].strftime('%Y-%m-%d')} ({best_day[stock_ticker]:+.1%})
        - Worst Event: {worst_day['Date'].strftime('%Y-%m-%d')} ({worst_day[stock_ticker]:+.1%})
        """

    # I. Factor Drag Analysis (Deep Dive)
    factor_context = "N/A"
    try:
        # 1. Identify significant exposures (>0.5)
        # curr_loadings is already defined above, but might contain 'Date' which fails abs()
        # Filter for numeric only
        numeric_loadings = curr_loadings.drop(['Date'], errors='ignore')
        if not numeric_loadings.empty:
             sig_factors = numeric_loadings[pd.to_numeric(numeric_loadings, errors='coerce').abs() > 0.5].index.tolist()
        else:
             sig_factors = []
        # Filter strictly for factors present in factor_cols (to avoid '_Load' suffixes messing up lookup)
        sig_factors = [f for f in sig_factors if f in factor_cols or f.replace('_Load', '') in factor_cols]
        
        # 2. Check recent performance (Last 5 days) of these factors
        factor_headwinds = []
        factor_tailwinds = []
        
        # If no significant factors found, fallback to all factors to get *some* context
        factors_to_check = sig_factors if sig_factors else factor_cols
        
        for f in factors_to_check:
            clean_f = f.replace('_Load', '')
            # Try multiple column name variations
            possible_cols = [f"{clean_f}_Ret", clean_f, f"{clean_f} Return"]
            ret_col = next((c for c in possible_cols if c in full_df.columns), None)
            
            if ret_col:
                recent_perf = full_df[ret_col].tail(5).sum()
                exposure = curr_loadings.get(f, 0)
                if exposure == 0 and f in curr_loadings.index: exposure = curr_loadings[f]
                
                # Logic: Positive Exposure + Negative Perf = Headwind
                if exposure > 0 and recent_perf < 0:
                    factor_headwinds.append(f"{clean_f} (Exp: {exposure:.2f}, 5d Perf: {recent_perf:.1%})")
                elif exposure > 0 and recent_perf > 0:
                    factor_tailwinds.append(f"{clean_f} (Exp: {exposure:.2f}, 5d Perf: {recent_perf:.1%})")
                elif exposure < 0 and recent_perf > 0: # Short exp vs Rising factor = Headwind
                    factor_headwinds.append(f"{clean_f} (Short Exp: {exposure:.2f}, 5d Perf: {recent_perf:.1%})")
                    
        if factor_headwinds:
            factor_context = f"WARNING: Macro Headwinds. Stock fighting negative factor trends: {', '.join(factor_headwinds)}."
        elif factor_tailwinds:
            factor_context = f"SUPPORTIVE BACKDROP: Factor tailwinds present: {', '.join(factor_tailwinds)}."
        else:
            factor_context = "Neutral Factor Backdrop (No strong recent signals)."
            
    except Exception as e:
        factor_context = f"Factor Analysis Unavailable: {e}"

    # J. Advanced Risk Metrics for Prompt
    # 1. Idio Risk Ratio (Float)
    idio_ratio_val = 0.0
    if not reaction_days.empty and reaction_days[stock_ticker].abs().mean() > 0:
        idio_ratio_val = (reaction_days['Idiosyncratic_Return'].abs().mean() / reaction_days[stock_ticker].abs().mean()) * 100
    
    # 2. Drift Risk Rule
    drift_risk = "N/A"
    drift_corr = 0.0
    if not combined_events.empty:
        # Build paired arrays for correlation: T+1 Move vs Drift (T+5 - T+1)
        # reuse drifts logic or re-calc
        t1_moves = []
        drift_moves = []
        for eid, grp in combined_events.groupby('Event_ID'):
            try:
                p1 = grp.loc[grp['Rel_Day'] == 1, stock_ticker].values[0]
                p5 = grp.loc[grp['Rel_Day'] == 5, stock_ticker].values[0]
                t1_moves.append(p1)
                drift_moves.append(p5 - p1)
            except:
                continue
        
        if len(t1_moves) > 2:
            drift_corr = np.corrcoef(t1_moves, drift_moves)[0, 1]
            if np.isnan(drift_corr): drift_corr = 0.0
            
            if drift_corr > 0.15:
                drift_risk = "AMPLIFICATION RISK (Positive Correlation). If reaction is negative, losses historically widen."
            elif drift_corr < -0.15:
                drift_risk = "MEAN REVERSION (Negative Correlation). Initial moves tend to fade/reverse."
            else:
                drift_risk = "RANDOM WALK. No predictive signal."

    # 3. Seasonality Context (Current Quarter)
    current_q = (pd.Timestamp.now().month - 1) // 3 + 1
    seasonality_context = "N/A"
    if not reaction_days.empty:
        if 'Quarter' not in reaction_days.columns:
             reaction_days['Quarter'] = reaction_days['Date'].dt.quarter
        q_avg = reaction_days[reaction_days['Quarter'] == current_q][stock_ticker].mean()
        # handle nan
        q_avg = q_avg if not np.isnan(q_avg) else 0.0
        seasonality_context = f"Current Quarter (Q{current_q}) typically sees moves of {q_avg:.1%}."

    # 4. Critical Risk Flag (Removed by User Request)
    # kill_switch logic removed.

    # --- K. Factor Betrayal (Correlation Breakdown) ---
    # Compare Factor Correlation on "Normal Days" vs "Earnings Days"
    betrayal_table_str = "Data Not Available"
    try:
        if not reaction_days.empty and not full_df.empty:
            # 1. Identify Earnings Days (T+1)
            # We use reaction_days for the "Earnings Day" set
            earnings_dates = reaction_days['Date']
            
            # 2. Identify Normal Days (All days NOT in earnings_dates)
            normal_df = full_df[~full_df['Date'].isin(earnings_dates)]
            
            # 3. Calculate Correlations
            # Normal Days
            if len(normal_df) > 10:
                norm_corr = normal_df[[stock_ticker, 'Factor_Return']].corr().iloc[0, 1]
            else:
                norm_corr = np.nan
                
            # Earnings Days
            if len(reaction_days) > 2: # Need at least a few points for correlation
                earn_corr = reaction_days[[stock_ticker, 'Factor_Return']].corr().iloc[0, 1]
            else:
                earn_corr = np.nan
            
            # 4. Format Table
            if not np.isnan(norm_corr) and not np.isnan(earn_corr):
                betrayal_table_str = f"""
                | Condition | Beta Correlation (Stock vs Factor Model) | Interpretation |
                | :--- | :--- | :--- |
                | **Normal Days** | **{norm_corr:.2f}** | Typical market behavior. |
                | **Earnings Days (T+1)** | **{earn_corr:.2f}** | {"⚠️ FACTOR BETRAYAL (Broken Link)" if earn_corr < 0.3 else "Linkage Intact"} |
                """
            else:
                 betrayal_table_str = "Insufficient data points for correlation calculation."
                 
            # 5. Calculate Earnings Volatility Multiplier (Event Vol / Normal Vol)
            normal_vol = normal_df[stock_ticker].std()
            event_vol = reaction_days[stock_ticker].std()
            earnings_vol_multiplier = (event_vol / normal_vol) if normal_vol > 0 else 1.0
            
    except Exception as e:
        betrayal_table_str = f"Error calculating Factor Betrayal: {e}"
        earnings_vol_multiplier = 1.0

    # --- L. Strategy Backtest (for AI Prompt) ---
    strategy_str = "Data Not Available"
    try:
        # Using imported function
        st_stats, _ = calculate_fade_strategy(full_df, reaction_days['Date'], stock_ticker, hold_days=backtest_hold_days)
        
        if st_stats:
            win_rate = st_stats['Win Rate']
            # Dynamic Interpretation
            if win_rate < 0.40:
                conclusion = "FAILED. The stock exhibits **MOMENTUM EXTENSION** (The move tends to continue, and fading it causes losses)."
            elif win_rate > 0.60:
                conclusion = "SUCCEEDED. The stock exhibits **MEAN REVERSION** (The move tends to reverse)."
            else:
                conclusion = "Inconclusive / Random Walk."

            strategy_str = f"""
            Backtesting a 'Fade the Move' (Mean Reversion) strategy yielded a {win_rate:.1%} Win Rate.
            Conclusion: The strategy {conclusion}
            - Average Trade Return: {st_stats['Avg Return']:.2%} (Holding {backtest_hold_days} Days).
            - Total Cumulative Return: {st_stats['Total Return']:.2%}.
            """
        else:
            strategy_str = "Insufficient data for Strategy Backtest."
    except Exception as e:
        strategy_str = f"Strategy Backtest Error: {e}"

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
    - **Systematic Risk Breakdown (Variance Contrib):** {factor_contrib_str}
    - **Factor Crowding / Z-Scores:** {crowding_str}

    3. Risk Decomposition
    - Annualized Volatility: {total_vol:.1%}
    - Idiosyncratic Risk Ratio (Specific Risk): {pct_idio:.0%} (General 60d)
    - Event-Specific Idiosyncratic Ratio: {event_idio_ratio_str}
    - **CASE STUDY METRICS**:
        - **Idiosyncratic Risk Pct (Event):** {idio_ratio_val:.1f}% (Matches User Case Study)
        - **Earnings Volatility Multiplier:** {earnings_vol_multiplier:.1f}x (Event Vol vs Normal Vol)
    - Factor Risk Ratio: {1-pct_idio:.0%}
    {vol_ramp_str}

    **Factor Betrayal Table (Correlation Breakdown):**
    [[IMAGE:vol_comp]]
    {betrayal_table_str}

    **3. RISK & BEHAVIORAL PROFILE (DEEP DIVE):**
    - **Idiosyncratic Risk Ratio:** {idio_ratio_val:.1f}% (If >60%, Beta Hedging is INEFFECTIVE).
    - **Drift Profile:** {drift_corr:.2f} -> {drift_risk}
    - **Seasonality Context:** {seasonality_context}
    - **Factor Environment:** {factor_context}
    
    **4. TACTICAL TRADING IMPLICATIONS:**
    [[IMAGE:equity]]
    {strategy_str}

    6. Performance Attribution (Latest Earnings Event)
    - Total Move: {evt_ret:+.1%}
    - Alpha Component: {evt_alpha:+.1%} (Remainder explained by factors)
    
    7. Event Risk & Earnings Diagnostics
    [[IMAGE:histogram]]
    - Implied Move vs Realized: Realized move was {sigma:.1f} Standard Deviations (Sigma).
    
    8. Scenario Analysis (Hypothetical P&L Impact)
    {scenario_str}

    9. Alerts & Risk Flags Triggered
    {alerts_str}
    
    10. Historical Context (The Ledger)
    Last 5 Earnings Events:
    {history_str}
    
    Records (All-Time):
    {records_str}
    
    11. ADVANCED CONTEXT & PROBABILITIES
    - Historical Win Rate (T+1): {win_rate_str}
    - Tail Risk Profile: {tail_risk_str}
    - Asymmetry Profile: {asymmetry_str} (Skew check).
    - Current Setup (30d Pre-Event): {setup_str} -> Market is potentially {setup_desc}.
    - Run-up Reliability: {runup_corr_str} (Correlation of Run-up to Event Return).
    - Market Regime: {regime_str}
    - Seasonality Patterns: {season_str}
    - Post-Earnings Drift: {drift_str}
    
    **VISUAL EVIDENCE CHECK (CRITICAL):**
    I have attached 1) **Equity Curve** and 2) **Earnings Return Histogram**. 
    - **Equity Curve**: Does it trend UP? (Confirmation of Mean Reversion).
    - **Histogram**: Do you see "Fat Tails"? (Bars at the far left/right edges that are taller than the white dashed Normal Curve).
    - **Remarks**: Validating these visuals must be part of your "Event Risk" or "Tactical Implications" analysis. Mention "Fat Tails" if seen.
    
    *** DATA INTEGRITY PROTOCOL (STRICT) ***
    1. DO NOT use external knowledge, news, or general assumptions about {stock_ticker}.
    2. ONLY use the specific metrics provided above (Win Rate, Drift, Volatility, etc.).
    3. If a metric is "N/A" or missing, state "Data Not Available" - do NOT guess.
    4. Your analysis must be purely derivative of the provided numbers.
    5. FORMATTING RULE: Produce a FORMAL RESEARCH REPORT. Do NOT write an email. Do NOT use salutations like "Hi Team" or "Dear Risk Manager". Start directly with the Executive Summary.
    *****************************************
    """
    
    try:
        content = [prompt]
        if images:
            content.extend(images)
            
        response = model.generate_content(content)
        
        # Extract Token Usage
        usage = {}
        if hasattr(response, 'usage_metadata'):
            usage = {
                'prompt_tokens': response.usage_metadata.prompt_token_count,
                'candidates_tokens': response.usage_metadata.candidates_token_count,
                'total_tokens': response.usage_metadata.total_token_count
            }
            
        return response.text, usage
    except Exception as e:
        return f"AI Generation Error: {e}", {}
