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
    You are a conservative Chief Risk QUANT (CRO). Your job is to provide insights on how a stock performs and behaves around earnings announcements, both in terms of returns and volatility.
    
    PRIMARY OBJECTIVE: Utilize Quarterly Earnings Dates and Equity Factor Returns to explore how the stock performs and behaves around earnings announcements, both in terms of Returns and Volatility.
    
    CRITICAL INTERPRETATION GUIDELINES:
    1. Win Rate vs Magnitude: A ~60% win rate is a coin flip. Focus on MAGNITUDE risk. If the stock drops 8-10%, that wipes out weeks of alpha. That is the risk.
    2. Drift Profile: If drift is "Momentum/Amplification", it means a bad earnings reaction gets worse over the week.
    3. Idiosyncratic Risk: If the moves are >60% Idiosyncratic, BETA HEDGING (shorting sector) WILL FAIL.
    
    Investigate:
    1. Returns: Does it typically sell off or rally? Is the move asymmetric?
    2. Volatility: Analyze the "Fear Cycle" (Ramp & Crush). Does risk entitle (rise) leading into the event?
    3. Regime Sensitivity: Does the stock behavior change when the broader market is in "Crisis Mode" (High Correlation)?
    
    Tone: Critical, defensive, cautious.
    
    Structure:
    1. Executive Risk Summary (Focus on the primary objective paraphrase your findings)
    2. Earnings Behavior Analysis (Deep Dive)
        - Intraday Reaction: Win Rate & Asymmetry (Upside vs Downside Skew)
        - Volatility Dynamics: The "Fear Cycle" (Ramp up vs Crush down)
        - Drift Profile: Do moves sustain or reverse in days T+2 to T+5?
    3. Factor Ecosystem Deep Dive (Exposure & Crowding)
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
        if not vol_profile.empty:
            try:
                # Robust lookup: Find available days
                days = vol_profile['Rel_Day'].values
                
                # T-Start (Closest to -30, or min)
                start_day = -30
                if -30 not in days:
                    # Find closest day <= 0
                    neg_days = days[days < 0]
                    start_day = neg_days.min() if len(neg_days) > 0 else days.min()
                
                # T-End (Closest to +10, or max)
                end_day = 10
                if 10 not in days:
                    pos_days = days[days > 0]
                    end_day = pos_days.max() if len(pos_days) > 0 else days.max()
                
                # Get Values
                t_start_val = vol_profile[vol_profile['Rel_Day'] == start_day]['Total_Vol'].iloc[0]
                
                # Get T=0 (or closest to 0 if exact 0 missing, though unlikely for event study)
                t_0_val = 0
                if 0 in days:
                     t_0_val = vol_profile[vol_profile['Rel_Day'] == 0]['Total_Vol'].iloc[0]
                     t_0_idio = vol_profile[vol_profile['Rel_Day'] == 0]['Idio_Vol'].iloc[0]
                else:
                    # Fallback to day closest to 0
                    idx_0 = np.abs(days).argmin()
                    day_0 = days[idx_0]
                    t_0_val = vol_profile[vol_profile['Rel_Day'] == day_0]['Total_Vol'].iloc[0]
                    t_0_idio = vol_profile[vol_profile['Rel_Day'] == day_0]['Idio_Vol'].iloc[0]

                t_end_val = vol_profile[vol_profile['Rel_Day'] == end_day]['Total_Vol'].iloc[0]
                
                t_0_idio_ratio = t_0_idio / t_0_val if t_0_val > 0 else 0
                
                # Calc Ramps using actual days found
                ramp_pct = (t_0_val - t_start_val) / t_start_val if t_start_val > 0 else 0
                crush_pct = (t_end_val - t_0_val) / t_0_val if t_0_val > 0 else 0
                
                vol_ramp_str = f"""
                - Pre-Event Ramp (T{start_day} to T0): {ramp_pct:+.1%}
                - Post-Event Crush (T0 to T+{end_day}): {crush_pct:+.1%}
                - Risk Composition at Event: {t_0_idio_ratio:.0%} Idiosyncratic / {1-t_0_idio_ratio:.0%} Systematic
                """
            except Exception as e:
                vol_ramp_str = f"Error interpreting Volatility Profile: {e}"
        else:
             vol_ramp_str = "Volatility Profile Data is Empty"
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
    drift_str = "N/A"
    if not combined_events.empty:
        drifts = []
        signs_match = []
        for eid, grp in combined_events.groupby('Event_ID'):
            try:
                p1 = grp[grp['Rel_Day'] == 1][stock_ticker].values[0]
                p_end = grp[grp['Rel_Day'] == 5][stock_ticker].values[0] # Using T+5 for weekly drift
                drift_val = p_end - p1
                drifts.append(drift_val)
                # Check if drift amplifies the move
                if (p1 > 0 and drift_val > 0) or (p1 < 0 and drift_val < 0):
                    signs_match.append(1) # Amplification
                else:
                    signs_match.append(0) # Reversal/Damping
            except:
                continue
        
        if drifts:
            avg_drift = np.mean(drifts)
            match_pct = np.mean(signs_match)
            # Logic: If >50% of time drift matches move, it's Momentum/Amplification
            tendency = "MOMENTUM / AMPLIFICATION" if match_pct > 0.5 else "MEAN REVERSION / DAMPING"
            drift_str = f"{avg_drift:+.1%} (Avg T+1 to T+5). Tendency: {tendency} ({match_pct:.0%} of time trajectory continues)."

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
        # curr_loadings is already defined above
        sig_factors = curr_loadings[curr_loadings.abs() > 0.5].index.tolist()
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

    # 4. Kill Criteria Switch
    # Flag if Idio risk > 60% AND Avg Move > 6%
    avg_abs_move_val = reaction_days[stock_ticker].abs().mean() if not reaction_days.empty else 0
    kill_switch = "TRUE" if (idio_ratio_val > 60 and avg_abs_move_val > 0.06) else "FALSE"

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
    - Idiosyncratic Risk Ratio (Specific Risk): {pct_idio:.0%} (General 60d)
    - Event-Specific Idiosyncratic Ratio: {event_idio_ratio_str}
    - Factor Risk Ratio: {1-pct_idio:.0%}
    {vol_ramp_str}

    **3. RISK & BEHAVIORAL PROFILE (DEEP DIVE):**
    - **Idiosyncratic Risk Ratio:** {idio_ratio_val:.1f}% (If >60%, Beta Hedging is INEFFECTIVE).
    - **Drift Profile:** {drift_corr:.2f} -> {drift_risk}
    - **Seasonality Context:** {seasonality_context}
    - **Factor Environment:** {factor_context}
    
    **4. KILL CRITERIA STATUS:**
    - **Critical Risk Flag:** {kill_switch}

    4. Performance Attribution (Latest Earnings Event)
    - Total Move: {evt_ret:+.1%}
    - Alpha Component: {evt_alpha:+.1%} (Remainder explained by factors)
    
    5. Event Risk & Earnings Diagnostics
    - Implied Move vs Realized: Realized move was {sigma:.1f} Standard Deviations (Sigma).
    
    6. Scenario Analysis (Hypothetical P&L Impact)
    {scenario_str}

    7. Alerts & Risk Flags Triggered
    {alerts_str}
    
    8. Historical Context (The Ledger)
    Last 5 Earnings Events:
    {history_str}
    
    Records (All-Time):
    {records_str}
    
    9. ADVANCED CONTEXT & PROBABILITIES
    - Historical Win Rate (T+1): {win_rate_str}
    - Tail Risk Profile: {tail_risk_str}
    - Asymmetry Profile: {asymmetry_str} (Skew check).
    - Current Setup (30d Pre-Event): {setup_str} -> Market is potentially {setup_desc}.
    - Run-up Reliability: {runup_corr_str} (Correlation of Run-up to Event Return).
    - Market Regime: {regime_str}
    - Seasonality Patterns: {season_str}
    - Post-Earnings Drift: {drift_str}
    
    *** DATA INTEGRITY PROTOCOL (STRICT) ***
    1. DO NOT use external knowledge, news, or general assumptions about {stock_ticker}.
    2. ONLY use the specific metrics provided above (Win Rate, Drift, Volatility, etc.).
    3. If a metric is "N/A" or missing, state "Data Not Available" - do NOT guess.
    4. Your analysis must be purely derivative of the provided numbers.
    5. FORMATTING RULE: Produce a FORMAL RESEARCH REPORT. Do NOT write an email. Do NOT use salutations like "Hi Team" or "Dear Risk Manager". Start directly with the Executive Summary.
    *****************************************
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Generation Error: {e}"
