import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy.stats.mstats import winsorize

# --- Page Config ---
st.set_page_config(layout="wide", page_title="KR Capital: Earnings Dashboard")

from utils.data_loader import get_stock_list, load_data, process_data, get_event_window
from utils.calculations import get_garch_forecast, calculate_rolling_stats, calculate_volatility_profiles
from utils.visualizations import create_waterfall, plot_volatility_analysis
from utils.ai_summary import get_available_models, generate_ai_summary

# --- MAIN APP ---
def main():
    # --- Sidebar Controls ---
    st.sidebar.header("Stock Selection")
    stock_list = get_stock_list()
    
    if not stock_list:
        st.error("⚠️ No stock folders found in 'res/'. Please create a subdirectory for each stock (e.g., 'res/NVDA') containing the 3 required CSV files.")
        st.stop()
    
    selected_stock = st.sidebar.selectbox("Select Stock", stock_list)
    st.sidebar.markdown("---")

    st.title(f"{selected_stock} Earnings Analysis Dashboard 📈")
    st.markdown("### Quantitative Risk & Portfolio Construction Case Study")
    
    # Load and Process
    try:
        returns_df, loadings_df, earnings_df = load_data(selected_stock)
        full_df, factor_cols = process_data(returns_df, loadings_df, selected_stock)
    except FileNotFoundError as e:
        st.error(f"Critical Error: Missing file {e} in the '{selected_stock}' folder.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # --- Calculate Rolling Stats ---
    # We use the raw returns_df for this, but we need to know which columns are factors. 
    # factor_cols from process_data tells us the base names.
    # In returns_df, they should exist as is.
    rolling_stats = calculate_rolling_stats(returns_df, selected_stock, factor_cols, window=60)


    # --- DATA HEALTH CHECK ---
    with st.expander("Data Health Check"):
        col_h1, col_h2, col_h3 = st.columns(3)
        with col_h1:
            st.metric("Date Range", f"{full_df['Date'].min().strftime('%Y-%m-%d')} to {full_df['Date'].max().strftime('%Y-%m-%d')}")
        
        missing_vals = full_df.isnull().sum().sum()
        with col_h2:
            st.metric("Missing Values Found", f"{missing_vals}")
        
        duplicate_dates = full_df['Date'].duplicated().sum()
        with col_h3:
            st.metric("Duplicate Dates Found", f"{duplicate_dates}")

        if missing_vals > 0:
            st.warning("Missing values were found and have been forward-filled. This assumes that on a non-trading day, the last known value is carried forward.")
            # Simple forward-fill for all columns
            full_df = full_df.fillna(method='ffill').dropna() # drop any remaining NaNs at the start
        
        if duplicate_dates > 0:
            st.error("Duplicate dates were found and removed. Please check the source data integrity.")
            full_df = full_df.drop_duplicates(subset=['Date'], keep='first')

    st.sidebar.header("Configuration")
    window_size = st.sidebar.slider("Event Window (+/- Days)", 1, 10, 5)
    st.sidebar.markdown("---")
    st.sidebar.markdown("---")
    api_key = st.sidebar.text_input("Enter Google API Key (Optional)", type="password", help="Required for AI Summary tab.")
    
    selected_model = "models/gemini-1.5-flash" # Default
    if api_key:
        available_models = get_available_models(api_key)
        if available_models:
            # Try to find a good default
            default_ix = 0
            if 'models/gemini-1.5-flash' in available_models:
                default_ix = available_models.index('models/gemini-1.5-flash')
            elif 'models/gemini-pro' in available_models:
                default_ix = available_models.index('models/gemini-pro')
                
            selected_model = st.sidebar.selectbox("Select AI Model", available_models, index=default_ix)
        else:
            st.sidebar.warning("Could not fetch models. Check API Key.")
    
    st.sidebar.markdown("---")
    st.sidebar.header(f"{selected_stock} Event Deep Dive")
    
    # --- ADDED: Winsorization Control ---
    st.sidebar.markdown("---")
    st.sidebar.header("Advanced Settings")
    winsorize_level = st.sidebar.slider("Winsorize Returns Level (%)", 0, 5, 0, help="Set extreme outliers to a specified percentile. 0% means no winsorization.")
    if winsorize_level > 0:
        limit = winsorize_level / 100.0
        # Apply winsorization to all return columns in the dataframe
        return_cols = [col for col in returns_df.columns if col != 'Date']
        for col in return_cols:
            returns_df[col] = winsorize(returns_df[col], limits=[limit, limit])
        st.sidebar.info(f"Returns are being winsorized at the {winsorize_level}th percentile.")
    
    z_score_view = st.sidebar.checkbox("Enable Z-Score View in Factor Tab")


    # Select Event Logic
    default_index = len(earnings_df) - 1 if len(earnings_df) > 0 else 0
    selected_date = st.sidebar.selectbox("Select Earnings Date", earnings_df['Earnings Date'].dt.strftime('%Y-%m-%d'), index=default_index)
    selected_date_dt = pd.to_datetime(selected_date)
    
    # --- Main Page Layout (Tabs) ---
    tab_flow, tab1, tab2, tab3, tab4 = st.tabs(["📋 Workflow Monitoring", "📊 Aggregate Analysis", "⚖️ Event Comparison", "🔍 Event & Factor Deep Dive", "🤖 AI Summary"])

    # --- Tab 0: Workflow Monitoring (New Landing Page) ---
    with tab_flow:
        st.header("Risk Workflow Monitoring")
        st.markdown("Follow this 8-step process to assess earnings risk.")
        
        # --- Pre-calculate Globals for Workflow ---
        latest_event_date = earnings_df['Earnings Date'].max() if not earnings_df.empty else None
        
        # Beta Drift (Used in Step 7 Flags)
        current_beta = rolling_stats['Rolling_Beta'].iloc[-1] if 'Rolling_Beta' in rolling_stats.columns else np.nan
        avg_beta_60d = rolling_stats['Rolling_Beta'].rolling(60).mean().iloc[-1] if 'Rolling_Beta' in rolling_stats.columns else np.nan
        drift_val = current_beta - avg_beta_60d if not np.isnan(current_beta) and not np.isnan(avg_beta_60d) else 0.0
        
        # Volatility Forecast (Used in Step 7 Flags)
        latest_series = full_df[selected_stock]
        forecasted_vol, vol_msg = get_garch_forecast(latest_series)
        
        # --- Step 1: Data Alignment & Validation ---
        st.subheader("1. Data Alignment & Validation")
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            missing_vals = full_df.isnull().sum().sum()
            duplicate_dates = full_df['Date'].duplicated().sum()
            if missing_vals == 0 and duplicate_dates == 0:
                st.success(f"✅ Data Quality: READY ({len(full_df)} days loaded)")
            else:
                st.error(f"❌ Data Quality Issue: {missing_vals} missing, {duplicate_dates} duplicates")
        with col_q2:
            st.info(f"Analysis Range: {full_df['Date'].min().date()} to {full_df['Date'].max().date()}")
        
        st.markdown("---")

        # --- Step 2: Earnings Event Study (Monthly Horizon & Drift) ---
        st.subheader("2. Earnings Event Study (Monthly Horizon)")
        
        # A. Monthly Horizon Analysis
        # Defined approx windows: T-3m (-63d), T-2m (-42d), T-1m (-21d), T=0 (+21d post), T+1m (+42d), T+2m (+63d)
        # We need to aggregate across all events
        horizon_results = {
            "T-3 Months": [], "T-2 Months": [], "T-1 Month": [],
            "Earnings Month (T=0)": [], "T+1 Month": [], "T+2 Months": []
        }
        
        for date in earnings_df['Earnings Date']:
            # Find index in full_df
            idx_list = full_df.index[full_df['Date'] == date].tolist()
            if not idx_list: continue
            idx = idx_list[0]
            
            # Helper to get return in window relative to idx
            def get_ret(d_start, d_end):
                # Check bounds
                if idx + d_start < 0 or idx + d_end >= len(full_df): return np.nan
                # Sum log returns or simple returns? Simple returns sum approx.
                return full_df.iloc[idx + d_start : idx + d_end][selected_stock].sum()

            horizon_results["T-3 Months"].append(get_ret(-63, -42))
            horizon_results["T-2 Months"].append(get_ret(-42, -21))
            horizon_results["T-1 Month"].append(get_ret(-21, 0))
            horizon_results["Earnings Month (T=0)"].append(get_ret(0, 21))
            horizon_results["T+1 Month"].append(get_ret(21, 42))
            horizon_results["T+2 Months"].append(get_ret(42, 63))

        # Average and Format
        horizon_summary = []
        for period, rets in horizon_results.items():
            valid_rets = [r for r in rets if not np.isnan(r)]
            if valid_rets:
                avg_ret = np.mean(valid_rets)
                # Interpretation
                interp = "Neutral"
                if period == "T-1 Month" and avg_ret > 0.05: interp = "🔴 Pre-earnings Run-up"
                elif period == "T-1 Month" and avg_ret < -0.05: interp = "🟢 Oversold into Print"
                elif "T+" in period and avg_ret > 0.02: interp = "Momentum Drift"
                
                horizon_summary.append({"Period": period, "Avg Return": avg_ret, "Interpretation": interp})
        
        if horizon_summary:
            h_df = pd.DataFrame(horizon_summary)
            # Formatting for display
            h_df_disp = h_df.copy()
            h_df_disp['Avg Return'] = h_df_disp['Avg Return'].map('{:.2%}'.format)
            st.table(h_df_disp.set_index('Period'))
            
            # Critical Discoveries
            t_minus_1 = h_df.loc[h_df['Period'] == 'T-1 Month', 'Avg Return'].values[0] if 'T-1 Month' in h_df['Period'].values else 0
            if t_minus_1 > 0.10:
                st.error(f"🔴 CRITICAL: Pre-earnings drift is REAL. Returns 1 month before are extraordinarily high ({t_minus_1:.1%}).")

        # B. Factor Rotation Cycle (T-10 to T+10)
        st.caption("Factor Rotation Cycle (Beta Trajectory)")
        # We want to see how betas evolve around the event. 
        # Let's focus on the latest event for clarity, or an average of last 4. 
        # Using Latest Event for actionable "Now" analysis.
        if latest_event_date:
            # Define window: T-10 to T+10
            start_drift = latest_event_date - pd.Timedelta(days=15)
            end_drift = latest_event_date + pd.Timedelta(days=15)
            
            # Get loadings in this window
            # Use full_df because it has the _Load suffixes we need
            drift_window = full_df[(full_df['Date'] >= start_drift) & (full_df['Date'] <= end_drift)].copy()
            
            if not drift_window.empty:
                # Identify "Active" factors (those with highest max beta in window)
                max_betas = drift_window[ [f"{c}_Load" for c in factor_cols if f"{c}_Load" in drift_window.columns] ].max().sort_values(ascending=False)
                top_drift_factors = max_betas.head(5).index.tolist()
                
                # Check if we have data for T+10 (Post-Earnings)
                has_post = drift_window['Date'].max() > latest_event_date
                
                # Plot
                fig_cycle = px.line(drift_window, x='Date', y=top_drift_factors, 
                                    title=f"Factor Beta Cycle: {'Pre-Earnings' if not has_post else 'Pre & Post Earnings'}",
                                    labels={'value': 'Factor Beta', 'variable': 'Factor'})
                fig_cycle.add_vline(x=latest_event_date.timestamp() * 1000, line_dash="dash", line_color="white", annotation_text="Earnings")
                st.plotly_chart(fig_cycle, width="stretch")
                
                # Insight: Check for rotation (Crossing lines)
                st.info(f"Visualizing the top 5 active factors. Look for lines crossing or diverging at the vertical line (Earnings Date).")
            else:
                st.warning("Insufficient beta data for rotation analysis.")

        st.markdown("---")
        
        # --- Step 2b: Volatility Analysis ---
        st.subheader("2b. Volatility Analysis (Fear Cycle)")
        st.caption("Analyzing Realized Volatility behavior (21-Day Rolling) around earnings. Does risk accumulate before the event?")
        
        # Calculate Profiles
        vol_profile = calculate_volatility_profiles(full_df, selected_stock, earnings_df['Earnings Date'], window_size)
        
        if vol_profile is not None:
            fig_ramp, fig_comp = plot_volatility_analysis(vol_profile, selected_stock)
            
            col_vol1, col_vol2 = st.columns(2)
            with col_vol1:
                st.markdown("**1. The Fear Cycle (Total Volatility Ramp)**")
                st.plotly_chart(fig_ramp, width="stretch")
                st.caption("Higher peaks at T=0 indicate market stress/uncertainty pricing.")
            
            with col_vol2:
                st.markdown("**2. Risk Composition (Idio vs. Market)**")
                st.plotly_chart(fig_comp, width="stretch")
                st.caption("Red Area = Stock-Specific Risk (Earnings Attention). Blue Area = General Market Noise.")
        else:
            st.info("Insufficient data to calculate volatility profiles.")

        st.markdown("---")

        # --- Step 3: Factor Attribution (Latest) ---
        st.subheader("3. Factor Attribution (Latest Event)")
        # Use the logic from Deep Dive but summary style
        # Get latest event window
        if latest_event_date:
             # Find T+1 for latest event
            latest_t1 = full_df[full_df['Date'] == latest_event_date + pd.Timedelta(days=1)]
            if not latest_t1.empty:
                row = latest_t1.iloc[0]
                total_ret = row[selected_stock]
                idio_ret = row['Idiosyncratic_Return']
                factor_ret = row['Factor_Return']
                
                # Check Flags
                flag_single_factor = False
                factors = {col: row[f'Contrib_{col}'] for col in factor_cols}
                sorted_factors = sorted(factors.items(), key=lambda item: abs(item[1]), reverse=True)
                top_factor_name, top_factor_val = sorted_factors[0]
                
                if abs(top_factor_val) > 0.5 * abs(factor_ret) and abs(factor_ret) > 0.005:
                    flag_single_factor = True
                
                # Re-calculate vol for this check
                latest_series_pre = full_df[full_df['Date'] < latest_event_date][selected_stock]
                fc_vol, _ = get_garch_forecast(latest_series_pre)
                
                flag_unexpected = abs(idio_ret) > 2 * fc_vol # Approx 2 sigma
                
                col_att1, col_att2, col_att3 = st.columns(3)
                col_att1.metric("Latest Earnings Move", f"{total_ret:.2%}")
                col_att2.metric("Factor Contribution", f"{factor_ret:.2%}", help="Explained by Market/Sector")
                col_att3.metric("Idiosyncratic (Alpha)", f"{idio_ret:.2%}", help="Unexplained / Earnings Surprise")
                
                if flag_single_factor:
                    st.warning(f"⚠️ Single-Factor Dependency: {top_factor_name} drove {top_factor_val/factor_ret:.0%} of the factor move.")
                if flag_unexpected:
                    st.error(f"⚠️ Unexpected Move: Alpha > 2x Volatility Forecast. True Earnings Surprise.")
            else:
                st.info("No T+1 data for latest earnings.")
        
        st.markdown("---")
        
        # --- Step 4: Risk Decomposition (Ex-ante 60d) ---
        st.subheader("4. Risk Decomposition (Ex-ante 60d)")
        # Calculate recent variance decomposition
        # Var(Stock) approx Var(Factor Component) + Var(Idiosyncratic)
        window_60 = full_df.iloc[-60:].copy()
        var_factor = window_60['Factor_Return'].var() * 252 # Annualized
        var_idio = window_60['Idiosyncratic_Return'].var() * 252
        var_total = var_factor + var_idio
        
        pct_factor = var_factor / var_total
        pct_idio = var_idio / var_total
        
        col_decomp1, col_decomp2 = st.columns([1, 2])
        
        with col_decomp1:
            st.metric("Total Annualized Volatility", f"{np.sqrt(var_total):.2%}")
            if pct_idio > 0.5:
                st.error(f"⚠️ High Specific Risk: {pct_idio:.0%} of vol is Idiosyncratic.")
            elif pct_factor > 0.8:
                st.warning(f"⚠️ Factor Dominated: {pct_factor:.0%} of vol is Systematic.")
            else:
                st.success("✅ Balanced Risk Profile")
                
        with col_decomp2:
            fig_risk = go.Figure()
            fig_risk.add_trace(go.Bar(y=['Risk Source'], x=[pct_factor], name='Factor Risk', orientation='h', marker_color='#3366CC'))
            fig_risk.add_trace(go.Bar(y=['Risk Source'], x=[pct_idio], name='Specific Risk', orientation='h', marker_color='#DC3912'))
            fig_risk.update_layout(barmode='stack', title="Variance Decomposition", xaxis_tickformat='.0%', height=200, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_risk, width="stretch")
            
        # Detailed Systematic Breakdown
        st.caption("Drivers of Systematic Risk")
        # Calc var of each contrib col
        factor_vars = {}
        for col in factor_cols:
            c_name = f'Contrib_{col}'
            if c_name in window_60.columns:
                factor_vars[col] = window_60[c_name].var() * 252
        
        if factor_vars:
            # Sort and normalize
            sorted_vars = sorted(factor_vars.items(), key=lambda x: x[1], reverse=True)
            top_drivers = sorted_vars[:5] # Top 5
            
            driver_names = [x[0] for x in top_drivers]
            driver_vals = [x[1] for x in top_drivers]
            total_driver_var = sum(factor_vars.values()) # Sum of all factor vars (approx var_factor)
            
            # Normalize to % of Systematic
            driver_pcts = [v / total_driver_var for v in driver_vals]
            
            fig_drivers = px.bar(x=driver_pcts, y=driver_names, orientation='h', 
                                 title="Top Factors Driving Systematic Risk",
                                 labels={'x': '% of Systematic Variance', 'y': 'Factor'})
            fig_drivers.update_layout(yaxis=dict(autorange="reversed"), xaxis_tickformat='.0%', height=300)
            st.plotly_chart(fig_drivers, width="stretch")
            
        st.markdown("---")

        # --- Step 5: Scenario Analysis (Interactive) ---
        st.subheader("5. Scenario Analysis (Interactive Builder)")
        st.caption("Define market shocks to test portfolio resilience.")
        
        if not loadings_df.empty:
            latest_loadings = loadings_df.sort_values('Date').iloc[-1]
            
            # Input Columns
            col_sc1, col_sc2, col_sc3, col_sc4 = st.columns(4)
            with col_sc1:
                shock_mkt = st.number_input("Market Shock %", value=-5.0, step=1.0) / 100
            with col_sc2:
                shock_semi = st.number_input("Semis Shock %", value=0.0, step=1.0) / 100
            with col_sc3:
                shock_val = st.number_input("Value Factor %", value=0.0, step=1.0) / 100
            with col_sc4:
                shock_mom = st.number_input("Momentum %", value=0.0, step=1.0) / 100
            
            # Calculation
            total_impact = 0.0
            
            # Map inputs to likely column names
            # We assume columns exist like 'Market_Load', 'Semiconductors_Load', etc.
            # Using basic factor_cols list for fuzzy matching if needed, but direct look up is safer
            
            def get_beta(name):
                return latest_loadings[f"{name}_Load"] if f"{name}_Load" in latest_loadings else (latest_loadings[name] if name in latest_loadings else 0.0)

            impact_mkt = get_beta("Market") * shock_mkt
            impact_semi = get_beta("Semiconductors") * shock_semi
            impact_val = get_beta("Value") * shock_val
            impact_mom = get_beta("Momentum") * shock_mom
            
            total_impact = impact_mkt + impact_semi + impact_val + impact_mom
            
            # Display Result Card
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: #262730; border: 1px solid #484b55; text-align: center;">
                <h3>Estimated Price Impact</h3>
                <h1 style="color: {'#ff4b4b' if total_impact < 0 else '#00cc96'};">{total_impact:+.2%}</h1>
                <p>Based on current factor loadings (Betas).</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("View Beta Details"):
                st.write(f"**Market Beta**: {get_beta('Market'):.2f}")
                st.write(f"**Semiconductors Beta**: {get_beta('Semiconductors'):.2f}")
                st.write(f"**Value Beta**: {get_beta('Value'):.2f}")
                st.write(f"**Momentum Beta**: {get_beta('Momentum'):.2f}")

        else:
            st.warning("No loadings available for scenario analysis.")
        
        st.markdown("---")
        st.markdown("---")

        # --- Step 6: Exposure Trend Monitoring ---
        st.subheader("6. Exposure Trend Monitoring")
        # Sparkline of Beta
        if 'Rolling_Beta' in rolling_stats.columns:
            st.caption("Rolling Market Beta (60-day)")
            # Use last year for trend
            trend_data = rolling_stats.iloc[-252:] 
            fig_trend = px.line(trend_data, x='Date', y='Rolling_Beta', height=150)
            fig_trend.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            st.plotly_chart(fig_trend, width="stretch")
            
        st.markdown("---")

        # --- Step 7: Event Flags & Alerts ---
        st.subheader("7. Event Flags & Alerts")
        
        flags_triggered = []
        
        # 1. Regime Flag
        if 'Correlation_Regime' in rolling_stats.columns:
            curr_regime = rolling_stats['Correlation_Regime'].iloc[-1]
            if curr_regime > 0.5:
                flags_triggered.append(f"🔴 **CRITICAL**: Crisis Regime Detected (Avg Corr {curr_regime:.2f}). Diversification likely failing.")
        
        # 2. Beta Drift Flag
        # Already calculated in Step 2: drift_val
        if abs(drift_val) > 0.5: # Arbitrary large drift
            flags_triggered.append(f"🟠 **WARNING**: Major Beta Drift. Risk profile changed by +{abs(drift_val):.2f}.")
            
        # 3. Earnings Shock (from Step 3 attribution)
        # We need to pull that flag from Step 3... easiest is to re-check
        # flag_unexpected was logical variable in Step 3. 
        # Re-eval:
        if latest_event_date:
            latest_t1 = full_df[full_df['Date'] == latest_event_date + pd.Timedelta(days=1)]
            if not latest_t1.empty:
                last_idio = latest_t1.iloc[0]['Idiosyncratic_Return']
                if abs(last_idio) > 2.5 * forecasted_vol:  # 2.5 sigma
                    flags_triggered.append(f"🟡 **NOTE**: Last earnings was a >2.5 Sigma Surprise.")

        if flags_triggered:
            for f in flags_triggered:
                st.markdown(f)
        else:
            st.success("✅ No Critical Risk Flags Triggered.")

        st.markdown("---")
        
        # --- Step 8: Report Generation ---
        st.subheader("8. Report Generation")
        st.markdown("Ready to finalize? Generate the AI Summary to synthesize these metrics into a text report.")
        if st.button("Go to AI Summary ➔"):
            # We can't easily jump tabs programmatically in pure Streamlit without rerun tricks
            # Just encourage user to click tab 4
            st.info("Please click the '🤖 AI Summary' tab at the top of the page.")
            
    with tab1:
        st.header(f"Aggregate Earnings Impact for {selected_stock}")
        
        all_events = []
        for date in earnings_df['Earnings Date']:
            w_df = get_event_window(full_df, date, window_size)
            if w_df is not None:
                w_df['Event_ID'] = date.strftime('%Y-%m-%d')
                w_df['Quarter'] = date.quarter # Add quarter info
                all_events.append(w_df)
        
        if all_events:
            combined_events = pd.concat(all_events)
            reaction_days = combined_events[combined_events['Rel_Day'] == 1]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("**Avg Abs Move (T+1)**", f"{reaction_days[selected_stock].abs().mean():.2%}")
            with col2:
                st.metric("**Avg Idiosyncratic Move**", f"{reaction_days['Idiosyncratic_Return'].abs().mean():.2%}")
            with col3:
                corr_t1 = reaction_days[selected_stock].corr(reaction_days['Idiosyncratic_Return'])
                st.metric("**Total/Idio Correlation**", f"{corr_t1:.2f}")
            with col4:
                up_moves = (reaction_days[selected_stock] > 0).sum()
                st.metric("**Positive Reactions**", f"{up_moves}/{len(reaction_days)}")

            # Cumulative Return Chart
            avg_cum_ret = combined_events.groupby('Rel_Day')[[selected_stock, 'Factor_Return', 'Idiosyncratic_Return']].mean().cumsum()
            
            fig_agg = px.line(avg_cum_ret, x=avg_cum_ret.index, y=[selected_stock, 'Factor_Return', 'Idiosyncratic_Return'],
                              title=f"Average Cumulative Returns (T-{window_size} to T+{window_size})",
                              labels={'value': 'Cumulative Return', 'Rel_Day': 'Days Relative to Earnings'})
            
            fig_agg.add_vline(x=0, line_dash="dash", line_color="white", annotation_text="Announcement")
            st.plotly_chart(fig_agg, width="stretch")

            st.markdown("---")
            
            # --- NEW SECTION: Advanced Analytics ---
            st.subheader("Advanced Analytics")
            col_adv1, col_adv2 = st.columns(2)

            with col_adv1:
                # 1. Seasonality Analysis
                st.markdown("**Seasonality: Avg Absolute Move by Quarter**")
                quarterly_stats = reaction_days.groupby('Quarter')[selected_stock].apply(lambda x: x.abs().mean()).reset_index()
                quarterly_stats['Quarter'] = 'Q' + quarterly_stats['Quarter'].astype(str)
                fig_season = px.bar(quarterly_stats, x='Quarter', y=selected_stock, 
                                    labels={selected_stock: "Avg Abs Return"},
                                    color=selected_stock, color_continuous_scale="Blues")
                fig_season.update_layout(yaxis_tickformat=".2%")
                st.plotly_chart(fig_season, width="stretch")

            with col_adv2:
                # 2. Alpha vs Total Correlation Plot
                st.markdown("**Correlation: Total Return vs. Alpha (T+1)**")
                fig_corr = px.scatter(reaction_days, x=selected_stock, y='Idiosyncratic_Return',
                                      hover_data=['Event_ID'],
                                      labels={selected_stock: 'Total T+1 Return', 'Idiosyncratic_Return': 'Idiosyncratic (Alpha)'},
                                      trendline="ols")
                fig_corr.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".1%")
                st.plotly_chart(fig_corr, width="stretch")

            # --- ENHANCED SECTION: Pre & Post-Event Patterns ---
            st.subheader("Pre & Post-Event Patterns (Anticipation & Drift)")
            
            # --- Calculate Pre-event and Post-event data ---
            drift_data = []
            pre_event_data = []
            for ev_df in all_events:
                try:
                    # Pre-event run-up
                    pre_event_ret = ev_df[ev_df['Rel_Day'] < 0][selected_stock].sum()
                    pre_event_data.append({'Event': ev_df['Event_ID'].iloc[0], 'Pre-Event Run-up': pre_event_ret})

                    # Post-event drift
                    t1_close = ev_df[ev_df['Rel_Day'] == 1][selected_stock].values[0]
                    drift_ret = ev_df[(ev_df['Rel_Day'] > 1) & (ev_df['Rel_Day'] <= window_size)][selected_stock].sum()
                    drift_data.append({'Event': ev_df['Event_ID'].iloc[0], 'T+1 Move': t1_close, 'Drift (T+1 to End)': drift_ret})
                except:
                    continue
            
            col_pre, col_post = st.columns(2)

            with col_pre:
                # --- 1. Pre-Earnings Anticipation ---
                st.markdown("**Pre-Earnings Anticipation**")
                if pre_event_data:
                    pre_event_df = pd.DataFrame(pre_event_data)
                    fig_pre_event = px.box(pre_event_df, y='Pre-Event Run-up', points="all",
                                           title=f"Distribution of Pre-Event Run-up (T-{window_size} to T-1)")
                    fig_pre_event.update_layout(yaxis_tickformat=".1%")
                    st.plotly_chart(fig_pre_event, width="stretch")

            with col_post:
                # --- 2. Conditional Post-Earnings Drift ---
                st.markdown("**Conditional Post-Earnings Drift**")
                if drift_data:
                    drift_df = pd.DataFrame(drift_data)
                    pos_t1_drift = drift_df[drift_df['T+1 Move'] > 0]['Drift (T+1 to End)'].mean()
                    neg_t1_drift = drift_df[drift_df['T+1 Move'] < 0]['Drift (T+1 to End)'].mean()

                    drift_analysis_df = pd.DataFrame({
                        "Condition": ["After Positive T+1 Move", "After Negative T+1 Move"],
                        "Average Drift": [pos_t1_drift, neg_t1_drift]
                    })
                    fig_cond_drift = px.bar(drift_analysis_df, x="Condition", y="Average Drift",
                                            color="Condition", title=f"Avg. Drift (T+2 to T+{window_size})")
                    fig_cond_drift.update_layout(yaxis_tickformat=".2%")
                    st.plotly_chart(fig_cond_drift, width="stretch")

            # --- Original PEAD Scatter Plot for context ---
            if drift_data:
                drift_df = pd.DataFrame(drift_data)
                fig_drift = px.scatter(drift_df, x='T+1 Move', y='Drift (T+1 to End)', 
                                       hover_name='Event',
                                       title=f"Price Drift (Days +2 to +{window_size}) vs. Initial Reaction (Day +1)",
                                       labels={'T+1 Move': 'Initial Reaction (T+1)', 'Drift (T+1 to End)': f'Subsequent Drift (Next {window_size-1} Days)'})
                fig_drift.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_drift.add_vline(x=0, line_dash="dash", line_color="gray")
                fig_drift.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".1%")
                st.plotly_chart(fig_drift, width="stretch")
                st.caption("Points in Top-Right/Bottom-Left quadrants indicate **Momentum** (Drift follows move). Top-Left/Bottom-Right indicate **Reversal**.")

            st.markdown("---")
            st.subheader("Historical Correlation with Risk Factors")
            st.caption("This chart shows the historical correlation between the stock's daily returns and the daily returns of each risk factor over the entire dataset.")
            
            # Calculate correlations from the returns dataframe
            if selected_stock in returns_df.columns:
                factor_correlations = returns_df.corr(numeric_only=True)[selected_stock].drop(selected_stock).sort_values()
                
                fig_corr_bar = px.bar(
                    factor_correlations, 
                    x=factor_correlations.values, 
                    y=factor_correlations.index, 
                    orientation='h',
                    title=f"Historical Correlation of {selected_stock} with Risk Factors"
                )
                st.plotly_chart(fig_corr_bar, width="stretch")



    # --- Tab 2: Comparison ---
    with tab2:
        st.header(f"Compare Two {selected_stock} Earnings Events")
        comp_col1, comp_col2 = st.columns(2)
        
        default_idx_a = len(earnings_df)-1 if len(earnings_df) >= 1 else 0
        default_idx_b = len(earnings_df)-2 if len(earnings_df) >= 2 else 0

        # --- Event A Selection ---
        with comp_col1:
            date_a_str = st.selectbox("Event A", earnings_df['Earnings Date'].dt.strftime('%Y-%m-%d'), index=default_idx_a, key="comp_a")
            date_a = pd.to_datetime(date_a_str)
            t1_a = full_df[full_df['Date'] == date_a + pd.Timedelta(days=1)]
            if not t1_a.empty:
                st.plotly_chart(create_waterfall(t1_a.iloc[0], factor_cols, f"Attribution: {date_a_str}"), width="stretch")
        
        # --- Event B Selection ---
        with comp_col2:
            date_b_str = st.selectbox("Event B", earnings_df['Earnings Date'].dt.strftime('%Y-%m-%d'), index=default_idx_b, key="comp_b")
            date_b = pd.to_datetime(date_b_str)
            t1_b = full_df[full_df['Date'] == date_b + pd.Timedelta(days=1)]
            if not t1_b.empty:
                st.plotly_chart(create_waterfall(t1_b.iloc[0], factor_cols, f"Attribution: {date_b_str}"), width="stretch")

        st.markdown("---")

        # --- ADDED: Metrics Comparison Table ---
        st.subheader("Key Metrics Comparison (T+1)")
        if not t1_a.empty and not t1_b.empty:
            row_a = t1_a.iloc[0]
            row_b = t1_b.iloc[0]
            
            alpha_pct_a = (row_a['Idiosyncratic_Return'] / row_a[selected_stock]) if row_a[selected_stock] != 0 else 0
            alpha_pct_b = (row_b['Idiosyncratic_Return'] / row_b[selected_stock]) if row_b[selected_stock] != 0 else 0

            comp_data = {
                "Metric": ["Total Return", "Idiosyncratic (Alpha) Return", "Factor-driven Return", "Alpha as % of Total"],
                f"Event A ({date_a_str})": [f"{row_a[selected_stock]:.2%}", f"{row_a['Idiosyncratic_Return']:.2%}", f"{row_a['Factor_Return']:.2%}", f"{alpha_pct_a:.1%}"],
                f"Event B ({date_b_str})": [f"{row_b[selected_stock]:.2%}", f"{row_b['Idiosyncratic_Return']:.2%}", f"{row_b['Factor_Return']:.2%}", f"{alpha_pct_b:.1%}"]
            }
            st.dataframe(pd.DataFrame(comp_data), width="stretch", hide_index=True)

        # --- ADDED: Cumulative Return Window Comparison ---
        st.subheader("Cumulative Return Window Comparison")
        window_a = get_event_window(full_df, date_a, window_size)
        window_b = get_event_window(full_df, date_b, window_size)

        if window_a is not None and window_b is not None:
            # Calculate cumulative returns for each window
            window_a['Cumulative Return'] = window_a[selected_stock].cumsum()
            window_b['Cumulative Return'] = window_b[selected_stock].cumsum()

            # Combine for plotting
            window_a['Event'] = f"Event A ({date_a_str})"
            window_b['Event'] = f"Event B ({date_b_str})"
            combined_windows = pd.concat([window_a, window_b])

            fig_cum_comp = px.line(combined_windows, x='Rel_Day', y='Cumulative Return', color='Event',
                                   title="Cumulative Performance During Event Windows",
                                   labels={'Rel_Day': 'Days Relative to Earnings', 'Cumulative Return': 'Cumulative Stock Return'})
            fig_cum_comp.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Announcement")
            st.plotly_chart(fig_cum_comp, width="stretch")

    # --- Tab 3: Individual Deep Dive (Enhanced) ---
    with tab3:
        st.header(f"Individual Deep Dive: {selected_stock} - {selected_date}")
        event_window = get_event_window(full_df, selected_date_dt, window_size)
        
        if event_window is not None:
            t_plus_1 = event_window[event_window['Rel_Day'] == 1]
            
            if not t_plus_1.empty:
                row = t_plus_1.iloc[0]
                
                # 1. Return Attribution Waterfall
                st.subheader(f"1. Return Attribution (T+1)")
                st.plotly_chart(create_waterfall(row, factor_cols, ""), width="stretch")

                # 2. Top Contributors Text
                all_contribs = {col: row[f'Contrib_{col}'] for col in factor_cols}
                all_contribs['Idiosyncratic'] = row['Idiosyncratic_Return']
                sorted_contribs = sorted(all_contribs.items(), key=lambda x: x[1], reverse=True)
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.markdown("**Top Positive Drivers:**")
                    for k, v in [i for i in sorted_contribs if i[1] > 0][:3]:
                        st.write(f"- {k}: `{v:.2%}`")
                with col_d2:
                    st.markdown("**Top Negative Drivers:**")
                    for k, v in [i for i in reversed(sorted_contribs) if i[1] < 0][:3]:
                        st.write(f"- {k}: `{v:.2%}`")

                st.markdown("---")

                # 3. Historical Context Histogram (New Feature)
                st.subheader("2. Historical Context")
                col_h1, col_h2 = st.columns([2, 1])
                
                with col_h1:
                    fig_hist = px.histogram(full_df, x=selected_stock, nbins=100, title="Distribution of All Historical Daily Returns",
                                            labels={selected_stock: 'Daily Return'})
                    fig_hist.add_vline(x=row[selected_stock], line_dash="dash", line_color="red", 
                                       annotation_text=f"Event: {row[selected_stock]:.2%}")
                    st.plotly_chart(fig_hist, width="stretch")
                
                with col_h2:
                    percentile = (full_df[selected_stock] < row[selected_stock]).mean() * 100
                    st.info(f"The T+1 return of **{row[selected_stock]:.2%}** is in the **{percentile:.1f}th percentile** of all daily returns since Sept 2022.")

                st.markdown("---")

                # 4. Factor Exposure Evolution (New Feature)
                st.subheader("3. Factor Beta Evolution (Risk Exposure)")
                top_factors = [x[0] for x in sorted(all_contribs.items(), key=lambda x: abs(x[1]), reverse=True) if x[0] != 'Idiosyncratic'][:5]
                selected_factors = st.multiselect("Select Factors to View Betas:", factor_cols, default=top_factors)
                
                if selected_factors:
                    loading_cols = [f"{f}_Load" for f in selected_factors]
                    fig_load = px.line(event_window, x='Rel_Day', y=loading_cols, 
                                       title="Factor Loadings (Betas) During Event Window",
                                       labels={'value': 'Beta', 'Rel_Day': 'Days Relative to Earnings'})
                    fig_load.add_vline(x=0, line_dash="dash", line_color="white", annotation_text="Announcement")
                    st.plotly_chart(fig_load, width="stretch")

                st.markdown("---")
                
                # --- 5. Volatility Context (GARCH Implementation) ---
                st.subheader("4. Volatility Context")
                
                # Get returns history before the event day
                pre_event_returns = full_df[full_df['Date'] < row['Date']][selected_stock]
                
                forecasted_vol, vol_method = get_garch_forecast(pre_event_returns)
                sigma_move = abs(row[selected_stock] / forecasted_vol) if forecasted_vol > 0 else 0
                
                col_v1, col_v2 = st.columns(2)
                col_v1.metric("Forecasted Daily Volatility (T+1)", f"{forecasted_vol:.2%}", help=f"Volatility forecast using {vol_method}.")
                col_v2.metric("Sigma of T+1 Move", f"{sigma_move:.2f}x", help="How many standard deviations the T+1 move was, based on the forecast.")

                st.markdown("---")

                # 4. Rolling Beta Verification (New Feature)
                st.subheader("5. Rolling Beta Verification")
                st.caption("Comparison of the 'Provided Beta' (from data vendor) vs. an 'Estimated Rolling Beta' (calculated on-the-fly using 60-day OLS). Large divergences may indicate methodology differences or data issues.")
                
                # Merge rolling stats into event window for plotting
                # event_window has 'Date', rolling_stats has 'Date'
                verify_df = pd.merge(event_window, rolling_stats, on='Date', how='left')
                
                # Plot
                # Plot
                # Identify "Provided Beta" column (usually Market_Load)
                provided_beta_col = None
                if 'Beta_Load' in verify_df.columns:
                    provided_beta_col = 'Beta_Load'
                elif 'Market_Load' in verify_df.columns:
                    provided_beta_col = 'Market_Load'
                
                # Check what we have
                has_rolling = 'Rolling_Beta' in verify_df.columns
                has_provided = provided_beta_col is not None
                
                if has_rolling or has_provided:
                    fig_beta = go.Figure()
                    
                    if has_provided:
                        fig_beta.add_trace(go.Scatter(x=verify_df['Rel_Day'], y=verify_df[provided_beta_col], name='Provided Beta (Source)', line=dict(color='orange')))
                    
                    if has_rolling:
                        fig_beta.add_trace(go.Scatter(x=verify_df['Rel_Day'], y=verify_df['Rolling_Beta'], name='Estimated Beta (60d OLS)', line=dict(color='cyan', dash='dot')))
                    
                    fig_beta.update_layout(title="Beta Stability Verification", xaxis_title="Days Relative to Earnings", yaxis_title="Beta")
                    fig_beta.add_vline(x=0, line_dash="dash", line_color="white")
                    st.plotly_chart(fig_beta, width="stretch")
                else:
                    st.warning("Could not find 'Rolling_Beta' or 'Market_Load' columns for verification.")


                with st.expander("View Raw Event Data"):
                    st.dataframe(event_window)
                    
                    # CSV Download Button
                    csv = event_window.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Event CSV",
                        data=csv,
                        file_name=f"{selected_stock}_event_{selected_date}.csv",
                        mime='text/csv',
                    )

    # --- Tab 4: AI Summary (Multi-Agent) ---
    with tab4:
        st.header("🤖 AI Risk & Strategy Report")
        
        # --- Multi-Agent Selector ---
        st.markdown("### Choose Your Analyst Persona")
        agent_options = ["Risk Manager 🛡️", "Portfolio Manager 💼", "Quantitative Analyst 🔢"]
        selected_agent = st.selectbox("Select Agent Persona:", agent_options, index=0)
        
        st.info(f"**Current Agent**: {selected_agent}")
        
        if st.button("Generate Report"):
            if not api_key:
                st.error("Please enter an API Key in the sidebar.")
            else:
                with st.spinner(f"🤖 {selected_agent} is analyzing {selected_stock}..."):
                        
                        # Gather necessary data
                        # We need 'reaction_days' and 'combined_events' computed in Tab 1
                        # Re-computing strictly what is needed here to avoid global variable dependency hell if tab 1 wasn't visited
                        # Ideally, these should be session_state, but for now we follow the "stateless" tab pattern
                        
                        # Re-calc event window data for AI context
                        ai_events = []
                        for date in earnings_df['Earnings Date']:
                            w_df = get_event_window(full_df, date, window_size)
                            if w_df is not None:
                                w_df['Event_ID'] = date.strftime('%Y-%m-%d')
                                ai_events.append(w_df)
                        
                        ai_combined = pd.concat(ai_events) if ai_events else pd.DataFrame()
                        ai_reaction = ai_combined[ai_combined['Rel_Day'] == 1] if not ai_combined.empty else pd.DataFrame()
                        
                        # Get latest two events for commentary
                        latest_evt = ai_reaction.sort_values('Date').iloc[-1:] if not ai_reaction.empty else pd.DataFrame()
                        second_latest = ai_reaction.sort_values('Date').iloc[-2:-1] if len(ai_reaction) > 1 else pd.DataFrame()

                        # Calculate Volatility Profile for AI Context
                        vol_profile = calculate_volatility_profiles(full_df, selected_stock, earnings_df['Earnings Date'], window_size)

                        report = generate_ai_summary(
                            api_key=api_key, 
                            stock_ticker=selected_stock, 
                            full_df=full_df, 
                            reaction_days=ai_reaction, 
                            latest_reaction=latest_evt, 
                            second_latest_reaction=second_latest, 
                            factor_cols=factor_cols, 
                            combined_events=ai_combined, 
                            rolling_stats=rolling_stats, 
                            loadings_df=loadings_df,
                            vol_profile=vol_profile, # Passed new data
                            model_name=selected_model,
                            agent_persona=selected_agent
                        )
                        
                        st.markdown("### AI Report")
                        st.markdown(report)
                        st.success("Generation Complete.")
                        
                        # Download button for the generated report
                        st.download_button(
                            label="📥 Download Report",
                            data=report,
                            file_name=f"{selected_stock}_AI_Report_{selected_agent.replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d')}.md",
                            mime="text/markdown",
                        )





if __name__ == "__main__":
    main()