import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import google.generativeai as genai
from scipy.stats.mstats import winsorize
from arch import arch_model

# --- Page Config ---
st.set_page_config(layout="wide", page_title="KR Capital: Earnings Dashboard")

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

# --- Plotly Waterfall Chart Function ---
def create_waterfall(row, factor_cols, title):
    factors = {col: row[f'Contrib_{col}'] for col in factor_cols}
    sorted_factors = dict(sorted(factors.items(), key=lambda item: abs(item[1]), reverse=True))
    
    top_n = 5
    wf_labels = ['Idiosyncratic (Alpha)'] + list(sorted_factors.keys())[:top_n] + ['Other Factors']
    wf_values = [row['Idiosyncratic_Return']] + list(sorted_factors.values())[:top_n]
    
    other_sum = sum(list(sorted_factors.values())[top_n:])
    wf_values.append(other_sum)

    fig = go.Figure(go.Waterfall(
        name = "Return Attribution", orientation = "v",
        measure = ["relative"] * len(wf_values) + ["total"],
        x = wf_labels + ["Total Return"],
        textposition = "outside",
        textfont_size=10, 
        text = [f"{v:.1%}" for v in wf_values + [sum(wf_values)]],
        y = wf_values + [0],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title = title,
        showlegend = False,
        yaxis_tickformat = '.1%',
        height=500
    )
    return fig

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

def generate_ai_summary(api_key, stock_ticker, full_df, reaction_days, latest_reaction, second_latest_reaction, factor_cols, combined_events):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro-latest')
    except Exception as e:
        return f"Error configuring AI model: {e}"

    # --- 1. Aggregate Metrics ---
    avg_abs_move = reaction_days[stock_ticker].abs().mean()
    avg_idio_move = reaction_days['Idiosyncratic_Return'].abs().mean()
    avg_alpha_contrib = (reaction_days['Idiosyncratic_Return'].abs() / reaction_days[stock_ticker].abs()).mean()
    up_moves = (reaction_days[stock_ticker] > 0).sum()
    total_moves = len(reaction_days)

    
    # Calculate average absolute move per quarter to identify seasonal volatility
    seasonality = reaction_days.groupby('Quarter')[stock_ticker].apply(lambda x: x.abs().mean()).to_dict()
    seasonality_str = ", ".join([f"Q{int(k)}: {v:.1%}" for k, v in seasonality.items()])

    # --- 2. Behavior Metrics (Drift) ---
    drift_correlations = []
    for event_id in reaction_days['Event_ID'].unique():
        event_window = combined_events[combined_events['Event_ID'] == event_id]
        if len(event_window) >= 5:
            t1_move = event_window[event_window['Rel_Day'] == 1][stock_ticker].values
            drift_move = event_window[(event_window['Rel_Day'] > 1) & (event_window['Rel_Day'] <= 5)][stock_ticker].sum()
            if len(t1_move) > 0:
                drift_correlations.append({'t1': t1_move[0], 'drift': drift_move})
    
    drift_df = pd.DataFrame(drift_correlations)
    drift_corr = drift_df['t1'].corr(drift_df['drift']) if not drift_df.empty else 0
    behavior_desc = "Momentum (Continuation)" if drift_corr > 0.15 else "Mean Reversion (Reversal)" if drift_corr < -0.15 else "Random Walk (No pattern)"

    # --- 3. Latest Event Data ---
    row = latest_reaction.iloc[0]
    latest_date = row['Date'].strftime('%Y-%m-%d')
    latest_ret = row[stock_ticker]
    latest_idio = row['Idiosyncratic_Return']
    
    # Volatility Context (Sigma)
    pre_event_vol_series = full_df[full_df['Date'] < row['Date']][stock_ticker].rolling(30).std()
    pre_event_vol = pre_event_vol_series.iloc[-1] if not pre_event_vol_series.empty else 0.02
    sigma_move = abs(latest_ret / pre_event_vol) if pre_event_vol > 0 else 0
    
    
    percentile = (full_df[stock_ticker] < latest_ret).mean() * 100

    # Comparison Data
    prev_ret = second_latest_reaction.iloc[0][stock_ticker] if not second_latest_reaction.empty else 0
    
    # Top Factors
    factors = {col: row[f'Contrib_{col}'] for col in factor_cols}
    sorted_factors = sorted(factors.items(), key=lambda item: item[1], reverse=True)
    top_pos = [f"{k} ({v:.1%})" for k, v in sorted_factors if v > 0][:3]
    top_neg = [f"{k} ({v:.1%})" for k, v in sorted_factors if v < 0][-3:]

    prompt = f"""
    You are a fundamental Portfolio Manager at a quantitative hedge fund. Write a strategic post-earnings analysis for {stock_ticker}.
    
    **1. AGGREGATE PROFILE:**
    - Avg Earnings Move: {avg_abs_move:.1%} (of which {avg_alpha_contrib:.0%} is usually Idiosyncratic Alpha).
    - Seasonality (Avg Move by Quarter): {seasonality_str}. (Note any quarters that are typically more volatile).
    - Win Rate: {up_moves}/{total_moves} positive reactions.
    - Post-Earnings Drift Correlation: {drift_corr:.2f} ({behavior_desc}).

    **2. LATEST EVENT ({latest_date}):**
    - Move: {latest_ret:.1%} ({sigma_move:.1f}x Sigma vs 30-day baseline).
    - Historical Context: This return is in the {percentile:.1f}th percentile of all daily moves.
    - Alpha: {latest_idio:.1%}.
    - Top Drivers: Positive [{', '.join(top_pos)}], Negative [{', '.join(top_neg)}].
    - Comparison to Previous Q: Previous move was {prev_ret:.1%}.

    **OUTPUT FORMAT (Markdown):**
    
    ### 1. Executive Summary
    One sentence characterizing the event (e.g., "A volatility shock driven by pure alpha" or "A non-event dampening volatility").

    ### 2. Return Attribution & Seasonality
    Analyze the latest move vs the previous quarter. Was it sector-driven or company-specific? Does this align with the typical seasonality for this quarter ({seasonality_str})?

    ### 3. Volatility & Behavioral Analysis
    Discuss the risk. Is the {sigma_move:.1f}-Sigma move abnormal? Note the percentile ranking. Based on the Drift Correlation ({drift_corr:.2f}), what is the trading bias for the next week?

    ### 4. Conclusion
    Synthesize the "Earnings Personality" of {stock_ticker}. Is it a reliable trend-follower post-earnings, or a mean-reverting volatility trap?
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Generation Error: {e}"
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
    api_key = st.sidebar.text_input("Enter Google API Key (Optional)", type="password", help="Required for AI Summary tab.")
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Aggregate Analysis", "⚖️ Event Comparison", "🔍 Individual Deep Dive", "🤖 AI Summary", "📈 Factor Performance"])

    # --- Tab 1: Aggregate Statistics ---
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
            st.plotly_chart(fig_agg, use_container_width=True)

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
                st.plotly_chart(fig_season, use_container_width=True)

            with col_adv2:
                # 2. Alpha vs Total Correlation Plot
                st.markdown("**Correlation: Total Return vs. Alpha (T+1)**")
                fig_corr = px.scatter(reaction_days, x=selected_stock, y='Idiosyncratic_Return',
                                      hover_data=['Event_ID'],
                                      labels={selected_stock: 'Total T+1 Return', 'Idiosyncratic_Return': 'Idiosyncratic (Alpha)'},
                                      trendline="ols")
                fig_corr.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".1%")
                st.plotly_chart(fig_corr, use_container_width=True)

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
                    st.plotly_chart(fig_pre_event, use_container_width=True)

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
                    st.plotly_chart(fig_cond_drift, use_container_width=True)

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
                st.plotly_chart(fig_drift, use_container_width=True)
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
                st.plotly_chart(fig_corr_bar, use_container_width=True)

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
                st.plotly_chart(create_waterfall(t1_a.iloc[0], factor_cols, f"Attribution: {date_a_str}"), use_container_width=True)
        
        # --- Event B Selection ---
        with comp_col2:
            date_b_str = st.selectbox("Event B", earnings_df['Earnings Date'].dt.strftime('%Y-%m-%d'), index=default_idx_b, key="comp_b")
            date_b = pd.to_datetime(date_b_str)
            t1_b = full_df[full_df['Date'] == date_b + pd.Timedelta(days=1)]
            if not t1_b.empty:
                st.plotly_chart(create_waterfall(t1_b.iloc[0], factor_cols, f"Attribution: {date_b_str}"), use_container_width=True)

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
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

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
            st.plotly_chart(fig_cum_comp, use_container_width=True)

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
                st.plotly_chart(create_waterfall(row, factor_cols, ""), use_container_width=True)

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
                    st.plotly_chart(fig_hist, use_container_width=True)
                
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
                    st.plotly_chart(fig_load, use_container_width=True)

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

    # --- Tab 4: AI Summary ---
    with tab4:
        st.header(f"AI-Generated Analyst Summary for {selected_stock}")

        # Use session state to hold the summary
        if 'ai_summary' not in st.session_state:
            st.session_state.ai_summary = None
        if 'summary_stock' not in st.session_state:
            st.session_state.summary_stock = None

        if st.button("Generate Report"):
            if not api_key:
                st.warning("Please enter an API Key in the sidebar.")
                st.session_state.ai_summary = None # Clear previous summary
            else:
                # Prepare data
                reaction_days = combined_events[combined_events['Rel_Day'] == 1]
                
                # Get latest and second latest event data
                latest_event_date = earnings_df['Earnings Date'].max()
                latest_reaction = full_df[full_df['Date'] == latest_event_date + pd.Timedelta(days=1)]
                
                second_latest_event_date = earnings_df['Earnings Date'].nlargest(2).iloc[-1] if len(earnings_df) > 1 else None
                second_latest_reaction = full_df[full_df['Date'] == second_latest_event_date + pd.Timedelta(days=1)] if second_latest_event_date else pd.DataFrame()
                
                if not latest_reaction.empty:
                    with st.spinner("Analyzing..."):
                        st.session_state.ai_summary = generate_ai_summary(api_key, selected_stock, full_df, reaction_days, latest_reaction, second_latest_reaction, factor_cols, combined_events)
                        st.session_state.summary_stock = selected_stock # Store which stock this summary is for
                else:
                    st.error("No data available for the most recent earnings date.")
                    st.session_state.ai_summary = None

        # Display the summary and download button if it exists for the current stock
        if st.session_state.ai_summary and st.session_state.summary_stock == selected_stock:
            st.markdown("---")
            st.markdown(st.session_state.ai_summary)
            st.download_button(
                label="📥 Download Summary",
                data=st.session_state.ai_summary,
                file_name=f"AI_Summary_{selected_stock}_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
            )

    # --- Tab 5: Factor Performance ---
    with tab5:
        st.header("Historical Risk Factor Performance")
        st.markdown("Analyze the cumulative performance and relationships of the underlying risk factors.")

        # --- Section 1: Cumulative Performance vs. Stock ---
        st.subheader("1. Cumulative Performance Comparison")
        default_factors = ['Market', 'Semiconductors', 'Value']
        default_factors = [f for f in default_factors if f in factor_cols]

        selected_factors_perf = st.multiselect("Select factors to plot:", factor_cols, default=default_factors, key="factor_perf_select")
        compare_with_stock = st.checkbox(f"Compare with {selected_stock}", value=True)

        if selected_factors_perf or compare_with_stock:
            cols_to_plot = selected_factors_perf[:]
            if compare_with_stock:
                cols_to_plot.append(selected_stock)
            
            # Calculate cumulative returns from the original returns_df
            perf_df = returns_df[['Date'] + cols_to_plot].copy()
            perf_df[cols_to_plot] = perf_df[cols_to_plot].cumsum()

            fig_factor_perf = px.line(perf_df, x='Date', y=cols_to_plot, title="Cumulative Performance: Factors vs. Stock")
            st.plotly_chart(fig_factor_perf, use_container_width=True)

        st.markdown("---")

        # --- Section 2: Factor Correlation Matrix ---
        st.subheader("2. Factor Correlation Matrix")
        st.caption("This heatmap shows which risk factors tend to move together (high positive correlation, dark red) or in opposite directions (high negative correlation, dark blue).")
        
        # Calculate correlation on the factor returns
        factor_return_cols = [f for f in returns_df.columns if f in factor_cols]
        factor_corr = returns_df[factor_return_cols].corr()

        fig_corr_heatmap = px.imshow(factor_corr, text_auto=".2f", aspect="auto",
                                     color_continuous_scale='RdBu_r', range_color=[-1, 1],
                                     title="Historical Correlation Between Risk Factors")
        st.plotly_chart(fig_corr_heatmap, use_container_width=True)
        
        # --- Section 3: Z-Score Normalized View ---
        if z_score_view:
            st.markdown("---")
            st.subheader("3. Z-Score Normalized Factor Returns")
            st.caption("This chart shows each factor's daily return as a Z-score (number of standard deviations from its own mean). It helps identify days with unusually large moves across multiple factors on a comparable scale.")
            
            z_score_df = returns_df[['Date']].copy()
            for col in factor_return_cols:
                z_score_df[col] = (returns_df[col] - returns_df[col].mean()) / returns_df[col].std()
            
            selected_z_factors = st.multiselect("Select factors to view Z-Scores:", factor_return_cols, default=default_factors, key="z_score_select")
            if selected_z_factors:
                fig_z_score = px.line(z_score_df, x='Date', y=selected_z_factors, title="Z-Score of Daily Factor Returns")
                st.plotly_chart(fig_z_score, use_container_width=True)


if __name__ == "__main__":
    main()