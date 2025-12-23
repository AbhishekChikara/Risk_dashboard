import plotly.graph_objects as go

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

def plot_volatility_analysis(vol_df, stock_ticker):
    # Chart 1: Total Vol Ramp
    fig_ramp = go.Figure()
    fig_ramp.add_trace(go.Scatter(x=vol_df['Rel_Day'], y=vol_df['Total_Vol'], mode='lines', name='Realized Vol (21D)', line=dict(color='white', width=3)))
    
    fig_ramp.update_layout(
        title="Avg. Realized Volatility Ramp (T-30 to T+30)",
        xaxis_title="Days Relative to Earnings",
        yaxis_title="Annualized Volatility",
        yaxis_tickformat='.0%',
        height=350,
        showlegend=True
    )
    fig_ramp.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Earnings")

    # Chart 2: Stacked Area (Idio vs Factor)
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(
        x=vol_df['Rel_Day'], y=vol_df['Factor_Vol'], 
        mode='lines', name='Systematic (Market) Risk', stackgroup='one', 
        line=dict(width=0), fillcolor='rgba(50, 100, 200, 0.6)' 
    ))
    fig_comp.add_trace(go.Scatter(
        x=vol_df['Rel_Day'], y=vol_df['Idio_Vol'], 
        mode='lines', name='Idiosyncratic (Stock) Risk', stackgroup='one', 
        line=dict(width=0), fillcolor='rgba(200, 50, 50, 0.7)'
    ))

    fig_comp.update_layout(
        title="Risk Composition: Idiosyncratic vs. Systematic",
        xaxis_title="Days Relative to Earnings",
        yaxis_title="Contribution to Volatility",
        yaxis_tickformat='.0%',
        height=350,
        hovermode="x unified"
    )
    fig_comp.add_vline(x=0, line_dash="dash", line_color="white")
    
    return fig_ramp, fig_comp

def plot_earnings_histogram(reaction_df, stock_ticker):
    """
    Plots a histogram of Earnings Day returns overlaid with a Normal Distribution curve.
    Used to visualize 'Fat Tails' and outliers.
    """
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    import plotly.express as px
    import scipy.stats as stats
    import numpy as np
    
    # 1. Get Data
    data = reaction_df[stock_ticker].dropna()
    if data.empty:
        return go.Figure()

    mu = data.mean()
    sigma = data.std()
    
    # 2. Create Histogram with Curve
    # Simple Histogram first
    fig = px.histogram(reaction_df, x=stock_ticker, nbins=20, 
                       title=f"Distribution of {stock_ticker} Earnings Returns vs Normal Curve",
                       labels={stock_ticker: "Return"},
                       opacity=0.7,
                       color_discrete_sequence=['#636EFA']) # Streamlit Blueish
    
    # 3. Generate Normal Curve
    if sigma > 0:
        x_range = np.linspace(data.min() - 0.05, data.max() + 0.05, 100)
        pdf = stats.norm.pdf(x_range, mu, sigma)
        
        # Scale PDF to match Histogram frequency? 
        # Easier to standardize histogram to Probability Density or just show shape
        # Let's use Histnorm='probability density'
        fig.update_traces(histnorm='probability density')
        
        fig.add_trace(go.Scatter(x=x_range, y=pdf, mode='lines', 
                                 name='Normal Distribution', 
                                 line=dict(color='white', dash='dash')))
                                 
    fig.update_layout(xaxis_tickformat='.1%', showlegend=True)
    return fig
