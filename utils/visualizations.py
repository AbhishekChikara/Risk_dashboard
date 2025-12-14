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
