# Strategic Earnings Dashboard

A professional-grade quantitative dashboard for analyzing stock behavior around earnings events. Designed for Portfolio Managers and Risk Officers to visualize return attribution, volatility dynamics, and generate AI-driven hedging strategies.

## 🚀 Key Features

### 1. Workflow Monitoring (Tab 0) & Volatility Analysis
- **Volatility Ramp ("The Fear Cycle")**: Visualize how Implied Volatility rises into earnings (T-30 to T=0) and crushes post-event.
- **Risk Composition**: Breakdown of Total Volatility into Idiosyncratic (Stock-Specific) vs Systematic (Market) components.
- **Step-by-Step Workflow**: Track the loading, cleaning, and processing of factor data.

### 2. Multi-Agent AI Strategy (Tab 4)
- **Three Distinct Personas**:
    - **🛡️ Risk Manager**: Focuses on "Kill Criteria", Magnitude Risk, and Downside Protection (Options/Collars). Uses **Event-Specific Idiosyncratic Ratios** to validate hedging strategy.
    - **💼 Portfolio Manager**: Focuses on Upside/Downside Skew, Sizing, and "The Edge".
    - **🔢 Quant Analyst**: Focuses on Statistical Significance (Sigma moves), Regimes, and Anomalies.
- **Advanced Context Engine**: 
    - **Win Rate**: Historical probability of positive reactions.
    - **Drift Amplification**: Detects if post-earnings moves tend to "run" (Momentum) or reverse.
    - **Market Regime**: Identifies "Crisis Correlation" vs "Stock Picker" environments.

### 3. Quantitative Deep Dives
- **Waterfalls**: Attribution of returns to specific factors (Market, Semi, Value, Momentum).
- **Event Comparison**: Side-by-side analysis of two historical quarters.
- **Scenario Analysis**: Stress-test portfolio P&L against market shocks (e.g., "Tech Rally +5%").

---

## 🛠️ Data Infrastructure

The project is now fully modularized for robustness and scale:
- `main_file.py`: Streamlit entry point.
- `utils/data_loader.py`: Robust CSV reading and pre-processing.
- `utils/calculations.py`: Rolling statistics, GARCH volatility forecasting, and Event-Study metrics.
- `utils/visualizations.py`: Plotly charts for waterfalls, ramps, and ramps.
- `utils/ai_summary.py`: Prompt engineering and Gemini API integration.

### File Structure
```
project_root/
├── main_file.py
├── requirements.txt
├── README.md
├── utils/              <-- Core Logic
│   ├── data_loader.py
│   ├── calculations.py
│   ├── visualizations.py
│   └── ai_summary.py
└── res/
    └── [Stock_Ticker]/ (e.g., NVDA)
        ├── 01_case_study_returns.csv
        ├── 02_case_study_factor_loadings.csv
        └── 03_case_study_earnings_dates.csv
```

---

## ⚡ Installation & Usage

1.  **Clone & Install**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Dashboard**:
    ```bash
    streamlit run main_file.py
    ```

3.  **AI Configuration**:
    - Enter your **Google Gemini API Key** in the **Sidebar**.
    - Navigate to **Tab 4 (AI Summary)** to generate reports.

---

## 📊 Methodology Notes

- **Idiosyncratic Ratio**: We calculate this *specifically* for the earnings event window (T+1), as general 60-day stats often understate event risk.
- **Drift Logic**: "Amplification" is flagged when the T+2 to T+5 move shares the same sign as the T+1 reaction.
- **Volatility**: Forecasts use a GARCH(1,1) model on pre-event returns.
