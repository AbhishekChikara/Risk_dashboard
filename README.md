# KR Capital: Earnings Dashboard

A comprehensive quantitative dashboard for analyzing stock performance around earnings events. This tool allows Portfolio Managers to visualize return attribution, factor exposure, and volatility dynamics to make data-driven decisions.

## Features

- **Aggregate Analysis**: View average absolute moves, idiosyncratic (alpha) contribution, and seasonality trends.
- **Event Comparison**: Compare two specific earnings events side-by-side with waterfall attribution charts.
- **Individual Deep Dive**: Detailed breakdown of a single event, including return attribution, historical context, and factor beta evolution.
- **AI Summary**: Generate strategic post-earnings analysis reports using Google's Gemini AI.
- **Factor Performance**: Analyze historical performance and correlations of underlying risk factors.

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.
2.  **Create a Conda environment** (recommended):
    ```bash
    conda create -n kr_capital python=3.11
    conda activate kr_capital
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Data Setup

The dashboard expects a specific directory structure for stock data. Ensure your project folder looks like this:

```
project_root/
├── main_file.py
├── requirements.txt
└── res/
    └── [Stock_Ticker]/  (e.g., NVDA)
        ├── 01_case_study_returns.csv
        ├── 02_case_study_factor_loadings.csv
        └── 03_case_study_earnings_dates.csv
```

-   **`res/`**: The main resource directory.
-   **`[Stock_Ticker]/`**: Create a subdirectory for each stock you want to analyze (e.g., `res/NVDA`, `res/AAPL`).
-   **CSV Files**: Inside each stock folder, you must have the three specific CSV files listed above.

## Usage

1.  **Run the Streamlit application**:
    ```bash
    streamlit run main_file.py
    ```
2.  **Navigate**: Open the provided local URL in your web browser (usually `http://localhost:8501`).
3.  **Select Stock**: Use the sidebar to select a stock from the available folders in `res/`.
4.  **Explore**: Use the tabs to navigate between different analysis modules.

## Configuration

-   **Event Window**: Adjust the `Event Window (+/- Days)` slider in the sidebar to change the analysis period.
-   **Winsorization**: Use the "Advanced Settings" in the sidebar to winsorize returns and reduce the impact of outliers.
-   **AI Summary**: To use the "AI Summary" tab, you must provide a valid Google API Key in the sidebar input field.

## Requirements

See `requirements.txt` for the full list of Python dependencies. Key libraries include:
-   `streamlit`
-   `pandas`
-   `plotly`
-   `google-generativeai`
-   `arch`
