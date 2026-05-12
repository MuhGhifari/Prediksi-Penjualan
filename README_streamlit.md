# Sales Prediction Website

Interactive Streamlit website for Bogor sales prediction and store clustering.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the website:

```bash
streamlit run app.py
```

The app can use the default local CSV path or a CSV uploaded from the sidebar.

## Features

- Sales summary metrics
- Channel, brand, and store filters
- 7 to 90 day forecast control
- Model comparison: Linear Regression, Random Forest, Gradient Boosting
- Backtest actual vs predicted chart
- Store clustering with automatic or manual K-Means cluster count
- Forecast CSV download
- Combined Excel download for forecast, clustering, and model evaluation
