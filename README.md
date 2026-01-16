# Project Kassandra - Universal Sentiment Engine

A professional machine learning system for stock price prediction that combines historical price data with alternative sentiment data from multiple sources.

## Features

- **Multi-Source Data Fetching**: Automatically collects data from:
  - Yahoo Finance (historical price data)
  - Google Trends (search interest)
  - News RSS Feeds (sentiment analysis)
  - Reddit (social sentiment)
  - Wikipedia (page view statistics)

- **Advanced ML Pipeline**:
  - Ensemble model (XGBoost, Random Forest, Gradient Boosting)
  - Strict temporal leak prevention
  - Walk-forward validation
  - Feature importance analysis

- **Professional Web Dashboard**: Real-time Flask-based dashboard with:
  - Interactive price charts
  - Sentiment indicators
  - Model metrics visualization
  - Prediction logging

## Project Structure

```
ML Codefest/
├── README.md                 # Documentation
├── requirements.txt          # Python dependencies
├── app.py                    # Flask web server
├── run.py                    # CLI pipeline runner
├── data_check.py             # Data verification script
├── templates/
│   └── index.html            # Dashboard HTML
├── static/
│   ├── css/
│   │   └── styles.css        # Dashboard styles
│   └── js/
│       └── app.js            # Dashboard JavaScript
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py       # Multi-source data collection
│   ├── feature_engineering.py # Feature creation
│   ├── model.py              # ML models and training
│   ├── prediction_logger.py  # Logging and CSV export
│   └── pipeline.py           # Pipeline orchestration
└── outputs/                  # Generated outputs
    ├── features_*.csv        # Processed features
    ├── predictions_*.csv     # Prediction logs
    ├── model_*.pkl           # Saved models
    └── metrics_report.txt    # Evaluation metrics
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline (CLI)

```bash
# Default: TSLA with 1 year of data
python run.py

# Custom stock
python run.py AAPL

# Custom date range
python run.py GOOGL --start 2024-01-01 --end 2025-01-15

# Quick mode (skip validation)
python run.py TSLA --quick
```

### 3. Launch Web Dashboard

```bash
python app.py
```

Open your browser to: **http://localhost:8080**

### 4. Data Verification

```bash
python data_check.py
```

## Web Dashboard

The dashboard provides:

- **Configuration Panel**: Input stock ticker and date range
- **Prediction Display**: Next-day price prediction with confidence score
- **Metrics Panel**: MAE, RMSE, MAPE, R² scores
- **Price Chart**: Historical price visualization
- **Sentiment Indicators**: News, Reddit, and Google Trends sentiment
- **Model Comparison**: Individual model predictions
- **Feature Importance**: Top contributing features
- **Prediction Log**: Historical prediction accuracy

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/api/status` | GET | System status |
| `/api/run` | POST | Run pipeline |
| `/api/price-history` | GET | Historical prices |
| `/api/predictions-log` | GET | Prediction history |
| `/api/download/<file>` | GET | Download outputs |

## Model Architecture

### Ensemble Approach
The system uses an ensemble of multiple models:
- **XGBoost**: Gradient boosted trees
- **Random Forest**: Bagged decision trees
- **Gradient Boosting**: Sequential boosting
- **Ridge Regression**: Linear baseline

### Temporal Leak Prevention

All features use ONLY historical data:
- Rolling/lagged features are shifted appropriately
- Target variable represents FUTURE price
- Train/test split maintains temporal ordering
- Gap between train and test prevents leakage

## Feature Engineering

### Price-Based Features
- Lagged closing prices (1-10 days)
- Returns and price changes
- Moving averages (5, 10, 20, 50 day)
- RSI, MACD, Bollinger Bands
- Volatility measures

### Sentiment Features
- Google Trends interest scores
- News sentiment (VADER analysis)
- Reddit sentiment and bullish ratio
- Wikipedia page view momentum

### Temporal Features
- Day of week, month, quarter
- Month start/end indicators

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error (USD) |
| RMSE | Root Mean Squared Error (USD) |
| MAPE | Mean Absolute Percentage Error (%) |
| R² | Coefficient of Determination |
| Directional Accuracy | Correct up/down predictions (%) |

## Output Files

After running the pipeline:

| File | Description |
|------|-------------|
| `features_TICKER.csv` | Processed features for training |
| `prediction_log_TICKER.csv` | Date, Actual, Predicted prices |
| `model_TICKER.pkl` | Saved trained model |
| `metrics_report.txt` | Evaluation summary |

## Technical Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`
- All data sources are free and open

## Competition Deliverables

### Phase 1: Mid-Evaluation
- `data_check.py` - Data fetching verification

### Phase 2: Final Pipeline
- Complete repository with documentation
- Feature CSV export
- Prediction log CSV

### Phase 3: Mystery Stock
- Live web dashboard at http://localhost:8080
- Last Updated timestamp on UI
- Prediction report export

---

**Project Kassandra** | Prometeo '26 - ML Codefest | FinTech Track
