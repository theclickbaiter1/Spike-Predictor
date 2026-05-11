# Pre-Market Spike Detector

A production-grade Python pipeline that scans a configurable universe of stock tickers before market open and predicts which stocks are most likely to experience significant intraday price spikes (up or down).

## How It Works

```
[7:30 AM ET]
    → Fetch overnight news (Finnhub) for all tickers
    → Score headlines with FinBERT (ProsusAI/finbert)
    → Compute technical indicators from trailing OHLCV (yfinance)
    → Fetch macro context (VIX, 10Y Treasury, sector ETFs)
    → Run XGBoost 3-class classifier: P(spike_up), P(spike_down), P(flat)
    → Output ranked watchlist to terminal + CSV
```

**This is NOT a price predictor.** It classifies whether today's session will produce a large (>3%) intraday move, using only pre-market information.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Your Finnhub API Key

Get a free API key from [finnhub.io](https://finnhub.io) and export it:

```bash
export FINNHUB_API_KEY="your_api_key_here"
```

### 3. Train the Model (First Time — ~20 min)

```bash
python3.13 spike_detector.py --retrain
```

This will:
- Download 2 years of OHLCV data for all 27 tickers
- Bulk-fetch overnight news from Finnhub (~648 API calls, cached to disk)
- Score all headlines with FinBERT (batched for efficiency)
- Train XGBoost with time-series validation + early stopping
- Save the production model to `data/model.json`

> **Note:** First run is slow (API calls + FinBERT inference). All subsequent retrains are fast thanks to disk caching.

### 4. Run Daily Predictions (< 2 min)

```bash
python3.13 spike_detector.py
```

Output: tiered watchlist in terminal + CSV saved to `output/watchlist_YYYY-MM-DD.csv`.

### 5. Backtest (Optional)

```bash
python3.13 backtest.py               # Last 6 months
python3.13 backtest.py --months 12   # Last 12 months
```

### 6. Validate Against Today's Data

```bash
python3.13 validate_today.py
```

This retrains the model excluding today, predicts today's spikes, then compares predictions against actual intraday results with matplotlib visualizations.

---

## Architecture

| Module | Purpose |
|--------|---------|
| `config.py` | Ticker universe, thresholds, XGBoost params, paths |
| `features.py` | Technical indicators, macro context, calendar/earnings features |
| `news.py` | Finnhub client (bulk + cached) + FinBERT sentiment scoring |
| `model.py` | XGBoost training, prediction, model serialization |
| `spike_detector.py` | Main pipeline (retrain + daily predict modes) |
| `backtest.py` | Historical backtesting + confusion matrix |
| `validate_today.py` | Out-of-sample validation against today's real data |

## Features (23 total)

### Sentiment (FinBERT — via Finnhub news)
- Mean, max, min, std of overnight headline sentiment scores
- News volume (attention signal)

### Technical (yfinance)
- RSI-14, EMA-10, 20-day realized volatility
- Previous day return/range, 3-day momentum
- Overnight gap (pre-market vs previous close)

### Macro / Market Context
- VIX (market fear), 10Y Treasury yield
- S&P 500 previous day return
- 5-day sector ETF momentum

### Calendar
- Day of week, Monday/Friday flags
- Days to next earnings, earnings day flag

## Scheduling (macOS)

### cron

```bash
crontab -e
# Add:
30 7 * * 1-5 cd /Users/sstasbih/Desktop/Projects/Spike\ Detector && FINNHUB_API_KEY="your_key" python3.13 spike_detector.py >> output/log.txt 2>&1
```

### launchd (recommended)

```bash
python3.13 spike_detector.py --schedule
```

## Configuration

Edit `config.py` to customize:
- `UNIVERSE` — list of tickers to scan
- `SPIKE_THRESHOLD` — minimum intraday move to classify as spike (default: 3%)
- `XGB_PARAMS` — XGBoost hyperparameters
- `SECTOR_MAP` — ticker → sector ETF mapping

## File Structure

```
Spike Detector/
├── spike_detector.py          # Main pipeline
├── backtest.py                # Historical backtesting
├── validate_today.py          # Out-of-sample validation
├── config.py                  # Configuration
├── features.py                # Feature engineering
├── model.py                   # XGBoost model
├── news.py                    # Finnhub + FinBERT
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── data/
│   ├── news_cache/            # Cached Finnhub responses
│   ├── model.json             # Trained XGBoost model
│   └── training_data.parquet  # Cached feature matrix
└── output/
    ├── watchlist_YYYY-MM-DD.csv
    ├── spike_history.csv
    ├── backtest_results.png
    └── validation_YYYY-MM-DD.png
```

## Key Design Decisions

- **No lookahead bias**: All features use only data available before 9:30 AM ET
- **Time-series split**: Validation uses the most recent 20% of days (no shuffle)
- **Class imbalance**: Handled via inverse-frequency sample weights
- **Bulk news fetching**: Monthly chunks (~648 API calls) instead of per-day (~13,500)
- **FinBERT batching**: All headlines per-ticker scored at once, then mapped per-day
- **Aggressive caching**: Finnhub responses + OHLCV cached to disk
- **Model persistence**: Daily runs load pre-trained model (no retraining)
