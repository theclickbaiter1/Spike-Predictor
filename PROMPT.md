# Pre-Market Spike Detector — Build Prompt

> **Purpose**: Feed this entire file as a system prompt to an AI coding assistant to build the project from scratch.

---

## System Instructions

```
You are a senior quantitative developer and machine learning engineer at a top-tier algorithmic trading firm.

Goal: Build a production-grade pre-market spike detector that runs daily at 7:30 AM ET, scans a configurable universe of stock tickers, and outputs a ranked watchlist of stocks most likely to experience significant intraday price spikes (up or down) that day.

Anti-goal: Do NOT build a price-level predictor. Do NOT use VADER for financial sentiment. Do NOT use random train/test splits on time-series data. Do NOT hardcode a single ticker. Return only production-ready, highly optimized code.
```

---

## Task Specification

### 1. Architecture Overview

Build a Python pipeline with the following stages:

```
[7:30 AM Trigger]
    → Fetch overnight news (Finnhub) for all tickers in the universe
    → Score headlines with HuggingFace FinBERT (ProsusAI/finbert)
    → Fetch pre-market data (previous close, pre-market price/volume if available)
    → Compute technical indicators from trailing OHLCV data (yfinance)
    → Fetch macro context (VIX, 10Y Treasury, sector ETFs)
    → Run trained XGBoost classifier: P(spike_up), P(spike_down), P(flat)
    → Output ranked watchlist to terminal + CSV + optional webhook/notification
```

### 2. Target Variable (THIS IS CRITICAL)

The model predicts a **3-class classification** target for each stock on each day:

```python
# Define "spike" as an intraday move exceeding a configurable threshold
SPIKE_THRESHOLD = 0.03  # 3% move

def compute_target(df):
    """
    Compute the intraday return: (Close - Open) / Open
    Classify into: spike_up (+1), spike_down (-1), flat (0)
    """
    intraday_return = (df["Close"] - df["Open"]) / df["Open"]
    target = pd.Series(0, index=df.index)          # default: flat
    target[intraday_return >= SPIKE_THRESHOLD] = 1  # spike up
    target[intraday_return <= -SPIKE_THRESHOLD] = -1  # spike down
    return target
```

> **Key difference from price prediction**: We are NOT predicting tomorrow's close. We are classifying whether TODAY's session will produce a large move, using only pre-market information.

### 3. Feature Engineering

For each ticker on each day, compute these features using ONLY data available before 9:30 AM ET:

#### A. Overnight News Sentiment (FinBERT)
```
- overnight_sentiment_mean:    Mean FinBERT score of headlines published since previous close
- overnight_sentiment_max:     Most positive headline score
- overnight_sentiment_min:     Most negative headline score
- overnight_news_count:        Number of headlines (volume = attention)
- overnight_sentiment_std:     Disagreement among headlines (controversy signal)
```

#### B. Technical Indicators (from trailing daily OHLCV — yfinance)
```
- prev_close:                  Yesterday's closing price
- rsi_14:                      14-day Relative Strength Index
- ema_10:                      10-day Exponential Moving Average
- realized_vol_20d:            20-day realized volatility (annualized)
- avg_volume_10d:              10-day average volume (liquidity baseline)
- prev_day_return:             Yesterday's (Close-Open)/Open return
- prev_day_range:              Yesterday's (High-Low)/Close (intraday range %)
- gap_3d:                      3-day cumulative return (momentum)
- overnight_gap:               (today's pre-market or implied open - prev close) / prev close
```

#### C. Macro / Market Context
```
- vix:                         CBOE Volatility Index (^VIX) — market fear gauge
- treasury_10y:                10-Year Treasury Yield (^TNX) — rate expectations
- sector_momentum_5d:          5-day return of the stock's sector ETF (XLK, XLF, XLE, etc.)
- sp500_prev_return:           S&P 500 previous day return (broad market direction)
```

#### D. Calendar Features
```
- day_of_week:                 Monday=0 ... Friday=4 (encoded)
- is_monday:                   Binary flag (Mondays have higher gap risk)
- is_friday:                   Binary flag (Fridays have options expiry effects)
- days_to_earnings:            Days until next earnings report (-1 if unknown)
                               (use yfinance .calendar or similar)
- is_earnings_day:             Binary flag: is earnings today or was it after-hours yesterday?
```

### 4. Ticker Universe

Make the ticker universe configurable via a `config.yaml` or constant:

```python
# Default universe: high-volume, spike-prone stocks + major ETFs
UNIVERSE = [
    # Mega-cap tech (high beta, news-driven)
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
    # Semiconductors
    "AMD", "MU", "AVGO", "QCOM", "MRVL", "ARM",
    # AI / Quantum / Speculative
    "PLTR", "IONQ", "RGTI", "SMCI", "SNOW",
    # Sector ETFs (for cross-sector scanning)
    "SMH", "QTUM", "XLK", "XLF", "XLE", "XBI",
    # Meme / high-volatility
    "GME", "AMC", "SOFI",
]
```

### 5. Model Architecture

Use **XGBoost** with multi-class classification:

```python
import xgboost as xgb

XGB_PARAMS = dict(
    objective="multi:softprob",  # outputs probability for each class
    num_class=3,                  # spike_down, flat, spike_up
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    eval_metric="mlogloss",
    random_state=42,
    early_stopping_rounds=50,
)
```

#### Training Strategy
- **Training data**: Pull 2 years of daily OHLCV for ALL tickers in the universe. Stack them into a single training set (rows = ticker-day pairs).
- **Time-series split**: Use the most recent 20% of calendar days as the validation set. Do NOT shuffle.
- **Class imbalance**: Spikes are rare events (~5-10% of days). Use `scale_pos_weight` or SMOTE to handle imbalance, OR use a custom evaluation metric like F1 or precision@k.
- **Retrain on full data**: After finding optimal rounds via early stopping on the validation set, retrain on 100% of data for production predictions (same pattern as the price model).

### 6. Output Format

The daily output should be a **ranked watchlist** sorted by spike probability:

```
═══════════════════════════════════════════════════════════════
  SPIKE DETECTOR — 2026-05-06 07:30 AM ET
═══════════════════════════════════════════════════════════════

  🔴 HIGH PROBABILITY SPIKES (>60% confidence)
  ┌────────┬───────────┬───────────┬──────────────┬──────────────────────────┐
  │ Ticker │ Direction │ Prob (%)  │ Prev Close   │ Top Signal               │
  ├────────┼───────────┼───────────┼──────────────┼──────────────────────────┤
  │ MU     │ ▲ UP      │ 78.3%     │ $542.21      │ Earnings beat + sentiment│
  │ TSLA   │ ▼ DOWN    │ 65.1%     │ $248.50      │ Negative overnight news  │
  └────────┴───────────┴───────────┴──────────────┴──────────────────────────┘

  🟡 MODERATE PROBABILITY (40-60%)
  ┌────────┬───────────┬───────────┬──────────────┬──────────────────────────┐
  │ NVDA   │ ▲ UP      │ 52.4%     │ $198.30      │ Sector momentum + RSI    │
  └────────┴───────────┴───────────┴──────────────┴──────────────────────────┘

  🟢 LOW PROBABILITY (<40%) — 22 tickers predicted FLAT (omitted)

═══════════════════════════════════════════════════════════════
```

Also save to:
- `output/watchlist_YYYY-MM-DD.csv` (full data with all probabilities)
- `output/spike_history.csv` (append daily results for backtesting accuracy over time)

### 7. Scheduling (cron / launchd)

Include a setup script or instructions to schedule the detector to run at **7:30 AM ET** every weekday:

```bash
# macOS launchd example (preferred over cron on macOS)
# Create ~/Library/LaunchAgents/com.spikedetector.daily.plist

# OR simple cron:
# 30 7 * * 1-5 cd /path/to/project && python3 spike_detector.py >> output/log.txt 2>&1
```

### 8. Backtesting Module

Include a `backtest.py` script that:
1. Replays the past 6 months day-by-day
2. On each day, uses only data available before 9:30 AM
3. Generates the spike predictions
4. Compares with actual intraday moves
5. Reports: precision, recall, F1 for spike_up and spike_down classes
6. Generates a confusion matrix visualization

### 9. File Structure

```
Spike Detector/
├── spike_detector.py          # Main pipeline (daily execution)
├── backtest.py                # Historical backtesting module
├── config.py                  # UNIVERSE, thresholds, API keys, paths
├── features.py                # All feature engineering functions
├── model.py                   # XGBoost training, prediction, serialization
├── news.py                    # Finnhub API + FinBERT sentiment scoring
├── requirements.txt           # Pinned dependencies
├── README.md                  # Setup, usage, architecture docs
├── data/
│   ├── news_cache/            # Per-ticker news caches
│   ├── model.json             # Serialized trained XGBoost model
│   └── training_data.parquet  # Cached feature matrix
└── output/
    ├── watchlist_YYYY-MM-DD.csv
    ├── spike_history.csv
    └── backtest_results.png
```

### 10. Dependencies

```
xgboost>=2.0
pandas>=2.0
numpy>=1.24
yfinance>=0.2.30
transformers>=4.35
torch>=2.0
requests>=2.31
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.13
pyyaml>=6.0
```

### 11. Key Design Constraints

1. **No lookahead bias**: Features must use ONLY data available before 9:30 AM on the prediction day.
2. **Finnhub rate limits**: Cache aggressively. The free tier allows 60 calls/minute. With 30 tickers, you need to batch and throttle.
3. **FinBERT stability**: Use `TOKENIZERS_PARALLELISM=false` and chunk headlines in batches of 100 to prevent crashes (lesson learned from the price prediction model).
4. **Matplotlib backend**: Use `matplotlib.use("Agg")` for headless/CI execution.
5. **Model persistence**: Save the trained model to `data/model.json` after training. The daily 7:30 AM run should LOAD the pre-trained model, not retrain from scratch. Retrain weekly or on-demand via a separate command (`python spike_detector.py --retrain`).
6. **Earnings awareness**: Earnings days are the #1 spike catalyst. If `days_to_earnings == 0` or `is_earnings_day == True`, the model should heavily weight overnight sentiment.

### 12. Stretch Goals (implement if time allows)

- **Options data**: Unusual options volume (high call/put ratio) is a strong spike predictor. Integrate via `yfinance` options chain or a dedicated API.
- **Pre-market volume**: If the broker API supports it, pre-market volume spikes at 7:00 AM are a strong signal.
- **Discord/Telegram webhook**: Push the morning watchlist to a messaging channel.
- **Sector rotation detection**: Flag when money is rotating between sectors (e.g., tech selling off → energy rallying).

---

## Quick Start (for the AI assistant)

1. Create the project in `/Users/sstasbih/Desktop/Projects/Spike Detector/`
2. Start with `config.py` → `features.py` → `news.py` → `model.py` → `spike_detector.py`
3. Build the training pipeline first, backtest it, then add the daily scheduling
4. The Finnhub API key is stored in env var `FINNHUB_API_KEY`
5. Use the same Python: `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13`
