# Pre-Market Spike Detector v2

A Python pipeline that predicts which stocks will experience significant intraday price spikes before market open. Uses a two-stage XGBoost model with FinBERT sentiment analysis, adaptive per-ticker thresholds, and 38 features spanning sentiment, technical, macro, earnings, and calendar signals.

**Last updated:** May 11, 2026

## How It Works

Every trading day before 8:00 AM ET, the pipeline:

1. Fetches overnight news headlines and scores them with FinBERT
2. Pulls pre-market price/volume data (Alpaca or Finnhub)
3. Computes 38 features from trailing OHLCV, macro indicators, and earnings history
4. Runs a two-stage XGBoost model:
   - **Stage 1** predicts whether a spike will occur (binary)
   - **Stage 2** predicts the direction — up or down (binary)
5. Outputs a ranked watchlist with spike probabilities and top signals

**Important:** This is not a price predictor. It classifies whether today's session will produce a large intraday move relative to the stock's own volatility, using only data available before 9:30 AM ET.

## Quick Start

### 1. Set Up Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set API Keys

```bash
export FINNHUB_API_KEY="your_key"
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_key"
```

### 3. Train the Model

```bash
python3 spike_detector.py --retrain
```

Downloads 2 years of OHLCV data for 60 tickers, fetches overnight news from Finnhub, scores headlines with FinBERT, and trains the two-stage XGBoost model with time-series validation.

### 4. Run Daily Predictions

```bash
python3 spike_detector.py
```

### 5. Validate Against Actuals

```bash
python3 validate_today.py
```

### 6. Backtest

```bash
python3 backtest.py               # Last 6 months
python3 backtest.py --months 12   # Last 12 months
```

## Two-Stage Model Architecture

### Stage 1: Spike Detection (Binary)

Predicts P(spike) — whether any significant move will occur. Trained with `scale_pos_weight` to handle the ~88% flat / 12% spike class imbalance. **Excludes `realized_vol_20d`** from its feature set to prevent the "always flag volatile stocks" bias — `vol_z_score` (relative volatility) captures regime shifts without the absolute-level shortcut.

### Stage 2: Direction Classification (Binary)

Predicts P(up | spike) — if a spike happens, which way. Trained on **spike + near-spike samples** (days where the return exceeded 50% of the spike threshold) rather than all samples. This gives the model ~10,000 training rows of meaningful directional moves without diluting signal with flat days that would bias it bearish.

### Inference

```
p_spike = Stage1.predict_proba(X)
p_up    = p_spike × Stage2.predict_proba(X)
p_down  = p_spike × (1 - Stage2.predict_proba(X))
p_flat  = 1 - p_spike
```

### Adaptive Spike Threshold

Each ticker gets its own threshold based on its historical volatility:

```
threshold = max(20-day avg |intraday return| × 1.5, 3%)
```

NVDA's threshold might be 5%, while AAPL's might be 3%. This prevents volatile stocks from always being labeled as "spiking."

## Features (38 total)

### Sentiment (6)
Overnight news headlines scored with [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert). Mean, max, min, std of sentiment scores, plus news count and a z-scored news volume (abnormal attention signal).

### Pre-Market (2)
Pre-market price change and volume ratio vs 10-day average. Live data from Alpaca with Finnhub fallback; historical data from yfinance extended hours.

### Technical (10)
RSI-14, EMA-10, 20-day realized volatility, volatility z-score, average volume, previous day return/range, 3-day momentum, and overnight gap. All shifted to avoid lookahead.

### Macro (10)
VIX level and daily change, 10Y Treasury yield, yield curve spread, S&P 500 1-day and 5-day returns, sector ETF 5-day momentum, USD index change, crude oil change, and gold change.

### Calendar (5)
Day of week, Monday/Friday flags, days to next earnings, and earnings day flag.

### Earnings (5)
Most recent EPS surprise %, revenue surprise %, consecutive beat/miss streak, post-earnings 1-day drift, and average absolute return on earnings days (last 4 quarters).

## Ticker Universe (60 stocks)

| Sector | Tickers |
|--------|---------|
| Mega-Cap Tech | NVDA, AAPL, MSFT, GOOGL, AMZN, META, TSLA |
| Semiconductors | AMD, MU, AVGO, QCOM, MRVL, ARM, INTC, ON |
| Growth / AI / Cloud | PLTR, IONQ, RGTI, SMCI, SNOW, ANET, CRWD, NET |
| Biotech / Healthcare | MRNA, BNTX, CRSP, DXCM, ISRG, VRTX |
| Energy | XOM, CVX, OXY, FSLR, ENPH |
| Financials | JPM, GS, COIN, HOOD, XYZ, SOFI |
| Consumer / Retail | NFLX, DIS, NKE, BABA, JD |
| Industrial / EV | RIVN, LCID, LI, BA, CAT |
| Meme / Speculative | GME, AMC |
| Sector ETFs | SMH, QTUM, XLK, XLF, XLE, XBI, XLY, XLI |

## Automation

### GitHub Actions (included)

Two workflows in `.github/workflows/`:

- **daily_predict.yml** — Runs predictions Mon-Fri at ~7 AM ET, sends results via Telegram
- **retrain.yml** — Retrains the model weekly on Sundays (or manually via workflow_dispatch)

Required GitHub Secrets: `FINNHUB_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`

### Telegram Notifications

```bash
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"
python3 notify.py
```

## Architecture

| Module | Purpose |
|--------|---------|
| `config.py` | Ticker universe, feature columns, XGBoost params, sector/ETF mappings |
| `features.py` | Feature engineering: technical, macro, sentiment, earnings, calendar |
| `model.py` | Two-stage XGBoost: spike detection + direction classification |
| `news.py` | Finnhub API client + FinBERT sentiment scoring with disk caching |
| `premarket.py` | Pre-market data from Alpaca (live) and yfinance (historical) |
| `spike_detector.py` | Main pipeline: `--retrain` and predict modes |
| `notify.py` | Telegram bot notification of daily watchlist |
| `validate_today.py` | Out-of-sample validation against actual intraday results |
| `backtest.py` | Historical backtesting with confusion matrices |

## Recent Changes (May 2026)

### Model Improvements

- **Dropped `realized_vol_20d` from Stage 1.** It accounted for 10.3% of feature importance — 2x the next feature — teaching the model "volatile stock = spike" rather than detecting actual spike conditions. `vol_z_score` remains to capture relative volatility shifts.

- **Stage 2 now trains on spike + near-spike samples** instead of all samples. The previous approach (training on all ~29,000 rows) introduced a bearish bias from the 88% flat days. The new approach uses ~10,000 rows of days with meaningful directional moves, improving direction accuracy from 65% to 92.6% on the validation set.

- **Added `scale_pos_weight`** for Stage 1 class imbalance instead of manual sample weights. XGBoost handles this natively and more efficiently.

### New Features (+13)

- **Earnings (5):** EPS surprise, revenue surprise, earnings streak, post-earnings drift, earnings volatility
- **Macro (5):** Yield curve spread, USD index change, crude oil change, gold change, S&P 500 5-day return
- **Technical (2):** Volatility z-score, news count z-score
- **Macro (1):** VIX daily change

### Expanded Universe

27 → 60 tickers across biotech, energy, financials, consumer, industrial, and more semiconductors. Training data roughly doubled from ~13,500 to ~30,000 rows.

### Why These Changes Should Help Going Forward

The core problem with v1 was that it flagged every volatile stock every day and couldn't tell up from down. Removing the raw volatility shortcut forces Stage 1 to rely on actual predictive signals (overnight gaps, earnings timing, macro shifts). Training Stage 2 on directional data rather than mostly-flat data means it learns what actually distinguishes an up-spike from a down-spike. More tickers and more features give the model a broader base to generalize from rather than overfitting to the behavior of 27 stocks.

## Design Decisions

- **No lookahead bias** — every feature uses only data available before 9:30 AM ET
- **Time-series validation** — most recent 20% of data as validation (no shuffle)
- **Adaptive thresholds** — per-ticker spike definition based on historical volatility
- **Aggressive caching** — Finnhub responses, earnings data, and OHLCV cached to disk
- **Graceful degradation** — missing pre-market or earnings data defaults to 0, model still runs
