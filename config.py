"""
config.py — Central configuration for the Pre-Market Spike Detector.
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
NEWS_CACHE_DIR = DATA_DIR / "news_cache"
OUTPUT_DIR = PROJECT_DIR / "output"
MODEL_PATH = DATA_DIR / "model.json"
TRAINING_DATA_PATH = DATA_DIR / "training_data.parquet"

for d in [DATA_DIR, NEWS_CACHE_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ──────────────────────────────────────────────────────────────────
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

# ── Ticker Universe ──────────────────────────────────────────────────────────
UNIVERSE = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
    "AMD", "MU", "AVGO", "QCOM", "MRVL", "ARM",
    "PLTR", "IONQ", "RGTI", "SMCI", "SNOW",
    "SMH", "QTUM", "XLK", "XLF", "XLE", "XBI",
    "GME", "AMC", "SOFI",
]

# ── Adaptive Spike Threshold ─────────────────────────────────────────────────
# A spike = intraday move > ADAPTIVE_MULTIPLIER × 20-day avg |intraday return|
# This replaces the old flat 3% threshold. Each ticker gets its own threshold.
ADAPTIVE_MULTIPLIER = 1.5
SPIKE_THRESHOLD = 0.03  # Kept as fallback / minimum floor

# ── Sector Mapping ───────────────────────────────────────────────────────────
SECTOR_MAP = {
    "NVDA": "XLK", "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK",
    "AMZN": "XLK", "META": "XLK", "TSLA": "XLK", "PLTR": "XLK",
    "SNOW": "XLK",
    "AMD": "SMH", "MU": "SMH", "AVGO": "SMH", "QCOM": "SMH",
    "MRVL": "SMH", "ARM": "SMH", "SMCI": "SMH",
    "IONQ": "QTUM", "RGTI": "QTUM",
    "SOFI": "XLF",
    "GME": "XLK", "AMC": "XLK",
    "SMH": "SMH", "QTUM": "QTUM", "XLK": "XLK",
    "XLF": "XLF", "XLE": "XLE", "XBI": "XBI",
}

# ── XGBoost — Stage 1: Spike Detector (binary) ──────────────────────────────
XGB_PARAMS_S1 = dict(
    objective="binary:logistic",
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    eval_metric="logloss",
    random_state=42,
    early_stopping_rounds=50,
)

# ── XGBoost — Stage 2: Direction Classifier (binary) ────────────────────────
XGB_PARAMS_S2 = dict(
    objective="binary:logistic",
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    eval_metric="logloss",
    random_state=42,
    early_stopping_rounds=30,
)

# ── Legacy 3-class params (kept for reference) ──────────────────────────────
XGB_PARAMS = dict(
    objective="multi:softprob", num_class=3, n_estimators=500,
    max_depth=6, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    tree_method="hist", eval_metric="mlogloss", random_state=42,
    early_stopping_rounds=50,
)

# ── Training Settings ────────────────────────────────────────────────────────
TRAINING_LOOKBACK_YEARS = 2
VAL_FRACTION = 0.20

# ── Finnhub Rate Limiting ────────────────────────────────────────────────────
FINNHUB_CALLS_PER_MIN = 60
FINNHUB_DELAY_SEC = 1.05

# ── FinBERT Settings ─────────────────────────────────────────────────────────
FINBERT_MODEL_NAME = "ProsusAI/finbert"
FINBERT_BATCH_SIZE = 100

# ── Feature Column Order (28 features) ───────────────────────────────────────
FEATURE_COLUMNS = [
    # Sentiment (6)
    "overnight_sentiment_mean", "overnight_sentiment_max",
    "overnight_sentiment_min", "overnight_news_count", "overnight_sentiment_std",
    "news_count_z_score",
    # Pre-market (2)
    "premarket_change", "premarket_volume_ratio",
    # Technical (10)
    "prev_close", "rsi_14", "ema_10", "realized_vol_20d", "avg_volume_10d",
    "prev_day_return", "prev_day_range", "gap_3d", "overnight_gap",
    "vol_z_score",
    # Macro (5)
    "vix", "treasury_10y", "sector_momentum_5d", "sp500_prev_return",
    "vix_change",
    # Calendar (5)
    "day_of_week", "is_monday", "is_friday", "days_to_earnings", "is_earnings_day",
]

# ── Label Encoding ───────────────────────────────────────────────────────────
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}
LABEL_NAMES = ["spike_down", "flat", "spike_up"]
