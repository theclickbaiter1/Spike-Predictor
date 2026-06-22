"""
config.py — Central configuration for the Pre-Market Spike Detector.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")
DATA_DIR = PROJECT_DIR / "data"
NEWS_CACHE_DIR = DATA_DIR / "news_cache"
EARNINGS_CACHE_DIR = DATA_DIR / "earnings_cache"
OUTPUT_DIR = PROJECT_DIR / "output"
MODEL_PATH = DATA_DIR / "model.json"
MODEL_BACKUP_DIR = DATA_DIR / "model_backups"
TRAINING_DATA_PATH = DATA_DIR / "training_data.parquet"
TRADE_LOG_PATH = OUTPUT_DIR / "trade_log.csv"

for d in [DATA_DIR, NEWS_CACHE_DIR, EARNINGS_CACHE_DIR, OUTPUT_DIR, MODEL_BACKUP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ──────────────────────────────────────────────────────────────────
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_API_KEY_DELAYED = os.environ.get("ALPACA_API_KEY_DELAYED", "")
ALPACA_SECRET_KEY_DELAYED = os.environ.get("ALPACA_SECRET_KEY_DELAYED", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── Alpaca Endpoints ─────────────────────────────────────────────────────────
ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL = "https://api.alpaca.markets"

# ── Trading Strategy ─────────────────────────────────────────────────────────
TRADE_THRESHOLD = 0.40           # Minimum P(spike) to enter a trade
MAX_POSITIONS_PER_DAY = 3        # Max simultaneous positions
MAX_POSITION_PCT = 0.10          # Max 10% of account per position
MAX_DAILY_LOSS_PCT = 0.05        # Stop trading if account drops 5% in a day
TAKE_PROFIT_PCT = 0.05           # Bracket order take-profit at +5%
STOP_LOSS_PCT = 0.03             # Bracket order stop-loss at -3%
MAX_CONSECUTIVE_TICKER_DAYS = 3  # Don't trade same ticker >3 days in a row

# Limit-order entry: only fill if price moves against the signal by this much
# (i.e. a "dip" for LONG, a "rip" for SHORT) before we enter. Day-bound; if
# no fill by 4pm Alpaca cancels. Tradeoff: cleaner average entry price,
# but you skip days that gap-and-run without a pullback.
LIMIT_ENTRY_DIP_PCT = 0.005      # 0.5% favorable move required to fill

# ── Ticker Universe (~58 tickers) ────────────────────────────────────────────
UNIVERSE = [
    # Original — Mega-cap Tech
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
    # Original — Semiconductors
    "AMD", "MU", "AVGO", "QCOM", "MRVL", "ARM",
    # Original — Growth / Quantum / Cloud
    "PLTR", "IONQ", "RGTI", "SMCI", "SNOW",
    # Original — Meme / Speculative
    "GME", "AMC", "SOFI",
    # NEW — Biotech / Healthcare
    "MRNA", "BNTX", "CRSP", "DXCM", "ISRG", "VRTX",
    # NEW — Energy
    "XOM", "CVX", "OXY", "FSLR", "ENPH",
    # NEW — Financials
    "JPM", "GS", "COIN", "HOOD", "XYZ",
    # NEW — Consumer / Retail
    "NFLX", "DIS", "NKE", "BABA", "JD",
    # NEW — Industrial / EV
    "RIVN", "LCID", "LI", "BA", "CAT",
    # NEW — More Semis / Tech
    "INTC", "ON", "ANET", "CRWD", "NET",
    # Sector ETFs
    "SMH", "QTUM", "XLK", "XLF", "XLE", "XBI", "XLY", "XLI",
]

# ── Adaptive Spike Threshold ─────────────────────────────────────────────────
# A spike = intraday move > ADAPTIVE_MULTIPLIER × 20-day avg |intraday return|
# This replaces the old flat 3% threshold. Each ticker gets its own threshold.
ADAPTIVE_MULTIPLIER = 1.5
SPIKE_THRESHOLD = 0.03  # Kept as fallback / minimum floor

# ── Sector Mapping ───────────────────────────────────────────────────────────
SECTOR_MAP = {
    # Mega-cap Tech
    "NVDA": "XLK", "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK",
    "AMZN": "XLK", "META": "XLK", "TSLA": "XLK", "PLTR": "XLK",
    "SNOW": "XLK",
    # Semiconductors
    "AMD": "SMH", "MU": "SMH", "AVGO": "SMH", "QCOM": "SMH",
    "MRVL": "SMH", "ARM": "SMH", "SMCI": "SMH",
    # Quantum
    "IONQ": "QTUM", "RGTI": "QTUM",
    # Meme / Speculative
    "SOFI": "XLF", "GME": "XLK", "AMC": "XLK",
    # NEW — Biotech / Healthcare
    "MRNA": "XBI", "BNTX": "XBI", "CRSP": "XBI",
    "DXCM": "XBI", "ISRG": "XBI", "VRTX": "XBI",
    # NEW — Energy
    "XOM": "XLE", "CVX": "XLE", "OXY": "XLE",
    "FSLR": "XLE", "ENPH": "XLE",
    # NEW — Financials
    "JPM": "XLF", "GS": "XLF", "COIN": "XLF",
    "HOOD": "XLF", "XYZ": "XLF",
    # NEW — Consumer / Retail
    "NFLX": "XLY", "DIS": "XLY", "NKE": "XLY",
    "BABA": "XLY", "JD": "XLY",
    # NEW — Industrial / EV
    "RIVN": "XLI", "LCID": "XLI", "LI": "XLI",
    "BA": "XLI", "CAT": "XLI",
    # NEW — More Semis / Tech
    "INTC": "SMH", "ON": "SMH",
    "ANET": "XLK", "CRWD": "XLK", "NET": "XLK",
    # Sector ETFs (self-map)
    "SMH": "SMH", "QTUM": "QTUM", "XLK": "XLK",
    "XLF": "XLF", "XLE": "XLE", "XBI": "XBI",
    "XLY": "XLY", "XLI": "XLI",
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
# Heavily regularized to combat 92% val → 50% live overfitting.
XGB_PARAMS_S2 = dict(
    objective="binary:logistic",
    n_estimators=300,
    max_depth=3,              # was 5 — shallower trees generalize better
    learning_rate=0.02,       # was 0.05 — slower learning, less memorization
    subsample=0.6,            # was 0.8 — more row dropout
    colsample_bytree=0.5,    # was 0.8 — force feature diversity per tree
    min_child_weight=20,      # was default 1 — require more samples per leaf
    gamma=1.0,                # was default 0 — prune splits that don't help enough
    reg_alpha=1.0,            # was 0.1 — stronger L1 (sparsity)
    reg_lambda=5.0,           # was 1.0 — stronger L2 (shrinkage)
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

# ── Statistical mechanics ─────────────────────────────────────────────────────
ISING_J_MAX = 0.3
ISING_LAMBDA_DEFAULT = 0.85
ISING_M_THRESHOLD = 0.15
ISING_COUPLING_WINDOW = 60
BETA_VIX_FLOOR = 10.0

STAT_MECH_COLUMNS = [
    "sector_magnetization",
    "sector_abs_magnetization",
    "local_field",
    "coupling_alignment",
    "cross_section_entropy",
    "sentiment_entropy",
    "inverse_temperature",
    "susceptibility_proxy",
    "criticality_proxy",
]

# ── Feature Column Order (45 + 9 stat-mech = 54 features) ─────────────────────
FEATURE_COLUMNS = [
    # Sentiment (7)
    "overnight_sentiment_mean", "overnight_sentiment_max",
    "overnight_sentiment_min", "overnight_news_count", "overnight_sentiment_std",
    "news_count_z_score", "news_spike",
    # Technical (10)
    "prev_close", "rsi_14", "ema_10", "realized_vol_20d", "avg_volume_10d",
    "prev_day_return", "prev_day_range", "gap_3d", "overnight_gap",
    "vol_z_score",
    # Macro (10 original + 8 lagged = 18)
    "vix", "treasury_10y", "sector_momentum_5d", "sp500_prev_return",
    "vix_change",
    "yield_curve_spread", "dxy_change", "crude_oil_change",
    "gold_change", "sp500_5d_return",
    # Lagged macro (multi-day momentum / regime)
    "vix_change_3d", "vix_change_5d", "vix_regime",
    "dxy_change_5d", "crude_oil_change_5d", "gold_change_5d",
    "treasury_10y_delta_5d", "sp500_return_3d",
    # Calendar (5)
    "day_of_week", "is_monday", "is_friday", "days_to_earnings", "is_earnings_day",
    # Earnings (5)
    "eps_surprise_last", "revenue_surprise_last", "earnings_streak",
    "post_earnings_drift_1d", "earnings_volatility",
] + STAT_MECH_COLUMNS

# ── Label Encoding ───────────────────────────────────────────────────────────
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}
LABEL_NAMES = ["spike_down", "flat", "spike_up"]
