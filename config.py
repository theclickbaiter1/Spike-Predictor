"""
config.py — Central configuration for the Pre-Market Spike Detector.
"""

import os
from pathlib import Path

import numpy as np
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
TRADE_THRESHOLD = 0.55           # Default; overridden by data/tuned_threshold.json if present
WATCHLIST_THRESHOLD_LOW = 0.50   # Two-tier: alert-only watchlist floor
WATCHLIST_THRESHOLD_HIGH = 0.70  # Trade tier uses get_trade_threshold() (tuned)
TRADE_PROB_COLUMN = "p_spike_trade"
VIX_LOW_MAX = 15.0
VIX_MID_MAX = 25.0
DIRECTION_MARGIN_MIN = 0.18        # min max(p_up,p_down)/p_spike to trade direction
SECTOR_AGREEMENT_REQUIRED = True # coupling_alignment sign must match direction
REQUIRE_GAP_SENTIMENT_AGREEMENT = True  # gap direction must agree with overnight sentiment
SKIP_TRADE_VIX_ABOVE = 25.0      # no new entries when VIX at/above this level
SKIP_TRADE_NEAR_EARNINGS_DAYS = 1  # skip if days_to_earnings <= N or is_earnings_day
MIN_TRADE_THRESHOLD = 0.75       # floor — never trade below this even if tuned lower
MIN_TUNED_PRECISION = 0.40       # threshold tuning target precision on tune slice
TUNED_THRESHOLD_PATH = DATA_DIR / "tuned_threshold.json"
MAX_POSITIONS_PER_DAY = 3        # Max simultaneous positions
MAX_POSITION_PCT = 0.10          # Max 10% of account per position
MAX_DAILY_LOSS_PCT = 0.05        # Stop trading if account drops 5% in a day
TAKE_PROFIT_PCT = 0.04           # Symmetric bracket take-profit at +4%
STOP_LOSS_PCT = 0.04             # Symmetric bracket stop-loss at -4%
MAX_CONSECUTIVE_TICKER_DAYS = 3  # Don't trade same ticker >3 days in a row

# Limit-order entry: only fill if price moves against the signal by this much
# (i.e. a "dip" for LONG, a "rip" for SHORT) before we enter. Day-bound; if
# no fill by 4pm Alpaca cancels. Tradeoff: cleaner average entry price,
# but you skip days that gap-and-run without a pullback.
LIMIT_ENTRY_DIP_PCT = 0.005      # 0.5% favorable move required to fill
# "market" = bracket with market entry (fills at open/intraday quote).
# "limit_dip" = day limit entry LIMIT_ENTRY_DIP_PCT away (often expires unfilled).
ENTRY_ORDER_TYPE = "market"

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
S1_EVAL_METRIC = "aucpr"           # PR-AUC for rare-event early stopping
S1_POS_WEIGHT_MULTIPLIER = 1.0     # Multiply scale_pos_weight; sweep via ablation_study.py
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
    eval_metric="aucpr",
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

# ── XGBoost — Stage 3: Return Magnitude Regressor ───────────────────────────
XGB_PARAMS_RET = dict(
    objective="reg:squarederror",
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=3.0,
    tree_method="hist",
    eval_metric="rmse",
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

# ── Feature Column Order (50 + 6 catalyst + 9 stat-mech = 65 features) ─────
CATALYST_COLUMNS = [
    "premarket_gap_pct",
    "premarket_volume_ratio",
    "premarket_session_range_pct",
    "premarket_session_rel_volume",
    "gap_sentiment_agreement",
    "earnings_catalyst_score",
]

FEATURE_COLUMNS = [
    # Sentiment (7)
    "overnight_sentiment_mean", "overnight_sentiment_max",
    "overnight_sentiment_min", "overnight_news_count", "overnight_sentiment_std",
    "news_count_z_score", "news_spike",
    # Technical (10)
    "prev_close", "rsi_14", "ema_10", "realized_vol_20d", "avg_volume_10d",
    "prev_day_return", "prev_day_range", "gap_3d", "overnight_gap",
    "vol_z_score", "ret_1d", "ret_5d", "ret_10d", "momentum_slope_5d",
    "vol_regime_shift",
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
] + CATALYST_COLUMNS + STAT_MECH_COLUMNS

# ── EV Trading Overlay ────────────────────────────────────────────────────────
EV_MIN_EDGE = 0.0025               # minimum expected edge after costs (0.25%)
EV_COST_BUFFER = 0.0015            # slippage/fees buffer (0.15%)
EV_POSITION_MULTIPLIER = 3.0       # scales edge into position size fraction
EV_MAX_POSITION_PCT = 0.12          # cap single position by EV sizing

# ── Label Encoding ───────────────────────────────────────────────────────────
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}
LABEL_NAMES = ["spike_down", "flat", "spike_up"]


def get_trade_threshold(vix: float | None = None) -> float:
    """
    Load walk-forward tuned threshold if available, else TRADE_THRESHOLD.
    When vix is provided and tuned_threshold.json has regime buckets, pick by VIX.
    Never returns below MIN_TRADE_THRESHOLD.
    """
    import json

    def _floor(t: float) -> float:
        return max(float(t), MIN_TRADE_THRESHOLD)

    if TUNED_THRESHOLD_PATH.exists():
        try:
            with open(TUNED_THRESHOLD_PATH) as f:
                data = json.load(f)
            if vix is not None and not (isinstance(vix, float) and np.isnan(vix)):
                vix = float(vix)
                if vix >= VIX_MID_MAX and "vix_high" in data:
                    return _floor(data["vix_high"].get("threshold", data.get("default", TRADE_THRESHOLD)))
                if vix >= VIX_LOW_MAX and "vix_mid" in data:
                    return _floor(data["vix_mid"].get("threshold", data.get("default", TRADE_THRESHOLD)))
                if "vix_low" in data:
                    return _floor(data["vix_low"].get("threshold", data.get("default", TRADE_THRESHOLD)))
            return _floor(data.get("threshold", data.get("default", TRADE_THRESHOLD)))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return _floor(TRADE_THRESHOLD)
