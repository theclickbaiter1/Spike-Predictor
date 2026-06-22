# Pre-Market Spike Detector v2

A fully automated Python pipeline that predicts intraday stock spikes before market open and executes paper trades through Alpaca. Built around a two-stage XGBoost backbone with a statistical-mechanics inference layer (entropy, sector magnetization, Boltzmann calibration, Ising overlay), FinBERT sentiment, adaptive per-ticker thresholds, and 54 features spanning sentiment, technical, macro, earnings, calendar, and stat-mech signals. The full daily loop — predict, alert, trade, validate, retrain — runs on its own with no human in the loop.

**Last updated:** May 26, 2026 (added A/B timing experiment via dual paper accounts; fixed Alpaca data-API URL bug; added Telegram /start subscriber broadcast)

## What problem this solves

The model is **not a price predictor**. It's a binary spike classifier — given the state of the world at 9:15 AM ET on any given trading day, will this stock produce a significant intraday move (up or down) relative to its own recent volatility? "Significant" is defined adaptively per ticker (NVDA's threshold is higher than AAPL's because NVDA is intrinsically more volatile). The model uses **only data available before 9:30 AM ET** — no peeking at intraday news or prices.

The end goal: identify the 2–3 tickers each morning most likely to make a 3%+ intraday move, predict the direction, and trade them with risk-limited bracket orders.

## Daily flow at a glance

| Time (ET) | What happens | Telegram you'll receive |
|---|---|---|
| **9:15 AM** | Predict → save watchlist artifact → place trades on **Account OPEN** | Watchlist + `[OPEN]` trade confirmations |
| **10:00 AM** | Load morning watchlist → place same signals on **Account DELAYED** | `[DELAYED]` trade confirmations |
| **4:30 PM** | Pull actual intraday returns → compute metrics → update trend chart | Daily validation summary + 10d rolling avg |
| **Sunday 7:00 AM** | Retrain model on the past 2 years of data | (silent) |

Everything is triggered by a single Cloudflare Worker cron, which dispatches GitHub Actions workflows. Each workflow runs on a fresh Ubuntu runner, downloads the trained model artifact, executes its task, and uploads its outputs as artifacts.

## System architecture

```mermaid
flowchart TB
    subgraph CFW["Cloudflare Worker (DST-aware, holiday-aware)"]
        C1[9:15 AM ET cron → daily_trade.yml]
        C2[10:00 AM ET cron → daily_trade_delayed.yml]
        C3[4:30 PM ET cron → daily_validate.yml]
        C4[Sunday 7:00 AM ET cron → retrain.yml (bi-weekly)]
    end

    subgraph Sources["External Data Sources"]
        FH[Finnhub<br/>company news]
        YF[yfinance<br/>OHLCV + macro + earnings]
        ALP_DATA[Alpaca Market Data<br/>data.alpaca.markets<br/>latest trade prices]
        ALP_OPEN[Alpaca Paper Account OPEN<br/>paper-api.alpaca.markets]
        ALP_DEL[Alpaca Paper Account DELAYED<br/>paper-api.alpaca.markets]
    end

    subgraph Pipeline["GitHub Actions"]
        GTR[daily_trade.yml<br/>predict + Telegram + trade OPEN]
        GDEL[daily_trade_delayed.yml<br/>load watchlist + trade DELAYED]
        GVAL[daily_validate.yml<br/>actuals + rolling metrics]
        GRET[retrain.yml<br/>bi-weekly model rebuild]
    end

    subgraph Model["Two-Stage XGBoost — 45 features"]
        FEAT[Feature Engineering<br/>sentiment ×7, technical ×10<br/>macro ×18, calendar ×5, earnings ×5]
        S1[Stage 1: P_spike<br/>binary, scale_pos_weight]
        S2[Stage 2: P_up given spike<br/>regularized, near-spike training]
    end

    subgraph State["Persistent State (GHA Cache + Artifacts)"]
        ART_MODEL[(spike-model<br/>bi-weekly retrain artifact)]
        ART_WL[(watchlist-YYYY-MM-DD<br/>9:15 AM watchlist)]
        ART_VAL[(validation-state<br/>history, metrics, trend chart)]
        ART_SUBS[(subscribers.json<br/>Telegram /start chat_ids)]
        ART_LOG_O[(trade_log_open.csv)]
        ART_LOG_D[(trade_log_delayed.csv)]
    end

    subgraph TG["Telegram Broadcast"]
        TG_OUT[Owner + all /start subscribers]
    end

    C1 --> GTR
    C2 --> GDEL
    C3 --> GVAL
    C4 --> GRET

    GRET --> ART_MODEL
    ART_MODEL --> GTR
    ART_MODEL -.->|fallback path<br/>not needed daily| GDEL

    GTR --> FH
    GTR --> YF
    FH --> FEAT
    YF --> FEAT
    FEAT --> S1 --> S2
    S2 --> ART_WL
    S2 --> TG_OUT

    ART_WL --> GDEL
    ART_WL --> GVAL

    GTR --> ALP_DATA
    GTR --> ALP_OPEN
    GDEL --> ALP_DATA
    GDEL --> ALP_DEL

    GTR --> ART_LOG_O
    GDEL --> ART_LOG_D
    GTR --> TG_OUT
    GDEL --> TG_OUT

    GVAL --> YF
    GVAL --> ART_VAL
    GVAL --> TG_OUT

    TG_OUT --> ART_SUBS
    ART_SUBS --> TG_OUT
```

## Repo layout

```
.
├── config.py                  # Universe, feature columns, XGBoost params, risk limits, API URLs
├── features.py                # Feature engineering (technical, macro, sentiment, earnings, calendar)
├── model.py                   # Two-stage XGBoost + Boltzmann calibrator + Ising overlay
├── stat_mech_features.py      # Entropy, magnetization, β(VIX) feature engineering
├── stat_mech/
│   ├── calibrator.py          # BoltzmannCalibrator (3-state Gibbs / MaxEnt)
│   └── ising.py               # Sector Ising mean-field overlay + λ blend
├── news.py                    # Finnhub client + FinBERT scorer (disk-cached)
├── predict/
│   ├── spike_detector.py      # Main pipeline — predict + --retrain modes
│   ├── trade.py               # Alpaca bracket order execution
│   │                          #   --label {open|delayed} → per-account trade log
│   │                          #   --watchlist <csv>     → skip regeneration, use saved snapshot
│   ├── notify.py              # Telegram alerts + /start subscriber polling
│   │                          #   --trades --label open|delayed
│   │                          #   --validate
│   │                          #   --poll-only
│   └── run_daily.sh           # Wrapper used by GitHub Actions
├── backtest/
│   ├── backtest.py            # Historical model performance
│   ├── backtest_strategy.py   # Strategy backtest with TP/SL
│   ├── daily_validation.py    # Lightweight daily pred-vs-actual + rolling metrics
│   ├── validate_today.py      # Full retrain-and-validate for a single date (heavy)
│   ├── validate_week.py       # Full retrain-and-validate for a date range (heavy)
│   └── compare_calibrators.py # Raw vs calibrated NLL, Brier, reliability diagram
├── data/                      # gitignored — cached news, earnings, OHLCV, model, subscribers.json
├── output/                    # gitignored — daily watchlists, per-account trade logs, validation_state/
├── pyrefly.toml               # Points Pyrefly at venv/bin/python (IDE diagnostics)
├── .vscode/settings.json      # Same for Pylance + Python extension
└── .github/workflows/
    ├── daily_predict.yml          # Prediction-only manual run (workflow_dispatch only)
    ├── daily_trade.yml            # 9:15 AM ET — predict + trade OPEN account
    ├── daily_trade_delayed.yml    # 10:00 AM ET — trade morning watchlist on DELAYED account
    ├── daily_validate.yml         # 4:30 PM ET — EOD close + pred-vs-actual rolling metrics
    ├── retrain.yml                # Sunday 7 AM ET — bi-weekly retrain (even ISO weeks)
    └── weekly_validate.yml        # Saturday 8 AM ET — heavy OOS weekly validation
```

The Cloudflare Worker source lives outside this repo at `~/spike-scheduler/` (separate concern, separate deploy). It's just `worker.js` + `wrangler.toml` — see [Cloudflare Worker](#cloudflare-worker) below.

## Quick start

### 1. Set up environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set API keys in `.env`

```bash
# Finnhub (overnight company news)
FINNHUB_API_KEY=...

# Alpaca — Account OPEN (trades at 9:30 open)
ALPACA_API_KEY=PK...
ALPACA_SECRET_KEY=...

# Alpaca — Account DELAYED (trades at 10:00 — A/B experiment)
ALPACA_API_KEY_DELAYED=PK...
ALPACA_SECRET_KEY_DELAYED=...

# Telegram bot — owner is always notified; /start subscribers also added at runtime
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

For automation, these same names exist as **GitHub Actions secrets** at `https://github.com/<owner>/<repo>/settings/secrets/actions`. Locally they live in `.env` (which is gitignored).

### 3. Train the model (first time)

```bash
python predict/spike_detector.py --retrain
```

Downloads 2 years of OHLCV data for ~60 tickers, fetches overnight news from Finnhub, scores headlines with FinBERT, and trains the two-stage XGBoost model with time-series validation. ~15-20 min on first run, faster after caching (`data/news_cache/`, `data/earnings_cache/`).

### 4. Run predictions / trades locally

```bash
# Just predict (writes output/watchlist_YYYY-MM-DD.csv)
python predict/spike_detector.py

# Dry-run trade simulation (no actual orders)
python predict/trade.py --dry-run

# Paper trade (default — uses ALPACA_API_KEY)
python predict/trade.py --paper --label open

# Paper trade the delayed account from a saved watchlist
python predict/trade.py --paper --label delayed \
  --watchlist output/watchlist_2026-05-26.csv

# Live trading (requires explicit flag + interactive 'yes' confirmation)
python predict/trade.py --live
```

### 5. Validate today's predictions

```bash
python backtest/daily_validation.py            # validate today
python backtest/daily_validation.py --date 2026-05-26   # backfill a past date
```

This is the lightweight one — reads `output/watchlist_<date>.csv`, fetches actuals via yfinance, appends to `output/validation_state/{history,metrics}.csv`, regenerates `trend.png`. The "heavy" alternative (`backtest/validate_today.py`) retrains the model from scratch first and is meant for ad-hoc deeper investigation, not daily use.

## Two-stage model architecture

### Stage 1: Spike detection (binary)

Predicts `P(spike)` — whether any significant intraday move will occur. Trained with `scale_pos_weight` to handle the ~88% flat / 12% spike class imbalance. **Excludes `realized_vol_20d`** from its feature set to prevent the "always flag volatile stocks" bias — `vol_z_score` (relative volatility) captures regime shifts without the absolute-level shortcut.

### Stage 2: Direction classification (binary)

Predicts `P(up | spike)` — if a spike happens, which way. Trained on **spike + near-spike samples** (days where the return exceeded 50% of the spike threshold) rather than all samples. This gives the model ~10,000 training rows of meaningful directional moves without diluting signal with flat days that would bias it bearish. Heavily regularized (`max_depth=3`, `min_child_weight=20`, `gamma=1.0`, `reg_alpha=1.0`, `reg_lambda=5.0`) to combat the 92% val → 50% live overfitting seen in v1.

### Inference

```
p_spike_raw = Stage1.predict_proba(X)
p_up_raw    = p_spike_raw * Stage2.predict_proba(X)
```

Raw factorized probabilities are then passed through the **stat-mech inference layer** (see below) to produce calibrated `p_spike`, `p_up`, `p_down`, `p_flat`. Trade logs record both `p_spike_raw` and `p_spike` for A/B calibration analysis.

### Statistical mechanics hybrid layer

XGBoost remains the backbone learner. Three stat-mech components sit on top:

1. **Feature engineering** (`stat_mech_features.py`) — sector magnetization, Shannon entropy of overnight gaps and headline scores, inverse temperature β(VIX), susceptibility/criticality proxies.
2. **Boltzmann calibrator** (`stat_mech/calibrator.py`) — fits a 3-state Gibbs distribution on validation data, mapping XGBoost logits + local field/coupling features to a thermodynamically consistent joint law over {up, flat, down}.
3. **Sector Ising overlay** (`stat_mech/ising.py`) — mean-field solve on same-sector couplings; blends with calibrated XGBoost via tunable λ (default 0.85, favoring XGBoost).

Retrain acceptance gate rejects deploy if val F1 drops >0.03 **or** calibrated NLL worsens by >0.05 vs the backup model.

Offline calibration benchmarks: `python backtest/compare_calibrators.py` (NLL, Brier, reliability diagram → `output/reliability_diagram.png`).

### Adaptive spike threshold

Each ticker gets its own threshold based on its historical volatility:

```
threshold = max(20-day avg |intraday return| * 1.5, 3%)
```

NVDA's threshold might be 5%, while AAPL's might be 3%. This prevents volatile stocks from being labeled as "spiking" every day just because they always move.

## Features (54 total)

### Sentiment (7)
Overnight news headlines scored with [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert). Mean, max, min, std of sentiment scores, raw news count, z-scored news volume (abnormal-attention signal), and a binary news-spike flag (≥3× normal overnight volume).

### Technical (10)
Previous close, RSI-14, EMA-10, 20-day realized volatility, volatility z-score, 10-day average volume, previous day return/range, 3-day momentum, overnight gap. All `.shift(1)` to avoid lookahead.

### Macro (18)
VIX level + daily change, 10Y treasury yield, yield curve spread (10Y minus 3M), SP500 1d/5d returns, sector ETF 5-day momentum, USD index change, crude oil change, gold change, **plus 8 lagged 3-day/5-day variants** so the model can see sustained regime shifts vs single-day blips.

### Calendar (5)
Day of week, Monday/Friday flags, days to next earnings, earnings day flag.

### Earnings (5)
Most recent EPS surprise %, revenue surprise %, consecutive beat/miss streak, post-earnings 1-day drift, average absolute return on earnings days (last 4 quarters).

### Statistical mechanics (9)
`sector_magnetization`, `sector_abs_magnetization`, `local_field` (macro + sentiment external field), `coupling_alignment` (σᵢ · m_sector), `cross_section_entropy` (market-wide disorder), `sentiment_entropy` (FinBERT score histogram entropy), `inverse_temperature` β(VIX) z-scored, `susceptibility_proxy` (20d std of sector magnetization), `criticality_proxy` (susceptibility × cross-section entropy). Stage 1 excludes `inverse_temperature` to avoid double-counting with the Boltzmann layer.

### Planned (not yet implemented)
- Live pre-market price/volume features (requires paid Alpaca data tier)
- Options flow / unusual options activity (requires paid feed)

## Ticker universe (60 stocks)

| Sector | Tickers |
|--------|---------|
| Mega-Cap Tech | NVDA, AAPL, MSFT, GOOGL, AMZN, META, TSLA |
| Semiconductors | AMD, MU, AVGO, QCOM, MRVL, ARM, INTC, ON, SMCI |
| Growth / AI / Cloud | PLTR, IONQ, RGTI, SNOW, ANET, CRWD, NET |
| Biotech / Healthcare | MRNA, BNTX, CRSP, DXCM, ISRG, VRTX |
| Energy | XOM, CVX, OXY, FSLR, ENPH |
| Financials | JPM, GS, COIN, HOOD, XYZ, SOFI |
| Consumer / Retail | NFLX, DIS, NKE, BABA, JD |
| Industrial / EV | RIVN, LCID, LI, BA, CAT |
| Meme / Speculative | GME, AMC |
| Sector ETFs | SMH, QTUM, XLK, XLF, XLE, XBI, XLY, XLI |

## Trading strategy

Configured in [config.py](config.py):

| Setting | Value | Purpose |
|---|---|---|
| `TRADE_THRESHOLD` | 0.40 | Minimum `P(spike)` to enter a position |
| `MAX_POSITIONS_PER_DAY` | 3 | Cap on simultaneous positions |
| `MAX_POSITION_PCT` | 10% | Per-position cap as % of equity |
| `MAX_DAILY_LOSS_PCT` | 5% | Circuit breaker — stop trading if account drops 5% intraday |
| `TAKE_PROFIT_PCT` | +5% | Bracket-order take-profit |
| `STOP_LOSS_PCT` | -3% | Bracket-order stop-loss |
| `MAX_CONSECUTIVE_TICKER_DAYS` | 3 | Don't trade same ticker more than 3 days in a row |

Alpaca bracket orders use take-profit (+5%) and stop-loss (-3%) for intraday exits. At 4:30 PM ET the `daily_validate.yml` workflow **liquidates any remaining open positions** on both paper accounts (`--close-all`) before running validation — so unfilled bracket legs that expired at 4 PM do not leave positions open overnight.

## A/B timing experiment: dual paper accounts

This is the headline operational change as of May 2026. We run the same signals through two separate Alpaca paper accounts at two different times of day to compare execution timing without confounding from day-to-day market drift.

| | Account OPEN | Account DELAYED |
|---|---|---|
| **Trade fire time** | 9:15 AM ET (orders queue, fill at 9:30 opening auction) | 10:00 AM ET (fill at intraday price) |
| **Workflow** | [daily_trade.yml](.github/workflows/daily_trade.yml) | [daily_trade_delayed.yml](.github/workflows/daily_trade_delayed.yml) |
| **Signal source** | Re-generated from morning data | Loaded from morning's saved watchlist artifact |
| **Alpaca keys** | `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | `ALPACA_API_KEY_DELAYED` / `ALPACA_SECRET_KEY_DELAYED` |
| **Trade log** | `output/trade_log_open.csv` | `output/trade_log_delayed.csv` |
| **Telegram tag** | `*Spike Trader [OPEN] — DATE*` | `*Spike Trader [DELAYED] — DATE*` |

**Hypothesis being tested:** The first 20–30 minutes of trading is dominated by retail order flow ("amateur hour"). For our long-biased signals on liquid mega-cap tech, waiting until ~10 AM may produce better fills on average. The DELAYED account tests this.

**Why dual accounts beats sequential testing:** Running 9:30 fills for 3 weeks then switching to 10:00 fills for 3 weeks confounds market-condition differences across the windows. Running them in parallel — same signals, same day, only timing differs — eliminates that.

**Both accounts start at $100k.** Compare equity after ~3–4 weeks of trading.

**After resetting a paper account on the Alpaca dashboard:** API keys are invalidated. Generate new keys (Paper Trading → API Keys), update `.env` locally, and update GitHub secrets `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` (and `_DELAYED` variants for the second account). Verify with:

```bash
python predict/trade.py --paper --label open --status
python predict/trade.py --paper --label open --close-all   # flatten any leftover positions
```

**Critical detail:** the DELAYED workflow does **not** regenerate predictions at 10:00 AM. It loads the 9:15 watchlist artifact (`watchlist-YYYY-MM-DD`) and trades that. Regenerating at 10:00 would leak intraday news into the sentiment features (Finnhub returns articles published since the last call, with no time-of-day filter) — predictions would be artificially "accurate" because they'd already know what happened in the morning. The saved-artifact path is the only way to keep the comparison fair.

## Automation

### Cloudflare Worker

A small Cloudflare Worker fires cron triggers, checks NY local time (DST-aware), checks the date against a hardcoded NYSE holiday set, and dispatches the appropriate GitHub Actions workflow via the API. Worker source: `~/spike-scheduler/` (see `~/spike-scheduler/README.md` for deploy).

If `wrangler deploy` fails with **Authentication error [code: 10000]**, run `wrangler login` in your terminal (OAuth session expired). `account_id` is pinned in `wrangler.toml`.

**Cron triggers (consolidated; one line per slot covers both EDT and EST):**

```toml
[triggers]
crons = [
  "15 13,14 * * 1-5",   # 9:15 AM ET Mon-Fri → daily_trade.yml
  "0 14,15 * * 1-5",    # 10:00 AM ET Mon-Fri → daily_trade_delayed.yml
  "30 20,21 * * 1-5",   # 4:30 PM ET Mon-Fri → daily_validate.yml (+ EOD close)
  "0 12,13 * * 6",      # 8:00 AM ET Saturday → weekly_validate.yml
  "0 11,12 * * 7",      # 7:00 AM ET Sunday  → retrain.yml (bi-weekly, even ISO weeks)
]
```

4 cron lines total — under the Cloudflare free-tier limit of 5. Each line fires at both the EDT (winter) and EST (summer) UTC offsets; the worker's runtime guard at `resolveWorkflow(ny)` only dispatches when NY local time matches the intended slot, so DST switches require zero manual intervention.

**Holiday handling:** `MARKET_HOLIDAYS` in `worker.js` is a hardcoded Set of YYYY-MM-DD strings for full-closure NYSE holidays through 2027. Weekday slots skip dispatch on these days. Sunday retrain runs regardless (uses historical data only). The list needs annual update against [nyse.com/markets/hours-calendars](https://www.nyse.com/markets/hours-calendars).

**Manual trigger (bypasses both guards):**
```bash
curl 'https://spike-scheduler.<subdomain>.workers.dev/?workflow=daily_trade.yml'
```

### GitHub Actions workflows

| Workflow | Cron-triggered at | What it does | Uploads as artifact |
|---|---|---|---|
| `daily_trade.yml` | Mon–Fri 9:15 AM ET | Predict + Telegram watchlist + trade OPEN + Telegram [OPEN] confirmations | `watchlist-YYYY-MM-DD` (date-stable name for downstream pickup) + `daily-output-<run_id>` (full output dir) |
| `daily_trade_delayed.yml` | Mon–Fri 10:00 AM ET | Download morning watchlist + trade DELAYED + Telegram [DELAYED] confirmations | `delayed-trade-YYYY-MM-DD` (trade log) |
| `daily_validate.yml` | Mon–Fri 4:30 PM ET | Download morning watchlist + fetch actuals via yfinance + update rolling metrics + Telegram validation summary | `validation-YYYY-MM-DD` (rolling state) |
| `retrain.yml` | Sunday 7:00 AM ET (bi-weekly) | Build 2-year training set + retrain two-stage XGBoost + upload model | `spike-model` (consumed by daily workflows) |
| `weekly_validate.yml` | Saturday 8:00 AM ET | Heavy OOS `validate_week.py` retrain + predict prior week | `weekly-validation-*` artifacts |
| `daily_predict.yml` | (manual only) | Prediction-only run for debugging — no trades | `watchlist-<run_id>` |

**Key cross-workflow dependency:** daily workflows download the `spike-model` artifact from the most recent successful `retrain.yml` run. The download step uses `dawidd6/action-download-artifact@v3` with `workflow: retrain.yml` and `workflow_conclusion: success` — without `workflow:` it would default to searching the current workflow's artifacts and never find anything (this was a real 10-day-silent bug fixed in commit `3fb60fb`).

### State persistence (GitHub Actions cache + artifacts)

GitHub Actions runners are stateless. State that needs to live across runs uses one of two mechanisms:

| State | Mechanism | Cache key | Lives in |
|---|---|---|---|
| Finnhub news cache | `actions/cache@v4` | `spike-data-cache-<run_id>` w/ prefix restore | `data/news_cache/` |
| Earnings cache | same | same | `data/earnings_cache/` |
| Telegram subscriber list | `actions/cache@v4` | `telegram-subscribers-<run_id>` w/ prefix restore | `data/subscribers.json` |
| Validation rolling state | `actions/cache@v4` | `validation-state-<run_id>` w/ prefix restore | `output/validation_state/` |
| Delayed trade log | `actions/cache@v4` | `trade-log-delayed-<run_id>` w/ prefix restore | `output/trade_log_delayed.csv` |
| Trained model | upload-artifact | `spike-model` (latest successful retrain.yml) | `data/model_s1.json`, `model_s2.json`, `model_meta.json` |
| Daily watchlist | upload-artifact | `watchlist-YYYY-MM-DD` (date-stable) | `output/watchlist_YYYY-MM-DD.csv` |

The cache pattern is: key is unique per run (`-<run_id>` suffix), restore-keys is the prefix only. So every run restores the most recently saved snapshot, modifies it, then saves a fresh copy under a new key. No race conditions because the daily workflows never run concurrently.

### Telegram broadcast (multi-recipient)

Every send goes to the **owner** (chat_id from `TELEGRAM_CHAT_ID` env) plus any chat_ids that have sent `/start` to the bot. The `/start` listener is implemented as polling: each `notify.py` invocation calls Telegram's `getUpdates` first, registers new subscribers (and removes `/stop` senders), persists the list to `data/subscribers.json`, then broadcasts.

| Daily message | When | Sender |
|---|---|---|
| Watchlist | 9:15 AM ET (after prediction) | `notify.py` |
| `[OPEN]` trade confirmation | 9:15 AM ET (after trade.py) | `notify.py --trades --label open` |
| `[DELAYED]` trade confirmation | 10:00 AM ET (after delayed trade.py) | `notify.py --trades --label delayed` |
| Validation summary | 4:30 PM ET | `notify.py --validate` |

**Caveat:** Telegram only retains undelivered `/start` messages for ~24 hours. If a friend sends `/start` Friday afternoon and the next workflow run is Tuesday morning (over a long weekend), the message may expire before being polled. Workaround: manually run `python predict/notify.py --poll-only` locally, or have the friend re-send `/start` on a trading morning.

### Validation feedback loop

`backtest/daily_validation.py` runs every weekday at 4:30 PM ET. For each ticker in the morning watchlist:

1. Fetch today's actual OHLC bar via yfinance
2. Classify the actual intraday return as `SPIKE UP` / `SPIKE DOWN` / `FLAT`
3. Compare against predicted class (derived from `p_spike` vs `TRADE_THRESHOLD` and `p_up` vs `p_down`)
4. Append to rolling per-day state

State files in `output/validation_state/`:
- `history.csv` — per-ticker per-day prediction vs actual
- `metrics.csv` — per-day aggregate metrics (accuracy, spike precision, spike recall, direction accuracy on caught spikes, signal P&L proxy)
- `trend.png` — 4-panel chart auto-regenerated each run, showing how each metric evolves over time with 10-day rolling averages

Telegram summary surfaces today's numbers plus the 10-day rolling average + cumulative signal P&L. The signal P&L proxy assumes you took every `SPIKE UP` long and every `SPIKE DOWN` short and held to the close — it's not real P&L (no slippage, no TP/SL, no position sizing) but it's directionally informative.

## Recent changes

### May 2026 — Dual-account A/B + Alpaca bug fix

- **Fixed Alpaca data API URL.** Market data endpoints live on `data.alpaca.markets`, not `paper-api.alpaca.markets`. `get_latest_price()` was hitting the wrong host, returning 404, and silently skipping every order. This blocked all paper trades for the first ~10 days the workflow was running. Commit `5d5d993`.
- **Split execution into two workflows for A/B timing test.** OPEN account trades at 9:30 auction, DELAYED account trades at 10:00 from the same morning watchlist. Both start at $100k. Same commit.
- **Multi-recipient Telegram via `/start` polling.** Friends can subscribe by sending `/start` to the bot; chat_ids persist in `data/subscribers.json` across runs. Commit `13c734f`.
- **Holiday-aware Cloudflare Worker.** Hardcoded NYSE holiday set through 2027 — weekday slots skip on closures (e.g., Memorial Day 2026-05-25 didn't fire). Commit `423faf1`.
- **DST-aware scheduling.** Worker fires at both EDT and EST UTC offsets per slot; runtime guard dispatches only when NY local time matches.
- **Daily post-market validation workflow.** Lightweight pred-vs-actual that doesn't retrain — runs in ~2 min vs ~30+ min for the heavy `validate_today.py`. Commit `6a01d63`.
- **Fixed `download-artifact` defaulting to current workflow.** Daily workflows now explicitly point at `retrain.yml` to find `spike-model`. This had silently broken every scheduled daily_predict run for 10+ days. Commit `3fb60fb`.

### Model improvements (May 11–22, 2026)

- **Dropped `realized_vol_20d` from Stage 1.** It accounted for 10.3% of feature importance — 2× the next feature — teaching the model "volatile stock = spike" rather than detecting actual spike conditions. `vol_z_score` remains to capture relative volatility shifts.
- **Stage 2 now trains on spike + near-spike samples** instead of all samples. Direction accuracy went from 65% → 92.6% on the validation set.
- **Stage 2 heavily regularized** — `max_depth=3`, `min_child_weight=20`, `gamma=1.0`, `reg_alpha=1.0`, `reg_lambda=5.0` — to combat val/live divergence.
- **Added `scale_pos_weight`** for Stage 1 class imbalance instead of manual sample weights.

### Feature expansion (May 2026)

- **+8 lagged macro features** — 3-day and 5-day variants of VIX, DXY, oil, gold, treasury, SP500
- **+5 earnings features** — EPS/rev surprise, beat-miss streak, post-earnings drift, earnings-day volatility
- **+5 macro features** — yield curve spread, USD index, crude oil, gold, SP500 5-day return
- **+2 sentiment features** — news count z-score, news spike flag
- **+2 technical features** — volatility z-score, overnight gap

### Universe expansion

27 → 60 tickers across biotech, energy, financials, consumer, industrial, more semiconductors. Training data roughly doubled from ~13,500 to ~30,000 rows.

## Design decisions

- **No lookahead bias** — every feature uses only data available before 9:30 AM ET. yfinance's `end` parameter is exclusive, so today's bar is never pulled into technical features.
- **Time-series validation** — most recent 20% of training data as validation. No shuffle. Prevents leakage that would inflate val scores.
- **Adaptive thresholds** — per-ticker spike definition based on historical volatility; prevents mega-cap tech from being "always spiking."
- **Aggressive caching** — Finnhub news, earnings data, and OHLCV cached to disk under `data/`. Retrains go from ~30 min cold to ~3 min warm.
- **Graceful degradation** — missing earnings or news data defaults to 0; model still runs. (Macro NaNs are NOT zeroed because "0% VIX change" is a real signal, not absence.)
- **Server-side bracket exits** — Alpaca manages TP/SL after order fill. No intraday monitor process needed.
- **Stable-named artifacts for cross-workflow handoff** — `watchlist-YYYY-MM-DD` rather than `<run_id>` so downstream workflows can find by date without knowing the upstream run ID.
- **Per-account state isolation** — OPEN and DELAYED have separate trade logs, separate API keys, separate Alpaca accounts. No cross-contamination of the A/B experiment.

## Known caveats

- **News leakage if predictions run after market open.** Finnhub returns all articles in a date range with no time filter. Running predictions at 3 PM ET would include 10 AM news in "overnight" sentiment features, artificially inflating accuracy. This is why the DELAYED workflow loads the saved 9:15 watchlist instead of regenerating predictions.
- **24-hour Telegram /start expiry.** If no workflow runs within ~24 hours of someone sending `/start`, their subscription may be dropped from `getUpdates`. Workaround: manual `--poll-only` run or re-send `/start` on a trading morning.
- **Cloudflare Worker free-tier cap of 5 crons.** All 5 slots are in use (trade, delayed trade, daily validate, weekly validate, bi-weekly retrain). Adding more requires consolidation or upgrading the plan.
- **NYSE holiday list is hardcoded.** Update annually against the official NYSE calendar. Last updated through 2027.
- **Paper data feed is IEX only.** `get_latest_price()` uses `feed=iex` — the free-tier feed. Prices are accurate but represent only IEX flow, not the full SIP consolidated tape. For position sizing, this is fine; for sub-second backtesting, it's not.
- **No real P&L tracking yet.** Validation metrics include a "signal P&L proxy" but the system doesn't pull realized P&L from Alpaca's order history. This is a future enhancement; for now, log into the Alpaca dashboard once a week to see equity.

## Anti-features (deliberately not built)

- **No intraday monitoring process.** Alpaca handles bracket exits server-side. No need for a long-running daemon.
- **No webhook listener for Telegram.** Polling via `getUpdates` is enough for a daily-cadence bot and avoids running a public HTTP server.
- **No retraining inside the daily workflow.** Retraining happens bi-weekly on Sunday (even ISO weeks). Daily workflows download the artifact. Keeps daily runs fast (~3 min) and avoids non-determinism from retraining mid-week.

## Files an AI assistant would want to read first

If you're an LLM trying to understand this codebase quickly, start with:

1. [config.py](config.py) — universe, feature columns, all tunable knobs in one place
2. [predict/spike_detector.py](predict/spike_detector.py) — the predict + retrain entry point; everything else hangs off this
3. [predict/trade.py](predict/trade.py) — Alpaca integration, position sizing, risk filters
4. [.github/workflows/daily_trade.yml](.github/workflows/daily_trade.yml) — the canonical workflow; daily_trade_delayed and daily_validate are variations
5. The "System architecture" Mermaid diagram above for the overall data flow
