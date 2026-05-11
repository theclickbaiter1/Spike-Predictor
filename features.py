"""
features.py — Feature engineering for the Pre-Market Spike Detector.

Improvements:
  A. Adaptive per-ticker threshold (replaces flat 3%)
  B. Pre-market price/volume features (Alpaca + yfinance)
  C. All features use ONLY data available before market open.
"""

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    ADAPTIVE_MULTIPLIER,
    FEATURE_COLUMNS,
    LABEL_MAP,
    SECTOR_MAP,
    SPIKE_THRESHOLD,
    TRAINING_LOOKBACK_YEARS,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Target Variable (Adaptive Threshold) ─────────────────────────────────────

def compute_adaptive_target(ohlcv: pd.DataFrame, multiplier: float = ADAPTIVE_MULTIPLIER) -> pd.Series:
    """
    Compute 3-class target using per-ticker adaptive threshold.
    Spike = intraday move > multiplier × 20-day rolling avg |intraday return|.
    The threshold adapts to each stock's own volatility.
    """
    intraday_return = (ohlcv["Close"] - ohlcv["Open"]) / ohlcv["Open"]
    abs_return = intraday_return.abs()

    # 20-day avg absolute return, shifted to avoid lookahead
    avg_abs = abs_return.rolling(20, min_periods=10).mean().shift(1)

    # Adaptive threshold with a minimum floor
    threshold = (avg_abs * multiplier).clip(lower=SPIKE_THRESHOLD)

    target = pd.Series(0, index=ohlcv.index)
    target[intraday_return >= threshold] = 1    # spike up
    target[intraday_return <= -threshold] = -1  # spike down
    return target


def compute_target(df: pd.DataFrame) -> pd.Series:
    """Legacy flat-threshold target (kept for backward compatibility)."""
    intraday_return = (df["Close"] - df["Open"]) / df["Open"]
    target = pd.Series(0, index=df.index)
    target[intraday_return >= SPIKE_THRESHOLD] = 1
    target[intraday_return <= -SPIKE_THRESHOLD] = -1
    return target


# ── Technical Indicators ─────────────────────────────────────────────────────

def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_technical_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features from OHLCV. No lookahead."""
    df = ohlcv.copy()
    close, high, low, volume, open_ = df["Close"], df["High"], df["Low"], df["Volume"], df["Open"]
    f = pd.DataFrame(index=df.index)
    f["prev_close"] = close.shift(1)
    f["rsi_14"] = _compute_rsi(close, 14)
    f["ema_10"] = close.ewm(span=10, adjust=False).mean()
    f["realized_vol_20d"] = close.pct_change().rolling(20).std() * np.sqrt(252)
    f["avg_volume_10d"] = volume.rolling(10).mean()
    f["prev_day_return"] = ((close - open_) / open_).shift(1)
    f["prev_day_range"] = ((high - low) / close).shift(1)
    f["gap_3d"] = close.pct_change(3).shift(1)
    f["overnight_gap"] = (open_ - close.shift(1)) / close.shift(1)
    # Volatility z-score: is current vol elevated vs its own 60-day history?
    vol_20d = close.pct_change().rolling(20).std()
    vol_mean = vol_20d.rolling(60, min_periods=20).mean().shift(1)
    vol_std = vol_20d.rolling(60, min_periods=20).std().shift(1)
    f["vol_z_score"] = ((vol_20d.shift(1) - vol_mean) / vol_std.replace(0, np.nan))
    return f


# ── Macro / Market Context ───────────────────────────────────────────────────

def _download_safe(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        return data
    except Exception as e:
        print(f"  ⚠ yfinance download failed for {ticker}: {e}")
        return pd.DataFrame()


def compute_macro_features(trading_dates, start_date, end_date):
    vix_data = _download_safe("^VIX", start_date, end_date)
    tnx_data = _download_safe("^TNX", start_date, end_date)
    spy_data = _download_safe("^GSPC", start_date, end_date)
    macro = pd.DataFrame(index=trading_dates)
    macro["vix"] = vix_data["Close"].reindex(trading_dates, method="ffill") if not vix_data.empty else np.nan
    macro["vix_change"] = vix_data["Close"].pct_change().reindex(trading_dates, method="ffill").shift(1) if not vix_data.empty else np.nan
    macro["treasury_10y"] = tnx_data["Close"].reindex(trading_dates, method="ffill") if not tnx_data.empty else np.nan
    if not spy_data.empty:
        macro["sp500_prev_return"] = spy_data["Close"].pct_change().reindex(trading_dates, method="ffill").shift(1)
    else:
        macro["sp500_prev_return"] = np.nan
    return macro


def compute_sector_momentum(sector_etf, trading_dates, start_date, end_date):
    data = _download_safe(sector_etf, start_date, end_date)
    if data.empty:
        return pd.Series(np.nan, index=trading_dates, name="sector_momentum_5d")
    return data["Close"].pct_change(5).reindex(trading_dates, method="ffill").rename("sector_momentum_5d")


# ── Calendar Features ────────────────────────────────────────────────────────

def compute_calendar_features(ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    cal = pd.DataFrame(index=dates)
    cal["day_of_week"] = dates.dayofweek
    cal["is_monday"] = (dates.dayofweek == 0).astype(int)
    cal["is_friday"] = (dates.dayofweek == 4).astype(int)
    cal["days_to_earnings"] = -1
    cal["is_earnings_day"] = 0

    try:
        t = yf.Ticker(ticker)
        earnings_dates = None
        if hasattr(t, "earnings_dates") and t.earnings_dates is not None:
            earnings_dates = t.earnings_dates.index
        elif hasattr(t, "calendar") and t.calendar is not None:
            cd = t.calendar
            if isinstance(cd, dict) and "Earnings Date" in cd:
                earnings_dates = pd.to_datetime(cd["Earnings Date"])
            elif isinstance(cd, pd.DataFrame) and "Earnings Date" in cd.index:
                earnings_dates = pd.to_datetime(cd.loc["Earnings Date"])

        if earnings_dates is not None and len(earnings_dates) > 0:
            earnings_dates = pd.DatetimeIndex(earnings_dates).normalize()
            for idx, d in enumerate(dates):
                d_norm = pd.Timestamp(d).normalize()
                future = earnings_dates[earnings_dates >= d_norm]
                if len(future) > 0:
                    days = (future[0] - d_norm).days
                    cal.iloc[idx, cal.columns.get_loc("days_to_earnings")] = days
                    if days == 0:
                        cal.iloc[idx, cal.columns.get_loc("is_earnings_day")] = 1
                past = earnings_dates[earnings_dates == d_norm - timedelta(days=1)]
                if len(past) > 0:
                    cal.iloc[idx, cal.columns.get_loc("is_earnings_day")] = 1
    except Exception:
        pass

    return cal


# ── Full Feature Matrix Builder ──────────────────────────────────────────────

def build_training_dataset(
    tickers: list[str], news_client, sentiment_scorer,
    lookback_years: int = TRAINING_LOOKBACK_YEARS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix + target with adaptive threshold and pre-market data."""
    from news import filter_articles_for_date, compute_sentiment_from_scores
    from premarket import PremarketProvider

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
    buffer_start = (datetime.now() - timedelta(days=lookback_years * 365 + 60)).strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"  BUILDING TRAINING DATASET (v2 — adaptive threshold)")
    print(f"  Period: {start_date} → {end_date}")
    print(f"  Tickers: {len(tickers)}")
    print(f"{'='*60}\n")

    # Step 1: Macro
    print("  [1/4] Fetching macro data...")
    spy_data = _download_safe("^GSPC", buffer_start, end_date)
    if spy_data.empty:
        raise RuntimeError("Failed to fetch S&P 500 data.")
    trading_dates = spy_data.index
    macro = compute_macro_features(trading_dates, buffer_start, end_date)

    # Step 2: Sector ETFs
    print("  [2/4] Fetching sector ETF momentum...")
    sector_cache = {}
    for etf in set(SECTOR_MAP.values()):
        sector_cache[etf] = compute_sector_momentum(etf, trading_dates, buffer_start, end_date)

    # Step 3: Per-ticker features (with bulk sentiment + pre-market)
    print("  [3/4] Computing per-ticker features + sentiment + pre-market...")
    all_rows = []

    for i, ticker in enumerate(tickers):
        print(f"    [{i+1}/{len(tickers)}] {ticker}...", end=" ", flush=True)

        # OHLCV
        ohlcv = _download_safe(ticker, buffer_start, end_date)
        if ohlcv.empty or len(ohlcv) < 30:
            print("SKIP")
            continue

        # Technical features
        tech = compute_technical_features(ohlcv)
        cal = compute_calendar_features(ticker, ohlcv.index)
        sector_etf = SECTOR_MAP.get(ticker, "XLK")
        sector_mom = sector_cache.get(sector_etf, pd.Series(np.nan, index=trading_dates, name="sector_momentum_5d"))

        # Bulk sentiment
        all_articles = news_client.prefetch_ticker_news(ticker, buffer_start, end_date)
        unique_headlines = list(set(
            a.get("headline", "").strip() for a in all_articles if a.get("headline", "").strip()
        ))
        headline_scores = {}
        if unique_headlines:
            scores = sentiment_scorer.score_headlines(unique_headlines)
            headline_scores = dict(zip(unique_headlines, scores))

        sent_rows = []
        for date in ohlcv.index:
            overnight = filter_articles_for_date(all_articles, date)
            day_hl = [a.get("headline", "").strip() for a in overnight if a.get("headline", "").strip()]
            day_sc = [headline_scores[h] for h in day_hl if h in headline_scores]
            sent_rows.append(compute_sentiment_from_scores(day_sc))
        sent_df = pd.DataFrame(sent_rows, index=ohlcv.index)
        # Relative news volume: z-score vs ticker's own 60-day baseline
        nc = sent_df["overnight_news_count"]
        nc_mean = nc.rolling(60, min_periods=10).mean().shift(1)
        nc_std = nc.rolling(60, min_periods=10).std().shift(1)
        sent_df["news_count_z_score"] = ((nc - nc_mean) / nc_std.replace(0, np.nan)).fillna(0)

        # Historical pre-market data (yfinance hourly)
        pm_data = PremarketProvider.get_historical_premarket(ticker, buffer_start, end_date)
        pm_features = pd.DataFrame(index=ohlcv.index)
        if not pm_data.empty:
            pm_data = pm_data.reindex(ohlcv.index, method=None)
            prev_close = ohlcv["Close"].shift(1)
            pm_features["premarket_change"] = (pm_data["premarket_price"] - prev_close) / prev_close
            avg_vol = ohlcv["Volume"].rolling(10).mean()
            pm_features["premarket_volume_ratio"] = pm_data["premarket_volume"] / avg_vol.replace(0, np.nan)
        else:
            pm_features["premarket_change"] = ohlcv["Open"].shift(0).sub(ohlcv["Close"].shift(1)).div(ohlcv["Close"].shift(1))
            pm_features["premarket_volume_ratio"] = np.nan

        # Combine all features
        ticker_features = sent_df.copy()
        ticker_features = ticker_features.join(pm_features, how="left")
        ticker_features = ticker_features.join(tech, how="left")
        ticker_features = ticker_features.join(macro[["vix", "vix_change", "treasury_10y", "sp500_prev_return"]], how="left")
        ticker_features["sector_momentum_5d"] = sector_mom.reindex(ticker_features.index, method="ffill")
        ticker_features = ticker_features.join(cal, how="left")

        # Adaptive target
        target = compute_adaptive_target(ohlcv)
        ticker_features["_ticker"] = ticker
        ticker_features["_target"] = target

        start_ts = pd.Timestamp(start_date)
        ticker_features = ticker_features[ticker_features.index >= start_ts]
        all_rows.append(ticker_features)

        n_news = sum(1 for s in sent_rows if s["overnight_news_count"] > 0)
        n_spikes = (target[ohlcv.index >= start_ts] != 0).sum()
        print(f"OK ({len(ticker_features)} days, {len(unique_headlines)} headlines, {n_spikes} spikes)")

    if not all_rows:
        raise RuntimeError("No valid ticker data collected.")

    combined = pd.concat(all_rows, axis=0)

    # Step 4: Clean up
    print(f"\n  [4/4] Finalizing dataset...")
    y_raw = combined["_target"]
    X = combined[FEATURE_COLUMNS].copy()
    y = y_raw.map(LABEL_MAP)
    # Pre-market features are optional — fill NaN with 0 (graceful degradation
    # for dates where yfinance hourly data isn't available)
    for col in ["premarket_change", "premarket_volume_ratio"]:
        if col in X.columns:
            X[col] = X[col].fillna(0)

    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask].astype(int)

    print(f"\n  Dataset built: {len(X)} rows × {len(FEATURE_COLUMNS)} features")
    for label_idx, name in enumerate(["spike_down", "flat", "spike_up"]):
        count = (y == label_idx).sum()
        print(f"    {name}: {count} ({count/len(y)*100:.1f}%)")

    return X, y


# ── Single-Day Feature Builder (for live prediction) ────────────────────────

def build_single_day_features(
    ticker: str, date: datetime, news_client, sentiment_scorer,
    ohlcv_cache=None, macro_cache=None, premarket_data=None,
) -> pd.Series:
    """Build one feature row for a ticker on a given date (live prediction)."""
    from news import compute_sentiment_features

    end_str = date.strftime("%Y-%m-%d")
    start_str = (date - timedelta(days=60)).strftime("%Y-%m-%d")

    sent = compute_sentiment_features(ticker, date, news_client, sentiment_scorer)

    if ohlcv_cache and ticker in ohlcv_cache:
        ohlcv = ohlcv_cache[ticker]
    else:
        ohlcv = _download_safe(ticker, start_str, end_str)

    if ohlcv.empty:
        return pd.Series({col: np.nan for col in FEATURE_COLUMNS})

    tech = compute_technical_features(ohlcv)
    cal = compute_calendar_features(ticker, ohlcv.index)

    if macro_cache:
        macro_row = macro_cache
    else:
        macro_df = compute_macro_features(ohlcv.index, start_str, end_str)
        macro_row = macro_df.iloc[-1].to_dict() if not macro_df.empty else {}

    sector_etf = SECTOR_MAP.get(ticker, "XLK")
    sector_mom = compute_sector_momentum(sector_etf, ohlcv.index, start_str, end_str)

    last_idx = ohlcv.index[-1]
    row = {}
    row.update(sent)
    # For live prediction, news_count_z_score defaults to 0 (no rolling baseline available)
    row["news_count_z_score"] = 0

    # Pre-market features
    if premarket_data and ticker in premarket_data:
        pm = premarket_data[ticker]
        pc = pm.get("prev_close", np.nan)
        pp = pm.get("premarket_price", np.nan)
        pv = pm.get("premarket_volume", np.nan)
        row["premarket_change"] = (pp - pc) / pc if pc and pc > 0 and not np.isnan(pp) else 0
        avg_vol = tech.loc[last_idx, "avg_volume_10d"] if last_idx in tech.index else 1
        row["premarket_volume_ratio"] = pv / avg_vol if avg_vol and avg_vol > 0 and not np.isnan(pv) else 0
    else:
        row["premarket_change"] = tech.loc[last_idx, "overnight_gap"] if last_idx in tech.index else 0
        row["premarket_volume_ratio"] = 0

    if last_idx in tech.index:
        for col in tech.columns:
            row[col] = tech.loc[last_idx, col]

    if isinstance(macro_row, dict):
        for k in ["vix", "vix_change", "treasury_10y", "sp500_prev_return"]:
            row[k] = macro_row.get(k, np.nan)

    if last_idx in sector_mom.index:
        row["sector_momentum_5d"] = sector_mom.loc[last_idx]

    if last_idx in cal.index:
        for col in cal.columns:
            row[col] = cal.loc[last_idx, col]

    return pd.Series({col: row.get(col, np.nan) for col in FEATURE_COLUMNS})
