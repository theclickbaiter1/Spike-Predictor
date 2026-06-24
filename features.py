"""
features.py — Feature engineering for the Pre-Market Spike Detector.

Improvements:
  A. Adaptive per-ticker threshold (replaces flat 3%)
  B. Pre-market price/volume features (Alpaca + yfinance)
  C. All features use ONLY data available before market open.
"""

import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    ADAPTIVE_MULTIPLIER,
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    DATA_DIR,
    EARNINGS_CACHE_DIR,
    FEATURE_COLUMNS,
    FINNHUB_API_KEY,
    LABEL_MAP,
    SECTOR_MAP,
    SPIKE_THRESHOLD,
    TRAINING_LOOKBACK_YEARS,
)
from stat_mech_features import (
    attach_live_stat_mech,
    build_live_cross_section_cache,
    compute_beta_norm_params,
    enrich_training_frame,
    sentiment_entropy_from_scores,
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


def compute_adaptive_threshold_series(ohlcv: pd.DataFrame, multiplier: float = ADAPTIVE_MULTIPLIER) -> pd.Series:
    """Per-day adaptive spike threshold (aligned with compute_adaptive_target)."""
    intraday_return = (ohlcv["Close"] - ohlcv["Open"]) / ohlcv["Open"]
    abs_return = intraday_return.abs()
    avg_abs = abs_return.rolling(20, min_periods=10).mean().shift(1)
    return (avg_abs * multiplier).clip(lower=SPIKE_THRESHOLD)


def classify_intraday_return(ret: float, threshold: float = None) -> str:
    """Classify an intraday return using flat or adaptive threshold."""
    th = SPIKE_THRESHOLD if threshold is None else threshold
    if ret >= th:
        return "SPIKE UP"
    if ret <= -th:
        return "SPIKE DOWN"
    return "FLAT"


# Features filled with 0 when missing (same list used in training + prediction)
OPTIONAL_FILL_COLS = [
    "eps_surprise_last", "revenue_surprise_last", "earnings_streak",
    "post_earnings_drift_1d", "earnings_volatility",
    "premarket_gap_pct", "premarket_volume_ratio", "gap_sentiment_agreement",
    "earnings_catalyst_score",
    "vix_change_3d", "vix_change_5d", "vix_regime",
    "dxy_change_5d", "crude_oil_change_5d", "gold_change_5d",
    "treasury_10y_delta_5d", "sp500_return_3d",
    "susceptibility_proxy", "criticality_proxy",
]


def impute_features_for_predict(X: pd.DataFrame) -> pd.DataFrame:
    """
    Align prediction-time feature handling with training:
    fill optional NaNs with 0, drop rows missing required (macro) features.
    If macro edge-cases would drop every ticker, fall back to 0-fill with a warning.
    """
    X = X.copy()
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = 0.0
    for col in OPTIONAL_FILL_COLS:
        if col in X.columns:
            X[col] = X[col].fillna(0)
    required = [c for c in FEATURE_COLUMNS if c not in OPTIONAL_FILL_COLS]
    valid = X[required].notna().all(axis=1)
    if valid.any():
        return X.loc[valid]
    # Weekend / stale macro: avoid empty watchlist — degrade gracefully at predict time
    print("  ⚠ All tickers missing required features; using 0-fill fallback for predict.")
    return X.fillna(0)


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


def compute_catalyst_features(
    ohlcv: pd.DataFrame,
    sentiment_mean: pd.Series | None = None,
    earnings_catalyst: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Catalyst confirmation features (no lookahead).
    Training uses open-gap proxy for premarket_gap when intraday pre-market unavailable.
    """
    close = ohlcv["Close"]
    open_ = ohlcv["Open"]
    volume = ohlcv["Volume"]
    f = pd.DataFrame(index=ohlcv.index)
    prev_close = close.shift(1)
    f["premarket_gap_pct"] = (open_ - prev_close) / prev_close.replace(0, np.nan)
    avg_vol = volume.rolling(10, min_periods=5).mean().shift(1)
    f["premarket_volume_ratio"] = (volume.shift(1) / avg_vol.replace(0, np.nan)).fillna(0)

    if sentiment_mean is not None:
        sent = sentiment_mean.reindex(ohlcv.index).fillna(0)
        gap = f["premarket_gap_pct"].fillna(0)
        agree = (
            (gap.abs() > 0.002)
            & (sent.abs() > 0.1)
            & (np.sign(gap) == np.sign(sent))
        ).astype(int)
        f["gap_sentiment_agreement"] = agree
    else:
        f["gap_sentiment_agreement"] = 0

    if earnings_catalyst is not None:
        f["earnings_catalyst_score"] = earnings_catalyst.reindex(ohlcv.index).fillna(0)
    else:
        f["earnings_catalyst_score"] = 0.0

    return f


def fetch_live_premarket_price(ticker: str) -> float | None:
    """Fetch latest trade price from Alpaca data API (pre-market). Returns None if unavailable."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return None
    try:
        import requests
        url = f"https://data.alpaca.markets/v2/stocks/{ticker}/trades/latest"
        headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        }
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.ok:
            trade = resp.json().get("trade", {})
            return float(trade.get("p", 0)) or None
    except Exception:
        pass
    return None


def apply_live_premarket_overrides(row: dict, ticker: str, ohlcv: pd.DataFrame) -> dict:
    """Override premarket_gap_pct with live price when available before the open."""
    if ohlcv.empty:
        return row
    prev_close = float(ohlcv["Close"].iloc[-2]) if len(ohlcv) >= 2 else float(ohlcv["Close"].iloc[-1])
    live_px = fetch_live_premarket_price(ticker)
    if live_px and prev_close > 0:
        row["premarket_gap_pct"] = (live_px - prev_close) / prev_close
        sent = row.get("overnight_sentiment_mean", 0) or 0
        gap = row["premarket_gap_pct"]
        if abs(gap) > 0.002 and abs(sent) > 0.1 and np.sign(gap) == np.sign(sent):
            row["gap_sentiment_agreement"] = 1
        else:
            row["gap_sentiment_agreement"] = 0
    return row


def _download_safe(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    """
    Download OHLCV with retries. yfinance bulk threads often rate-limit mid-run;
    we use threads=False and fall back to Ticker.history().
    """
    import time

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts <= start_ts:
        end_ts = start_ts + pd.Timedelta(days=1)

    for attempt in range(retries):
        try:
            data = yf.download(
                ticker,
                start=start_ts.strftime("%Y-%m-%d"),
                end=end_ts.strftime("%Y-%m-%d"),
                progress=False,
                threads=False,
                auto_adjust=True,
            )
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            if not data.empty and len(data) >= 5:
                return data

            # Fallback: Ticker.history (often succeeds when download() fails)
            hist = yf.Ticker(ticker).history(
                start=start_ts.strftime("%Y-%m-%d"),
                end=end_ts.strftime("%Y-%m-%d"),
                auto_adjust=True,
            )
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.droplevel(1)
            if not hist.empty:
                return hist
        except Exception as e:
            if attempt == retries - 1:
                print(f"  ⚠ yfinance download failed for {ticker}: {e}")
        time.sleep(0.35 * (attempt + 1))

    return pd.DataFrame()


def compute_macro_features(trading_dates, start_date, end_date):
    vix_data = _download_safe("^VIX", start_date, end_date)
    tnx_data = _download_safe("^TNX", start_date, end_date)
    spy_data = _download_safe("^GSPC", start_date, end_date)
    irx_data = _download_safe("^IRX", start_date, end_date)   # 13-week T-bill
    dxy_data = _download_safe("DX-Y.NYB", start_date, end_date)  # US Dollar Index
    oil_data = _download_safe("CL=F", start_date, end_date)    # WTI Crude
    gold_data = _download_safe("GC=F", start_date, end_date)   # Gold

    macro = pd.DataFrame(index=trading_dates)

    # Original 5 macro features
    macro["vix"] = vix_data["Close"].reindex(trading_dates, method="ffill") if not vix_data.empty else np.nan
    macro["vix_change"] = vix_data["Close"].pct_change().reindex(trading_dates, method="ffill").shift(1) if not vix_data.empty else np.nan
    macro["treasury_10y"] = tnx_data["Close"].reindex(trading_dates, method="ffill") if not tnx_data.empty else np.nan
    if not spy_data.empty:
        macro["sp500_prev_return"] = spy_data["Close"].pct_change().reindex(trading_dates, method="ffill").shift(1)
    else:
        macro["sp500_prev_return"] = np.nan

    # NEW — Yield curve spread: 10Y minus 3-month (recession indicator)
    if not tnx_data.empty and not irx_data.empty:
        tnx_close = tnx_data["Close"].reindex(trading_dates, method="ffill")
        irx_close = irx_data["Close"].reindex(trading_dates, method="ffill")
        macro["yield_curve_spread"] = (tnx_close - irx_close).shift(1)
    else:
        macro["yield_curve_spread"] = np.nan

    # NEW — US Dollar Index daily change
    if not dxy_data.empty:
        macro["dxy_change"] = dxy_data["Close"].pct_change().reindex(trading_dates, method="ffill").shift(1)
    else:
        macro["dxy_change"] = np.nan

    # NEW — WTI Crude Oil daily change
    if not oil_data.empty:
        macro["crude_oil_change"] = oil_data["Close"].pct_change().reindex(trading_dates, method="ffill").shift(1)
    else:
        macro["crude_oil_change"] = np.nan

    # NEW — Gold daily change
    if not gold_data.empty:
        macro["gold_change"] = gold_data["Close"].pct_change().reindex(trading_dates, method="ffill").shift(1)
    else:
        macro["gold_change"] = np.nan

    # NEW — S&P 500 5-day return (broader trend)
    if not spy_data.empty:
        macro["sp500_5d_return"] = spy_data["Close"].pct_change(5).reindex(trading_dates, method="ffill").shift(1)
    else:
        macro["sp500_5d_return"] = np.nan

    # ── Lagged macro features (multi-day momentum / regime context) ──────────
    # Single-day snapshots miss sustained macro moves. These rolling features
    # let the model see "VIX has been elevated for days" vs a one-day blip.

    if not vix_data.empty:
        vix_close = vix_data["Close"].reindex(trading_dates, method="ffill")
        macro["vix_change_3d"] = vix_close.pct_change(3).shift(1)
        macro["vix_change_5d"] = vix_close.pct_change(5).shift(1)
        vix_mean20 = vix_close.rolling(20, min_periods=10).mean()
        vix_std20 = vix_close.rolling(20, min_periods=10).std()
        macro["vix_regime"] = ((vix_close - vix_mean20) / vix_std20.replace(0, np.nan)).shift(1)
    else:
        macro["vix_change_3d"] = np.nan
        macro["vix_change_5d"] = np.nan
        macro["vix_regime"] = np.nan

    if not dxy_data.empty:
        dxy_close = dxy_data["Close"].reindex(trading_dates, method="ffill")
        macro["dxy_change_5d"] = dxy_close.pct_change(5).shift(1)
    else:
        macro["dxy_change_5d"] = np.nan

    if not oil_data.empty:
        oil_close = oil_data["Close"].reindex(trading_dates, method="ffill")
        macro["crude_oil_change_5d"] = oil_close.pct_change(5).shift(1)
    else:
        macro["crude_oil_change_5d"] = np.nan

    if not gold_data.empty:
        gold_close = gold_data["Close"].reindex(trading_dates, method="ffill")
        macro["gold_change_5d"] = gold_close.pct_change(5).shift(1)
    else:
        macro["gold_change_5d"] = np.nan

    if not tnx_data.empty:
        tnx_close = tnx_data["Close"].reindex(trading_dates, method="ffill")
        macro["treasury_10y_delta_5d"] = (tnx_close - tnx_close.shift(5)).shift(1)
    else:
        macro["treasury_10y_delta_5d"] = np.nan

    if not spy_data.empty:
        spy_close = spy_data["Close"].reindex(trading_dates, method="ffill")
        macro["sp500_return_3d"] = spy_close.pct_change(3).shift(1)
    else:
        macro["sp500_return_3d"] = np.nan

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
    cal["days_to_earnings"] = 90  # Default: no known upcoming earnings
    cal["is_earnings_day"] = 0

    # Point-in-time: only use reported historical earnings (no yfinance future dates).
    earnings_data = _fetch_earnings_data(ticker)
    earnings_dates_list = []
    for rec in earnings_data:
        try:
            earnings_dates_list.append(pd.Timestamp(rec["date"]).normalize())
        except Exception:
            continue

    if not earnings_dates_list:
        return cal

    earnings_dates = pd.DatetimeIndex(sorted(set(earnings_dates_list)))

    for idx, d in enumerate(dates):
        d_norm = pd.Timestamp(d).normalize()
        # Only earnings already reported on or before d_norm (no lookahead)
        known = earnings_dates[earnings_dates <= d_norm]
        if len(known) == 0:
            continue

        if d_norm in known:
            cal.iloc[idx, cal.columns.get_loc("is_earnings_day")] = 1
            cal.iloc[idx, cal.columns.get_loc("days_to_earnings")] = 0
        else:
            days_since = (d_norm - known[-1]).days
            cal.iloc[idx, cal.columns.get_loc("days_to_earnings")] = min(days_since, 90)

        yesterday_was_earnings = earnings_dates[earnings_dates == d_norm - timedelta(days=1)]
        if len(yesterday_was_earnings) > 0:
            cal.iloc[idx, cal.columns.get_loc("is_earnings_day")] = 1

    return cal


# ── Earnings Features ────────────────────────────────────────────────────────

def _normalize_earnings_records(records: list[dict]) -> list[dict]:
    """Sort by date and backfill missing revenue estimates from prior quarters."""
    if not records:
        return records

    parsed = []
    for rec in records:
        try:
            parsed.append({
                "date": str(pd.Timestamp(rec["date"]).normalize().date()),
                "epsActual": rec.get("epsActual", np.nan),
                "epsEstimate": rec.get("epsEstimate", np.nan),
                "revenueActual": rec.get("revenueActual", np.nan),
                "revenueEstimate": rec.get("revenueEstimate", np.nan),
            })
        except Exception:
            continue

    parsed.sort(key=lambda x: x["date"])
    for i, rec in enumerate(parsed):
        est = rec.get("revenueEstimate")
        if pd.notna(est) and est != 0:
            continue
        prior = [
            p["revenueActual"] for p in parsed[:i]
            if pd.notna(p.get("revenueActual")) and p.get("revenueActual", 0) != 0
        ]
        if len(prior) >= 4:
            rec["revenueEstimate"] = prior[-4]
        elif prior:
            rec["revenueEstimate"] = prior[-1]

    return parsed


def _fetch_earnings_data(ticker: str) -> list[dict]:
    """
    Fetch earnings history for a ticker via yfinance. Returns a list of dicts
    with keys: date, epsActual, epsEstimate, revenueActual, revenueEstimate.
    Results are cached to disk to avoid redundant API calls.
    """
    import json

    cache_path = EARNINGS_CACHE_DIR / f"{ticker}_earnings.json"
    if cache_path.exists():
        with open(cache_path, "r") as f:
            cached = json.load(f)
        if cached:
            normalized = _normalize_earnings_records(cached)
            has_surprise_inputs = any(
                pd.notna(r.get("revenueActual")) and r.get("revenueActual", 0) != 0
                and pd.notna(r.get("revenueEstimate")) and r.get("revenueEstimate", 0) != 0
                for r in normalized
            )
            if has_surprise_inputs:
                return normalized

    records = []
    try:
        t = yf.Ticker(ticker)

        # Finnhub earnings (has revenue actual/estimate when available)
        if FINNHUB_API_KEY:
            try:
                import requests
                resp = requests.get(
                    "https://finnhub.io/api/v1/stock/earnings",
                    params={"symbol": ticker, "token": FINNHUB_API_KEY},
                    timeout=10,
                )
                if resp.ok:
                    for item in resp.json():
                        records.append({
                            "date": str(pd.Timestamp(item.get("period", item.get("date", ""))).normalize().date()),
                            "epsActual": float(item.get("actual", np.nan)) if item.get("actual") is not None else np.nan,
                            "epsEstimate": float(item.get("estimate", np.nan)) if item.get("estimate") is not None else np.nan,
                            "revenueActual": float(item.get("revenueActual", np.nan)) if item.get("revenueActual") is not None else np.nan,
                            "revenueEstimate": float(item.get("revenueEstimate", np.nan)) if item.get("revenueEstimate") is not None else np.nan,
                        })
            except Exception:
                pass

        # Try earnings_history (preferred — has EPS surprise data)
        eh = getattr(t, "earnings_history", None)
        if eh is not None and isinstance(eh, pd.DataFrame) and not eh.empty:
            for idx, row in eh.iterrows():
                rec = {
                    "date": str(pd.Timestamp(idx).normalize().date()) if not isinstance(idx, str) else idx,
                    "epsActual": float(row.get("epsActual", row.get("Reported EPS", np.nan))),
                    "epsEstimate": float(row.get("epsEstimate", row.get("EPS Estimate", np.nan))),
                    "revenueActual": np.nan,
                    "revenueEstimate": np.nan,
                }
                records.append(rec)
        elif not records:
            # Fallback: use earnings_dates which has Reported/Estimate EPS
            ed = getattr(t, "earnings_dates", None)
            if ed is not None and isinstance(ed, pd.DataFrame) and not ed.empty:
                for idx, row in ed.iterrows():
                    reported = row.get("Reported EPS", np.nan)
                    estimate = row.get("EPS Estimate", np.nan)
                    rev_actual = row.get("Reported Revenue", row.get("Revenue", np.nan))
                    rev_est = row.get("Revenue Estimate", np.nan)
                    if pd.notna(reported):  # Only include quarters with actual results
                        rec = {
                            "date": str(pd.Timestamp(idx).normalize().date()),
                            "epsActual": float(reported),
                            "epsEstimate": float(estimate) if pd.notna(estimate) else np.nan,
                            "revenueActual": float(rev_actual) if pd.notna(rev_actual) else np.nan,
                            "revenueEstimate": float(rev_est) if pd.notna(rev_est) else np.nan,
                        }
                        records.append(rec)

        # Try to get revenue data from quarterly_financials + income statement
        qf = getattr(t, "quarterly_financials", None)
        if qf is not None and isinstance(qf, pd.DataFrame) and not qf.empty:
            rev_row = None
            for label in ["Total Revenue", "Revenue", "Net Revenue"]:
                if label in qf.index:
                    rev_row = qf.loc[label]
                    break
            if rev_row is not None:
                for rec in records:
                    if pd.notna(rec.get("revenueActual")):
                        continue
                    rec_date = pd.Timestamp(rec["date"])
                    for col_date in rev_row.index:
                        if abs((pd.Timestamp(col_date) - rec_date).days) < 45:
                            rec["revenueActual"] = float(rev_row[col_date]) if pd.notna(rev_row[col_date]) else np.nan
                            break

        # Revenue from income statement (yfinance) — map quarters to earnings dates
        for stmt_attr in ("quarterly_income_stmt", "quarterly_financials"):
            stmt = getattr(t, stmt_attr, None)
            if stmt is None or not isinstance(stmt, pd.DataFrame) or stmt.empty:
                continue
            rev_row = None
            for label in ["Total Revenue", "Revenue", "Net Revenue"]:
                if label in stmt.index:
                    rev_row = stmt.loc[label]
                    break
            if rev_row is None:
                continue
            sorted_cols = sorted(rev_row.index, key=lambda x: pd.Timestamp(x))
            rev_by_date = {pd.Timestamp(c): float(rev_row[c]) for c in sorted_cols if pd.notna(rev_row[c])}
            for rec in records:
                if pd.notna(rec.get("revenueActual")) and rec.get("revenueActual", 0) != 0:
                    continue
                rec_date = pd.Timestamp(rec["date"])
                best_col, best_diff = None, 999
                for col_date, val in rev_by_date.items():
                    diff = abs((col_date - rec_date).days)
                    if diff < best_diff and diff < 60:
                        best_diff = diff
                        best_col = col_date
                if best_col is not None:
                    rec["revenueActual"] = rev_by_date[best_col]
                if pd.isna(rec.get("revenueEstimate")) or rec.get("revenueEstimate", 0) == 0:
                    prior = [v for d, v in rev_by_date.items() if d < rec_date]
                    if len(prior) >= 1:
                        rec["revenueEstimate"] = prior[-1]
                    if len(prior) >= 4:
                        rec["revenueEstimate"] = prior[-4]

    except Exception as e:
        print(f"  ⚠ Earnings fetch failed for {ticker}: {e}")

    # Deduplicate by date (prefer Finnhub rows with revenue)
    by_date: dict[str, dict] = {}
    for rec in records:
        d = rec.get("date", "")
        if d not in by_date:
            by_date[d] = rec
        else:
            existing = by_date[d]
            if pd.isna(existing.get("revenueActual")) and pd.notna(rec.get("revenueActual")):
                by_date[d] = rec
            elif pd.isna(existing.get("revenueEstimate")) and pd.notna(rec.get("revenueEstimate")):
                existing["revenueEstimate"] = rec["revenueEstimate"]

    records = _normalize_earnings_records(list(by_date.values()))

    # Cache to disk
    try:
        with open(cache_path, "w") as f:
            json.dump(records, f)
    except Exception:
        pass

    return records


def compute_earnings_features(ticker: str, dates: pd.DatetimeIndex, ohlcv: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute 5 earnings-based features for a ticker across multiple dates.
    All features use only data available before each date (no lookahead).

    Features:
      - eps_surprise_last:       Most recent EPS surprise %
      - revenue_surprise_last:   Most recent revenue surprise %
      - earnings_streak:         Consecutive beat (+) or miss (-) count
      - post_earnings_drift_1d:  1-day stock return after last earnings
      - earnings_volatility:     Avg |1-day return| on earnings days (last 4 Qs)
    """
    earnings_data = _fetch_earnings_data(ticker)
    ef = pd.DataFrame(index=dates, columns=[
        "eps_surprise_last", "revenue_surprise_last", "earnings_streak",
        "post_earnings_drift_1d", "earnings_volatility",
    ], dtype=float)
    ef[:] = 0.0  # Default to 0 (graceful degradation for ETFs)

    if not earnings_data:
        return ef

    # Parse earnings dates and sort ascending
    parsed = []
    for rec in earnings_data:
        try:
            ed = pd.Timestamp(rec["date"]).normalize()
            parsed.append({
                "date": ed,
                "epsActual": rec.get("epsActual", np.nan),
                "epsEstimate": rec.get("epsEstimate", np.nan),
                "revenueActual": rec.get("revenueActual", np.nan),
                "revenueEstimate": rec.get("revenueEstimate", np.nan),
            })
        except Exception:
            continue

    if not parsed:
        return ef

    parsed.sort(key=lambda x: x["date"])

    # Precompute 1-day returns on earnings days (needs OHLCV)
    earnings_1d_returns = {}
    if ohlcv is not None and not ohlcv.empty:
        for p in parsed:
            ed = p["date"]
            # Find the next trading day after earnings
            future_days = ohlcv.index[ohlcv.index >= ed]
            if len(future_days) >= 2:
                day_of = future_days[0]
                day_after = future_days[1]
                ret = (ohlcv.loc[day_after, "Close"] - ohlcv.loc[day_of, "Close"]) / ohlcv.loc[day_of, "Close"]
                earnings_1d_returns[ed] = float(ret)

    for idx, d in enumerate(dates):
        d_norm = pd.Timestamp(d).normalize()
        # Get earnings reports BEFORE this date
        past = [p for p in parsed if p["date"] < d_norm]
        if not past:
            continue

        # Most recent earnings
        latest = past[-1]

        # eps_surprise_last
        if pd.notna(latest["epsActual"]) and pd.notna(latest["epsEstimate"]) and abs(latest["epsEstimate"]) > 0.001:
            ef.iloc[idx, 0] = (latest["epsActual"] - latest["epsEstimate"]) / abs(latest["epsEstimate"])

        # revenue_surprise_last
        if pd.notna(latest["revenueActual"]) and pd.notna(latest["revenueEstimate"]) and abs(latest["revenueEstimate"]) > 0:
            ef.iloc[idx, 1] = (latest["revenueActual"] - latest["revenueEstimate"]) / abs(latest["revenueEstimate"])

        # earnings_streak
        streak = 0
        for p in reversed(past):
            if pd.notna(p["epsActual"]) and pd.notna(p["epsEstimate"]):
                if p["epsActual"] > p["epsEstimate"]:
                    if streak >= 0:
                        streak += 1
                    else:
                        break
                elif p["epsActual"] < p["epsEstimate"]:
                    if streak <= 0:
                        streak -= 1
                    else:
                        break
                else:
                    break  # Exact match ends streak
            else:
                break
        ef.iloc[idx, 2] = streak

        # post_earnings_drift_1d
        if latest["date"] in earnings_1d_returns:
            ef.iloc[idx, 3] = earnings_1d_returns[latest["date"]]

        # earnings_volatility: avg |1-day return| across last 4 earnings
        last_4 = past[-4:]
        abs_rets = [abs(earnings_1d_returns[p["date"]]) for p in last_4 if p["date"] in earnings_1d_returns]
        if abs_rets:
            ef.iloc[idx, 4] = np.mean(abs_rets)

    return ef


# ── Full Feature Matrix Builder ──────────────────────────────────────────────

def build_training_dataset(
    tickers: list[str], news_client, sentiment_scorer,
    lookback_years: int = TRAINING_LOOKBACK_YEARS,
    end_date_str: str = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix + target with adaptive threshold."""
    from news import filter_articles_for_date, compute_sentiment_from_scores

    ref_date = datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str else datetime.now()
    end_date = ref_date.strftime("%Y-%m-%d")
    start_date = (ref_date - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
    buffer_start = (ref_date - timedelta(days=lookback_years * 365 + 60)).strftime("%Y-%m-%d")

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

    import time

    # Step 3: Per-ticker features (with bulk sentiment + pre-market)
    print("  [3/4] Computing per-ticker features + sentiment + pre-market...")
    all_rows = []

    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(0.25)
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
        sent_entropy_rows = []
        for date in ohlcv.index:
            overnight = filter_articles_for_date(all_articles, date)
            day_hl = [a.get("headline", "").strip() for a in overnight if a.get("headline", "").strip()]
            day_sc = [headline_scores[h] for h in day_hl if h in headline_scores]
            sent_rows.append(compute_sentiment_from_scores(day_sc))
            sent_entropy_rows.append(sentiment_entropy_from_scores(day_sc))
        sent_df = pd.DataFrame(sent_rows, index=ohlcv.index)
        sent_df["sentiment_entropy"] = sent_entropy_rows
        # Relative news volume: z-score vs ticker's own 60-day baseline
        nc = sent_df["overnight_news_count"]
        nc_mean = nc.rolling(60, min_periods=10).mean().shift(1)
        nc_std = nc.rolling(60, min_periods=10).std().shift(1)
        sent_df["news_count_z_score"] = ((nc - nc_mean) / nc_std.replace(0, np.nan)).fillna(0)
        # Binary flag: did this ticker get 3x its normal overnight news volume?
        sent_df["news_spike"] = (nc >= (nc_mean * 3).clip(lower=3)).astype(int)

        # Earnings features
        earnings_feats = compute_earnings_features(ticker, ohlcv.index, ohlcv)

        # Catalyst confirmation (premarket proxy + sentiment agreement)
        catalyst = compute_catalyst_features(
            ohlcv,
            sentiment_mean=sent_df["overnight_sentiment_mean"],
            earnings_catalyst=(
                0.6 * earnings_feats["eps_surprise_last"]
                + 0.4 * earnings_feats["revenue_surprise_last"]
            ),
        )

        # Combine all features
        ticker_features = sent_df.copy()
        ticker_features = ticker_features.join(tech, how="left")
        macro_cols = ["vix", "vix_change", "treasury_10y", "sp500_prev_return",
                      "yield_curve_spread", "dxy_change", "crude_oil_change",
                      "gold_change", "sp500_5d_return",
                      "vix_change_3d", "vix_change_5d", "vix_regime",
                      "dxy_change_5d", "crude_oil_change_5d", "gold_change_5d",
                      "treasury_10y_delta_5d", "sp500_return_3d"]
        macro_available = [c for c in macro_cols if c in macro.columns]
        ticker_features = ticker_features.join(macro[macro_available], how="left")
        ticker_features["sector_momentum_5d"] = sector_mom.reindex(ticker_features.index, method="ffill")
        ticker_features = ticker_features.join(cal, how="left")
        ticker_features = ticker_features.join(earnings_feats, how="left")
        ticker_features = ticker_features.join(catalyst, how="left")

        # Adaptive target + raw intraday return (for near-spike filtering in model)
        target = compute_adaptive_target(ohlcv)
        adaptive_thresh = compute_adaptive_threshold_series(ohlcv)
        intraday_return = (ohlcv["Close"] - ohlcv["Open"]) / ohlcv["Open"]
        ticker_features["_ticker"] = ticker
        ticker_features["_target"] = target
        ticker_features["_intraday_return"] = intraday_return
        ticker_features["_adaptive_threshold"] = adaptive_thresh

        start_ts = pd.Timestamp(start_date)
        ticker_features = ticker_features[ticker_features.index >= start_ts]
        all_rows.append(ticker_features)

        n_news = sum(1 for s in sent_rows if s["overnight_news_count"] > 0)
        n_spikes = (target[ohlcv.index >= start_ts] != 0).sum()
        print(f"OK ({len(ticker_features)} days, {len(unique_headlines)} headlines, {n_spikes} spikes)")

    if not all_rows:
        raise RuntimeError("No valid ticker data collected.")

    combined = pd.concat(all_rows, axis=0)

    # Stat-mech features (cross-section + sector coupling)
    print("  [4/5] Computing statistical mechanics features...")
    beta_mean, beta_std = compute_beta_norm_params(combined)
    combined = enrich_training_frame(combined, beta_mean, beta_std)
    beta_norm_path = DATA_DIR / "beta_norm.json"
    with open(beta_norm_path, "w") as f:
        json.dump({"mean": beta_mean, "std": beta_std}, f)

    # Step 5: Clean up
    print(f"\n  [5/5] Finalizing dataset...")
    y_raw = combined["_target"]
    intraday_ret = combined["_intraday_return"]
    adaptive_thresh = combined["_adaptive_threshold"]
    X = combined[FEATURE_COLUMNS].copy()
    y = y_raw.map(LABEL_MAP)
    # Pre-market + earnings features are optional — fill NaN with 0
    # (graceful degradation for dates where data isn't available)
    # Only fill features where NaN genuinely means "no data = no signal".
    # Macro features are NOT filled here — a 0% change is a real signal,
    # not "missing". They stay NaN and get dropped by valid_mask if absent.
    optional_fill_cols = OPTIONAL_FILL_COLS
    for col in optional_fill_cols:
        if col in X.columns:
            X[col] = X[col].fillna(0)

    tickers = combined["_ticker"]

    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask].astype(int)
    intraday_ret = intraday_ret[valid_mask]
    adaptive_thresh = adaptive_thresh[valid_mask]
    tickers = tickers[valid_mask]

    print(f"\n  Dataset built: {len(X)} rows × {len(FEATURE_COLUMNS)} features")
    for label_idx, name in enumerate(["spike_down", "flat", "spike_up"]):
        count = (y == label_idx).sum()
        print(f"    {name}: {count} ({count/len(y)*100:.1f}%)")

    return X, y, intraday_ret, tickers, adaptive_thresh


def fetch_adaptive_threshold(ticker: str, date_str: str) -> float:
    """Adaptive spike threshold for a ticker on a given calendar date."""
    ts = pd.Timestamp(date_str)
    start = (ts - timedelta(days=60)).strftime("%Y-%m-%d")
    end = (ts + timedelta(days=1)).strftime("%Y-%m-%d")
    ohlcv = _download_safe(ticker, start, end)
    if ohlcv.empty:
        return SPIKE_THRESHOLD
    thresh = compute_adaptive_threshold_series(ohlcv)
    if ts in thresh.index and pd.notna(thresh.loc[ts]):
        return float(thresh.loc[ts])
    prior = thresh[thresh.index <= ts]
    if len(prior) > 0 and pd.notna(prior.iloc[-1]):
        return float(prior.iloc[-1])
    return SPIKE_THRESHOLD


# ── Single-Day Feature Builder (for live prediction) ────────────────────────

def build_single_day_features(
    ticker: str, date: datetime, news_client, sentiment_scorer,
    ohlcv_cache=None, macro_cache=None, cross_section_cache=None,
    beta_mean: float = 0.0, beta_std: float = 1.0,
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
    # Compute news_count_z_score and news_spike from cached news history.
    today_count = sent.get("overnight_news_count", 0)
    try:
        from news import filter_articles_for_date
        cached_articles = news_client.fetch_news(
            ticker,
            (date - timedelta(days=65)).strftime("%Y-%m-%d"),
            date.strftime("%Y-%m-%d"),
        )
        daily_counts = []
        for d in ohlcv.index[-60:]:
            day_articles = filter_articles_for_date(cached_articles, d)
            daily_counts.append(len(day_articles))
        if len(daily_counts) >= 10:
            nc_mean = np.mean(daily_counts)
            nc_std = np.std(daily_counts)
            row["news_count_z_score"] = (today_count - nc_mean) / nc_std if nc_std > 0 else 0
            row["news_spike"] = 1 if today_count >= max(nc_mean * 3, 3) else 0
        else:
            row["news_count_z_score"] = 0
            row["news_spike"] = 0
    except Exception:
        row["news_count_z_score"] = 0
        row["news_spike"] = 0

    if last_idx in tech.index:
        for col in tech.columns:
            row[col] = tech.loc[last_idx, col]

    if isinstance(macro_row, dict):
        for k in ["vix", "vix_change", "treasury_10y", "sp500_prev_return",
                  "yield_curve_spread", "dxy_change", "crude_oil_change",
                  "gold_change", "sp500_5d_return",
                  "vix_change_3d", "vix_change_5d", "vix_regime",
                  "dxy_change_5d", "crude_oil_change_5d", "gold_change_5d",
                  "treasury_10y_delta_5d", "sp500_return_3d"]:
            row[k] = macro_row.get(k, np.nan)

    if last_idx in sector_mom.index:
        row["sector_momentum_5d"] = sector_mom.loc[last_idx]

    if last_idx in cal.index:
        for col in cal.columns:
            row[col] = cal.loc[last_idx, col]

    # Earnings features
    earnings_feats = compute_earnings_features(ticker, ohlcv.index, ohlcv)
    if last_idx in earnings_feats.index:
        for col in earnings_feats.columns:
            row[col] = earnings_feats.loc[last_idx, col]

    earn_cat = None
    if last_idx in earnings_feats.index:
        earn_cat = (
            0.6 * earnings_feats.loc[last_idx, "eps_surprise_last"]
            + 0.4 * earnings_feats.loc[last_idx, "revenue_surprise_last"]
        )
    catalyst = compute_catalyst_features(
        ohlcv,
        sentiment_mean=pd.Series({last_idx: sent.get("overnight_sentiment_mean", 0)}),
        earnings_catalyst=pd.Series({last_idx: earn_cat}) if earn_cat is not None else None,
    )
    if last_idx in catalyst.index:
        for col in catalyst.columns:
            row[col] = catalyst.loc[last_idx, col]

    row = apply_live_premarket_overrides(row, ticker, ohlcv)

    if cross_section_cache is not None:
        attach_live_stat_mech(row, ticker, cross_section_cache, beta_mean, beta_std)
    else:
        for col in [
            "sector_magnetization", "sector_abs_magnetization", "local_field",
            "coupling_alignment", "cross_section_entropy", "sentiment_entropy",
            "inverse_temperature", "susceptibility_proxy", "criticality_proxy",
        ]:
            row[col] = np.nan

    return pd.Series({col: row.get(col, np.nan) for col in FEATURE_COLUMNS})


def load_beta_norm() -> tuple[float, float]:
    """Load train-set inverse_temperature normalization from disk."""
    path = DATA_DIR / "beta_norm.json"
    if not path.exists():
        return 0.0, 1.0
    with open(path) as f:
        data = json.load(f)
    return float(data.get("mean", 0.0)), float(data.get("std", 1.0))


def finalize_live_stat_mech(feature_rows: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Second pass: attach cross-section stat-mech features for live prediction."""
    beta_mean, beta_std = load_beta_norm()
    gaps = {
        t: float(r.get("overnight_gap", 0) or 0)
        for t, r in feature_rows.items()
        if pd.notna(r.get("overnight_gap"))
    }
    sent_scores = {}
    for t, r in feature_rows.items():
        sent = r.get("overnight_sentiment_mean", 0)
        if pd.notna(sent) and sent != 0:
            sent_scores[t] = [float(sent)]
    cache = build_live_cross_section_cache(gaps, sent_scores or None)

    out = {}
    for ticker, row in feature_rows.items():
        row_dict = row.to_dict()
        attach_live_stat_mech(row_dict, ticker, cache, beta_mean, beta_std)
        out[ticker] = pd.Series({col: row_dict.get(col, np.nan) for col in FEATURE_COLUMNS})
    return out
