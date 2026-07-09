"""
trade.py — Automated trading execution via Alpaca.

Integrates with the spike detector model to place bracket orders at market open.
Alpaca handles exits server-side (take-profit + stop-loss), so no intraday
monitoring process is needed.

Usage:
    python predict/trade.py --dry-run   # Show signals, no orders placed
    python predict/trade.py --paper     # Paper trading (default)
    python predict/trade.py --live      # Live trading (requires explicit flag)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from config import (
    ALPACA_API_KEY,
    ALPACA_API_KEY_DELAYED,
    ALPACA_LIVE_URL,
    ALPACA_PAPER_URL,
    ALPACA_SECRET_KEY,
    ALPACA_SECRET_KEY_DELAYED,
    DIRECTION_MARGIN_MIN,
    ENTRY_ORDER_TYPE,
    EV_COST_BUFFER,
    EV_MAX_POSITION_PCT,
    EV_MIN_EDGE,
    EV_POSITION_MULTIPLIER,
    FEATURE_COLUMNS,
    LIMIT_ENTRY_DIP_PCT,
    MAX_CONSECUTIVE_TICKER_DAYS,
    MAX_DAILY_LOSS_PCT,
    MAX_POSITIONS_PER_DAY,
    MAX_POSITION_PCT,
    MODEL_PATH,
    OUTPUT_DIR,
    REQUIRE_GAP_SENTIMENT_AGREEMENT,
    SECTOR_AGREEMENT_REQUIRED,
    SKIP_TRADE_NEAR_EARNINGS_DAYS,
    SKIP_TRADE_VIX_ABOVE,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TRADE_LOG_PATH,
    TRADE_PROB_COLUMN,
    get_trade_threshold,
    UNIVERSE,
)

ET = ZoneInfo("America/New_York")

CANONICAL_TRADE_FIELDS = [
    "date", "time", "ticker", "direction", "qty", "entry_price",
    "take_profit", "stop_loss", "p_spike", "p_spike_raw", "p_spike_trade", "p_dir",
    "expected_value", "order_id", "mode", "order_status", "filled_qty",
]
LEGACY_TRADE_FIELDS_V2 = [
    "date", "time", "ticker", "direction", "qty", "entry_price",
    "take_profit", "stop_loss", "p_spike", "p_spike_raw", "p_spike_trade", "p_dir", "order_id", "mode",
]


LEGACY_TRADE_FIELDS = [
    "date", "time", "ticker", "direction", "qty", "entry_price",
    "take_profit", "stop_loss", "p_spike", "p_dir", "order_id", "mode",
]


def _normalize_trade_row(row: list[str]) -> dict | None:
    if len(row) == len(CANONICAL_TRADE_FIELDS):
        rec = dict(zip(CANONICAL_TRADE_FIELDS, row))
    elif len(row) == len(LEGACY_TRADE_FIELDS_V2):
        rec = dict(zip(LEGACY_TRADE_FIELDS_V2, row))
        rec["expected_value"] = ""
        rec["order_status"] = ""
        rec["filled_qty"] = ""
    elif len(row) == len(LEGACY_TRADE_FIELDS):
        rec = dict(zip(LEGACY_TRADE_FIELDS, row))
        rec["p_spike_raw"] = rec["p_spike"]
        rec["p_spike_trade"] = rec["p_spike"]
        rec["expected_value"] = ""
        rec["order_status"] = ""
        rec["filled_qty"] = ""
    else:
        return None
    return rec


def read_trade_log(path: Path) -> pd.DataFrame:
    """Load trade log, tolerating legacy rows mixed with newer formats."""
    if not path.exists():
        return pd.DataFrame(columns=CANONICAL_TRADE_FIELDS)

    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header (may be stale)
        for row in reader:
            rec = _normalize_trade_row(row)
            if rec:
                rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=CANONICAL_TRADE_FIELDS)
    return pd.DataFrame(rows)


def write_trade_log(path: Path, df: pd.DataFrame) -> None:
    """Write trade log with canonical header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.reindex(columns=CANONICAL_TRADE_FIELDS)
    out.to_csv(path, index=False)


def alpaca_credentials(label: str = "open") -> tuple[str, str]:
    """Return API key/secret for the given account label."""
    if label == "delayed":
        return ALPACA_API_KEY_DELAYED or ALPACA_API_KEY, ALPACA_SECRET_KEY_DELAYED or ALPACA_SECRET_KEY
    return ALPACA_API_KEY, ALPACA_SECRET_KEY


# ── Alpaca Client ────────────────────────────────────────────────────────────

class AlpacaClient:
    """Thin wrapper around Alpaca REST API for order execution."""

    def __init__(self, paper=True, label: str = "open"):
        self.base_url = ALPACA_PAPER_URL if paper else ALPACA_LIVE_URL
        api_key, secret_key = alpaca_credentials(label)
        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        self.paper = paper
        self.label = label

    # Trading endpoints live on paper-api.alpaca.markets (paper) /
    # api.alpaca.markets (live). Market data endpoints live on a separate
    # host: data.alpaca.markets. Sending data queries to the trading host
    # returns 404 — which is what was silently breaking get_latest_price
    # and stopping every order from being placed.
    DATA_URL = "https://data.alpaca.markets"

    def _request(self, method, endpoint, json_data=None, base_url=None):
        import requests
        url = f"{base_url or self.base_url}{endpoint}"
        resp = requests.request(method, url, headers=self.headers, json=json_data, timeout=15)
        if not resp.ok:
            print(f"  ❌ Alpaca API error: {resp.status_code} {resp.text}")
            return None
        return resp.json()

    def get_account(self):
        return self._request("GET", "/v2/account")

    def get_positions(self):
        return self._request("GET", "/v2/positions")

    def get_orders(self, status="open"):
        return self._request("GET", f"/v2/orders?status={status}&limit=100")

    def get_order(self, order_id: str):
        return self._request("GET", f"/v2/orders/{order_id}")

    def place_bracket_order(self, ticker, side, qty, take_profit_price, stop_loss_price,
                            limit_price=None):
        """
        Place a bracket order: entry + take-profit limit + stop-loss stop.
        If limit_price is given the entry is a day-limit order (fills only on a
        favorable move); otherwise it's a market order. Alpaca handles the OCO
        exit automatically on the entry-fill side; unfilled day-limit entries
        cancel at the close.
        """
        order = {
            "symbol": ticker,
            "qty": str(qty),
            "side": side,        # "buy" or "sell"
            "type": "limit" if limit_price is not None else "market",
            "time_in_force": "day",
            "order_class": "bracket",
            "take_profit": {"limit_price": f"{take_profit_price:.2f}"},
            "stop_loss": {"stop_price": f"{stop_loss_price:.2f}"},
        }
        if limit_price is not None:
            order["limit_price"] = f"{limit_price:.2f}"
        return self._request("POST", "/v2/orders", json_data=order)

    def get_latest_price(self, ticker):
        """Get the latest trade price for a ticker (market-data API host)."""
        data = self._request(
            "GET",
            f"/v2/stocks/{ticker}/trades/latest?feed=iex",
            base_url=self.DATA_URL,
        )
        if data and "trade" in data:
            return float(data["trade"]["p"])
        return None

    def cancel_all_orders(self):
        return self._request("DELETE", "/v2/orders")

    def close_position(self, symbol):
        """Close a single position at market."""
        return self._request("DELETE", f"/v2/positions/{symbol}")

    def close_all_positions(self, cancel_orders=True):
        """
        Liquidate all open positions at market.
        Cancels open orders first so bracket legs don't block the close.
        """
        if cancel_orders:
            self.cancel_all_orders()
        return self._request("DELETE", "/v2/positions")


# ── Trade History (for consecutive-day tracking) ─────────────────────────────

def trade_log_path(label: str) -> Path:
    """Return the trade log path for a given account label."""
    if label == "open":
        return TRADE_LOG_PATH
    return OUTPUT_DIR / f"trade_log_{label}.csv"


def load_recent_trades(days=5, label: str = "open"):
    """Load trade log and return tickers traded in recent days."""
    path = trade_log_path(label)
    df = read_trade_log(path)
    if df.empty or "date" not in df.columns:
        return {}

    df["date"] = pd.to_datetime(df["date"])
    # Both sides are calendar dates (no time component) — the CSV writes
    # `now.strftime("%Y-%m-%d")` and pandas parses that as tz-naive midnight.
    # The cutoff must match: take "today in ET" as a plain date, subtract N
    # days, lift back to a tz-naive Timestamp. DST is irrelevant — we're
    # comparing calendar dates, not wall-clock instants.
    cutoff = pd.Timestamp((datetime.now(ET).date() - timedelta(days=days)))
    recent = df[df["date"] >= cutoff]

    # Count consecutive trading days per ticker (from most recent backward)
    ticker_streaks = {}
    for ticker in recent["ticker"].unique():
        ticker_dates = sorted(recent[recent["ticker"] == ticker]["date"].dt.date.unique(), reverse=True)
        streak = 0
        for i, d in enumerate(ticker_dates):
            if i == 0:
                streak = 1
            else:
                # Check if consecutive trading day (skip weekends)
                prev = ticker_dates[i - 1]
                gap = (prev - d).days
                if gap <= 3:  # Allow weekends
                    streak += 1
                else:
                    break
        ticker_streaks[ticker] = streak

    return ticker_streaks


def log_trades(trades, label: str = "open"):
    """Append trades to the trade log CSV for this account label."""
    if not trades:
        return

    path = trade_log_path(label)
    existing = read_trade_log(path)
    new_df = pd.DataFrame(trades)
    for col in ("p_spike_raw", "p_spike_trade"):
        if col not in new_df.columns:
            new_df[col] = new_df["p_spike"]
    for col in ("order_status", "filled_qty"):
        if col not in new_df.columns:
            new_df[col] = ""
    new_df = new_df.reindex(columns=CANONICAL_TRADE_FIELDS)
    write_trade_log(path, pd.concat([existing, new_df], ignore_index=True))


# ── Signal Generation ────────────────────────────────────────────────────────

def passes_direction_gates(
    prob_row,
    direction: str,
    coupling_alignment: float,
    gap_sentiment_agreement: float | None = None,
) -> bool:
    """Direction confidence + sector magnetization + gap/sentiment agreement."""
    p_spike = float(prob_row.get("p_spike_trade", prob_row["p_spike"]))
    if p_spike <= 0:
        return False
    p_dir = max(float(prob_row["p_up"]), float(prob_row["p_down"]))
    if p_dir / p_spike < DIRECTION_MARGIN_MIN:
        return False
    if SECTOR_AGREEMENT_REQUIRED:
        coupling = float(coupling_alignment or 0)
        if direction == "LONG" and coupling < 0:
            return False
        if direction == "SHORT" and coupling > 0:
            return False
    if REQUIRE_GAP_SENTIMENT_AGREEMENT and gap_sentiment_agreement is not None:
        if float(gap_sentiment_agreement or 0) < 1:
            return False
    return True


def passes_entry_filters(feature_row, vix: float | None) -> bool:
    """Macro and calendar gates for new entries."""
    if vix is not None and pd.notna(vix) and float(vix) >= SKIP_TRADE_VIX_ABOVE:
        return False
    days = feature_row.get("days_to_earnings", 99)
    if days is not None and pd.notna(days) and int(days) <= SKIP_TRADE_NEAR_EARNINGS_DAYS:
        return False
    if int(feature_row.get("is_earnings_day", 0) or 0) == 1:
        return False
    return True


def apply_top_signal_cap(signals: list[dict], max_pool: int) -> list[dict]:
    """Raise effective threshold if too many tickers pass (Jun 18 guard)."""
    if len(signals) <= max_pool:
        return signals
    signals.sort(key=lambda s: s["p_spike_trade"], reverse=True)
    floor = signals[max_pool - 1]["p_spike_trade"]
    capped = [s for s in signals if s["p_spike_trade"] >= floor]
    print(f"    ⚠ Top-signal cap: {len(signals)} → {len(capped)} (floor P={floor*100:.1f}%)")
    return capped[:max_pool]


def expected_edge(prob_row) -> float:
    """
    Expected net return per trade from model outputs.
    Uses magnitude head if present; otherwise falls back to 0.
    """
    p_trade = float(prob_row.get("p_spike_trade", prob_row.get("p_spike", 0)))
    signed = float(prob_row.get("expected_signed_return", 0) or 0)
    return (p_trade * signed) - EV_COST_BUFFER


def generate_signals():
    """Run the spike detector model and return ranked trade signals."""
    from features import _download_safe, build_single_day_features, compute_macro_features
    from features import impute_features_for_predict, finalize_live_stat_mech
    from model import TwoStageModel
    from news import FinBERTScorer, FinnhubClient

    s1_path = str(MODEL_PATH).replace(".json", "_s1.json")
    if not Path(s1_path).exists():
        print("  ❌ No trained model found. Run: python spike_detector.py --retrain")
        sys.exit(1)

    model = TwoStageModel()
    model.load()

    client = FinnhubClient()
    scorer = FinBERTScorer()

    today = datetime.now(ET)
    end_str = today.strftime("%Y-%m-%d")
    start_str = (today - timedelta(days=60)).strftime("%Y-%m-%d")

    # Macro data (shared across all tickers)
    print("  Fetching macro data...")
    spy_data = _download_safe("^GSPC", start_str, end_str)
    macro_df = compute_macro_features(
        spy_data.index if not spy_data.empty else pd.DatetimeIndex([]),
        start_str, end_str,
    )
    macro_cache = macro_df.iloc[-1].to_dict() if not macro_df.empty else {}

    # Build features for all tickers
    print(f"  Scanning {len(UNIVERSE)} tickers for signals...\n")
    feature_rows = {}
    for i, ticker in enumerate(UNIVERSE):
        print(f"    [{i+1}/{len(UNIVERSE)}] {ticker}...", end=" ", flush=True)
        row = build_single_day_features(
            ticker, today, client, scorer,
            macro_cache=macro_cache,
        )
        feature_rows[ticker] = row
        print("OK")

    feature_rows = finalize_live_stat_mech(feature_rows)
    X = pd.DataFrame(feature_rows).T
    X.columns = FEATURE_COLUMNS
    X = impute_features_for_predict(X)
    if X.empty:
        print("  No tickers with complete macro data for signals.")
        return []

    probs = model.predict_for_trade(X)
    vix = macro_cache.get("vix")
    if vix is not None and pd.notna(vix) and float(vix) >= SKIP_TRADE_VIX_ABOVE:
        print(f"\n  ⚠ VIX {float(vix):.1f} ≥ {SKIP_TRADE_VIX_ABOVE:.0f} — no new entries today.")
        return []
    threshold = get_trade_threshold(float(vix) if vix is not None else None)

    # Build signal list
    signals = []
    for ticker in X.index:
        r = probs.loc[ticker]
        p_trade = float(r.get("p_spike_trade", r["p_spike"]))
        if p_trade < threshold:
            continue
        row = feature_rows[ticker]
        if not passes_entry_filters(row, vix):
            continue
        direction = "LONG" if r["p_up"] > r["p_down"] else "SHORT"
        coupling = float(row.get("coupling_alignment", 0) or 0)
        gap_agree = row.get("gap_sentiment_agreement")
        if not passes_direction_gates(r, direction, coupling, gap_agree):
            continue
        edge = expected_edge(r)
        if model.return_model is not None and edge < EV_MIN_EDGE:
            continue
        p_dir = max(r["p_up"], r["p_down"])
        signals.append({
            "ticker": ticker,
            "direction": direction,
            "p_spike": float(r["p_spike"]),
            "p_spike_raw": float(r.get("p_spike_raw", r["p_spike"])),
            "p_spike_trade": p_trade,
            "p_dir": float(p_dir),
            "coupling_alignment": coupling,
            "expected_signed_return": float(r.get("expected_signed_return", 0) or 0),
            "expected_value": edge,
        })

    signals = apply_top_signal_cap(signals, MAX_POSITIONS_PER_DAY * 4)
    signals.sort(key=lambda s: s["p_spike_trade"], reverse=True)
    return signals


def load_signals_from_watchlist(csv_path: Path) -> list[dict]:
    """
    Load signals from a watchlist CSV produced by the morning prediction run.

    Used by the delayed-execution workflow so we trade off the 9:15 prediction
    snapshot (clean, pre-market data) rather than regenerating predictions at
    10:00 AM with intraday news leakage in the features.
    """
    if not csv_path.exists():
        print(f"  ❌ Watchlist not found at {csv_path}.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    vix = df["vix"].iloc[0] if "vix" in df.columns and len(df) else None
    if vix is not None and pd.notna(vix) and float(vix) >= SKIP_TRADE_VIX_ABOVE:
        print(f"  ⚠ VIX {float(vix):.1f} ≥ {SKIP_TRADE_VIX_ABOVE:.0f} — no new entries from watchlist.")
        return []
    threshold = get_trade_threshold(float(vix) if vix is not None and pd.notna(vix) else None)
    prob_col = TRADE_PROB_COLUMN if TRADE_PROB_COLUMN in df.columns else "p_spike"
    ev_gate_active = "expected_signed_return" in df.columns
    if not ev_gate_active:
        print("  ℹ Watchlist has no magnitude columns — EV gate skipped (retrain for EV sizing).")
    signals = []
    for _, r in df.iterrows():
        p_trade = float(r.get(prob_col, r["p_spike"]))
        if p_trade >= threshold:
            if not passes_entry_filters(r, vix):
                continue
            direction = "LONG" if r["p_up"] > r["p_down"] else "SHORT"
            coupling = float(r.get("coupling_alignment", 0) or 0)
            gap_agree = r.get("gap_sentiment_agreement") if "gap_sentiment_agreement" in r.index else None
            if not passes_direction_gates(r, direction, coupling, gap_agree):
                continue
            if "expected_value" in r.index and pd.notna(r.get("expected_value")):
                edge = float(r["expected_value"])
            else:
                edge = expected_edge(r)
            if ev_gate_active and edge < EV_MIN_EDGE:
                continue
            signals.append({
                "ticker": r["ticker"],
                "direction": direction,
                "p_spike": float(r["p_spike"]),
                "p_spike_raw": float(r.get("p_spike_raw", r["p_spike"])),
                "p_spike_trade": p_trade,
                "p_dir": float(max(r["p_up"], r["p_down"])),
                "expected_signed_return": float(r.get("expected_signed_return", 0) or 0),
                "expected_value": edge,
            })
    signals = apply_top_signal_cap(signals, MAX_POSITIONS_PER_DAY * 4)
    signals.sort(key=lambda s: s["p_spike_trade"], reverse=True)
    return signals


# ── Risk Filters ─────────────────────────────────────────────────────────────

def get_open_position_symbols(client) -> set[str]:
    """Return symbols with open positions on this account."""
    positions = client.get_positions()
    if not positions:
        return set()
    return {p["symbol"] for p in positions if float(p.get("qty", 0)) != 0}


def apply_risk_filters(signals, label: str = "open", client=None):
    """Apply risk management rules to filter and limit signals."""
    # 0. Skip tickers already held (open positions from prior days)
    held = set()
    if client is not None:
        held = get_open_position_symbols(client)
        if held:
            print(f"    Open positions: {', '.join(sorted(held))}")

    # 1. Remove tickers traded too many consecutive days
    streaks = load_recent_trades(label=label)
    filtered = []
    for sig in signals:
        if sig["ticker"] in held:
            print(f"    ⚠ {sig['ticker']} skipped — already holding open position")
            continue
        streak = streaks.get(sig["ticker"], 0)
        if streak >= MAX_CONSECUTIVE_TICKER_DAYS:
            print(f"    ⚠ {sig['ticker']} skipped — traded {streak} consecutive days")
            continue
        filtered.append(sig)

    # 2. Cap new entries: MAX_POSITIONS_PER_DAY minus existing holdings
    slots = max(0, MAX_POSITIONS_PER_DAY - len(held))
    if slots == 0 and filtered:
        print(f"    ⚠ All {MAX_POSITIONS_PER_DAY} position slots filled — no new entries today")
        return []
    if len(filtered) > slots:
        dropped = filtered[slots:]
        for d in dropped:
            print(f"    ⚠ {d['ticker']} skipped — max {MAX_POSITIONS_PER_DAY} positions/day "
                  f"({len(held)} already open)")
        filtered = filtered[:slots]

    return filtered


# ── Order Execution ──────────────────────────────────────────────────────────

def execute_trades(signals, client, dry_run=False):
    """Place bracket orders for each signal."""
    if not signals:
        print("\n  No signals passed risk filters. No trades today.")
        return []

    # Get account info for position sizing
    if not dry_run:
        account = client.get_account()
        if not account:
            print("  ❌ Failed to get account info. Aborting.")
            return []

        equity = float(account["equity"])
        buying_power = float(account["buying_power"])
        daily_pnl = float(account.get("equity", 0)) - float(account.get("last_equity", equity))
        daily_pnl_pct = daily_pnl / float(account.get("last_equity", equity)) if float(account.get("last_equity", equity)) > 0 else 0

        print(f"\n  Account: ${equity:,.2f} equity, ${buying_power:,.2f} buying power")

        # Daily loss circuit breaker
        if daily_pnl_pct <= -MAX_DAILY_LOSS_PCT:
            print(f"  🛑 Daily loss limit hit ({daily_pnl_pct*100:.1f}%). No trading today.")
            return []

        max_position_value = equity * MAX_POSITION_PCT
    else:
        equity = 100_000  # Simulated for dry run
        max_position_value = equity * MAX_POSITION_PCT

    now = datetime.now(ET)
    trade_records = []

    for sig in signals:
        ticker = sig["ticker"]
        direction = sig["direction"]
        side = "buy" if direction == "LONG" else "sell"
        edge = float(sig.get("expected_value", 0) or 0)
        ev_size_frac = min(EV_MAX_POSITION_PCT, max(0.0, edge * EV_POSITION_MULTIPLIER))
        position_value = max_position_value if ev_size_frac <= 0 else (equity * ev_size_frac)

        # Get current price for position sizing
        if not dry_run:
            price = client.get_latest_price(ticker)
            if not price:
                print(f"    ⚠ Could not get price for {ticker}. Skipping.")
                continue
        else:
            price = 100.0  # Placeholder for dry run

        # Entry: market bracket (default) or limit-dip day order.
        use_market_entry = ENTRY_ORDER_TYPE == "market"
        if use_market_entry:
            entry_ref = price
            if direction == "LONG":
                take_profit_price = round(entry_ref * (1 + TAKE_PROFIT_PCT), 2)
                stop_loss_price = round(entry_ref * (1 - STOP_LOSS_PCT), 2)
            else:
                take_profit_price = round(entry_ref * (1 - TAKE_PROFIT_PCT), 2)
                stop_loss_price = round(entry_ref * (1 + STOP_LOSS_PCT), 2)
            limit_price = None
            qty = int(position_value / entry_ref)
        else:
            if direction == "LONG":
                limit_price = round(price * (1 - LIMIT_ENTRY_DIP_PCT), 2)
                take_profit_price = round(limit_price * (1 + TAKE_PROFIT_PCT), 2)
                stop_loss_price = round(limit_price * (1 - STOP_LOSS_PCT), 2)
            else:  # SHORT
                limit_price = round(price * (1 + LIMIT_ENTRY_DIP_PCT), 2)
                take_profit_price = round(limit_price * (1 - TAKE_PROFIT_PCT), 2)
                stop_loss_price = round(limit_price * (1 + STOP_LOSS_PCT), 2)
            entry_ref = limit_price
            qty = int(position_value / limit_price)
        if qty < 1:
            print(f"    ⚠ {ticker} price ${entry_ref:.2f} too high for position size ${position_value:.0f}. Skipping.")
            continue

        trade_value = qty * entry_ref

        if dry_run:
            entry_label = "market" if use_market_entry else f"limit ${limit_price:.2f}"
            print(f"    🏷️  {ticker:6s} {direction:5s}  {qty:4d} shares @ {entry_label}"
                  f"  (ref ${price:.2f}, ${trade_value:,.0f})  TP=${take_profit_price:.2f}  SL=${stop_loss_price:.2f}"
                  f"  P(spike)={sig['p_spike']*100:.0f}% EV={edge*100:+.2f}%")
            order_id = "DRY-RUN"
            order_status = "dry-run"
            filled_qty = 0
        else:
            entry_label = "market" if use_market_entry else f"limit ${limit_price:.2f}"
            print(f"    📤 {ticker:6s} {direction:5s}  {qty:4d} shares @ {entry_label}"
                  f"  (ref ${price:.2f})  TP=${take_profit_price:.2f}  SL=${stop_loss_price:.2f}"
                  f"  P(spike)={sig['p_spike']*100:.0f}% EV={edge*100:+.2f}%", end=" ", flush=True)

            result = client.place_bracket_order(
                ticker, side, qty, take_profit_price, stop_loss_price,
                limit_price=limit_price,
            )
            if result:
                order_id = result.get("id", "unknown")
                order_status = result.get("status", "submitted")
                filled_qty = int(float(result.get("filled_qty", 0) or 0))
                # Brief poll — market entries often fill immediately during session.
                import time
                for _ in range(5):
                    if filled_qty >= qty or order_status in {"filled", "canceled", "expired", "rejected"}:
                        break
                    time.sleep(1)
                    detail = client.get_order(order_id)
                    if detail:
                        order_status = detail.get("status", order_status)
                        filled_qty = int(float(detail.get("filled_qty", 0) or 0))
                fill_note = f"filled {filled_qty}/{qty}" if filled_qty else order_status
                print(f"✅ ({order_id[:8]}, {fill_note})")
            else:
                print("❌ FAILED")
                continue

        trade_records.append({
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "ticker": ticker,
            "direction": direction,
            "qty": qty,
            "entry_price": round(entry_ref, 2),
            "take_profit": take_profit_price,
            "stop_loss": stop_loss_price,
            "p_spike": round(sig["p_spike"], 4),
            "p_spike_raw": round(sig.get("p_spike_raw", sig["p_spike"]), 4),
            "p_spike_trade": round(sig.get("p_spike_trade", sig["p_spike"]), 4),
            "p_dir": round(sig["p_dir"], 4),
            "expected_value": round(edge, 5),
            "order_id": order_id,
            "mode": "dry-run" if dry_run else ("paper" if client.paper else "live"),
            "order_status": order_status,
            "filled_qty": filled_qty,
        })

    return trade_records


def close_all_open_positions(client, dry_run=False, label: str = "open") -> list[dict]:
    """
    Cancel open orders and liquidate all positions at market (EOD flatten).
    Returns a list of closed position records for logging/notify.
    """
    positions = client.get_positions() if not dry_run else []
    if dry_run:
        print("  [DRY RUN] Would cancel orders and close all positions.")
        return []

    if not positions:
        print("  No open positions to close.")
        return []

    symbols = [p["symbol"] for p in positions]
    print(f"  Closing {len(symbols)} position(s): {', '.join(symbols)}")

    client.cancel_all_orders()
    result = client.close_all_positions(cancel_orders=False)
    if result is None:
        print("  ❌ close_all_positions API call failed.")
        return []

    import time
    for _ in range(15):
        remaining = client.get_positions() or []
        if not remaining:
            break
        time.sleep(2)
    remaining = client.get_positions() or []
    if remaining:
        syms = ", ".join(p["symbol"] for p in remaining)
        print(f"  ⚠ Still holding after close: {syms}")

    now = datetime.now(ET)
    records = []
    for p in positions:
        records.append({
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "ticker": p["symbol"],
            "direction": "LONG" if float(p.get("qty", 0)) > 0 else "SHORT",
            "qty": abs(int(float(p.get("qty", 0)))),
            "entry_price": float(p.get("avg_entry_price", 0)),
            "unrealized_pl": float(p.get("unrealized_pl", 0)),
            "label": label,
        })
        print(f"    ✅ Closed {p['symbol']} ({p.get('qty')} shares)")

    return records


# ── Main ─────────────────────────────────────────────────────────────────────

def print_banner(mode, label):
    now = datetime.now(ET)
    print()
    print("═" * 65)
    print(f"  SPIKE TRADER [{label.upper()}] — {now.strftime('%Y-%m-%d %I:%M %p ET')}")
    print(f"  Mode: {mode.upper()}")
    print("═" * 65)


def main():
    parser = argparse.ArgumentParser(description="Automated spike trading via Alpaca")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--paper", action="store_true", default=True,
                       help="Paper trading (default)")
    group.add_argument("--live", action="store_true",
                       help="Live trading — real money!")
    group.add_argument("--dry-run", action="store_true",
                       help="Show signals and simulated orders, no actual trades")
    parser.add_argument("--label", type=str, default="open",
                        help="Account label (e.g. 'open', 'delayed'). Determines trade log filename.")
    parser.add_argument("--watchlist", type=str, default=None,
                        help="Path to a watchlist CSV. If provided, skips signal "
                             "regeneration and trades off this snapshot.")
    parser.add_argument("--close-all", action="store_true",
                        help="Cancel open orders and liquidate all positions (EOD flatten)")
    parser.add_argument("--status", action="store_true",
                        help="Print account equity, cash, and open positions; then exit")
    args = parser.parse_args()

    if args.live:
        mode = "live"
    elif args.dry_run:
        mode = "dry-run"
    else:
        mode = "paper"

    label = args.label
    print_banner(mode, label)

    # Safety check for live trading
    if mode == "live":
        confirm = input("\n  ⚠️  LIVE TRADING — real money! Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("  Aborted.")
            sys.exit(0)

    # Validate API keys (except for dry run)
    api_key, secret_key = alpaca_credentials(label)
    if mode != "dry-run" and (not api_key or not secret_key):
        print(f"\n  ❌ Alpaca API keys must be set for label '{label}'.")
        if label == "delayed":
            print("     Set ALPACA_API_KEY_DELAYED and ALPACA_SECRET_KEY_DELAYED in .env")
        else:
            print("     Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        sys.exit(1)

    if args.close_all:
        print("\n  EOD CLOSE — liquidating all open positions\n")
        if mode == "dry-run":
            close_all_open_positions(None, dry_run=True, label=label)
        else:
            client = AlpacaClient(paper=(mode == "paper"), label=label)
            records = close_all_open_positions(client, dry_run=False, label=label)
            if records:
                close_log = OUTPUT_DIR / f"eod_close_{label}.json"
                import json
                close_log.parent.mkdir(parents=True, exist_ok=True)
                with open(close_log, "w") as f:
                    json.dump(records, f, indent=2)
                print(f"\n  📄 Close log saved to {close_log}")
        print("═" * 65)
        return

    if args.status:
        client = AlpacaClient(paper=(mode == "paper"), label=label)
        account = client.get_account()
        if not account:
            print("\n  ❌ Could not fetch account (check API keys — they change after a paper reset).")
            sys.exit(1)
        positions = client.get_positions() or []
        equity = float(account["equity"])
        print(f"\n  Account [{label.upper()}]")
        print(f"    Equity:    ${equity:,.2f}")
        print(f"    Cash:      ${float(account['cash']):,.2f}")
        print(f"    Buying power: ${float(account['buying_power']):,.2f}")
        print(f"    Positions: {len(positions)}")
        for p in positions:
            print(f"      {p['symbol']}: {p['qty']} @ ${float(p['avg_entry_price']):.2f}")
        open_orders = client.get_orders(status="open") or []
        print(f"    Open orders: {len(open_orders)}")
        for o in open_orders[:5]:
            print(f"      {o['symbol']} {o['side']} {o['type']} {o['status']} qty={o['qty']}")
        recent = client.get_orders(status="all") or []
        if recent:
            print(f"\n    Recent orders (paper dashboard: app.alpaca.markets/paper):")
            for o in recent[:5]:
                print(f"      {o.get('submitted_at','')[:16]} {o['symbol']:6} {o['side']:4} "
                      f"{o['status']:10} filled={o.get('filled_qty','0')}/{o.get('qty','?')}")
        print("═" * 65)
        return

    # Step 1: Get signals (regenerate or load from watchlist)
    if args.watchlist:
        watchlist_path = Path(args.watchlist)
        print(f"\n  Step 1: Loading signals from {watchlist_path}...")
        signals = load_signals_from_watchlist(watchlist_path)
    else:
        print("\n  Step 1: Generating spike predictions...\n")
        signals = generate_signals()

    if not signals:
        print("\n  📭 No tickers above threshold today. No trades.")
        print("═" * 65)
        return

    threshold = get_trade_threshold()
    print(f"\n  Found {len(signals)} signals above {threshold*100:.0f}% threshold:")
    for s in signals:
        print(
            f"    {s['ticker']:6s} {s['direction']:5s}  "
            f"P(trade)={s['p_spike_trade']*100:.1f}%  EV={float(s.get('expected_value', 0))*100:+.2f}%"
        )

    # Step 2: Apply risk filters
    print(f"\n  Step 2: Applying risk filters...")
    trade_client = None if mode == "dry-run" else AlpacaClient(paper=(mode == "paper"), label=label)
    filtered = apply_risk_filters(signals, label=label, client=trade_client)

    if not filtered:
        print("\n  All signals filtered out by risk rules. No trades today.")
        print("═" * 65)
        return

    # Step 3: Execute trades
    print(f"\n  Step 3: {'Simulating' if mode == 'dry-run' else 'Placing'} orders...\n")

    if mode == "dry-run":
        trade_records = execute_trades(filtered, None, dry_run=True)
    else:
        trade_records = execute_trades(filtered, trade_client, dry_run=False)

    # Step 4: Log trades
    if trade_records:
        log_trades(trade_records, label=label)
        print(f"\n  📄 {len(trade_records)} trades logged to {trade_log_path(label)}")

    # Step 5: Summary
    filled = sum(int(t.get("filled_qty") or 0) for t in trade_records)
    print(f"\n  {'=' * 60}")
    print(f"  SUMMARY: {len(trade_records)} orders {'simulated' if mode == 'dry-run' else 'submitted'}"
          + (f", {filled} shares filled" if not mode == "dry-run" else ""))
    for t in trade_records:
        status = t.get("order_status", "")
        fill = t.get("filled_qty", "")
        extra = f"  [{status}, filled {fill}/{t['qty']}]" if status and mode != "dry-run" else ""
        print(f"    {t['ticker']:6s} {t['direction']:5s}  {t['qty']:4d} shares"
              f"  TP=${t['take_profit']:.2f}  SL=${t['stop_loss']:.2f}{extra}")
    print(f"  {'=' * 60}\n")


if __name__ == "__main__":
    main()
