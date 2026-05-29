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
    ALPACA_LIVE_URL,
    ALPACA_PAPER_URL,
    ALPACA_SECRET_KEY,
    FEATURE_COLUMNS,
    LIMIT_ENTRY_DIP_PCT,
    MAX_CONSECUTIVE_TICKER_DAYS,
    MAX_DAILY_LOSS_PCT,
    MAX_POSITIONS_PER_DAY,
    MAX_POSITION_PCT,
    MODEL_PATH,
    OUTPUT_DIR,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TRADE_LOG_PATH,
    TRADE_THRESHOLD,
    UNIVERSE,
)

ET = ZoneInfo("America/New_York")


# ── Alpaca Client ────────────────────────────────────────────────────────────

class AlpacaClient:
    """Thin wrapper around Alpaca REST API for order execution."""

    def __init__(self, paper=True):
        self.base_url = ALPACA_PAPER_URL if paper else ALPACA_LIVE_URL
        self.headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        }
        self.paper = paper

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


# ── Trade History (for consecutive-day tracking) ─────────────────────────────

def trade_log_path(label: str) -> Path:
    """Return the trade log path for a given account label."""
    if label == "open":
        return TRADE_LOG_PATH
    return OUTPUT_DIR / f"trade_log_{label}.csv"


def load_recent_trades(days=5, label: str = "open"):
    """Load trade log and return tickers traded in recent days."""
    path = trade_log_path(label)
    if not path.exists():
        return {}

    df = pd.read_csv(path)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "date", "time", "ticker", "direction", "qty", "entry_price",
            "take_profit", "stop_loss", "p_spike", "p_dir", "order_id", "mode",
        ])
        if not file_exists:
            writer.writeheader()
        for t in trades:
            writer.writerow(t)


# ── Signal Generation ────────────────────────────────────────────────────────

def generate_signals():
    """Run the spike detector model and return ranked trade signals."""
    from features import _download_safe, build_single_day_features, compute_macro_features
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

    X = pd.DataFrame(feature_rows).T
    X.columns = FEATURE_COLUMNS
    X = X.fillna(0)

    probs = model.predict(X)

    # Build signal list
    signals = []
    for ticker in UNIVERSE:
        r = probs.loc[ticker]
        if r["p_spike"] >= TRADE_THRESHOLD:
            direction = "LONG" if r["p_up"] > r["p_down"] else "SHORT"
            p_dir = max(r["p_up"], r["p_down"])
            signals.append({
                "ticker": ticker,
                "direction": direction,
                "p_spike": r["p_spike"],
                "p_dir": p_dir,
            })

    # Sort by spike probability descending
    signals.sort(key=lambda s: s["p_spike"], reverse=True)
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
    signals = []
    for _, r in df.iterrows():
        if float(r["p_spike"]) >= TRADE_THRESHOLD:
            direction = "LONG" if r["p_up"] > r["p_down"] else "SHORT"
            signals.append({
                "ticker": r["ticker"],
                "direction": direction,
                "p_spike": float(r["p_spike"]),
                "p_dir": float(max(r["p_up"], r["p_down"])),
            })
    signals.sort(key=lambda s: s["p_spike"], reverse=True)
    return signals


# ── Risk Filters ─────────────────────────────────────────────────────────────

def apply_risk_filters(signals, label: str = "open"):
    """Apply risk management rules to filter and limit signals."""
    # 1. Remove tickers traded too many consecutive days
    streaks = load_recent_trades(label=label)
    filtered = []
    for sig in signals:
        streak = streaks.get(sig["ticker"], 0)
        if streak >= MAX_CONSECUTIVE_TICKER_DAYS:
            print(f"    ⚠ {sig['ticker']} skipped — traded {streak} consecutive days")
            continue
        filtered.append(sig)

    # 2. Cap at MAX_POSITIONS_PER_DAY (already sorted by p_spike)
    if len(filtered) > MAX_POSITIONS_PER_DAY:
        dropped = filtered[MAX_POSITIONS_PER_DAY:]
        for d in dropped:
            print(f"    ⚠ {d['ticker']} skipped — max {MAX_POSITIONS_PER_DAY} positions/day")
        filtered = filtered[:MAX_POSITIONS_PER_DAY]

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

        # Get current price for position sizing
        if not dry_run:
            price = client.get_latest_price(ticker)
            if not price:
                print(f"    ⚠ Could not get price for {ticker}. Skipping.")
                continue
        else:
            price = 100.0  # Placeholder for dry run

        # Limit-entry price: only fill if the market moves favorably by
        # LIMIT_ENTRY_DIP_PCT off the current quote. LONG = dip below current,
        # SHORT = rip above current. Bracket TP/SL anchor to the limit price
        # (the planned fill), not the current quote.
        if direction == "LONG":
            limit_price = round(price * (1 - LIMIT_ENTRY_DIP_PCT), 2)
            take_profit_price = round(limit_price * (1 + TAKE_PROFIT_PCT), 2)
            stop_loss_price = round(limit_price * (1 - STOP_LOSS_PCT), 2)
        else:  # SHORT
            limit_price = round(price * (1 + LIMIT_ENTRY_DIP_PCT), 2)
            take_profit_price = round(limit_price * (1 - TAKE_PROFIT_PCT), 2)
            stop_loss_price = round(limit_price * (1 + STOP_LOSS_PCT), 2)

        # Size off the planned limit price, not the current quote.
        qty = int(max_position_value / limit_price)
        if qty < 1:
            print(f"    ⚠ {ticker} price ${limit_price:.2f} too high for position size ${max_position_value:.0f}. Skipping.")
            continue

        trade_value = qty * limit_price

        if dry_run:
            print(f"    🏷️  {ticker:6s} {direction:5s}  {qty:4d} shares @ limit ${limit_price:.2f}"
                  f"  (ref ${price:.2f}, ${trade_value:,.0f})  TP=${take_profit_price:.2f}  SL=${stop_loss_price:.2f}"
                  f"  P(spike)={sig['p_spike']*100:.0f}%")
            order_id = "DRY-RUN"
        else:
            print(f"    📤 {ticker:6s} {direction:5s}  {qty:4d} shares @ limit ${limit_price:.2f}"
                  f"  (ref ${price:.2f})  TP=${take_profit_price:.2f}  SL=${stop_loss_price:.2f}"
                  f"  P(spike)={sig['p_spike']*100:.0f}%", end=" ", flush=True)

            result = client.place_bracket_order(
                ticker, side, qty, take_profit_price, stop_loss_price,
                limit_price=limit_price,
            )
            if result:
                order_id = result.get("id", "unknown")
                print(f"✅ (order {order_id[:8]})")
            else:
                print("❌ FAILED")
                continue

        trade_records.append({
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "ticker": ticker,
            "direction": direction,
            "qty": qty,
            "entry_price": limit_price,
            "take_profit": take_profit_price,
            "stop_loss": stop_loss_price,
            "p_spike": round(sig["p_spike"], 4),
            "p_dir": round(sig["p_dir"], 4),
            "order_id": order_id,
            "mode": "dry-run" if dry_run else ("paper" if client.paper else "live"),
        })

    return trade_records


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
    if mode != "dry-run" and (not ALPACA_API_KEY or not ALPACA_SECRET_KEY):
        print("\n  ❌ ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env or workflow env.")
        print("     Sign up at https://alpaca.markets for free paper trading.")
        sys.exit(1)

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

    print(f"\n  Found {len(signals)} signals above {TRADE_THRESHOLD*100:.0f}% threshold:")
    for s in signals:
        print(f"    {s['ticker']:6s} {s['direction']:5s}  P(spike)={s['p_spike']*100:.1f}%")

    # Step 2: Apply risk filters
    print(f"\n  Step 2: Applying risk filters...")
    filtered = apply_risk_filters(signals, label=label)

    if not filtered:
        print("\n  All signals filtered out by risk rules. No trades today.")
        print("═" * 65)
        return

    # Step 3: Execute trades
    print(f"\n  Step 3: {'Simulating' if mode == 'dry-run' else 'Placing'} orders...\n")

    if mode == "dry-run":
        trade_records = execute_trades(filtered, None, dry_run=True)
    else:
        client = AlpacaClient(paper=(mode == "paper"))
        trade_records = execute_trades(filtered, client, dry_run=False)

    # Step 4: Log trades
    if trade_records:
        log_trades(trade_records, label=label)
        print(f"\n  📄 {len(trade_records)} trades logged to {trade_log_path(label)}")

    # Step 5: Summary
    print(f"\n  {'=' * 60}")
    print(f"  SUMMARY: {len(trade_records)} orders {'simulated' if mode == 'dry-run' else 'placed'}")
    for t in trade_records:
        print(f"    {t['ticker']:6s} {t['direction']:5s}  {t['qty']:4d} shares"
              f"  TP=${t['take_profit']:.2f}  SL=${t['stop_loss']:.2f}")
    print(f"  {'=' * 60}\n")


if __name__ == "__main__":
    main()
