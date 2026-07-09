"""
daily_validation.py — Post-market validation of the morning's predictions.

Runs after 3:55 PM ET (before market close). Reads today's watchlist (generated at 9:15 AM ET by
spike_detector.py), fetches actual intraday returns via yfinance, computes
per-day metrics, and appends to a rolling history so trends are visible.

State files (persisted across runs via GitHub Actions cache):
  output/validation_state/history.csv   — per-ticker per-day pred vs actual
  output/validation_state/metrics.csv   — per-day aggregate metrics
  output/validation_state/trend.png     — rolling-window metric chart

Usage:
  python backtest/daily_validation.py
  python backtest/daily_validation.py --date 2026-05-21    # backfill
  python backtest/daily_validation.py --watchlist <path>   # explicit input
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from config import OUTPUT_DIR, TRADE_PROB_COLUMN, UNIVERSE, get_trade_threshold
from features import classify_intraday_return, fetch_adaptive_threshold

STATE_DIR = OUTPUT_DIR / "validation_state"
HISTORY_CSV = STATE_DIR / "history.csv"
METRICS_CSV = STATE_DIR / "metrics.csv"
TREND_PNG = STATE_DIR / "trend.png"


def classify(ret: float, ticker: str, date_str: str) -> str:
    threshold = fetch_adaptive_threshold(ticker, date_str)
    return classify_intraday_return(ret, threshold)


def fetch_actuals(date_str: str) -> dict:
    """Fetch today's OHLC for every ticker. yfinance end is exclusive."""
    start = date_str
    end = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    actuals = {}
    for ticker in UNIVERSE:
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, threads=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            if data.empty:
                continue
            row = data.iloc[0]
            open_, close_ = float(row["Open"]), float(row["Close"])
            actuals[ticker] = {
                "open": open_,
                "close": close_,
                "intraday_return": (close_ - open_) / open_ if open_ else 0.0,
            }
        except Exception as e:
            print(f"  ! Failed to fetch {ticker}: {e}")
    return actuals


def fetch_vix_for_date(date_str: str) -> float | None:
    try:
        end = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        data = yf.download("^VIX", start=date_str, end=end, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        if not data.empty:
            return float(data.iloc[0]["Close"])
    except Exception:
        pass
    return None


def compute_trade_prob_metrics(per_ticker: list[dict], threshold: float) -> dict:
    """Precision using p_spike_trade vs p_spike_raw if present in history rows."""
    df = pd.DataFrame(per_ticker)
    actual_spike = df["actual_class"] != "FLAT"
    out = {}
    for col in ["p_spike_trade", "p_spike"]:
        if col not in df.columns:
            continue
        pred = df[col] >= threshold
        if pred.sum() == 0:
            out[f"{col}_precision"] = 0.0
        else:
            out[f"{col}_precision"] = float((pred & actual_spike).sum() / pred.sum())
    return out


def rolling_precision_alert(metrics_df: pd.DataFrame, window: int = 10, floor: float = 0.25) -> str | None:
    if len(metrics_df) < window:
        return None
    recent = metrics_df.tail(window)["spike_precision"].mean()
    if recent < floor:
        return f"ALERT: {window}d rolling spike precision {recent:.1%} < {floor:.0%}"
    return None


def compute_daily_metrics(per_ticker: list[dict]) -> dict:
    df = pd.DataFrame(per_ticker)
    actual_spikes = df[df["actual_class"] != "FLAT"]
    pred_spikes = df[df["pred_class"] != "FLAT"]
    caught = actual_spikes[actual_spikes["pred_class"] != "FLAT"]
    true_pos = pred_spikes[pred_spikes["actual_class"] != "FLAT"]
    dir_correct = caught[caught["pred_class"] == caught["actual_class"]]

    longs = df[df["pred_class"] == "SPIKE UP"]
    shorts = df[df["pred_class"] == "SPIKE DOWN"]

    return {
        "n_tickers": len(df),
        "accuracy": df["correct"].mean() if len(df) else 0.0,
        "n_actual_spikes": len(actual_spikes),
        "n_pred_spikes": len(pred_spikes),
        "spike_recall": len(caught) / len(actual_spikes) if len(actual_spikes) else 0.0,
        "spike_precision": len(true_pos) / len(pred_spikes) if len(pred_spikes) else 0.0,
        "direction_accuracy": len(dir_correct) / len(caught) if len(caught) else 0.0,
        # P&L proxy: avg return on long signals minus avg return on short signals
        # (rough but useful — captures whether signals are profitable in aggregate)
        "long_signal_return": longs["actual_return"].mean() if len(longs) else 0.0,
        "short_signal_return": -shorts["actual_return"].mean() if len(shorts) else 0.0,
    }


def generate_trend_chart(metrics_df: pd.DataFrame):
    if len(metrics_df) < 2:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = metrics_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    window = min(10, len(df))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    ax.plot(df["date"], df["accuracy"] * 100, "o-", alpha=0.5, label="Daily")
    ax.plot(df["date"], df["accuracy"].rolling(window, min_periods=1).mean() * 100,
            "-", lw=2, label=f"{window}d avg")
    ax.set_title("Overall Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(df["date"], df["spike_precision"] * 100, "o-", label="Precision", color="C0")
    ax.plot(df["date"], df["spike_recall"] * 100, "o-", label="Recall", color="C1")
    ax.set_title("Spike Precision & Recall (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(df["date"], df["direction_accuracy"] * 100, "o-", color="purple")
    ax.set_title("Direction Accuracy (on caught spikes, %)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    pnl = df["long_signal_return"].fillna(0) + df["short_signal_return"].fillna(0)
    cumulative = pnl.cumsum() * 100
    ax.plot(df["date"], cumulative, "-", lw=2, color="green" if cumulative.iloc[-1] >= 0 else "red")
    ax.axhline(0, color="black", lw=0.5)
    ax.fill_between(df["date"], 0, cumulative, alpha=0.2,
                    color="green" if cumulative.iloc[-1] >= 0 else "red")
    ax.set_title(f"Cumulative P&L Proxy (signal-aligned, last: {cumulative.iloc[-1]:+.1f}%)")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Spike Detector — {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(TREND_PNG, dpi=100, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None,
                        help="Date to validate (YYYY-MM-DD); default = today")
    parser.add_argument("--watchlist", type=str, default=None,
                        help="Watchlist CSV path; default = output/watchlist_<date>.csv")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    watchlist_path = Path(args.watchlist) if args.watchlist else (OUTPUT_DIR / f"watchlist_{date_str}.csv")

    if not watchlist_path.exists():
        print(f"  ! No watchlist found at {watchlist_path}; skipping (morning run may have failed).")
        sys.exit(0)

    watchlist = pd.read_csv(watchlist_path)
    print(f"\n  Validating watchlist: {watchlist_path}")
    print(f"  Predictions for {len(watchlist)} tickers\n")

    actuals = fetch_actuals(date_str)
    if not actuals:
        print("  ! No actuals fetched — yfinance returned empty for all tickers. Aborting.")
        sys.exit(1)

    vix_level = fetch_vix_for_date(date_str)
    trade_threshold = get_trade_threshold(vix_level)

    per_ticker = []
    for _, r in watchlist.iterrows():
        ticker = r["ticker"]
        if ticker not in actuals:
            continue
        actual_ret = actuals[ticker]["intraday_return"]
        actual_class = classify(actual_ret, ticker, date_str)
        pred_dir = "UP" if r["p_up"] > r["p_down"] else "DOWN"
        prob_col = TRADE_PROB_COLUMN if TRADE_PROB_COLUMN in r.index else "p_spike"
        p_trade = float(r.get(prob_col, r["p_spike"]))
        p_raw = float(r.get("p_spike_raw", r["p_spike"]))
        pred_class = "FLAT" if p_trade < trade_threshold else f"SPIKE {pred_dir}"

        per_ticker.append({
            "date": date_str,
            "ticker": ticker,
            "p_spike": float(r["p_spike"]),
            "p_spike_raw": p_raw,
            "p_spike_trade": p_trade,
            "p_up": float(r["p_up"]),
            "p_down": float(r["p_down"]),
            "pred_class": pred_class,
            "actual_return": actual_ret,
            "actual_class": actual_class,
            "correct": pred_class == actual_class,
        })

    if not per_ticker:
        print("  ! No tickers matched between watchlist and actuals. Aborting.")
        sys.exit(1)

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    new_history = pd.DataFrame(per_ticker)
    if HISTORY_CSV.exists():
        existing = pd.read_csv(HISTORY_CSV)
        existing = existing[existing["date"] != date_str]
        history = pd.concat([existing, new_history], ignore_index=True)
    else:
        history = new_history
    history.to_csv(HISTORY_CSV, index=False)

    today_metrics = compute_daily_metrics(per_ticker)
    today_metrics.update(compute_trade_prob_metrics(per_ticker, trade_threshold))
    today_metrics["date"] = date_str
    today_metrics["trade_threshold"] = trade_threshold
    if vix_level is not None:
        today_metrics["vix"] = vix_level
    try:
        from predict.alpaca_pnl import both_accounts_summary
        today_metrics.update(both_accounts_summary())
    except Exception as e:
        print(f"  ⚠ Alpaca P&L snapshot skipped: {e}")
    if METRICS_CSV.exists():
        metrics_df = pd.read_csv(METRICS_CSV)
        metrics_df = metrics_df[metrics_df["date"] != date_str]
        metrics_df = pd.concat([metrics_df, pd.DataFrame([today_metrics])], ignore_index=True)
    else:
        metrics_df = pd.DataFrame([today_metrics])
    metrics_df = metrics_df.sort_values("date").reset_index(drop=True)
    metrics_df.to_csv(METRICS_CSV, index=False)

    # Print summary
    print(f"  {'=' * 65}")
    print(f"  VALIDATION — {date_str}")
    print(f"  {'=' * 65}")
    n = today_metrics["n_tickers"]
    print(f"  Accuracy:           {today_metrics['accuracy']*100:5.1f}%  ({int(today_metrics['accuracy']*n)}/{n})")
    print(f"  Actual spikes:      {today_metrics['n_actual_spikes']}")
    print(f"  Predicted spikes:   {today_metrics['n_pred_spikes']}")
    print(f"  Spike precision:    {today_metrics['spike_precision']*100:5.1f}%")
    print(f"  Spike recall:       {today_metrics['spike_recall']*100:5.1f}%")
    print(f"  Direction acc:      {today_metrics['direction_accuracy']*100:5.1f}%  (on caught spikes)")
    print(f"  Signal return long: {today_metrics['long_signal_return']*100:+.2f}%")
    print(f"  Signal return short:{today_metrics['short_signal_return']*100:+.2f}%")
    if "p_spike_raw_precision" in today_metrics:
        print(f"  Raw prec @ {trade_threshold:.2f}:   {today_metrics['p_spike_raw_precision']*100:5.1f}%")
    if "p_spike_trade_precision" in today_metrics:
        print(f"  Trade prec @ {trade_threshold:.2f}: {today_metrics['p_spike_trade_precision']*100:5.1f}%")
    if "open_equity" in today_metrics:
        print(f"\n  Alpaca OPEN:    ${today_metrics['open_equity']:,.0f}  "
              f"day P&L ${today_metrics.get('open_daily_pnl', 0):+,.0f}")
    if "delayed_equity" in today_metrics:
        print(f"  Alpaca DELAYED: ${today_metrics['delayed_equity']:,.0f}  "
              f"day P&L ${today_metrics.get('delayed_daily_pnl', 0):+,.0f}")
    if "open_equity" in today_metrics:
        print(f"\n  Alpaca OPEN:    ${today_metrics['open_equity']:,.0f}  "
              f"day P&L ${today_metrics.get('open_daily_pnl', 0):+,.0f}")
    if "delayed_equity" in today_metrics:
        print(f"  Alpaca DELAYED: ${today_metrics['delayed_equity']:,.0f}  "
              f"day P&L ${today_metrics.get('delayed_daily_pnl', 0):+,.0f}")

    alert = rolling_precision_alert(metrics_df)
    if alert:
        print(f"\n  ⚠ {alert}")

    # Side-by-side: predicted spikes vs reality
    pred_spikes_today = [r for r in per_ticker if r["pred_class"] != "FLAT"]
    if pred_spikes_today:
        print(f"\n  Predicted spikes (today):")
        print(f"  {'Ticker':<7} {'Pred':<13} {'P(spike)':<10} {'Actual':<13} {'Return':>8}  ok?")
        print(f"  {'-' * 60}")
        for r in sorted(pred_spikes_today, key=lambda x: -x["p_spike"]):
            check = "+" if r["correct"] else "-"
            print(f"  {r['ticker']:<7} {r['pred_class']:<13} {r['p_spike']*100:6.1f}%   "
                  f"{r['actual_class']:<13} {r['actual_return']*100:>+7.2f}%  {check}")

    generate_trend_chart(metrics_df)
    print(f"\n  History: {HISTORY_CSV}")
    print(f"  Metrics: {METRICS_CSV}")
    if TREND_PNG.exists():
        print(f"  Trend:   {TREND_PNG}")
    print(f"  ({len(metrics_df)} days of validation history)")


if __name__ == "__main__":
    main()
