"""
backtest_strategy.py — Simulate a trading strategy from spike predictions.

Strategy:
  - Each day, the model predicts spike probabilities for all tickers.
  - Enter a position at market open for any ticker with P(spike) > threshold.
  - Direction: go LONG if P(up) > P(down), SHORT if P(down) > P(up).
  - Exit rule: sell when (pct_gain * hours_held) >= 10, or at market close.
  - Equal capital allocation across open positions each day.
  - Compare cumulative returns vs S&P 500 buy-and-hold.

Fast mode: builds all features in one pass, only downloads hourly data for actual trades.

Usage:
    python backtest/backtest_strategy.py
    python backtest/backtest_strategy.py --train-until 2025-05-31 --trade-start 2025-06-02 --trade-end 2026-05-15
    python backtest/backtest_strategy.py --threshold 0.5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf

from config import FEATURE_COLUMNS, OUTPUT_DIR, SPIKE_THRESHOLD, UNIVERSE


def download_safe(ticker, start, end, interval="1d"):
    try:
        data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        return data
    except Exception:
        return pd.DataFrame()


def simulate_trade_intraday(ticker, date_str, direction, hourly_cache):
    """
    Simulate a single trade using hourly candles.
    Enter at market open, exit when pct_gain * hours >= 10 or at close.
    """
    key = (ticker, date_str)
    if key not in hourly_cache:
        next_day = (pd.Timestamp(date_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        hourly_cache[key] = download_safe(ticker, date_str, next_day, interval="1h")

    hourly = hourly_cache[key]
    if hourly.empty or len(hourly) < 2:
        return 0.0

    open_price = float(hourly.iloc[0]["Open"])
    if open_price <= 0:
        return 0.0

    for i in range(1, len(hourly)):
        price = float(hourly.iloc[i]["Close"])
        if direction == "LONG":
            pct_gain = (price - open_price) / open_price * 100
        else:
            pct_gain = (open_price - price) / open_price * 100

        hours = i
        score = pct_gain * hours

        if score >= 10:
            if direction == "LONG":
                return (price - open_price) / open_price
            else:
                return (open_price - price) / open_price

        if pct_gain < -3:
            if direction == "LONG":
                return (price - open_price) / open_price
            else:
                return (open_price - price) / open_price

    close_price = float(hourly.iloc[-1]["Close"])
    if direction == "LONG":
        return (close_price - open_price) / open_price
    else:
        return (open_price - close_price) / open_price


def run_backtest(train_until, trade_start, trade_end, threshold):
    from features import build_training_dataset
    from model import TwoStageModel, time_series_split
    from news import FinBERTScorer, FinnhubClient

    print(f"\n{'=' * 65}")
    print(f"  BACKTEST: Train ≤ {train_until}, Trade {trade_start} → {trade_end}")
    print(f"  Spike threshold: {threshold*100:.0f}%")
    print(f"{'=' * 65}")

    # ── Phase 1: Build ALL features in one pass ──
    print("\n  Phase 1: Building features for full period...")
    client = FinnhubClient()
    scorer = FinBERTScorer()

    X_all, y_all, ret_all, tickers_all, adaptive_all = build_training_dataset(
        UNIVERSE, client, scorer, end_date_str=trade_end
    )

    # Split by date
    train_cutoff = pd.Timestamp(train_until)
    trade_start_ts = pd.Timestamp(trade_start)

    train_mask = X_all.index <= train_cutoff
    trade_mask = X_all.index >= trade_start_ts

    X_train_full = X_all[train_mask]
    y_train_full = y_all[train_mask]
    ret_train_full = ret_all[train_mask]
    adaptive_train_full = adaptive_all[train_mask]

    X_trade = X_all[trade_mask].copy()
    ret_trade = ret_all[trade_mask]
    tickers_trade = tickers_all[trade_mask]

    print(f"\n  Training rows: {len(X_train_full)}")
    print(f"  Trading rows:  {len(X_trade)} ({len(X_trade) // len(UNIVERSE)} days approx)")

    # ── Phase 2: Train model ──
    print("\n  Phase 2: Training model...")
    X_tr, y_tr, X_val, y_val = time_series_split(X_train_full, y_train_full)
    ret_tr = ret_train_full.iloc[:len(X_tr)]
    ret_val = ret_train_full.iloc[len(X_tr):]
    thresh_tr = adaptive_train_full.iloc[:len(X_tr)]
    thresh_val = adaptive_train_full.iloc[len(X_tr):]

    model = TwoStageModel()
    model.train(X_tr, y_tr, X_val, y_val, ret_tr, ret_val, thresh_tr, thresh_val)
    model.retrain_full(X_train_full, y_train_full, ret_train_full, adaptive_train_full)

    print("\n  Top 10 Spike Feature Importances:")
    importance = model.get_spike_feature_importance()
    for feat, score in importance.head(10).items():
        bar = "█" * int(score * 100)
        print(f"    {feat:30s} {score:.3f} {bar}")
    print()

    # ── Phase 3: Predict all trade rows at once (fast) ──
    print("  Phase 3: Predicting all trade rows...")
    probs = model.predict(X_trade)
    probs["ticker"] = tickers_trade.values
    probs["date"] = X_trade.index
    probs["actual_return"] = ret_trade.values

    # ── Phase 4: Walk-forward trading simulation ──
    trade_dates = sorted(X_trade.index.unique())
    print(f"  Phase 4: Simulating {len(trade_dates)} trading days...\n")

    hourly_cache = {}
    daily_returns = []
    trade_log = []
    total_trades = 0
    winning_trades = 0

    for day_idx, trade_date in enumerate(trade_dates):
        date_str = trade_date.strftime("%Y-%m-%d")

        if day_idx % 20 == 0:
            print(f"    [{day_idx+1}/{len(trade_dates)}] {date_str}...")

        day_probs = probs[probs["date"] == trade_date]

        # Select trades above threshold
        trades_today = []
        for _, row in day_probs.iterrows():
            if row["p_spike"] >= threshold:
                direction = "LONG" if row["p_up"] > row["p_down"] else "SHORT"
                trades_today.append({
                    "ticker": row["ticker"],
                    "direction": direction,
                    "p_spike": row["p_spike"],
                })

        if not trades_today:
            daily_returns.append({"date": date_str, "return": 0.0, "n_trades": 0})
            continue

        # Simulate each trade with hourly data
        trade_returns = []
        for trade in trades_today:
            ret = simulate_trade_intraday(
                trade["ticker"], date_str, trade["direction"], hourly_cache
            )
            trade_returns.append(ret)
            total_trades += 1
            if ret > 0:
                winning_trades += 1
            trade_log.append({
                "date": date_str,
                "ticker": trade["ticker"],
                "direction": trade["direction"],
                "p_spike": round(trade["p_spike"], 3),
                "return": round(ret, 4),
            })

        avg_ret = np.mean(trade_returns) if trade_returns else 0.0
        daily_returns.append({
            "date": date_str,
            "return": avg_ret,
            "n_trades": len(trades_today),
        })

    # ── Phase 5: Results ──
    print(f"\n  Phase 5: Computing results...")

    df_daily = pd.DataFrame(daily_returns)
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily = df_daily.set_index("date")
    df_daily["cumulative"] = (1 + df_daily["return"]).cumprod()

    # S&P 500 buy-and-hold
    spy = download_safe("^GSPC", trade_start, trade_end)
    if not spy.empty:
        spy_ret = spy["Close"].pct_change().fillna(0)
        spy_cum = (1 + spy_ret).cumprod()
        spy_cum = spy_cum.reindex(df_daily.index, method="ffill")
    else:
        spy_cum = pd.Series(1.0, index=df_daily.index)

    strategy_total = (df_daily["cumulative"].iloc[-1] - 1) * 100
    spy_total = (spy_cum.iloc[-1] - 1) * 100
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    trading_days_active = (df_daily["n_trades"] > 0).sum()
    avg_trades_per_day = df_daily["n_trades"].mean()

    cum = df_daily["cumulative"]
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min() * 100

    daily_rets = df_daily["return"]
    sharpe = (daily_rets.mean() / daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0

    print(f"\n  {'=' * 65}")
    print(f"  BACKTEST RESULTS")
    print(f"  {'=' * 65}")
    print(f"  Period:              {trade_start} → {trade_end}")
    print(f"  Strategy return:     {strategy_total:+.1f}%")
    print(f"  S&P 500 return:      {spy_total:+.1f}%")
    print(f"  Alpha:               {strategy_total - spy_total:+.1f}%")
    print(f"  Sharpe ratio:        {sharpe:.2f}")
    print(f"  Max drawdown:        {max_dd:.1f}%")
    print(f"  Total trades:        {total_trades}")
    print(f"  Win rate:            {win_rate:.1f}%")
    print(f"  Active trading days: {trading_days_active}/{len(df_daily)}")
    print(f"  Avg trades/day:      {avg_trades_per_day:.1f}")
    print(f"  {'=' * 65}")

    # ── Phase 6: Plot ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(df_daily.index, df_daily["cumulative"], label=f"Strategy ({strategy_total:+.1f}%)",
             color="#2563eb", linewidth=2)
    ax1.plot(spy_cum.index, spy_cum, label=f"S&P 500 ({spy_total:+.1f}%)",
             color="#9ca3af", linewidth=1.5, linestyle="--")
    ax1.axhline(y=1.0, color="gray", linestyle="-", alpha=0.3)
    ax1.fill_between(df_daily.index, df_daily["cumulative"], 1.0,
                     where=df_daily["cumulative"] >= 1.0, alpha=0.1, color="#2563eb")
    ax1.fill_between(df_daily.index, df_daily["cumulative"], 1.0,
                     where=df_daily["cumulative"] < 1.0, alpha=0.1, color="#ef4444")
    ax1.set_ylabel("Cumulative Return")
    ax1.set_title(
        f"Spike Detector Strategy vs S&P 500\n"
        f"Train ≤ {train_until} | Trade {trade_start} → {trade_end} | "
        f"Threshold {threshold*100:.0f}% | Sharpe {sharpe:.2f}",
        fontweight="bold",
    )
    ax1.legend(fontsize=11)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.grid(alpha=0.2)

    ax2.fill_between(drawdown.index, drawdown * 100, 0, color="#ef4444", alpha=0.4)
    ax2.plot(drawdown.index, drawdown * 100, color="#ef4444", linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "backtest_strategy.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\n  Chart saved to {plot_path}")

    df_trades = pd.DataFrame(trade_log)
    trades_path = OUTPUT_DIR / "backtest_trades.csv"
    df_trades.to_csv(trades_path, index=False)
    print(f"  Trade log saved to {trades_path}")

    daily_path = OUTPUT_DIR / "backtest_daily.csv"
    df_daily.to_csv(daily_path)
    print(f"  Daily returns saved to {daily_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest spike detector trading strategy")
    parser.add_argument("--train-until", type=str, default="2025-05-31")
    parser.add_argument("--trade-start", type=str, default="2025-06-02")
    parser.add_argument("--trade-end", type=str, default="2026-05-15")
    parser.add_argument("--threshold", type=float, default=0.40)
    args = parser.parse_args()

    run_backtest(args.train_until, args.trade_start, args.trade_end, args.threshold)
