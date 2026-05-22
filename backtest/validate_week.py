"""
validate_week.py — Train on data up to a cutoff date, predict the following week, compare vs actuals.

Usage:
    python backtest/validate_week.py                           # defaults: train ≤ last Friday, test this week
    python backtest/validate_week.py --train-until 2026-05-09 --test-start 2026-05-11 --test-end 2026-05-15
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from config import FEATURE_COLUMNS, MODEL_PATH, SPIKE_THRESHOLD, UNIVERSE


# ── Helpers ──────────────────────────────────────────────────────────────────

def download_safe(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        return data
    except Exception:
        return pd.DataFrame()


def classify_return(ret):
    if ret >= SPIKE_THRESHOLD:
        return "SPIKE UP"
    elif ret <= -SPIKE_THRESHOLD:
        return "SPIKE DOWN"
    else:
        return "FLAT"


# ── Phase 1: Train on data up to May 9 ──────────────────────────────────────

def train_model(train_until):
    from features import build_training_dataset
    from model import TwoStageModel, time_series_split
    from news import FinBERTScorer, FinnhubClient

    print("=" * 65)
    print(f"  PHASE 1: TRAINING (data up to {train_until})")
    print("=" * 65)

    client = FinnhubClient()
    scorer = FinBERTScorer()

    X, y, intraday_ret, _ = build_training_dataset(
        UNIVERSE, client, scorer, end_date_str=train_until
    )

    print(f"\n  Dataset: {len(X)} rows, {len(X.columns)} features")

    X_train, y_train, X_val, y_val = time_series_split(X, y)
    ret_train = intraday_ret.iloc[: len(X_train)]
    ret_val = intraday_ret.iloc[len(X_train) :]

    model = TwoStageModel()
    model.train(X_train, y_train, X_val, y_val, ret_train, ret_val)

    print("\n  Top 10 Spike Feature Importances:")
    importance = model.get_spike_feature_importance()
    for feat, score in importance.head(10).items():
        bar = "█" * int(score * 100)
        print(f"    {feat:30s} {score:.3f} {bar}")

    model.retrain_full(X, y, intraday_ret)
    model.save()
    print("\n  ✅ Model saved.\n")
    return model


# ── Phase 2: Predict May 11-15 and compare vs actuals ───────────────────────

def predict_and_compare(model, test_start, test_end):
    from features import (
        _download_safe,
        build_single_day_features,
        compute_macro_features,
    )
    from news import FinBERTScorer, FinnhubClient

    print("=" * 65)
    print(f"  PHASE 2: PREDICT {test_start} to {test_end} vs ACTUALS")
    print("=" * 65)

    client = FinnhubClient()
    scorer = FinBERTScorer()

    test_dates = pd.bdate_range(test_start, test_end)

    all_results = []

    for pred_date in test_dates:
        pred_dt = pred_date.to_pydatetime()
        date_str = pred_dt.strftime("%Y-%m-%d")
        next_day_str = (pred_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        start_str = (pred_dt - timedelta(days=60)).strftime("%Y-%m-%d")

        print(f"\n  ── {date_str} ──")

        # Compute macro once per day
        spy_data = _download_safe("^GSPC", start_str, date_str)
        if not spy_data.empty:
            macro_df = compute_macro_features(spy_data.index, start_str, date_str)
            macro_cache = macro_df.iloc[-1].to_dict() if not macro_df.empty else {}
        else:
            macro_cache = {}

        # Build features for each ticker
        feature_rows = {}
        for ticker in UNIVERSE:
            row = build_single_day_features(
                ticker, pred_dt, client, scorer, macro_cache=macro_cache
            )
            feature_rows[ticker] = row

        X_pred = pd.DataFrame(feature_rows).T
        X_pred.columns = FEATURE_COLUMNS
        X_pred = X_pred.fillna(0)
        probs = model.predict(X_pred)

        # Fetch actuals for this date
        for ticker in UNIVERSE:
            actual_data = download_safe(ticker, date_str, next_day_str)
            if actual_data.empty or len(actual_data) == 0:
                continue

            row = actual_data.iloc[0]
            actual_ret = (row["Close"] - row["Open"]) / row["Open"]

            p = probs.loc[ticker]
            pred_dir = "UP" if p["p_up"] > p["p_down"] else "DOWN"
            pred_class = "FLAT" if p["p_spike"] < 0.40 else f"SPIKE {pred_dir}"
            actual_class = classify_return(actual_ret)

            all_results.append(
                {
                    "date": date_str,
                    "ticker": ticker,
                    "p_spike": round(p["p_spike"], 3),
                    "p_up": round(p["p_up"], 3),
                    "p_down": round(p["p_down"], 3),
                    "pred_class": pred_class,
                    "actual_return": round(actual_ret, 4),
                    "actual_class": actual_class,
                    "correct": pred_class == actual_class,
                }
            )

        day_results = [r for r in all_results if r["date"] == date_str]
        correct = sum(1 for r in day_results if r["correct"])
        total = len(day_results)
        print(f"  {correct}/{total} correct ({correct / total * 100:.0f}%)" if total else "  No data")

    return pd.DataFrame(all_results)


# ── Phase 3: Print summary ──────────────────────────────────────────────────

def print_summary(df):
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)

    # Overall accuracy
    total = len(df)
    correct = df["correct"].sum()
    print(f"\n  Overall: {correct}/{total} ({correct / total * 100:.1f}%)")

    # Per-day accuracy
    print("\n  By Day:")
    for date, grp in df.groupby("date"):
        c = grp["correct"].sum()
        n = len(grp)
        print(f"    {date}: {c}/{n} ({c / n * 100:.0f}%)")

    # Spike detection performance
    actual_spikes = df[df["actual_class"] != "FLAT"]
    pred_spikes = df[df["pred_class"] != "FLAT"]

    print(f"\n  Actual spikes in period: {len(actual_spikes)}")
    print(f"  Predicted spikes: {len(pred_spikes)}")

    if len(actual_spikes) > 0:
        caught = actual_spikes[actual_spikes["pred_class"] != "FLAT"]
        print(f"  Spikes caught (recall): {len(caught)}/{len(actual_spikes)} ({len(caught) / len(actual_spikes) * 100:.0f}%)")

    if len(pred_spikes) > 0:
        true_pos = pred_spikes[pred_spikes["actual_class"] != "FLAT"]
        print(f"  Spike precision: {len(true_pos)}/{len(pred_spikes)} ({len(true_pos) / len(pred_spikes) * 100:.0f}%)")

        # Direction accuracy (among correctly predicted spikes)
        if len(true_pos) > 0:
            dir_correct = true_pos[true_pos["pred_class"] == true_pos["actual_class"]]
            print(f"  Direction accuracy (on true spikes): {len(dir_correct)}/{len(true_pos)} ({len(dir_correct) / len(true_pos) * 100:.0f}%)")

    # Side-by-side: show biggest movers
    print(f"\n  {'─' * 63}")
    print(f"  TOP MOVERS vs PREDICTIONS (|return| > 3%)")
    print(f"  {'─' * 63}")
    big_movers = df[df["actual_return"].abs() >= SPIKE_THRESHOLD].sort_values(
        "actual_return", key=abs, ascending=False
    )
    if len(big_movers) == 0:
        print("  No moves > 3% in this period.")
    else:
        print(f"  {'Date':<12} {'Ticker':<7} {'Predicted':<14} {'P(spike)':<10} {'Actual':<10} {'Return':>8} {'✓?'}")
        print(f"  {'─' * 63}")
        for _, r in big_movers.iterrows():
            check = "✅" if r["correct"] else "❌"
            print(
                f"  {r['date']:<12} {r['ticker']:<7} {r['pred_class']:<14} "
                f"{r['p_spike'] * 100:6.1f}%   {r['actual_class']:<10} {r['actual_return'] * 100:>+7.2f}%  {check}"
            )

    # Full side-by-side for all predicted spikes
    print(f"\n  {'─' * 63}")
    print(f"  ALL PREDICTED SPIKES")
    print(f"  {'─' * 63}")
    if len(pred_spikes) == 0:
        print("  No spikes predicted.")
    else:
        pred_spikes_sorted = pred_spikes.sort_values("p_spike", ascending=False)
        print(f"  {'Date':<12} {'Ticker':<7} {'Predicted':<14} {'P(spike)':<10} {'Actual':<10} {'Return':>8} {'✓?'}")
        print(f"  {'─' * 63}")
        for _, r in pred_spikes_sorted.iterrows():
            check = "✅" if r["correct"] else "❌"
            print(
                f"  {r['date']:<12} {r['ticker']:<7} {r['pred_class']:<14} "
                f"{r['p_spike'] * 100:6.1f}%   {r['actual_class']:<10} {r['actual_return'] * 100:>+7.2f}%  {check}"
            )

    # Save CSV
    csv_path = f"output/validation_{df['date'].min()}_{df['date'].max()}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  📄 Full results saved to {csv_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def _default_dates():
    """Default: train up to last Friday, test this Monday-Friday."""
    today = datetime.now()
    # Find last Friday
    days_since_friday = (today.weekday() - 4) % 7
    if days_since_friday == 0 and today.hour < 16:
        days_since_friday = 7
    last_friday = today - timedelta(days=days_since_friday)
    train_until = last_friday.strftime("%Y-%m-%d")
    # Test week = Monday after that Friday
    test_start = (last_friday + timedelta(days=3)).strftime("%Y-%m-%d")
    test_end = (last_friday + timedelta(days=7)).strftime("%Y-%m-%d")
    return train_until, test_start, test_end


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate spike detector on a date range")
    parser.add_argument("--train-until", type=str, default=None, help="Last date for training data (YYYY-MM-DD)")
    parser.add_argument("--test-start", type=str, default=None, help="First test date (YYYY-MM-DD)")
    parser.add_argument("--test-end", type=str, default=None, help="Last test date (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.train_until and args.test_start and args.test_end:
        train_until, test_start, test_end = args.train_until, args.test_start, args.test_end
    else:
        train_until, test_start, test_end = _default_dates()
        print(f"  Using default dates: train ≤ {train_until}, test {test_start} → {test_end}")

    model = train_model(train_until)
    results = predict_and_compare(model, test_start, test_end)
    if len(results) > 0:
        print_summary(results)
    else:
        print("\n  ❌ No results collected. Check data availability.")
