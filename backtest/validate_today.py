"""
validate_today.py — Retrain excluding today, predict today, compare with reality.

Usage:
    python backtest/validate_today.py                    # validate today
    python backtest/validate_today.py --date 2026-05-14  # validate a specific date
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from config import FEATURE_COLUMNS, MODEL_PATH, OUTPUT_DIR, SPIKE_THRESHOLD, UNIVERSE


def download_safe(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        return data
    except Exception:
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Validate spike detector on a single day")
    parser.add_argument("--date", type=str, default=None, help="Date to validate (YYYY-MM-DD, default: today)")
    args = parser.parse_args()

    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d")
    else:
        target_date = datetime.now()

    target_str = target_date.strftime("%Y-%m-%d")
    # Train up to the day before
    train_until = (target_date - timedelta(days=1)).strftime("%Y-%m-%d")
    next_day_str = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")

    from features import build_training_dataset, build_single_day_features, _download_safe, compute_macro_features
    from model import TwoStageModel, time_series_split
    from news import FinBERTScorer, FinnhubClient

    print(f"\n{'=' * 65}")
    print(f"  VALIDATE: Train ≤ {train_until}, Predict {target_str}")
    print(f"{'=' * 65}")

    client = FinnhubClient()
    scorer = FinBERTScorer()

    # Phase 1: Train
    print("\n  Phase 1: Training...")
    X, y, intraday_ret, _, adaptive_thresh = build_training_dataset(
        UNIVERSE, client, scorer, end_date_str=train_until
    )

    X_train, y_train, X_val, y_val = time_series_split(X, y)
    ret_train = intraday_ret.iloc[:len(X_train)]
    ret_val = intraday_ret.iloc[len(X_train):]
    thresh_train = adaptive_thresh.iloc[:len(X_train)]
    thresh_val = adaptive_thresh.iloc[len(X_train):]

    model = TwoStageModel()
    model.train(X_train, y_train, X_val, y_val, ret_train, ret_val,
                thresh_train, thresh_val)
    model.retrain_full(X, y, intraday_ret, adaptive_thresh)
    model.save()

    # Phase 2: Predict
    print(f"\n  Phase 2: Predicting {target_str}...")
    start_str = (target_date - timedelta(days=60)).strftime("%Y-%m-%d")
    spy_data = _download_safe("^GSPC", start_str, target_str)
    if not spy_data.empty:
        macro_df = compute_macro_features(spy_data.index, start_str, target_str)
        macro_cache = macro_df.iloc[-1].to_dict() if not macro_df.empty else {}
    else:
        macro_cache = {}

    feature_rows = {}
    for i, ticker in enumerate(UNIVERSE):
        print(f"    [{i+1}/{len(UNIVERSE)}] {ticker}...", end=" ", flush=True)
        row = build_single_day_features(
            ticker, target_date, client, scorer, macro_cache=macro_cache
        )
        feature_rows[ticker] = row
        nc = row.get("overnight_news_count", 0)
        sm = row.get("overnight_sentiment_mean", 0)
        print(f"OK (news={int(nc)}, sent={sm:+.2f})")

    X_pred = pd.DataFrame(feature_rows).T
    X_pred.columns = FEATURE_COLUMNS
    X_pred = X_pred.fillna(0)
    probs = model.predict(X_pred)

    # Phase 3: Compare with actuals
    print(f"\n  Phase 3: Fetching actuals for {target_str}...\n")
    results = []
    for ticker in UNIVERSE:
        actual_data = download_safe(ticker, target_str, next_day_str)
        if actual_data.empty:
            continue

        r = actual_data.iloc[0]
        actual_ret = (r["Close"] - r["Open"]) / r["Open"]
        actual_class = (
            "SPIKE UP" if actual_ret >= SPIKE_THRESHOLD
            else "SPIKE DOWN" if actual_ret <= -SPIKE_THRESHOLD
            else "FLAT"
        )

        p = probs.loc[ticker]
        pred_dir = "UP" if p["p_up"] > p["p_down"] else "DOWN"
        pred_class = "FLAT" if p["p_spike"] < 0.40 else f"SPIKE {pred_dir}"

        results.append({
            "ticker": ticker,
            "p_spike": round(p["p_spike"], 3),
            "p_up": round(p["p_up"], 3),
            "p_down": round(p["p_down"], 3),
            "pred_class": pred_class,
            "actual_return": round(actual_ret, 4),
            "actual_class": actual_class,
            "correct": pred_class == actual_class,
        })

    df = pd.DataFrame(results)
    if df.empty:
        print("  No data available for this date.")
        return

    correct = df["correct"].sum()
    total = len(df)
    print(f"  {'=' * 65}")
    print(f"  RESULTS: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  {'=' * 65}")

    # Spike metrics
    actual_spikes = df[df["actual_class"] != "FLAT"]
    pred_spikes = df[df["pred_class"] != "FLAT"]
    caught = actual_spikes[actual_spikes["pred_class"] != "FLAT"]

    print(f"  Actual spikes: {len(actual_spikes)}")
    print(f"  Predicted spikes: {len(pred_spikes)}")
    if len(actual_spikes) > 0:
        print(f"  Recall: {len(caught)}/{len(actual_spikes)} ({len(caught)/len(actual_spikes)*100:.0f}%)")
    if len(pred_spikes) > 0:
        true_pos = pred_spikes[pred_spikes["actual_class"] != "FLAT"]
        print(f"  Precision: {len(true_pos)}/{len(pred_spikes)} ({len(true_pos)/len(pred_spikes)*100:.0f}%)")

    # Print biggest movers
    big = df[df["actual_return"].abs() >= SPIKE_THRESHOLD].sort_values("actual_return", key=abs, ascending=False)
    if len(big) > 0:
        print(f"\n  {'Ticker':<7} {'Predicted':<14} {'P(spike)':<10} {'Actual':<10} {'Return':>8}")
        print(f"  {'─' * 55}")
        for _, r in big.iterrows():
            check = "+" if r["correct"] else "-"
            print(f"  {r['ticker']:<7} {r['pred_class']:<14} {r['p_spike']*100:6.1f}%   {r['actual_class']:<10} {r['actual_return']*100:>+7.2f}% {check}")

    csv_path = OUTPUT_DIR / f"validation_{target_str}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved to {csv_path}")


if __name__ == "__main__":
    main()
