"""
tune_threshold.py — Walk-forward grid search for TRADE_THRESHOLD on p_spike_trade.

Usage:
    python backtest/tune_threshold.py
    python backtest/tune_threshold.py --train-until 2026-06-13 --test-start 2026-06-16 --test-end 2026-06-20
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from config import (
    DATA_DIR, FEATURE_COLUMNS, MAX_POSITIONS_PER_DAY, TRAINING_DATA_PATH,
    TRADE_THRESHOLD, TUNED_THRESHOLD_PATH, UNIVERSE, get_trade_threshold,
)
from model import TwoStageModel, time_series_split
from stat_mech.ising import sign_returns_from_training


THRESHOLD_GRID = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
MIN_RECALL = 0.25


def _default_dates():
    today = datetime.now()
    days_since_friday = (today.weekday() - 4) % 7 or 7
    last_friday = today - timedelta(days=days_since_friday)
    train_until = last_friday.strftime("%Y-%m-%d")
    test_start = (last_friday + timedelta(days=3)).strftime("%Y-%m-%d")
    test_end = (last_friday + timedelta(days=7)).strftime("%Y-%m-%d")
    return train_until, test_start, test_end


def train_model_fast(train_until: str) -> TwoStageModel:
    df = pd.read_parquet(TRAINING_DATA_PATH)
    df = df[df.index <= pd.Timestamp(train_until)]
    X = df[FEATURE_COLUMNS]
    y = df["_target"].astype(int)
    intraday_ret = df["_intraday_return"]
    tickers = df["_ticker"]
    adaptive_thresh = df.get("_adaptive_threshold")

    X_train, y_train, X_val, y_val = time_series_split(X, y)
    ret_train = intraday_ret.iloc[:len(X_train)]
    thresh_train = adaptive_thresh.iloc[:len(X_train)] if adaptive_thresh is not None else None
    thresh_val = adaptive_thresh.iloc[len(X_train):] if adaptive_thresh is not None else None

    model = TwoStageModel()
    model.train(
        X_train, y_train, X_val, y_val,
        ret_train, intraday_ret.iloc[len(X_train):],
        thresh_train, thresh_val,
    )
    sign_returns = sign_returns_from_training(ret_train, tickers.iloc[:len(X_train)])
    model.fit_stat_mech_layers(X_val, y_val, sign_returns, tickers_val=tickers.iloc[len(X_train):])
    model.retrain_full(X, y, intraday_ret, adaptive_thresh)
    return model


def eval_threshold_on_parquet(model: TwoStageModel, train_until: str, threshold: float) -> dict:
    """Evaluate p_spike_trade on val rows after train_until cutoff."""
    df = pd.read_parquet(TRAINING_DATA_PATH)
    df = df[df.index <= pd.Timestamp(train_until)]
    _, _, X_val, y_val = time_series_split(df[FEATURE_COLUMNS], df["_target"].astype(int))

    probs = model.predict_for_trade(X_val)
    y_spike = (y_val != 1).astype(int).values
    p_trade = probs["p_spike_trade"].values
    pred = (p_trade >= threshold).astype(int)

    dates = pd.Index(X_val.index)
    n_days = max(len(dates.unique()), 1)
    signals = int(pred.sum())

    return {
        "threshold": threshold,
        "precision": float(precision_score(y_spike, pred, zero_division=0)),
        "recall": float(recall_score(y_spike, pred, zero_division=0)),
        "signals_per_day": signals / n_days,
        "total_signals": signals,
    }


def main():
    parser = argparse.ArgumentParser(description="Tune trade threshold (precision-first)")
    parser.add_argument("--train-until", type=str, default=None)
    parser.add_argument("--test-start", type=str, default=None)
    parser.add_argument("--test-end", type=str, default=None)
    args = parser.parse_args()

    if args.train_until:
        train_until = args.train_until
        test_start = args.test_start or ""
        test_end = args.test_end or ""
    else:
        train_until, test_start, test_end = _default_dates()

    if not TRAINING_DATA_PATH.exists():
        print("No training_data.parquet — run retrain first.")
        sys.exit(1)

    print(f"Tuning threshold: train ≤ {train_until}")
    model = train_model_fast(train_until)

    best = None
    print(f"\n{'Thresh':>7} {'Prec':>7} {'Rec':>7} {'Sig/day':>8}")
    print("-" * 35)
    for t in THRESHOLD_GRID:
        m = eval_threshold_on_parquet(model, train_until, t)
        print(f"{m['threshold']:7.2f} {m['precision']:6.1%} {m['recall']:6.1%} {m['signals_per_day']:8.1f}")
        if m["recall"] < MIN_RECALL:
            continue
        if m["signals_per_day"] > MAX_POSITIONS_PER_DAY:
            continue
        if best is None or m["precision"] > best["precision"]:
            best = m

    if best is None:
        best = eval_threshold_on_parquet(model, train_until, TRADE_THRESHOLD)
        print(f"\nNo grid point met constraints; falling back to {TRADE_THRESHOLD}")

    out = {
        "threshold": best["threshold"],
        "expected_precision": best["precision"],
        "expected_recall": best["recall"],
        "signals_per_day": best["signals_per_day"],
        "tuning_window": {"train_until": train_until, "test_start": test_start, "test_end": test_end},
        "tuned_at": datetime.now().isoformat(),
    }
    TUNED_THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TUNED_THRESHOLD_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved tuned threshold {best['threshold']:.2f} → {TUNED_THRESHOLD_PATH}")
    print(f"  Expected precision: {best['precision']:.1%}, recall: {best['recall']:.1%}")


if __name__ == "__main__":
    main()
