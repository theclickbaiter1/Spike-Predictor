"""
tune_threshold.py — Walk-forward grid search for TRADE_THRESHOLD on p_spike_trade.

Includes VIX regime buckets and optional nested OOS summary.

Usage:
    python backtest/tune_threshold.py
    python backtest/tune_threshold.py --train-until 2026-06-13 --nested
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from datetime import datetime, timedelta

from config import (
    MAX_POSITIONS_PER_DAY,
    TRADE_THRESHOLD,
    TRAINING_DATA_PATH,
    TUNED_THRESHOLD_PATH,
)
from walkforward_utils import (
    THRESHOLD_GRID,
    eval_probs_slice,
    load_training_df,
    pick_regime_thresholds,
    run_nested_walkforward,
    save_tuned_threshold_config,
    train_model_fast,
)
from model import time_series_split


def _default_dates():
    today = datetime.now()
    days_since_friday = (today.weekday() - 4) % 7 or 7
    last_friday = today - timedelta(days=days_since_friday)
    train_until = last_friday.strftime("%Y-%m-%d")
    test_start = (last_friday + timedelta(days=3)).strftime("%Y-%m-%d")
    test_end = (last_friday + timedelta(days=7)).strftime("%Y-%m-%d")
    return train_until, test_start, test_end


def main():
    parser = argparse.ArgumentParser(description="Tune trade threshold (precision-first)")
    parser.add_argument("--train-until", type=str, default=None)
    parser.add_argument("--test-start", type=str, default=None)
    parser.add_argument("--test-end", type=str, default=None)
    parser.add_argument("--nested", action="store_true", help="Run nested walk-forward summary")
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

    df = load_training_df(train_until)
    from config import FEATURE_COLUMNS
    _, _, X_val, y_val = time_series_split(df[FEATURE_COLUMNS], df["_target"].astype(int))

    print(f"\n{'Thresh':>7} {'Prec':>7} {'Rec':>7} {'Sig/day':>8}")
    print("-" * 35)
    for t in THRESHOLD_GRID:
        m = eval_probs_slice(model, X_val, y_val, t)
        print(f"{m['threshold']:7.2f} {m['precision']:6.1%} {m['recall']:6.1%} {m['signals_per_day']:8.1f}")

    regime = pick_regime_thresholds(model, X_val, y_val)
    nested_summary = None
    if args.nested:
        print("\nRunning nested walk-forward (may take several minutes)...")
        _, nested_summary = run_nested_walkforward(n_oos_weeks=4, tune_days=20)
        print(f"  Nested mean OOS precision: {nested_summary['mean_oos_precision']:.1%}")
        print(f"  Nested val→OOS gap:       {nested_summary['mean_val_oos_gap']:+.1%}")

    path = save_tuned_threshold_config(
        regime,
        train_until,
        test_start,
        test_end,
        nested_summary=nested_summary,
    )
    print(f"\nSaved tuned thresholds → {path}")
    print(f"  Default: {regime['default']:.2f}")
    print(f"  VIX low (<15):  {regime['vix_low']['threshold']:.2f}")
    print(f"  VIX mid (15-25): {regime['vix_mid']['threshold']:.2f}")
    print(f"  VIX high (≥25): {regime['vix_high']['threshold']:.2f}")
    print(f"  Expected precision: {regime.get('expected_precision', 0):.1%}, "
          f"recall: {regime.get('expected_recall', 0):.1%}")


if __name__ == "__main__":
    main()
