"""
check_weekly_precision.py — Fail CI if weekly OOS precision falls below graduated floor.

Usage:
    python backtest/check_weekly_precision.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import OUTPUT_DIR, get_trade_threshold

WEEKLY_DIR = OUTPUT_DIR / "validation_state" / "weekly"
MIN_PRECISION_BASE = 0.25
MIN_PRECISION_WEEK4 = 0.30
MIN_PRECISION_WEEK8 = 0.35
PROB_COL = "p_spike_trade"


def precision_from_csv(path: Path, threshold: float) -> float:
    df = pd.read_csv(path)
    if df.empty or PROB_COL not in df.columns:
        if "p_spike_raw" in df.columns:
            col = "p_spike_raw"
        else:
            return 1.0
    else:
        col = PROB_COL
    actual_spike = df["actual_class"] != "FLAT"
    pred_spike = df[col] >= threshold
    if pred_spike.sum() == 0:
        return 0.0
    tp = ((pred_spike) & actual_spike).sum()
    return float(tp / pred_spike.sum())


def required_precision_floor(n_weeks: int) -> float:
    if n_weeks >= 8:
        return MIN_PRECISION_WEEK8
    if n_weeks >= 4:
        return MIN_PRECISION_WEEK4
    return MIN_PRECISION_BASE


def main():
    if not WEEKLY_DIR.exists():
        print("No weekly validation history — skipping precision gate.")
        sys.exit(0)

    files = sorted(WEEKLY_DIR.glob("validation_*.csv"), key=lambda p: p.stat().st_mtime)
    if len(files) < 1:
        print("No weekly validation CSVs found.")
        sys.exit(0)

    threshold = get_trade_threshold()
    precisions = [(f.name, precision_from_csv(f, threshold)) for f in files]
    floor = required_precision_floor(len(files))

    print("Weekly OOS precision check (trade prob):")
    for name, prec in precisions[-4:]:
        print(f"  {name}: {prec:.1%}")

    recent = precisions[-2:] if len(precisions) >= 2 else precisions[-1:]
    print(f"\n  Required floor ({len(files)} weeks history): {floor:.0%}")

    if len(recent) >= 2 and all(p < floor for _, p in recent):
        print(f"\nFAIL: precision below {floor:.0%} for two consecutive weeks.")
        sys.exit(1)

    print("\nOK: weekly precision gate passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
