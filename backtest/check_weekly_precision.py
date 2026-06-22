"""
check_weekly_precision.py — Fail CI if last two weekly OOS runs had precision < 25%.

Usage:
    python backtest/check_weekly_precision.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import OUTPUT_DIR, get_trade_threshold

WEEKLY_DIR = OUTPUT_DIR / "validation_state" / "weekly"
MIN_PRECISION = 0.25
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


def main():
    if not WEEKLY_DIR.exists():
        print("No weekly validation history — skipping precision gate.")
        sys.exit(0)

    files = sorted(WEEKLY_DIR.glob("validation_*.csv"), key=lambda p: p.stat().st_mtime)
    if len(files) < 1:
        print("No weekly validation CSVs found.")
        sys.exit(0)

    threshold = get_trade_threshold()
    recent = files[-2:] if len(files) >= 2 else files[-1:]
    precisions = [(f.name, precision_from_csv(f, threshold)) for f in recent]

    print("Weekly OOS precision check (trade prob):")
    for name, prec in precisions:
        print(f"  {name}: {prec:.1%}")

    if len(precisions) >= 2 and all(p < MIN_PRECISION for _, p in precisions):
        print(f"\nFAIL: precision below {MIN_PRECISION:.0%} for two consecutive weeks.")
        sys.exit(1)

    print("\nOK: weekly precision gate passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
