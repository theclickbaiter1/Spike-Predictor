"""
nested_walkforward.py — Nested walk-forward threshold evaluation.

Tunes threshold on fold N validation window, evaluates on fold N+1 OOS week
(never seen during tuning). Reports per-fold and aggregate OOS precision.

Usage:
    python backtest/nested_walkforward.py
    python backtest/nested_walkforward.py --oos-weeks 8 --tune-days 20
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import pandas as pd

from walkforward_utils import run_nested_walkforward
from config import OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description="Nested walk-forward OOS evaluation")
    parser.add_argument("--oos-weeks", type=int, default=8, help="Number of OOS weeks")
    parser.add_argument("--tune-days", type=int, default=20, help="Tune window (trading days)")
    parser.add_argument("--save", action="store_true", help="Save results CSV to output/")
    args = parser.parse_args()

    print(f"\nNested walk-forward: {args.oos_weeks} OOS weeks, {args.tune_days}-day tune window\n")
    results, summary = run_nested_walkforward(
        n_oos_weeks=args.oos_weeks,
        tune_days=args.tune_days,
    )

    print(f"{'OOS week':<22} {'Thresh':>7} {'TunePrec':>9} {'OOSPrec':>8} {'OOSRec':>7} {'Gap':>7}")
    print("-" * 65)
    for r in results:
        week = f"{r['oos_start']} → {r['oos_end']}"
        print(
            f"{week:<22} {r['tuned_threshold']:7.2f} "
            f"{r['tune_precision']:8.1%} {r['oos_precision']:7.1%} "
            f"{r['oos_recall']:6.1%} {r['val_oos_gap']:+6.1%}"
        )

    print("\n" + "=" * 65)
    print(f"  Folds:              {summary['n_folds']}")
    print(f"  Mean OOS precision: {summary['mean_oos_precision']:.1%}")
    print(f"  Median OOS prec:    {summary['median_oos_precision']:.1%}")
    print(f"  Mean val→OOS gap:   {summary['mean_val_oos_gap']:+.1%}")
    print(f"  Median threshold:   {summary['median_threshold']:.2f} ± {summary['std_threshold']:.2f}")
    print("=" * 65 + "\n")

    if args.save:
        out_dir = OUTPUT_DIR / "validation_state" / "nested"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out_dir / "nested_walkforward.csv", index=False)
        with open(out_dir / "nested_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved to {out_dir}/")


if __name__ == "__main__":
    main()
