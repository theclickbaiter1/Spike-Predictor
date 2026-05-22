"""
backtest.py — Backtesting for the 2-stage Spike Detector.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

from config import FEATURE_COLUMNS, LABEL_NAMES, MODEL_PATH, OUTPUT_DIR, TRAINING_DATA_PATH


def run_backtest(months=6):
    from model import TwoStageModel
    from pathlib import Path

    print(f"\n{'='*65}\n  SPIKE DETECTOR v2 — BACKTEST\n{'='*65}")

    s1_path = str(MODEL_PATH).replace(".json", "_s1.json")
    if not Path(s1_path).exists():
        print("\n  ❌ No model. Run --retrain first.")
        sys.exit(1)

    model = TwoStageModel()
    model.load()

    if not TRAINING_DATA_PATH.exists():
        print("\n  ❌ No training data. Run --retrain first.")
        sys.exit(1)

    df = pd.read_parquet(TRAINING_DATA_PATH)
    y_all = df["_target"].astype(int)
    X_all = df[FEATURE_COLUMNS]

    dates = pd.DatetimeIndex(df.index)
    cutoff = dates.max() - pd.DateOffset(months=months)
    bt_mask = dates >= cutoff
    X_bt, y_bt = X_all[bt_mask].fillna(0), y_all[bt_mask]

    if len(X_bt) == 0:
        print(f"\n  ❌ No data in backtest window.")
        sys.exit(1)

    print(f"  Window: {dates[bt_mask].min().date()} → {dates[bt_mask].max().date()}")
    print(f"  Samples: {len(X_bt)}")

    probs = model.predict(X_bt)

    # Binary spike evaluation
    actual_spike = (y_bt != 1).astype(int)
    pred_spike = (probs["p_spike"] >= 0.40).astype(int)

    prec = precision_score(actual_spike, pred_spike, zero_division=0)
    rec = recall_score(actual_spike, pred_spike, zero_division=0)
    f1 = f1_score(actual_spike, pred_spike, zero_division=0)

    print(f"\n{'─'*65}")
    print(f"  SPIKE DETECTION (binary)")
    print(f"{'─'*65}")
    print(f"  Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")

    # Confidence tiers
    print(f"\n{'─'*65}")
    print(f"  CONFIDENCE TIERS")
    print(f"{'─'*65}")
    for thresh, label in [(0.60, "HIGH >60%"), (0.40, "MOD 40-60%")]:
        if thresh == 0.60:
            mask = probs["p_spike"] >= thresh
        else:
            mask = (probs["p_spike"] >= thresh) & (probs["p_spike"] < 0.60)
        if mask.sum() == 0:
            print(f"  {label}: No predictions")
            continue
        correct = actual_spike[mask].sum()
        total = mask.sum()
        print(f"  {label}: {total} predictions, {correct} actual spikes ({correct/total:.1%})")

    # Direction accuracy (among correctly detected spikes)
    detected_spikes = (probs["p_spike"] >= 0.40) & (actual_spike == 1)
    if detected_spikes.sum() > 0:
        pred_up = probs.loc[detected_spikes, "p_up"] > probs.loc[detected_spikes, "p_down"]
        actual_up = y_bt[detected_spikes] == 2
        dir_acc = (pred_up.values == actual_up.values).mean()
        print(f"\n  Direction accuracy (among caught spikes): {dir_acc:.1%}")

    # Confusion matrix
    y_pred_3class = pd.Series(1, index=X_bt.index)  # default flat
    spike_pred = probs["p_spike"] >= 0.40
    up_pred = probs["p_up"] > probs["p_down"]
    y_pred_3class[spike_pred & up_pred] = 2
    y_pred_3class[spike_pred & ~up_pred] = 0

    cm = confusion_matrix(y_bt, y_pred_3class, labels=[0, 1, 2])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)", fontweight="bold")
    axes[0].set_ylabel("Actual"); axes[0].set_xlabel("Predicted")

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Oranges", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalized)", fontweight="bold")
    axes[1].set_ylabel("Actual"); axes[1].set_xlabel("Predicted")

    plt.suptitle(f"Spike Detector v2 Backtest — Last {months} Months\nSpike F1: {f1:.3f}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUTPUT_DIR / "backtest_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  📊 Saved to {out}")
    print(f"\n  ✅ Backtest complete.\n{'='*65}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=6)
    run_backtest(parser.parse_args().months)


if __name__ == "__main__":
    main()
