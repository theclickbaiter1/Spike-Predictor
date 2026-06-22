"""
compare_calibrators.py — Compare raw XGBoost vs Boltzmann-calibrated probabilities.

Metrics: NLL, Brier score, reliability diagram (saved to output/).

Usage:
    python backtest/compare_calibrators.py
    python backtest/compare_calibrators.py --val-frac 0.2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from config import FEATURE_COLUMNS, MODEL_PATH, OUTPUT_DIR, UNIVERSE
from features import build_training_dataset
from model import TwoStageModel, time_series_split
from news import FinBERTScorer, FinnhubClient
from stat_mech.ising import sign_returns_from_training


def reliability_curve(probs: np.ndarray, y_binary: np.ndarray, n_bins: int = 10):
    """Bin predicted probabilities and return mean predicted vs observed frequency."""
    bins = np.linspace(0, 1, n_bins + 1)
    centers, observed = [], []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        centers.append((bins[i] + bins[i + 1]) / 2)
        observed.append(y_binary[mask].mean())
    return np.array(centers), np.array(observed)


def plot_reliability(raw_p, cal_p, y_binary, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for probs, label, color in [
        (raw_p, "Raw XGBoost", "tab:orange"),
        (cal_p, "Boltzmann calibrated", "tab:blue"),
    ]:
        centers, obs = reliability_curve(probs, y_binary)
        ax.plot(centers, obs, "o-", label=label, color=color)

    ax.set_xlabel("Mean predicted P(spike)")
    ax.set_ylabel("Observed spike frequency")
    ax.set_title("Reliability diagram")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare raw vs calibrated spike probabilities")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction")
    args = parser.parse_args()

    if not Path(str(MODEL_PATH).replace(".json", "_s1.json")).exists():
        print("No trained model found. Run: python predict/spike_detector.py --retrain")
        sys.exit(1)

    client = FinnhubClient()
    scorer = FinBERTScorer()
    X, y, intraday_ret, tickers, adaptive_thresh = build_training_dataset(UNIVERSE, client, scorer)

    X_train, y_train, X_val, y_val = time_series_split(X, y, val_frac=args.val_frac)
    ret_train = intraday_ret.iloc[:len(X_train)]
    tickers_val = tickers.iloc[len(X_train):]

    model = TwoStageModel()
    model.load()
    if not model.calibrator.fitted:
        print("Calibrator not fitted — fitting on validation split...")
        model.train(X_train, y_train, X_val, y_val,
                    ret_train, intraday_ret.iloc[len(X_train):],
                    adaptive_thresh.iloc[:len(X_train)], adaptive_thresh.iloc[len(X_train):])
        sign_returns = sign_returns_from_training(ret_train, tickers.iloc[:len(X_train)])
        model.fit_stat_mech_layers(X_val, y_val, sign_returns, tickers_val=tickers_val)

    raw = model.predict_raw(X_val)
    cal = model.predict(X_val)
    trade = model.predict_for_trade(X_val)
    y_spike = (y_val != 1).astype(int).values

    raw_nll = model.calibrator.nll(raw, X_val, y_val) if model.calibrator.fitted else float("nan")
    cal_nll = model.calibrated_val_nll(X_val, y_val)

    raw_brier = brier_score_loss(y_spike, raw["p_spike"].values)
    cal_brier = brier_score_loss(y_spike, cal["p_spike"].values)
    trade_brier = brier_score_loss(y_spike, trade["p_spike_trade"].values)

    bypass_rate = float(trade.get("calibrator_bypassed", pd.Series([False])).mean())

    print("\n" + "=" * 55)
    print("  CALIBRATOR COMPARISON (validation split)")
    print("=" * 55)
    print(f"  Samples: {len(X_val)}")
    print(f"  Raw NLL:         {raw_nll:.4f}")
    print(f"  Calibrated NLL:  {cal_nll:.4f}")
    print(f"  Raw Brier:       {raw_brier:.4f}")
    print(f"  Calibrated Brier:{cal_brier:.4f}")
    print(f"  Trade Brier:     {trade_brier:.4f}")
    print(f"  Ising λ:         {model.ising.lambda_blend:.3f} (enabled={model.ising.enabled})")
    print(f"  Calibrator bypass rate: {bypass_rate:.1%}")

    out_path = OUTPUT_DIR / "reliability_diagram.png"
    plot_reliability(raw["p_spike"].values, trade["p_spike_trade"].values, y_spike, out_path)
    print(f"\n  Reliability diagram saved to {out_path}")

    summary = pd.DataFrame([{
        "raw_nll": raw_nll,
        "cal_nll": cal_nll,
        "raw_brier": raw_brier,
        "cal_brier": cal_brier,
        "trade_brier": trade_brier,
        "bypass_rate": bypass_rate,
        "ising_lambda": model.ising.lambda_blend,
        "n_val": len(X_val),
    }])
    summary_path = OUTPUT_DIR / "calibrator_comparison.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Summary saved to {summary_path}\n")


if __name__ == "__main__":
    main()
