"""
compare_calibrators.py — Compare raw XGBoost vs Boltzmann vs Platt vs isotonic.

Optimizes for OOS precision when parquet is available, plus NLL/Brier on val.

Usage:
    python backtest/compare_calibrators.py
    python backtest/compare_calibrators.py --val-frac 0.2 --threshold 0.70
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, precision_score, recall_score

from config import FEATURE_COLUMNS, MODEL_PATH, OUTPUT_DIR, TRAINING_DATA_PATH, UNIVERSE, get_trade_threshold
from features import build_training_dataset
from model import TwoStageModel, time_series_split
from news import FinBERTScorer, FinnhubClient
from stat_mech.ising import sign_returns_from_training
from stat_mech.platt import PlattCalibrator, apply_platt_to_probs


def reliability_curve(probs: np.ndarray, y_binary: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins + 1)
    centers, observed = [], []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        centers.append((bins[i] + bins[i + 1]) / 2)
        observed.append(y_binary[mask].mean())
    return np.array(centers), np.array(observed)


def precision_at_threshold(p_spike: np.ndarray, y_spike: np.ndarray, threshold: float) -> dict:
    pred = (p_spike >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_spike, pred, zero_division=0)),
        "recall": float(recall_score(y_spike, pred, zero_division=0)),
        "signals": int(pred.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare calibration methods")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    threshold = args.threshold if args.threshold is not None else get_trade_threshold()

    if TRAINING_DATA_PATH.exists():
        df = pd.read_parquet(TRAINING_DATA_PATH)
        X = df[FEATURE_COLUMNS]
        y = df["_target"].astype(int)
        intraday_ret = df["_intraday_return"]
        tickers = df["_ticker"]
        adaptive_thresh = df.get("_adaptive_threshold")
    else:
        client = FinnhubClient()
        scorer = FinBERTScorer()
        X, y, intraday_ret, tickers, adaptive_thresh = build_training_dataset(UNIVERSE, client, scorer)

    X_train, y_train, X_val, y_val = time_series_split(X, y, val_frac=args.val_frac)
    ret_train = intraday_ret.iloc[: len(X_train)]
    tickers_val = tickers.iloc[len(X_train) :]
    y_spike = (y_val != 1).astype(int).values

    model = TwoStageModel()
    s1_path = Path(str(MODEL_PATH).replace(".json", "_s1.json"))
    if s1_path.exists():
        try:
            model.load()
        except Exception:
            model.train(X_train, y_train, X_val, y_val,
                        ret_train, intraday_ret.iloc[len(X_train) :],
                        adaptive_thresh.iloc[: len(X_train)], adaptive_thresh.iloc[len(X_train) :])
    else:
        model.train(X_train, y_train, X_val, y_val,
                    ret_train, intraday_ret.iloc[len(X_train) :],
                    adaptive_thresh.iloc[: len(X_train)], adaptive_thresh.iloc[len(X_train) :])

    if not model.calibrator.fitted:
        sign_returns = sign_returns_from_training(ret_train, tickers.iloc[: len(X_train)])
        model.fit_stat_mech_layers(X_val, y_val, sign_returns, tickers_val=tickers_val)

    raw = model.predict_raw(X_val)
    boltz = model.predict(X_val)
    trade = model.predict_for_trade(X_val)

    platt = PlattCalibrator()
    platt.fit(raw["p_spike"].values, y_spike)
    platt_df = apply_platt_to_probs(raw, platt)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw["p_spike"].values, y_spike)
    iso_p = iso.predict(raw["p_spike"].values)

    methods = {
        "raw": raw["p_spike"].values,
        "boltzmann": boltz["p_spike"].values,
        "platt": platt_df["p_spike"].values,
        "isotonic": iso_p,
        "trade_path": trade["p_spike_trade"].values,
    }

    print("\n" + "=" * 60)
    print(f"  CALIBRATOR BAKE-OFF @ threshold {threshold:.2f}")
    print("=" * 60)
    print(f"  {'Method':<14} {'Prec':>7} {'Rec':>7} {'Brier':>8} {'Signals':>8}")
    print("-" * 48)

    rows = []
    best_prec = ("", 0.0)
    for name, p in methods.items():
        m = precision_at_threshold(p, y_spike, threshold)
        brier = brier_score_loss(y_spike, p)
        print(f"  {name:<14} {m['precision']:6.1%} {m['recall']:6.1%} {brier:8.4f} {m['signals']:8d}")
        rows.append({"method": name, **m, "brier": brier})
        if m["precision"] > best_prec[1]:
            best_prec = (name, m["precision"])

    print(f"\n  Best val precision: {best_prec[0]} ({best_prec[1]:.1%})")

    # OOS slice from last 5% of data if enough rows
    if len(X) > 500:
        oos_cut = int(len(X) * 0.95)
        X_oos = X.iloc[oos_cut:]
        y_oos = (y.iloc[oos_cut] != 1).astype(int).values
        raw_oos = model.predict_raw(X_oos)
        platt_oos = apply_platt_to_probs(raw_oos, platt)
        boltz_oos = model.predict(X_oos)
        print(f"\n  OOS tail precision (last 5% rows, threshold {threshold:.2f}):")
        for name, p in [
            ("raw", raw_oos["p_spike"].values),
            ("platt", platt_oos["p_spike"].values),
            ("boltzmann", boltz_oos["p_spike"].values),
        ]:
            m = precision_at_threshold(p, y_oos, threshold)
            print(f"    {name:<12} prec={m['precision']:.1%} rec={m['recall']:.1%}")

    out_path = OUTPUT_DIR / "reliability_diagram.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    for probs, label, color in [
        (methods["raw"], "Raw", "tab:orange"),
        (methods["platt"], "Platt", "tab:green"),
        (methods["boltzmann"], "Boltzmann", "tab:blue"),
        (methods["trade_path"], "Trade path", "tab:red"),
    ]:
        centers, obs = reliability_curve(probs, y_spike)
        ax.plot(centers, obs, "o-", label=label, color=color)
    ax.set_xlabel("Mean predicted P(spike)")
    ax.set_ylabel("Observed spike frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\n  Reliability diagram: {out_path}")

    summary_path = OUTPUT_DIR / "calibrator_comparison.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"  Summary: {summary_path}\n")


if __name__ == "__main__":
    main()
