"""
ablation_study.py — Feature-group ablation and Stage 1 objective sweep.

Usage:
    python backtest/ablation_study.py --mode ablation
    python backtest/ablation_study.py --mode stage1-sweep
    python backtest/ablation_study.py --mode ablation --oos-weeks 4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np
import pandas as pd

from config import CATALYST_COLUMNS, FEATURE_COLUMNS, STAT_MECH_COLUMNS, TRAINING_DATA_PATH
from walkforward_utils import trading_week_slices, load_training_df
from model import TwoStageModel, time_series_split
from stat_mech.ising import sign_returns_from_training

FEATURE_GROUPS = {
    "sentiment": [
        "overnight_sentiment_mean", "overnight_sentiment_max",
        "overnight_sentiment_min", "overnight_news_count", "overnight_sentiment_std",
        "news_count_z_score", "news_spike", "has_overnight_news",
    ],
    "technical": [
        "prev_close", "rsi_14", "ema_10", "realized_vol_20d", "avg_volume_10d",
        "prev_day_return", "prev_day_range", "gap_3d", "overnight_gap", "vol_z_score",
    ],
    "macro": [
        "vix", "treasury_10y", "sector_momentum_5d", "sp500_prev_return", "vix_change",
        "yield_curve_spread", "dxy_change", "crude_oil_change", "gold_change", "sp500_5d_return",
        "vix_change_3d", "vix_change_5d", "vix_regime", "dxy_change_5d", "crude_oil_change_5d",
        "gold_change_5d", "treasury_10y_delta_5d", "sp500_return_3d",
    ],
    "calendar": [
        "day_of_week", "is_monday", "is_friday", "days_to_earnings", "is_earnings_day",
    ],
    "earnings": [
        "eps_surprise_last", "revenue_surprise_last", "earnings_streak",
        "post_earnings_drift_1d", "earnings_volatility",
    ],
    "catalyst": list(CATALYST_COLUMNS),
    "stat_mech": list(STAT_MECH_COLUMNS),
}


def _zero_group(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = 0.0
    return out


def train_model_on_df(df: pd.DataFrame, train_until: str) -> TwoStageModel:
    """Train on modified dataframe up to train_until."""
    df = df[df.index <= pd.Timestamp(train_until)]
    X = df[FEATURE_COLUMNS]
    y = df["_target"].astype(int)
    intraday_ret = df["_intraday_return"]
    tickers = df["_ticker"]
    adaptive_thresh = df.get("_adaptive_threshold")

    X_train, y_train, X_val, y_val = time_series_split(X, y)
    ret_train = intraday_ret.iloc[: len(X_train)]
    thresh_train = adaptive_thresh.iloc[: len(X_train)] if adaptive_thresh is not None else None
    thresh_val = adaptive_thresh.iloc[len(X_train) :] if adaptive_thresh is not None else None

    model = TwoStageModel()
    model.train(
        X_train, y_train, X_val, y_val,
        ret_train, intraday_ret.iloc[len(X_train) :],
        thresh_train, thresh_val,
    )
    sign_returns = sign_returns_from_training(ret_train, tickers.iloc[: len(X_train)])
    model.fit_stat_mech_layers(X_val, y_val, sign_returns, tickers_val=tickers.iloc[len(X_train) :])
    model.retrain_full(X, y, intraday_ret, adaptive_thresh)
    return model


def ablation_oos_precision(
    df: pd.DataFrame,
    zero_cols: list[str] | None,
    n_oos_weeks: int = 4,
) -> float:
    """Mean OOS precision over last n folds with optional feature group zeroed."""
    work = _zero_group(df, zero_cols) if zero_cols else df
    folds = trading_week_slices(work, n_oos_weeks=n_oos_weeks, tune_days=20)
    if not folds:
        return float("nan")
    precs = []
    for fold in folds[-n_oos_weeks:]:
        # Custom train using zeroed df
        train_until = fold["train_until"]
        model = train_model_on_df(work, train_until)
        from walkforward_utils import pick_best_threshold, eval_probs_slice
        tune_mask = work.index.isin(fold["tune_dates"])
        oos_mask = work.index.isin(fold["oos_dates"])
        X_tune = work.loc[tune_mask, FEATURE_COLUMNS]
        y_tune = work.loc[tune_mask, "_target"].astype(int)
        X_oos = work.loc[oos_mask, FEATURE_COLUMNS]
        y_oos = work.loc[oos_mask, "_target"].astype(int)
        best = pick_best_threshold(model, X_tune, y_tune)
        oos_m = eval_probs_slice(model, X_oos, y_oos, best["threshold"])
        precs.append(oos_m["precision"])
    return float(np.mean(precs))


def run_ablation(n_oos_weeks: int = 4):
    df = load_training_df()
    baseline = ablation_oos_precision(df, None, n_oos_weeks)
    print(f"\nAblation study ({n_oos_weeks} OOS weeks)\n")
    print(f"{'Group':<12} {'OOS Prec':>10} {'Delta':>10}")
    print("-" * 34)
    print(f"{'baseline':<12} {baseline:9.1%} {'—':>10}")

    rows = [{"group": "baseline", "oos_precision": baseline, "delta": 0.0}]
    for name, cols in FEATURE_GROUPS.items():
        prec = ablation_oos_precision(df, cols, n_oos_weeks)
        delta = prec - baseline if not np.isnan(prec) and not np.isnan(baseline) else float("nan")
        print(f"{name:<12} {prec:9.1%} {delta:+9.1%}")
        rows.append({"group": name, "oos_precision": prec, "delta": delta})

    from config import OUTPUT_DIR
    out = OUTPUT_DIR / "ablation_study.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")


def run_stage1_sweep(n_oos_weeks: int = 2):
    """Grid search S1_POS_WEIGHT_MULTIPLIER on nested OOS precision."""
    import config as cfg

    df = load_training_df()
    multipliers = [1.0, 1.5, 2.0]
    print(f"\nStage 1 scale_pos_weight sweep ({n_oos_weeks} OOS weeks)\n")
    print(f"{'Multiplier':>12} {'OOS Prec':>10}")
    print("-" * 24)

    rows = []
    for mult in multipliers:
        cfg.S1_POS_WEIGHT_MULTIPLIER = mult
        prec = ablation_oos_precision(df, None, n_oos_weeks)
        print(f"{mult:12.1f} {prec:9.1%}")
        rows.append({"multiplier": mult, "oos_precision": prec})
    cfg.S1_POS_WEIGHT_MULTIPLIER = 1.0

    from config import OUTPUT_DIR
    out = OUTPUT_DIR / "stage1_sweep.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")


def main():
    parser = argparse.ArgumentParser(description="Feature ablation / Stage1 sweep")
    parser.add_argument("--mode", choices=["ablation", "stage1-sweep"], default="ablation")
    parser.add_argument("--oos-weeks", type=int, default=4)
    args = parser.parse_args()

    if not TRAINING_DATA_PATH.exists():
        print("No training_data.parquet — run retrain first.")
        sys.exit(1)

    if args.mode == "ablation":
        run_ablation(args.oos_weeks)
    else:
        run_stage1_sweep(args.oos_weeks)


if __name__ == "__main__":
    main()
