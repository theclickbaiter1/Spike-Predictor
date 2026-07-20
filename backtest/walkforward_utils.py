"""
walkforward_utils.py — Shared helpers for threshold tuning and nested walk-forward.

Used by tune_threshold.py, nested_walkforward.py, and retrain acceptance gates.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from config import (
    FEATURE_COLUMNS,
    MAX_POSITIONS_PER_DAY,
    MIN_TRADE_THRESHOLD,
    MIN_TUNED_PRECISION,
    TRADE_THRESHOLD,
    TRAINING_DATA_PATH,
    TUNED_THRESHOLD_PATH,
    VAL_FRACTION,
)
from model import TwoStageModel, time_series_split
from stat_mech.ising import sign_returns_from_training

THRESHOLD_GRID = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
MIN_RECALL = 0.25
VIX_LOW_MAX = 15.0
VIX_MID_MAX = 25.0


def load_training_df(train_until: str | None = None) -> pd.DataFrame:
    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(
            f"No {TRAINING_DATA_PATH} — run: python predict/spike_detector.py --retrain"
        )
    df = pd.read_parquet(TRAINING_DATA_PATH)
    # Backfill new columns when reading parquet from before a feature upgrade
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    if train_until:
        df = df[df.index <= pd.Timestamp(train_until)]
    return df


def train_model_fast(train_until: str, df: pd.DataFrame | None = None) -> TwoStageModel:
    """Train model on parquet rows up to train_until."""
    if df is None:
        df = load_training_df(train_until)
    else:
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
        X_train,
        y_train,
        X_val,
        y_val,
        ret_train,
        intraday_ret.iloc[len(X_train) :],
        thresh_train,
        thresh_val,
    )
    sign_returns = sign_returns_from_training(ret_train, tickers.iloc[: len(X_train)])
    model.fit_stat_mech_layers(X_val, y_val, sign_returns, tickers_val=tickers.iloc[len(X_train) :])
    model.retrain_full(X, y, intraday_ret, adaptive_thresh)
    return model


def eval_probs_slice(
    model: TwoStageModel,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
    prob_col: str = "p_spike_trade",
) -> dict:
    """Precision/recall on a feature slice at threshold."""
    probs = model.predict_for_trade(X)
    y_spike = (y != 1).astype(int).values
    p_trade = probs[prob_col].values
    pred = (p_trade >= threshold).astype(int)
    dates = pd.Index(X.index)
    n_days = max(len(dates.unique()), 1)
    signals = int(pred.sum())
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_spike, pred, zero_division=0)),
        "recall": float(recall_score(y_spike, pred, zero_division=0)),
        "signals_per_day": signals / n_days,
        "total_signals": signals,
        "n_days": n_days,
    }


def pick_best_threshold(
    model: TwoStageModel,
    X_tune: pd.DataFrame,
    y_tune: pd.Series,
    grid: list[float] | None = None,
    vix_series: pd.Series | None = None,
) -> dict:
    """Precision-first threshold selection on a tune slice."""
    grid = grid or THRESHOLD_GRID
    best = None
    for t in grid:
        m = eval_probs_slice(model, X_tune, y_tune, t)
        if m["recall"] < MIN_RECALL:
            continue
        if m["precision"] < MIN_TUNED_PRECISION:
            continue
        if m["signals_per_day"] > MAX_POSITIONS_PER_DAY:
            continue
        if best is None or m["precision"] > best["precision"]:
            best = m
    if best is None:
        best = eval_probs_slice(model, X_tune, y_tune, max(TRADE_THRESHOLD, MIN_TRADE_THRESHOLD))
    best["threshold"] = max(best["threshold"], MIN_TRADE_THRESHOLD)
    best["vix_mean"] = float(vix_series.mean()) if vix_series is not None and len(vix_series) else None
    return best


def pick_regime_thresholds(
    model: TwoStageModel,
    X_tune: pd.DataFrame,
    y_tune: pd.Series,
) -> dict:
    """Tune separate thresholds per VIX bucket on tune slice."""
    if "vix" not in X_tune.columns:
        default = pick_best_threshold(model, X_tune, y_tune)
        return {
            "default": default["threshold"],
            "vix_low": {"max_vix": VIX_LOW_MAX, "threshold": default["threshold"]},
            "vix_mid": {"max_vix": VIX_MID_MAX, "threshold": default["threshold"]},
            "vix_high": {"threshold": default["threshold"]},
            "expected_precision": default["precision"],
            "expected_recall": default["recall"],
            "signals_per_day": default["signals_per_day"],
        }

    vix = X_tune["vix"]
    # Boolean masks (not .loc on DatetimeIndex): training rows share dates across
    # tickers, so index-based lookup explodes y vs X lengths.
    bucket_masks = {
        "vix_low": vix < VIX_LOW_MAX,
        "vix_mid": (vix >= VIX_LOW_MAX) & (vix < VIX_MID_MAX),
        "vix_high": vix >= VIX_MID_MAX,
    }
    y_tune = y_tune.iloc[: len(X_tune)]
    if len(y_tune) != len(X_tune):
        raise ValueError(f"X/y length mismatch for regime tune: {len(X_tune)} vs {len(y_tune)}")
    regime = {}
    for name, mask in bucket_masks.items():
        X_b = X_tune.loc[mask]
        if len(X_b) < 50:
            continue
        y_b = y_tune.loc[mask]
        regime[name] = pick_best_threshold(model, X_b, y_b)

    default = pick_best_threshold(model, X_tune, y_tune)
    out = {
        "default": default["threshold"],
        "vix_low": {"max_vix": VIX_LOW_MAX, "threshold": regime.get("vix_low", default)["threshold"]},
        "vix_mid": {"max_vix": VIX_MID_MAX, "threshold": regime.get("vix_mid", default)["threshold"]},
        "vix_high": {"threshold": regime.get("vix_high", default)["threshold"]},
        "expected_precision": default["precision"],
        "expected_recall": default["recall"],
        "signals_per_day": default["signals_per_day"],
    }
    return out


def trading_week_slices(
    df: pd.DataFrame,
    n_oos_weeks: int = 8,
    tune_days: int = 20,
) -> list[dict]:
    """
    Build rolling OOS week slices from parquet index.
    Each fold: tune on `tune_days` before OOS week; OOS = 5 trading days.
    """
    dates = sorted(pd.Index(df.index).unique())
    if len(dates) < tune_days + n_oos_weeks * 5 + 60:
        n_oos_weeks = max(1, (len(dates) - tune_days - 60) // 5)

    folds = []
    # Walk backwards: last n_oos_weeks non-overlapping-ish windows
    end_idx = len(dates) - 1
    for w in range(n_oos_weeks):
        oos_end_idx = end_idx - w * 5
        oos_start_idx = max(0, oos_end_idx - 4)
        if oos_start_idx >= len(dates):
            break
        oos_dates = dates[oos_start_idx : oos_end_idx + 1]
        if not oos_dates:
            break

        tune_end_idx = oos_start_idx - 1
        tune_start_idx = max(0, tune_end_idx - tune_days + 1)
        if tune_start_idx >= tune_end_idx:
            break
        tune_dates = dates[tune_start_idx : tune_end_idx + 1]
        train_until = dates[tune_start_idx - 1] if tune_start_idx > 0 else dates[0]

        folds.append(
            {
                "oos_start": str(pd.Timestamp(oos_dates[0]).date()),
                "oos_end": str(pd.Timestamp(oos_dates[-1]).date()),
                "train_until": str(pd.Timestamp(train_until).date()),
                "tune_dates": tune_dates,
                "oos_dates": oos_dates,
            }
        )
    folds.reverse()
    return folds


def run_nested_fold(
    df: pd.DataFrame,
    fold: dict,
    grid: list[float] | None = None,
) -> dict:
    """Train ≤ train_until, tune on tune_dates, evaluate on oos_dates."""
    train_until = fold["train_until"]
    model = train_model_fast(train_until, df)

    tune_mask = df.index.isin(fold["tune_dates"])
    oos_mask = df.index.isin(fold["oos_dates"])
    X_tune = df.loc[tune_mask, FEATURE_COLUMNS]
    y_tune = df.loc[tune_mask, "_target"].astype(int)
    X_oos = df.loc[oos_mask, FEATURE_COLUMNS]
    y_oos = df.loc[oos_mask, "_target"].astype(int)

    best = pick_best_threshold(model, X_tune, y_tune, grid=grid)
    oos_m = eval_probs_slice(model, X_oos, y_oos, best["threshold"])
    val_m = eval_probs_slice(model, X_tune, y_tune, best["threshold"])

    return {
        "train_until": train_until,
        "oos_start": fold["oos_start"],
        "oos_end": fold["oos_end"],
        "tuned_threshold": best["threshold"],
        "tune_precision": val_m["precision"],
        "oos_precision": oos_m["precision"],
        "oos_recall": oos_m["recall"],
        "oos_signals_per_day": oos_m["signals_per_day"],
        "val_oos_gap": val_m["precision"] - oos_m["precision"],
    }


def run_nested_walkforward(
    n_oos_weeks: int = 8,
    tune_days: int = 20,
    grid: list[float] | None = None,
) -> tuple[list[dict], dict]:
    """Run full nested walk-forward; return per-fold results and aggregate summary."""
    df = load_training_df()
    folds = trading_week_slices(df, n_oos_weeks=n_oos_weeks, tune_days=tune_days)
    if not folds:
        raise RuntimeError("Not enough data for nested walk-forward.")

    results = []
    for fold in folds:
        results.append(run_nested_fold(df, fold, grid=grid))

    precisions = [r["oos_precision"] for r in results]
    gaps = [r["val_oos_gap"] for r in results]
    thresholds = [r["tuned_threshold"] for r in results]
    summary = {
        "n_folds": len(results),
        "mean_oos_precision": float(np.mean(precisions)),
        "median_oos_precision": float(np.median(precisions)),
        "mean_val_oos_gap": float(np.mean(gaps)),
        "median_threshold": float(np.median(thresholds)),
        "std_threshold": float(np.std(thresholds)),
    }
    return results, summary


def save_tuned_threshold_config(
    regime: dict,
    train_until: str,
    test_start: str = "",
    test_end: str = "",
    nested_summary: dict | None = None,
) -> Path:
    """Write tuned_threshold.json with regime buckets."""
    from datetime import datetime

    out = {
        "threshold": regime["default"],
        "default": regime["default"],
        "vix_low": regime["vix_low"],
        "vix_mid": regime["vix_mid"],
        "vix_high": regime["vix_high"],
        "expected_precision": regime.get("expected_precision", 0.0),
        "expected_recall": regime.get("expected_recall", 0.0),
        "signals_per_day": regime.get("signals_per_day", 0.0),
        "tuning_window": {
            "train_until": train_until,
            "test_start": test_start,
            "test_end": test_end,
        },
        "tuned_at": datetime.now().isoformat(),
    }
    if nested_summary:
        out["nested_oos"] = nested_summary
    TUNED_THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TUNED_THRESHOLD_PATH, "w") as f:
        json.dump(out, f, indent=2)
    return TUNED_THRESHOLD_PATH


def quick_holdout_oos_precision(
    model: TwoStageModel,
    df: pd.DataFrame,
    holdout_frac: float = 0.05,
    threshold: float | None = None,
) -> float:
    """Fast OOS precision on trailing holdout dates (no retrain). For retrain gates."""
    if threshold is None:
        from config import get_trade_threshold
        threshold = get_trade_threshold()
    dates = sorted(pd.Index(df.index).unique())
    if len(dates) < 20:
        return float("nan")
    cut = max(0, int(len(dates) * (1 - holdout_frac)))
    holdout = set(dates[cut:])
    mask = df.index.isin(holdout)
    if mask.sum() < 10:
        return float("nan")
    X_h = df.loc[mask, FEATURE_COLUMNS]
    y_h = df.loc[mask, "_target"].astype(int)
    m = eval_probs_slice(model, X_h, y_h, threshold)
    return m["precision"]


def quick_holdout_signal_pnl(
    model: TwoStageModel,
    df: pd.DataFrame,
    holdout_frac: float = 0.05,
    threshold: float | None = None,
) -> float:
    """Mean direction-aligned intraday return on trade signals in trailing holdout."""
    if threshold is None:
        from config import get_trade_threshold
        threshold = get_trade_threshold()
    dates = sorted(pd.Index(df.index).unique())
    if len(dates) < 20:
        return float("nan")
    cut = max(0, int(len(dates) * (1 - holdout_frac)))
    holdout = set(dates[cut:])
    mask = df.index.isin(holdout)
    if mask.sum() < 10 or "_intraday_return" not in df.columns:
        return float("nan")
    X_h = df.loc[mask, FEATURE_COLUMNS]
    rets = df.loc[mask, "_intraday_return"]
    probs = model.predict_for_trade(X_h)
    signed = []
    for idx in X_h.index:
        # Training frames use a shared DatetimeIndex across tickers; use positional lookup.
        r = probs.loc[idx]
        if isinstance(r, pd.DataFrame):
            # Duplicate index: evaluate each row
            rows = [r.iloc[i] for i in range(len(r))]
            ret_vals = rets.loc[idx]
            if isinstance(ret_vals, pd.Series):
                ret_list = ret_vals.tolist()
            else:
                ret_list = [float(ret_vals)] * len(rows)
        else:
            rows = [r]
            ret_vals = rets.loc[idx]
            ret_list = [float(ret_vals.iloc[0]) if isinstance(ret_vals, pd.Series) else float(ret_vals)]
        for i, row in enumerate(rows):
            p_trade = float(row.get("p_spike_trade", row["p_spike"]))
            if p_trade < threshold:
                continue
            direction = 1 if row["p_up"] > row["p_down"] else -1
            signed.append(float(ret_list[min(i, len(ret_list) - 1)]) * direction)
    if not signed:
        return float("nan")
    return float(np.mean(signed))


def quick_nested_oos_precision(n_folds: int = 2, tune_days: int = 20) -> tuple[float, float]:
    """
    Fast check for retrain gate: mean OOS precision and val-OOS gap over last n folds.
    Returns (mean_oos_precision, mean_val_oos_gap).
    """
    try:
        _, summary = run_nested_walkforward(n_oos_weeks=n_folds, tune_days=tune_days)
        return summary["mean_oos_precision"], summary["mean_val_oos_gap"]
    except Exception:
        return float("nan"), float("nan")
