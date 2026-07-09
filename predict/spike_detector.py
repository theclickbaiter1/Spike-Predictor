"""
spike_detector.py — Main pipeline for the Pre-Market Spike Detector (v2).

Usage:
    python predict/spike_detector.py              # Daily prediction (8 AM run)
    python predict/spike_detector.py --retrain    # Retrain model from scratch
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import shutil
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import (
    FEATURE_COLUMNS, MODEL_BACKUP_DIR, MODEL_PATH, OUTPUT_DIR,
    TRAINING_DATA_PATH, UNIVERSE, WATCHLIST_THRESHOLD_HIGH, WATCHLIST_THRESHOLD_LOW,
    get_trade_threshold,
)

MIN_TRAINING_TICKERS = 50
ET = ZoneInfo("America/New_York")


def print_banner():
    now = datetime.now(ET)
    print()
    print("═" * 65)
    print(f"  SPIKE DETECTOR v2 — {now.strftime('%Y-%m-%d %I:%M %p ET')}")
    print("═" * 65)


def determine_top_signal(features: pd.Series) -> str:
    signals = []
    sent = features.get("overnight_sentiment_mean", 0)
    nc = features.get("overnight_news_count", 0)
    if nc > 3 and abs(sent) > 0.3:
        signals.append(f"{'Pos' if sent > 0 else 'Neg'} sentiment")
    if features.get("is_earnings_day", 0) == 1:
        signals.append("Earnings day")
    gap = features.get("overnight_gap", 0)
    if abs(gap) > 0.02:
        signals.append(f"Gap {'+'if gap>0 else ''}{gap*100:.1f}%")
    rsi = features.get("rsi_14", 50)
    if rsi > 70: signals.append(f"RSI {rsi:.0f}")
    elif rsi < 30: signals.append(f"RSI {rsi:.0f}")
    vol = features.get("realized_vol_20d", 0)
    if vol > 0.6: signals.append(f"High vol")
    ns = features.get("news_spike", 0)
    if ns > 0: signals.append("News spike")
    if not signals: signals.append("Technical composite")
    return " + ".join(signals[:3])


def print_watchlist(results: pd.DataFrame):
    if results.empty:
        print("\n  No predictions.\n")
        return

    trade_thresh = get_trade_threshold()
    watch_hi = min(WATCHLIST_THRESHOLD_HIGH, trade_thresh)
    watch_lo = WATCHLIST_THRESHOLD_LOW
    pcol = "p_spike_trade" if "p_spike_trade" in results.columns else "p_spike"

    trade = results[results[pcol] >= trade_thresh]
    watchlist = results[(results[pcol] >= watch_lo) & (results[pcol] < watch_hi)]
    high = results[(results[pcol] >= 0.60) & (results[pcol] < trade_thresh)]
    moderate = results[(results[pcol] >= 0.40) & (results[pcol] < 0.60)]
    flat_count = len(results[results[pcol] < 0.40])

    def _table(df, emoji, header, pcol_name=pcol):
        if df.empty:
            return
        print(f"\n  {emoji} {header}")
        print("  ┌────────┬───────────┬──────────┬───────────┬──────────────────────────┐")
        print("  │ Ticker │ Direction │ P(spike) │ P(dir)    │ Top Signal               │")
        print("  ├────────┼───────────┼──────────┼───────────┼──────────────────────────┤")
        for _, r in df.iterrows():
            d = "▲ UP" if r["p_up"] > r["p_down"] else "▼ DOWN"
            pdir = max(r["p_up"], r["p_down"])
            psp = r[pcol_name]
            print(f"  │ {r['ticker']:<6} │ {d:<9} │ {psp*100:6.1f}%  │ {pdir*100:6.1f}%    │ {r['top_signal'][:24]:<24} │")
        print("  └────────┴───────────┴──────────┴───────────┴──────────────────────────┘")

    _table(trade, "🔴", f"TRADE TIER (≥{trade_thresh*100:.0f}%) — executes")
    _table(watchlist, "🟠", f"WATCHLIST ({watch_lo*100:.0f}-{watch_hi*100:.0f}%) — alert only")
    _table(high, "🟡", "HIGH PROBABILITY (60%+ below trade)")
    _table(moderate, "🟢", "MODERATE PROBABILITY (40-60%)")
    if flat_count > 0:
        print(f"\n  ⚪ LOW PROBABILITY (<40%) — {flat_count} tickers predicted FLAT")
    print("\n" + "═" * 65)


def data_quality_report(X, y, intraday_ret):
    """Print a data quality audit before training."""
    print("\n" + "─" * 65)
    print("  📊 DATA QUALITY REPORT")
    print("─" * 65)

    n_rows, n_cols = X.shape
    print(f"  Dataset: {n_rows:,} rows × {n_cols} features")

    # NaN counts per feature
    nan_counts = X.isna().sum()
    has_nans = nan_counts[nan_counts > 0]
    if len(has_nans) > 0:
        print(f"\n  ⚠ Features with NaN values ({len(has_nans)}/{n_cols}):")
        for feat, count in has_nans.sort_values(ascending=False).items():
            pct = count / n_rows * 100
            print(f"    {feat:35s} {count:6d} ({pct:5.1f}%)")
    else:
        print("\n  ✅ No NaN values in any feature.")

    # Zero rate per feature
    print(f"\n  Zero rates (features with >30% zeros):")
    zero_warnings = 0
    for col in X.columns:
        zero_rate = (X[col] == 0).sum() / n_rows
        if zero_rate > 0.30:
            flag = " ⚠ HIGH" if zero_rate > 0.50 else ""
            print(f"    {col:35s} {zero_rate*100:5.1f}% zeros{flag}")
            zero_warnings += 1
    if zero_warnings == 0:
        print("    None — all features have <30% zeros.")

    # All-zero rows (a row where every feature is 0 = likely bad data)
    all_zero_rows = (X == 0).all(axis=1).sum()
    if all_zero_rows > 0:
        print(f"\n  🔴 {all_zero_rows} rows have ALL features = 0 (likely bad data)")
    else:
        print(f"\n  ✅ No all-zero rows detected.")

    # Target distribution
    print(f"\n  Target distribution:")
    for label_idx, name in enumerate(["spike_down", "flat", "spike_up"]):
        count = (y == label_idx).sum()
        print(f"    {name:12s} {count:6d} ({count/len(y)*100:5.1f}%)")

    # Intraday return stats
    print(f"\n  Intraday return distribution:")
    print(f"    Mean:   {intraday_ret.mean()*100:+.3f}%")
    print(f"    Median: {intraday_ret.median()*100:+.3f}%")
    print(f"    Std:    {intraday_ret.std()*100:.3f}%")
    print(f"    Min:    {intraday_ret.min()*100:+.2f}%")
    print(f"    Max:    {intraday_ret.max()*100:+.2f}%")

    # Feature range checks
    print(f"\n  Feature range check (suspicious if min == max == 0):")
    suspicious = 0
    all_zero_features = []
    for col in X.columns:
        if X[col].min() == 0 and X[col].max() == 0:
            print(f"    🔴 {col} — all values are 0!")
            suspicious += 1
            all_zero_features.append(col)
    if suspicious == 0:
        print("    ✅ All features have non-trivial value ranges.")
    elif all_zero_features:
        print(f"\n  ⚠ {len(all_zero_features)} dead feature(s): {', '.join(all_zero_features[:5])}"
              f"{'...' if len(all_zero_features) > 5 else ''}")

    # Known issue: days_to_earnings — now point-in-time (historical only)
    if "days_to_earnings" in X.columns:
        print(f"\n  ℹ 'days_to_earnings' uses reported historical earnings only (no lookahead).")

    print("─" * 65 + "\n")


def backup_current_model():
    """Back up the current model files before retraining. Returns backup dir or None."""
    from pathlib import Path

    base = str(MODEL_PATH).replace(".json", "")
    s1_path = Path(f"{base}_s1.json")
    s2_path = Path(f"{base}_s2.json")
    meta_path = Path(f"{base}_meta.json")

    if not s1_path.exists():
        print("  No existing model to back up.")
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    backup_dir = MODEL_BACKUP_DIR / f"backup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for src in [s1_path, s2_path, meta_path, MODEL_PATH]:
        if src.exists():
            shutil.copy2(src, backup_dir / src.name)

    base = Path(base)
    for suffix in ["_calibrator.json", "_ising.json", "_ret.json"]:
        extra = base.parent / f"{base.name}{suffix}"
        if extra.exists():
            shutil.copy2(extra, backup_dir / extra.name)

    beta_norm = Path(str(MODEL_PATH).replace("model.json", "beta_norm.json"))
    if not beta_norm.exists():
        beta_norm = Path(__file__).resolve().parent.parent / "data" / "beta_norm.json"
    if beta_norm.exists():
        shutil.copy2(beta_norm, backup_dir / "beta_norm.json")

    print(f"  📦 Current model backed up to {backup_dir}")
    return backup_dir


def restore_model_from_backup(backup_dir: Path):
    """Restore model files from a backup directory."""
    base = str(MODEL_PATH).replace(".json", "")
    model_name = Path(base).name
    for suffix in ["_s1.json", "_s2.json", "_ret.json", "_meta.json", "_calibrator.json", "_ising.json"]:
        src = backup_dir / f"{model_name}{suffix}"
        dest = Path(f"{base}{suffix}")
        if src.exists():
            shutil.copy2(src, dest)

    beta_src = backup_dir / "beta_norm.json"
    if beta_src.exists():
        from config import DATA_DIR
        shutil.copy2(beta_src, DATA_DIR / "beta_norm.json")
    print(f"  ↩️  Model restored from {backup_dir}")


# Retrain acceptance: reject new model if val F1 or calibrated NLL worsens vs backup
RETRAIN_MIN_F1_DELTA = -0.03
RETRAIN_MAX_NLL_DELTA = 0.05
RETRAIN_MIN_TRADE_PRECISION = 0.30
RETRAIN_MAX_SIGNALS_PER_DAY = 15
RETRAIN_MAX_VAL_OOS_GAP = 0.20  # reject if val prec - nested OOS prec > 20pp
RETRAIN_MIN_HOLDOUT_SIGNAL_PNL = 0.0  # mean signed return on holdout signals


def _mini_oos_gate(model, X_val, y_val, threshold: float | None = None) -> tuple[bool, str]:
    """Check last 5 val trading days at val-tuned threshold (not stale tuned_threshold.json)."""
    dates = pd.Index(X_val.index)
    unique_dates = sorted(dates.unique())[-5:]
    if len(unique_dates) < 2:
        return True, "skipped (insufficient val dates)"

    mask = dates.isin(unique_dates)
    X_slice = X_val.loc[mask]
    y_slice = y_val.loc[mask]

    if threshold is None:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backtest"))
            from walkforward_utils import pick_best_threshold
            threshold = pick_best_threshold(model, X_val, y_val)["threshold"]
        except Exception:
            threshold = get_trade_threshold()

    trade_m = model.trade_val_metrics(X_slice, y_slice, threshold)
    raw_m = model.spike_val_metrics(X_slice, y_slice, threshold=threshold, calibrated=False)
    sig_per_day = trade_m["signals"] / max(len(unique_dates), 1)

    reasons = []
    # Allow low precision only when almost no signals (threshold too strict on short window)
    if trade_m["signals"] >= 3 and trade_m["precision"] < RETRAIN_MIN_TRADE_PRECISION:
        reasons.append(f"trade precision {trade_m['precision']:.2f} < {RETRAIN_MIN_TRADE_PRECISION}")
    if sig_per_day > RETRAIN_MAX_SIGNALS_PER_DAY:
        reasons.append(f"signals/day {sig_per_day:.1f} > {RETRAIN_MAX_SIGNALS_PER_DAY}")
    if trade_m["signals"] >= 3 and trade_m["precision"] < raw_m["precision"]:
        reasons.append(f"trade prec {trade_m['precision']:.2f} < raw {raw_m['precision']:.2f}")

    summary = (f"mini-OOS @ {threshold:.2f}: trade prec={trade_m['precision']:.2f} "
               f"rec={trade_m['recall']:.2f} sig/day={sig_per_day:.1f} vs raw prec={raw_m['precision']:.2f}")
    return len(reasons) == 0, summary if not reasons else summary + " — " + "; ".join(reasons)


def run_retrain(backup=True, force=False):
    from features import build_training_dataset
    from model import TwoStageModel, time_series_split
    from news import FinBERTScorer, FinnhubClient
    from stat_mech.ising import sign_returns_from_training
    from config import get_trade_threshold

    print_banner()
    print("  MODE: RETRAIN (2-stage model + adaptive threshold)\n")

    # Back up existing model
    backup_dir = None
    if backup:
        backup_dir = backup_current_model()

    client = FinnhubClient()
    scorer = FinBERTScorer()

    X, y, intraday_ret, tickers, adaptive_thresh = build_training_dataset(UNIVERSE, client, scorer)

    n_tickers = tickers.nunique()
    if n_tickers < MIN_TRAINING_TICKERS:
        print(f"\n  🛑 RETRAIN ABORTED — only {n_tickers}/{len(UNIVERSE)} tickers loaded.")
        print("     Check yfinance connectivity and re-run.")
        if backup_dir is not None:
            restore_model_from_backup(backup_dir)
        sys.exit(1)

    training_df = X.copy()
    training_df["_target"] = y
    training_df["_intraday_return"] = intraday_ret
    training_df["_ticker"] = tickers.values
    training_df.to_parquet(TRAINING_DATA_PATH)
    print(f"\n  Training data saved to {TRAINING_DATA_PATH}")

    # Data quality audit
    data_quality_report(X, y, intraday_ret)

    X_train, y_train, X_val, y_val = time_series_split(X, y)
    ret_train = intraday_ret.iloc[:len(X_train)]
    ret_val = intraday_ret.iloc[len(X_train):]
    thresh_train = adaptive_thresh.iloc[:len(X_train)]
    thresh_val = adaptive_thresh.iloc[len(X_train):]

    # Evaluate previous model on same val split (acceptance gate)
    old_metrics = None
    old_nll = None
    feature_migration = False
    if backup_dir is not None and not force:
        old_model = TwoStageModel()
        try:
            old_model.load()
            old_s1 = set(old_model.spike_model.get_booster().feature_names)
            new_s1 = set(c for c in FEATURE_COLUMNS if c not in {"realized_vol_20d", "inverse_temperature"})
            if old_s1 != new_s1:
                feature_migration = True
                print(f"\n  ℹ Feature migration detected ({len(old_s1)} → {len(new_s1)} S1 features); "
                      "skipping old-model comparison gates.")
            else:
                old_metrics = old_model.spike_val_metrics(X_val, y_val)
                old_nll = old_model.calibrated_val_nll(X_val, y_val)
                print(f"\n  Previous model val F1: {old_metrics['f1']:.3f} "
                      f"(prec {old_metrics['precision']:.3f}, rec {old_metrics['recall']:.3f})")
                if old_nll < float("inf"):
                    print(f"  Previous model val NLL: {old_nll:.4f}")
        except Exception as e:
            feature_migration = True
            print(f"\n  ⚠ Could not load previous model for comparison: {e}")

    model = TwoStageModel()
    model.train(X_train, y_train, X_val, y_val, ret_train, ret_val,
                thresh_train, thresh_val)

    sign_returns = sign_returns_from_training(ret_train, tickers.iloc[:len(X_train)])
    model.fit_stat_mech_layers(X_val, y_val, sign_returns, tickers_val=tickers.iloc[len(X_train):])

    new_metrics = model.spike_val_metrics(X_val, y_val)
    new_nll = model.calibrated_val_nll(X_val, y_val)
    trade_val = model.trade_val_metrics(X_val, y_val, get_trade_threshold())
    oos_ok, oos_msg = _mini_oos_gate(model, X_val, y_val)
    if force or feature_migration:
        oos_ok = True
        oos_msg = f"skipped (force={force}, migration={feature_migration})"

    nested_oos_prec, nested_gap = float("nan"), float("nan")
    holdout_pnl = float("nan")
    overfit_ok = True
    economic_ok = True
    if not force and not feature_migration:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backtest"))
            from walkforward_utils import quick_holdout_oos_precision, quick_holdout_signal_pnl
            holdout_df = pd.read_parquet(TRAINING_DATA_PATH)
            nested_oos_prec = quick_holdout_oos_precision(model, holdout_df, holdout_frac=0.05)
            holdout_pnl = quick_holdout_signal_pnl(model, holdout_df, holdout_frac=0.05)
            val_trade_prec = trade_val["precision"]
            if not np.isnan(nested_oos_prec):
                gap = val_trade_prec - nested_oos_prec
                if gap > RETRAIN_MAX_VAL_OOS_GAP:
                    overfit_ok = False
                print(f"  Holdout OOS prec (5%): {nested_oos_prec:.2f}, val→OOS gap: {gap:+.2f}")
            if not np.isnan(holdout_pnl) and holdout_pnl < RETRAIN_MIN_HOLDOUT_SIGNAL_PNL:
                economic_ok = False
                print(f"  🛑 Holdout signal P&L {holdout_pnl:+.4f} < {RETRAIN_MIN_HOLDOUT_SIGNAL_PNL:+.4f}")
            elif not np.isnan(holdout_pnl):
                print(f"  Holdout signal P&L proxy: {holdout_pnl:+.4f}")
        except Exception as e:
            print(f"  ⚠ Holdout OOS check skipped: {e}")
    else:
        print("  Holdout OOS check skipped (force/migration).")

    print(f"\n  New model val F1: {new_metrics['f1']:.3f} "
          f"(prec {new_metrics['precision']:.3f}, rec {new_metrics['recall']:.3f})")
    print(f"  New model val NLL: {new_nll:.4f}")
    print(f"  Mini-OOS gate: {oos_msg}")

    if old_metrics is not None and not force:
        f1_delta = new_metrics["f1"] - old_metrics["f1"]
        reject = False
        if f1_delta < RETRAIN_MIN_F1_DELTA:
            print(f"\n  🛑 RETRAIN REJECTED — val F1 dropped {f1_delta:+.3f} "
                  f"(threshold {RETRAIN_MIN_F1_DELTA:+.3f}).")
            reject = True
        if old_nll is not None and old_nll < float("inf") and new_nll < float("inf"):
            nll_delta = new_nll - old_nll
            if nll_delta > RETRAIN_MAX_NLL_DELTA:
                print(f"  🛑 RETRAIN REJECTED — val NLL increased {nll_delta:+.4f} "
                      f"(threshold {RETRAIN_MAX_NLL_DELTA:+.4f}).")
                reject = True
        if not oos_ok:
            print(f"  🛑 RETRAIN REJECTED — {oos_msg}")
            reject = True
        if not overfit_ok:
            print(f"  🛑 RETRAIN REJECTED — val/OOS overfit gap > {RETRAIN_MAX_VAL_OOS_GAP:.0%}")
            reject = True
        if not economic_ok:
            print("  🛑 RETRAIN REJECTED — holdout signal P&L below floor")
            reject = True
        if reject:
            restore_model_from_backup(backup_dir)
            sys.exit(1)
        print(f"  ✅ Acceptance gate passed (F1 delta {f1_delta:+.3f})")
    elif (not oos_ok or not overfit_ok or not economic_ok) and not force:
        if not oos_ok:
            print(f"  🛑 RETRAIN REJECTED — {oos_msg}")
        if not overfit_ok:
            print(f"  🛑 RETRAIN REJECTED — val/OOS overfit gap > {RETRAIN_MAX_VAL_OOS_GAP:.0%}")
        if not economic_ok:
            print("  🛑 RETRAIN REJECTED — holdout signal P&L below floor")
        if backup_dir is not None:
            restore_model_from_backup(backup_dir)
        sys.exit(1)

    print("\n  Top 10 Spike Detection Feature Importances:")
    importance = model.get_spike_feature_importance()
    for feat, score in importance.head(10).items():
        bar = "█" * int(score * 100)
        print(f"    {feat:30s} {score:.3f} {bar}")

    model.retrain_full(X, y, intraday_ret, adaptive_thresh)
    model.save()

    print("\n  Tuning trade threshold (walk-forward)...")
    import subprocess
    tune_script = Path(__file__).resolve().parent.parent / "backtest" / "tune_threshold.py"
    subprocess.run([sys.executable, str(tune_script), "--nested"], check=False)

    print("\n  ✅ Retraining complete.")
    print("═" * 65)


def run_predict():
    from features import (
        build_single_day_features, _download_safe, compute_macro_features,
        impute_features_for_predict, finalize_live_stat_mech,
    )
    from model import TwoStageModel
    from news import FinBERTScorer, FinnhubClient
    from predict.trade import expected_edge

    print_banner()
    print("  MODE: PREDICT\n")

    s1_path = str(MODEL_PATH).replace(".json", "_s1.json")
    from pathlib import Path
    if not Path(s1_path).exists():
        print("  ❌ No trained model found. Run with --retrain first.")
        sys.exit(1)

    model = TwoStageModel()
    model.load()

    client = FinnhubClient()
    scorer = FinBERTScorer()

    today = datetime.now(ET)
    end_str = today.strftime("%Y-%m-%d")
    start_str = (today - timedelta(days=60)).strftime("%Y-%m-%d")

    # Macro — use last valid row (ffill handles weekends / partial downloads)
    print("  Fetching macro data...")
    spy_data = _download_safe("^GSPC", start_str, end_str)
    if spy_data.empty:
        trading_dates = pd.bdate_range(end=pd.Timestamp(end_str), periods=30)
    else:
        trading_dates = spy_data.index
    macro_df = compute_macro_features(trading_dates, start_str, end_str)
    if not macro_df.empty:
        macro_cache = macro_df.ffill().bfill().iloc[-1].to_dict()
    else:
        macro_cache = {}

    print(f"\n  Scanning {len(UNIVERSE)} tickers...\n")
    feature_rows = {}
    for i, ticker in enumerate(UNIVERSE):
        print(f"    [{i+1}/{len(UNIVERSE)}] {ticker}...", end=" ", flush=True)
        row = build_single_day_features(
            ticker, today, client, scorer,
            macro_cache=macro_cache,
        )
        feature_rows[ticker] = row
        print("OK")

    feature_rows = finalize_live_stat_mech(feature_rows)
    X = pd.DataFrame(feature_rows).T
    X.columns = FEATURE_COLUMNS
    valid_tickers = list(impute_features_for_predict(X).index)
    skipped = set(UNIVERSE) - set(valid_tickers)
    if skipped:
        print(f"\n  ⚠ Skipped {len(skipped)} tickers with missing macro data: "
              f"{', '.join(sorted(skipped)[:8])}{'...' if len(skipped) > 8 else ''}")
    X = impute_features_for_predict(X)
    probs = model.predict_for_trade(X)

    results = []
    for ticker in valid_tickers:
        r = probs.loc[ticker]
        row = feature_rows[ticker]
        results.append({
            "ticker": ticker,
            "p_spike": r["p_spike"],
            "p_spike_raw": r.get("p_spike_raw", r["p_spike"]),
            "p_spike_trade": r.get("p_spike_trade", r["p_spike"]),
            "expected_abs_return": float(r.get("expected_abs_return", 0) or 0),
            "expected_signed_return": float(r.get("expected_signed_return", 0) or 0),
            "expected_value": expected_edge(r),
            "p_up": r["p_up"],
            "p_down": r["p_down"],
            "p_flat": r["p_flat"],
            "calibrator_bypassed": bool(r.get("calibrator_bypassed", False)),
            "coupling_alignment": row.get("coupling_alignment", 0),
            "gap_sentiment_agreement": row.get("gap_sentiment_agreement", 0),
            "days_to_earnings": row.get("days_to_earnings", 99),
            "is_earnings_day": row.get("is_earnings_day", 0),
            "top_signal": determine_top_signal(row),
        })

    if not results:
        print("\n  📭 No valid predictions (missing market data). Saving empty watchlist.")
        results_df = pd.DataFrame(columns=[
            "ticker", "p_spike", "p_spike_raw", "p_spike_trade",
            "expected_abs_return", "expected_signed_return", "expected_value",
            "p_up", "p_down", "p_flat", "calibrator_bypassed",
            "coupling_alignment", "top_signal",
        ])
    else:
        results_df = pd.DataFrame(results).sort_values("p_spike_trade", ascending=False)
        results_df["vix"] = macro_cache.get("vix")
        results_df["tier"] = "flat"
        trade_t = get_trade_threshold(
            float(macro_cache.get("vix", 0)) if macro_cache.get("vix") is not None else None
        )
        pcol = "p_spike_trade"
        results_df.loc[results_df[pcol] >= trade_t, "tier"] = "trade"
        results_df.loc[
            (results_df[pcol] >= WATCHLIST_THRESHOLD_LOW) & (results_df[pcol] < trade_t),
            "tier",
        ] = "watchlist"
        print_watchlist(results_df)

    date_str = today.strftime("%Y-%m-%d")
    csv_path = OUTPUT_DIR / f"watchlist_{date_str}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  📄 Watchlist saved to {csv_path}")

    history_path = OUTPUT_DIR / "spike_history.csv"
    results_df["date"] = date_str
    if history_path.exists():
        existing = pd.read_csv(history_path)
        pd.concat([existing, results_df], ignore_index=True).to_csv(history_path, index=False)
    else:
        results_df.to_csv(history_path, index=False)
    print(f"  📊 History appended to {history_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Pre-Market Spike Detector v2")
    parser.add_argument("--retrain", action="store_true", help="Retrain model")
    parser.add_argument("--force", action="store_true",
                        help="Skip acceptance gates (feature migration deploy)")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip model backup before retrain")
    args = parser.parse_args()
    if args.retrain:
        run_retrain(backup=not args.no_backup, force=args.force)
    else:
        run_predict()


if __name__ == "__main__":
    main()
