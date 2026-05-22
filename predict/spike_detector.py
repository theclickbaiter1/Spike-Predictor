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

import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS, MODEL_BACKUP_DIR, MODEL_PATH, OUTPUT_DIR, TRAINING_DATA_PATH, UNIVERSE


def print_banner():
    now = datetime.now()
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

    high = results[results["p_spike"] >= 0.60]
    moderate = results[(results["p_spike"] >= 0.40) & (results["p_spike"] < 0.60)]
    flat_count = len(results[results["p_spike"] < 0.40])

    def _table(df, emoji, header):
        if df.empty: return
        print(f"\n  {emoji} {header}")
        print("  ┌────────┬───────────┬──────────┬───────────┬──────────────────────────┐")
        print("  │ Ticker │ Direction │ P(spike) │ P(dir)    │ Top Signal               │")
        print("  ├────────┼───────────┼──────────┼───────────┼──────────────────────────┤")
        for _, r in df.iterrows():
            d = "▲ UP" if r["p_up"] > r["p_down"] else "▼ DOWN"
            pdir = max(r["p_up"], r["p_down"])
            print(f"  │ {r['ticker']:<6} │ {d:<9} │ {r['p_spike']*100:6.1f}%  │ {pdir*100:6.1f}%    │ {r['top_signal'][:24]:<24} │")
        print("  └────────┴───────────┴──────────┴───────────┴──────────────────────────┘")

    _table(high, "🔴", "HIGH PROBABILITY SPIKES (>60%)")
    _table(moderate, "🟡", "MODERATE PROBABILITY (40-60%)")
    if flat_count > 0:
        print(f"\n  🟢 LOW PROBABILITY (<40%) — {flat_count} tickers predicted FLAT")
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
    for col in X.columns:
        if X[col].min() == 0 and X[col].max() == 0:
            print(f"    🔴 {col} — all values are 0!")
            suspicious += 1
    if suspicious == 0:
        print("    ✅ All features have non-trivial value ranges.")

    # Known issue: days_to_earnings lookahead bias
    if "days_to_earnings" in X.columns:
        print(f"\n  ⚠ KNOWN ISSUE: 'days_to_earnings' has lookahead bias.")
        print(f"    The model knows future earnings dates. This inflates backtest performance.")
        print(f"    Consider removing in a future iteration.")

    print("─" * 65 + "\n")


def backup_current_model():
    """Back up the current model files before retraining."""
    from pathlib import Path

    base = str(MODEL_PATH).replace(".json", "")
    s1_path = Path(f"{base}_s1.json")
    s2_path = Path(f"{base}_s2.json")
    meta_path = Path(f"{base}_meta.json")

    if not s1_path.exists():
        print("  No existing model to back up.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    backup_dir = MODEL_BACKUP_DIR / f"backup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for src in [s1_path, s2_path, meta_path, MODEL_PATH]:
        if src.exists():
            shutil.copy2(src, backup_dir / src.name)

    print(f"  📦 Current model backed up to {backup_dir}")


def run_retrain(backup=True):
    from features import build_training_dataset
    from model import TwoStageModel, time_series_split
    from news import FinBERTScorer, FinnhubClient

    print_banner()
    print("  MODE: RETRAIN (2-stage model + adaptive threshold)\n")

    # Back up existing model
    if backup:
        backup_current_model()

    client = FinnhubClient()
    scorer = FinBERTScorer()

    X, y, intraday_ret, _ = build_training_dataset(UNIVERSE, client, scorer)

    training_df = X.copy()
    training_df["_target"] = y
    training_df["_intraday_return"] = intraday_ret
    training_df.to_parquet(TRAINING_DATA_PATH)
    print(f"\n  Training data saved to {TRAINING_DATA_PATH}")

    # Data quality audit
    data_quality_report(X, y, intraday_ret)

    X_train, y_train, X_val, y_val = time_series_split(X, y)
    ret_train = intraday_ret.iloc[:len(X_train)]
    ret_val = intraday_ret.iloc[len(X_train):]

    model = TwoStageModel()
    model.train(X_train, y_train, X_val, y_val, ret_train, ret_val)

    print("\n  Top 10 Spike Detection Feature Importances:")
    importance = model.get_spike_feature_importance()
    for feat, score in importance.head(10).items():
        bar = "█" * int(score * 100)
        print(f"    {feat:30s} {score:.3f} {bar}")

    model.retrain_full(X, y, intraday_ret)
    model.save()

    print("\n  ✅ Retraining complete.")
    print("═" * 65)


def run_predict():
    from features import build_single_day_features, _download_safe, compute_macro_features
    from model import TwoStageModel
    from news import FinBERTScorer, FinnhubClient

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

    today = datetime.now()
    end_str = today.strftime("%Y-%m-%d")
    start_str = (today - timedelta(days=60)).strftime("%Y-%m-%d")

    # Macro
    print("  Fetching macro data...")
    spy_data = _download_safe("^GSPC", start_str, end_str)
    macro_df = compute_macro_features(
        spy_data.index if not spy_data.empty else pd.DatetimeIndex([]),
        start_str, end_str,
    )
    macro_cache = macro_df.iloc[-1].to_dict() if not macro_df.empty else {}

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

    X = pd.DataFrame(feature_rows).T
    X.columns = FEATURE_COLUMNS
    X = X.fillna(0)
    probs = model.predict(X)

    results = []
    for ticker in UNIVERSE:
        r = probs.loc[ticker]
        results.append({
            "ticker": ticker,
            "p_spike": r["p_spike"],
            "p_up": r["p_up"],
            "p_down": r["p_down"],
            "p_flat": r["p_flat"],
            "top_signal": determine_top_signal(feature_rows[ticker]),
        })

    results_df = pd.DataFrame(results).sort_values("p_spike", ascending=False)
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
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip model backup before retrain")
    args = parser.parse_args()
    if args.retrain:
        run_retrain(backup=not args.no_backup)
    else:
        run_predict()


if __name__ == "__main__":
    main()
