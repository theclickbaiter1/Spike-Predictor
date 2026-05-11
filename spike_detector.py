"""
spike_detector.py — Main pipeline for the Pre-Market Spike Detector (v2).

Usage:
    python spike_detector.py              # Daily prediction (8 AM run)
    python spike_detector.py --retrain    # Retrain model from scratch
"""

import argparse
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS, MODEL_PATH, OUTPUT_DIR, TRAINING_DATA_PATH, UNIVERSE


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
    pm = features.get("premarket_change", 0)
    if abs(pm) > 0.02:
        signals.append(f"Pre-mkt {'+'if pm>0 else ''}{pm*100:.1f}%")
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
    pvr = features.get("premarket_volume_ratio", 0)
    if pvr > 2: signals.append(f"PM vol {pvr:.1f}×")
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


def run_retrain():
    from features import build_training_dataset
    from model import TwoStageModel, time_series_split
    from news import FinBERTScorer, FinnhubClient

    print_banner()
    print("  MODE: RETRAIN (2-stage model + adaptive threshold)\n")

    client = FinnhubClient()
    scorer = FinBERTScorer()

    X, y = build_training_dataset(UNIVERSE, client, scorer)

    training_df = X.copy()
    training_df["_target"] = y
    training_df.to_parquet(TRAINING_DATA_PATH)
    print(f"\n  Training data saved to {TRAINING_DATA_PATH}")

    X_train, y_train, X_val, y_val = time_series_split(X, y)

    model = TwoStageModel()
    model.train(X_train, y_train, X_val, y_val)

    print("\n  Top 10 Spike Detection Feature Importances:")
    importance = model.get_spike_feature_importance()
    for feat, score in importance.head(10).items():
        bar = "█" * int(score * 100)
        print(f"    {feat:30s} {score:.3f} {bar}")

    model.retrain_full(X, y)
    model.save()

    print("\n  ✅ Retraining complete.")
    print("═" * 65)


def run_predict():
    from features import build_single_day_features, _download_safe, compute_macro_features
    from model import TwoStageModel
    from news import FinBERTScorer, FinnhubClient
    from premarket import PremarketProvider

    print_banner()
    print("  MODE: PREDICT (2-stage + pre-market)\n")

    s1_path = str(MODEL_PATH).replace(".json", "_s1.json")
    from pathlib import Path
    if not Path(s1_path).exists():
        print("  ❌ No trained model found. Run with --retrain first.")
        sys.exit(1)

    model = TwoStageModel()
    model.load()

    client = FinnhubClient()
    scorer = FinBERTScorer()
    pm_provider = PremarketProvider()

    today = datetime.now()
    end_str = today.strftime("%Y-%m-%d")
    start_str = (today - timedelta(days=60)).strftime("%Y-%m-%d")

    # Pre-market data
    print("  Fetching pre-market data...")
    premarket_data = pm_provider.get_live_premarket(UNIVERSE)

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
            macro_cache=macro_cache, premarket_data=premarket_data,
        )
        feature_rows[ticker] = row
        pm = premarket_data.get(ticker, {})
        pc = pm.get("premarket_change", 0)
        if pc and not np.isnan(pc if isinstance(pc, float) else 0):
            pm_val = pm.get("premarket_price", 0)
            prev = pm.get("prev_close", 0)
            if prev and prev > 0 and pm_val and not np.isnan(pm_val):
                chg = (pm_val - prev) / prev * 100
                print(f"OK (PM: {chg:+.1f}%)")
            else:
                print("OK")
        else:
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
    args = parser.parse_args()
    if args.retrain:
        run_retrain()
    else:
        run_predict()


if __name__ == "__main__":
    main()
