"""
validate_today.py — Retrain with Finnhub sentiment, predict today, compare with reality.

Usage:
    FINNHUB_API_KEY="your_key" python3.13 validate_today.py
"""
import os
os.environ["FINNHUB_API_KEY"] = os.environ.get("FINNHUB_API_KEY", "")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import matplotlib
matplotlib.use("Agg")

import sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from config import (
    FEATURE_COLUMNS, LABEL_NAMES, MODEL_PATH, OUTPUT_DIR,
    SPIKE_THRESHOLD, TRAINING_DATA_PATH, UNIVERSE,
)


def main():
    from features import (
        build_training_dataset, build_single_day_features,
        _download_safe, compute_macro_features,
    )
    from model import (
        load_model, predict, retrain_full, save_model,
        time_series_split, train_model, get_feature_importance,
    )
    from news import FinBERTScorer, FinnhubClient

    today = datetime(2026, 5, 6)
    yesterday_str = "2026-05-05"
    today_str = today.strftime("%Y-%m-%d")
    tomorrow_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    print("\n" + "=" * 70)
    print(f"  SPIKE DETECTOR — FULL VALIDATION WITH SENTIMENT")
    print(f"  Training cutoff: {yesterday_str} (no today data)")
    print(f"  Prediction date: {today_str}")
    print("=" * 70)

    client = FinnhubClient()
    scorer = FinBERTScorer()

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Retrain with sentiment (excluding today)
    # ═══════════════════════════════════════════════════════════════════
    print("\n  ▶ PHASE 1: Retraining model WITH Finnhub sentiment...")
    print("    (bulk-fetching news in monthly chunks — ~648 API calls)\n")

    X, y = build_training_dataset(UNIVERSE, client, scorer)

    # Exclude today's data (if any leaked in)
    if hasattr(X, 'index'):
        today_ts = pd.Timestamp(today_str)
        mask = X.index < today_ts
        X = X[mask]
        y = y[mask]
        print(f"\n  After excluding today: {len(X)} rows")

    # Save training data
    training_df = X.copy()
    training_df["_target"] = y
    training_df.to_parquet(TRAINING_DATA_PATH)

    # Time-series split + train
    X_train, y_train, X_val, y_val = time_series_split(X, y)
    model, best_rounds = train_model(X_train, y_train, X_val, y_val)

    # Feature importance
    print("\n  Top 10 Feature Importances:")
    importance = get_feature_importance(model)
    for feat, score in importance.head(10).items():
        bar = "█" * int(score * 100)
        print(f"    {feat:30s} {score:.3f} {bar}")

    # Retrain on full (excluding today) and save
    final_model = retrain_full(X, y, best_rounds)
    save_model(final_model)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: Predict today's spikes (with sentiment)
    # ═══════════════════════════════════════════════════════════════════
    print("\n  ▶ PHASE 2: Predicting today's spikes...\n")

    # Macro data (pre-fetch for efficiency)
    start_str = (today - timedelta(days=60)).strftime("%Y-%m-%d")
    spy_data = _download_safe("^GSPC", start_str, yesterday_str)
    macro_df = compute_macro_features(
        spy_data.index if not spy_data.empty else pd.DatetimeIndex([]),
        start_str, yesterday_str,
    )
    macro_cache = macro_df.iloc[-1].to_dict() if not macro_df.empty else {}

    feature_rows = {}
    for i, ticker in enumerate(UNIVERSE):
        print(f"    [{i+1}/{len(UNIVERSE)}] {ticker}...", end=" ", flush=True)
        row = build_single_day_features(
            ticker, today, client, scorer, macro_cache=macro_cache
        )
        feature_rows[ticker] = row
        nc = row.get("overnight_news_count", 0)
        sm = row.get("overnight_sentiment_mean", 0)
        print(f"OK (news={int(nc)}, sent={sm:+.2f})")

    X_pred = pd.DataFrame(feature_rows).T
    X_pred.columns = FEATURE_COLUMNS
    X_pred = X_pred.fillna(0)
    probs = predict(final_model, X_pred)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: Fetch actual results and compare
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n  ▶ PHASE 3: Fetching actual market data for {today_str}...\n")

    actuals = {}
    for ticker in UNIVERSE:
        data = _download_safe(ticker, today_str, tomorrow_str)
        if not data.empty and len(data) > 0:
            r = data.iloc[-1]
            o, c = float(r["Open"]), float(r["Close"])
            ret = (c - o) / o if o > 0 else 0
            cls = "SPIKE UP" if ret >= SPIKE_THRESHOLD else (
                "SPIKE DOWN" if ret <= -SPIKE_THRESHOLD else "FLAT"
            )
            actuals[ticker] = {"open": o, "close": c, "return": ret, "class": cls}
        else:
            actuals[ticker] = {"open": 0, "close": 0, "return": 0, "class": "N/A"}

    # Build results table
    results = []
    for ticker in UNIVERSE:
        p_down = probs.loc[ticker, "p_spike_down"]
        p_up = probs.loc[ticker, "p_spike_up"]
        spike_prob = max(p_up, p_down)
        pred_dir = "UP" if p_up > p_down else "DOWN"
        a = actuals[ticker]
        pred_label = f"SPIKE {pred_dir}" if spike_prob >= 0.40 else "FLAT"
        correct = pred_label == a["class"] if a["class"] != "N/A" else None
        results.append({
            "ticker": ticker, "pred_dir": pred_dir, "spike_prob": spike_prob,
            "p_up": p_up, "p_down": p_down,
            "actual_return": a["return"], "actual_class": a["class"],
            "pred_label": pred_label, "correct": correct,
            "sent_mean": feature_rows[ticker].get("overnight_sentiment_mean", 0),
            "news_count": feature_rows[ticker].get("overnight_news_count", 0),
        })

    results.sort(key=lambda x: x["spike_prob"], reverse=True)
    df = pd.DataFrame(results)

    # Print table
    print(f"{'=' * 78}")
    print(f"  PREDICTION vs REALITY — {today_str}")
    print(f"{'=' * 78}\n")
    print(f"  {'Ticker':<7} {'Pred':<10} {'Prob':>6} {'Sent':>6} {'News':>4} {'Actual':>8} {'Class':<11} {'':>4}")
    print(f"  {'─'*7} {'─'*10} {'─'*6} {'─'*6} {'─'*4} {'─'*8} {'─'*11} {'─'*4}")

    correct_ct = total_ct = 0
    for r in results:
        tier = "🔴" if r["spike_prob"] >= 0.6 else ("🟡" if r["spike_prob"] >= 0.4 else "  ")
        mark = "✅" if r["correct"] else "❌" if r["correct"] is not None else "?"
        print(
            f"{tier} {r['ticker']:<7} {'▲ '+r['pred_dir'] if r['pred_dir']=='UP' else '▼ '+r['pred_dir']:<10}"
            f" {r['spike_prob']*100:5.1f}% {r['sent_mean']:+5.2f} {int(r['news_count']):4d}"
            f" {r['actual_return']*100:+7.2f}% {r['actual_class']:<11} {mark}"
        )
        if r["correct"] is not None:
            total_ct += 1
            correct_ct += int(r["correct"])

    acc = correct_ct / total_ct * 100 if total_ct else 0
    actual_spikes = sum(1 for r in results if r["actual_class"] in ("SPIKE UP", "SPIKE DOWN"))
    pred_spikes = sum(1 for r in results if r["spike_prob"] >= 0.40)
    caught = sum(
        1 for r in results
        if r["spike_prob"] >= 0.40 and r["actual_class"] in ("SPIKE UP", "SPIKE DOWN")
    )
    dir_correct = sum(
        1 for r in results if r["correct"]
        and r["actual_class"] in ("SPIKE UP", "SPIKE DOWN")
    )

    print(f"\n{'=' * 78}")
    print(f"  Overall accuracy:        {correct_ct}/{total_ct} ({acc:.1f}%)")
    print(f"  Actual spikes today:     {actual_spikes}/{len(results)}")
    print(f"  Predicted spikes (≥40%): {pred_spikes}")
    print(f"  Spikes caught:           {caught}/{actual_spikes}")
    print(f"  Direction correct:       {dir_correct}/{actual_spikes}")
    print(f"{'=' * 78}")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 4: Matplotlib Visualizations
    # ═══════════════════════════════════════════════════════════════════
    print("\n  ▶ PHASE 4: Generating visualizations...\n")

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        f"Spike Detector Validation — {today_str}\n"
        f"Model trained with Finnhub + FinBERT sentiment (cutoff: {yesterday_str})",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # ── Plot 1: Predicted Probability vs Actual Return (bar chart) ────
    ax1 = fig.add_subplot(2, 2, 1)

    tickers_sorted = [r["ticker"] for r in results]
    spike_probs = [r["spike_prob"] * 100 for r in results]
    actual_rets = [r["actual_return"] * 100 for r in results]
    bar_colors = []
    for r in results:
        if r["actual_class"] == "SPIKE UP":
            bar_colors.append("#2ecc71")
        elif r["actual_class"] == "SPIKE DOWN":
            bar_colors.append("#e74c3c")
        else:
            bar_colors.append("#95a5a6")

    x = np.arange(len(tickers_sorted))
    bars = ax1.bar(x, spike_probs, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax1.axhline(y=60, color="#e74c3c", linestyle="--", alpha=0.5, label="High threshold (60%)")
    ax1.axhline(y=40, color="#f39c12", linestyle="--", alpha=0.5, label="Moderate threshold (40%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tickers_sorted, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Predicted Spike Probability (%)")
    ax1.set_title("Predicted Spike Probability by Ticker", fontweight="bold")
    ax1.legend(fontsize=8)
    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="Actual Spike UP"),
        mpatches.Patch(color="#e74c3c", label="Actual Spike DOWN"),
        mpatches.Patch(color="#95a5a6", label="Actual FLAT"),
    ]
    ax1.legend(handles=legend_patches + ax1.get_legend_handles_labels()[0][-2:], fontsize=7, loc="upper right")

    # ── Plot 2: Actual Intraday Returns ──────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    ret_colors = ["#2ecc71" if r > 0 else "#e74c3c" for r in actual_rets]
    ax2.bar(x, actual_rets, color=ret_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax2.axhline(y=SPIKE_THRESHOLD * 100, color="#2ecc71", linestyle="--", alpha=0.5, label=f"+{SPIKE_THRESHOLD*100:.0f}% threshold")
    ax2.axhline(y=-SPIKE_THRESHOLD * 100, color="#e74c3c", linestyle="--", alpha=0.5, label=f"-{SPIKE_THRESHOLD*100:.0f}% threshold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tickers_sorted, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Actual Intraday Return (%)")
    ax2.set_title("Actual Intraday Returns", fontweight="bold")
    ax2.legend(fontsize=8)

    # ── Plot 3: Scatter — Predicted Probability vs Actual Return ─────
    ax3 = fig.add_subplot(2, 2, 3)
    scatter_colors = []
    for r in results:
        if r["correct"]:
            scatter_colors.append("#2ecc71")
        elif r["correct"] is False:
            scatter_colors.append("#e74c3c")
        else:
            scatter_colors.append("#95a5a6")

    signed_probs = [
        r["spike_prob"] * 100 * (1 if r["pred_dir"] == "UP" else -1) for r in results
    ]
    ax3.scatter(signed_probs, actual_rets, c=scatter_colors, s=80, alpha=0.8, edgecolors="white", linewidth=0.5)
    for i, r in enumerate(results):
        ax3.annotate(r["ticker"], (signed_probs[i], actual_rets[i]),
                     fontsize=6, ha="center", va="bottom", alpha=0.7)
    ax3.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax3.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
    ax3.axhline(y=3, color="#2ecc71", linestyle="--", alpha=0.3)
    ax3.axhline(y=-3, color="#e74c3c", linestyle="--", alpha=0.3)
    ax3.set_xlabel("Signed Predicted Probability (%)")
    ax3.set_ylabel("Actual Intraday Return (%)")
    ax3.set_title("Prediction vs Reality", fontweight="bold")
    legend_s = [
        mpatches.Patch(color="#2ecc71", label="Correct ✅"),
        mpatches.Patch(color="#e74c3c", label="Wrong ❌"),
    ]
    ax3.legend(handles=legend_s, fontsize=8)

    # ── Plot 4: Summary Stats ────────────────────────────────────────
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")

    summary_text = (
        f"{'━' * 42}\n"
        f"  VALIDATION SUMMARY — {today_str}\n"
        f"{'━' * 42}\n\n"
        f"  Overall Accuracy:        {correct_ct}/{total_ct} ({acc:.1f}%)\n\n"
        f"  Actual Spikes Today:     {actual_spikes}/{len(results)}\n"
        f"  Predicted Spikes (≥40%): {pred_spikes}\n"
        f"  Spikes Caught:           {caught}/{actual_spikes}\n"
        f"  Direction Correct:       {dir_correct}/{actual_spikes}\n\n"
        f"  Spike Threshold:         ±{SPIKE_THRESHOLD*100:.0f}%\n"
        f"  Model:                   XGBoost 3-class\n"
        f"  Features:                {len(FEATURE_COLUMNS)} (with sentiment)\n"
        f"  Training Rows:           {len(X)}\n"
        f"{'━' * 42}\n\n"
    )

    # Add per-ticker spike breakdown
    spike_tickers = [r for r in results if r["actual_class"] in ("SPIKE UP", "SPIKE DOWN")]
    if spike_tickers:
        summary_text += "  ACTUAL SPIKES:\n"
        for r in spike_tickers:
            caught_mark = "CAUGHT ✓" if r["spike_prob"] >= 0.40 else "MISSED ✗"
            summary_text += f"    {r['ticker']:6s} {r['actual_return']*100:+5.1f}%  → {caught_mark}\n"

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6"))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = OUTPUT_DIR / "validation_2026-05-06.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  📊 Visualization saved to {output_path}")

    # Save results CSV
    csv_path = OUTPUT_DIR / "validation_2026-05-06.csv"
    df.to_csv(csv_path, index=False)
    print(f"  📄 Results saved to {csv_path}")

    print("\n  ✅ Validation complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
