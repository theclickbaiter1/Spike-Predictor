"""
notify.py — Send spike detector watchlist to Telegram.

Usage:
    python predict/notify.py                        # Send today's watchlist
    python predict/notify.py --file output/watchlist_YYYY-MM-DD.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from datetime import datetime

import requests

from config import OUTPUT_DIR, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TRADE_LOG_PATH


def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set.")
        sys.exit(1)

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    })

    if resp.ok:
        print("Telegram message sent.")
    else:
        print(f"Telegram send failed: {resp.status_code} {resp.text}")
        sys.exit(1)


def format_watchlist(csv_path: str) -> str:
    import pandas as pd

    df = pd.read_csv(csv_path)
    date_str = datetime.now().strftime("%Y-%m-%d")

    lines = [f"*Spike Detector — {date_str}*\n"]

    high = df[df["p_spike"] >= 0.60]
    moderate = df[(df["p_spike"] >= 0.40) & (df["p_spike"] < 0.60)]
    low_count = len(df[df["p_spike"] < 0.40])

    if not high.empty:
        lines.append("🔴 *HIGH PROBABILITY (>60%)*")
        for _, r in high.iterrows():
            d = "▲" if r["p_up"] > r["p_down"] else "▼"
            lines.append(f"  `{r['ticker']:<6}` {d} {r['p_spike']*100:.0f}%  — {r['top_signal']}")
        lines.append("")

    if not moderate.empty:
        lines.append("🟡 *MODERATE (40-60%)*")
        for _, r in moderate.iterrows():
            d = "▲" if r["p_up"] > r["p_down"] else "▼"
            lines.append(f"  `{r['ticker']:<6}` {d} {r['p_spike']*100:.0f}%  — {r['top_signal']}")
        lines.append("")

    if high.empty and moderate.empty:
        lines.append("🟢 No significant spikes predicted today.")
    elif low_count > 0:
        lines.append(f"🟢 {low_count} tickers predicted FLAT (<40%)")

    return "\n".join(lines)


def format_trade_summary(csv_path: str = None) -> str:
    """Format today's trades from trade_log.csv for Telegram."""
    import pandas as pd

    path = Path(csv_path) if csv_path else TRADE_LOG_PATH
    if not path.exists():
        return "*Spike Trader* — No trade log found."

    df = pd.read_csv(path)
    if df.empty:
        return "*Spike Trader* — No trades recorded."

    # Filter to today's trades
    today = datetime.now().strftime("%Y-%m-%d")
    df_today = df[df["date"] == today]

    if df_today.empty:
        return f"*Spike Trader — {today}*\n\n📭 No trades placed today."

    lines = [f"*Spike Trader — {today}*\n"]
    lines.append(f"📊 *{len(df_today)} orders placed*\n")

    for _, t in df_today.iterrows():
        d = "▲" if t["direction"] == "LONG" else "▼"
        lines.append(
            f"  `{t['ticker']:<6}` {d} {t['direction']}  "
            f"{t['qty']} shares @ ${t['entry_price']:.2f}\n"
            f"        TP=${t['take_profit']:.2f}  SL=${t['stop_loss']:.2f}  "
            f"P(spike)={t['p_spike']*100:.0f}%"
        )

    mode = df_today.iloc[0].get("mode", "unknown")
    lines.append(f"\n_Mode: {mode}_")

    return "\n".join(lines)


def format_validation_summary(metrics_csv: str = None) -> str:
    """Format the latest validation metrics + rolling trend for Telegram."""
    import pandas as pd

    path = Path(metrics_csv) if metrics_csv else (OUTPUT_DIR / "validation_state" / "metrics.csv")
    if not path.exists():
        return "*Validation* — No metrics yet (first run?)."

    df = pd.read_csv(path)
    if df.empty:
        return "*Validation* — No metrics yet."

    df = df.sort_values("date").reset_index(drop=True)
    today = df.iloc[-1]
    n = int(today["n_tickers"])

    lines = [f"*Validation — {today['date']}*\n"]
    lines.append(f"📊 Accuracy: *{today['accuracy']*100:.1f}%* ({int(today['accuracy']*n)}/{n})")
    lines.append(f"🎯 Spike precision: *{today['spike_precision']*100:.0f}%* "
                 f"({int(today['n_pred_spikes'])} predicted)")
    lines.append(f"🔍 Spike recall: *{today['spike_recall']*100:.0f}%* "
                 f"({int(today['n_actual_spikes'])} actual)")
    lines.append(f"↕️ Direction acc: *{today['direction_accuracy']*100:.0f}%* (on caught spikes)")

    pnl_today = today["long_signal_return"] + today["short_signal_return"]
    pnl_emoji = "🟢" if pnl_today >= 0 else "🔴"
    lines.append(f"{pnl_emoji} Signal P&L: *{pnl_today*100:+.2f}%* "
                 f"(L {today['long_signal_return']*100:+.2f}% / S {today['short_signal_return']*100:+.2f}%)")

    # Rolling 10-day trend
    if len(df) >= 5:
        recent = df.tail(10)
        cum_pnl = (df["long_signal_return"].fillna(0) + df["short_signal_return"].fillna(0)).sum() * 100
        lines.append("")
        lines.append(f"📈 *{len(recent)}d avg:* "
                     f"acc {recent['accuracy'].mean()*100:.1f}%, "
                     f"prec {recent['spike_precision'].mean()*100:.0f}%, "
                     f"rec {recent['spike_recall'].mean()*100:.0f}%")
        lines.append(f"💰 Cumulative signal P&L ({len(df)}d): *{cum_pnl:+.1f}%*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to watchlist CSV")
    parser.add_argument("--trades", action="store_true",
                        help="Send trade execution summary instead of watchlist")
    parser.add_argument("--validate", action="store_true",
                        help="Send daily validation summary instead of watchlist")
    args = parser.parse_args()

    if args.validate:
        msg = format_validation_summary()
        send_telegram(msg)
        return

    if args.trades:
        msg = format_trade_summary()
        send_telegram(msg)
        return

    if args.file:
        csv_path = args.file
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")
        csv_path = OUTPUT_DIR / f"watchlist_{date_str}.csv"

    if not Path(csv_path).exists():
        print(f"ERROR: Watchlist not found at {csv_path}")
        sys.exit(1)

    msg = format_watchlist(str(csv_path))
    send_telegram(msg)


if __name__ == "__main__":
    main()
