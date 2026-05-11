"""
notify.py — Send spike detector watchlist to Telegram.

Usage:
    python notify.py                        # Send today's watchlist
    python notify.py --file output/watchlist_2026-05-06.csv
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import requests

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to watchlist CSV")
    args = parser.parse_args()

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
