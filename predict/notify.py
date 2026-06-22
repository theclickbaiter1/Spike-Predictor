"""
notify.py — Send spike detector messages to all Telegram subscribers.

Subscribers are anyone who has sent /start to the bot. The owner (whose
chat_id is set as TELEGRAM_CHAT_ID in env) is always included. Subscribers
are persisted in data/subscribers.json, cached across GitHub Actions runs.

Each notify.py invocation:
  1. Polls Telegram getUpdates for new /start (or /stop) commands
  2. Registers/unregisters chat_ids; sends a welcome reply to new subscribers
  3. Sends the requested message to all current subscribers + owner

Usage:
    python predict/notify.py                        # Send today's watchlist
    python predict/notify.py --trades               # Trade confirmations
    python predict/notify.py --validate             # Validation summary
    python predict/notify.py --validate-week        # Weekly OOS validation summary
    python predict/notify.py --eod-close            # EOD position close summary
    python predict/notify.py --file <path>          # Custom watchlist CSV
    python predict/notify.py --poll-only            # Just register /start; send no message
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from datetime import datetime, timezone

import requests

from config import DATA_DIR, OUTPUT_DIR, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TRADE_LOG_PATH

SUBSCRIBERS_PATH = DATA_DIR / "subscribers.json"


# ── Subscriber persistence ───────────────────────────────────────────────────

def load_subscribers() -> dict:
    if not SUBSCRIBERS_PATH.exists():
        return {"last_update_id": 0, "subscribers": []}
    try:
        with open(SUBSCRIBERS_PATH) as f:
            state = json.load(f)
        state.setdefault("last_update_id", 0)
        state.setdefault("subscribers", [])
        return state
    except Exception as e:
        print(f"  ! Failed to read {SUBSCRIBERS_PATH}: {e} — starting fresh.")
        return {"last_update_id": 0, "subscribers": []}


def save_subscribers(state: dict):
    SUBSCRIBERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUBSCRIBERS_PATH, "w") as f:
        json.dump(state, f, indent=2)


# ── Low-level Telegram I/O ───────────────────────────────────────────────────

def _send(chat_id, text: str, markdown: bool = True) -> bool:
    if not TELEGRAM_BOT_TOKEN:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    if markdown:
        payload["parse_mode"] = "Markdown"
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if not resp.ok:
            print(f"  ! Send to {chat_id} failed: {resp.status_code} {resp.text[:200]}")
        return resp.ok
    except Exception as e:
        print(f"  ! Send to {chat_id} exception: {e}")
        return False


def poll_and_register() -> dict:
    """Call getUpdates; register /start senders, remove /stop senders."""
    state = load_subscribers()
    if not TELEGRAM_BOT_TOKEN:
        return state

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    params = {
        "offset": state["last_update_id"] + 1,
        "timeout": 0,
        "allowed_updates": json.dumps(["message"]),
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json() if resp.ok else None
    except Exception as e:
        print(f"  ! getUpdates exception: {e}")
        return state

    if not data or not data.get("ok"):
        print(f"  ! getUpdates not-ok: {data}")
        return state

    subs_by_id = {s["chat_id"]: s for s in state["subscribers"]}
    new_count = 0
    removed_count = 0

    for upd in data.get("result", []):
        state["last_update_id"] = max(state["last_update_id"], upd["update_id"])
        msg = upd.get("message") or {}
        text = (msg.get("text") or "").strip()
        chat = msg.get("chat") or {}
        chat_id = chat.get("id")
        if not chat_id:
            continue

        if text.startswith("/start"):
            if chat_id not in subs_by_id:
                sub = {
                    "chat_id": chat_id,
                    "first_name": chat.get("first_name", ""),
                    "username": chat.get("username", ""),
                    "subscribed_at": datetime.now(timezone.utc).isoformat(),
                }
                subs_by_id[chat_id] = sub
                new_count += 1
                # Welcome message (plain text to avoid Markdown escaping issues with names)
                _send(
                    chat_id,
                    f"Welcome{(' ' + sub['first_name']) if sub['first_name'] else ''}! "
                    f"You're subscribed to Spike Detector alerts.\n\n"
                    f"You'll receive:\n"
                    f"• 9:15 AM ET — daily watchlist + paper trade confirmations\n"
                    f"• 4:30 PM ET — post-market validation summary\n\n"
                    f"Send /stop to unsubscribe at any time.",
                    markdown=False,
                )

        elif text.startswith("/stop"):
            if chat_id in subs_by_id:
                del subs_by_id[chat_id]
                removed_count += 1
                _send(chat_id, "You've been unsubscribed. Send /start to resubscribe.", markdown=False)

    state["subscribers"] = list(subs_by_id.values())
    save_subscribers(state)

    if new_count or removed_count:
        print(f"  Subscriber changes: +{new_count} new, -{removed_count} removed. Total: {len(state['subscribers'])}.")
    return state


def send_telegram(text: str):
    """Poll for new subscribers, then broadcast `text` to owner + all subscribers."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set.")
        sys.exit(1)

    state = poll_and_register()

    # Owner is always a recipient (even if they never /start'd the bot)
    recipients = {int(TELEGRAM_CHAT_ID)}
    for s in state["subscribers"]:
        recipients.add(int(s["chat_id"]))

    sent = 0
    for chat_id in recipients:
        if _send(chat_id, text, markdown=True):
            sent += 1

    print(f"Telegram broadcast: {sent}/{len(recipients)} recipient(s) reached.")
    if sent == 0:
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


def format_trade_summary(csv_path: str = None, label: str = "open") -> str:
    """Format today's trades from the per-account trade log for Telegram."""
    import pandas as pd

    if csv_path:
        path = Path(csv_path)
    elif label == "open":
        path = TRADE_LOG_PATH
    else:
        path = OUTPUT_DIR / f"trade_log_{label}.csv"

    tag = label.upper()
    if not path.exists():
        return f"*Spike Trader [{tag}]* — No trade log found."

    df = pd.read_csv(path)
    if df.empty:
        return f"*Spike Trader [{tag}]* — No trades recorded."

    # Filter to today's trades
    today = datetime.now().strftime("%Y-%m-%d")
    df_today = df[df["date"] == today]

    if df_today.empty:
        return f"*Spike Trader [{tag}] — {today}*\n\n📭 No trades placed today."

    lines = [f"*Spike Trader [{tag}] — {today}*\n"]
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


def format_eod_close_summary() -> str:
    """Format EOD position close logs for both accounts."""
    import json

    today = datetime.now().strftime("%Y-%m-%d")
    lines = [f"*EOD Close — {today}*\n"]

    for label in ("open", "delayed"):
        path = OUTPUT_DIR / f"eod_close_{label}.json"
        tag = label.upper()
        if not path.exists():
            lines.append(f"*{tag}:* No positions closed (or no log).")
            continue
        with open(path) as f:
            records = json.load(f)
        if not records:
            lines.append(f"*{tag}:* No open positions.")
            continue
        lines.append(f"*{tag}:* Closed {len(records)} position(s)")
        for r in records:
            pl = r.get("unrealized_pl", 0)
            pl_emoji = "🟢" if pl >= 0 else "🔴"
            lines.append(
                f"  `{r['ticker']:<6}` {r['direction']} {r['qty']} sh  "
                f"{pl_emoji} P&L ${pl:+.2f}"
            )

    return "\n".join(lines)


def format_weekly_validation_summary(csv_path: str = None) -> str:
    """Format the latest weekly OOS validation results for Telegram."""
    import pandas as pd

    if csv_path:
        path = Path(csv_path)
    else:
        weekly_dir = OUTPUT_DIR / "validation_state" / "weekly"
        if weekly_dir.exists():
            candidates = sorted(weekly_dir.glob("validation_*.csv"), reverse=True)
            path = candidates[0] if candidates else None
        else:
            path = None

    if not path or not path.exists():
        return "*Weekly Validation* — No results yet."

    df = pd.read_csv(path)
    if df.empty:
        return "*Weekly Validation* — Empty results file."

    total = len(df)
    correct = int(df["correct"].sum())
    actual_spikes = df[df["actual_class"] != "FLAT"]
    pred_spikes = df[df["pred_class"] != "FLAT"]
    caught = actual_spikes[actual_spikes["pred_class"] != "FLAT"]
    true_pos = pred_spikes[pred_spikes["actual_class"] != "FLAT"]
    dir_correct = true_pos[true_pos["pred_class"] == true_pos["actual_class"]]

    date_range = f"{df['date'].min()} → {df['date'].max()}"
    lines = [f"*Weekly OOS Validation — {date_range}*\n"]
    lines.append(f"📊 Accuracy: *{correct/total*100:.1f}%* ({correct}/{total})")
    lines.append(f"🎯 Spike precision: *{len(true_pos)/len(pred_spikes)*100:.0f}%* "
                 f"({len(pred_spikes)} predicted)" if len(pred_spikes) else "🎯 Spike precision: N/A")
    lines.append(f"🔍 Spike recall: *{len(caught)/len(actual_spikes)*100:.0f}%* "
                 f"({len(actual_spikes)} actual)" if len(actual_spikes) else "🔍 Spike recall: N/A")
    if len(true_pos):
        lines.append(f"↕️ Direction acc: *{len(dir_correct)/len(true_pos)*100:.0f}%* (on true spikes)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to watchlist CSV")
    parser.add_argument("--trades", action="store_true",
                        help="Send trade execution summary instead of watchlist")
    parser.add_argument("--validate", action="store_true",
                        help="Send daily validation summary instead of watchlist")
    parser.add_argument("--validate-week", action="store_true",
                        help="Send weekly OOS validation summary")
    parser.add_argument("--eod-close", action="store_true",
                        help="Send EOD position close summary")
    parser.add_argument("--poll-only", action="store_true",
                        help="Poll for /start and /stop commands; do not send any broadcast")
    parser.add_argument("--label", type=str, default="open",
                        help="Account label for --trades summary (e.g. 'open', 'delayed').")
    args = parser.parse_args()

    if args.poll_only:
        state = poll_and_register()
        print(f"Subscribers: {len(state['subscribers'])}")
        return

    if args.validate:
        msg = format_validation_summary()
        send_telegram(msg)
        return

    if args.validate_week:
        msg = format_weekly_validation_summary()
        send_telegram(msg)
        return

    if args.eod_close:
        msg = format_eod_close_summary()
        send_telegram(msg)
        return

    if args.trades:
        msg = format_trade_summary(label=args.label)
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
