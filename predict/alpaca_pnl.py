"""Fetch Alpaca paper account equity and realized round-trip P&L."""

from __future__ import annotations

from collections import defaultdict

from predict.trade import AlpacaClient


def account_snapshot(label: str = "open") -> dict:
    """Equity, cash, and open position count for a paper account label."""
    client = AlpacaClient(paper=True, label=label)
    acct = client.get_account()
    if not acct:
        return {}
    positions = client.get_positions() or []
    return {
        "label": label,
        "equity": float(acct["equity"]),
        "last_equity": float(acct.get("last_equity", acct["equity"])),
        "cash": float(acct["cash"]),
        "buying_power": float(acct.get("buying_power", 0)),
        "n_positions": len(positions),
        "daily_pnl": float(acct["equity"]) - float(acct.get("last_equity", acct["equity"])),
    }


def round_trip_pnl(label: str = "open", limit: int = 500) -> dict:
    """
    Reconstruct FIFO round-trip P&L from filled orders.
    Returns cumulative stats and today's realized P&L.
    """
    client = AlpacaClient(paper=True, label=label)
    orders = client._request("GET", f"/v2/orders?status=all&limit={limit}&direction=asc") or []
    fills = [
        o for o in orders
        if o.get("status") == "filled" and float(o.get("filled_qty", 0) or 0) > 0
    ]

    lots: dict[str, list] = defaultdict(list)
    trades = []
    for o in fills:
        sym = o["symbol"]
        qty = float(o["filled_qty"])
        px = float(o.get("filled_avg_price") or 0)
        day = (o.get("filled_at") or o.get("submitted_at") or "")[:10]
        if o["side"] == "buy":
            lots[sym].append({"qty": qty, "px": px, "day": day})
        elif o["side"] == "sell" and lots[sym]:
            lot = lots[sym].pop(0)
            q = min(qty, lot["qty"])
            trades.append({
                "symbol": sym,
                "pnl": (px - lot["px"]) * q,
                "sell_day": day,
            })

    from datetime import datetime
    from zoneinfo import ZoneInfo

    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    today_pnl = sum(t["pnl"] for t in trades if t["sell_day"] == today)
    total_pnl = sum(t["pnl"] for t in trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    return {
        "label": label,
        "round_trips": len(trades),
        "wins": wins,
        "losses": len(trades) - wins,
        "total_realized_pnl": total_pnl,
        "today_realized_pnl": today_pnl,
    }


def both_accounts_summary() -> dict:
    """OPEN + DELAYED equity and realized P&L for daily validation metrics."""
    out = {}
    for label in ("open", "delayed"):
        try:
            snap = account_snapshot(label)
            rt = round_trip_pnl(label)
            if snap:
                out[f"{label}_equity"] = snap["equity"]
                out[f"{label}_daily_pnl"] = snap["daily_pnl"]
                out[f"{label}_positions"] = snap["n_positions"]
            if rt:
                out[f"{label}_today_realized_pnl"] = rt["today_realized_pnl"]
                out[f"{label}_round_trips"] = rt["round_trips"]
        except Exception:
            continue
    return out
