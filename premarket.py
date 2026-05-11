"""
premarket.py — Pre-market data provider using Alpaca, Finnhub, and yfinance.

Live (8 AM): Alpaca real-time → Finnhub /quote fallback
Training:    yfinance hourly with prepost=True
"""

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY,
    FINNHUB_API_KEY, FINNHUB_DELAY_SEC,
)


class PremarketProvider:
    """Fetches pre-market price and volume from multiple sources."""

    def __init__(self):
        self._alpaca = None
        if ALPACA_API_KEY and ALPACA_SECRET_KEY:
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                self._alpaca = StockHistoricalDataClient(
                    ALPACA_API_KEY, ALPACA_SECRET_KEY
                )
                print("  Alpaca client initialized.")
            except Exception as e:
                print(f"  ⚠ Alpaca init failed: {e}")

    # ── Live pre-market (for daily 8 AM prediction) ──────────────────────

    def get_live_premarket(self, tickers: list[str]) -> dict:
        """
        Get current pre-market data for all tickers.
        Returns: {ticker: {"premarket_price": float, "prev_close": float,
                           "premarket_volume": float}}
        """
        if self._alpaca:
            return self._alpaca_live(tickers)
        if FINNHUB_API_KEY:
            return self._finnhub_live(tickers)
        return {t: {"premarket_price": np.nan, "prev_close": np.nan,
                     "premarket_volume": np.nan} for t in tickers}

    def _alpaca_live(self, tickers: list[str]) -> dict:
        """Fetch latest snapshots from Alpaca."""
        from alpaca.data.requests import StockSnapshotRequest
        try:
            req = StockSnapshotRequest(symbol_or_symbols=tickers)
            snapshots = self._alpaca.get_stock_snapshot(req)
            result = {}
            for ticker in tickers:
                if ticker in snapshots:
                    snap = snapshots[ticker]
                    price = snap.latest_trade.price if snap.latest_trade else np.nan
                    prev = snap.previous_daily_bar.close if snap.previous_daily_bar else np.nan
                    vol = snap.minute_bar.volume if snap.minute_bar else 0
                    result[ticker] = {
                        "premarket_price": price,
                        "prev_close": prev,
                        "premarket_volume": float(vol),
                    }
                else:
                    result[ticker] = {"premarket_price": np.nan, "prev_close": np.nan,
                                      "premarket_volume": np.nan}
            return result
        except Exception as e:
            print(f"  ⚠ Alpaca snapshot failed: {e}")
            return self._finnhub_live(tickers)

    def _finnhub_live(self, tickers: list[str]) -> dict:
        """Fallback: Finnhub /quote endpoint."""
        result = {}
        for ticker in tickers:
            try:
                time.sleep(FINNHUB_DELAY_SEC)
                resp = requests.get(
                    "https://finnhub.io/api/v1/quote",
                    params={"symbol": ticker, "token": FINNHUB_API_KEY},
                    timeout=10,
                )
                data = resp.json()
                result[ticker] = {
                    "premarket_price": data.get("c", np.nan),
                    "prev_close": data.get("pc", np.nan),
                    "premarket_volume": np.nan,  # Finnhub doesn't provide this
                }
            except Exception:
                result[ticker] = {"premarket_price": np.nan, "prev_close": np.nan,
                                  "premarket_volume": np.nan}
        return result

    # ── Historical pre-market (for training) ─────────────────────────────

    @staticmethod
    def get_historical_premarket(ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Download hourly data with extended hours from yfinance.
        Extract daily pre-market features: premarket_price, premarket_volume.

        Returns DataFrame indexed by date with columns:
            premarket_price, premarket_volume
        """
        # yfinance limits hourly data to 730 days — clamp start date
        from datetime import datetime as dt
        max_start = (dt.now() - timedelta(days=725)).strftime("%Y-%m-%d")
        clamped_start = max(start, max_start)

        try:
            data = yf.download(
                ticker, start=clamped_start, end=end,
                interval="1h", prepost=True, progress=False,
            )
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
        except Exception:
            return pd.DataFrame()

        if data.empty:
            return pd.DataFrame()

        # yfinance returns timezone-aware index for intraday data
        if data.index.tz is not None:
            data.index = data.index.tz_convert("America/New_York")
        else:
            data.index = data.index.tz_localize("America/New_York", ambiguous="NaT")

        daily_features = []
        for date, group in data.groupby(data.index.date):
            # Pre-market: before 9:30 AM ET
            premarket = group[group.index.hour < 9]
            # Also include the 9:00-9:29 bar
            h9 = group[(group.index.hour == 9) & (group.index.minute < 30)]
            premarket = pd.concat([premarket, h9])

            if premarket.empty:
                daily_features.append({
                    "date": pd.Timestamp(date),
                    "premarket_price": np.nan,
                    "premarket_volume": np.nan,
                })
                continue

            daily_features.append({
                "date": pd.Timestamp(date),
                "premarket_price": float(premarket.iloc[-1]["Close"]),
                "premarket_volume": float(premarket["Volume"].sum()),
            })

        if not daily_features:
            return pd.DataFrame()

        return pd.DataFrame(daily_features).set_index("date")
