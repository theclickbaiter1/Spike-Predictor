"""
news.py — Finnhub news fetching + FinBERT sentiment scoring.

Handles API rate limiting, disk caching, and batched FinBERT inference.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests

# Suppress tokenizer parallelism warnings from HuggingFace
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import (
    FINBERT_BATCH_SIZE,
    FINBERT_MODEL_NAME,
    FINNHUB_API_KEY,
    FINNHUB_DELAY_SEC,
    NEWS_CACHE_DIR,
)


class FinnhubClient:
    """
    Lightweight Finnhub news client with rate limiting and disk caching.
    
    Free tier: 60 requests/minute. We throttle to ~1 req/sec and cache
    every response to avoid redundant API calls on retraining.
    """

    BASE_URL = "https://finnhub.io/api/v1/company-news"

    def __init__(self, api_key: str = FINNHUB_API_KEY):
        self.api_key = api_key
        self._last_call_time = 0.0

    def _throttle(self):
        """Enforce minimum delay between API calls."""
        elapsed = time.time() - self._last_call_time
        if elapsed < FINNHUB_DELAY_SEC:
            time.sleep(FINNHUB_DELAY_SEC - elapsed)
        self._last_call_time = time.time()

    def _cache_path(self, ticker: str, from_date: str, to_date: str) -> Path:
        """Return the cache file path for a given ticker + date range."""
        return NEWS_CACHE_DIR / f"{ticker}_{from_date}_{to_date}.json"

    def fetch_news(
        self, ticker: str, from_date: str, to_date: str
    ) -> list[dict]:
        """
        Fetch company news for a ticker within a date range.

        Args:
            ticker: Stock symbol (e.g., "NVDA")
            from_date: Start date "YYYY-MM-DD"
            to_date: End date "YYYY-MM-DD"

        Returns:
            List of news article dicts from Finnhub.
        """
        cache_file = self._cache_path(ticker, from_date, to_date)

        # Return cached data if available
        if cache_file.exists():
            with open(cache_file, "r") as f:
                return json.load(f)

        # No API key → return empty (graceful degradation)
        if not self.api_key:
            return []

        self._throttle()

        try:
            resp = requests.get(
                self.BASE_URL,
                params={
                    "symbol": ticker,
                    "from": from_date,
                    "to": to_date,
                    "token": self.api_key,
                },
                timeout=10,
            )
            resp.raise_for_status()
            articles = resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"  ⚠ Finnhub error for {ticker} ({from_date}→{to_date}): {e}")
            return []

        # Cache to disk
        with open(cache_file, "w") as f:
            json.dump(articles, f)

        return articles

    def fetch_overnight_news(self, ticker: str, date: datetime) -> list[dict]:
        """
        Fetch news published between previous close (4 PM ET day-1) and
        pre-market (9:30 AM ET today). We approximate by fetching the
        previous calendar day and the current day.

        Args:
            ticker: Stock symbol
            date: The trading day (datetime or date)

        Returns:
            List of overnight news articles.
        """
        prev_day = (date - timedelta(days=1)).strftime("%Y-%m-%d")
        curr_day = date.strftime("%Y-%m-%d")
        return self.fetch_news(ticker, prev_day, curr_day)

    def prefetch_ticker_news(
        self, ticker: str, start_date: str, end_date: str
    ) -> list[dict]:
        """
        Bulk-fetch ALL news for a ticker in monthly chunks.
        Each chunk is cached independently. ~24 API calls per ticker
        instead of ~500 per-day calls.

        Returns:
            Flat list of unique articles across the full date range.
        """
        from datetime import datetime as dt

        current = dt.strptime(start_date, "%Y-%m-%d")
        end = dt.strptime(end_date, "%Y-%m-%d")

        all_articles = []
        while current < end:
            chunk_end = min(current + timedelta(days=30), end)
            from_str = current.strftime("%Y-%m-%d")
            to_str = chunk_end.strftime("%Y-%m-%d")
            articles = self.fetch_news(ticker, from_str, to_str)
            all_articles.extend(articles)
            current = chunk_end + timedelta(days=1)

        # Deduplicate by article ID
        seen = set()
        unique = []
        for a in all_articles:
            aid = a.get("id", id(a))
            if aid not in seen:
                seen.add(aid)
                unique.append(a)
        return unique


def filter_articles_for_date(articles: list[dict], date) -> list[dict]:
    """
    Filter articles to the overnight window for a given trading date.
    Returns articles published on (date - 1 day) or date itself.
    """
    if hasattr(date, "to_pydatetime"):
        date = date.to_pydatetime()
    prev_day = date - timedelta(days=1)

    result = []
    for a in articles:
        ts = a.get("datetime", 0)
        if ts == 0:
            continue
        article_date = datetime.utcfromtimestamp(ts).date()
        if article_date == prev_day.date() or article_date == date.date():
            result.append(a)
    return result


def compute_sentiment_from_scores(scores: list[float]) -> dict:
    """Compute the 5 sentiment features from pre-computed score values."""
    if not scores:
        return {
            "overnight_sentiment_mean": 0.0,
            "overnight_sentiment_max": 0.0,
            "overnight_sentiment_min": 0.0,
            "overnight_news_count": 0,
            "overnight_sentiment_std": 0.0,
        }
    arr = np.array(scores)
    return {
        "overnight_sentiment_mean": float(np.mean(arr)),
        "overnight_sentiment_max": float(np.max(arr)),
        "overnight_sentiment_min": float(np.min(arr)),
        "overnight_news_count": len(arr),
        "overnight_sentiment_std": float(np.std(arr)),
    }


class FinBERTScorer:
    """
    Scores financial headlines using the ProsusAI/finbert model.
    
    Outputs a sentiment score in [-1, 1] for each headline:
      -1 = maximally negative
       0 = neutral
      +1 = maximally positive
    """

    def __init__(self):
        self._pipeline = None  # Lazy-loaded

    def _load(self):
        """Lazy-load the FinBERT pipeline to avoid import-time GPU allocation."""
        if self._pipeline is not None:
            return
        from transformers import pipeline

        print("  Loading FinBERT model (first time may download ~400MB)...")
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=FINBERT_MODEL_NAME,
            tokenizer=FINBERT_MODEL_NAME,
            top_k=None,  # Return all class probabilities
            truncation=True,
            max_length=512,
        )
        print("  FinBERT loaded.")

    def score_headlines(self, headlines: list[str]) -> np.ndarray:
        """
        Score a list of headlines.

        Args:
            headlines: List of headline strings.

        Returns:
            np.ndarray of sentiment scores in [-1, 1], one per headline.
        """
        if not headlines:
            return np.array([])

        self._load()

        scores = []
        # Batch to prevent OOM on large sets
        for i in range(0, len(headlines), FINBERT_BATCH_SIZE):
            batch = headlines[i : i + FINBERT_BATCH_SIZE]
            results = self._pipeline(batch)
            for result in results:
                # result is a list of dicts: [{'label': 'positive', 'score': 0.9}, ...]
                score_map = {item["label"]: item["score"] for item in result}
                pos = score_map.get("positive", 0.0)
                neg = score_map.get("negative", 0.0)
                # Net score: positive probability minus negative probability
                scores.append(pos - neg)

        return np.array(scores)


def compute_sentiment_features(
    ticker: str,
    date: datetime,
    client: FinnhubClient,
    scorer: FinBERTScorer,
) -> dict:
    """
    Compute the 5 sentiment features for a ticker on a given date.

    Returns:
        Dict with keys: overnight_sentiment_mean, overnight_sentiment_max,
        overnight_sentiment_min, overnight_news_count, overnight_sentiment_std
    """
    articles = client.fetch_overnight_news(ticker, date)

    # Extract headlines
    headlines = [
        a.get("headline", "").strip()
        for a in articles
        if a.get("headline", "").strip()
    ]

    # Default features when no news is available
    defaults = {
        "overnight_sentiment_mean": 0.0,
        "overnight_sentiment_max": 0.0,
        "overnight_sentiment_min": 0.0,
        "overnight_news_count": 0,
        "overnight_sentiment_std": 0.0,
    }

    if not headlines:
        return defaults

    scores = scorer.score_headlines(headlines)

    return {
        "overnight_sentiment_mean": float(np.mean(scores)),
        "overnight_sentiment_max": float(np.max(scores)),
        "overnight_sentiment_min": float(np.min(scores)),
        "overnight_news_count": len(scores),
        "overnight_sentiment_std": float(np.std(scores)),
    }
