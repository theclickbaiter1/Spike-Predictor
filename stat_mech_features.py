"""
stat_mech_features.py — Statistical mechanics features for spike detection.

Spin representation, sector magnetization, entropy, and thermodynamic regime
proxies. Used by features.py during training and live prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import SECTOR_MAP, STAT_MECH_COLUMNS, UNIVERSE

SPIN_EPS = 0.001


def spin_from_gap(overnight_gap: float) -> float:
    """Map overnight gap to spin σ ∈ {-1, 0, +1}."""
    if pd.isna(overnight_gap) or abs(overnight_gap) < SPIN_EPS:
        return 0.0
    return 1.0 if overnight_gap > 0 else -1.0


def shannon_entropy(counts: np.ndarray) -> float:
    """Shannon entropy H = -Σ p log p for nonnegative counts."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def sentiment_entropy_from_scores(scores: list[float]) -> float:
    """Entropy of FinBERT score histogram (3 bins: neg / neutral / pos)."""
    if not scores:
        return 0.0
    arr = np.asarray(scores, dtype=float)
    neg = (arr < -0.1).sum()
    neu = ((arr >= -0.1) & (arr <= 0.1)).sum()
    pos = (arr > 0.1).sum()
    return shannon_entropy(np.array([neg, neu, pos]))


def cross_section_entropy_from_gaps(gaps: pd.Series) -> float:
    """Market-wide disorder from sign distribution of overnight gaps."""
    spins = gaps.dropna().apply(spin_from_gap)
    up = (spins > 0).sum()
    down = (spins < 0).sum()
    flat = (spins == 0).sum()
    return shannon_entropy(np.array([up, down, flat]))


def compute_local_field(row: pd.Series) -> float:
    """External field h_i on ticker spin (macro + sentiment + gap)."""
    sent = row.get("overnight_sentiment_mean", 0) or 0
    gap = row.get("overnight_gap", 0) or 0
    spy = row.get("sp500_prev_return", 0) or 0
    vix_ch = row.get("vix_change", 0) or 0
    if pd.isna(sent):
        sent = 0
    if pd.isna(gap):
        gap = 0
    if pd.isna(spy):
        spy = 0
    if pd.isna(vix_ch):
        vix_ch = 0
    return float(0.4 * sent + 0.3 * spy + 0.2 * gap + 0.1 * vix_ch)


def compute_inverse_temperature(vix: float, vix_floor: float = 10.0) -> float:
    """β = 1 / max(VIX, floor)."""
    if pd.isna(vix) or vix <= 0:
        vix = vix_floor
    return 1.0 / max(float(vix), vix_floor)


def normalize_inverse_temperature(
    beta: pd.Series,
    mean: float | None = None,
    std: float | None = None,
) -> pd.Series:
    """Z-score inverse temperature using train-set mean/std."""
    if mean is None or std is None or std <= 0:
        return beta
    return (beta - mean) / std


def _positions_for_date(index: pd.Index, date, n: int) -> list[int]:
    """Integer row positions for all rows on a given date (duplicate index safe)."""
    loc = index.get_loc(date)
    if isinstance(loc, np.ndarray):
        if loc.dtype == bool:
            return np.where(loc)[0].tolist()
        return loc.tolist()
    if isinstance(loc, slice):
        return list(range(loc.start or 0, loc.stop or n))
    return [int(loc)]


def enrich_training_frame(
    df: pd.DataFrame,
    beta_mean: float | None = None,
    beta_std: float | None = None,
) -> pd.DataFrame:
    """
    Add stat-mech columns to combined training frame (index=date, _ticker column).
    Uses integer positions for assignment — index has duplicate dates (one per ticker).
    """
    out = df.copy()
    if "_ticker" not in out.columns:
        raise ValueError("enrich_training_frame requires _ticker column")

    out["_spin"] = out["overnight_gap"].apply(spin_from_gap)
    n = len(out)
    sector_mag = np.zeros(n, dtype=float)

    for date, grp in out.groupby(out.index):
        pos_list = _positions_for_date(out.index, date, n)
        tickers = out.iloc[pos_list]["_ticker"].values
        spins = dict(zip(tickers, out.iloc[pos_list]["_spin"].values))
        for pos, ticker in zip(pos_list, tickers):
            sector = SECTOR_MAP.get(ticker, "XLK")
            peer_spins = [
                s for t, s in spins.items()
                if t != ticker and SECTOR_MAP.get(t, "XLK") == sector
            ]
            sector_mag[pos] = float(np.mean(peer_spins)) if peer_spins else 0.0

    xs_map = {
        date: cross_section_entropy_from_gaps(grp["overnight_gap"])
        for date, grp in out.groupby(out.index)
    }
    cross_section = out.index.map(xs_map).astype(float)
    out["cross_section_entropy"] = cross_section
    out["sector_magnetization"] = sector_mag
    out["sector_abs_magnetization"] = np.abs(sector_mag)
    out["coupling_alignment"] = out["_spin"].values * sector_mag
    out["local_field"] = out.apply(compute_local_field, axis=1)

    if "vix" in out.columns:
        beta_raw = out["vix"].apply(compute_inverse_temperature)
    else:
        beta_raw = pd.Series(0.1, index=out.index)
    out["inverse_temperature"] = normalize_inverse_temperature(beta_raw, beta_mean, beta_std)

    sus = np.zeros(n, dtype=float)
    for ticker in out["_ticker"].unique():
        mask = (out["_ticker"] == ticker).values
        rolling_std = pd.Series(sector_mag[mask]).rolling(20, min_periods=5).std().values
        sus[mask] = np.nan_to_num(rolling_std, nan=0.0)
    out["susceptibility_proxy"] = sus
    out["criticality_proxy"] = sus * cross_section

    if "sentiment_entropy" not in out.columns:
        out["sentiment_entropy"] = 0.0

    out.drop(columns=["_spin"], inplace=True, errors="ignore")
    return out


def compute_beta_norm_params(df: pd.DataFrame) -> tuple[float, float]:
    """Train-set mean/std for inverse_temperature normalization."""
    if "vix" not in df.columns:
        return 0.0, 1.0
    beta = df["vix"].apply(compute_inverse_temperature)
    return float(beta.mean()), float(beta.std()) if beta.std() > 0 else 1.0


def build_live_cross_section_cache(
    ticker_gaps: dict[str, float],
    ticker_sentiment_scores: dict[str, list[float]] | None = None,
) -> dict:
    """
    Per-date shared cache for live prediction.
    Keys: cross_section_entropy, per-ticker sector_magnetization, sentiment_entropy.
    """
    gaps = pd.Series(ticker_gaps)
    xs_ent = cross_section_entropy_from_gaps(gaps)

    spins = {t: spin_from_gap(g) for t, g in ticker_gaps.items()}
    sector_mag = {}
    for ticker in ticker_gaps:
        sector = SECTOR_MAP.get(ticker, "XLK")
        peer_spins = [
            spins[t] for t, s in spins.items()
            if t != ticker and SECTOR_MAP.get(t, "XLK") == sector
        ]
        sector_mag[ticker] = float(np.mean(peer_spins)) if peer_spins else 0.0

    sent_ent = {}
    if ticker_sentiment_scores:
        for t, scores in ticker_sentiment_scores.items():
            sent_ent[t] = sentiment_entropy_from_scores(scores)

    return {
        "cross_section_entropy": xs_ent,
        "sector_magnetization": sector_mag,
        "sentiment_entropy": sent_ent,
    }


def attach_live_stat_mech(
    row: dict,
    ticker: str,
    cross_section_cache: dict,
    beta_mean: float = 0.0,
    beta_std: float = 1.0,
) -> dict:
    """Attach stat-mech features to a single-ticker feature row at predict time."""
    spin = spin_from_gap(row.get("overnight_gap", 0))
    m_sector = cross_section_cache.get("sector_magnetization", {}).get(ticker, 0.0)
    xs_ent = cross_section_cache.get("cross_section_entropy", 0.0)
    sent_ent_map = cross_section_cache.get("sentiment_entropy", {})
    sent_ent = sent_ent_map.get(ticker, 0.0)

    vix = row.get("vix", np.nan)
    beta = compute_inverse_temperature(vix)
    beta_z = (beta - beta_mean) / beta_std if beta_std > 0 else beta

    row["sector_magnetization"] = m_sector
    row["sector_abs_magnetization"] = abs(m_sector)
    row["coupling_alignment"] = spin * m_sector
    row["local_field"] = compute_local_field(pd.Series(row))
    row["cross_section_entropy"] = xs_ent
    row["sentiment_entropy"] = sent_ent
    row["inverse_temperature"] = beta_z
    # Live: no history for rolling std — use |m_sector| as proxy
    row["susceptibility_proxy"] = abs(m_sector)
    row["criticality_proxy"] = abs(m_sector) * xs_ent
    return row
