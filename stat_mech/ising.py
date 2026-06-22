"""
Sector Ising mean-field overlay — blends with calibrated XGBoost probabilities.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ISING_J_MAX,
    ISING_LAMBDA_DEFAULT,
    ISING_M_THRESHOLD,
    MODEL_PATH,
    SECTOR_MAP,
    UNIVERSE,
)


def _sector_neighbors() -> dict[str, list[str]]:
    """Same-sector ticker pairs (excluding self)."""
    neighbors = {t: [] for t in UNIVERSE}
    for t in UNIVERSE:
        sec = SECTOR_MAP.get(t, "XLK")
        neighbors[t] = [u for u in UNIVERSE if u != t and SECTOR_MAP.get(u, "XLK") == sec]
    return neighbors


def sign_returns_from_training(
    intraday_ret: pd.Series,
    tickers: pd.Series,
) -> pd.DataFrame:
    """Pivot intraday returns into date × ticker matrix for coupling estimation."""
    tmp = pd.DataFrame(
        {"ret": intraday_ret.values, "_ticker": tickers.values},
        index=intraday_ret.index,
    )
    pivot = tmp.pivot_table(index=tmp.index, columns="_ticker", values="ret", aggfunc="first")
    return pivot.reindex(columns=UNIVERSE).fillna(0)


def estimate_coupling_matrix(
    sign_returns: pd.DataFrame,
    window: int = 60,
    j_max: float = ISING_J_MAX,
) -> pd.DataFrame:
    """
    J_ij from rolling covariance of sign(intraday_return) / variance.
    sign_returns: index=date, columns=tickers, values in {-1,0,1} or returns.
    """
    signs = np.sign(sign_returns).fillna(0)
    J = pd.DataFrame(0.0, index=UNIVERSE, columns=UNIVERSE)
    recent = signs.tail(window)
    if len(recent) < 10:
        return J

    for i in UNIVERSE:
        if i not in recent.columns:
            continue
        for j in UNIVERSE:
            if i == j or j not in recent.columns:
                continue
            if SECTOR_MAP.get(i) != SECTOR_MAP.get(j):
                continue
            cov = recent[i].cov(recent[j])
            var = recent[j].var()
            if var > 0:
                J.loc[i, j] = np.clip(cov / var, -j_max, j_max)
    return J


def mean_field_magnetizations(
    local_fields: dict[str, float],
    J: pd.DataFrame,
    max_iter: int = 10,
    tol: float = 1e-4,
) -> dict[str, float]:
    """Solve m_i = tanh(h_i + Σ_j J_ij m_j) by iteration."""
    m = {t: 0.0 for t in local_fields}
    neighbors = _sector_neighbors()

    for _ in range(max_iter):
        m_new = {}
        for t, h in local_fields.items():
            field = h
            for u in neighbors.get(t, []):
                if u in J.columns and t in J.index:
                    field += J.loc[t, u] * m.get(u, 0.0)
            m_new[t] = float(np.tanh(field))
        delta = max(abs(m_new[t] - m[t]) for t in m)
        m = m_new
        if delta < tol:
            break
    return m


def ising_probs(magnetizations: dict[str, float], m_threshold: float = ISING_M_THRESHOLD) -> pd.DataFrame:
    """Map |m_i| to spike probability; sign(m) to direction."""
    rows = []
    for t, m in magnetizations.items():
        p_spike = min(1.0, max(0.0, abs(m) - m_threshold) / max(1 - m_threshold, 1e-6))
        if m >= 0:
            p_up, p_down = p_spike, 0.0
        else:
            p_up, p_down = 0.0, p_spike
        rows.append({
            "ticker": t,
            "p_spike_ising": p_spike,
            "p_up_ising": p_up,
            "p_down_ising": p_down,
        })
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


def blend_probs(
    calibrated: pd.DataFrame,
    ising_df: pd.DataFrame,
    lam: float = ISING_LAMBDA_DEFAULT,
) -> pd.DataFrame:
    """P_final = λ P_xgb + (1-λ) P_ising. Index of calibrated must match ising_df index (tickers)."""
    out = calibrated.copy()
    if ising_df.empty:
        out["p_spike_raw"] = out.get("p_spike_raw", out["p_spike"])
        return out

    for col_cal, col_ising in [
        ("p_spike", "p_spike_ising"),
        ("p_up", "p_up_ising"),
        ("p_down", "p_down_ising"),
    ]:
        if col_ising in ising_df.columns:
            ising_aligned = ising_df[col_ising].reindex(out.index).fillna(0)
            out[col_cal] = lam * out[col_cal] + (1 - lam) * ising_aligned

    out["p_flat"] = np.clip(1.0 - out["p_spike"], 0.0, 1.0)
    return out


def blend_panel_day(
    calibrated_day: pd.DataFrame,
    local_fields_day: pd.Series,
    tickers_day: pd.Series,
    J: pd.DataFrame,
    lam: float,
    m_threshold: float,
) -> pd.DataFrame:
    """Blend calibrated probs with Ising for one trading day (panel row index preserved)."""
    h_dict = dict(zip(tickers_day.values, local_fields_day.values))
    m = mean_field_magnetizations(h_dict, J)
    ising_df = ising_probs(m, m_threshold)
    cal_by_ticker = calibrated_day.copy()
    cal_by_ticker.index = tickers_day.values
    blended = blend_probs(cal_by_ticker, ising_df, lam)
    blended.index = calibrated_day.index
    return blended


def blend_panel(
    calibrated: pd.DataFrame,
    local_fields: pd.Series,
    tickers: pd.Series,
    dates: pd.Index,
    J: pd.DataFrame,
    lam: float,
    m_threshold: float,
) -> pd.DataFrame:
    """Blend across a validation panel grouped by date."""
    parts = []
    for date in pd.Index(dates).unique():
        mask = dates == date
        parts.append(
            blend_panel_day(
                calibrated.loc[mask],
                local_fields.loc[mask],
                tickers.loc[mask],
                J,
                lam,
                m_threshold,
            )
        )
    return pd.concat(parts) if parts else calibrated.copy()


class IsingOverlay:
    """Frozen coupling matrix + blend weight; fit λ on validation."""

    def __init__(self):
        self.J = pd.DataFrame(0.0, index=UNIVERSE, columns=UNIVERSE)
        self.lambda_blend = ISING_LAMBDA_DEFAULT
        self.m_threshold = ISING_M_THRESHOLD
        self.fitted = False

    def fit_couplings(self, sign_returns: pd.DataFrame):
        self.J = estimate_coupling_matrix(sign_returns)

    def fit_lambda(
        self,
        calibrated_probs: pd.DataFrame,
        local_fields: pd.Series,
        y_spike_binary: pd.Series,
        tickers: pd.Series | None = None,
        dates: pd.Index | None = None,
        lam_grid: np.ndarray | None = None,
    ):
        """Pick λ maximizing spike F1 on validation."""
        from sklearn.metrics import f1_score

        if lam_grid is None:
            lam_grid = np.linspace(0.5, 1.0, 6)

        y_true = (y_spike_binary != 1).astype(int).values
        use_panel = tickers is not None and dates is not None

        best_lam, best_f1 = self.lambda_blend, -1.0
        for lam in lam_grid:
            if use_panel:
                blended = blend_panel(
                    calibrated_probs, local_fields, tickers, dates,
                    self.J, lam, self.m_threshold,
                )
            else:
                h_dict = local_fields.to_dict()
                m = mean_field_magnetizations(h_dict, self.J)
                ising_df = ising_probs(m, self.m_threshold)
                blended = blend_probs(calibrated_probs, ising_df, lam)

            y_pred = (blended["p_spike"].values >= 0.5).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_lam = f1, lam

        self.lambda_blend = float(best_lam)
        self.fitted = True

    def transform(
        self,
        calibrated_probs: pd.DataFrame,
        local_fields: pd.Series,
        tickers: pd.Series | None = None,
        dates: pd.Index | None = None,
    ) -> pd.DataFrame:
        if tickers is not None and dates is not None and len(pd.Index(dates).unique()) > 1:
            return blend_panel(
                calibrated_probs, local_fields, tickers, dates,
                self.J, self.lambda_blend, self.m_threshold,
            )
        h_dict = (
            dict(zip(tickers.values, local_fields.values))
            if tickers is not None
            else local_fields.to_dict()
        )
        m = mean_field_magnetizations(h_dict, self.J)
        ising_df = ising_probs(m, self.m_threshold)
        cal = calibrated_probs.copy()
        if tickers is not None:
            cal.index = tickers.values
        blended = blend_probs(cal, ising_df, self.lambda_blend)
        if tickers is not None:
            blended.index = calibrated_probs.index
        return blended

    def save(self, path: Path | None = None):
        path = path or Path(str(MODEL_PATH).replace(".json", "_ising.json"))
        data = {
            "lambda_blend": self.lambda_blend,
            "m_threshold": self.m_threshold,
            "fitted": self.fitted,
            "J": self.J.to_dict(),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: Path | None = None):
        path = path or Path(str(MODEL_PATH).replace(".json", "_ising.json"))
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        self.lambda_blend = data.get("lambda_blend", ISING_LAMBDA_DEFAULT)
        self.m_threshold = data.get("m_threshold", ISING_M_THRESHOLD)
        self.fitted = data.get("fitted", False)
        if "J" in data:
            self.J = pd.DataFrame(data["J"]).reindex(index=UNIVERSE, columns=UNIVERSE).fillna(0)
