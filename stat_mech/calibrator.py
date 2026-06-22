"""
Boltzmann / MaxEnt calibrator — maps XGBoost logits + stat-mech features
to a thermodynamically consistent 3-state distribution.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.metrics import precision_score

from config import MODEL_PATH, TRADE_THRESHOLD
from stat_mech.guards import apply_shrinkage
from stat_mech_features import compute_inverse_temperature

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _beta_from_vix(vix: pd.Series | np.ndarray | None, n: int) -> np.ndarray:
    """Physical β(VIX) = 1/max(VIX, floor) — always positive."""
    if vix is None:
        return np.ones(n)
    if isinstance(vix, pd.Series):
        vix = vix.values
    return np.array([compute_inverse_temperature(v) for v in vix], dtype=float)


def _softmax3(energies: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """energies shape (n, 3), beta shape (n,) -> probs (n, 3) for [up, flat, down]."""
    scaled = -energies * beta[:, None]
    log_z = logsumexp(scaled, axis=1, keepdims=True)
    return np.exp(scaled - log_z)


def _nll(params: np.ndarray, logit_spike, logit_dir, local_field, coupling, beta, y_class) -> float:
    """Negative log-likelihood for 3-class labels (0=down, 1=flat, 2=up)."""
    a1, a2, a3, a4, a5, beta0 = params
    beta_eff = beta0 * beta

    e_up = -(a1 * logit_spike + a2 * logit_dir + a3 * local_field + a4 * coupling)
    e_down = -(a1 * logit_spike - a2 * logit_dir + a3 * local_field - a4 * coupling)
    e_flat = -(a5 * (-logit_spike))
    energies = np.column_stack([e_up, e_flat, e_down])

    probs = _softmax3(energies, beta_eff)
    idx = y_class.astype(int)
    p = probs[np.arange(len(idx)), idx]
    return float(-np.log(np.clip(p, _EPS, 1.0)).sum())


class BoltzmannCalibrator:
    """Fit 3-state Gibbs distribution on validation split; transform at predict time."""

    def __init__(self):
        self.params = np.array([1.0, 1.0, 0.5, 0.5, 1.0, 1.0])
        self.shrink = 1.0
        self.fitted = False

    def fit(
        self,
        xgb_probs: pd.DataFrame,
        stat_features: pd.DataFrame,
        y_3class: pd.Series,
        vix: pd.Series | None = None,
        trade_threshold: float = TRADE_THRESHOLD,
    ) -> "BoltzmannCalibrator":
        p_spike = xgb_probs["p_spike"].values
        p_up_gs = np.clip(xgb_probs["p_up"].values / np.maximum(p_spike, _EPS), _EPS, 1 - _EPS)
        logit_spike = _logit(p_spike)
        logit_dir = _logit(p_up_gs)

        local_field = stat_features["local_field"].fillna(0).values
        coupling = stat_features["coupling_alignment"].fillna(0).values
        beta = _beta_from_vix(vix, len(logit_spike))

        y = y_3class.values.astype(int)

        def objective(params):
            return _nll(params, logit_spike, logit_dir, local_field, coupling, beta, y)

        res = minimize(objective, self.params, method="L-BFGS-B",
                       bounds=[(0.1, 5)] * 5 + [(0.1, 5)])
        if res.success:
            self.params = res.x

        self._fit_shrinkage(xgb_probs, stat_features, y_3class, vix, trade_threshold)
        self.fitted = True
        return self

    def _fit_shrinkage(
        self,
        xgb_probs: pd.DataFrame,
        stat_features: pd.DataFrame,
        y_3class: pd.Series,
        vix: pd.Series | None,
        trade_threshold: float,
    ):
        """Pick shrink ∈ [0.3, 1.0] minimizing NLL while precision >= raw @ threshold."""
        y_spike = (y_3class != 1).astype(int).values
        raw_p = xgb_probs["p_spike"].values
        raw_prec = precision_score(
            y_spike, (raw_p >= trade_threshold).astype(int), zero_division=0,
        )

        boltz = self._transform_boltzmann(xgb_probs, stat_features, vix)
        best_shrink, best_nll = 1.0, float("inf")

        for shrink in np.linspace(0.3, 1.0, 8):
            p_spike = apply_shrinkage(boltz["p_spike"].values, raw_p, shrink)
            prec = precision_score(
                y_spike, (p_spike >= trade_threshold).astype(int), zero_division=0,
            )
            if prec < raw_prec:
                continue
            tmp = boltz.copy()
            tmp["p_spike"] = p_spike
            nll = self.nll_from_probs(tmp, y_3class)
            if nll < best_nll:
                best_nll, best_shrink = nll, float(shrink)

        self.shrink = best_shrink

    def _transform_boltzmann(
        self,
        xgb_probs: pd.DataFrame,
        stat_features: pd.DataFrame,
        vix: pd.Series | None,
    ) -> pd.DataFrame:
        p_spike = xgb_probs["p_spike"].values
        p_up_gs = np.clip(xgb_probs["p_up"].values / np.maximum(p_spike, _EPS), _EPS, 1 - _EPS)
        logit_spike = _logit(p_spike)
        logit_dir = _logit(p_up_gs)

        local_field = stat_features["local_field"].fillna(0).values
        coupling = stat_features["coupling_alignment"].fillna(0).values
        beta = _beta_from_vix(vix, len(logit_spike))

        a1, a2, a3, a4, a5, beta0 = self.params
        beta_eff = beta0 * beta

        e_up = -(a1 * logit_spike + a2 * logit_dir + a3 * local_field + a4 * coupling)
        e_down = -(a1 * logit_spike - a2 * logit_dir + a3 * local_field - a4 * coupling)
        e_flat = -(a5 * (-logit_spike))
        energies = np.column_stack([e_up, e_flat, e_down])
        probs = _softmax3(energies, beta_eff)

        p_up = probs[:, 0]
        p_flat = probs[:, 1]
        p_down = probs[:, 2]
        p_spike_out = p_up + p_down

        return pd.DataFrame({
            "p_spike": p_spike_out,
            "p_up": p_up,
            "p_down": p_down,
            "p_flat": p_flat,
            "p_spike_raw": p_spike,
        }, index=xgb_probs.index)

    def transform(
        self,
        xgb_probs: pd.DataFrame,
        stat_features: pd.DataFrame,
        vix: pd.Series | None = None,
    ) -> pd.DataFrame:
        out = self._transform_boltzmann(xgb_probs, stat_features, vix)
        if self.shrink < 1.0:
            out["p_spike"] = apply_shrinkage(
                out["p_spike"].values, out["p_spike_raw"].values, self.shrink,
            )
            # Rescale directional probs proportionally when shrinking spike mass
            scale = np.clip(
                out["p_spike"].values / np.maximum(out["p_spike_raw"].values, _EPS),
                0.0, 1.0,
            )
            out["p_up"] = out["p_up"].values * scale
            out["p_down"] = out["p_down"].values * scale
            out["p_flat"] = 1.0 - out["p_spike"]
        return out

    def nll_from_probs(self, probs: pd.DataFrame, y_3class: pd.Series) -> float:
        y = y_3class.values.astype(int)
        p3 = np.column_stack([
            probs["p_down"].values, probs["p_flat"].values, probs["p_up"].values,
        ])
        p3 = np.clip(p3, _EPS, 1 - _EPS)
        p3 = p3 / p3.sum(axis=1, keepdims=True)
        return float(-np.log(p3[np.arange(len(y)), y]).mean())

    def nll(self, xgb_probs: pd.DataFrame, stat_features: pd.DataFrame,
            y_3class: pd.Series, vix: pd.Series | None = None) -> float:
        """Mean negative log-likelihood of transformed (shrunk) probs."""
        probs = self.transform(xgb_probs, stat_features, vix)
        return self.nll_from_probs(probs, y_3class)

    def save(self, path: Path | None = None):
        path = path or Path(str(MODEL_PATH).replace(".json", "_calibrator.json"))
        with open(path, "w") as f:
            json.dump({
                "params": self.params.tolist(),
                "shrink": self.shrink,
                "fitted": self.fitted,
            }, f)

    def load(self, path: Path | None = None):
        path = path or Path(str(MODEL_PATH).replace(".json", "_calibrator.json"))
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        self.params = np.array(data["params"])
        self.shrink = float(data.get("shrink", 1.0))
        self.fitted = data.get("fitted", True)
