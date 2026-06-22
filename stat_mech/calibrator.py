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

from config import MODEL_PATH

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _softmax3(energies: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """energies shape (n, 3), beta shape (n,) -> probs (n, 3) for [up, flat, down]."""
    scaled = -energies * beta[:, None]
    log_z = logsumexp(scaled, axis=1, keepdims=True)
    return np.exp(scaled - log_z)


def _nll(params: np.ndarray, logit_spike, logit_dir, local_field, coupling, beta, y_class) -> float:
    """Negative log-likelihood for 3-class labels (0=down, 1=flat, 2=up)."""
    a1, a2, a3, a4, a5, beta0 = params
    beta_eff = beta0 * np.maximum(beta, _EPS)

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
        self.fitted = False

    def fit(
        self,
        xgb_probs: pd.DataFrame,
        stat_features: pd.DataFrame,
        y_3class: pd.Series,
        vix: pd.Series | None = None,
    ) -> "BoltzmannCalibrator":
        p_spike = xgb_probs["p_spike"].values
        p_up_gs = np.clip(xgb_probs["p_up"].values / np.maximum(p_spike, _EPS), _EPS, 1 - _EPS)
        logit_spike = _logit(p_spike)
        logit_dir = _logit(p_up_gs)

        local_field = stat_features["local_field"].fillna(0).values
        coupling = stat_features["coupling_alignment"].fillna(0).values
        if vix is not None and "inverse_temperature" in stat_features.columns:
            beta = stat_features["inverse_temperature"].fillna(0).values
        else:
            beta = np.ones(len(logit_spike))

        y = y_3class.values.astype(int)

        def objective(params):
            return _nll(params, logit_spike, logit_dir, local_field, coupling, beta, y)

        res = minimize(objective, self.params, method="L-BFGS-B",
                         bounds=[(0.1, 5)] * 5 + [(0.1, 5)])
        if res.success:
            self.params = res.x
        self.fitted = True
        return self

    def transform(
        self,
        xgb_probs: pd.DataFrame,
        stat_features: pd.DataFrame,
        vix: pd.Series | None = None,
    ) -> pd.DataFrame:
        p_spike = xgb_probs["p_spike"].values
        p_up_gs = np.clip(xgb_probs["p_up"].values / np.maximum(p_spike, _EPS), _EPS, 1 - _EPS)
        logit_spike = _logit(p_spike)
        logit_dir = _logit(p_up_gs)

        local_field = stat_features["local_field"].fillna(0).values
        coupling = stat_features["coupling_alignment"].fillna(0).values
        if vix is not None and "inverse_temperature" in stat_features.columns:
            beta = stat_features["inverse_temperature"].fillna(0).values
        else:
            beta = np.ones(len(logit_spike))

        a1, a2, a3, a4, a5, beta0 = self.params
        beta_eff = beta0 * np.maximum(beta, _EPS)

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

    def nll(self, xgb_probs: pd.DataFrame, stat_features: pd.DataFrame,
            y_3class: pd.Series, vix: pd.Series | None = None) -> float:
        """Mean negative log-likelihood on labeled data."""
        p_spike = xgb_probs["p_spike"].values
        p_up_gs = np.clip(xgb_probs["p_up"].values / np.maximum(p_spike, _EPS), _EPS, 1 - _EPS)
        logit_spike = _logit(p_spike)
        logit_dir = _logit(p_up_gs)
        local_field = stat_features["local_field"].fillna(0).values
        coupling = stat_features["coupling_alignment"].fillna(0).values
        if vix is not None and "inverse_temperature" in stat_features.columns:
            beta = stat_features["inverse_temperature"].fillna(0).values
        else:
            beta = np.ones(len(logit_spike))
        total = _nll(self.params, logit_spike, logit_dir, local_field, coupling, beta, y_3class.values)
        return total / len(y_3class)

    def save(self, path: Path | None = None):
        path = path or Path(str(MODEL_PATH).replace(".json", "_calibrator.json"))
        with open(path, "w") as f:
            json.dump({"params": self.params.tolist(), "fitted": self.fitted}, f)

    def load(self, path: Path | None = None):
        path = path or Path(str(MODEL_PATH).replace(".json", "_calibrator.json"))
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        self.params = np.array(data["params"])
        self.fitted = data.get("fitted", True)
