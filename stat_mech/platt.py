"""
Platt scaling — simple logistic calibration on raw spike probabilities.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Map raw P(spike) through Platt scaling (logistic on logit)."""

    def __init__(self):
        self.model = LogisticRegression(C=1e10, max_iter=1000)
        self.fitted = False

    def fit(self, p_raw: np.ndarray, y_spike: np.ndarray) -> "PlattCalibrator":
        x = _logit(np.asarray(p_raw, dtype=float)).reshape(-1, 1)
        y = np.asarray(y_spike, dtype=int)
        if len(np.unique(y)) < 2:
            self.fitted = False
            return self
        self.model.fit(x, y)
        self.fitted = True
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return np.asarray(p_raw, dtype=float)
        x = _logit(np.asarray(p_raw, dtype=float)).reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1]

    def save(self, path: Path):
        if not self.fitted:
            return
        coef = self.model.coef_.ravel().tolist()
        intercept = float(self.model.intercept_[0])
        with open(path, "w") as f:
            json.dump({"coef": coef, "intercept": intercept, "fitted": True}, f)

    def load(self, path: Path):
        if not path.exists():
            self.fitted = False
            return
        with open(path) as f:
            data = json.load(f)
        if not data.get("fitted"):
            self.fitted = False
            return
        self.model.coef_ = np.array([data["coef"]])
        self.model.intercept_ = np.array([data["intercept"]])
        self.fitted = True


def apply_platt_to_probs(raw: pd.DataFrame, calibrator: PlattCalibrator) -> pd.DataFrame:
    """Return probability frame with Platt-calibrated p_spike."""
    out = raw.copy()
    p_cal = calibrator.transform(raw["p_spike"].values)
    out["p_spike_raw"] = raw["p_spike"]
    out["p_spike"] = p_cal
    p_up_gs = np.clip(raw["p_up"].values / np.maximum(raw["p_spike"].values, _EPS), _EPS, 1 - _EPS)
    out["p_up"] = p_cal * p_up_gs
    out["p_down"] = p_cal * (1 - p_up_gs)
    out["p_flat"] = 1 - p_cal
    return out
