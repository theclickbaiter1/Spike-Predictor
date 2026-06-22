"""
Degeneracy guards for stat-mech calibration at inference time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_calibrator_degeneracy(p_spike_cal: pd.Series | np.ndarray) -> bool:
    """
    Detect daily collapse when calibrated probs cluster (Jun 18 pattern).
    Triggers when ALL conditions hold across the cross-section.
    """
    p = np.asarray(p_spike_cal, dtype=float)
    if len(p) < 5:
        return False

    std_p = float(np.std(p))
    mean_p = float(np.mean(p))
    median_p = float(np.median(p))
    near_median = float(np.mean(np.abs(p - median_p) < 0.02))

    return std_p < 0.03 and mean_p > 0.45 and near_median > 0.5


def apply_shrinkage(p_boltz: np.ndarray, p_raw: np.ndarray, shrink: float) -> np.ndarray:
    """Blend Boltzmann p_spike toward raw XGBoost."""
    shrink = float(np.clip(shrink, 0.0, 1.0))
    return shrink * p_boltz + (1.0 - shrink) * p_raw
