"""
model.py — Two-stage XGBoost model for spike detection.

Stage 1: Binary spike detector (spike vs flat)
         — excludes realized_vol_20d (prevents "always flag volatile stocks" bias)
Stage 2: Direction classifier (up vs down)
         — trained on spike + near-spike samples (not all samples, not spike-only)

Stat-mech layers (optional): Boltzmann calibrator + sector Ising mean-field blend.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score

from config import (
    FEATURE_COLUMNS, MODEL_PATH, SPIKE_THRESHOLD, UNIVERSE,
    VAL_FRACTION, XGB_PARAMS_S1, XGB_PARAMS_S2,
    S1_POS_WEIGHT_MULTIPLIER, get_trade_threshold,
)
from stat_mech.calibrator import BoltzmannCalibrator
from stat_mech.guards import detect_calibrator_degeneracy
from stat_mech.ising import IsingOverlay

# Stage 1 sees all features except realized_vol_20d and inverse_temperature
# (β(VIX) is used downstream in the Boltzmann calibrator).
_S1_EXCLUDE = {"realized_vol_20d", "inverse_temperature"}
S1_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in _S1_EXCLUDE]


def _align_to_model_features(X: pd.DataFrame, model: xgb.XGBClassifier) -> pd.DataFrame:
    """Subset/reorder columns to match a saved XGBoost model (handles feature migrations)."""
    feat_names = model.get_booster().feature_names
    if not feat_names:
        return X
    aligned = X.copy()
    for col in feat_names:
        if col not in aligned.columns:
            aligned[col] = 0.0
    return aligned[list(feat_names)]


def time_series_split(X, y, val_frac=VAL_FRACTION):
    n = len(X)
    split_idx = int(n * (1 - val_frac))
    print(f"  Time-series split: Train={split_idx}, Val={n - split_idx} ({val_frac*100:.0f}%)")
    return X.iloc[:split_idx], y.iloc[:split_idx], X.iloc[split_idx:], y.iloc[split_idx:]


def _near_spike_mask(y, intraday_ret, adaptive_threshold=None, threshold_frac=0.35):
    """
    Select spike samples + near-spike samples (return > 35% of spike threshold).
    This gives Stage 2 enough data to learn direction without flat-day bias.
    """
    is_spike = y != 1
    if adaptive_threshold is not None:
        near_thresh = adaptive_threshold * threshold_frac
    else:
        near_thresh = SPIKE_THRESHOLD * threshold_frac
    is_near_spike = (~is_spike) & (intraday_ret.abs() >= near_thresh)
    mask = is_spike | is_near_spike
    return mask


class TwoStageModel:
    """
    Stage 1: Binary — is there a spike? (spike=1 vs flat=0)
    Stage 2: Binary — if spike, which direction? (up=1 vs down=0)
    """

    def __init__(self):
        self.spike_model = None
        self.direction_model = None
        self.best_rounds_s1 = 500
        self.best_rounds_s2 = 300
        self.calibrator = BoltzmannCalibrator()
        self.ising = IsingOverlay()

    def train(self, X_train, y_train, X_val, y_val,
              ret_train=None, ret_val=None,
              thresh_train=None, thresh_val=None):
        """
        Train both stages with early stopping.
        y values: 0=spike_down, 1=flat, 2=spike_up
        ret_train/ret_val: raw intraday returns for near-spike filtering
        """
        # ── Stage 1: Spike vs Flat (without realized_vol_20d) ────────────
        print("\n  ▶ Stage 1: Spike Detector (binary, excl. realized_vol_20d)")
        y_s1_train = (y_train != 1).astype(int)  # 1=spike, 0=flat
        y_s1_val = (y_val != 1).astype(int)

        X_s1_train = X_train[S1_FEATURE_COLUMNS]
        X_s1_val = X_val[S1_FEATURE_COLUMNS]

        params = XGB_PARAMS_S1.copy()
        es = params.pop("early_stopping_rounds", 50)
        n_flat = (y_s1_train == 0).sum()
        n_spike = (y_s1_train == 1).sum()
        params["scale_pos_weight"] = (n_flat / max(n_spike, 1)) * S1_POS_WEIGHT_MULTIPLIER

        self.spike_model = xgb.XGBClassifier(**params, early_stopping_rounds=es)
        self.spike_model.fit(
            X_s1_train, y_s1_train,
            eval_set=[(X_s1_val, y_s1_val)], verbose=False,
        )
        self.best_rounds_s1 = self.spike_model.best_iteration + 1
        print(f"    Best iteration: {self.best_rounds_s1}")
        print(f"    scale_pos_weight: {n_flat / max(n_spike, 1):.2f}")

        y_s1_pred = self.spike_model.predict(X_s1_val)
        prec = precision_score(y_s1_val, y_s1_pred, zero_division=0)
        rec = recall_score(y_s1_val, y_s1_pred, zero_division=0)
        f1 = f1_score(y_s1_val, y_s1_pred, zero_division=0)
        print(f"    Spike detection — Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

        # ── Stage 2: Direction (spike + near-spike samples) ─────────────
        print("\n  ▶ Stage 2: Direction Classifier (spike + near-spike samples)")

        if ret_train is not None:
            train_mask = _near_spike_mask(y_train, ret_train, thresh_train)
            val_mask = _near_spike_mask(y_val, ret_val, thresh_val) if ret_val is not None else (y_val != 1)
        else:
            # Fallback: spike-only if no return data
            train_mask = y_train != 1
            val_mask = y_val != 1

        X_s2_train = X_train[train_mask]
        # Direction label: 1 = positive return, 0 = negative return
        y_s2_train = (ret_train[train_mask] > 0).astype(int) if ret_train is not None else (y_train[train_mask] == 2).astype(int)

        X_s2_val = X_val[val_mask]
        y_s2_val = (ret_val[val_mask] > 0).astype(int) if ret_val is not None else (y_val[val_mask] == 2).astype(int)

        print(f"    Training samples: {len(X_s2_train)} (spike + near-spike)")

        if len(X_s2_train) < 20 or len(X_s2_val) < 5:
            print("    ⚠ Not enough samples for Stage 2. Skipping.")
            return self.best_rounds_s1, self.best_rounds_s2

        params2 = XGB_PARAMS_S2.copy()
        es2 = params2.pop("early_stopping_rounds", 30)

        self.direction_model = xgb.XGBClassifier(**params2, early_stopping_rounds=es2)
        self.direction_model.fit(
            X_s2_train, y_s2_train,
            eval_set=[(X_s2_val, y_s2_val)], verbose=False,
        )
        self.best_rounds_s2 = self.direction_model.best_iteration + 1
        print(f"    Best iteration: {self.best_rounds_s2}")

        # Evaluate direction accuracy on spike samples only (the ones that matter)
        spike_val = y_val != 1
        if spike_val.sum() >= 5:
            y_s2_pred_spike = self.direction_model.predict(X_val[spike_val])
            y_s2_true_spike = (y_val[spike_val] == 2).astype(int)
            acc = (y_s2_pred_spike == y_s2_true_spike.values).mean()
            print(f"    Direction accuracy (spike subset): {acc:.3f} ({spike_val.sum()} samples)")
        else:
            print(f"    Direction accuracy: N/A (only {spike_val.sum()} spike samples in val)")

        return self.best_rounds_s1, self.best_rounds_s2

    def fit_stat_mech_layers(self, X_val, y_val, sign_returns: pd.DataFrame,
                             tickers_val: pd.Series | None = None):
        """Fit Boltzmann calibrator and Ising overlay on validation split."""
        print("\n  ▶ Stat-mech layers: Boltzmann calibrator + Ising overlay")
        raw_val = self.predict_raw(X_val)
        vix = X_val["vix"] if "vix" in X_val.columns else None
        threshold = get_trade_threshold()
        self.calibrator.fit(raw_val, X_val, y_val, vix=vix, trade_threshold=threshold)
        calibrated = self.calibrator.transform(raw_val, X_val, vix=vix)
        nll = self.calibrator.nll(raw_val, X_val, y_val, vix=vix)
        print(f"    Calibrator val NLL: {nll:.4f} (shrink={self.calibrator.shrink:.2f})")

        self.ising.fit_couplings(sign_returns)
        if "local_field" in X_val.columns:
            dates = X_val.index if tickers_val is not None else None
            self.ising.fit_lambda(
                calibrated, X_val["local_field"], y_val,
                tickers=tickers_val, dates=dates,
                trade_threshold=threshold,
            )
            enabled = "on" if self.ising.enabled else "off"
            print(f"    Ising blend λ: {self.ising.lambda_blend:.3f} ({enabled})")

    def retrain_full(self, X, y, intraday_ret=None, adaptive_threshold=None):
        """Retrain on full dataset using discovered optimal rounds."""
        print(f"\n  Retraining on full dataset ({len(X)} rows)...")

        # Stage 1 (without realized_vol_20d)
        y_s1 = (y != 1).astype(int)
        n_flat = (y_s1 == 0).sum()
        n_spike = (y_s1 == 1).sum()
        params = XGB_PARAMS_S1.copy()
        params.pop("early_stopping_rounds", None)
        params["n_estimators"] = self.best_rounds_s1
        params["scale_pos_weight"] = (n_flat / max(n_spike, 1)) * S1_POS_WEIGHT_MULTIPLIER
        self.spike_model = xgb.XGBClassifier(**params)
        self.spike_model.fit(X[S1_FEATURE_COLUMNS], y_s1, verbose=False)

        # Stage 2: spike + near-spike samples
        if intraday_ret is not None:
            mask = _near_spike_mask(y, intraday_ret, adaptive_threshold)
            X_s2 = X[mask]
            y_s2 = (intraday_ret[mask] > 0).astype(int)
        else:
            mask = y != 1
            X_s2 = X[mask]
            y_s2 = (y[mask] == 2).astype(int)

        params2 = XGB_PARAMS_S2.copy()
        params2.pop("early_stopping_rounds", None)
        params2["n_estimators"] = self.best_rounds_s2
        self.direction_model = xgb.XGBClassifier(**params2)
        self.direction_model.fit(X_s2, y_s2, verbose=False)

        print("  Full retrain complete.")

    def spike_val_metrics(self, X_val, y_val, threshold=0.5, calibrated=False):
        """Spike detection precision/recall/F1 on a validation set (raw XGBoost by default)."""
        if calibrated and self.calibrator.fitted:
            probs = self.predict(X_val)
            p_spike = probs["p_spike"].values
        else:
            X_s1 = _align_to_model_features(X_val, self.spike_model)
            p_spike = self.spike_model.predict_proba(X_s1)[:, 1]
        y_true = (y_val != 1).astype(int)
        y_pred = (p_spike >= threshold).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return {"precision": prec, "recall": rec, "f1": f1}

    def calibrated_val_nll(self, X_val, y_val) -> float:
        """Mean negative log-likelihood of 3-class labels on validation set."""
        if not self.calibrator.fitted:
            return float("inf")
        raw = self.predict_raw(X_val)
        vix = X_val["vix"] if "vix" in X_val.columns else None
        return self.calibrator.nll(raw, X_val, y_val, vix=vix)

    def predict_raw(self, X):
        """Raw factorized XGBoost probabilities (no stat-mech layers)."""
        if isinstance(X, pd.DataFrame):
            X_s1 = _align_to_model_features(X, self.spike_model)
            X_s2 = _align_to_model_features(X, self.direction_model) if self.direction_model else X
        else:
            X_s1, X_s2 = X, X
        p_spike = self.spike_model.predict_proba(X_s1)[:, 1]

        if self.direction_model is not None:
            p_up_given_spike = self.direction_model.predict_proba(X_s2)[:, 1]
        else:
            p_up_given_spike = np.full(len(X), 0.5)

        return pd.DataFrame({
            "p_spike": p_spike,
            "p_up": p_spike * p_up_given_spike,
            "p_down": p_spike * (1 - p_up_given_spike),
            "p_flat": 1 - p_spike,
        }, index=X.index if hasattr(X, "index") else None)

    def predict(self, X):
        """
        Returns DataFrame with: p_spike, p_up, p_down, p_flat, p_spike_raw
        Applies Boltzmann calibrator and Ising overlay when fitted.
        """
        raw = self.predict_raw(X)
        if self.calibrator.fitted:
            vix = X["vix"] if isinstance(X, pd.DataFrame) and "vix" in X.columns else None
            out = self.calibrator.transform(raw, X, vix=vix)
        else:
            out = raw.copy()
            out["p_spike_raw"] = raw["p_spike"]

        if self.ising.fitted and isinstance(X, pd.DataFrame) and "local_field" in X.columns:
            tickers = pd.Series(X.index, index=X.index) if set(X.index).issubset(set(UNIVERSE)) else None
            out = self.ising.transform(out, X["local_field"], tickers=tickers)
        elif "p_spike_raw" not in out.columns:
            out["p_spike_raw"] = raw["p_spike"]

        return out

    def predict_for_trade(self, X):
        """
        Production inference path: calibrated + optional Ising with degeneracy bypass.
        Returns p_spike_trade for execution; keeps p_spike / p_spike_raw for logging.
        """
        raw = self.predict_raw(X)
        out = raw.copy()
        out["p_spike_raw"] = raw["p_spike"]
        out["calibrator_bypassed"] = False

        if self.calibrator.fitted:
            vix = X["vix"] if isinstance(X, pd.DataFrame) and "vix" in X.columns else None
            cal = self.calibrator.transform(raw, X, vix=vix)
            bypass = detect_calibrator_degeneracy(cal["p_spike"])
            if bypass:
                out = raw.copy()
                out["p_spike_raw"] = raw["p_spike"]
            else:
                out = cal

            out["calibrator_bypassed"] = bypass
            if (
                not bypass
                and self.ising.enabled
                and self.ising.fitted
                and isinstance(X, pd.DataFrame)
                and "local_field" in X.columns
            ):
                tickers = pd.Series(X.index, index=X.index) if set(X.index).issubset(set(UNIVERSE)) else None
                out = self.ising.transform(out, X["local_field"], tickers=tickers)
        else:
            out["calibrator_bypassed"] = False

        out["p_spike_trade"] = out["p_spike"]
        return out

    def trade_val_metrics(self, X_val, y_val, threshold=None):
        """Precision/recall on p_spike_trade at trade threshold."""
        if threshold is None:
            threshold = get_trade_threshold()
        probs = self.predict_for_trade(X_val)
        y_true = (y_val != 1).astype(int)
        y_pred = (probs["p_spike_trade"].values >= threshold).astype(int)
        return {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "signals": int(y_pred.sum()),
        }

    def save(self, path=MODEL_PATH):
        base = str(path).replace(".json", "")
        self.spike_model.save_model(f"{base}_s1.json")
        if self.direction_model:
            self.direction_model.save_model(f"{base}_s2.json")
        meta = {"best_rounds_s1": self.best_rounds_s1, "best_rounds_s2": self.best_rounds_s2}
        with open(f"{base}_meta.json", "w") as f:
            json.dump(meta, f)
        self.calibrator.save(Path(f"{base}_calibrator.json"))
        self.ising.save(Path(f"{base}_ising.json"))
        print(f"  Model saved to {base}_s1.json + {base}_s2.json")

    def load(self, path=MODEL_PATH):
        base = str(path).replace(".json", "")
        self.spike_model = xgb.XGBClassifier()
        self.spike_model.load_model(f"{base}_s1.json")
        s2_path = f"{base}_s2.json"
        if Path(s2_path).exists():
            self.direction_model = xgb.XGBClassifier()
            self.direction_model.load_model(s2_path)
        meta_path = f"{base}_meta.json"
        if Path(meta_path).exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.best_rounds_s1 = meta.get("best_rounds_s1", 500)
            self.best_rounds_s2 = meta.get("best_rounds_s2", 300)
        self.calibrator.load(Path(f"{base}_calibrator.json"))
        self.ising.load(Path(f"{base}_ising.json"))
        print(f"  Model loaded from {base}_s1.json")

    def get_spike_feature_importance(self):
        imp = self.spike_model.feature_importances_
        names = self.spike_model.get_booster().feature_names or S1_FEATURE_COLUMNS
        return pd.Series(imp, index=names).sort_values(ascending=False)
