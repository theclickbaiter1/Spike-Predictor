"""
model.py — Two-stage XGBoost model for spike detection.

Stage 1: Binary spike detector (spike vs flat)
         — excludes realized_vol_20d (prevents "always flag volatile stocks" bias)
Stage 2: Direction classifier (up vs down)
         — trained on spike + near-spike samples (not all samples, not spike-only)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from config import (
    FEATURE_COLUMNS, LABEL_NAMES, MODEL_PATH,
    SPIKE_THRESHOLD, VAL_FRACTION, XGB_PARAMS_S1, XGB_PARAMS_S2,
)

# Stage 1 sees all features except realized_vol_20d.
# vol_z_score (relative volatility) remains — it captures regime shifts.
# realized_vol_20d just tells the model "this is a volatile stock" which
# inflates spike probability for every volatile ticker every day.
S1_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c != "realized_vol_20d"]


def time_series_split(X, y, val_frac=VAL_FRACTION):
    n = len(X)
    split_idx = int(n * (1 - val_frac))
    print(f"  Time-series split: Train={split_idx}, Val={n - split_idx} ({val_frac*100:.0f}%)")
    return X.iloc[:split_idx], y.iloc[:split_idx], X.iloc[split_idx:], y.iloc[split_idx:]


def _near_spike_mask(y, intraday_ret, threshold_frac=0.5):
    """
    Select spike samples + near-spike samples (return > 50% of spike threshold).
    This gives Stage 2 enough data to learn direction without flat-day bias.
    """
    is_spike = y != 1
    # Near-spike: not a spike but absolute return > threshold_frac × SPIKE_THRESHOLD
    # These are days with meaningful directional moves, just below the spike cutoff
    is_near_spike = (~is_spike) & (intraday_ret.abs() >= SPIKE_THRESHOLD * threshold_frac)
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

    def train(self, X_train, y_train, X_val, y_val,
              ret_train=None, ret_val=None):
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
        params["scale_pos_weight"] = n_flat / max(n_spike, 1)

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
            train_mask = _near_spike_mask(y_train, ret_train)
            val_mask = _near_spike_mask(y_val, ret_val) if ret_val is not None else (y_val != 1)
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

    def retrain_full(self, X, y, intraday_ret=None):
        """Retrain on full dataset using discovered optimal rounds."""
        print(f"\n  Retraining on full dataset ({len(X)} rows)...")

        # Stage 1 (without realized_vol_20d)
        y_s1 = (y != 1).astype(int)
        n_flat = (y_s1 == 0).sum()
        n_spike = (y_s1 == 1).sum()
        params = XGB_PARAMS_S1.copy()
        params.pop("early_stopping_rounds", None)
        params["n_estimators"] = self.best_rounds_s1
        params["scale_pos_weight"] = n_flat / max(n_spike, 1)
        self.spike_model = xgb.XGBClassifier(**params)
        self.spike_model.fit(X[S1_FEATURE_COLUMNS], y_s1, verbose=False)

        # Stage 2: spike + near-spike samples
        if intraday_ret is not None:
            mask = _near_spike_mask(y, intraday_ret)
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

    def predict(self, X):
        """
        Returns DataFrame with: p_spike, p_up, p_down, p_flat
        Stage 1 uses S1_FEATURE_COLUMNS (excl. realized_vol_20d).
        Stage 2 uses all FEATURE_COLUMNS.
        """
        X_s1 = X[S1_FEATURE_COLUMNS] if isinstance(X, pd.DataFrame) else X
        p_spike = self.spike_model.predict_proba(X_s1)[:, 1]

        if self.direction_model is not None:
            p_up_given_spike = self.direction_model.predict_proba(X)[:, 1]
        else:
            p_up_given_spike = np.full(len(X), 0.5)

        return pd.DataFrame({
            "p_spike": p_spike,
            "p_up": p_spike * p_up_given_spike,
            "p_down": p_spike * (1 - p_up_given_spike),
            "p_flat": 1 - p_spike,
        }, index=X.index if hasattr(X, "index") else None)

    def save(self, path=MODEL_PATH):
        base = str(path).replace(".json", "")
        self.spike_model.save_model(f"{base}_s1.json")
        if self.direction_model:
            self.direction_model.save_model(f"{base}_s2.json")
        meta = {"best_rounds_s1": self.best_rounds_s1, "best_rounds_s2": self.best_rounds_s2}
        with open(f"{base}_meta.json", "w") as f:
            json.dump(meta, f)
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
        print(f"  Model loaded from {base}_s1.json")

    def get_spike_feature_importance(self):
        imp = self.spike_model.feature_importances_
        return pd.Series(imp, index=S1_FEATURE_COLUMNS).sort_values(ascending=False)
