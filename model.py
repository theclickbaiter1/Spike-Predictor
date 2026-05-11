"""
model.py — Two-stage XGBoost model for spike detection.

Stage 1: Binary spike detector (spike vs flat)
Stage 2: Direction classifier (up vs down, trained only on spike days)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from config import (
    FEATURE_COLUMNS, LABEL_NAMES, MODEL_PATH,
    VAL_FRACTION, XGB_PARAMS_S1, XGB_PARAMS_S2,
)


def time_series_split(X, y, val_frac=VAL_FRACTION):
    n = len(X)
    split_idx = int(n * (1 - val_frac))
    print(f"  Time-series split: Train={split_idx}, Val={n - split_idx} ({val_frac*100:.0f}%)")
    return X.iloc[:split_idx], y.iloc[:split_idx], X.iloc[split_idx:], y.iloc[split_idx:]


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

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train both stages with early stopping.
        y values: 0=spike_down, 1=flat, 2=spike_up
        """
        # ── Stage 1: Spike vs Flat ───────────────────────────────────────
        print("\n  ▶ Stage 1: Spike Detector (binary)")
        y_s1_train = (y_train != 1).astype(int)  # 1=spike, 0=flat
        y_s1_val = (y_val != 1).astype(int)

        params = XGB_PARAMS_S1.copy()
        es = params.pop("early_stopping_rounds", 50)
        # Use scale_pos_weight for class imbalance (more efficient than sample_weight)
        n_flat = (y_s1_train == 0).sum()
        n_spike = (y_s1_train == 1).sum()
        params["scale_pos_weight"] = n_flat / max(n_spike, 1)

        self.spike_model = xgb.XGBClassifier(**params, early_stopping_rounds=es)
        self.spike_model.fit(
            X_train, y_s1_train,
            eval_set=[(X_val, y_s1_val)], verbose=False,
        )
        self.best_rounds_s1 = self.spike_model.best_iteration + 1
        print(f"    Best iteration: {self.best_rounds_s1}")
        print(f"    scale_pos_weight: {n_flat / max(n_spike, 1):.2f}")

        y_s1_pred = self.spike_model.predict(X_val)
        prec = precision_score(y_s1_val, y_s1_pred, zero_division=0)
        rec = recall_score(y_s1_val, y_s1_pred, zero_division=0)
        f1 = f1_score(y_s1_val, y_s1_pred, zero_division=0)
        print(f"    Spike detection — Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

        # ── Stage 2: Direction (trained on ALL samples for more data) ────
        print("\n  ▶ Stage 2: Direction Classifier (trained on all samples)")
        # Use intraday return sign from all samples (up=1 vs down/flat=0)
        # This gives the model much more training data than spike-only training
        y_s2_train = (y_train == 2).astype(int)  # 1=up, 0=down or flat
        y_s2_val = (y_val == 2).astype(int)

        params2 = XGB_PARAMS_S2.copy()
        es2 = params2.pop("early_stopping_rounds", 30)

        self.direction_model = xgb.XGBClassifier(**params2, early_stopping_rounds=es2)
        self.direction_model.fit(
            X_train, y_s2_train,
            eval_set=[(X_val, y_s2_val)], verbose=False,
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

    def retrain_full(self, X, y):
        """Retrain on full dataset using discovered optimal rounds."""
        print(f"\n  Retraining on full dataset ({len(X)} rows)...")

        # Stage 1
        y_s1 = (y != 1).astype(int)
        n_flat = (y_s1 == 0).sum()
        n_spike = (y_s1 == 1).sum()
        params = XGB_PARAMS_S1.copy()
        params.pop("early_stopping_rounds", None)
        params["n_estimators"] = self.best_rounds_s1
        params["scale_pos_weight"] = n_flat / max(n_spike, 1)
        self.spike_model = xgb.XGBClassifier(**params)
        self.spike_model.fit(X, y_s1, verbose=False)

        # Stage 2: trained on all samples (direction signal from full dataset)
        y_s2 = (y == 2).astype(int)
        params2 = XGB_PARAMS_S2.copy()
        params2.pop("early_stopping_rounds", None)
        params2["n_estimators"] = self.best_rounds_s2
        self.direction_model = xgb.XGBClassifier(**params2)
        self.direction_model.fit(X, y_s2, verbose=False)

        print("  Full retrain complete.")

    def predict(self, X):
        """
        Returns DataFrame with: p_spike, p_up, p_down, p_flat
        """
        p_spike = self.spike_model.predict_proba(X)[:, 1]

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
        return pd.Series(imp, index=FEATURE_COLUMNS).sort_values(ascending=False)
