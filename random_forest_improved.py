"""
random_forest_improved.py — Space Debris Collision Risk Predictor
Unified, robust Random Forest Classifier implementation.
Features:
- Leakage-free feature input (excludes minimum distance & relative velocity)
- Class imbalance mitigation (class_weight='balanced')
- Stratified K-Fold cross validation and compact randomized hyperparameter optimization
- Out-of-fold calibration & threshold tuning
- Feature importance extraction
"""

from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score, confusion_matrix
)

LEAKAGE_FREE_FEATURES = [
    "altitude_difference_km",
    "time_to_tca_min",
    "sat_altitude_km",
    "debris_altitude_km",
    "inclination_difference_deg",
    "eccentricity_difference",
    "period_difference_min",
    "sma_difference_km",
]


class ImprovedRandomForest:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model: RandomForestClassifier | None = None
        self.best_params: dict | None = None
        self.cv_results: dict | None = None
        self.optimal_threshold: float = 0.50

    def tune_and_fit(self, X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = 10) -> ImprovedRandomForest:
        """Perform compact RandomizedSearchCV over sensible hyperparameter space."""
        base_rf = RandomForestClassifier(
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=1
        )

        param_dist = {
            "n_estimators": [50, 100, 150, 200],
            "max_depth": [4, 6, 8, 10, None],
            "min_samples_split": [2, 4, 6],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5],
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="f1",
            cv=cv,
            random_state=self.random_state,
            n_jobs=1,
            refit=True
        )

        search.fit(X_train[LEAKAGE_FREE_FEATURES], y_train)
        self.model = search.best_estimator_
        self.best_params = search.best_params_

        # Determine threshold on out-of-fold training probabilities
        oof_probs = np.zeros(len(y_train))
        for train_idx, val_idx in cv.split(X_train, y_train):
            fold_rf = RandomForestClassifier(**self.best_params, class_weight="balanced", random_state=self.random_state, n_jobs=1)
            fold_rf.fit(X_train.iloc[train_idx][LEAKAGE_FREE_FEATURES], y_train.iloc[train_idx])
            oof_probs[val_idx] = fold_rf.predict_proba(X_train.iloc[val_idx][LEAKAGE_FREE_FEATURES])[:, 1]

        # Scan thresholds between 0.3 and 0.7 to maximize F1
        best_thr = 0.50
        best_f1 = -1.0
        for thr in np.linspace(0.30, 0.70, 41):
            preds = (oof_probs >= thr).astype(int)
            f1 = f1_score(y_train, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)

        self.optimal_threshold = best_thr
        return self

    def predict(self, X: pd.DataFrame, use_threshold: bool = True) -> np.ndarray:
        probs = self.predict_proba(X)
        thr = self.optimal_threshold if use_threshold else 0.50
        return (probs >= thr).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict_proba(X[LEAKAGE_FREE_FEATURES])[:, 1]

    def get_feature_importances(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return pd.Series(self.model.feature_importances_, index=LEAKAGE_FREE_FEATURES).sort_values(ascending=False)

    def save(self, filepath: str | Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)


def build_and_evaluate_rf(X_train, y_train, X_test, y_test):
    rf = ImprovedRandomForest(random_state=42)
    rf.tune_and_fit(X_train, y_train, n_iter=12)

    # Train eval
    y_train_pred = rf.predict(X_train)
    y_train_prob = rf.predict_proba(X_train)
    train_f1 = f1_score(y_train, y_train_pred)

    # Test eval
    y_test_pred = rf.predict(X_test)
    y_test_prob = rf.predict_proba(X_test)
    test_f1 = f1_score(y_test, y_test_pred)

    metrics = {
        "model_name": "Improved Random Forest",
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "f1": test_f1,
        "macro_f1": f1_score(y_test, y_test_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "roc_auc": roc_auc_score(y_test, y_test_prob),
        "average_precision": average_precision_score(y_test, y_test_prob),
        "train_f1": train_f1,
        "test_f1": test_f1,
        "optimal_threshold": rf.optimal_threshold,
        "best_params": rf.best_params,
        "confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist(),
    }
    return rf, metrics, (y_test.values, y_test_pred, y_test_prob)
