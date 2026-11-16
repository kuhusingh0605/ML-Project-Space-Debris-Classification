"""
logistic_regression_improved.py — Space Debris Collision Risk Predictor
Unified Logistic Regression Pipeline:
- Leakage-free features
- Integrated StandardScaler within Pipeline (fit only on training fold)
- Class imbalance handling (class_weight='balanced')
- Regularization parameter tuning over StratifiedKFold
"""

from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
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


class ImprovedLogisticRegression:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self.best_params: dict | None = None
        self.optimal_threshold: float = 0.50

    def tune_and_fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> ImprovedLogisticRegression:
        base_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", random_state=self.random_state, solver="lbfgs", max_iter=1000))
        ])

        param_grid = {
            "clf__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        search = GridSearchCV(
            estimator=base_pipe,
            param_grid=param_grid,
            scoring="f1",
            cv=cv,
            n_jobs=1,
            refit=True
        )

        search.fit(X_train[LEAKAGE_FREE_FEATURES], y_train)
        self.pipeline = search.best_estimator_
        self.best_params = search.best_params_

        # Threshold calibration on OOF cross validation
        oof_probs = np.zeros(len(y_train))
        for train_idx, val_idx in cv.split(X_train, y_train):
            c_val = self.best_params["clf__C"]
            fold_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=c_val, class_weight="balanced", random_state=self.random_state, solver="lbfgs", max_iter=1000))
            ])
            fold_pipe.fit(X_train.iloc[train_idx][LEAKAGE_FREE_FEATURES], y_train.iloc[train_idx])
            oof_probs[val_idx] = fold_pipe.predict_proba(X_train.iloc[val_idx][LEAKAGE_FREE_FEATURES])[:, 1]

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
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not fitted.")
        return self.pipeline.predict_proba(X[LEAKAGE_FREE_FEATURES])[:, 1]

    def save(self, filepath: str | Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)


def build_and_evaluate_lr(X_train, y_train, X_test, y_test):
    lr = ImprovedLogisticRegression(random_state=42)
    lr.tune_and_fit(X_train, y_train)

    y_train_pred = lr.predict(X_train)
    train_f1 = f1_score(y_train, y_train_pred)

    y_test_pred = lr.predict(X_test)
    y_test_prob = lr.predict_proba(X_test)
    test_f1 = f1_score(y_test, y_test_pred)

    metrics = {
        "model_name": "Improved Logistic Regression",
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
        "optimal_threshold": lr.optimal_threshold,
        "best_params": lr.best_params,
        "confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist(),
    }
    return lr, metrics, (y_test.values, y_test_pred, y_test_prob)
