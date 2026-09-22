"""
models.py - Classical Machine Learning Models Module.

Implements Logistic Regression (baseline) and Random Forest Classifier (primary model)
for binary classification of candidate conjunction pairs into priority risk tiers.

Design Decisions:
-----------------
- Group-aware pair splitting: Prevents the same satellite-debris pair (across different
  timestamps) from appearing in both training and test sets, avoiding data leakage.
- Stratified sampling: Preserves class balance in train/test split.
- Pipeline-based: Uses sklearn Pipeline for clean preprocessing + classification.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("space_debris.models")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Ordered list of feature columns used for training
FEATURE_COLUMNS: List[str] = [
    "minimum_distance_km",
    "relative_velocity_km_s",
    "altitude_difference_km",
    "time_to_tca_min",
    "sat_altitude_km",
    "debris_altitude_km",
    "inclination_difference_deg",
    "eccentricity_difference",
    "period_difference_min",
    "sma_difference_km",
]

LABEL_COLUMN: str = "label"


class ConjunctionDataset:
    """Handles ML dataset preparation, including train/test splitting."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def prepare(
        self,
        df_labeled: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        test_size: float = 0.25,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare train/test split with group-aware separation.

        Group-awareness: each unique (sat_id, debris_id) pair is treated as a group.
        Ensures that observations from the same pair do not span train and test sets,
        preventing optimistic leakage when the same pair appears at multiple timesteps.
        """
        feat_cols = feature_columns or FEATURE_COLUMNS
        available_feats = [c for c in feat_cols if c in df_labeled.columns]
        missing_feats = [c for c in feat_cols if c not in df_labeled.columns]
        if missing_feats:
            logger.warning(f"Features not found in dataset (will be skipped): {missing_feats}")

        if LABEL_COLUMN not in df_labeled.columns:
            raise ValueError("'label' column missing from dataset.")

        # Drop rows where label is missing
        df_clean = df_labeled.dropna(subset=[LABEL_COLUMN]).copy()

        # Drop rows with any NaN in feature columns (avoids imputation issues)
        df_clean = df_clean.dropna(subset=available_feats)

        if len(df_clean) < 4:
            raise ValueError(
                f"Insufficient samples for training: only {len(df_clean)} clean rows after NaN removal. "
                "Run propagation over a longer window / wider proximity threshold."
            )

        X = df_clean[available_feats]
        y = df_clean[LABEL_COLUMN].astype(int)

        # Group-aware split if pair identifiers present
        if "sat_id" in df_clean.columns and "debris_id" in df_clean.columns:
            groups = df_clean["sat_id"].astype(str) + "_" + df_clean["debris_id"].astype(str)
            unique_groups = groups.nunique()
            min_test_groups = max(1, int(np.ceil(unique_groups * test_size)))
            if unique_groups >= 4:
                gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=self.random_state)
                train_idx, test_idx = next(gss.split(X, y, groups=groups))
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                logger.info(
                    f"Group-aware split: {unique_groups} unique pairs, "
                    f"train={len(X_train)}, test={len(X_test)}"
                )
                return X_train, X_test, y_train, y_test

        # Fallback: stratified standard split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=self.random_state
            )
            logger.info(f"Stratified split: train={len(X_train)}, test={len(X_test)}")
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
            logger.info(f"Random split (no stratify): train={len(X_train)}, test={len(X_test)}")

        return X_train, X_test, y_train, y_test


class ModelTrainer:
    """
    Trains and manages classical ML classification models for conjunction risk prioritization.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.trained_models: Dict[str, Pipeline] = {}
        self.feature_names: List[str] = []

    def build_logistic_regression(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = "balanced",
    ) -> Pipeline:
        """Build a Logistic Regression pipeline with StandardScaler preprocessing."""
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=C,
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=self.random_state,
                solver="lbfgs",
            ))
        ])

    def build_random_forest(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 8,
        class_weight: str = "balanced",
    ) -> Pipeline:
        """Build a Random Forest pipeline (StandardScaler not strictly needed but included for uniformity)."""
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight=class_weight,
                random_state=self.random_state,
                n_jobs=-1,
            ))
        ])

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str,
        pipeline: Pipeline,
    ) -> Pipeline:
        """Fit a pipeline and store it."""
        self.feature_names = list(X_train.columns)
        logger.info(f"Training {model_name} on {len(X_train)} samples with {len(X_train.columns)} features...")
        pipeline.fit(X_train, y_train)
        self.trained_models[model_name] = pipeline
        logger.info(f"{model_name} training complete.")
        return pipeline

    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        lr_params: Optional[Dict[str, Any]] = None,
        rf_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Pipeline]:
        """Train all models and return a dict of fitted pipelines."""
        lr_kwargs = lr_params or {}
        rf_kwargs = rf_params or {}

        pipelines = {
            "Logistic Regression": self.build_logistic_regression(**lr_kwargs),
            "Random Forest": self.build_random_forest(**rf_kwargs),
        }

        for name, pipeline in pipelines.items():
            self.train(X_train, y_train, name, pipeline)

        return self.trained_models

    def predict(
        self,
        model_name: str,
        X: pd.DataFrame,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (y_pred, y_proba) for a named model on new data."""
        if model_name not in self.trained_models:
            raise KeyError(f"Model '{model_name}' not found. Available: {list(self.trained_models.keys())}")
        pipeline = self.trained_models[model_name]
        y_pred = pipeline.predict(X)
        y_proba = pipeline.predict_proba(X)[:, 1] if hasattr(pipeline.named_steps["clf"], "predict_proba") else None
        return y_pred, y_proba

    def save_model(self, model_name: str, directory: Union[str, Path] = "models/") -> Path:
        """Save a trained pipeline to disk with pickle."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        filename = model_name.lower().replace(" ", "_") + ".pkl"
        filepath = path / filename
        with open(filepath, "wb") as f:
            pickle.dump(self.trained_models[model_name], f)
        logger.info(f"Model '{model_name}' saved to {filepath.resolve()}")
        return filepath

    def load_model(self, model_name: str, directory: Union[str, Path] = "models/") -> Pipeline:
        """Load a trained pipeline from disk."""
        filename = model_name.lower().replace(" ", "_") + ".pkl"
        filepath = Path(directory) / filename
        with open(filepath, "rb") as f:
            pipeline = pickle.load(f)
        self.trained_models[model_name] = pipeline
        logger.info(f"Model '{model_name}' loaded from {filepath.resolve()}")
        return pipeline

    def get_feature_importances(self, model_name: str) -> Optional[pd.Series]:
        """Extract feature importances for tree-based models."""
        if model_name not in self.trained_models:
            return None
        clf = self.trained_models[model_name].named_steps.get("clf")
        if hasattr(clf, "feature_importances_"):
            return pd.Series(clf.feature_importances_, index=self.feature_names).sort_values(ascending=False)
        return None
