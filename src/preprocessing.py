"""
preprocessing.py - Centralized Data Validation and Preprocessing Utilities.
Provides helpers for feature normalization, NaN handling, and dataset health checks.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("space_debris.preprocessing")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def validate_feature_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate that a feature dataframe has required columns and sensible values.
    Returns (is_valid, list_of_issues).
    """
    issues = []
    if df.empty:
        issues.append("Dataframe is empty.")
        return False, issues

    required = required_columns or []
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")

    numeric_df = df.select_dtypes(include=[np.number])
    all_nan_cols = [c for c in numeric_df.columns if numeric_df[c].isna().all()]
    if all_nan_cols:
        issues.append(f"Fully-NaN numeric columns: {all_nan_cols}")

    if "minimum_distance_km" in df.columns:
        neg_dist = (df["minimum_distance_km"] < 0).sum()
        if neg_dist > 0:
            issues.append(f"{neg_dist} rows have negative minimum_distance_km.")

    return len(issues) == 0, issues


def report_dataset_health(df: pd.DataFrame, name: str = "Dataset") -> None:
    """Print a dataset health report to logger."""
    logger.info(f"\n--- {name} Health Report ---")
    logger.info(f"  Shape: {df.shape}")
    numeric_df = df.select_dtypes(include=[np.number])
    nan_counts = numeric_df.isna().sum()
    if nan_counts.sum() > 0:
        logger.info("  NaN counts per numeric column:")
        for col, cnt in nan_counts[nan_counts > 0].items():
            logger.info(f"    {col}: {cnt}")
    else:
        logger.info("  No NaN values in numeric columns. ✓")
    if "label" in df.columns:
        logger.info(f"  Label distribution: {df['label'].value_counts().to_dict()}")
