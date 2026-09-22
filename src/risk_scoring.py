"""
risk_scoring.py - Risk Scoring and Prioritization Module.

Applies trained ML model probabilities to compute standardized risk scores and
categorize conjunction pairs into LOW / MEDIUM / HIGH risk tiers.

DISCLAIMER:
-----------
Risk categories are experimental model-priority labels, NOT operational collision
probability estimates. HIGH does not mean a collision will occur; it means the
ML model assigns this pair the highest screening priority.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("space_debris.risk_scoring")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


RISK_CATEGORIES = {
    "HIGH": "HIGH (Model-Priority)",
    "MEDIUM": "MEDIUM (Monitoring-Priority)",
    "LOW": "LOW (Low Priority)",
}


class RiskScorer:
    """
    Assigns risk scores and categorical risk tiers to candidate conjunction pairs
    using predictions from trained ML classification pipelines.
    """

    def __init__(
        self,
        low_threshold: float = 0.3,
        high_threshold: float = 0.7,
    ) -> None:
        """
        Configurable risk thresholds:
          score < low_threshold  → LOW
          low_threshold <= score < high_threshold  → MEDIUM
          score >= high_threshold → HIGH
        """
        self.low_threshold = float(low_threshold)
        self.high_threshold = float(high_threshold)

    def assign_risk_tier(self, score: float) -> str:
        """Map a probability score (0–1) to a risk category label."""
        if score >= self.high_threshold:
            return "HIGH"
        elif score >= self.low_threshold:
            return "MEDIUM"
        else:
            return "LOW"

    def score_conjunctions(
        self,
        df_features: pd.DataFrame,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        model_name: str = "Random Forest",
    ) -> pd.DataFrame:
        """
        Add risk scores and tier labels to the conjunction feature dataframe.
        Sorts results by descending risk score for easy triage.
        """
        df = df_features.copy()

        if y_proba is not None:
            df["risk_score"] = y_proba
        else:
            # If no probability available (e.g., pure SVM), use binary prediction
            df["risk_score"] = y_pred.astype(float)

        df["predicted_label"] = y_pred.astype(int)
        df["risk_category"] = df["risk_score"].apply(self.assign_risk_tier)
        df["risk_model"] = model_name

        # Sort by risk score descending — highest priority pairs at the top
        df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        # Summary log
        tier_counts = df["risk_category"].value_counts()
        logger.info(
            f"Risk scoring complete for {len(df)} pairs (model: {model_name}):\n"
            + "\n".join([f"  {tier}: {count}" for tier, count in tier_counts.items()])
        )

        return df

    def top_risky_pairs(self, df_scored: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """Return the top-N highest-risk conjunction pairs."""
        return df_scored[df_scored["risk_category"].isin(["HIGH", "MEDIUM"])].head(n).copy()

    def save_scored_results(
        self,
        df_scored: pd.DataFrame,
        filepath: Union[str, Path] = "results/risk_scored_conjunctions.csv",
    ) -> Path:
        """Save the full risk-scored conjunction table to CSV."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Format columns for output
        display_cols = [
            "rank", "sat_name", "debris_name",
            "minimum_distance_km", "relative_velocity_km_s",
            "altitude_difference_km", "time_to_tca_min",
            "risk_score", "risk_category", "predicted_label",
            "tca_timestamp", "risk_model",
        ]
        out_cols = [c for c in display_cols if c in df_scored.columns]
        df_scored[out_cols].to_csv(path, index=False)
        logger.info(f"Risk-scored results saved to {path.resolve()} ({len(df_scored)} pairs)")
        return path
