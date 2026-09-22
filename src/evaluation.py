"""
evaluation.py - Model Evaluation, Metrics and Visualization Module.

Reports comprehensive classification performance metrics and generates publication-quality
visualizations for confusion matrices, ROC curves, PR curves, and feature importances.
All plots are saved to the results/ directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger("space_debris.evaluation")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Consistent dark aerospace palette for all plots
PALETTE = {
    "bg": "#0A0F1E",
    "panel": "#111827",
    "accent": "#00D4FF",
    "highlight": "#FF4D4D",
    "muted": "#374151",
    "text": "#E5E7EB",
}


def _apply_dark_style(ax: plt.Axes, title: str = "") -> None:
    """Apply consistent aerospace dark theme to a matplotlib axes."""
    ax.set_facecolor(PALETTE["panel"])
    ax.figure.patch.set_facecolor(PALETTE["bg"])
    ax.title.set_color(PALETTE["text"])
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["muted"])
    if title:
        ax.set_title(title, color=PALETTE["text"], fontsize=13, fontweight="bold", pad=10)


class ModelEvaluator:
    """Evaluates trained classification pipelines and generates result artifacts."""

    def __init__(self, results_dir: Union[str, Path] = "results/") -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.all_metrics: Dict[str, Dict[str, Any]] = {}

    def evaluate(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute and store all relevant classification metrics."""
        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)

        metrics: Dict[str, Any] = {
            "model": model_name,
            "n_samples_test": int(len(y_true)),
            "n_positive": int(np.sum(y_true)),
            "n_negative": int(len(y_true) - np.sum(y_true)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "classification_report": classification_report(y_true, y_pred, zero_division=0),
        }

        if y_proba is not None and n_classes == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
                metrics["avg_precision"] = float(average_precision_score(y_true, y_proba))
            except Exception:
                metrics["roc_auc"] = None
                metrics["avg_precision"] = None

        logger.info(
            f"[{model_name}] Metrics:\n"
            f"  Accuracy={metrics['accuracy']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"F1={metrics['f1_score']:.4f}"
            + (f" | ROC-AUC={metrics.get('roc_auc', 'N/A'):.4f}" if metrics.get("roc_auc") else "")
        )

        self.all_metrics[model_name] = metrics
        return metrics

    def plot_confusion_matrix(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Path:
        """Generate and save a confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        labels = ["Low Priority (0)", "High Priority (1)"]

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            xticklabels=labels,
            yticklabels=labels,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Count"},
        )
        _apply_dark_style(ax, title=f"Confusion Matrix — {model_name}")
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        fig.tight_layout()

        safe_name = model_name.lower().replace(" ", "_")
        filepath = self.results_dir / f"confusion_matrix_{safe_name}.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
        plt.close(fig)
        logger.info(f"Confusion matrix saved: {filepath.resolve()}")
        return filepath

    def plot_roc_curves(
        self,
        model_results: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    ) -> Path:
        """Plot ROC curves for all models on a single axes."""
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = [PALETTE["accent"], "#FF6B35", "#8B5CF6", "#10B981"]

        has_valid_roc = False
        for i, (model_name, (y_true, _, y_proba)) in enumerate(model_results.items()):
            if y_proba is None or len(np.unique(y_true)) < 2:
                continue
            try:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc_val = auc(fpr, tpr)
                ax.plot(
                    fpr, tpr,
                    color=colors[i % len(colors)],
                    linewidth=2.5,
                    label=f"{model_name} (AUC = {roc_auc_val:.3f})"
                )
                has_valid_roc = True
            except Exception:
                pass

        if has_valid_roc:
            ax.plot([0, 1], [0, 1], "w--", linewidth=1.2, alpha=0.5, label="Random Classifier")
            ax.set_xlabel("False Positive Rate", fontsize=11)
            ax.set_ylabel("True Positive Rate", fontsize=11)
            _apply_dark_style(ax, title="ROC Curves — Conjunction Risk Classifier")
            legend = ax.legend(loc="lower right", facecolor=PALETTE["panel"], labelcolor=PALETTE["text"], fontsize=10)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
        else:
            ax.text(0.5, 0.5, "Insufficient class samples\nfor ROC calculation",
                    ha="center", va="center", color=PALETTE["text"], fontsize=12, transform=ax.transAxes)
            _apply_dark_style(ax, title="ROC Curves (Unavailable)")

        fig.tight_layout()
        filepath = self.results_dir / "roc_curves.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
        plt.close(fig)
        logger.info(f"ROC curves saved: {filepath.resolve()}")
        return filepath

    def plot_feature_importance(
        self,
        model_name: str,
        feature_importances: pd.Series,
        top_n: int = 10,
    ) -> Path:
        """Generate and save a horizontal bar chart of feature importances."""
        top = feature_importances.nlargest(top_n)

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh(
            top.index[::-1],
            top.values[::-1],
            color=PALETTE["accent"],
            alpha=0.85,
            edgecolor=PALETTE["muted"],
        )

        for bar, val in zip(bars, top.values[::-1]):
            ax.text(
                bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                color=PALETTE["text"],
                fontsize=9,
            )

        ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)", fontsize=11)
        _apply_dark_style(ax, title=f"Feature Importances — {model_name}")
        fig.tight_layout()

        safe_name = model_name.lower().replace(" ", "_")
        filepath = self.results_dir / f"feature_importance_{safe_name}.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
        plt.close(fig)
        logger.info(f"Feature importance plot saved: {filepath.resolve()}")
        return filepath

    def save_metrics_json(self) -> Path:
        """Serialize all computed metrics to a JSON file."""
        filepath = self.results_dir / "metrics.json"
        # Make serializable
        output = {}
        for model_name, metrics in self.all_metrics.items():
            output[model_name] = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float, str, bool, type(None)))
            }
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Metrics JSON saved: {filepath.resolve()}")
        return filepath

    def print_comparison_table(self) -> None:
        """Print a formatted model comparison table."""
        rows = []
        for model_name, m in self.all_metrics.items():
            rows.append({
                "Model": model_name,
                "Accuracy": f"{m.get('accuracy', 0):.4f}",
                "Precision": f"{m.get('precision', 0):.4f}",
                "Recall": f"{m.get('recall', 0):.4f}",
                "F1-Score": f"{m.get('f1_score', 0):.4f}",
                "ROC-AUC": f"{m.get('roc_auc', 0):.4f}" if m.get("roc_auc") else "N/A",
            })
        df_comparison = pd.DataFrame(rows)
        print("\n" + "=" * 70)
        print("  MODEL COMPARISON TABLE")
        print("=" * 70)
        print(df_comparison.to_string(index=False))
        print("=" * 70)
