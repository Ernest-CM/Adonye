"""
Unified metrics computation for all six model configurations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)
from typing import Dict, Tuple


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    low: float = 0.3,
    high: float = 0.7,
) -> float:
    """
    Return the threshold that maximises Youden's J, clamped to [low, high].
    Clamping prevents small validation sets from producing extreme thresholds
    that overfit and fail to generalise.
    """
    fpr, tpr, thresholds = roc_curve(y_true.flatten().astype(int), y_prob.flatten())
    j = tpr - fpr
    best = float(thresholds[np.argmax(j)])
    return float(np.clip(best, low, high))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes accuracy, precision, recall, F1, ROC-AUC, specificity, balanced accuracy.
    """
    y_true = y_true.flatten().astype(int)
    y_prob = y_pred_prob.flatten()
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy":          round(float(accuracy_score(y_true, y_pred)),       4),
        "precision":         round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":            round(float(recall_score(y_true, y_pred, zero_division=0)),    4),
        "f1":                round(float(f1_score(y_true, y_pred, zero_division=0)),        4),
        "roc_auc":           round(float(roc_auc_score(y_true, y_prob)),                    4),
        "specificity":       round(specificity,                                             4),
        "balanced_accuracy": round(float((recall_score(y_true, y_pred, zero_division=0) + specificity) / 2), 4),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: str,
) -> None:
    cm = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["HC", "PD"], yticklabels=["HC", "PD"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    model_name: str,
    save_path: str,
) -> None:
    fpr, tpr, _ = roc_curve(y_true.astype(int), y_pred_prob.flatten())
    auc = roc_auc_score(y_true.astype(int), y_pred_prob.flatten())

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def build_metrics_table(
    results: Dict[str, Dict[str, float]],
    save_path: str,
) -> pd.DataFrame:
    """
    Input:  {model_name: {metric_name: value}}
    Output: DataFrame (rows=models, cols=metrics), also saved as CSV.
    """
    df = pd.DataFrame(results).T
    df.index.name = "model"
    df.to_csv(save_path)
    print(f"\nMetrics table saved to {save_path}")
    print(df.to_string())
    return df
