"""Imbalance-aware evaluation metrics for SCOTUS classification."""
from __future__ import annotations

from sklearn.metrics import classification_report, confusion_matrix, f1_score


def full_report(y_true, y_pred, label_names: list[str]) -> dict:
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "per_class_report": classification_report(
            y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
