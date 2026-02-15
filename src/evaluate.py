"""Metric computation and result serialization."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    labels: Sequence[int],
) -> dict[str, float]:
    """Compute headline classification metrics over a fixed label set."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(labels),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(labels),
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=list(labels),
                average="macro",
                zero_division=0,
            )
        ),
    }


def per_class_recall_frame(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Return per-class support and recall for every configured class."""
    labels = list(range(len(class_names)))
    recalls = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    support = np.bincount(np.asarray(y_true, dtype=np.int64), minlength=len(labels))
    return pd.DataFrame(
        {
            "class_id": labels,
            "class_name": list(class_names),
            "support": support.astype(np.int64),
            "recall": np.asarray(recalls, dtype=float),
        }
    )


def confusion_matrix_frame(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Return a labelled confusion matrix."""
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=list(class_names), columns=list(class_names))
