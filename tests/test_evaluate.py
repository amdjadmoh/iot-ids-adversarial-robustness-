from __future__ import annotations

import numpy as np

from src.evaluate import (
    compute_metrics,
    confusion_matrix_frame,
    per_class_recall_frame,
)


def test_perfect_predictions_score_one() -> None:
    y_true = [0, 1, 2, 1]

    metrics = compute_metrics(y_true, y_true, labels=range(3))

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["macro_recall"] == 1.0


def test_per_class_recall_reports_support_and_recall() -> None:
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 1])

    frame = per_class_recall_frame(y_true, y_pred, ["a", "b", "c"])

    assert list(frame["class_name"]) == ["a", "b", "c"]
    assert list(frame["support"]) == [2, 1, 1]
    assert np.allclose(frame["recall"], [0.5, 1.0, 0.0])


def test_confusion_matrix_is_labelled() -> None:
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 1])

    matrix = confusion_matrix_frame(y_true, y_pred, ["a", "b", "c"])

    assert matrix.shape == (3, 3)
    assert matrix.loc["a", "a"] == 1
    assert matrix.loc["a", "b"] == 1
    assert matrix.loc["c", "b"] == 1
