from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    baseline_test_summary,
    benign_class_index,
    compute_metrics,
    confusion_matrix_frame,
    per_class_recall_frame,
    transfer_summary,
    worst_case_attack_summary,
    write_summary,
)


def test_benign_class_index_matches_each_target() -> None:
    assert benign_class_index("multiclass", ("BenignTraffic", "DDoS-ICMP_Flood")) == 0
    assert benign_class_index("family", ("Benign", "DDoS")) == 0
    assert benign_class_index("binary", ("Benign", "Attack")) == 0
    assert benign_class_index("family", ("DDoS", "Benign")) == 1


def test_benign_class_index_rejects_unknown_target() -> None:
    with pytest.raises(ValueError, match="Unknown target"):
        benign_class_index("unknown", ("Benign",))


def test_benign_class_index_requires_the_benign_class() -> None:
    with pytest.raises(ValueError, match="benign class"):
        benign_class_index("family", ("DDoS", "DoS"))


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


def test_baseline_summary_filters_validation_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "target": "binary",
                "model": "mlp",
                "split": "val",
                "accuracy": 0.5,
                "macro_f1": 0.5,
                "weighted_f1": 0.5,
                "macro_recall": 0.5,
            },
            {
                "target": "binary",
                "model": "mlp",
                "split": "test",
                "accuracy": 0.8,
                "macro_f1": 0.7,
                "weighted_f1": 0.78,
                "macro_recall": 0.72,
            },
        ]
    )

    summary = baseline_test_summary(frame)

    assert len(summary) == 1
    assert summary.iloc[0]["accuracy"] == 0.8


def test_worst_case_attack_summary_reports_extremes() -> None:
    frame = pd.DataFrame(
        [
            {
                "target": "binary",
                "model": "mlp",
                "attack": "fgsm",
                "epsilon": 0.01,
                "clean_accuracy": 0.9,
                "adversarial_accuracy": 0.8,
                "accuracy_drop": 0.1,
                "clean_macro_f1": 0.85,
                "adversarial_macro_f1": 0.75,
                "macro_f1_drop": 0.1,
                "clean_macro_recall": 0.8,
                "adversarial_macro_recall": 0.7,
                "macro_recall_drop": 0.1,
                "attack_success_rate": 0.2,
            },
            {
                "target": "binary",
                "model": "mlp",
                "attack": "fgsm",
                "epsilon": 0.05,
                "clean_accuracy": 0.9,
                "adversarial_accuracy": 0.4,
                "accuracy_drop": 0.5,
                "clean_macro_f1": 0.85,
                "adversarial_macro_f1": 0.3,
                "macro_f1_drop": 0.55,
                "clean_macro_recall": 0.8,
                "adversarial_macro_recall": 0.25,
                "macro_recall_drop": 0.55,
                "attack_success_rate": 0.7,
            },
        ]
    )

    summary = worst_case_attack_summary(frame)
    row = summary.iloc[0]

    assert row["worst_accuracy"] == 0.4
    assert row["epsilon_at_worst_accuracy"] == 0.05
    assert row["max_accuracy_drop"] == 0.5
    assert row["worst_macro_f1"] == 0.3
    assert row["max_attack_success_rate"] == 0.7


def _attack_row(
    *,
    attack: str,
    epsilon: float,
    clean_accuracy: float,
    adversarial_accuracy: float,
    model: str = "mlp",
    target: str = "binary",
    evasion_rate: float = 0.0,
) -> dict[str, float | str]:
    return {
        "target": target,
        "model": model,
        "attack": attack,
        "epsilon": epsilon,
        "clean_accuracy": clean_accuracy,
        "adversarial_accuracy": adversarial_accuracy,
        "accuracy_drop": clean_accuracy - adversarial_accuracy,
        "clean_macro_f1": clean_accuracy,
        "adversarial_macro_f1": adversarial_accuracy,
        "macro_f1_drop": clean_accuracy - adversarial_accuracy,
        "clean_macro_recall": clean_accuracy,
        "adversarial_macro_recall": adversarial_accuracy,
        "macro_recall_drop": clean_accuracy - adversarial_accuracy,
        "attack_success_rate": clean_accuracy - adversarial_accuracy,
        "evasion_rate": evasion_rate,
        "evasion_population": 100,
    }


def test_transfer_summary_subtracts_the_random_noise_control() -> None:
    frame = pd.DataFrame(
        [
            _attack_row(
                attack="pgd",
                epsilon=0.1,
                clean_accuracy=0.99,
                adversarial_accuracy=0.20,
            ),
            _attack_row(
                attack="random_noise",
                epsilon=0.1,
                clean_accuracy=0.99,
                adversarial_accuracy=0.55,
            ),
        ]
    )

    summary = transfer_summary(frame)
    row = summary.iloc[0]

    assert row["random_noise_accuracy"] == 0.55
    assert np.isclose(row["accuracy_drop_vs_clean"], 0.79)
    assert np.isclose(row["accuracy_drop_vs_noise"], 0.35)


def test_transfer_summary_is_empty_without_a_noise_control() -> None:
    frame = pd.DataFrame(
        [_attack_row(attack="pgd", epsilon=0.1, clean_accuracy=0.9, adversarial_accuracy=0.2)]
    )

    assert transfer_summary(frame).empty


def test_write_summary_writes_traceable_files(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    pd.DataFrame(
        [
            {
                "target": "binary",
                "model": "mlp",
                "split": "test",
                "accuracy": 0.9,
                "macro_f1": 0.8,
                "weighted_f1": 0.9,
                "macro_recall": 0.85,
            }
        ]
    ).to_csv(results / "baselines_metrics.csv", index=False)
    pd.DataFrame(
        [
            _attack_row(attack="pgd", epsilon=0.05, clean_accuracy=0.9, adversarial_accuracy=0.3),
            _attack_row(
                attack="random_noise",
                epsilon=0.05,
                clean_accuracy=0.9,
                adversarial_accuracy=0.6,
            ),
        ]
    ).to_csv(results / "attack_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "target": "binary",
                "model": "mlp",
                "attack": "pgd",
                "epsilon": 0.05,
                "clean_accuracy": 0.9,
                "adversarial_accuracy": 0.3,
                "accuracy_drop": 0.6,
                "clean_macro_f1": 0.8,
                "adversarial_macro_f1": 0.2,
                "macro_f1_drop": 0.6,
                "clean_macro_recall": 0.8,
                "adversarial_macro_recall": 0.2,
                "macro_recall_drop": 0.6,
                "attack_success_rate": 0.7,
            }
        ]
    ).to_csv(results / "attack_metrics.csv", index=False)

    paths = write_summary(results)

    assert {path.name for path in paths} == {
        "baseline_test_summary.csv",
        "attack_worst_case.csv",
        "experiment_summary.json",
    }
    assert all(path.exists() for path in paths)
