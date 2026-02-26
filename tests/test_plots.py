from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.plots import generate_all


def _write_resources(results: Path) -> None:
    targets = ["multiclass", "family", "binary"]
    models = ["random_forest", "xgboost", "mlp"]
    baseline_rows = []
    recall_rows = []
    attack_rows = []
    for target in targets:
        for model in models:
            for split in ("val", "test"):
                baseline_rows.append(
                    {
                        "target": target,
                        "model": model,
                        "split": split,
                        "n_samples": 30,
                        "accuracy": 0.8,
                        "macro_f1": 0.7,
                        "weighted_f1": 0.78,
                        "macro_recall": 0.72,
                    }
                )
            for class_id, class_name in enumerate(("a", "b", "c")):
                recall_rows.append(
                    {
                        "target": target,
                        "model": model,
                        "split": "test",
                        "class_id": class_id,
                        "class_name": class_name,
                        "support": 10,
                        "recall": 0.5 + 0.1 * class_id,
                    }
                )
            for attack in ("fgsm", "pgd"):
                for epsilon in (0.01, 0.02):
                    attack_rows.append(
                        {
                            "target": target,
                            "model": model,
                            "attack": attack,
                            "epsilon": epsilon,
                            "n_samples": 30,
                            "clean_accuracy": 0.8,
                            "adversarial_accuracy": 0.8 - epsilon,
                            "accuracy_drop": epsilon,
                            "clean_macro_f1": 0.7,
                            "adversarial_macro_f1": 0.7 - epsilon,
                            "macro_f1_drop": epsilon,
                            "clean_macro_recall": 0.72,
                            "adversarial_macro_recall": 0.72 - epsilon,
                            "macro_recall_drop": epsilon,
                            "attack_success_rate": 0.2,
                        }
                    )
    pd.DataFrame(baseline_rows).to_csv(
        results / "baselines_metrics.csv",
        index=False,
    )
    pd.DataFrame(recall_rows).to_csv(
        results / "baselines_per_class_recall.csv",
        index=False,
    )
    pd.DataFrame(attack_rows).to_csv(results / "attack_metrics.csv", index=False)

    count_rows = []
    for target in targets:
        for class_name, test_count in (("a", 4), ("b", 120), ("c", 90)):
            count_rows.append(
                {
                    "target": target,
                    "class_name": class_name,
                    "train": 300,
                    "val": 40,
                    "test": test_count,
                    "low_test_support": test_count < 30,
                }
            )
    pd.DataFrame(count_rows).to_csv(
        results / "test_class_counts.csv",
        index=False,
    )

    defense_rows = []
    for target in targets:
        for defense, parameter, clean_accuracy, adversarial_accuracy in (
            ("adversarial_training", 0.01, 0.72, 0.55),
            ("feature_squeezing", 8.0, 0.79, 0.60),
            ("feature_squeezing", 4.0, 0.75, 0.65),
        ):
            defense_rows.append(
                {
                    "target": target,
                    "defense": defense,
                    "defense_param": parameter,
                    "model": "mlp",
                    "attack": "clean",
                    "epsilon": 0.0,
                    "clean_accuracy": clean_accuracy,
                    "adversarial_accuracy": clean_accuracy,
                }
            )
            defense_rows.append(
                {
                    "target": target,
                    "defense": defense,
                    "defense_param": parameter,
                    "model": "mlp",
                    "attack": "pgd",
                    "epsilon": 0.05,
                    "clean_accuracy": clean_accuracy,
                    "adversarial_accuracy": adversarial_accuracy,
                }
            )
    pd.DataFrame(defense_rows).to_csv(
        results / "defense_metrics.csv",
        index=False,
    )

    confusion_dir = results / "confusion_matrices"
    confusion_dir.mkdir()
    matrix = pd.DataFrame(
        [[5, 1, 0], [0, 4, 2], [1, 0, 3]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )
    matrix.to_csv(confusion_dir / "multiclass_mlp_test.csv")
    matrix.to_csv(confusion_dir / "family_xgboost_test.csv")


def test_generate_all_writes_every_supported_figure(tmp_path: Path) -> None:
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    results.mkdir()
    _write_resources(results)

    paths = generate_all(results, figures)

    assert len(paths) == 2 + 3 + 2 + 6 + 3
    assert all(path.exists() for path in paths)
    assert all(path.stat().st_size > 0 for path in paths)


def test_low_support_classes_are_marked_in_per_class_figures(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    results.mkdir()
    _write_resources(results)

    generate_all(results, figures, min_test_samples=30)

    # The figure is regenerated with a footnote when any class is flagged; the
    # helper is the single source of truth for which classes those are.
    from src.plots import _low_support_classes

    for target in ("multiclass", "family", "binary"):
        assert _low_support_classes(results, target, 30) == {"a"}
        assert _low_support_classes(results, target, 5) == {"a"}
        assert _low_support_classes(results, target, 2) == set()


def test_generate_all_skips_missing_inputs(tmp_path: Path) -> None:
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    results.mkdir()

    paths = generate_all(results, figures)

    assert paths == []
