"""Metric computation and result serialization."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from src.config import configure_logging, load_config

LOGGER = logging.getLogger(__name__)


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


def baseline_test_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the clean test metrics for every target/model pair."""
    required = {
        "target",
        "model",
        "split",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_recall",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Baseline metrics are missing columns: {missing}")
    test = frame[frame["split"] == "test"].copy()
    columns = [
        "target",
        "model",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_recall",
    ]
    return (
        test[columns]
        .sort_values(["target", "model"])
        .reset_index(drop=True)
    )


def worst_case_attack_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the strongest observed evasion effect for each target/model/attack."""
    required = {
        "target",
        "model",
        "attack",
        "epsilon",
        "clean_accuracy",
        "adversarial_accuracy",
        "accuracy_drop",
        "clean_macro_f1",
        "adversarial_macro_f1",
        "macro_f1_drop",
        "clean_macro_recall",
        "adversarial_macro_recall",
        "macro_recall_drop",
        "attack_success_rate",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Attack metrics are missing columns: {missing}")

    rows: list[dict[str, Any]] = []
    for (target, model, attack), group in frame.groupby(
        ["target", "model", "attack"]
    ):
        worst_accuracy_index = group["adversarial_accuracy"].idxmin()
        rows.append(
            {
                "target": target,
                "model": model,
                "attack": attack,
                "clean_accuracy": float(group["clean_accuracy"].iloc[0]),
                "worst_accuracy": float(
                    group["adversarial_accuracy"].min()
                ),
                "epsilon_at_worst_accuracy": float(
                    group.loc[worst_accuracy_index, "epsilon"]
                ),
                "max_accuracy_drop": float(group["accuracy_drop"].max()),
                "clean_macro_f1": float(group["clean_macro_f1"].iloc[0]),
                "worst_macro_f1": float(
                    group["adversarial_macro_f1"].min()
                ),
                "max_macro_f1_drop": float(group["macro_f1_drop"].max()),
                "clean_macro_recall": float(
                    group["clean_macro_recall"].iloc[0]
                ),
                "worst_macro_recall": float(
                    group["adversarial_macro_recall"].min()
                ),
                "max_macro_recall_drop": float(
                    group["macro_recall_drop"].max()
                ),
                "max_attack_success_rate": float(
                    group["attack_success_rate"].max()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["target", "model", "attack"]
    ).reset_index(drop=True)


def write_summary(results_dir: Path) -> list[Path]:
    """Aggregate saved metrics into compact, traceable summary files."""
    results_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    summary: dict[str, Any] = {}

    baseline_path = results_dir / "baselines_metrics.csv"
    if baseline_path.exists():
        baselines = baseline_test_summary(pd.read_csv(baseline_path))
        output_path = results_dir / "baseline_test_summary.csv"
        baselines.to_csv(output_path, index=False, float_format="%.6f")
        paths.append(output_path)
        summary["baseline_test"] = baselines.to_dict(orient="records")
    else:
        LOGGER.warning("Missing baseline metrics: %s", baseline_path)

    attack_path = results_dir / "attack_metrics.csv"
    if attack_path.exists():
        attacks = worst_case_attack_summary(pd.read_csv(attack_path))
        output_path = results_dir / "attack_worst_case.csv"
        attacks.to_csv(output_path, index=False, float_format="%.6f")
        paths.append(output_path)
        summary["attack_worst_case"] = attacks.to_dict(orient="records")
    else:
        LOGGER.warning("Missing attack metrics: %s", attack_path)

    if summary:
        summary_path = results_dir / "experiment_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        paths.append(summary_path)
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()
    config = load_config(args.config)
    results_dir = Path(config["paths"].get("results", "results"))
    paths = write_summary(results_dir)
    LOGGER.info("Wrote %d summary files to %s", len(paths), results_dir)


if __name__ == "__main__":
    main()
