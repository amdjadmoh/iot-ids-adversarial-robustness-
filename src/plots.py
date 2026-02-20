"""Figure generation from files under results/."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from src.config import configure_logging, load_config

LOGGER = logging.getLogger(__name__)

TARGET_ORDER = ("multiclass", "family", "binary")
MODEL_ORDER = ("random_forest", "xgboost", "mlp")
ATTACK_ORDER = ("fgsm", "pgd")
DEFENSE_ORDER = ("adversarial_training", "feature_squeezing")


def _ordered_present(values: Iterable[str], order: tuple[str, ...]) -> list[str]:
    present = set(values)
    return [value for value in order if value in present]


def _save(figure: plt.Figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def plot_baseline_metrics(results_dir: Path, figures_dir: Path) -> list[Path]:
    path = results_dir / "baselines_metrics.csv"
    if not path.exists():
        LOGGER.warning("Skipping baseline metrics; %s is missing", path)
        return []
    frame = pd.read_csv(path)
    test = frame[frame["split"] == "test"]
    paths: list[Path] = []
    for metric in ("accuracy", "macro_f1"):
        table = test.pivot_table(
            index="target",
            columns="model",
            values=metric,
            aggfunc="mean",
        )
        rows = _ordered_present(table.index, TARGET_ORDER)
        columns = _ordered_present(table.columns, MODEL_ORDER)
        if not rows or not columns:
            continue
        table = table.reindex(index=rows, columns=columns)
        axis = table.plot(kind="bar", figsize=(8, 4), rot=0)
        figure = axis.get_figure()
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.set_xlabel("Target")
        axis.set_title(f"Baseline test {metric.replace('_', ' ')}")
        axis.set_ylim(0.0, 1.0)
        axis.legend(title="Model", fontsize=8)
        paths.append(
            _save(figure, figures_dir / f"baselines_{metric}.png")
        )
    return paths


def plot_per_class_recall(results_dir: Path, figures_dir: Path) -> list[Path]:
    path = results_dir / "baselines_per_class_recall.csv"
    if not path.exists():
        LOGGER.warning("Skipping per-class recall; %s is missing", path)
        return []
    frame = pd.read_csv(path)
    test = frame[frame["split"] == "test"]
    paths: list[Path] = []
    for target in _ordered_present(test["target"], TARGET_ORDER):
        subset = test[test["target"] == target]
        models = _ordered_present(subset["model"], MODEL_ORDER)
        if not models:
            continue
        class_names = (
            subset[subset["model"] == models[0]]["class_name"].tolist()
        )
        width = max(10.0, 0.45 * len(class_names) + 3.0)
        figure, axis = plt.subplots(figsize=(width, 5))
        positions = np.arange(len(class_names))
        bar_width = 0.8 / len(models)
        for index, model in enumerate(models):
            values = (
                subset[subset["model"] == model]
                .set_index("class_name")
                .reindex(class_names)["recall"]
            )
            offset = (index - (len(models) - 1) / 2) * bar_width
            axis.bar(
                positions + offset,
                values.to_numpy(),
                width=bar_width,
                label=model,
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(class_names, rotation=90, fontsize=7)
        axis.set_ylabel("Recall")
        axis.set_xlabel("Class")
        axis.set_ylim(0.0, 1.05)
        axis.set_title(f"Per-class test recall ({target})")
        axis.legend(title="Model", fontsize=8)
        paths.append(
            _save(figure, figures_dir / f"per_class_recall_{target}.png")
        )
    return paths


def plot_confusion_matrices(results_dir: Path, figures_dir: Path) -> list[Path]:
    directory = results_dir / "confusion_matrices"
    if not directory.exists():
        LOGGER.warning("Skipping confusion matrices; %s is missing", directory)
        return []
    paths: list[Path] = []
    for path in sorted(directory.glob("*_test.csv")):
        matrix = pd.read_csv(path, index_col=0)
        if matrix.empty:
            continue
        values = matrix.to_numpy()
        size = max(6.0, 0.35 * len(matrix) + 2.5)
        figure, axis = plt.subplots(figsize=(size, size))
        masked = np.ma.masked_where(values == 0, values)
        image = axis.imshow(
            masked,
            cmap="Blues",
            norm=LogNorm(vmin=1, vmax=max(1, values.max())),
        )
        axis.set_xticks(range(len(matrix.columns)))
        axis.set_yticks(range(len(matrix.index)))
        label_size = 6 if len(matrix) > 12 else 8
        axis.set_xticklabels(
            matrix.columns,
            rotation=90,
            fontsize=label_size,
        )
        axis.set_yticklabels(matrix.index, fontsize=label_size)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(path.stem)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        paths.append(_save(figure, figures_dir / f"confusion_{path.stem}.png"))
    return paths


def plot_attack_curves(results_dir: Path, figures_dir: Path) -> list[Path]:
    path = results_dir / "attack_metrics.csv"
    if not path.exists():
        LOGGER.warning("Skipping attack curves; %s is missing", path)
        return []
    frame = pd.read_csv(path)
    paths: list[Path] = []
    for target in _ordered_present(frame["target"], TARGET_ORDER):
        subset = frame[frame["target"] == target]
        for metric, filename in (
            ("adversarial_accuracy", "attack_accuracy"),
            ("adversarial_macro_f1", "attack_macro_f1"),
        ):
            figure, axis = plt.subplots(figsize=(8, 5))
            for model in _ordered_present(subset["model"], MODEL_ORDER):
                for attack in _ordered_present(subset["attack"], ATTACK_ORDER):
                    curve = subset[
                        (subset["model"] == model)
                        & (subset["attack"] == attack)
                    ].sort_values("epsilon")
                    if curve.empty:
                        continue
                    axis.plot(
                        curve["epsilon"],
                        curve[metric],
                        marker="o",
                        label=f"{model} / {attack}",
                    )
            axis.set_xlabel("Epsilon (scaled L-infinity)")
            axis.set_ylabel(metric.replace("adversarial_", "").replace("_", " "))
            axis.set_ylim(0.0, 1.0)
            axis.grid(True, alpha=0.3)
            axis.set_title(f"Evasion on {target} ({metric})")
            axis.legend(fontsize=8, ncol=2)
            paths.append(
                _save(figure, figures_dir / f"{filename}_{target}.png")
            )
    return paths


def plot_defense_tradeoff(results_dir: Path, figures_dir: Path) -> list[Path]:
    path = results_dir / "defense_metrics.csv"
    if not path.exists():
        LOGGER.warning("Skipping defense trade-off; %s is missing", path)
        return []
    frame = pd.read_csv(path)
    paths: list[Path] = []
    for target in _ordered_present(frame["target"], TARGET_ORDER):
        subset = frame[
            (frame["target"] == target) & (frame["model"] == "mlp")
        ]
        if subset.empty:
            continue
        labels: list[str] = []
        clean_values: list[float] = []
        adversarial_values: list[float] = []
        for (defense, parameter), group in sorted(
            subset.groupby(["defense", "defense_param"]),
            key=lambda item: (
                DEFENSE_ORDER.index(item[0][0])
                if item[0][0] in DEFENSE_ORDER
                else len(DEFENSE_ORDER),
                item[0][1],
            ),
        ):
            clean_rows = group[group["attack"] == "clean"]
            pgd_rows = group[group["attack"] == "pgd"]
            if clean_rows.empty or pgd_rows.empty:
                continue
            if defense == "adversarial_training":
                label = f"adv train eps={parameter:g}"
            elif defense == "feature_squeezing":
                label = f"squeeze {int(parameter)}-bit"
            else:
                label = f"{defense} {parameter:g}"
            labels.append(label)
            clean_values.append(
                float(clean_rows["clean_accuracy"].iloc[0])
            )
            adversarial_values.append(
                float(pgd_rows["adversarial_accuracy"].min())
            )
        if not labels:
            continue

        positions = np.arange(len(labels))
        width = 0.38
        figure, axis = plt.subplots(
            figsize=(max(7.0, 1.8 * len(labels)), 4.5)
        )
        axis.bar(
            positions - width / 2,
            clean_values,
            width,
            label="Clean accuracy",
        )
        axis.bar(
            positions + width / 2,
            adversarial_values,
            width,
            label="Worst PGD accuracy",
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.set_ylabel("Accuracy")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(f"Defense trade-off ({target})")
        axis.legend(fontsize=8)
        paths.append(
            _save(figure, figures_dir / f"defense_tradeoff_{target}.png")
        )
    return paths


def generate_all(results_dir: Path, figures_dir: Path) -> list[Path]:
    """Generate every figure supported by the result files that exist."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.extend(plot_baseline_metrics(results_dir, figures_dir))
    paths.extend(plot_per_class_recall(results_dir, figures_dir))
    paths.extend(plot_confusion_matrices(results_dir, figures_dir))
    paths.extend(plot_attack_curves(results_dir, figures_dir))
    paths.extend(plot_defense_tradeoff(results_dir, figures_dir))
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
    figures_dir = Path(config["paths"].get("figures", "figures"))
    paths = generate_all(results_dir, figures_dir)
    LOGGER.info("Wrote %d figures to %s", len(paths), figures_dir)


if __name__ == "__main__":
    main()
