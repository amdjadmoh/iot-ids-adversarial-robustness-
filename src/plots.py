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


def _low_support_classes(
    results_dir: Path,
    target: str,
    min_test_samples: int,
) -> set[str]:
    """Return the classes flagged as having too few test rows to trust."""
    path = results_dir / "test_class_counts.csv"
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    subset = frame[
        (frame["target"] == target) & (frame["test"] < min_test_samples)
    ]
    return set(subset["class_name"])


def plot_per_class_recall(
    results_dir: Path,
    figures_dir: Path,
    min_test_samples: int = 30,
) -> list[Path]:
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
        low_support = _low_support_classes(
            results_dir,
            target,
            min_test_samples,
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
            bars = axis.bar(
                positions + offset,
                values.to_numpy(),
                width=bar_width,
                label=model,
            )
            for bar, class_name in zip(bars, class_names):
                if class_name in low_support:
                    bar.set_hatch("///")
                    bar.set_edgecolor("black")
                    bar.set_linewidth(0.4)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [
                f"{name}*" if name in low_support else name
                for name in class_names
            ],
            rotation=90,
            fontsize=7,
        )
        axis.set_ylabel("Recall")
        axis.set_xlabel("Class")
        axis.set_ylim(0.0, 1.05)
        title = f"Per-class test recall ({target})"
        if low_support:
            title += f"\n* fewer than {min_test_samples} test samples (hatched)"
        axis.set_title(title)
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
    """Plot clean cost and robustness as matched-epsilon comparisons.

    The left panel reports clean accuracy and accuracy under PGD at one shared
    epsilon; the right panel shows the whole robustness curve. Comparing a
    baseline at the largest epsilon with a defense at a smaller one would make
    the defense look better than it is, so every bar is measured at the same
    epsilon.
    """
    path = results_dir / "defense_metrics.csv"
    if not path.exists():
        LOGGER.warning("Skipping defense trade-off; %s is missing", path)
        return []
    frame = pd.read_csv(path)
    attack_path = results_dir / "attack_metrics.csv"
    baseline_frame = (
        pd.read_csv(attack_path)
        if attack_path.exists()
        else pd.DataFrame(columns=["target", "model", "attack", "epsilon"])
    )
    has_baseline_attacks = "attack" in baseline_frame.columns

    paths: list[Path] = []
    for target in _ordered_present(frame["target"], TARGET_ORDER):
        subset = frame[
            (frame["target"] == target) & (frame["model"] == "mlp")
        ]
        if subset.empty:
            continue
        baseline_pgd = (
            baseline_frame[
                (baseline_frame["target"] == target)
                & (baseline_frame["model"] == "mlp")
                & (baseline_frame["attack"] == "pgd")
            ].sort_values("epsilon")
            if has_baseline_attacks
            else pd.DataFrame()
        )

        groups = sorted(
            subset.groupby(["defense", "defense_param"]),
            key=lambda item: (
                DEFENSE_ORDER.index(item[0][0])
                if item[0][0] in DEFENSE_ORDER
                else len(DEFENSE_ORDER),
                item[0][1],
            ),
        )

        shared: set[float] | None = None
        if not baseline_pgd.empty:
            shared = set(baseline_pgd["epsilon"].tolist())
        for _, group in groups:
            pgd_rows = group[group["attack"] == "pgd"]
            if pgd_rows.empty:
                continue
            epsilons = set(pgd_rows["epsilon"].tolist())
            shared = epsilons if shared is None else shared & epsilons
        bar_epsilon = (
            max(shared)
            if shared
            else float(subset["epsilon"].max())
        )

        labels: list[str] = []
        clean_values: list[float] = []
        robust_values: list[float] = []
        curves: list[tuple[str, pd.DataFrame, str | None]] = []

        if not baseline_pgd.empty:
            baseline_row = baseline_pgd[baseline_pgd["epsilon"] == bar_epsilon]
            if not baseline_row.empty:
                labels.append("no defense")
                clean_values.append(float(baseline_row["clean_accuracy"].iloc[0]))
                robust_values.append(
                    float(baseline_row["adversarial_accuracy"].iloc[0])
                )
            curves.append(("no defense", baseline_pgd, "black"))

        for (defense, parameter), group in groups:
            clean_rows = group[group["attack"] == "clean"]
            pgd_rows = group[group["attack"] == "pgd"].sort_values("epsilon")
            if clean_rows.empty or pgd_rows.empty:
                continue
            if defense == "adversarial_training":
                label = f"adv train eps={parameter:g}"
            elif defense == "feature_squeezing":
                label = f"squeeze {int(parameter)}-bit"
            else:
                label = f"{defense} {parameter:g}"
            at_bar = pgd_rows[pgd_rows["epsilon"] == bar_epsilon]
            if at_bar.empty:
                continue
            labels.append(label)
            clean_values.append(float(clean_rows["clean_accuracy"].iloc[0]))
            robust_values.append(float(at_bar["adversarial_accuracy"].iloc[0]))
            curves.append((label, pgd_rows, None))
        if not labels:
            continue

        figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.8))
        positions = np.arange(len(labels))
        width = 0.38
        left.bar(
            positions - width / 2,
            clean_values,
            width,
            label="Clean accuracy",
        )
        left.bar(
            positions + width / 2,
            robust_values,
            width,
            label=f"PGD accuracy (eps={bar_epsilon:g})",
        )
        left.set_xticks(positions)
        left.set_xticklabels(labels, rotation=20, ha="right")
        left.set_ylabel("Accuracy")
        left.set_ylim(0.0, 1.0)
        left.set_title(f"Clean cost and robustness at eps={bar_epsilon:g}")
        left.legend(fontsize=8)
        left.grid(True, axis="y", alpha=0.3)

        for label, curve, colour in curves:
            right.plot(
                curve["epsilon"],
                curve["adversarial_accuracy"],
                marker="o",
                color=colour,
                label=label,
            )
        right.set_xlabel("Epsilon (scaled L-infinity)")
        right.set_ylabel("PGD accuracy")
        right.set_ylim(0.0, 1.0)
        right.set_title("Robustness across the perturbation budget")
        right.grid(True, alpha=0.3)
        right.legend(fontsize=7)
        figure.suptitle(f"Defense trade-off ({target})")
        paths.append(
            _save(figure, figures_dir / f"defense_tradeoff_{target}.png")
        )
    return paths


def generate_all(
    results_dir: Path,
    figures_dir: Path,
    min_test_samples: int = 30,
) -> list[Path]:
    """Generate every figure supported by the result files that exist."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.extend(plot_baseline_metrics(results_dir, figures_dir))
    paths.extend(
        plot_per_class_recall(results_dir, figures_dir, min_test_samples)
    )
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
    paths = generate_all(
        results_dir,
        figures_dir,
        min_test_samples=int(
            config.get("data", {}).get("rare_class_min_test_samples", 30)
        ),
    )
    LOGGER.info("Wrote %d figures to %s", len(paths), figures_dir)


if __name__ == "__main__":
    main()
