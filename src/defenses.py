"""Defenses: FGSM adversarial training and feature squeezing."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.attacks import (
    attack_success_rate,
    evasion_population,
    evasion_rate,
    fgsm_attack,
    invert_and_clip,
    pgd_attack,
    raw_feature_bounds,
    scaled_feature_bounds,
)
from src.config import configure_logging, load_config, set_seed
from src.evaluate import benign_class_index, compute_metrics
from src.models import build_mlp, load_mlp, save_mlp
from src.train import (
    MODEL_NAMES,
    TARGET_CONFIG,
    Preprocessor,
    load_splits,
    predict_mlp,
    resolve_features,
    target_arrays,
    transform_features,
)

LOGGER = logging.getLogger(__name__)

DEFENSE_NAMES = ("adversarial_training", "feature_squeezing")
ATTACK_NAMES = ("fgsm", "pgd")


def _tensor_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(lower, dtype=torch.float32, device=device),
        torch.as_tensor(upper, dtype=torch.float32, device=device),
    )


def _fgsm_batch(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    original = features.detach()
    perturbed = original.clone().requires_grad_(True)
    loss = nn.functional.cross_entropy(model(perturbed), labels)
    gradient = torch.autograd.grad(loss, perturbed)[0]
    adversarial = perturbed.detach() + epsilon * gradient.sign()
    adversarial = torch.max(
        torch.min(adversarial, original + epsilon),
        original - epsilon,
    )
    return torch.max(torch.min(adversarial, upper), lower).detach()


def train_mlp_adversarial(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    epsilon: float,
    scaled_bounds: tuple[np.ndarray, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame]:
    """Train an MLP on a mix of clean and FGSM examples."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    defense_cfg = config["defenses"]["adversarial_training"]
    clean_fraction = float(defense_cfg.get("clean_fraction", 0.5))
    if not 0.0 <= clean_fraction <= 1.0:
        raise ValueError("clean_fraction must be between 0 and 1")
    epochs = int(defense_cfg.get("epochs", 20))
    if epochs <= 0:
        raise ValueError("defenses.adversarial_training.epochs must be positive")

    params = config["models"]["mlp"]
    batch_size = int(params["batch_size"])
    model = build_mlp(X_train.shape[1], num_classes, config).to(device)
    dataset = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(int(config["seed"])),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    lower, upper = _tensor_bounds(scaled_bounds[0], scaled_bounds[1], device)
    evaluation_batch = int(
        config.get("evaluation", {}).get("batch_size", 4096)
    )

    best_score = -1.0
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            model.eval()
            adversarial = _fgsm_batch(
                model,
                features,
                labels,
                epsilon,
                lower,
                upper,
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            clean_loss = nn.functional.cross_entropy(
                model(features),
                labels,
            )
            adversarial_loss = nn.functional.cross_entropy(
                model(adversarial),
                labels,
            )
            loss = (
                clean_fraction * clean_loss
                + (1.0 - clean_fraction) * adversarial_loss
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(features)
            total_rows += len(features)

        model.eval()
        clean_predictions = predict_mlp(
            model,
            X_val,
            evaluation_batch,
            device,
        )
        clean_metrics = compute_metrics(
            y_val,
            clean_predictions,
            labels=range(num_classes),
        )
        adversarial_val = fgsm_attack(
            model,
            X_val,
            y_val,
            epsilon=epsilon,
            clip_min=scaled_bounds[0],
            clip_max=scaled_bounds[1],
            batch_size=evaluation_batch,
            device=device,
        )
        adversarial_predictions = predict_mlp(
            model,
            adversarial_val,
            evaluation_batch,
            device,
        )
        adversarial_metrics = compute_metrics(
            y_val,
            adversarial_predictions,
            labels=range(num_classes),
        )
        robust_score = 0.5 * (
            clean_metrics["macro_f1"] + adversarial_metrics["macro_f1"]
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / max(total_rows, 1),
                "val_clean_macro_f1": clean_metrics["macro_f1"],
                "val_robust_macro_f1": robust_score,
            }
        )
        if robust_score > best_score:
            best_score = robust_score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            LOGGER.info(
                "Adversarial MLP (%s) epoch %d/%d clean_f1=%.4f robust_f1=%.4f",
                epsilon,
                epoch,
                epochs,
                clean_metrics["macro_f1"],
                robust_score,
            )

    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)


@dataclass(frozen=True)
class FeatureSqueezer:
    """Per-feature quantile quantizer fitted on the training split only.

    A linear minimum-to-maximum quantizer is unusable on CICIoT2023: the flow
    statistics are heavy tailed and about 56% of the raw values are exactly
    zero, so nearly every observation lands in the first of the 256 bins. That
    version of the defense cost the multiclass Random Forest about 58 points of
    clean accuracy, which measured the quantizer's brokenness rather than the
    defense's strength.

    Two alternatives were measured against the clean-accuracy gate:

    * ``sign(x) * log1p(|x|)`` spacing kept 79.9% on the multiclass Random
      Forest against a 99.0% baseline, still far outside the one-point budget,
      because a handful of huge values still consume most of the levels.
    * Per-feature quantile bins, used here, spend the levels in proportion to
      how often values actually occur, so the dense region around zero keeps
      its resolution. Cost stays under half a point on every model.

    Quantile edges also make the transform lossless on discrete features: a
    value that coincides with a bin edge is passed through unchanged, so the
    protocol flags and the exact zeros are reproduced exactly instead of being
    nudged by half a bin. Edges and bin means are estimated from the training
    features only; values outside the training range are clamped to the nearest
    edge, which is what makes the quantizer safe on adversarial inputs.
    """

    bit_depth: int
    edges: np.ndarray
    bin_means: np.ndarray

    def __post_init__(self) -> None:
        if not 1 <= int(self.bit_depth) <= 32:
            raise ValueError("bit_depth must be between 1 and 32")
        if self.edges.ndim != 2 or self.bin_means.ndim != 2:
            raise ValueError("edges and bin_means must be 2D arrays")
        if self.edges.shape[0] != self.bin_means.shape[0] + 1:
            raise ValueError("edges need exactly one more row than bin_means")
        if self.edges.shape[1] != self.bin_means.shape[1]:
            raise ValueError("edges and bin_means must describe the same features")

    @property
    def levels(self) -> int:
        return int(self.edges.shape[0]) - 1

    @classmethod
    def fit(cls, features: np.ndarray, bit_depth: int) -> "FeatureSqueezer":
        """Estimate quantile edges and bin centres from training features."""
        if not 1 <= int(bit_depth) <= 32:
            raise ValueError("bit_depth must be between 1 and 32")
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("features must be a 2D array")

        levels = 2 ** int(bit_depth)
        num_features = values.shape[1]
        edges = np.empty((levels + 1, num_features), dtype=np.float64)
        bin_means = np.empty((levels, num_features), dtype=np.float64)

        for column in range(num_features):
            values_column = values[:, column]
            quantiles = np.quantile(
                values_column,
                np.linspace(0.0, 1.0, levels + 1),
            )
            edges[:, column] = quantiles
            indices = np.clip(
                np.searchsorted(quantiles, values_column, side="right") - 1,
                0,
                levels - 1,
            )
            totals = np.bincount(
                indices,
                weights=values_column,
                minlength=levels,
            )
            counts = np.bincount(indices, minlength=levels)
            bin_means[:, column] = np.where(
                counts > 0,
                totals / np.maximum(counts, 1),
                quantiles[:-1],
            )
        return cls(bit_depth=int(bit_depth), edges=edges, bin_means=bin_means)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Quantize features to per-feature quantile bins."""
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.edges.shape[1]:
            raise ValueError("features must be a 2D array matching the bounds")

        squeezed = np.empty_like(values)
        for column in range(values.shape[1]):
            values_column = values[:, column]
            column_edges = self.edges[:, column]
            indices = np.clip(
                np.searchsorted(column_edges, values_column, side="right") - 1,
                0,
                self.levels - 1,
            )
            mapped = self.bin_means[indices, column]
            # Exact hits on an edge are kept, which preserves zeros and the
            # discrete protocol flags bit for bit.
            on_edge = column_edges[indices] == values_column
            mapped = np.where(on_edge, values_column, mapped)
            mapped = np.where(values_column < column_edges[0], column_edges[0], mapped)
            mapped = np.where(
                values_column > column_edges[-1], column_edges[-1], mapped
            )
            squeezed[:, column] = mapped
        return squeezed.astype(np.float32)

    def reconstruction_error(self, features: np.ndarray) -> np.ndarray:
        """Return the absolute per-element error introduced by the quantizer."""
        values = np.asarray(features, dtype=np.float64)
        return np.abs(self.transform(values).astype(np.float64) - values)


def _generate_attack(
    attack_name: str,
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
    bounds: tuple[np.ndarray, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    batch_size = int(config.get("evaluation", {}).get("batch_size", 4096))
    if attack_name == "fgsm":
        return fgsm_attack(
            model,
            features,
            labels,
            epsilon=epsilon,
            clip_min=bounds[0],
            clip_max=bounds[1],
            batch_size=batch_size,
            device=device,
        )
    if attack_name == "pgd":
        params = config["attacks"]["pgd"]
        return pgd_attack(
            model,
            features,
            labels,
            epsilon=epsilon,
            steps=int(params["steps"]),
            clip_min=bounds[0],
            clip_max=bounds[1],
            step_size_factor=float(params.get("step_size_factor", 0.25)),
            random_start=bool(params.get("random_start", True)),
            batch_size=batch_size,
            device=device,
        )
    raise ValueError(f"Unknown attack: {attack_name!r}")


def _metric_row(
    *,
    target: str,
    defense: str,
    defense_param: float,
    model_name: str,
    attack: str,
    epsilon: float,
    labels: np.ndarray,
    clean_predictions: np.ndarray,
    adversarial_predictions: np.ndarray,
    baseline_clean_accuracy: float,
    num_classes: int,
    benign_index: int,
) -> dict[str, Any]:
    clean_metrics = compute_metrics(
        labels,
        clean_predictions,
        labels=range(num_classes),
    )
    adversarial_metrics = compute_metrics(
        labels,
        adversarial_predictions,
        labels=range(num_classes),
    )
    return {
        "target": target,
        "defense": defense,
        "defense_param": float(defense_param),
        "model": model_name,
        "attack": attack,
        "epsilon": float(epsilon),
        "n_samples": int(len(labels)),
        "baseline_clean_accuracy": float(baseline_clean_accuracy),
        "clean_accuracy": clean_metrics["accuracy"],
        "adversarial_accuracy": adversarial_metrics["accuracy"],
        "accuracy_drop": clean_metrics["accuracy"]
        - adversarial_metrics["accuracy"],
        "clean_macro_f1": clean_metrics["macro_f1"],
        "adversarial_macro_f1": adversarial_metrics["macro_f1"],
        "macro_f1_drop": clean_metrics["macro_f1"]
        - adversarial_metrics["macro_f1"],
        "attack_success_rate": attack_success_rate(
            clean_predictions,
            adversarial_predictions,
            labels,
        ),
        "evasion_rate": evasion_rate(
            labels,
            clean_predictions,
            adversarial_predictions,
            benign_index,
        ),
        "evasion_population": evasion_population(
            labels,
            clean_predictions,
            benign_index,
        ),
        "clean_accuracy_cost": float(baseline_clean_accuracy)
        - clean_metrics["accuracy"],
    }


def _load_tree_model(models_dir: Path, target: str, model_name: str) -> Any:
    path = models_dir / f"{target}_{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model {path}. Run `python -m src.train --config "
            "configs/default.yaml` first."
        )
    return joblib.load(path)


def run(
    config: dict[str, Any],
    targets: Sequence[str],
    model_names: Sequence[str],
    defense_names: Sequence[str],
) -> None:
    """Train/evaluate the requested defenses and write result files."""
    unknown_targets = sorted(set(targets) - set(TARGET_CONFIG))
    if unknown_targets:
        raise ValueError(f"Unknown targets: {unknown_targets}")
    unknown_models = sorted(set(model_names) - set(MODEL_NAMES))
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}")
    unknown_defenses = sorted(set(defense_names) - set(DEFENSE_NAMES))
    if unknown_defenses:
        raise ValueError(f"Unknown defenses: {unknown_defenses}")
    if not defense_names:
        raise ValueError("At least one defense must be requested")
    if "mlp" not in model_names:
        raise ValueError("The MLP must be among the evaluated models")

    set_seed(int(config["seed"]), bool(config.get("deterministic", True)))
    splits = load_splits(config)
    train_frame = splits["train"]
    val_frame = splits["val"]
    test_frame = splits["test"]
    features = resolve_features(train_frame)

    models_dir = Path(config["data"].get("models_dir", "data/models"))
    results_dir = Path(config["paths"].get("results", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    preprocessor_path = models_dir / "preprocessor.joblib"
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Missing preprocessor {preprocessor_path}. Run `python -m src.train "
            "--config configs/default.yaml` first."
        )
    preprocessor: Preprocessor = joblib.load(preprocessor_path)
    if preprocessor.features != features:
        raise ValueError(
            "The saved preprocessor does not match the available features"
        )

    train_scaled = transform_features(train_frame, preprocessor, scale=True)
    train_tree = transform_features(train_frame, preprocessor, scale=False)
    val_scaled = transform_features(val_frame, preprocessor, scale=True)
    test_scaled = transform_features(test_frame, preprocessor, scale=True)
    test_tree = transform_features(test_frame, preprocessor, scale=False)
    scaled_bounds = scaled_feature_bounds(train_scaled)
    raw_bounds = raw_feature_bounds(train_frame, preprocessor)
    batch_size = int(config.get("evaluation", {}).get("batch_size", 4096))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Defense device: %s", device)

    squeezing_cfg = config["defenses"]["feature_squeezing"]
    acceptance_bits = int(squeezing_cfg.get("acceptance_bit_depth", 8))
    acceptance_tolerance = float(squeezing_cfg.get("acceptance_tolerance", 0.01))
    enforce_acceptance = bool(squeezing_cfg.get("enforce_acceptance", True))

    rows: list[dict[str, Any]] = []
    acceptance_rows: list[dict[str, Any]] = []

    for target in targets:
        labels, class_names = target_arrays(test_frame, target)
        num_classes = len(class_names)
        benign_index = benign_class_index(target, class_names)
        baseline_mlp_path = models_dir / f"{target}_mlp.pt"
        if not baseline_mlp_path.exists():
            raise FileNotFoundError(
                f"Missing model {baseline_mlp_path}. Run `python -m src.train "
                "--config configs/default.yaml` first."
            )
        baseline_mlp = load_mlp(baseline_mlp_path, device)
        tree_models = {
            name: _load_tree_model(models_dir, target, name)
            for name in model_names
            if name != "mlp"
        }
        baseline_clean_predictions = {
            "mlp": predict_mlp(
                baseline_mlp,
                test_scaled,
                batch_size,
                device,
            )
        }
        for name, model in tree_models.items():
            baseline_clean_predictions[name] = model.predict(
                test_tree
            ).astype(np.int64)
        baseline_clean_accuracy = {
            name: compute_metrics(
                labels,
                predictions,
                labels=range(num_classes),
            )["accuracy"]
            for name, predictions in baseline_clean_predictions.items()
        }

        if "adversarial_training" in defense_names:
            train_labels, _ = target_arrays(train_frame, target)
            val_labels, _ = target_arrays(val_frame, target)
            for training_epsilon in config["defenses"]["adversarial_training"][
                "epsilons"
            ]:
                training_epsilon = float(training_epsilon)
                LOGGER.info(
                    "Target %s: adversarial training at epsilon=%s",
                    target,
                    training_epsilon,
                )
                model, history = train_mlp_adversarial(
                    train_scaled,
                    train_labels,
                    val_scaled,
                    val_labels,
                    num_classes,
                    training_epsilon,
                    scaled_bounds,
                    config,
                    device,
                )
                tag = f"{training_epsilon:g}"
                save_mlp(
                    model,
                    input_dim=len(features),
                    num_classes=num_classes,
                    path=models_dir / f"{target}_mlp_adv_{tag}.pt",
                    config=config,
                )
                history.to_csv(
                    results_dir
                    / f"defense_training_{target}_adv_{tag}.csv",
                    index=False,
                    float_format="%.6f",
                )
                clean_predictions = predict_mlp(
                    model,
                    test_scaled,
                    batch_size,
                    device,
                )
                rows.append(
                    _metric_row(
                        target=target,
                        defense="adversarial_training",
                        defense_param=training_epsilon,
                        model_name="mlp",
                        attack="clean",
                        epsilon=0.0,
                        labels=labels,
                        clean_predictions=clean_predictions,
                        adversarial_predictions=clean_predictions,
                        baseline_clean_accuracy=baseline_clean_accuracy["mlp"],
                        num_classes=num_classes,
                        benign_index=benign_index,
                    )
                )
                for attack_name in ATTACK_NAMES:
                    for epsilon in config["attacks"][attack_name]["epsilons"]:
                        epsilon = float(epsilon)
                        adversarial_scaled = _generate_attack(
                            attack_name,
                            model,
                            test_scaled,
                            labels,
                            epsilon,
                            scaled_bounds,
                            config,
                            device,
                        )
                        adversarial_predictions = predict_mlp(
                            model,
                            adversarial_scaled,
                            batch_size,
                            device,
                        )
                        rows.append(
                            _metric_row(
                                target=target,
                                defense="adversarial_training",
                                defense_param=training_epsilon,
                                model_name="mlp",
                                attack=attack_name,
                                epsilon=epsilon,
                                labels=labels,
                                clean_predictions=clean_predictions,
                                adversarial_predictions=adversarial_predictions,
                                baseline_clean_accuracy=baseline_clean_accuracy[
                                    "mlp"
                                ],
                                num_classes=num_classes,
                                benign_index=benign_index,
                            )
                        )

        if "feature_squeezing" in defense_names:
            adversarial_sets: dict[tuple[str, float], np.ndarray] = {}
            for attack_name in ATTACK_NAMES:
                for epsilon in config["attacks"][attack_name]["epsilons"]:
                    epsilon = float(epsilon)
                    adversarial_sets[(attack_name, epsilon)] = _generate_attack(
                        attack_name,
                        baseline_mlp,
                        test_scaled,
                        labels,
                        epsilon,
                        scaled_bounds,
                        config,
                        device,
                    )

            for bit_depth in config["defenses"]["feature_squeezing"][
                "bit_depths"
            ]:
                bit_depth = int(bit_depth)
                LOGGER.info(
                    "Target %s: feature squeezing at %d bits",
                    target,
                    bit_depth,
                )
                # Fit on the training split only, then apply to the test split.
                squeezer_mlp = FeatureSqueezer.fit(train_scaled, bit_depth)
                squeezer_tree = FeatureSqueezer.fit(train_tree, bit_depth)
                squeezed_test_mlp = squeezer_mlp.transform(test_scaled)
                squeezed_test_tree = squeezer_tree.transform(test_tree)
                clean_predictions = {
                    "mlp": predict_mlp(
                        baseline_mlp,
                        squeezed_test_mlp,
                        batch_size,
                        device,
                    )
                }
                for name, model in tree_models.items():
                    clean_predictions[name] = model.predict(
                        squeezed_test_tree
                    ).astype(np.int64)

                for name in model_names:
                    rows.append(
                        _metric_row(
                            target=target,
                            defense="feature_squeezing",
                            defense_param=float(bit_depth),
                            model_name=name,
                            attack="clean",
                            epsilon=0.0,
                            labels=labels,
                            clean_predictions=clean_predictions[name],
                            adversarial_predictions=clean_predictions[name],
                            baseline_clean_accuracy=baseline_clean_accuracy[name],
                            num_classes=num_classes,
                            benign_index=benign_index,
                        )
                    )
                    acceptance_rows.append(
                        {
                            "target": target,
                            "model": name,
                            "bit_depth": bit_depth,
                            "baseline_clean_accuracy": baseline_clean_accuracy[name],
                            "squeezed_clean_accuracy": float(
                                compute_metrics(
                                    labels,
                                    clean_predictions[name],
                                    labels=range(num_classes),
                                )["accuracy"]
                            ),
                            "max_allowed_drop": acceptance_tolerance,
                        }
                    )

                for (attack_name, epsilon), adversarial_scaled in (
                    adversarial_sets.items()
                ):
                    adversarial_raw = invert_and_clip(
                        adversarial_scaled,
                        preprocessor,
                        raw_bounds[0],
                        raw_bounds[1],
                    )
                    squeezed_adversarial_mlp = squeezer_mlp.transform(
                        adversarial_scaled
                    )
                    squeezed_adversarial_tree = squeezer_tree.transform(
                        adversarial_raw
                    )
                    adversarial_predictions = {
                        "mlp": predict_mlp(
                            baseline_mlp,
                            squeezed_adversarial_mlp,
                            batch_size,
                            device,
                        )
                    }
                    for name, model in tree_models.items():
                        adversarial_predictions[name] = model.predict(
                            squeezed_adversarial_tree
                        ).astype(np.int64)

                    for name in model_names:
                        rows.append(
                            _metric_row(
                                target=target,
                                defense="feature_squeezing",
                                defense_param=float(bit_depth),
                                model_name=name,
                                attack=attack_name,
                                epsilon=epsilon,
                                labels=labels,
                                clean_predictions=clean_predictions[name],
                                adversarial_predictions=adversarial_predictions[
                                    name
                                ],
                                baseline_clean_accuracy=baseline_clean_accuracy[
                                    name
                                ],
                                num_classes=num_classes,
                                benign_index=benign_index,
                            )
                        )

    pd.DataFrame(rows).to_csv(
        results_dir / "defense_metrics.csv",
        index=False,
        float_format="%.6f",
    )

    acceptance = pd.DataFrame(acceptance_rows)
    if not acceptance.empty:
        acceptance["drop"] = (
            acceptance["baseline_clean_accuracy"]
            - acceptance["squeezed_clean_accuracy"]
        )
        acceptance["passed"] = acceptance["drop"] <= acceptance["max_allowed_drop"]
        acceptance.to_csv(
            results_dir / "squeezing_acceptance.csv",
            index=False,
            float_format="%.6f",
        )
        failures = acceptance[
            (acceptance["bit_depth"] == acceptance_bits) & ~acceptance["passed"]
        ]
        if enforce_acceptance and not failures.empty:
            raise RuntimeError(
                "Feature squeezing failed the clean-accuracy acceptance check at "
                f"{acceptance_bits} bits; see results/squeezing_acceptance.csv. "
                "Failing rows: "
                + failures[["target", "model", "drop"]].to_string(index=False)
            )

    manifest = {
        "seed": int(config["seed"]),
        "features": features,
        "targets": list(targets),
        "models": list(model_names),
        "defenses": list(defense_names),
        "adversarial_training_epsilons": [
            float(epsilon)
            for epsilon in config["defenses"]["adversarial_training"][
                "epsilons"
            ]
        ],
        "adversarial_training_epochs": int(
            config["defenses"]["adversarial_training"].get("epochs", 20)
        ),
        "adversarial_training_clean_fraction": float(
            config["defenses"]["adversarial_training"].get(
                "clean_fraction",
                0.5,
            )
        ),
        "feature_squeezing_bit_depths": [
            int(bits)
            for bits in config["defenses"]["feature_squeezing"]["bit_depths"]
        ],
        "device": str(device),
        "test_rows": int(len(test_frame)),
    }
    with (results_dir / "defense_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2)

    LOGGER.info("Wrote defense metrics to %s", results_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=sorted(TARGET_CONFIG),
        default=None,
        help="Label granularities to defend (default: evaluation.targets).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=None,
        help="Models evaluated for feature squeezing (default: evaluation.models).",
    )
    parser.add_argument(
        "--defenses",
        nargs="+",
        choices=DEFENSE_NAMES,
        default=None,
        help="Defenses to run (default: both).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()
    config = load_config(args.config)
    evaluation_cfg = config.get("evaluation", {})
    targets = args.targets or evaluation_cfg.get("targets", list(TARGET_CONFIG))
    model_names = args.models or evaluation_cfg.get("models", list(MODEL_NAMES))
    defense_names = args.defenses or list(DEFENSE_NAMES)
    run(config, targets, model_names, defense_names)


if __name__ == "__main__":
    main()
