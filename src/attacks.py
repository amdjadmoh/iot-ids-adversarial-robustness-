"""White-box (FGSM/PGD), feature-constrained, noise-control, and transfer evasion.

The random-noise attack is the control condition for the transfer experiments:
it spends the same L-infinity budget as the gradient attacks without using any
information about the model. Transfer is only meaningful to the extent that a
gradient attack beats that control.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.config import configure_logging, load_config, set_seed
from src.constraints import project_feature_constraints
from src.evaluate import benign_class_index, compute_metrics, per_class_recall_frame
from src.features import FEATURE_NAMES
from src.models import load_mlp
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

ATTACK_NAMES = ("fgsm", "pgd", "constrained", "random_noise")


def _replace_non_finite(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan)


def _as_bounds(
    clip_min: float | np.ndarray,
    clip_max: float | np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower = torch.as_tensor(clip_min, device=device, dtype=dtype)
    upper = torch.as_tensor(clip_max, device=device, dtype=dtype)
    if torch.any(lower > upper):
        raise ValueError("clip_min must not exceed clip_max")
    return lower, upper


def _project(
    features: torch.Tensor,
    original: torch.Tensor,
    epsilon: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    features = torch.max(torch.min(features, original + epsilon), original - epsilon)
    return torch.max(torch.min(features, upper), lower)


def _batch_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, labels)


def fgsm_attack(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
    clip_min: float | np.ndarray,
    clip_max: float | np.ndarray,
    batch_size: int = 4096,
    device: torch.device | None = None,
) -> np.ndarray:
    """Return one-step FGSM adversarial examples in the model's input space."""
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    features_tensor = torch.as_tensor(features, dtype=torch.float32)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    lower, upper = _as_bounds(clip_min, clip_max, device, torch.float32)

    batches: list[np.ndarray] = []
    for start in range(0, len(features_tensor), batch_size):
        original = features_tensor[start : start + batch_size].to(device)
        labels_batch = labels_tensor[start : start + batch_size].to(device)
        perturbed = original.detach().clone().requires_grad_(True)
        loss = _batch_loss(model(perturbed), labels_batch)
        gradient = torch.autograd.grad(loss, perturbed)[0]
        adversarial = _project(
            perturbed.detach() + epsilon * gradient.sign(),
            original,
            epsilon,
            lower,
            upper,
        )
        batches.append(adversarial.cpu().numpy())

    if not batches:
        return np.empty_like(features, dtype=np.float32)
    return np.concatenate(batches, axis=0).astype(np.float32)


def random_noise_attack(
    features: np.ndarray,
    epsilon: float,
    clip_min: float | np.ndarray,
    clip_max: float | np.ndarray,
    seed: int,
    batch_size: int = 4096,
) -> np.ndarray:
    """Return a random-sign perturbation of magnitude ``epsilon``.

    This is the control condition for the transfer experiments: it moves every
    feature by the same L-infinity budget as FGSM/PGD but carries no information
    about the model, so any accuracy loss it causes is pure input fragility.
    A transfer attack only demonstrates transferability for the part of the loss
    that lies beyond this baseline.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    features_tensor = torch.as_tensor(features, dtype=torch.float32)
    device = torch.device("cpu")
    lower, upper = _as_bounds(clip_min, clip_max, device, torch.float32)
    rng = np.random.default_rng(seed)

    batches: list[np.ndarray] = []
    for start in range(0, len(features_tensor), batch_size):
        original = features_tensor[start : start + batch_size]
        signs = rng.integers(0, 2, size=tuple(original.shape), dtype=np.int8) * 2 - 1
        direction = torch.as_tensor(signs, dtype=torch.float32)
        adversarial = _project(
            original + epsilon * direction,
            original,
            epsilon,
            lower,
            upper,
        )
        batches.append(adversarial.numpy())

    if not batches:
        return np.empty_like(features, dtype=np.float32)
    return np.concatenate(batches, axis=0).astype(np.float32)


def pgd_attack(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
    steps: int,
    clip_min: float | np.ndarray,
    clip_max: float | np.ndarray,
    step_size_factor: float = 0.25,
    random_start: bool = True,
    batch_size: int = 4096,
    device: torch.device | None = None,
) -> np.ndarray:
    """Return PGD adversarial examples in the model's input space."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if step_size_factor <= 0:
        raise ValueError("step_size_factor must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    step_size = epsilon * step_size_factor
    features_tensor = torch.as_tensor(features, dtype=torch.float32)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    lower, upper = _as_bounds(clip_min, clip_max, device, torch.float32)

    batches: list[np.ndarray] = []
    for start in range(0, len(features_tensor), batch_size):
        original = features_tensor[start : start + batch_size].to(device)
        labels_batch = labels_tensor[start : start + batch_size].to(device)
        if random_start:
            adversarial = original + torch.empty_like(original).uniform_(
                -epsilon,
                epsilon,
            )
        else:
            adversarial = original.clone()
        adversarial = _project(adversarial, original, epsilon, lower, upper)

        for _ in range(steps):
            adversarial = adversarial.detach().requires_grad_(True)
            loss = _batch_loss(model(adversarial), labels_batch)
            gradient = torch.autograd.grad(loss, adversarial)[0]
            adversarial = _project(
                adversarial.detach() + step_size * gradient.sign(),
                original,
                epsilon,
                lower,
                upper,
            )
        batches.append(adversarial.cpu().numpy())

    if not batches:
        return np.empty_like(features, dtype=np.float32)
    return np.concatenate(batches, axis=0).astype(np.float32)


def constrained_pgd_attack(
    model: nn.Module,
    scaled_features: np.ndarray,
    original_raw: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
    steps: int,
    preprocessor: Preprocessor,
    raw_lower: np.ndarray,
    raw_upper: np.ndarray,
    scaled_lower: np.ndarray,
    scaled_upper: np.ndarray,
    step_size_factor: float = 0.25,
    max_feature_delta: float | None = 0.25,
    batch_size: int = 4096,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate feature-constrained adversarial examples adhering to threat-model rules.

    Each step computes gradients in scaled feature space, updates controllable features,
    inverts to raw space, projects onto physical and threat-model constraints, and
    re-scales.

    Returns:
        tuple (adversarial_scaled, adversarial_raw)
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    step_size = epsilon * step_size_factor

    batches_scaled: list[np.ndarray] = []
    batches_raw: list[np.ndarray] = []

    for start in range(0, len(scaled_features), batch_size):
        end = start + batch_size
        batch_scaled_orig = scaled_features[start:end]
        batch_raw_orig = original_raw[start:end]
        batch_labels = labels[start:end]

        current_scaled = batch_scaled_orig.copy()
        current_raw = batch_raw_orig.copy()

        labels_tensor = torch.as_tensor(batch_labels, dtype=torch.long, device=device)
        orig_scaled_tensor = torch.as_tensor(batch_scaled_orig, dtype=torch.float32, device=device)

        for _ in range(steps):
            tensor_scaled = torch.as_tensor(
                current_scaled, dtype=torch.float32, device=device
            ).requires_grad_(True)
            loss = _batch_loss(model(tensor_scaled), labels_tensor)
            gradient = torch.autograd.grad(loss, tensor_scaled)[0]

            with torch.no_grad():
                step_tensor = tensor_scaled + step_size * gradient.sign()
                # L-infinity clamp to epsilon in scaled space
                step_tensor = torch.max(
                    torch.min(step_tensor, orig_scaled_tensor + epsilon),
                    orig_scaled_tensor - epsilon,
                )
                step_scaled_np = step_tensor.cpu().numpy()

            # Invert to raw feature space
            raw_unprojected = preprocessor.scaler.inverse_transform(
                step_scaled_np.astype(np.float64)
            )

            # Project onto realistic threat model & physical constraints
            projected_raw = project_feature_constraints(
                adversarial_raw=raw_unprojected,
                original_raw=batch_raw_orig,
                feature_names=preprocessor.features,
                raw_lower=raw_lower,
                raw_upper=raw_upper,
                max_feature_delta=max_feature_delta,
            )

            # Re-scale back to model input space
            scaled_reprojected = preprocessor.scaler.transform(
                projected_raw.astype(np.float64)
            )
            if preprocessor.clip_value is not None:
                scaled_reprojected = np.clip(
                    scaled_reprojected,
                    -preprocessor.clip_value,
                    preprocessor.clip_value,
                )

            current_raw = projected_raw.astype(np.float32)
            current_scaled = scaled_reprojected.astype(np.float32)

        batches_scaled.append(current_scaled)
        batches_raw.append(current_raw)

    return (
        np.concatenate(batches_scaled, axis=0).astype(np.float32),
        np.concatenate(batches_raw, axis=0).astype(np.float32),
    )


def attack_success_rate(
    clean_predictions: np.ndarray,
    adversarial_predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Return the fraction of initially correct samples that become errors."""
    clean_predictions = np.asarray(clean_predictions)
    adversarial_predictions = np.asarray(adversarial_predictions)
    labels = np.asarray(labels)
    correct = clean_predictions == labels
    if not correct.any():
        return 0.0
    return float((adversarial_predictions[correct] != labels[correct]).mean())


def detected_attack_mask(
    labels: np.ndarray,
    clean_predictions: np.ndarray,
    benign_index: int,
) -> np.ndarray:
    """Return the rows that are true attacks and were detected as attacks."""
    labels = np.asarray(labels)
    clean_predictions = np.asarray(clean_predictions)
    return (labels != benign_index) & (clean_predictions != benign_index)


def evasion_rate(
    labels: np.ndarray,
    clean_predictions: np.ndarray,
    adversarial_predictions: np.ndarray,
    benign_index: int,
) -> float:
    """Fraction of detected attacks that are relabelled benign after the attack.

    Only rows whose true label is an attack *and* that the model already flagged
    as an attack count towards the denominator. Reclassifying a detected attack
    as a different attack class keeps the alarm firing, so it is not evasion;
    this metric is the one that answers "does the detector stay silent?".
    """
    labels = np.asarray(labels)
    clean_predictions = np.asarray(clean_predictions)
    adversarial_predictions = np.asarray(adversarial_predictions)
    mask = detected_attack_mask(labels, clean_predictions, benign_index)
    if not mask.any():
        return 0.0
    return float((adversarial_predictions[mask] == benign_index).mean())


def evasion_population(
    labels: np.ndarray,
    clean_predictions: np.ndarray,
    benign_index: int,
) -> int:
    """Return the number of rows the evasion rate is computed over."""
    return int(
        detected_attack_mask(labels, clean_predictions, benign_index).sum()
    )


def scaled_feature_bounds(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-feature minimum and maximum of the training input space."""
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")
    return (
        features.min(axis=0).astype(np.float32),
        features.max(axis=0).astype(np.float32),
    )


def raw_feature_bounds(
    train_frame: pd.DataFrame,
    preprocessor: Preprocessor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observed raw feature bounds after training-time imputation."""
    feature_frame = _replace_non_finite(train_frame[preprocessor.features])
    filled = feature_frame.fillna(preprocessor.medians)
    return (
        filled.min().to_numpy(dtype=np.float64),
        filled.max().to_numpy(dtype=np.float64),
    )


def invert_and_clip(
    scaled_features: np.ndarray,
    preprocessor: Preprocessor,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Map scaled adversarial examples back to clipped raw feature values."""
    if preprocessor.scaler is None:
        raise ValueError("The preprocessor does not contain a fitted scaler")
    raw = preprocessor.scaler.inverse_transform(
        np.asarray(scaled_features, dtype=np.float64)
    )
    raw = np.clip(raw, lower, upper)
    return raw.astype(np.float32)


def _generate_attack(
    attack_name: str,
    model: nn.Module,
    features_scaled: np.ndarray,
    features_raw: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
    scaled_bounds: tuple[np.ndarray, np.ndarray],
    raw_bounds: tuple[np.ndarray, np.ndarray],
    preprocessor: Preprocessor,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = int(config.get("evaluation", {}).get("batch_size", 4096))
    lower_s, upper_s = scaled_bounds
    lower_r, upper_r = raw_bounds

    if attack_name == "fgsm":
        adv_scaled = fgsm_attack(
            model,
            features_scaled,
            labels,
            epsilon=epsilon,
            clip_min=lower_s,
            clip_max=upper_s,
            batch_size=batch_size,
            device=device,
        )
        adv_raw = invert_and_clip(adv_scaled, preprocessor, lower_r, upper_r)
        return adv_scaled, adv_raw

    if attack_name == "pgd":
        params = config["attacks"]["pgd"]
        adv_scaled = pgd_attack(
            model,
            features_scaled,
            labels,
            epsilon=epsilon,
            steps=int(params["steps"]),
            clip_min=lower_s,
            clip_max=upper_s,
            step_size_factor=float(params.get("step_size_factor", 0.25)),
            random_start=bool(params.get("random_start", True)),
            batch_size=batch_size,
            device=device,
        )
        adv_raw = invert_and_clip(adv_scaled, preprocessor, lower_r, upper_r)
        return adv_scaled, adv_raw

    if attack_name == "constrained":
        params = config["attacks"]["constrained"]
        return constrained_pgd_attack(
            model,
            scaled_features=features_scaled,
            original_raw=features_raw,
            labels=labels,
            epsilon=epsilon,
            steps=int(params["steps"]),
            preprocessor=preprocessor,
            raw_lower=lower_r,
            raw_upper=upper_r,
            scaled_lower=lower_s,
            scaled_upper=upper_s,
            step_size_factor=float(params.get("step_size_factor", 0.25)),
            max_feature_delta=params.get("max_feature_delta", 0.25),
            batch_size=batch_size,
            device=device,
        )

    if attack_name == "random_noise":
        params = config["attacks"]["random_noise"]
        seed = int(params.get("seed", config["seed"]))
        adv_scaled = random_noise_attack(
            features_scaled,
            epsilon=epsilon,
            clip_min=lower_s,
            clip_max=upper_s,
            seed=seed,
            batch_size=batch_size,
        )
        adv_raw = invert_and_clip(adv_scaled, preprocessor, lower_r, upper_r)
        return adv_scaled, adv_raw

    raise ValueError(f"Unknown attack: {attack_name!r}")


def _load_tree_model(
    models_dir: Path,
    target: str,
    model_name: str,
) -> Any:
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
    attack_names: Sequence[str],
) -> None:
    """Generate attacks, evaluate MLP and transfer models, and write results."""
    unknown_targets = sorted(set(targets) - set(TARGET_CONFIG))
    if unknown_targets:
        raise ValueError(f"Unknown targets: {unknown_targets}")
    unknown_models = sorted(set(model_names) - set(MODEL_NAMES))
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}")
    unknown_attacks = sorted(set(attack_names) - set(ATTACK_NAMES))
    if unknown_attacks:
        raise ValueError(f"Unknown attacks: {unknown_attacks}")
    if "mlp" not in model_names:
        raise ValueError("The MLP must be among the evaluated models")

    set_seed(int(config["seed"]), bool(config.get("deterministic", True)))
    splits = load_splits(config)
    train_frame = splits["train"]
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
    test_scaled = transform_features(test_frame, preprocessor, scale=True)
    test_tree = transform_features(test_frame, preprocessor, scale=False)
    scaled_bounds = scaled_feature_bounds(train_scaled)
    raw_bounds = raw_feature_bounds(train_frame, preprocessor)
    batch_size = int(config.get("evaluation", {}).get("batch_size", 4096))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Attack device: %s", device)

    metrics_rows: list[dict[str, Any]] = []
    recall_frames: list[pd.DataFrame] = []

    for target in targets:
        labels, class_names = target_arrays(test_frame, target)
        num_classes = len(class_names)
        benign_index_value = benign_class_index(target, class_names)
        mlp_path = models_dir / f"{target}_mlp.pt"
        if not mlp_path.exists():
            raise FileNotFoundError(
                f"Missing model {mlp_path}. Run `python -m src.train --config "
                "configs/default.yaml` first."
            )
        mlp = load_mlp(mlp_path, device)
        tree_models = {
            name: _load_tree_model(models_dir, target, name)
            for name in model_names
            if name != "mlp"
        }

        clean_predictions = {
            "mlp": predict_mlp(mlp, test_scaled, batch_size, device)
        }
        for name, model in tree_models.items():
            clean_predictions[name] = model.predict(test_tree).astype(np.int64)

        clean_metrics = {
            name: compute_metrics(
                labels,
                predictions,
                labels=range(num_classes),
            )
            for name, predictions in clean_predictions.items()
        }
        clean_recall = {
            name: per_class_recall_frame(labels, predictions, class_names)
            for name, predictions in clean_predictions.items()
        }

        for attack_name in attack_names:
            epsilons = config["attacks"][attack_name]["epsilons"]
            for epsilon in epsilons:
                LOGGER.info(
                    "Target %s: %s epsilon=%s",
                    target,
                    attack_name,
                    epsilon,
                )
                adversarial_scaled, adversarial_raw = _generate_attack(
                    attack_name,
                    mlp,
                    test_scaled,
                    test_tree,
                    labels,
                    float(epsilon),
                    scaled_bounds,
                    raw_bounds,
                    preprocessor,
                    config,
                    device,
                )

                adversarial_predictions = {
                    "mlp": predict_mlp(
                        mlp,
                        adversarial_scaled,
                        batch_size,
                        device,
                    )
                }
                for name, model in tree_models.items():
                    adversarial_predictions[name] = model.predict(
                        adversarial_raw
                    ).astype(np.int64)

                for name in model_names:
                    metrics = compute_metrics(
                        labels,
                        adversarial_predictions[name],
                        labels=range(num_classes),
                    )
                    clean = clean_metrics[name]
                    metrics_rows.append(
                        {
                            "target": target,
                            "model": name,
                            "attack": attack_name,
                            "epsilon": float(epsilon),
                            "n_samples": int(len(labels)),
                            "clean_accuracy": clean["accuracy"],
                            "adversarial_accuracy": metrics["accuracy"],
                            "accuracy_drop": clean["accuracy"]
                            - metrics["accuracy"],
                            "clean_macro_f1": clean["macro_f1"],
                            "adversarial_macro_f1": metrics["macro_f1"],
                            "macro_f1_drop": clean["macro_f1"]
                            - metrics["macro_f1"],
                            "clean_macro_recall": clean["macro_recall"],
                            "adversarial_macro_recall": metrics["macro_recall"],
                            "macro_recall_drop": clean["macro_recall"]
                            - metrics["macro_recall"],
                            "attack_success_rate": attack_success_rate(
                                clean_predictions[name],
                                adversarial_predictions[name],
                                labels,
                            ),
                            "evasion_rate": evasion_rate(
                                labels,
                                clean_predictions[name],
                                adversarial_predictions[name],
                                benign_index_value,
                            ),
                            "evasion_population": evasion_population(
                                labels,
                                clean_predictions[name],
                                benign_index_value,
                            ),
                        }
                    )

                    adversarial_recall = per_class_recall_frame(
                        labels,
                        adversarial_predictions[name],
                        class_names,
                    )
                    recall_frame = clean_recall[name].rename(
                        columns={"recall": "clean_recall"}
                    )
                    recall_frame["adversarial_recall"] = adversarial_recall[
                        "recall"
                    ].to_numpy()
                    recall_frame["recall_drop"] = (
                        recall_frame["clean_recall"]
                        - recall_frame["adversarial_recall"]
                    )
                    recall_frame.insert(0, "epsilon", float(epsilon))
                    recall_frame.insert(0, "attack", attack_name)
                    recall_frame.insert(0, "model", name)
                    recall_frame.insert(0, "target", target)
                    recall_frames.append(recall_frame)

    pd.DataFrame(metrics_rows).to_csv(
        results_dir / "attack_metrics.csv",
        index=False,
        float_format="%.6f",
    )
    pd.concat(recall_frames, ignore_index=True).to_csv(
        results_dir / "attack_per_class_recall.csv",
        index=False,
        float_format="%.6f",
    )

    manifest = {
        "seed": int(config["seed"]),
        "features": features,
        "targets": list(targets),
        "models": list(model_names),
        "attacks": list(attack_names),
        "epsilons": {
            name: [
                float(epsilon)
                for epsilon in config["attacks"][name]["epsilons"]
            ]
            for name in attack_names
        },
        "pgd_steps": int(config["attacks"]["pgd"]["steps"]),
        "pgd_step_size_factor": float(
            config["attacks"]["pgd"].get("step_size_factor", 0.25)
        ),
        "pgd_random_start": bool(
            config["attacks"]["pgd"].get("random_start", True)
        ),
        "random_noise_seed": int(
            config["attacks"]["random_noise"].get("seed", config["seed"])
        ),
        "bounds": "training_min_max",
        "device": str(device),
        "test_rows": int(len(test_frame)),
    }
    with (results_dir / "attack_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2)

    LOGGER.info("Wrote attack metrics to %s", results_dir)


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
        help="Label granularities to attack (default: evaluation.targets).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=None,
        help="Models to evaluate, including the MLP (default: evaluation.models).",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        choices=ATTACK_NAMES,
        default=None,
        help="Attack families to run (default: every implemented attack).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()
    config = load_config(args.config)
    evaluation_cfg = config.get("evaluation", {})
    targets = args.targets or evaluation_cfg.get("targets", list(TARGET_CONFIG))
    model_names = args.models or evaluation_cfg.get("models", list(MODEL_NAMES))
    attack_names = args.attacks or list(ATTACK_NAMES)
    run(config, targets, model_names, attack_names)


if __name__ == "__main__":
    main()
