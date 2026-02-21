"""Training and evaluation entry point for the detector baselines."""

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
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import configure_logging, load_config, set_seed
from src.data import FAMILY_NAMES, FAMILY_TO_INDEX, MULTICLASS_LABELS
from src.evaluate import compute_metrics, confusion_matrix_frame, per_class_recall_frame
from src.features import FEATURE_NAMES
from src.models import (
    build_mlp,
    build_random_forest,
    build_xgboost,
    save_mlp,
)

LOGGER = logging.getLogger(__name__)

NON_FEATURE_COLUMNS = {
    "Label",
    "label",
    "source_file",
    "split",
    "label_multiclass",
    "label_family",
    "label_binary",
    "label_id",
}
TARGET_CONFIG = {
    "multiclass": ("label_id", MULTICLASS_LABELS),
    "family": ("label_family", FAMILY_NAMES),
    "binary": ("label_binary", ("Benign", "Attack")),
}
MODEL_NAMES = ("random_forest", "xgboost", "mlp")


@dataclass
class Preprocessor:
    """Training-fitted feature imputation and scaling."""

    features: list[str]
    medians: pd.Series
    scaler: StandardScaler | None
    clip_value: float | None


def _replace_non_finite(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan)


def resolve_features(frame: pd.DataFrame) -> list[str]:
    """Return the known feature columns present in a split, in dataset order."""
    features = [name for name in FEATURE_NAMES if name in frame.columns]
    missing = [name for name in FEATURE_NAMES if name not in frame.columns]
    if missing:
        LOGGER.warning("Features absent from the data: %s", missing)
    unexpected = [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS and column not in FEATURE_NAMES
    ]
    if unexpected:
        raise ValueError(f"Unexpected non-feature columns: {unexpected}")
    if not features:
        raise ValueError("No known feature columns found in the training split")
    return features


def fit_preprocessor(
    train_frame: pd.DataFrame,
    features: Sequence[str],
    config: dict[str, Any],
) -> Preprocessor:
    """Fit median imputation and scaling on the training split only."""
    feature_frame = _replace_non_finite(train_frame[list(features)])
    medians = feature_frame.median().fillna(0.0)
    filled = feature_frame.fillna(medians)

    preprocessing_cfg = config.get("preprocessing", {})
    scaler: StandardScaler | None = None
    if bool(preprocessing_cfg.get("scale_for_mlp", True)):
        scaler_name = str(preprocessing_cfg.get("scaler", "standard")).lower()
        if scaler_name != "standard":
            raise ValueError(f"Unsupported scaler: {scaler_name!r}")
        scaler = StandardScaler().fit(filled.to_numpy(dtype=np.float64))

    clip_value = preprocessing_cfg.get("clip_after_scaling")
    return Preprocessor(
        features=list(features),
        medians=medians,
        scaler=scaler,
        clip_value=float(clip_value) if clip_value is not None else None,
    )


def transform_features(
    frame: pd.DataFrame,
    preprocessor: Preprocessor,
    scale: bool,
) -> np.ndarray:
    """Apply the fitted imputation, and optionally the MLP scaling, to a split."""
    feature_frame = _replace_non_finite(frame[preprocessor.features])
    filled = feature_frame.fillna(preprocessor.medians)
    values = filled.to_numpy(dtype=np.float64)

    if scale:
        if preprocessor.scaler is None:
            raise ValueError("Scaler was not fitted")
        values = preprocessor.scaler.transform(values)
        if preprocessor.clip_value is not None:
            values = np.clip(
                values,
                -preprocessor.clip_value,
                preprocessor.clip_value,
            )

    if np.isnan(values).any():
        raise ValueError("Preprocessed features contain NaN values")
    return values.astype(np.float32)


def target_arrays(
    frame: pd.DataFrame,
    target: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return integer labels and human-readable class names for a target."""
    if target == "multiclass":
        return (
            frame["label_id"].to_numpy(dtype=np.int64, copy=True),
            MULTICLASS_LABELS,
        )
    if target == "family":
        mapped = frame["label_family"].map(FAMILY_TO_INDEX)
        if mapped.isna().any():
            raise ValueError("Unknown family label while building the target")
        return mapped.to_numpy(dtype=np.int64, copy=True), FAMILY_NAMES
    if target == "binary":
        return (
            frame["label_binary"].to_numpy(dtype=np.int64, copy=True),
            ("Benign", "Attack"),
        )
    raise ValueError(f"Unknown target: {target!r}")


def _check_train_classes(y_train: np.ndarray, num_classes: int, target: str) -> None:
    present = set(np.unique(y_train).tolist())
    missing = sorted(set(range(num_classes)) - present)
    if missing:
        raise ValueError(
            f"Target {target!r} is missing class ids {missing} from the training split"
        )


def load_splits(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Load the pickled splits produced by ``python -m src.data``."""
    processed_dir = Path(config["data"].get("processed_dir", "data/processed"))
    paths = {
        name: processed_dir / f"{name}.pkl"
        for name in ("train", "val", "test")
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing split files: "
            + ", ".join(missing)
            + ". Run `python -m src.data --config configs/default.yaml` first."
        )
    return {name: pd.read_pickle(path) for name, path in paths.items()}


def train_sklearn_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    config: dict[str, Any],
) -> Any:
    """Fit a Random Forest or XGBoost baseline."""
    if model_name == "random_forest":
        model = build_random_forest(config)
    elif model_name == "xgboost":
        model = build_xgboost(num_classes, config)
    else:
        raise ValueError(f"Unknown model: {model_name!r}")
    model.fit(X_train, y_train)
    return model


def predict_mlp(
    model: nn.Module,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return predicted class ids from the MLP."""
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(
                features[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            predictions.append(torch.argmax(model(batch), dim=1).cpu().numpy())
    if not predictions:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(predictions).astype(np.int64)


def predict_proba_mlp(
    model: nn.Module,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return class probabilities from the MLP."""
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(
                features[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            probabilities.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    if not probabilities:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(probabilities, axis=0)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame]:
    """Train the MLP and restore the best validation-macro-F1 weights."""
    params = config["models"]["mlp"]
    epochs = int(params["epochs"])
    if epochs <= 0:
        raise ValueError("models.mlp.epochs must be positive")

    model = build_mlp(X_train.shape[1], num_classes, config).to(device)
    dataset = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(params["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(int(config["seed"])),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )

    best_macro_f1 = -1.0
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    history: list[dict[str, float]] = []
    batch_size = int(config.get("evaluation", {}).get("batch_size", 4096))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(features)
            total_rows += len(features)

        val_predictions = predict_mlp(model, X_val, batch_size, device)
        val_metrics = compute_metrics(
            y_val,
            val_predictions,
            labels=range(num_classes),
        )
        train_loss = total_loss / max(total_rows, 1)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            LOGGER.info(
                "MLP epoch %d/%d train_loss=%.4f val_macro_f1=%.4f",
                epoch,
                epochs,
                train_loss,
                val_metrics["macro_f1"],
            )

    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)


def run(
    config: dict[str, Any],
    targets: Sequence[str],
    model_names: Sequence[str],
) -> None:
    """Train and evaluate the requested baselines, writing all metrics to results/."""
    unknown_targets = sorted(set(targets) - set(TARGET_CONFIG))
    if unknown_targets:
        raise ValueError(f"Unknown targets: {unknown_targets}")
    unknown_models = sorted(set(model_names) - set(MODEL_NAMES))
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}")

    set_seed(
        int(config["seed"]),
        bool(config.get("deterministic", True)),
    )
    splits = load_splits(config)
    train_frame = splits["train"]
    features = resolve_features(train_frame)
    LOGGER.info(
        "Training on %d rows, validating on %d, testing on %d; %d features",
        len(train_frame),
        len(splits["val"]),
        len(splits["test"]),
        len(features),
    )

    preprocessor = fit_preprocessor(train_frame, features, config)
    models_dir = Path(config["data"].get("models_dir", "data/models"))
    results_dir = Path(config["paths"].get("results", "results"))
    confusion_dir = results_dir / "confusion_matrices"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    if bool(config.get("evaluation", {}).get("save_confusion_matrix", True)):
        confusion_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")

    tree_features = {
        name: transform_features(frame, preprocessor, scale=False)
        for name, frame in splits.items()
    }
    mlp_features = {
        name: transform_features(frame, preprocessor, scale=True)
        for name, frame in splits.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Training device: %s", device)

    metrics_rows: list[dict[str, Any]] = []
    recall_frames: list[pd.DataFrame] = []

    for target in targets:
        labels_by_split = {
            name: target_arrays(frame, target)[0]
            for name, frame in splits.items()
        }
        class_names = target_arrays(train_frame, target)[1]
        num_classes = len(class_names)
        _check_train_classes(
            labels_by_split["train"],
            num_classes,
            target,
        )

        for model_name in model_names:
            LOGGER.info("Training %s for target %s", model_name, target)
            if model_name == "mlp":
                model, history = train_mlp(
                    mlp_features["train"],
                    labels_by_split["train"],
                    mlp_features["val"],
                    labels_by_split["val"],
                    num_classes,
                    config,
                    device,
                )
                save_mlp(
                    model,
                    input_dim=len(features),
                    num_classes=num_classes,
                    path=models_dir / f"{target}_mlp.pt",
                    config=config,
                )
                history.to_csv(
                    results_dir / f"mlp_training_{target}.csv",
                    index=False,
                    float_format="%.6f",
                )
            else:
                model = train_sklearn_model(
                    model_name,
                    tree_features["train"],
                    labels_by_split["train"],
                    num_classes,
                    config,
                )
                joblib.dump(
                    model,
                    models_dir / f"{target}_{model_name}.joblib",
                )

            for split_name in ("val", "test"):
                feature_matrix = (
                    mlp_features[split_name]
                    if model_name == "mlp"
                    else tree_features[split_name]
                )
                if model_name == "mlp":
                    predictions = predict_mlp(
                        model,
                        feature_matrix,
                        int(config.get("evaluation", {}).get("batch_size", 4096)),
                        device,
                    )
                else:
                    predictions = model.predict(feature_matrix).astype(np.int64)

                y_true = labels_by_split[split_name]
                metrics = compute_metrics(
                    y_true,
                    predictions,
                    labels=range(num_classes),
                )
                metrics_rows.append(
                    {
                        "target": target,
                        "model": model_name,
                        "split": split_name,
                        "n_samples": int(len(y_true)),
                        **metrics,
                    }
                )

                recall_frame = per_class_recall_frame(
                    y_true,
                    predictions,
                    class_names,
                )
                recall_frame.insert(0, "split", split_name)
                recall_frame.insert(0, "model", model_name)
                recall_frame.insert(0, "target", target)
                recall_frames.append(recall_frame)

                if (
                    split_name == "test"
                    and bool(
                        config.get("evaluation", {}).get(
                            "save_confusion_matrix",
                            True,
                        )
                    )
                ):
                    matrix = confusion_matrix_frame(
                        y_true,
                        predictions,
                        class_names,
                    )
                    matrix.to_csv(
                        confusion_dir / f"{target}_{model_name}_test.csv"
                    )

    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(
        results_dir / "baselines_metrics.csv",
        index=False,
        float_format="%.6f",
    )
    pd.concat(recall_frames, ignore_index=True).to_csv(
        results_dir / "baselines_per_class_recall.csv",
        index=False,
        float_format="%.6f",
    )

    manifest = {
        "seed": int(config["seed"]),
        "features": features,
        "targets": list(targets),
        "models": list(model_names),
        "device": str(device),
        "rows": {name: int(len(frame)) for name, frame in splits.items()},
    }
    with (results_dir / "baselines_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    LOGGER.info("Wrote baseline metrics to %s", results_dir)


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
        help="Label granularities to run (default: evaluation.targets).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=None,
        help="Baselines to run (default: evaluation.models).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()
    config = load_config(args.config)
    evaluation_cfg = config.get("evaluation", {})
    targets = args.targets or evaluation_cfg.get("targets", list(TARGET_CONFIG))
    model_names = args.models or evaluation_cfg.get("models", list(MODEL_NAMES))
    run(config, targets, model_names)


if __name__ == "__main__":
    main()
