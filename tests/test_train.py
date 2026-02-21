from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from src.data import FAMILY_NAMES
from src.features import FEATURE_NAMES
from src.train import (
    fit_preprocessor,
    predict_mlp,
    predict_proba_mlp,
    resolve_features,
    target_arrays,
    train_mlp,
    transform_features,
)


def _preprocessing_config() -> dict[str, Any]:
    return {
        "preprocessing": {
            "scale_for_mlp": True,
            "scaler": "standard",
            "clip_after_scaling": 8.0,
        }
    }


def test_fit_preprocessor_uses_training_statistics_only() -> None:
    train = pd.DataFrame({"a": [1.0, np.inf], "b": [np.nan, 4.0]})
    val = pd.DataFrame({"a": [100.0], "b": [np.nan]})

    preprocessor = fit_preprocessor(train, ["a", "b"], _preprocessing_config())

    assert preprocessor.medians["a"] == 1.0
    assert preprocessor.medians["b"] == 4.0
    transformed_train = transform_features(train, preprocessor, scale=True)
    transformed_val = transform_features(val, preprocessor, scale=True)
    assert np.isfinite(transformed_train).all()
    assert np.isfinite(transformed_val).all()
    assert np.allclose(transformed_train.mean(axis=0), 0.0, atol=1e-6)
    assert transformed_val[0, 0] == 8.0


def test_tree_transform_does_not_scale() -> None:
    train = pd.DataFrame({"a": [1.0, 3.0]})
    preprocessor = fit_preprocessor(train, ["a"], _preprocessing_config())

    values = transform_features(train, preprocessor, scale=False)

    assert np.allclose(values[:, 0], [1.0, 3.0])


def test_resolve_features_preserves_dataset_order() -> None:
    frame = pd.DataFrame({name: [0.0] for name in FEATURE_NAMES})
    frame["Label"] = "BenignTraffic"
    frame["split"] = "train"

    features = resolve_features(frame)

    assert features == list(FEATURE_NAMES)


def test_resolve_features_rejects_unknown_columns() -> None:
    frame = pd.DataFrame({name: [0.0] for name in FEATURE_NAMES})
    frame["mystery"] = 1.0

    with pytest.raises(ValueError, match="Unexpected non-feature"):
        resolve_features(frame)


def test_resolve_features_ignores_lowercase_label_column() -> None:
    frame = pd.DataFrame({name: [0.0] for name in FEATURE_NAMES})
    frame["label"] = "BenignTraffic"
    frame["split"] = "train"

    assert resolve_features(frame) == list(FEATURE_NAMES)


def test_target_arrays_maps_family_names() -> None:
    frame = pd.DataFrame(
        {
            "label_id": [0, 1, 2],
            "label_family": [FAMILY_NAMES[0], FAMILY_NAMES[1], FAMILY_NAMES[2]],
            "label_binary": [0, 1, 1],
        }
    )

    multiclass, multiclass_names = target_arrays(frame, "multiclass")
    family, family_names = target_arrays(frame, "family")
    binary, binary_names = target_arrays(frame, "binary")

    assert multiclass.tolist() == [0, 1, 2]
    assert family.tolist() == [0, 1, 2]
    assert binary.tolist() == [0, 1, 1]
    assert family_names == FAMILY_NAMES
    assert binary_names == ("Benign", "Attack")
    assert len(multiclass_names) == 34


def test_mlp_training_smoke() -> None:
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(90, 4)).astype(np.float32)
    y_train = np.tile(np.arange(3), 30).astype(np.int64)
    X_val = rng.normal(size=(30, 4)).astype(np.float32)
    y_val = np.tile(np.arange(3), 10).astype(np.int64)
    config: dict[str, Any] = {
        "seed": 3,
        "evaluation": {"batch_size": 16},
        "models": {
            "mlp": {
                "hidden_dims": [8],
                "dropout": 0.0,
                "batch_size": 16,
                "epochs": 2,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
            }
        },
    }
    device = torch.device("cpu")

    model, history = train_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        num_classes=3,
        config=config,
        device=device,
    )
    predictions = predict_mlp(model, X_val, 16, device)
    probabilities = predict_proba_mlp(model, X_val, 16, device)

    assert predictions.shape == (30,)
    assert probabilities.shape == (30, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert list(history.columns) == [
        "epoch",
        "train_loss",
        "val_accuracy",
        "val_macro_f1",
    ]
    assert len(history) == 2
