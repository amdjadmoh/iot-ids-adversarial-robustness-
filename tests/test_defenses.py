from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from src.defenses import squeeze_features, train_mlp_adversarial
from src.train import predict_mlp


def test_squeeze_features_quantizes_to_configured_levels() -> None:
    values = np.array([[0.0, 0.5], [0.1, 0.9]], dtype=np.float32)

    squeezed = squeeze_features(
        values,
        lower=np.zeros(2, dtype=np.float32),
        upper=np.ones(2, dtype=np.float32),
        bit_depth=2,
    )

    allowed = np.array([0.0, 1 / 3, 2 / 3, 1.0])
    for value in squeezed.ravel():
        assert any(np.isclose(value, level, atol=1e-6) for level in allowed)
    assert squeezed.min() >= 0.0
    assert squeezed.max() <= 1.0


def test_squeeze_features_clips_out_of_range_values() -> None:
    values = np.array([[-1.0, 2.0]], dtype=np.float32)

    squeezed = squeeze_features(
        values,
        lower=np.zeros(2, dtype=np.float32),
        upper=np.ones(2, dtype=np.float32),
        bit_depth=2,
    )

    assert squeezed.tolist() == [[0.0, 1.0]]


def test_squeeze_features_leaves_constant_features_unchanged() -> None:
    values = np.array([[7.0]], dtype=np.float32)

    squeezed = squeeze_features(
        values,
        lower=np.array([7.0], dtype=np.float32),
        upper=np.array([7.0], dtype=np.float32),
        bit_depth=4,
    )

    assert squeezed[0, 0] == 7.0


@pytest.mark.parametrize("bit_depth", [0, 33])
def test_squeeze_features_rejects_invalid_bit_depth(bit_depth: int) -> None:
    with pytest.raises(ValueError, match="bit_depth"):
        squeeze_features(
            np.zeros((1, 2), dtype=np.float32),
            lower=np.zeros(2, dtype=np.float32),
            upper=np.ones(2, dtype=np.float32),
            bit_depth=bit_depth,
        )


def test_adversarial_training_smoke() -> None:
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(90, 4)).astype(np.float32)
    y_train = np.tile(np.arange(3), 30).astype(np.int64)
    X_val = rng.normal(size=(30, 4)).astype(np.float32)
    y_val = np.tile(np.arange(3), 10).astype(np.int64)
    bounds = (X_train.min(axis=0), X_train.max(axis=0))
    config: dict[str, Any] = {
        "seed": 3,
        "evaluation": {"batch_size": 16},
        "defenses": {
            "adversarial_training": {
                "epsilons": [0.01],
                "epochs": 2,
                "clean_fraction": 0.5,
            }
        },
        "models": {
            "mlp": {
                "hidden_dims": [8],
                "dropout": 0.0,
                "batch_size": 16,
                "epochs": 1,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
            }
        },
    }
    device = torch.device("cpu")

    model, history = train_mlp_adversarial(
        X_train,
        y_train,
        X_val,
        y_val,
        num_classes=3,
        epsilon=0.01,
        scaled_bounds=bounds,
        config=config,
        device=device,
    )
    predictions = predict_mlp(model, X_val, 16, device)

    assert predictions.shape == (30,)
    assert list(history.columns) == [
        "epoch",
        "train_loss",
        "val_clean_macro_f1",
        "val_robust_macro_f1",
    ]
    assert len(history) == 2
