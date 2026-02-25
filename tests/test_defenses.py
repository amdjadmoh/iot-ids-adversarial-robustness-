from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from src.defenses import FeatureSqueezer, train_mlp_adversarial
from src.train import predict_mlp


def test_feature_squeezer_quantizes_to_configured_levels() -> None:
    values = np.array([[0.0, 0.5], [0.1, 0.9]], dtype=np.float32)
    squeezer = FeatureSqueezer.fit(values, bit_depth=2)

    squeezed = squeezer.transform(values)

    assert squeezed.shape == values.shape
    assert np.unique(squeezed[:, 0]).size <= 4
    assert squeezer.levels == 4


def test_feature_squeezer_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    train = rng.exponential(size=(200, 4)).astype(np.float32)
    test = rng.exponential(size=(50, 4)).astype(np.float32)
    squeezer = FeatureSqueezer.fit(train, bit_depth=8)

    first = squeezer.transform(test)
    second = squeezer.transform(test)

    assert np.array_equal(first, second)


def test_feature_squeezer_is_fitted_on_train_only() -> None:
    train = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    test = np.array([[1000.0]], dtype=np.float32)

    squeezer = FeatureSqueezer.fit(train, bit_depth=8)

    assert squeezer.edges[0, 0] == 0.0
    assert squeezer.edges[-1, 0] == 2.0
    # Test values outside the training range are clamped to the train edges,
    # which proves the fitted object never consulted the test split.
    assert squeezer.transform(test)[0, 0] == np.float32(2.0)


def test_feature_squeezer_preserves_zeros_and_discrete_values_exactly() -> None:
    train = np.concatenate(
        [np.zeros(900, dtype=np.float32), np.ones(100, dtype=np.float32)]
    ).reshape(-1, 1)
    squeezer = FeatureSqueezer.fit(train, bit_depth=8)

    probed = np.array([[0.0], [1.0]], dtype=np.float32)

    assert squeezer.transform(probed).ravel().tolist() == [0.0, 1.0]


def test_feature_squeezer_eight_bit_error_is_small_on_train() -> None:
    rng = np.random.default_rng(3)
    train = rng.lognormal(size=(500, 5)).astype(np.float32)
    squeezer = FeatureSqueezer.fit(train, bit_depth=8)

    error = squeezer.reconstruction_error(train)

    # Equal-mass bins keep the reconstruction error within one bin width, so
    # the error must stay small relative to each feature's observed spread.
    spread = np.quantile(train, 0.95, axis=0) - np.quantile(train, 0.05, axis=0)
    assert np.all(np.median(error, axis=0) < 0.1 * spread)

    # 8 bits must still resolve the training distribution, not collapse it.
    for column in range(train.shape[1]):
        assert np.unique(squeezer.transform(train)[:, column]).size >= 50


def test_feature_squeezer_handles_negative_values() -> None:
    train = np.array([[-5.0], [0.0], [5.0]], dtype=np.float32)
    squeezer = FeatureSqueezer.fit(train, bit_depth=8)

    squeezed = squeezer.transform(train)

    assert squeezed[0, 0] < 0.0 < squeezed[2, 0]
    assert np.isclose(squeezed[0, 0], -5.0, rtol=0.02)
    assert np.isclose(squeezed[2, 0], 5.0, rtol=0.02)
    assert abs(float(squeezed[1, 0])) < 0.05


def test_feature_squeezer_rejects_invalid_bit_depth() -> None:
    with pytest.raises(ValueError, match="bit_depth"):
        FeatureSqueezer.fit(np.zeros((2, 2)), bit_depth=0)


def test_feature_squeezer_rejects_mismatched_width() -> None:
    squeezer = FeatureSqueezer.fit(np.zeros((2, 2)), bit_depth=4)

    with pytest.raises(ValueError, match="matching the bounds"):
        squeezer.transform(np.zeros((2, 3)))


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
