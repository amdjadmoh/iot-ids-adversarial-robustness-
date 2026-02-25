from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.attacks import (
    attack_success_rate,
    fgsm_attack,
    invert_and_clip,
    pgd_attack,
    random_noise_attack,
    raw_feature_bounds,
    scaled_feature_bounds,
)
from src.train import Preprocessor


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(3, 4)


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    features = rng.uniform(-0.5, 0.5, size=(10, 3)).astype(np.float32)
    labels = rng.integers(0, 4, size=10).astype(np.int64)
    return features, labels


def test_fgsm_respects_epsilon_and_bounds() -> None:
    features, labels = _data()
    lower = np.full(3, -0.5, dtype=np.float32)
    upper = np.full(3, 0.5, dtype=np.float32)

    adversarial = fgsm_attack(
        _model(),
        features,
        labels,
        epsilon=0.1,
        clip_min=lower,
        clip_max=upper,
        batch_size=4,
        device=torch.device("cpu"),
    )

    assert adversarial.shape == features.shape
    assert np.all(np.abs(adversarial - features) <= 0.1 + 1e-6)
    assert np.all(adversarial >= lower - 1e-6)
    assert np.all(adversarial <= upper + 1e-6)


def test_fgsm_rejects_negative_epsilon() -> None:
    features, labels = _data()

    with pytest.raises(ValueError, match="epsilon"):
        fgsm_attack(
            _model(),
            features,
            labels,
            epsilon=-0.1,
            clip_min=-1.0,
            clip_max=1.0,
            device=torch.device("cpu"),
        )


def test_pgd_respects_epsilon_and_bounds() -> None:
    features, labels = _data()
    lower = np.full(3, -0.5, dtype=np.float32)
    upper = np.full(3, 0.5, dtype=np.float32)

    adversarial = pgd_attack(
        _model(),
        features,
        labels,
        epsilon=0.1,
        steps=4,
        clip_min=lower,
        clip_max=upper,
        step_size_factor=0.5,
        random_start=True,
        batch_size=4,
        device=torch.device("cpu"),
    )

    assert adversarial.shape == features.shape
    assert np.all(np.abs(adversarial - features) <= 0.1 + 1e-6)
    assert np.all(adversarial >= lower - 1e-6)
    assert np.all(adversarial <= upper + 1e-6)


def test_pgd_rejects_zero_epsilon() -> None:
    features, labels = _data()

    with pytest.raises(ValueError, match="epsilon"):
        pgd_attack(
            _model(),
            features,
            labels,
            epsilon=0.0,
            steps=4,
            clip_min=-1.0,
            clip_max=1.0,
            device=torch.device("cpu"),
        )


def test_random_noise_respects_epsilon_and_bounds() -> None:
    features, _ = _data()
    lower = np.full(3, -0.5, dtype=np.float32)
    upper = np.full(3, 0.5, dtype=np.float32)

    adversarial = random_noise_attack(
        features,
        epsilon=0.1,
        clip_min=lower,
        clip_max=upper,
        seed=7,
        batch_size=4,
    )

    assert adversarial.shape == features.shape
    assert np.all(np.abs(adversarial - features) <= 0.1 + 1e-6)
    assert np.all(adversarial >= lower - 1e-6)
    assert np.all(adversarial <= upper + 1e-6)


def test_random_noise_is_reproducible_with_the_same_seed() -> None:
    features, _ = _data()

    first = random_noise_attack(
        features, epsilon=0.1, clip_min=-1.0, clip_max=1.0, seed=11
    )
    second = random_noise_attack(
        features, epsilon=0.1, clip_min=-1.0, clip_max=1.0, seed=11
    )
    other = random_noise_attack(
        features, epsilon=0.1, clip_min=-1.0, clip_max=1.0, seed=12
    )

    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_random_noise_uses_the_full_epsilon_budget() -> None:
    features = np.zeros((200, 3), dtype=np.float32)

    adversarial = random_noise_attack(
        features, epsilon=0.05, clip_min=-1.0, clip_max=1.0, seed=3
    )

    magnitudes = np.abs(adversarial).ravel()
    assert np.allclose(magnitudes, 0.05)
    assert set(np.sign(adversarial).ravel().tolist()) == {-1.0, 1.0}


def test_attack_success_rate_uses_initially_correct_rows() -> None:
    labels = np.array([0, 1, 2, 0])
    clean = np.array([0, 1, 2, 0])
    adversarial = np.array([0, 2, 2, 1])

    assert attack_success_rate(clean, adversarial, labels) == 0.5


def test_attack_success_rate_ignores_clean_errors() -> None:
    labels = np.array([0, 1, 2])
    clean = np.array([0, 0, 2])
    adversarial = np.array([1, 0, 0])

    assert attack_success_rate(clean, adversarial, labels) == 1.0


def test_scaled_feature_bounds_are_per_feature() -> None:
    features = np.array([[0.0, 3.0], [2.0, 1.0]], dtype=np.float32)

    lower, upper = scaled_feature_bounds(features)

    assert lower.tolist() == [0.0, 1.0]
    assert upper.tolist() == [2.0, 3.0]


def test_raw_feature_bounds_impute_with_training_medians() -> None:
    train = pd.DataFrame({"a": [1.0, np.inf], "b": [np.nan, 4.0]})
    preprocessor = Preprocessor(
        features=["a", "b"],
        medians=pd.Series({"a": 1.0, "b": 4.0}),
        scaler=None,
        clip_value=None,
    )

    lower, upper = raw_feature_bounds(train, preprocessor)

    assert lower.tolist() == [1.0, 4.0]
    assert upper.tolist() == [1.0, 4.0]


def test_invert_and_clip_returns_raw_bounded_values() -> None:
    scaler = StandardScaler().fit(
        np.array([[0.0, 10.0], [2.0, 20.0], [1.0, 30.0]])
    )
    preprocessor = Preprocessor(
        features=["a", "b"],
        medians=pd.Series({"a": 1.0, "b": 20.0}),
        scaler=scaler,
        clip_value=None,
    )
    scaled = np.array([[100.0, -100.0]], dtype=np.float32)

    raw = invert_and_clip(
        scaled,
        preprocessor,
        lower=np.array([0.0, 10.0]),
        upper=np.array([2.0, 30.0]),
    )

    assert raw[0, 0] == 2.0
    assert raw[0, 1] == 10.0
