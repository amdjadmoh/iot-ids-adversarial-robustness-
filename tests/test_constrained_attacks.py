from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.attacks import constrained_pgd_attack
from src.constraints import check_sample_constraints
from src.features import FEATURE_NAMES
from src.train import Preprocessor


def _build_test_setup() -> tuple[nn.Module, np.ndarray, np.ndarray, Preprocessor]:
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    # 46 features
    dim = len(FEATURE_NAMES)
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    # Generate synthetic raw data (10 samples)
    raw_data = rng.uniform(10.0, 100.0, size=(10, dim)).astype(np.float32)

    # Make raw data strictly compliant with initial constraints
    for i in range(10):
        raw_data[i, idx["Min"]] = 50.0
        raw_data[i, idx["AVG"]] = 100.0
        raw_data[i, idx["Max"]] = 200.0
        raw_data[i, idx["Tot size"]] = 100.0
        raw_data[i, idx["Rate"]] = 50.0
        raw_data[i, idx["Srate"]] = 40.0
        raw_data[i, idx["IAT"]] = 0.05
        raw_data[i, idx["flow_duration"]] = 2.0
        raw_data[i, idx["Number"]] = 100.0
        raw_data[i, idx["Tot sum"]] = 100.0 * 100.0

    # Preprocessor
    scaler = StandardScaler().fit(raw_data)
    preprocessor = Preprocessor(
        features=list(FEATURE_NAMES),
        medians=raw_data.mean(axis=0),
        scaler=scaler,
        clip_value=8.0,
    )

    scaled_data = preprocessor.scaler.transform(raw_data).astype(np.float32)

    # Simple 2-class linear model
    model = nn.Linear(dim, 2)

    return model, raw_data, scaled_data, preprocessor


def test_constrained_pgd_satisfies_all_rules() -> None:
    model, raw_data, scaled_data, preprocessor = _build_test_setup()
    labels = np.array([0, 1] * 5, dtype=np.int64)

    raw_lower = raw_data.min(axis=0)
    raw_upper = raw_data.max(axis=0) + 50.0  # allow some headroom
    scaled_lower = scaled_data.min(axis=0)
    scaled_upper = scaled_data.max(axis=0)

    adv_scaled, adv_raw = constrained_pgd_attack(
        model=model,
        scaled_features=scaled_data,
        original_raw=raw_data,
        labels=labels,
        epsilon=0.05,
        steps=5,
        preprocessor=preprocessor,
        raw_lower=raw_lower,
        raw_upper=raw_upper,
        scaled_lower=scaled_lower,
        scaled_upper=scaled_upper,
        step_size_factor=0.25,
        max_feature_delta=0.25,
        batch_size=16,
        device=torch.device("cpu"),
    )

    assert adv_scaled.shape == scaled_data.shape
    assert adv_raw.shape == raw_data.shape

    # Verify every adversarial sample satisfies every constraint rule
    for i in range(len(raw_data)):
        checks = check_sample_constraints(
            adv_raw[i : i + 1],
            raw_data[i : i + 1],
            feature_names=preprocessor.features,
        )
        for check_name, passed in checks.items():
            assert passed, f"Sample {i} failed constraint check: {check_name}"
