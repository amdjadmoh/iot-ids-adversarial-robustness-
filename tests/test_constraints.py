from __future__ import annotations

import numpy as np
import pytest

from src.constraints import (
    FROZEN_DERIVED,
    check_sample_constraints,
    project_feature_constraints,
)
from src.features import FEATURE_NAMES, FIXED_FEATURES


def _synthetic_clean_sample() -> tuple[np.ndarray, dict[str, int]]:
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    row = np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    # Set controllable features to realistic positive values
    row[idx["flow_duration"]] = 1.0
    row[idx["Rate"]] = 50.0
    row[idx["Srate"]] = 50.0
    row[idx["Min"]] = 64.0
    row[idx["Max"]] = 512.0
    row[idx["AVG"]] = 256.0
    row[idx["Tot size"]] = 256.0
    row[idx["Std"]] = 10.0
    row[idx["IAT"]] = 0.02
    row[idx["Number"]] = 50.0

    # Fixed features
    row[idx["Header_Length"]] = 54.0
    row[idx["Protocol Type"]] = 6.0
    row[idx["TCP"]] = 1.0
    row[idx["syn_flag_number"]] = 1.0

    # Derived
    row[idx["Tot sum"]] = 50.0 * 256.0
    row[idx["syn_count"]] = 1.0
    row[idx["Magnitue"]] = 16.0

    return row.reshape(1, -1), idx


def test_project_freezes_fixed_features() -> None:
    orig, idx = _synthetic_clean_sample()
    perturbed = orig.copy()
    # Try to perturb fixed features
    perturbed[0, idx["Header_Length"]] = 999.0
    perturbed[0, idx["Protocol Type"]] = 17.0
    perturbed[0, idx["TCP"]] = 0.0

    projected = project_feature_constraints(perturbed, orig)

    assert projected[0, idx["Header_Length"]] == orig[0, idx["Header_Length"]]
    assert projected[0, idx["Protocol Type"]] == orig[0, idx["Protocol Type"]]
    assert projected[0, idx["TCP"]] == orig[0, idx["TCP"]]


def test_project_freezes_frozen_derived_features() -> None:
    orig, idx = _synthetic_clean_sample()
    perturbed = orig.copy()
    for name in FROZEN_DERIVED:
        perturbed[0, idx[name]] = 12345.0

    projected = project_feature_constraints(perturbed, orig)

    for name in FROZEN_DERIVED:
        assert projected[0, idx[name]] == orig[0, idx[name]]


def test_sizes_can_only_increase() -> None:
    orig, idx = _synthetic_clean_sample()
    perturbed = orig.copy()
    # Try to shrink sizes (illegal)
    perturbed[0, idx["Min"]] = 10.0
    perturbed[0, idx["AVG"]] = 50.0
    perturbed[0, idx["Max"]] = 100.0

    projected = project_feature_constraints(perturbed, orig)

    assert projected[0, idx["Min"]] >= orig[0, idx["Min"]]
    assert projected[0, idx["AVG"]] >= orig[0, idx["AVG"]]
    assert projected[0, idx["Max"]] >= orig[0, idx["Max"]]


def test_rates_can_only_decrease() -> None:
    orig, idx = _synthetic_clean_sample()
    perturbed = orig.copy()
    # Try to increase sending rate (illegal)
    perturbed[0, idx["Rate"]] = 5000.0
    perturbed[0, idx["Srate"]] = 5000.0

    projected = project_feature_constraints(perturbed, orig)

    assert projected[0, idx["Rate"]] <= orig[0, idx["Rate"]]
    assert projected[0, idx["Srate"]] <= orig[0, idx["Srate"]]


def test_enforces_min_le_avg_le_max() -> None:
    orig, idx = _synthetic_clean_sample()
    perturbed = orig.copy()
    # Perturb with inverted ordering
    perturbed[0, idx["Min"]] = 400.0
    perturbed[0, idx["AVG"]] = 100.0
    perturbed[0, idx["Max"]] = 200.0

    projected = project_feature_constraints(perturbed, orig)

    min_v = projected[0, idx["Min"]]
    avg_v = projected[0, idx["AVG"]]
    max_v = projected[0, idx["Max"]]

    assert min_v <= avg_v + 1e-6
    assert avg_v <= max_v + 1e-6


def test_tot_sum_is_recomputed() -> None:
    orig, idx = _synthetic_clean_sample()
    perturbed = orig.copy()
    # Increase AVG packet size
    perturbed[0, idx["Max"]] = 1000.0
    perturbed[0, idx["AVG"]] = 800.0

    projected = project_feature_constraints(perturbed, orig)

    expected_tot_sum = projected[0, idx["Number"]] * projected[0, idx["AVG"]]
    assert np.isclose(projected[0, idx["Tot sum"]], expected_tot_sum)


def test_check_sample_constraints_passes_on_projected() -> None:
    orig, _ = _synthetic_clean_sample()
    # Generate arbitrary noisy perturbation
    rng = np.random.default_rng(42)
    noise = rng.normal(scale=50.0, size=orig.shape)
    perturbed = orig + noise

    projected = project_feature_constraints(perturbed, orig)
    checks = check_sample_constraints(projected, orig)

    for check_name, passed in checks.items():
        assert passed, f"Constraint check {check_name} failed on projected sample"
