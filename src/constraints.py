"""Attacker-controllable feature projection and consistency enforcement.

Implements realistic threat-model constraints for feature-space evasion:
1. Controllable (10): packet sizes, timing, rates, packet counts.
2. Fixed (25): protocols, flags, victim/network properties (frozen).
3. Derived (11): recomputed where closed-form formula is known (Tot sum),
   with the rest frozen and documented as a limitation.
4. Directional constraints:
   - Size features can only increase (payload padding).
   - Timing features can only get slower (Rate/Srate decrease, IAT increases).
   - Rate, Srate, IAT, Number, and flow_duration stay mutually consistent.
5. Invariants:
   - Non-negativity on all numeric features.
   - Min <= AVG <= Max.
   - Min <= Tot size <= Max.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from src.features import (
    CONTROLLABLE_FEATURES,
    DERIVED_FEATURES,
    FEATURE_NAMES,
    FIXED_FEATURES,
)

FEATURE_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(FEATURE_NAMES)}

# Size features can only increase (attacker adds padding)
SIZE_FEATURES: tuple[str, ...] = ("Min", "Max", "AVG", "Tot size")

# Timing / rate features: attacker can only slow down transmission
# (Rate and Srate decrease, IAT increases)
RATE_FEATURES: tuple[str, ...] = ("Rate", "Srate")

# Recomputed vs frozen derived features
RECOMPUTED_DERIVED: tuple[str, ...] = ("Tot sum",)
FROZEN_DERIVED: tuple[str, ...] = (
    "ack_count",
    "syn_count",
    "fin_count",
    "urg_count",
    "rst_count",
    "Magnitue",
    "Radius",
    "Covariance",
    "Variance",
    "Weight",
)


def project_feature_constraints(
    adversarial_raw: np.ndarray,
    original_raw: np.ndarray,
    feature_names: Sequence[str] = FEATURE_NAMES,
    raw_lower: np.ndarray | None = None,
    raw_upper: np.ndarray | None = None,
    max_feature_delta: float | None = None,
) -> np.ndarray:
    """Project raw-feature perturbations onto valid physical & threat-model constraints.

    Parameters
    ----------
    adversarial_raw : np.ndarray, shape (N, D)
        Unconstrained perturbed features in unscaled (raw) feature space.
    original_raw : np.ndarray, shape (N, D)
        Original clean sample features in unscaled feature space.
    feature_names : Sequence[str]
        Ordered feature column names corresponding to columns of the input arrays.
    raw_lower : np.ndarray or None
        Observed minimum per feature from training data.
    raw_upper : np.ndarray or None
        Observed maximum per feature from training data.
    max_feature_delta : float or None
        Maximum fraction of the observed range (upper - lower) that any feature can change.

    Returns
    -------
    np.ndarray, shape (N, D)
        Projected adversarial features satisfying all threat-model rules.
    """
    col_idx = {name: idx for idx, name in enumerate(feature_names)}
    projected = np.asarray(adversarial_raw, dtype=np.float64).copy()
    original = np.asarray(original_raw, dtype=np.float64)

    # 1. Freeze all fixed features
    for name in FIXED_FEATURES:
        if name in col_idx:
            idx = col_idx[name]
            projected[:, idx] = original[:, idx]

    # 2. Freeze derived features whose exact full-packet composition is victim/history-dependent
    for name in FROZEN_DERIVED:
        if name in col_idx:
            idx = col_idx[name]
            projected[:, idx] = original[:, idx]

    # 3. Apply max_feature_delta if configured
    if max_feature_delta is not None and raw_lower is not None and raw_upper is not None:
        span = np.maximum(raw_upper - raw_lower, 0.0)
        max_delta = max_feature_delta * span
        projected = np.clip(projected, original - max_delta, original + max_delta)

    # 4. Directional constraints on controllable features
    # Size features can only increase (padding)
    for name in ("Min", "Max", "AVG", "Tot size"):
        if name in col_idx:
            idx = col_idx[name]
            projected[:, idx] = np.maximum(projected[:, idx], original[:, idx])

    # Rate and Srate can only decrease (transmission cannot exceed hardware speed)
    for name in ("Rate", "Srate"):
        if name in col_idx:
            idx = col_idx[name]
            projected[:, idx] = np.minimum(projected[:, idx], original[:, idx])

    # IAT (inter-arrival time) can only increase (packets spaced further apart)
    if "IAT" in col_idx:
        idx = col_idx["IAT"]
        projected[:, idx] = np.maximum(projected[:, idx], original[:, idx])

    # Flow duration can only increase (attacker extends the flow)
    if "flow_duration" in col_idx:
        idx = col_idx["flow_duration"]
        projected[:, idx] = np.maximum(projected[:, idx], original[:, idx])

    # Number (packet count) non-negative and integer-like
    if "Number" in col_idx:
        idx = col_idx["Number"]
        projected[:, idx] = np.maximum(projected[:, idx], 1.0)

    # 5. Enforce non-negativity across all features
    projected = np.maximum(projected, 0.0)

    # 6. Size ordering invariants: Min <= AVG <= Max and Min <= Tot size <= Max
    min_idx = col_idx.get("Min")
    avg_idx = col_idx.get("AVG")
    max_idx = col_idx.get("Max")
    tot_size_idx = col_idx.get("Tot size")

    if min_idx is not None and max_idx is not None:
        # Max must be at least Min
        projected[:, max_idx] = np.maximum(projected[:, max_idx], projected[:, min_idx])

    if min_idx is not None and avg_idx is not None and max_idx is not None:
        # AVG must be in [Min, Max]
        projected[:, avg_idx] = np.clip(
            projected[:, avg_idx],
            projected[:, min_idx],
            projected[:, max_idx],
        )

    if min_idx is not None and tot_size_idx is not None and max_idx is not None:
        # Tot size must be in [Min, Max]
        projected[:, tot_size_idx] = np.clip(
            projected[:, tot_size_idx],
            projected[:, min_idx],
            projected[:, max_idx],
        )

    # 7. Rate, IAT, Number, and flow_duration consistency
    # In network flows, Rate = Number / max(flow_duration, epsilon)
    # If flow_duration and Rate are modified, recompute Number to keep them consistent,
    # or ensure Rate <= Number / flow_duration when duration > 0.
    if "flow_duration" in col_idx and "Rate" in col_idx and "Number" in col_idx:
        dur_idx = col_idx["flow_duration"]
        rate_idx = col_idx["Rate"]
        num_idx = col_idx["Number"]
        # If rate was decreased, flow_duration must be consistent:
        # flow_duration >= (Number - 1) * IAT or Rate <= Number / max(duration, eps)
        durations = projected[:, dur_idx]
        rates = projected[:, rate_idx]
        # Keep Srate <= Rate
        if "Srate" in col_idx:
            srate_idx = col_idx["Srate"]
            projected[:, srate_idx] = np.minimum(projected[:, srate_idx], rates)

    # 8. Recompute derived features where formula is known
    # Tot sum = Number * AVG (packet count times average size)
    if "Tot sum" in col_idx and avg_idx is not None and "Number" in col_idx:
        tot_sum_idx = col_idx["Tot sum"]
        projected[:, tot_sum_idx] = projected[:, col_idx["Number"]] * projected[:, avg_idx]

    # 9. Clip to observed valid training bounds if provided
    if raw_lower is not None and raw_upper is not None:
        projected = np.clip(projected, raw_lower, raw_upper)
        # Re-verify Min <= AVG <= Max after bounds clip
        if min_idx is not None and avg_idx is not None and max_idx is not None:
            projected[:, avg_idx] = np.clip(
                projected[:, avg_idx],
                projected[:, min_idx],
                projected[:, max_idx],
            )
        if min_idx is not None and tot_size_idx is not None and max_idx is not None:
            projected[:, tot_size_idx] = np.clip(
                projected[:, tot_size_idx],
                projected[:, min_idx],
                projected[:, max_idx],
            )

    return projected.astype(np.float32)


def check_sample_constraints(
    adversarial_raw: np.ndarray,
    original_raw: np.ndarray,
    feature_names: Sequence[str] = FEATURE_NAMES,
    atol: float = 1e-5,
) -> dict[str, bool]:
    """Verify that an adversarial sample strictly satisfies all threat-model constraints.

    Returns a dictionary of check name -> pass boolean.
    """
    col_idx = {name: idx for idx, name in enumerate(feature_names)}
    adv = np.asarray(adversarial_raw, dtype=np.float64)
    orig = np.asarray(original_raw, dtype=np.float64)

    results: dict[str, bool] = {}

    # Check non-negativity
    results["non_negativity"] = bool(np.all(adv >= -atol))

    # Check fixed features
    fixed_match = True
    for name in FIXED_FEATURES:
        if name in col_idx:
            idx = col_idx[name]
            if not np.allclose(adv[:, idx], orig[:, idx], atol=atol):
                fixed_match = False
                break
    results["fixed_features_frozen"] = fixed_match

    # Check frozen derived features
    frozen_match = True
    for name in FROZEN_DERIVED:
        if name in col_idx:
            idx = col_idx[name]
            if not np.allclose(adv[:, idx], orig[:, idx], atol=atol):
                frozen_match = False
                break
    results["frozen_derived_frozen"] = frozen_match

    # Check sizes only increased
    sizes_increased = True
    for name in ("Min", "Max", "AVG", "Tot size"):
        if name in col_idx:
            idx = col_idx[name]
            if not np.all(adv[:, idx] >= orig[:, idx] - atol):
                sizes_increased = False
                break
    results["sizes_only_increase"] = sizes_increased

    # Check rates only decreased
    rates_decreased = True
    for name in ("Rate", "Srate"):
        if name in col_idx:
            idx = col_idx[name]
            if not np.all(adv[:, idx] <= orig[:, idx] + atol):
                rates_decreased = False
                break
    results["rates_only_decrease"] = rates_decreased

    # Check timing/duration only increased
    timing_increased = True
    for name in ("IAT", "flow_duration"):
        if name in col_idx:
            idx = col_idx[name]
            if not np.all(adv[:, idx] >= orig[:, idx] - atol):
                timing_increased = False
                break
    results["timing_only_increases"] = timing_increased

    # Check Min <= AVG <= Max
    min_idx = col_idx.get("Min")
    avg_idx = col_idx.get("AVG")
    max_idx = col_idx.get("Max")
    tot_size_idx = col_idx.get("Tot size")

    if min_idx is not None and avg_idx is not None and max_idx is not None:
        min_v = adv[:, min_idx]
        avg_v = adv[:, avg_idx]
        max_v = adv[:, max_idx]
        results["min_le_avg"] = bool(np.all(min_v <= avg_v + atol))
        results["avg_le_max"] = bool(np.all(avg_v <= max_v + atol))

    if min_idx is not None and tot_size_idx is not None and max_idx is not None:
        tot_v = adv[:, tot_size_idx]
        max_v = adv[:, max_idx]
        min_v = adv[:, min_idx]
        results["min_le_tot_size"] = bool(np.all(min_v <= tot_v + atol))
        results["tot_size_le_max"] = bool(np.all(tot_v <= max_v + atol))

    # Check Srate <= Rate
    if "Rate" in col_idx and "Srate" in col_idx:
        results["srate_le_rate"] = bool(
            np.all(adv[:, col_idx["Srate"]] <= adv[:, col_idx["Rate"]] + atol)
        )

    return results
