from __future__ import annotations

from src.features import (
    CONTROLLABLE_FEATURES,
    DERIVED_FEATURES,
    FEATURE_NAMES,
    FEATURE_SPECS,
    FIXED_FEATURES,
    feature_table,
)


def test_feature_table_is_complete_and_unique() -> None:
    assert len(FEATURE_SPECS) == 46
    assert len(FEATURE_NAMES) == 46
    assert len(set(FEATURE_NAMES)) == 46


def test_feature_groups_partition_the_table() -> None:
    grouped = set(CONTROLLABLE_FEATURES) | set(FIXED_FEATURES) | set(DERIVED_FEATURES)
    assert grouped == set(FEATURE_NAMES)
    assert not set(CONTROLLABLE_FEATURES) & set(FIXED_FEATURES)
    assert not set(CONTROLLABLE_FEATURES) & set(DERIVED_FEATURES)
    assert not set(FIXED_FEATURES) & set(DERIVED_FEATURES)


def test_known_feature_groups() -> None:
    assert "IAT" in CONTROLLABLE_FEATURES
    assert "Srate" in CONTROLLABLE_FEATURES
    assert "Drate" in FIXED_FEATURES
    assert "fin_flag_number" in FIXED_FEATURES
    assert "Tot sum" in DERIVED_FEATURES
    assert "Weight" in DERIVED_FEATURES


def test_feature_table_columns() -> None:
    table = feature_table()
    assert list(table.columns) == ["feature", "group", "justification"]
    assert set(table["group"]) == {"controllable", "fixed", "derived"}
    assert table["justification"].str.len().min() > 0
