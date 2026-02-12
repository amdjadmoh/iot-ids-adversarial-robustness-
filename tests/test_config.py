from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "seed: 7\ndata: {}\nlabels: {}\nsplit: {}\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config["seed"] == 7


def test_load_config_rejects_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("seed: 7\ndata: {}\n", encoding="utf-8")

    with pytest.raises(KeyError, match="labels"):
        load_config(path)


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- seed\n- data\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
