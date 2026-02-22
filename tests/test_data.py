from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data import (
    BENIGN_LABEL,
    FAMILY_BY_LABEL,
    LABEL_TO_INDEX,
    load_ciciot2023,
    save_outputs,
    split_dataset,
)


def _write_capture(path: Path, label: str, rows: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "feature_a": rng.normal(size=rows),
            "feature_b": rng.integers(0, 100, size=rows),
            "Label": [label] * rows,
        }
    )
    frame.to_csv(path, index=False)


@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "raw"
    directory.mkdir()
    specs = {
        "benign_1.csv": (BENIGN_LABEL, 20),
        "benign_2.csv": (BENIGN_LABEL, 20),
        "benign_3.csv": (BENIGN_LABEL, 20),
        "ddos_1.csv": ("DDoS-ICMP_Flood", 15),
        "ddos_2.csv": ("DDoS-ICMP_Flood", 15),
        "ddos_3.csv": ("DDoS-ICMP_Flood", 15),
        "xss_1.csv": ("XSS", 9),
        "upload_1.csv": ("Uploading_Attack", 6),
    }
    for seed, (name, (label, rows)) in enumerate(specs.items()):
        _write_capture(directory / name, label, rows, seed)
    return directory


@pytest.fixture()
def config(tmp_path: Path, raw_dir: Path) -> dict[str, Any]:
    return {
        "seed": 7,
        "data": {
            "raw_dir": str(raw_dir),
            "processed_dir": str(tmp_path / "processed"),
            "file_glob": "*.csv",
            "label_column": "Label",
            "sample_size": 100,
            "read_chunksize": 7,
            "drop_columns": [],
        },
        "labels": {"multiclass": 34, "family": 8, "binary": 2},
        "split": {
            "val_fraction": 0.2,
            "test_fraction": 0.2,
            "holdout_by_capture": True,
        },
        "paths": {
            "results": str(tmp_path / "results"),
            "figures": str(tmp_path / "figures"),
        },
    }


def test_taxonomy_is_complete() -> None:
    assert len(LABEL_TO_INDEX) == 34
    assert len(set(FAMILY_BY_LABEL.values())) == 8
    assert FAMILY_BY_LABEL[BENIGN_LABEL] == "Benign"
    assert FAMILY_BY_LABEL["DDoS-ICMP_Flood"] == "DDoS"
    assert FAMILY_BY_LABEL["VulnerabilityScan"] == "Recon"
    assert FAMILY_BY_LABEL["MITM-ArpSpoofing"] == "Spoofing"
    assert FAMILY_BY_LABEL["XSS"] == "Web"


def test_load_derives_labels_and_preserves_rare_classes(
    config: dict[str, Any],
) -> None:
    frame = load_ciciot2023(config)

    assert len(frame) == 100
    assert set(frame["label_multiclass"]) == {
        BENIGN_LABEL,
        "DDoS-ICMP_Flood",
        "XSS",
        "Uploading_Attack",
    }
    assert frame.loc[frame["label_multiclass"] == BENIGN_LABEL, "label_binary"].eq(0).all()
    assert frame.loc[frame["label_multiclass"] != BENIGN_LABEL, "label_binary"].eq(1).all()
    assert (
        frame.loc[frame["label_multiclass"] == "Uploading_Attack", "label_family"]
        .eq("Web")
        .all()
    )


def test_sampling_is_deterministic(config: dict[str, Any]) -> None:
    first = load_ciciot2023(config)
    second = load_ciciot2023(config)

    pd.testing.assert_frame_equal(first, second)


def test_split_keeps_captures_intact(config: dict[str, Any]) -> None:
    frame = split_dataset(load_ciciot2023(config), config)

    assert set(frame["split"]) == {"train", "val", "test"}
    for label in (BENIGN_LABEL, "DDoS-ICMP_Flood"):
        label_frame = frame[frame["label_multiclass"] == label]
        assert set(label_frame["split"]) == {"train", "val", "test"}
        assert label_frame.groupby("source_file")["split"].nunique().max() == 1

    xss = frame[frame["label_multiclass"] == "XSS"]
    assert set(xss["split"]) == {"train", "val", "test"}


def test_row_split_when_capture_holdout_disabled(config: dict[str, Any]) -> None:
    config["split"]["holdout_by_capture"] = False

    frame = split_dataset(load_ciciot2023(config), config)

    assert set(frame["split"]) == {"train", "val", "test"}


def test_save_outputs_writes_generated_files(
    config: dict[str, Any],
    tmp_path: Path,
) -> None:
    frame = split_dataset(load_ciciot2023(config), config)
    save_outputs(frame, config)

    processed = tmp_path / "processed"
    for split_name in ("train", "val", "test"):
        assert (processed / f"{split_name}.pkl").exists()
    assert (processed / "ciciot2023_sample.pkl").exists()

    results = tmp_path / "results"
    distribution = pd.read_csv(results / "label_distribution.csv")
    assert set(distribution.columns) == {
        "split",
        "label_multiclass",
        "label_family",
        "label_binary",
        "count",
        "fraction_within_split",
    }
    assert distribution["count"].sum() == len(frame)

    summary = json.loads((results / "split_summary.json").read_text(encoding="utf-8"))
    assert summary["sampled_rows"] == len(frame)
    assert set(summary["splits"]) == {"train", "val", "test"}
    assert (results / "split_files.csv").exists()


def test_unknown_label_is_rejected(
    config: dict[str, Any],
    raw_dir: Path,
) -> None:
    _write_capture(raw_dir / "unknown.csv", "NotAnAttack", 5, seed=99)

    with pytest.raises(ValueError, match="Unknown CICIoT2023 label"):
        load_ciciot2023(config)


def test_loader_strips_whitespace_from_headers(
    config: dict[str, Any],
    tmp_path: Path,
) -> None:
    directory = tmp_path / "spaced"
    directory.mkdir()
    pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            " Label": [BENIGN_LABEL] * 3,
        }
    ).to_csv(directory / "benign.csv", index=False)

    config["data"]["raw_dir"] = str(directory)
    config["data"]["sample_size"] = 3

    frame = load_ciciot2023(config)

    assert "Label" in frame.columns
    assert frame["label_multiclass"].eq(BENIGN_LABEL).all()


def test_loader_strips_byte_order_mark_from_headers(
    config: dict[str, Any],
    tmp_path: Path,
) -> None:
    directory = tmp_path / "bom"
    directory.mkdir()
    pd.DataFrame(
        {
            "\ufefffeature_a": [1.0, 2.0, 3.0],
            "label": [BENIGN_LABEL] * 3,
        }
    ).to_csv(directory / "benign.csv", index=False)

    config["data"]["raw_dir"] = str(directory)
    config["data"]["label_column"] = "label"
    config["data"]["sample_size"] = 3

    frame = load_ciciot2023(config)

    assert "feature_a" in frame.columns
    assert frame["label_multiclass"].eq(BENIGN_LABEL).all()
