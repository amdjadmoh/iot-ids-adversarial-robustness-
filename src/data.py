"""Dataset discovery, loading, sampling, and leakage-aware splitting for CICIoT2023.

The official distribution contains one CSV per capture/attack type. Sampling is
performed per capture file in bounded memory, so the full ~13 GB dataset can be
processed on a laptop without loading it all into RAM. Splits keep capture files
intact when a label has at least three files. Labels with one or two files fall
back to a row-level split and log a warning, because strict capture holdout would
otherwise leave those labels out of validation or test.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import configure_logging, load_config, set_seed

LOGGER = logging.getLogger(__name__)

BENIGN_LABEL = "BenignTraffic"

MULTICLASS_LABELS: tuple[str, ...] = (
    BENIGN_LABEL,
    "DictionaryBruteForce",
    "DDoS-ACK_Fragmentation",
    "DDoS-HTTP_Flood",
    "DDoS-ICMP_Fragmentation",
    "DDoS-ICMP_Flood",
    "DDoS-PSHACK_Flood",
    "DDoS-RSTFINFlood",
    "DDoS-SYN_Flood",
    "DDoS-SlowLoris",
    "DDoS-SynonymousIP_Flood",
    "DDoS-TCP_Flood",
    "DDoS-UDP_Fragmentation",
    "DDoS-UDP_Flood",
    "DoS-HTTP_Flood",
    "DoS-SYN_Flood",
    "DoS-TCP_Flood",
    "DoS-UDP_Flood",
    "Mirai-greeth_flood",
    "Mirai-greip_flood",
    "Mirai-udpplain",
    "Recon-HostDiscovery",
    "Recon-OSScan",
    "Recon-PingSweep",
    "Recon-PortScan",
    "VulnerabilityScan",
    "DNS_Spoofing",
    "MITM-ArpSpoofing",
    "Backdoor_Malware",
    "BrowserHijacking",
    "CommandInjection",
    "SqlInjection",
    "Uploading_Attack",
    "XSS",
)

FAMILY_NAMES: tuple[str, ...] = (
    "Benign",
    "BruteForce",
    "DDoS",
    "DoS",
    "Mirai",
    "Recon",
    "Spoofing",
    "Web",
)

LABEL_ALIASES = {
    "Benign": BENIGN_LABEL,
    "BENIGN": BENIGN_LABEL,
}


def _family_for_label(label: str) -> str:
    """Return the 8-class family for a canonical CICIoT2023 label."""
    if label == BENIGN_LABEL:
        return "Benign"
    if label == "DictionaryBruteForce":
        return "BruteForce"
    if label.startswith("DDoS-"):
        return "DDoS"
    if label.startswith("DoS-"):
        return "DoS"
    if label.startswith("Mirai-"):
        return "Mirai"
    if label in {
        "Recon-HostDiscovery",
        "Recon-OSScan",
        "Recon-PingSweep",
        "Recon-PortScan",
        "VulnerabilityScan",
    }:
        return "Recon"
    if label in {"DNS_Spoofing", "MITM-ArpSpoofing"}:
        return "Spoofing"
    if label in {
        "Backdoor_Malware",
        "BrowserHijacking",
        "CommandInjection",
        "SqlInjection",
        "Uploading_Attack",
        "XSS",
    }:
        return "Web"
    raise ValueError(f"Unmapped CICIoT2023 label: {label!r}")


FAMILY_BY_LABEL: dict[str, str] = {
    label: _family_for_label(label) for label in MULTICLASS_LABELS
}
FAMILY_TO_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(FAMILY_NAMES)
}
LABEL_TO_INDEX: dict[str, int] = {
    label: index for index, label in enumerate(MULTICLASS_LABELS)
}


@dataclass(frozen=True)
class FileStats:
    """Row and label counts for one CSV capture file."""

    rows: int
    label_counts: dict[str, int]


def normalize_label(value: Any) -> str:
    """Normalize a dataset label and reject unknown values early."""
    label = str(value).strip()
    label = LABEL_ALIASES.get(label, label)
    if label not in LABEL_TO_INDEX:
        raise ValueError(
            f"Unknown CICIoT2023 label {value!r}. "
            f"Expected one of {len(MULTICLASS_LABELS)} known labels."
        )
    return label


def discover_csv_files(raw_dir: str | Path, pattern: str = "*.csv") -> list[Path]:
    """Return all CSV files below ``raw_dir``, sorted for reproducibility."""
    directory = Path(raw_dir)
    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {directory}. "
            "See scripts/download_instructions.md."
        )
    files = sorted(path for path in directory.rglob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(
            f"No files matching {pattern!r} under {directory}. "
            "See scripts/download_instructions.md."
        )
    return files


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [
        str(column).strip().lstrip("\ufeff") for column in frame.columns
    ]
    return frame


def _find_label_column(columns: pd.Index, label_column: str) -> str:
    if label_column in columns:
        return label_column
    matches = [
        str(column)
        for column in columns
        if str(column).lower() == label_column.lower()
    ]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(
        f"Label column {label_column!r} not found in columns: {list(columns)}"
    )


def _clean_chunks(path: Path, chunksize: int) -> Any:
    for chunk in pd.read_csv(
        path,
        chunksize=chunksize,
        skipinitialspace=True,
        low_memory=False,
    ):
        yield _clean_columns(chunk)


def inspect_files(
    files: list[Path],
    label_column: str,
    chunksize: int,
) -> dict[Path, FileStats]:
    """Count rows and validate labels in every capture file."""
    if chunksize <= 0:
        raise ValueError("read_chunksize must be positive")

    stats: dict[Path, FileStats] = {}
    for path in files:
        header = _clean_columns(pd.read_csv(path, nrows=0, skipinitialspace=True))
        actual_label_column = _find_label_column(header.columns, label_column)
        rows = 0
        raw_counts: Counter[str] = Counter()

        for chunk in pd.read_csv(
            path,
            usecols=[actual_label_column],
            chunksize=chunksize,
            skipinitialspace=True,
        ):
            values = chunk[actual_label_column].astype("string").str.strip()
            raw_counts.update(values.value_counts(dropna=False).to_dict())
            rows += len(chunk)

        label_counts: Counter[str] = Counter()
        for value, count in raw_counts.items():
            if pd.isna(value):
                raise ValueError(f"Missing label in {path.name}")
            label_counts[normalize_label(value)] += int(count)

        if rows == 0:
            raise ValueError(f"Capture file is empty: {path}")
        stats[path] = FileStats(rows=rows, label_counts=dict(label_counts))
        LOGGER.debug("Inspected %s: %d rows", path.name, rows)
    return stats


def sample_file(
    path: Path,
    target_rows: int,
    seed: int,
    chunksize: int,
    label_column: str = "Label",
    min_per_class: int = 2,
) -> pd.DataFrame:
    """Sample ``target_rows`` rows from one CSV using bounded memory with rare-class protection."""
    if target_rows <= 0:
        raise ValueError("target_rows must be positive")

    rng = np.random.default_rng(seed)
    reservoir: pd.DataFrame | None = None

    for chunk in _clean_chunks(path, chunksize):
        if reservoir is None:
            combined = chunk
        else:
            combined = pd.concat([reservoir, chunk], ignore_index=True)

        if len(combined) > target_rows:
            # Guarantee rare classes retain at least `min_per_class` rows if available
            actual_lbl_col = _find_label_column(combined.columns, label_column)
            keep_indices = set()
            grouped = combined.groupby(actual_lbl_col, observed=True)
            for _, grp in grouped:
                if len(grp) <= min_per_class:
                    keep_indices.update(grp.index)
                else:
                    sample_idx = rng.choice(grp.index, size=min_per_class, replace=False)
                    keep_indices.update(sample_idx)

            remaining_slots = target_rows - len(keep_indices)
            if remaining_slots > 0:
                other_indices = combined.index.difference(list(keep_indices))
                sampled_other = rng.choice(other_indices, size=min(remaining_slots, len(other_indices)), replace=False)
                final_keep = np.sort(np.concatenate([list(keep_indices), sampled_other]))
            else:
                final_keep = np.sort(list(keep_indices)[:target_rows])

            combined = combined.iloc[final_keep].reset_index(drop=True)
        reservoir = combined

    if reservoir is None:
        raise ValueError(f"No rows read from {path}")
    return reservoir


def _sample_files(
    files: list[Path],
    stats: dict[Path, FileStats],
    sample_size: int | None,
    seed: int,
    chunksize: int,
    label_column: str = "Label",
) -> pd.DataFrame:
    total_rows = sum(stats[path].rows for path in files)
    if sample_size is not None and sample_size <= 0:
        raise ValueError("data.sample_size must be positive or null")

    if sample_size is None or sample_size >= total_rows:
        targets = {path: stats[path].rows for path in files}
        LOGGER.info("Loading all %d rows from %d files", total_rows, len(files))
    else:
        targets = {
            path: max(1, int(round(sample_size * stats[path].rows / total_rows)))
            for path in files
        }
        LOGGER.info(
            "Sampling %d of %d rows from %d files", sum(targets.values()),
            total_rows, len(files),
        )

    frames: list[pd.DataFrame] = []
    for index, path in enumerate(files):
        frame = sample_file(
            path,
            target_rows=targets[path],
            seed=seed + index,
            chunksize=chunksize,
            label_column=label_column,
        )
        frame["source_file"] = path.name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def _validate_taxonomy(config: dict[str, Any]) -> None:
    expected = config.get("labels", {})
    checks = {
        "multiclass": len(MULTICLASS_LABELS),
        "family": len(FAMILY_NAMES),
        "binary": 2,
    }
    for key, actual in checks.items():
        configured = expected.get(key, actual)
        if int(configured) != actual:
            raise ValueError(
                f"labels.{key}={configured} does not match the taxonomy ({actual})"
            )


def load_ciciot2023(config: dict[str, Any]) -> pd.DataFrame:
    """Load, sample, and label a CICIoT2023 CSV directory."""
    _validate_taxonomy(config)
    data_cfg = config["data"]
    files = discover_csv_files(
        data_cfg["raw_dir"],
        pattern=data_cfg.get("file_glob", "*.csv"),
    )
    chunksize = int(data_cfg.get("read_chunksize", 250_000))
    label_column = str(data_cfg.get("label_column", "Label"))

    stats = inspect_files(files, label_column, chunksize)
    frame = _sample_files(
        files,
        stats,
        sample_size=data_cfg.get("sample_size"),
        seed=int(config["seed"]),
        chunksize=chunksize,
        label_column=label_column,
    )
    frame = _clean_columns(frame)
    actual_label_column = _find_label_column(frame.columns, label_column)

    drop_columns = [column for column in data_cfg.get("drop_columns", []) if column in frame.columns]
    if actual_label_column in drop_columns:
        raise ValueError("The label column cannot be dropped")
    if drop_columns:
        frame = frame.drop(columns=drop_columns)

    labels = frame[actual_label_column].map(normalize_label)
    frame["label_multiclass"] = labels
    frame["label_family"] = labels.map(FAMILY_BY_LABEL)
    frame["label_binary"] = (labels != BENIGN_LABEL).astype("int8")
    frame["label_id"] = labels.map(LABEL_TO_INDEX).astype("int16")

    multi_label = (
        frame.groupby("source_file")["label_multiclass"].nunique().loc[lambda s: s > 1]
    )
    if not multi_label.empty:
        LOGGER.warning(
            "%d capture files contain more than one label: %s",
            len(multi_label),
            ", ".join(multi_label.index),
        )

    LOGGER.info(
        "Loaded %d sampled rows across %d labels from %d capture files",
        len(frame),
        frame["label_multiclass"].nunique(),
        frame["source_file"].nunique(),
    )
    return frame


def _row_split(
    group: pd.DataFrame,
    val_fraction: float,
    test_fraction: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Fallback row-level split for labels with too few capture files."""
    frame = group.copy()
    n_rows = len(frame)
    if n_rows < 3:
        LOGGER.warning(
            "Label %s has only %d rows; assigning all rows to train",
            frame["label_multiclass"].iloc[0],
            n_rows,
        )
        frame["split"] = "train"
        return frame

    n_test = max(1, int(round(n_rows * test_fraction)))
    n_val = max(1, int(round(n_rows * val_fraction)))
    if n_test + n_val >= n_rows:
        n_test = 1
        n_val = 1

    permutation = rng.permutation(n_rows)
    split = np.full(n_rows, "train", dtype=object)
    split[permutation[:n_test]] = "test"
    split[permutation[n_test : n_test + n_val]] = "val"
    frame["split"] = split
    return frame


def _assign_capture_files(
    row_counts: dict[str, int],
    val_fraction: float,
    test_fraction: float,
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    """Assign capture files to splits, guaranteeing non-empty train/val/test."""
    if len(row_counts) < 3:
        raise ValueError("Capture assignment requires at least three files")

    files = sorted(row_counts)
    ordered = sorted(files, key=lambda name: (row_counts[name], name))
    test_file = ordered[0]
    val_file = ordered[1]
    train_file = ordered[-1]
    remaining = [name for name in files if name not in {test_file, val_file, train_file}]
    rng.shuffle(remaining)

    train = [train_file]
    val = [val_file]
    test = [test_file]
    total_rows = sum(row_counts.values())
    target_val = val_fraction * total_rows
    target_test = test_fraction * total_rows
    val_rows = row_counts[val_file]
    test_rows = row_counts[test_file]

    for name in remaining:
        val_deficit = max(0.0, target_val - val_rows)
        test_deficit = max(0.0, target_test - test_rows)
        if val_deficit >= test_deficit and val_deficit > 0:
            val.append(name)
            val_rows += row_counts[name]
        elif test_deficit > 0:
            test.append(name)
            test_rows += row_counts[name]
        else:
            train.append(name)

    return {"train": train, "val": val, "test": test}


def split_dataset(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Add a split column, keeping captures intact wherever possible."""
    split_cfg = config["split"]
    val_fraction = float(split_cfg["val_fraction"])
    test_fraction = float(split_cfg["test_fraction"])
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("val_fraction and test_fraction must be in (0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be below 1")

    rng = np.random.default_rng(int(config["seed"]))
    parts: list[pd.DataFrame] = []

    # If files are named train.csv, validation.csv/val.csv, test.csv, adopt their natural splits
    file_stems = {str(name): Path(str(name)).stem.lower() for name in frame["source_file"].unique()}
    named_splits = {"train": "train", "val": "val", "validation": "val", "test": "test"}
    has_canonical_split_names = set(file_stems.values()).issubset({"train", "val", "validation", "test"}) and len(file_stems) >= 2

    if has_canonical_split_names:
        LOGGER.info("Adopting existing pre-split file roles: %s", file_stems)
        parts = []
        for file_name, stem in file_stems.items():
            assigned_split = named_splits[stem]
            file_part = frame[frame["source_file"] == file_name].copy()
            file_part["split"] = assigned_split
            parts.append(file_part)
        result = pd.concat(parts, ignore_index=True)
        counts = result["split"].value_counts().to_dict()
        LOGGER.info("Split sizes: %s", counts)
        return result

    if not bool(split_cfg.get("holdout_by_capture", True)):
        for _, group in frame.groupby("label_multiclass", sort=True):
            parts.append(_row_split(group, val_fraction, test_fraction, rng))
    else:
        labels_with_multi_files = (
            frame.groupby("source_file")["label_multiclass"]
            .nunique()
            .loc[lambda series: series > 1]
            .index
        )
        multi_label_files = set(labels_with_multi_files)
        if multi_label_files:
            LOGGER.warning(
                "Assigning %d multi-label capture files to train: %s",
                len(multi_label_files),
                ", ".join(sorted(multi_label_files)),
            )
            multi_frame = frame[frame["source_file"].isin(multi_label_files)].copy()
            multi_frame["split"] = "train"
            parts.append(multi_frame)

        single_frame = frame[~frame["source_file"].isin(multi_label_files)]
        for label, group in single_frame.groupby("label_multiclass", sort=True):
            files = sorted(group["source_file"].unique())
            if len(files) >= 3:
                row_counts = group.groupby("source_file").size().to_dict()
                assignments = _assign_capture_files(
                    {str(name): int(count) for name, count in row_counts.items()},
                    val_fraction,
                    test_fraction,
                    rng,
                )
                for split_name, split_files in assignments.items():
                    subset = group[group["source_file"].isin(split_files)].copy()
                    subset["split"] = split_name
                    parts.append(subset)
            else:
                LOGGER.warning(
                    "Label %s has %d capture file(s); using a row-level split",
                    label,
                    len(files),
                )
                parts.append(_row_split(group, val_fraction, test_fraction, rng))

    result = pd.concat(parts, ignore_index=True)
    counts = result["split"].value_counts().to_dict()
    LOGGER.info("Split sizes: %s", counts)
    for split_name in ("train", "val", "test"):
        if split_name not in counts:
            raise ValueError(f"Split {split_name!r} is empty")
    return result


CLASS_TARGETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "multiclass": ("label_multiclass", MULTICLASS_LABELS),
    "family": ("label_family", FAMILY_NAMES),
    "binary": ("label_binary", ("Benign", "Attack")),
}
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


def _target_names(frame: pd.DataFrame, target: str) -> pd.Series:
    """Return the class name of every row for one target granularity."""
    if target == "multiclass":
        return frame["label_multiclass"]
    if target == "family":
        return frame["label_family"]
    if target == "binary":
        return frame["label_binary"].map({0: "Benign", 1: "Attack"})
    raise ValueError(f"Unknown target: {target!r}")


def class_count_frame(
    frame: pd.DataFrame,
    min_test_samples: int,
) -> pd.DataFrame:
    """Return per-class train/validation/test counts for every target.

    Several CICIoT2023 classes have only a handful of test rows (CommandInjection
    has three), so their recall and the macro average built from it are noisy.
    Writing the counts next to the metrics keeps that visible instead of letting
    a macro-F1 hide it.
    """
    if min_test_samples < 0:
        raise ValueError("rare_class_min_test_samples must be non-negative")

    rows: list[dict[str, Any]] = []
    for target, (_, class_names) in CLASS_TARGETS.items():
        counts = {
            split: _target_names(
                frame[frame["split"] == split],
                target,
            ).value_counts()
            for split in SPLIT_NAMES
        }
        for class_name in class_names:
            row: dict[str, Any] = {"target": target, "class_name": class_name}
            for split in SPLIT_NAMES:
                row[split] = int(counts[split].get(class_name, 0))
            row["low_test_support"] = row["test"] < min_test_samples
            rows.append(row)
    return pd.DataFrame(rows)


def save_outputs(frame: pd.DataFrame, config: dict[str, Any]) -> None:
    """Persist split data under data/ and generated summaries under results/."""
    processed_dir = Path(config["data"].get("processed_dir", "data/processed"))
    results_dir = Path(config["paths"].get("results", "results"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    frame.to_pickle(processed_dir / "ciciot2023_sample.pkl")
    for split_name in ("train", "val", "test"):
        part = frame[frame["split"] == split_name]
        if part.empty:
            raise ValueError(f"Split {split_name!r} is empty")
        part.to_pickle(processed_dir / f"{split_name}.pkl")

    distribution = (
        frame.groupby(
            ["split", "label_multiclass", "label_family", "label_binary"],
            observed=True,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    distribution["fraction_within_split"] = distribution["count"] / distribution.groupby(
        "split"
    )["count"].transform("sum")
    distribution.to_csv(results_dir / "label_distribution.csv", index=False)

    file_summary = (
        frame.groupby(["split", "source_file", "label_multiclass"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    file_summary.to_csv(results_dir / "split_files.csv", index=False)

    class_counts = class_count_frame(
        frame,
        min_test_samples=int(
            config.get("data", {}).get("rare_class_min_test_samples", 30)
        ),
    )
    class_counts.to_csv(results_dir / "test_class_counts.csv", index=False)

    summary = {
        "seed": int(config["seed"]),
        "requested_sample_size": config["data"].get("sample_size"),
        "sampled_rows": int(len(frame)),
        "splits": {
            name: {
                "rows": int((frame["split"] == name).sum()),
                "classes": int(
                    frame.loc[frame["split"] == name, "label_multiclass"].nunique()
                ),
                "capture_files": int(
                    frame.loc[frame["split"] == name, "source_file"].nunique()
                ),
            }
            for name in ("train", "val", "test")
        },
    }
    with (results_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    LOGGER.info("Wrote splits to %s and summaries to %s", processed_dir, results_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()
    config = load_config(args.config)
    set_seed(int(config["seed"]), bool(config.get("deterministic", True)))
    frame = load_ciciot2023(config)
    splits = split_dataset(frame, config)
    save_outputs(splits, config)


if __name__ == "__main__":
    main()
