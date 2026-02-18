# iot-ids-adversarial

How robust are ML intrusion detectors for IoT networks to realistic adversarial evasion?

This project studies standard IoT intrusion detectors (XGBoost, Random Forest, and a
small MLP) on CICIoT2023, measures their degradation under gradient-based, transfer,
and feature-constrained evasion attacks, and evaluates adversarial training and feature
squeezing as defenses.

## Status

The data, baseline, attack, defense, evaluation, and plotting entry points are
implemented and unit-tested. The CICIoT2023 dataset is not committed; download it
before running the full pipeline. Every reported number is generated under `results/`
or `figures/`; no result values are hand-written.

The feature-controllability table in `src/features.py` is still a draft. Constrained
attacks and black-box attacks are not wired yet because they depend on the final
controllability decisions.

## Setup

Requires Python 3.12 (the pinned stack in `requirements.txt` targets 3.12) and an
NVIDIA GPU with a CUDA 13.0-compatible driver for the CUDA PyTorch build.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download CICIoT2023 as described in `scripts/download_instructions.md` and place the
CSV files under `data/raw/`.

## Run

```powershell
.\scripts\run_all.ps1
```

Equivalent on POSIX shells: `./scripts/run_all.sh`.

The pipeline runs in order:

| Step | Command | Main outputs |
| --- | --- | --- |
| Load and split | `python -m src.data` | `data/processed/*.pkl`, `results/label_distribution.csv`, `results/split_summary.json` |
| Baselines | `python -m src.train` | `results/baselines_metrics.csv`, `results/baselines_per_class_recall.csv`, `data/models/` |
| Evasion | `python -m src.attacks` | `results/attack_metrics.csv`, `results/attack_per_class_recall.csv` |
| Defenses | `python -m src.defenses` | `results/defense_metrics.csv`, `results/defense_training_*.csv` |
| Summaries | `python -m src.evaluate` | `results/baseline_test_summary.csv`, `results/attack_worst_case.csv`, `results/experiment_summary.json` |
| Figures | `python -m src.plots` | `figures/*.png` |

All commands accept `--config configs/default.yaml`. `src.train`, `src.attacks`, and
`src.defenses` also accept `--targets` and `--models`; `src.attacks` accepts
`--attacks fgsm pgd`, and `src.defenses` accepts
`--defenses adversarial_training feature_squeezing`.

## Evaluation protocol

- **Targets:** 34-class attack type, 8-class family, and binary Benign/Attack.
- **Features:** the 46 CICIoT2023 flow features. Trees consume median-imputed raw
  features; the MLP consumes median-imputed, standardized features clipped to the
  configured range.
- **Splits:** capture-aware where possible, with a documented row-level fallback for
  labels with too few capture files.
- **Attacks:** FGSM and PGD on the MLP in scaled feature space, bounded to the
  training feature ranges, then transferred to the tree models through the fitted
  scaler and observed raw ranges.
- **Defenses:** FGSM adversarial training of the MLP and inference-time feature
  squeezing at the configured bit depths.

The detector is frozen during evaluation. The threat model allows only sender-side
traffic shaping; protocol flags, victim-side measurements, and network-derived
statistics are fixed.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## License

MIT. See `LICENSE`.
