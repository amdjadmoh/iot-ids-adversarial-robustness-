#!/usr/bin/env bash
# Reproduce the full pipeline from data/raw to results/ and figures/.
# Run from the repository root with the virtual environment active.
set -euo pipefail

python -m src.data --config configs/default.yaml
python -m src.train --config configs/default.yaml
python -m src.attacks --config configs/default.yaml
python -m src.defenses --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml
python -m src.plots --config configs/default.yaml
