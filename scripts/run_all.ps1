# Reproduce the full pipeline from data/raw to results/ and figures/.
# Run from the repository root: .\scripts\run_all.ps1
$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment not found at .venv. Create it and install requirements first."
}

& $python -m src.data --config configs/default.yaml
& $python -m src.train --config configs/default.yaml
& $python -m src.attacks --config configs/default.yaml
& $python -m src.defenses --config configs/default.yaml
& $python -m src.evaluate --config configs/default.yaml
& $python -m src.plots --config configs/default.yaml
