# iot-ids-adversarial

How robust are ML intrusion detectors for IoT networks to realistic adversarial evasion?

This project investigates standard IoT intrusion detectors (XGBoost, Random Forest, and a
small MLP) on CICIoT2023, measures their degradation under white-box (FGSM/PGD),
feature-constrained evasion, and black-box transfer attacks, and evaluates adversarial training
and feature squeezing as defenses.

## Status

All experimental stages (data preprocessing, baseline training, adversarial evasion,
feature-constrained projection, defense evaluation, metrics aggregation, figure plotting,
and research reporting) are fully implemented, verified, and backed by generated results.
Every reported metric is traceable to files under `results/` and figures under `figures/`.

## Key Findings

1. **RQ1 (Detector Baselines):** While Random Forest and XGBoost achieve >99.0% clean test
   accuracy across multiclass (34 classes), family (8 classes), and binary settings, macro F1
   reveals class-imbalance vulnerability (multiclass macro F1 is ~0.70 for trees, ~0.59 for MLP).
2. **RQ2 (Adversarial Evasion & Transfer):**
   - **White-box Evasion (MLP):** Unconstrained PGD ($\epsilon=0.10$) drops MLP accuracy on
     multiclass from 91.51% to 41.92%, and on family from 98.13% to 46.57%.
   - **Realistic Feature-Constrained Evasion:** Restricting perturbations strictly to
     attacker-controlled features (packet sizes only increase, timing/rates only slow down,
     $Min \le AVG \le Max$, non-negativity enforced) still causes MLP multiclass accuracy
     to collapse from 91.51% to 42.65% (53.40% attack success rate).
   - **Transferability to Trees:** Gradient-based adversarial examples generated on the
     differentiable surrogate MLP transfer with devastating effect: Random Forest multiclass
     accuracy drops from 99.01% to 29.16%, and XGBoost drops from 99.06% to 16.96%.
3. **RQ3 (Defenses & Trade-offs):**
   - **Adversarial Training:** Restores multiclass adversarial accuracy under PGD ($\epsilon=0.05$)
     from 41.92% to 72.81%, at an ~11.1% clean accuracy cost.
   - **Feature Squeezing:** Spatial bit quantization offers limited defense on tabular flow features
     and significantly damages clean accuracy at lower bit depths (e.g. 4-bit drops clean accuracy to 58.95%).

## Threat Model & Constraints

The threat model evaluates feature-space evasion by an attacker controlling a compromised IoT device:
- **Controllable (10 features):** Packet sizes (`Min`, `Max`, `AVG`, `Std`, `Tot size`), rates (`Rate`, `Srate`),
  and timings (`IAT`, `flow_duration`, `Number`).
- **Directional rules:** Size features can only increase (payload padding); transmission rate can only decrease;
  timing (inter-arrival time and flow duration) can only increase.
- **Physical invariants:** Strict non-negativity, $Min \le AVG \le Max$, $Min \le \text{Tot size} \le Max$, and
  consistency between Rate, Srate, IAT, Number, and duration.
- **Fixed (25 features):** Network and transport protocol indicators, packet headers, flags, and victim-side measurements (`Drate`).
- **Derived features (11 features):** `Tot sum` is recomputed dynamically ($Number \times AVG$). Interaction
  features (`Magnitue`, `Radius`, `Covariance`, `Variance`, `Weight`, and flag counts) whose closed-form
  derivation depends on per-packet state without summary formulas are frozen during constrained evasion.

## Setup

Requires Python 3.12 (the pinned stack in `requirements.txt` targets 3.12) and an
NVIDIA GPU with a CUDA 13.0-compatible driver for the CUDA PyTorch build.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Extract the CICIoT2023 CSV captures into `data/raw/` as detailed in `scripts/download_instructions.md`.

## Reproducing the Complete Pipeline

Run the end-to-end automated pipeline from the repository root:

```powershell
.\scripts\run_all.ps1
```

Equivalent on POSIX systems:
```bash
./scripts/run_all.sh
```

### Pipeline Execution Order

| Step | Script / Module | Outputs |
| :--- | :--- | :--- |
| 1. Load & Split | `python -m src.data` | `data/processed/{train,val,test}.pkl`, `results/label_distribution.csv` |
| 2. Train Baselines | `python -m src.train` | `data/models/*.joblib`, `data/models/*.pt`, `results/baselines_metrics.csv` |
| 3. Run Attacks | `python -m src.attacks` | `results/attack_metrics.csv`, `results/attack_per_class_recall.csv` |
| 4. Run Defenses | `python -m src.defenses` | `data/models/*_adv_*.pt`, `results/defense_metrics.csv` |
| 5. Evaluation Summaries | `python -m src.evaluate` | `results/baseline_test_summary.csv`, `results/attack_worst_case.csv`, `results/experiment_summary.json` |
| 6. Generate Figures | `python -m src.plots` | `figures/*.png` (23 publication-quality plots) |

Detailed write-up and tabular breakdowns can be found in `report/report.md`.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```
All 59 unit and regression tests pass.

## License

MIT. See `LICENSE`.
