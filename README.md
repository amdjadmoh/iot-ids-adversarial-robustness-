# iot-ids-adversarial

How robust are ML intrusion detectors for IoT networks to realistic adversarial evasion?

This project studies three detectors (Random Forest, XGBoost, and a small MLP) on
CICIoT2023, measures how far their accuracy and attack-class recall degrade under
white-box (FGSM/PGD), feature-constrained, and transfer attacks, and evaluates
adversarial training and feature squeezing as defenses. Every number below is read
from a file under `results/`; nothing is hand-entered or estimated.

## Status

The data pipeline, baseline training, attacks, defenses, evaluation summaries, and
figures are implemented and were re-run end to end after the last review. The unit
suite (79 tests) passes. All result files quoted here were produced by that run.

## Key Findings

### RQ1 — clean accuracy is high, macro F1 is not

| Target | Model | Accuracy | Macro F1 |
| :--- | :--- | ---: | ---: |
| Multiclass (34) | Random Forest | 99.01% | 0.7054 |
| Multiclass (34) | XGBoost | 99.06% | 0.6947 |
| Multiclass (34) | MLP | 91.51% | 0.5937 |
| Family (8) | Random Forest | 99.31% | 0.6851 |
| Family (8) | XGBoost | 99.24% | 0.6937 |
| Family (8) | MLP | 98.13% | 0.6328 |
| Binary | Random Forest | 99.62% | 0.9578 |
| Binary | XGBoost | 99.56% | 0.9523 |
| Binary | MLP | 99.22% | 0.9152 |

(`results/baseline_test_summary.csv`.) The gap between accuracy and macro F1 comes
from rare classes: 10 of the 34 multiclass labels and 2 of the 8 family labels have
fewer than 30 test samples (`results/test_class_counts.csv`, see Limitations).

### RQ2 — accuracy collapse is not evasion

The primary metric is the **binary evasion rate**: among test flows that are true
attacks *and* classified as attacks before perturbation, the fraction predicted
benign afterwards. It answers the operational question — did the attack stop being
detected? — which accuracy alone does not.

| Target | Model | Constrained | FGSM | PGD | Random noise |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Binary | Random Forest | 0.01% | 0.13% | 0.13% | 0.18% |
| Binary | XGBoost | 0.04% | 0.15% | 0.15% | 0.14% |
| Binary | MLP | 0.04% | 0.37% | 0.39% | 0.07% |
| Multiclass | Random Forest | 0.11% | 0.50% | 0.48% | 0.45% |
| Multiclass | XGBoost | 0.13% | 0.25% | 0.24% | 0.24% |
| Multiclass | MLP | 0.06% | 0.21% | 0.25% | 0.08% |
| Family | Random Forest | 0.07% | 0.35% | 0.34% | 0.23% |
| Family | XGBoost | 0.06% | 0.27% | 0.27% | 0.18% |
| Family | MLP | 0.03% | 0.21% | 0.24% | 0.03% |

Maximum over the tested budgets, from `results/attack_worst_case.csv`. **No attack,
model, or target exceeds 0.5% evasion.** At the largest budget (ε = 0.10) multiclass
PGD moves MLP accuracy from 91.51% to 41.92% — a 49.59-point drop — while only 0.25%
of detected attacks become benign (`results/attack_metrics.csv`). The remaining
damage is attack-to-attack confusion: the sample is still flagged, just as the wrong
attack.

In the binary task the accuracy loss has the opposite sign of what "evasion" implies.
Under PGD at ε = 0.10 the Random Forest's benign recall falls from 91.37% to 7.91%
while its attack recall is unchanged (99.82% → 99.95%); XGBoost goes from 92.37% to
10.22% benign recall (`results/attack_per_class_recall.csv`). The detector does not
miss attacks — it raises false alarms on perturbed benign traffic.

**Transfer beyond plain noise is much smaller than the raw drop.** Comparing each
transferred attack with random-sign noise at the same budget (ε = 0.10):

| Target | Model | Clean | PGD | Random noise | Extra drop vs noise |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Multiclass | Random Forest | 99.01% | 29.16% | 45.09% | 15.94 pts |
| Multiclass | XGBoost | 99.06% | 16.96% | 34.18% | 17.22 pts |
| Family | Random Forest | 99.31% | 25.88% | 58.24% | 32.36 pts |
| Family | XGBoost | 99.24% | 9.41% | 47.37% | 37.96 pts |

(`results/transfer_summary.csv`, column `accuracy_drop_vs_noise`.) Trees are fragile
to any perturbation of this size; roughly half to two thirds of the reported "transfer
effect" is that fragility rather than a gradient-specific direction.

**Realistic constrained evasion mostly does not work.** Restricting changes to
attacker-controlled features (sizes may only grow, rates and timings may only slow)
leaves binary evasion at 0.01–0.04%, and multiclass/family evasion at 0.03–0.13%.
Tree accuracy under the constrained attack (81.90–87.62% multiclass) is *higher* than
under random noise at the same budget, so the constrained attacker is weaker than
unstructured noise for trees (`results/transfer_summary.csv`, negative
`accuracy_drop_vs_noise`). The MLP is the exception in accuracy terms — 42.65%
multiclass — but its evasion rate is still 0.06%.

### RQ3 — adversarial training works, squeezing does not

Baseline and adversarially-trained MLP are compared on the same attack at the same ε,
so no defense is credited for facing an easier budget (`results/defense_matched_epsilon.csv`):

| Target | PGD ε | Baseline | Adv. train ε=0.01 | Adv. train ε=0.02 |
| :--- | ---: | ---: | ---: | ---: |
| Multiclass | 0.05 | 42.79% | 72.36% | 72.81% |
| Multiclass | 0.10 | 41.92% | 57.08% | 66.71% |
| Family | 0.05 | 48.34% | 77.92% | 81.01% |
| Family | 0.10 | 48.06% | 69.69% | 74.22% |
| Binary | 0.05 | 98.90% | 98.93% | 98.97% |

Clean accuracy costs 10.29 points (multiclass, ε_train = 0.01), 11.10 (ε_train = 0.02),
12.00 and 12.38 on family, and 0.03–0.04 on binary. Attacks against the defended model
are **adaptive** — regenerated against the defended network — so these are not inflated
by reusing stale examples.

Feature squeezing in per-feature quantile space preserves clean accuracy at 8 bits
(largest drop 0.32 points across the nine model/target pairs,
`results/squeezing_acceptance.csv`) but does not defend: quantizing the attacked inputs
lowers accuracy further.

| Target | Clean | PGD ε=0.05 (baseline) | PGD ε=0.05 (8-bit squeezed) |
| :--- | ---: | ---: | ---: |
| Multiclass | 91.42% | 42.79% | 3.83% |
| Family | 97.81% | 48.34% | 37.33% |
| Binary | 99.19% | 98.90% | 97.34% |

Squeezing is evaluated **non-adaptively** (attacks generated on the undefended MLP,
then quantized), which is the conservative assumption for the defense. 4-bit
quantization fails the 1-point clean-accuracy gate for all nine pairs, 6-bit for seven
of nine; only 8-bit meets it.

## Threat Model & Constraints

The attacker controls a compromised IoT device and may change only sender-controlled
features (`src/features.py`): 10 of the 46 features are controllable (packet sizes
`Min`, `Max`, `AVG`, `Std`, `Tot size`; rates `Rate`, `Srate`; timings `IAT`,
`flow_duration`, `Number`), 25 are fixed (protocol identifiers, headers, flags,
victim-side `Drate`), and 11 are derived (`ack_count`-style counters, `Tot sum`,
`Magnitue`, `Radius`, `Covariance`, `Variance`, `Weight`). Modified values stay
non-negative, integral where required, within the observed range, and consistent with
`Min ≤ AVG ≤ Max`, `Min ≤ Tot size ≤ Max`. The detector is frozen; this is
feature-space evasion only.

## Setup

Python 3.12 with the pinned stack in `requirements.txt`; the CUDA build targets
PyTorch 2.14.1 + CUDA 13.0 (an RTX 3060 Laptop was used).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Extract the CICIoT2023 CSV captures into `data/raw/` as described in
`scripts/download_instructions.md`.

## Reproducing the Complete Pipeline

```powershell
.\scripts\run_all.ps1
```

```bash
./scripts/run_all.sh
```

| Step | Module | Outputs |
| :--- | :--- | :--- |
| 1. Load & split | `python -m src.data` | `data/processed/*.pkl`, `results/label_distribution.csv`, `results/test_class_counts.csv`, `results/split_summary.json` |
| 2. Train baselines | `python -m src.train` | `data/models/*`, `results/baselines_metrics.csv`, `results/baselines_per_class_recall.csv` |
| 3. Attacks | `python -m src.attacks` | `results/attack_metrics.csv`, `results/attack_per_class_recall.csv` |
| 4. Defenses | `python -m src.defenses` | `data/models/*_adv_*.pt`, `results/defense_metrics.csv`, `results/squeezing_acceptance.csv` |
| 5. Summaries | `python -m src.evaluate` | `results/attack_worst_case.csv`, `results/transfer_summary.csv`, `results/defense_matched_epsilon.csv`, `results/experiment_summary.json` |
| 6. Figures | `python -m src.plots` | `figures/*.png` (25 figures) |

The full write-up, including per-ε tables and per-class detail, is in
`report/report.md`.

## Limitations

1. **No capture-level holdout.** The dataset release used here already ships
   `train.csv`, `validation.csv`, and `test.csv`; each contains flows from a single
   capture file (`results/split_summary.json`). Those roles are adopted verbatim, so
   captures seen in training may also appear in test. Baselines are therefore
   potentially optimistic, and provenance of this pre-split layout still needs to be
   pinned down (`scripts/download_instructions.md`, TODO).
2. **Sample size.** 200,000 of the 7,845,673 available rows (140,000/30,000/30,000),
   with proportions preserved per file. Results describe this sample, not the full
   corpus.
3. **Rare classes.** 12 target/class combinations have fewer than 30 test samples
   (e.g. `Uploading_Attack` 2, `Recon-PingSweep` 2, `CommandInjection` 3,
   `Backdoor_Malware` 4, `XSS` 5, `BrowserHijacking` 6, `SqlInjection` 7,
   `DictionaryBruteForce` 9, `DDoS-HTTP_Flood` 18, `DDoS-SlowLoris` 24 in multiclass;
   `BruteForce` 9 and `Web` 27 in family). Classes are neither dropped nor merged;
   they are hatched in per-class figures and flagged in
   `results/baselines_per_class_recall.csv` and `results/attack_per_class_recall.csv`.
   Macro F1 is sensitive to them: `BruteForce` and `Web` both have recall 0.000 for the
   Random Forest, and their zero contribution alone removes up to 0.25 from the family
   macro average.
4. **Frozen derived features.** `Magnitue`, `Radius`, `Covariance`, `Variance`,
   `Weight`, and the flag counts are held constant in the constrained attack because
   their relation to per-packet padding has no closed form here. The constrained
   attacker is therefore weaker than a real one.
5. **Bidirectional aggregates.** CICIoT2023 features aggregate both directions, so
   size and timing features are only partly sender-controlled when the victim replies.
6. **Feature space, not packets.** Perturbations are applied to flow statistics, not
   to a live packet stream; MTU limits, retransmissions, and middlebox rewriting are
   not modelled.
7. **Squeezing is not adaptive.** The squeezing defense is probed with attacks
   generated on the undefended model and then quantized.
8. **Rejected variant, kept on record.** An 8-bit log1p-space quantizer was tried
   first and failed the 1-point gate by a wide margin, so per-feature quantile space
   was used instead.

## References

- Neto, E. C. P., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R., Ghorbani, A. A.
  *CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT
  environment.* Sensors 23(13):5941, 2023.
- Goodfellow, I. J., Shlens, J., Szegedy, C. *Explaining and harnessing adversarial
  examples.* ICLR 2015 (arXiv:1412.6572).
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., Vladu, A. *Towards deep learning
  models resistant to adversarial attacks.* ICLR 2018 (arXiv:1706.06083).
- Xu, W., Evans, D., Qi, Y. *Feature squeezing: Detecting adversarial examples in deep
  neural networks.* NDSS 2018.
- Chen, T., Guestrin, C. *XGBoost: A scalable tree boosting system.* KDD 2016,
  pp. 785–794.
- Breiman, L. *Random forests.* Machine Learning 45(1):5–32, 2001.
- Biggio, B., Corona, I., Maiorca, D., Nelson, B., Šrndić, N., Laskov, P., Giacinto, G.,
  Roli, F. *Evasion attacks against machine learning at test time.* ECML PKDD 2013,
  LNCS 8190, pp. 387–402.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

79 tests pass, covering the random-noise control, the evasion metric, the quantile
squeezer and its acceptance gate, class-count reporting, and the matched-ε defense
comparison.

## License

MIT. See `LICENSE`.
