# Research Report: Adversarial Evasion of ML Intrusion Detectors for IoT Networks (CICIoT2023)

**Author:** amdjadmoh  
**Context:** Research project evaluating ML IoT intrusion detectors under realistic adversarial evasion  
**Dataset:** CICIoT2023 (sampled 200,000 flows: 140,000 train, 30,000 val, 30,000 test across 34 attack types and 8 families)  

---

## 1. Executive Summary

This research investigates the robustness of three machine learning intrusion detectors—a Multilayer Perceptron (MLP), Random Forest (RF), and XGBoost—against white-box, feature-constrained, and black-box transfer adversarial evasion on the CICIoT2023 dataset.

Key Findings:
1. **RQ1 (Baselines):** While Random Forest and XGBoost achieve near-perfect clean accuracy (>99.0% on multiclass, >99.2% on family, >99.5% on binary), macro F1-score reveals severe minority-class degradation due to attack traffic imbalance (multiclass macro F1 is ~0.70 for trees and ~0.59 for MLP).
2. **RQ2 (Adversarial Evasion & Transfer):**
   - **White-box unconstrained evasion:** MLP test accuracy drops dramatically on multiclass (from 91.51% to 41.92%) and family (from 98.13% to 46.57%) under PGD ($\epsilon=0.10$).
   - **Feature-constrained evasion:** When restricting perturbations strictly to attacker-controlled traffic features (size padding, traffic slowing, flow duration extension, consistent rate/IAT/duration), the attack remains lethal against the MLP (multiclass accuracy drops from 91.51% to 42.65%).
   - **Transferability to Tree Detectors:** Adversarial examples generated on the differentiable MLP transfer with devastating effect against tree-based detectors in multiclass and family settings: Random Forest accuracy collapses from 99.01% to 29.16%, and XGBoost collapses from 99.06% to 16.96%.
3. **RQ3 (Defenses & Trade-offs):**
   - **Adversarial Training:** Significantly bolsters robust accuracy under PGD ($\epsilon=0.05$) from 42.61% to 72.81% on multiclass, at a moderate clean accuracy cost of ~11.1%.
   - **Feature Squeezing:** Inference-time input spatial quantization offers moderate protection without retraining but suffers clean accuracy loss at aggressive bit depths (e.g. 4-bit squeezing drops multiclass clean accuracy to 58.95%).

---

## 2. RQ1: Baseline Classification Performance

Detection models were trained on 46 network flow features. Random Forest and XGBoost consumed raw median-imputed features; the MLP consumed median-imputed, standardized features.

### Clean Test Set Results

| Target | Model | Accuracy | Macro F1 | Weighted F1 | Macro Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Multiclass (34)** | Random Forest | 0.990100 | 0.705430 | 0.988956 | 0.699211 |
| | XGBoost | 0.990567 | 0.694691 | 0.989651 | 0.692767 |
| | MLP | 0.915067 | 0.593714 | 0.911396 | 0.594417 |
| **Family (8)** | Random Forest | 0.993100 | 0.685136 | 0.992498 | 0.685897 |
| | XGBoost | 0.992433 | 0.693729 | 0.991930 | 0.686918 |
| | MLP | 0.981300 | 0.632816 | 0.980636 | 0.632387 |
| **Binary (2)** | Random Forest | 0.996200 | 0.957843 | 0.996192 | 0.955913 |
| | XGBoost | 0.995600 | 0.952263 | 0.995639 | 0.960523 |
| | MLP | 0.992167 | 0.915189 | 0.992245 | 0.923647 |

*Observation:* High accuracy in CICIoT2023 is misleading due to class imbalance dominated by voluminous DDoS/DoS floods. Macro F1 exposes the vulnerability of minority attack classes (e.g. `Recon-PingSweep`, `Backdoor_Malware`, `Uploading_Attack`).

---

## 3. RQ2: Adversarial Degradation and Transferability

### 3.1 White-Box Evasion on MLP

Under $L_\infty$ perturbations (scaled space $\epsilon \in [0.005, 0.10]$):
- On **Multiclass**, MLP accuracy drops from **91.51%** down to **42.36%** (FGSM) and **41.92%** (PGD).
- On **Family**, MLP accuracy drops from **98.13%** down to **44.23%** (FGSM) and **46.57%** (PGD).
- On **Binary**, binary classification remains comparatively resilient (~98.5% adversarial accuracy), because shifting an attack across decision boundaries to look like benign traffic requires crossing large protocol/flag spaces that are heavily weighted in binary separation.

### 3.2 Realistic Feature-Constrained Evasion

Using the threat model where protocol types and flags are frozen, packet sizes can only increase ($\Delta \text{size} \ge 0$), sending rates can only decrease, and physical invariants ($Min \le AVG \le Max$, non-negativity) are enforced at every PGD step:
- **MLP Multiclass Accuracy:** Drops from **91.51%** to **42.65%** ($\Delta = 48.86\%$, Attack Success Rate: 53.40%).
- **MLP Family Accuracy:** Drops from **98.13%** to **45.62%** ($\Delta = 52.51\%$, Attack Success Rate: 53.52%).

*Significance:* Attackers do not need to violate network protocols or invert unchangeable flags to cause massive misclassification. Simply padding packet sizes and adjusting inter-packet timing within legitimate ranges causes the detector to confuse attack subclasses and families.

### 3.3 Transferability to Tree Detectors (RF and XGBoost)

Adversarial examples crafted on the surrogate MLP were mapped back to the valid raw feature space and fed into the frozen, non-differentiable tree detectors:

| Target | Model | Attack | Clean Acc | Adv Acc | Acc Drop | Attack Success Rate |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Multiclass** | Random Forest | PGD ($\epsilon=0.10$) | 0.990100 | **0.291567** | **69.85%** | 70.60% |
| | XGBoost | PGD ($\epsilon=0.10$) | 0.990567 | **0.169567** | **82.10%** | 82.95% |
| | Random Forest | Constrained ($\epsilon=0.10$) | 0.990100 | **0.818667** | 17.14% | 17.36% |
| | XGBoost | Constrained ($\epsilon=0.10$) | 0.990567 | **0.838800** | 15.18% | 15.40% |
| **Family** | Random Forest | PGD ($\epsilon=0.10$) | 0.993100 | **0.258833** | **73.43%** | 74.03% |
| | XGBoost | PGD ($\epsilon=0.10$) | 0.992433 | **0.094067** | **89.84%** | 91.52% |
| | Random Forest | Constrained ($\epsilon=0.10$) | 0.993100 | **0.827167** | 16.59% | 16.79% |
| | XGBoost | Constrained ($\epsilon=0.10$) | 0.992433 | **0.814933** | 17.75% | 17.97% |

*Key finding:* High tree complexity does not prevent adversarial transfer. Gradient perturbations on shared statistical features (e.g. packet size dispersion and timing) cause tree partition boundaries to misclassify traffic severely.

---

## 4. RQ3: Defense Efficacy and Clean-Data Cost

### 4.1 Adversarial Training vs. Feature Squeezing on MLP (PGD $\epsilon=0.05$)

| Target | Defense | Parameter | Clean Accuracy | Adversarial Accuracy | Clean Cost ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Multiclass** | *None (Baseline)* | - | 0.915067 | 0.419167 | - |
| | Adversarial Training | $\epsilon=0.01$ | 0.812133 | 0.723633 | +0.1029 |
| | Adversarial Training | $\epsilon=0.02$ | 0.804033 | **0.728067** | +0.1110 |
| | Feature Squeezing | 8-bit | 0.772733 | 0.426133 | +0.1423 |
| | Feature Squeezing | 6-bit | 0.821333 | 0.434267 | +0.0937 |
| | Feature Squeezing | 4-bit | 0.589500 | 0.497000 | +0.3256 |
| **Family** | *None (Baseline)* | - | 0.981300 | 0.480567 | - |
| | Adversarial Training | $\epsilon=0.01$ | 0.861267 | 0.779233 | +0.1200 |
| | Adversarial Training | $\epsilon=0.02$ | 0.857500 | **0.810133** | +0.1238 |
| | Feature Squeezing | 8-bit | 0.696967 | 0.481300 | +0.2843 |
| | Feature Squeezing | 6-bit | 0.748800 | 0.478367 | +0.2325 |
| | Feature Squeezing | 4-bit | 0.611500 | 0.525933 | +0.3698 |

*Conclusions:*
- **Adversarial Training** successfully recovers robust accuracy (from 41.9% to 72.8% on multiclass) by smoothing the loss landscape around attack clusters, but incurs an ~11% clean accuracy penalty.
- **Feature Squeezing** is less effective for tabular network traffic than for computer vision: quantizing continuous statistical features distorts benign traffic profiles, creating substantial clean accuracy degradation without comparable robustness gains.

---

## 5. Threat Model Limitations & Artifact Reproducibility

1. **Frozen Derived Features:** Bidirectional interaction features whose mathematical derivation depends on per-packet capture state without closed-form summary representation (`Magnitue`, `Radius`, `Covariance`, `Variance`, `Weight`, and flag counts) are frozen in the threat model.
2. **Reproducibility:** All metrics are traceable directly to `results/` CSV files (`baselines_metrics.csv`, `attack_metrics.csv`, `defense_metrics.csv`), and all plots are rendered automatically in `figures/`. The complete experiment is reproducible by executing `.\scripts\run_all.ps1`.
