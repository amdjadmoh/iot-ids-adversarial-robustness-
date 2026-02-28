# Evaluating the Adversarial Robustness of Machine Learning Intrusion Detectors in IoT Networks

**Author:** amdjadmoh
**Context:** Final-year Cybersecurity Engineering research project (PFE portfolio)
**Dataset:** CICIoT2023, 200,000-flow sample drawn from 7,845,673 rows
**Repository:** `iot-ids-adversarial`

Every figure in this report is generated from a file under `results/`; the code that
produced each file is named in the text.

---

## 1. Introduction and Motivation

Machine-learning network intrusion detection systems (NIDS) for IoT are usually
evaluated on clean traffic, where tree ensembles reach accuracy figures above 99% on
recent benchmarks. Those figures say nothing about an adversary who can choose the
traffic it sends. A compromised IoT device can pad payloads, slow transmission, and
insert idle time without changing the attack itself, so the question is not only
whether accuracy falls but whether the attack stops being detected.

This project measures both, on three detectors, across three label granularities, and
asks three questions:

- **RQ1** — How well do XGBoost, Random Forest, and an MLP classify IoT attack traffic
  on CICIoT2023 (34-class, 8-family, and binary settings)?
- **RQ2** — How much do accuracy and attack-class recall drop under (a) unconstrained
  FGSM/PGD on the MLP, (b) feature-constrained evasion using only attacker-controlled
  features, and (c) transfer of MLP adversarial examples to XGBoost and Random Forest?
  How many *detected attacks* become benign in each case?
- **RQ3** — Do adversarial training and feature squeezing restore robustness, and
  what is the clean-data cost?

---

## 2. Threat Model

### 2.1 Controllability

The attacker controls a compromised IoT device sending traffic toward a victim, and
the detector runs passively on the gateway. Of the 46 CICIoT2023 features
(`src/features.py`):

- **10 controllable** — `Min`, `Max`, `AVG`, `Std`, `Tot size`, `Rate`, `Srate`,
  `IAT`, `flow_duration`, `Number`. Payload padding changes sizes; delaying packets
  changes rates, inter-arrival times and flow duration.
- **25 fixed** — protocol identifiers and application ports (`Protocol Type`, `TCP`,
  `UDP`, `ICMP`, `HTTP`, `HTTPS`, `DNS`, `Telnet`, `SMTP`, `SSH`, `IRC`, `DHCP`,
  `ARP`, `IPv`, `LLC`), `Header_Length`, `Duration` (IP TTL), the seven TCP flag
  numbers, and `Drate`, which is the victim's inbound rate.
- **11 derived** — `ack_count`, `syn_count`, `fin_count`, `urg_count`, `rst_count`,
  `Tot sum`, `Magnitue`, `Radius`, `Covariance`, `Variance`, `Weight`. `Tot sum` is
  recomputed as `Number × AVG`; the rest are frozen during constrained evasion because
  their dependence on per-packet padding has no closed form here.

Because CICIoT2023 aggregates both directions of a flow, size and timing features are
only partly sender-controlled whenever the victim replies (Section 6).

### 2.2 Validity constraints

Perturbed values must stay non-negative, integral where the raw feature is integral,
inside the observed range, and internally consistent: `Min ≤ AVG ≤ Max`,
`Min ≤ Tot size ≤ Max`, and rates/`IAT`/`Number`/`flow_duration` mutually consistent.
Sizes may only grow, rates may only fall, and timings may only lengthen. The detector
is frozen during evaluation; this is feature-space evasion, not poisoning and not
packet-level crafting.

---

## 3. Methodology

### 3.1 Data and splits

The release used here ships three files — `train.csv`, `validation.csv`, `test.csv` —
containing 7,845,673 flows in total. Each file corresponds to a single capture
(`results/split_summary.json` records `capture_files: 1` per split). The loaders adopt
those roles verbatim; **no capture-level holdout is performed**, so flows from the same
capture can appear in training and test. Provenance of this pre-split layout is still
to be documented (`scripts/download_instructions.md` carries a TODO), and the effect
on absolute accuracy is stated in Section 6.

From those files, 200,000 rows are sampled proportionally per file (seed 42), giving
140,000 training, 30,000 validation and 30,000 test flows with all 34 labels present.
Per-split counts are written to `results/test_class_counts.csv`; 12 target/class
combinations have fewer than 30 test samples and are flagged there and in the per-class
recall files.

### 3.2 Models

- **Random Forest** — 300 trees, `min_samples_leaf=2`, raw features.
- **XGBoost** — 400 trees, depth 6, learning rate 0.1, subsample 0.9, colsample 0.9,
  raw features.
- **MLP** — two hidden layers (128, 64), dropout 0.2, standardized inputs (training
  medians and standard deviations, clipped to ±8), AdamW at 1e-3 with weight decay
  1e-4, batch 512, up to 30 epochs, best validation macro F1 retained. Torch 2.14.1
  with CUDA on an RTX 3060 Laptop.

Trees consume raw features and the MLP consumes scaled features; preprocessing is fit
on the training split only.

### 3.3 Attacks and metrics

- **FGSM** and **PGD** (20 steps, step 0.25ε, random start) on the MLP at
  ε ∈ {0.005, 0.01, 0.02, 0.05, 0.10} in scaled space.
- **Constrained PGD** — the same gradient loop, projected back onto raw feature units
  and then onto the Section 2.2 constraints, at ε ∈ {0.01, 0.02, 0.05, 0.10}.
- **Random-noise control** — uniform random signs × ε through the same clipping path as
  FGSM/PGD, seeded from the config (seed 42), at the same five budgets
  (`results/attack_manifest.json`, `random_noise_seed`). It is evaluated on every
  model and target, and all transfer statements are made against it.
- **Transfer** — examples generated on the MLP are passed to the frozen trees.

Reported metrics (`src/evaluate.py`): accuracy, macro recall/F1, attack success rate
(clean-correct samples now wrong), and **evasion rate** — among rows that are true
attacks *and* classified as attacks before perturbation, the fraction predicted benign
afterwards (benign = `BenignTraffic` in multiclass, `Benign` in family and binary). The
evasion rate is the primary RQ2 metric because a wrong attack label still raises an
alert.

### 3.4 Defenses

- **Adversarial training** — the MLP is retrained from scratch on a 50/50 mix of clean
  and FGSM-perturbed samples, ε_train ∈ {0.01, 0.02}, 20 epochs. Attacks against the
  defended model are **adaptive**: every attack is regenerated against that network
  (`results/defense_manifest.json`,
  `adversarial_training_attack_generation`), so no stale examples are reused.
- **Feature squeezing** — per-feature quantile-space quantization at 8, 6 and 4 bits,
  with bin edges fitted on the training split only. Squeezing is probed
  **non-adaptively**: attacks are generated on the undefended MLP and then quantized
  (`feature_squeezing_attack_generation`). A hard acceptance gate
  (`results/squeezing_acceptance.csv`) requires 8-bit clean accuracy to stay within one
  point of the unsqueezed model; the pipeline fails if it does not. An earlier
  log1p-space quantizer failed that gate and was discarded.

---

## 4. Results

### 4.1 RQ1 — clean baselines

**Table 1** (`results/baseline_test_summary.csv`, 30,000 test flows):

| Target | Model | Accuracy | Macro F1 | Weighted F1 | Macro recall |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Multiclass (34) | Random Forest | 99.01% | 0.7054 | 0.9890 | 0.6992 |
| Multiclass (34) | XGBoost | **99.06%** | 0.6947 | **0.9897** | 0.6928 |
| Multiclass (34) | MLP | 91.51% | 0.5937 | 0.9114 | 0.5944 |
| Family (8) | Random Forest | **99.31%** | 0.6851 | **0.9925** | 0.6859 |
| Family (8) | XGBoost | 99.24% | **0.6937** | 0.9919 | 0.6869 |
| Family (8) | MLP | 98.13% | 0.6328 | 0.9806 | 0.6324 |
| Binary | Random Forest | **99.62%** | **0.9578** | **0.9962** | 0.9559 |
| Binary | XGBoost | 99.56% | 0.9523 | 0.9956 | **0.9605** |
| Binary | MLP | 99.22% | 0.9152 | 0.9922 | 0.9236 |

Accuracy is near-ceiling, but macro F1 separates the models. Two effects drive the
0.69–0.71 multiclass macro F1 of the trees and the 0.59 of the MLP, both visible in
`results/baselines_per_class_recall.csv`:

- **Rare labels.** Ten multiclass classes have fewer than 30 test samples (2–24 each),
  and eight of them have recall 0.000 for at least one model (`Uploading_Attack` 2,
  `Recon-PingSweep` 2, `CommandInjection` 3, `Backdoor_Malware` 4, `XSS` 5,
  `BrowserHijacking` 6, `SqlInjection` 7, `DictionaryBruteForce` 9). A handful of
  samples cannot support a class, and that is a property of the sample, not of the
  attack; `DDoS-HTTP_Flood` (18) and `DDoS-SlowLoris` (24) are the two low-support
  labels that are still classified.
- **Genuinely confusable classes with adequate support.** MLP recall is 0.000 on
  `Recon-OSScan` (70 test samples) and 0.220 on `DNS_Spoofing` (109), and 0.508 on
  `Recon-PortScan` (63) — reconnaissance traffic that resembles benign scanning.

In the family task the macro F1 (0.685–0.694) is dominated by the two rare families:
`BruteForce` (9 test samples) and `Web` (27) have recall 0.000 under the Random Forest,
which removes up to 0.25 from the macro average while accuracy stays at 99.31%.

### 4.2 RQ2 — evasion, transfer, and the noise control

#### 4.2.1 Evasion rate

**Table 2** — maximum evasion rate over the tested budgets
(`results/attack_worst_case.csv`, column `max_evasion_rate`):

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

No configuration exceeds 0.5%. The largest single value is 0.50% (Random Forest,
multiclass, FGSM). The evasion population is 29,173 (MLP), 29,251 (Random Forest) and
29,226 (XGBoost) detected attack flows out of 30,000 test flows, so these percentages
are not computed on a small denominator.

#### 4.2.2 Accuracy under attack

**Table 3** — accuracy at ε = 0.10 (`results/attack_metrics.csv`):

| Target | Model | Clean | Constrained | FGSM | PGD | Noise |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Multiclass | Random Forest | 99.01% | 81.90% | 36.74% | 29.16% | 45.09% |
| Multiclass | XGBoost | 99.06% | 87.62% | 19.98% | 16.96% | 34.18% |
| Multiclass | MLP | 91.51% | 42.65% | 42.62% | 41.92% | 66.07% |
| Family | Random Forest | 99.31% | 83.19% | 10.55% | 25.88% | 58.24% |
| Family | XGBoost | 99.24% | 82.60% | 9.86% | 9.41% | 47.37% |
| Family | MLP | 98.13% | 45.62% | 44.23% | 48.06% | 72.14% |
| Binary | Random Forest | 99.62% | 98.40% | 97.83% | 97.82% | 97.86% |
| Binary | XGBoost | 99.56% | 98.67% | 97.84% | 97.82% | 97.91% |
| Binary | MLP | 99.22% | 99.12% | 98.57% | 98.53% | 99.21% |

The MLP loses 49.59 points on multiclass under PGD at this budget
(attack success rate 0.5420), yet only 0.25% of its detected attacks become benign. The
damage is label confusion, not detection loss: at ε = 0.10 the multiclass MLP's recall
on `DDoS-SYN_Flood` falls from 0.911 to 0.004, on `DDoS-SynonymousIP_Flood` from 0.978
to 0.001, on `DDoS-TCP_Flood` from 0.971 to 0.001 and on `DDoS-UDP_Flood` from 0.903 to
0.000 (`results/attack_per_class_recall.csv`) — those flows are still classified as
attacks, just as the wrong one. That is a real operational problem for automated
response and attack attribution, but it is not evasion.

#### 4.2.3 Transfer is mostly input fragility

**Table 4** — ε = 0.10, MLP-generated examples on frozen trees
(`results/transfer_summary.csv`):

| Target | Model | Clean | PGD | Random noise | Extra drop vs noise |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Multiclass | Random Forest | 99.01% | 29.16% | 45.09% | 15.94 pts |
| Multiclass | XGBoost | 99.06% | 16.96% | 34.18% | 17.22 pts |
| Family | Random Forest | 99.31% | 25.88% | 58.24% | 32.36 pts |
| Family | XGBoost | 99.24% | 9.41% | 47.37% | 37.96 pts |
| Multiclass | MLP (own attack) | 91.51% | 41.92% | 66.07% | 24.16 pts |
| Family | MLP (own attack) | 98.13% | 48.06% | 72.14% | 24.08 pts |

Random noise of the same magnitude already costs the trees 41–65 points. The
gradient-specific part of the drop is therefore 15.9–38.0 points, not the 70–90 points
the raw numbers suggest. Reporting the raw drop as "transfer" would overstate the
effect by two to four times.

#### 4.2.4 Realistic constrained evasion barely works

Restricting changes to the 10 controllable features and enforcing Section 2.2 leaves
evasion at 0.03–0.13% for multiclass and family, and 0.01–0.04% for binary. The trees
end up *more* accurate under the constrained attack than under random noise at the same
budget: multiclass Random Forest 81.90% versus 45.09%, XGBoost 87.62% versus 34.18%;
family 83.19% versus 58.24% and 82.60% versus 47.37%. The corresponding
`accuracy_drop_vs_noise` values are negative at ε = 0.10 (between −24.95 and −53.44
points), so the
constrained attacker is weaker than an attacker who simply adds noise.

The MLP is the exception in accuracy terms — 42.65% on multiclass — because it is
fragile to any input shift, yet its constrained evasion is 0.06%. In other words, within
the part of the feature space an IoT device can actually reach, evasion of these
detectors was not achieved in this study. That is a negative result, and it is the most
important one here: the large unconstrained drops do not survive contact with the
threat model.

#### 4.2.5 Binary accuracy loss is false alarms

Under PGD at ε = 0.10 the binary Random Forest's benign recall falls from 91.37% to
7.91% while its attack recall is unchanged (99.82% → 99.95%); XGBoost goes from 92.37%
to 10.22% benign recall (attack recall 99.73% → 99.89%); the MLP goes from 85.18% to
71.65% (attack recall 99.55% → 99.16%). With only two classes, and with evasion of
detected attacks at or below 0.39%, the accuracy drop is almost entirely benign traffic
being pushed into the attack class. The operational reading is alert flooding, not
undetected intrusion — a denial-of-service effect on the analyst rather than on the
classifier's recall.

### 4.3 RQ3 — defenses

#### 4.3.1 Adversarial training at matching ε

**Tables 5–7** compare the baseline MLP and the adversarially-trained MLPs under PGD at
the *same* ε (`results/defense_matched_epsilon.csv`), with the clean column included so
the cost is visible. Attacks on the defended models are regenerated against those
models (adaptive).

**Table 5 — multiclass, PGD:**

| Model | Clean | ε=0.005 | ε=0.01 | ε=0.02 | ε=0.05 | ε=0.10 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 91.51% | 64.49% | 51.30% | 43.47% | 42.79% | 41.92% |
| Adv. train ε=0.01 | 81.21% | 80.52% | 80.15% | 79.04% | 72.36% | 57.08% |
| Adv. train ε=0.02 | 80.40% | 80.20% | 80.00% | 79.67% | 72.81% | 66.71% |

**Table 6 — family, PGD:**

| Model | Clean | ε=0.005 | ε=0.01 | ε=0.02 | ε=0.05 | ε=0.10 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 98.13% | 55.74% | 53.78% | 46.57% | 48.34% | 48.06% |
| Adv. train ε=0.01 | 86.13% | 84.43% | 83.62% | 82.31% | 77.92% | 69.69% |
| Adv. train ε=0.02 | 85.75% | 84.10% | 83.79% | 82.96% | 81.01% | 74.22% |

**Table 7 — binary, PGD:**

| Model | Clean | ε=0.005 | ε=0.01 | ε=0.02 | ε=0.05 | ε=0.10 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 99.22% | 99.18% | 99.16% | 99.11% | 98.90% | 98.53% |
| Adv. train ε=0.01 | 99.18% | 99.16% | 99.14% | 99.08% | 98.93% | 98.66% |
| Adv. train ε=0.02 | 99.19% | 99.17% | 99.14% | 99.12% | 98.97% | 98.72% |

Adversarial training is the only defense here that works. On multiclass it buys
+29.58 points at ε = 0.05 (ε_train = 0.01) and +24.79 at ε = 0.10 (ε_train = 0.02) for a
clean cost of 10.29–11.10 points. On family the gains are +29.59 to +32.68 points at
ε = 0.05 for 12.00–12.38 points of clean accuracy. On binary there is almost nothing to
gain (+0.03 to +0.19 points) because the baseline is already robust.

#### 4.3.2 Feature squeezing

**Table 8** — the 8-bit acceptance gate
(`results/squeezing_acceptance.csv`), clean accuracy before and after quantization:

| Target | Model | Clean | 8-bit squeezed | Drop |
| :--- | :--- | ---: | ---: | ---: |
| Multiclass | Random Forest | 99.01% | 98.79% | 0.22 pts |
| Multiclass | XGBoost | 99.06% | 99.04% | 0.02 pts |
| Multiclass | MLP | 91.51% | 91.42% | 0.09 pts |
| Family | Random Forest | 99.31% | 99.10% | 0.21 pts |
| Family | XGBoost | 99.24% | 99.23% | 0.02 pts |
| Family | MLP | 98.13% | 97.81% | 0.32 pts |
| Binary | Random Forest | 99.62% | 99.50% | 0.12 pts |
| Binary | XGBoost | 99.56% | 99.56% | 0.00 pts |
| Binary | MLP | 99.22% | 99.19% | 0.03 pts |

All nine 8-bit rows pass the 1-point tolerance. The gate is not met at lower depths:
4-bit fails for all nine pairs and 6-bit fails for seven of nine (the exceptions are the
multiclass and binary MLP), so 4-bit and 6-bit numbers are reported but not treated as
usable configurations.

**Table 9** — MLP accuracy on the same PGD examples, before and after quantization
(`results/defense_matched_epsilon.csv`, `defense = feature_squeezing`):

| Target | Variant | Clean | ε=0.005 | ε=0.05 | ε=0.10 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Multiclass | Unsqueezed | 91.51% | 64.49% | 42.79% | 41.92% |
| Multiclass | 8-bit | 91.42% | 23.26% | 3.83% | 3.62% |
| Multiclass | 6-bit | 90.77% | 21.91% | 5.54% | 2.77% |
| Multiclass | 4-bit | 85.13% | 12.52% | 12.75% | 1.51% |
| Family | Unsqueezed | 98.13% | 55.74% | 48.34% | 48.06% |
| Family | 8-bit | 97.81% | 64.89% | 37.33% | 37.57% |
| Family | 6-bit | 96.48% | 64.22% | 28.14% | 38.45% |
| Family | 4-bit | 93.70% | 50.72% | 31.10% | 20.91% |
| Binary | Unsqueezed | 99.22% | 99.18% | 98.90% | 98.53% |
| Binary | 8-bit | 99.19% | 97.57% | 97.34% | 97.07% |
| Binary | 6-bit | 98.81% | 97.34% | 97.09% | 96.86% |
| Binary | 4-bit | 97.73% | 97.46% | 97.38% | 97.28% |

Squeezing does not defend. On multiclass it destroys accuracy on attacked inputs
(3.83% at ε = 0.05 against 42.79% unsqueezed) while leaving clean accuracy intact — the
quantizer discards exactly the fine-grained magnitude information the MLP uses, and the
perturbation survives quantization. On family there is a small gain at ε = 0.005
(64.89% versus 55.74%) that reverses by ε = 0.05 (37.33% versus 48.34%). On binary it
costs 1.5–1.8 points at every budget. Since the evaluation is non-adaptive, a real
attacker who knew about the quantizer could do no worse.

---

## 5. Discussion

Three points follow from the numbers.

1. **Accuracy is a misleading robustness metric for multiclass NIDS.** A 49.59-point
   accuracy drop coexists with a 0.25% evasion rate. Benchmarks that report only
   accuracy conflate "the alarm named the wrong attack" with "the attack passed
   undetected"; the second is what matters for a detector's core function, and it stays
   at one quarter of one percent here.
2. **Most of the transfer effect is not transfer.** Random-sign noise at the same
   budget reproduces 56–79% of the tree accuracy loss. Any transfer claim needs a noise
   control, otherwise energy that any perturbation would inject is attributed to the
   gradient.
3. **Within the threat model, evasion was not achieved.** The constrained attacker —
   the only one a real IoT device can implement — reaches at best 0.13% evasion and, for
   trees, is less damaging than noise. The interesting failure mode it does produce is
   false alarms on benign traffic and misattribution among attacks.

---

## 6. Limitations

1. **No capture-level holdout.** `train.csv`, `validation.csv` and `test.csv` in the
   release each contain flows from a single capture (`results/split_summary.json`).
   They are used as shipped, so training and test flows can originate from the same
   capture. This is the standard caveat for pre-split releases and it makes the clean
   baselines, and therefore every drop measured against them, **potentially
   optimistic**. A capture-level re-split is the obvious next check; the provenance of
   the pre-split layout is still a TODO in `scripts/download_instructions.md`.
2. **Sample size.** 200,000 flows out of 7,845,673. Sampling is proportional per file,
   so the label mix is preserved but absolute counts for rare attacks are small.
3. **Rare classes.** 12 target/class combinations have fewer than 30 test samples
   (10 multiclass, 2 family). They are neither dropped nor merged; they are hatched in
   `figures/per_class_recall_*.png` and flagged with `low_test_support` in the result
   CSVs. Macro F1 is highly sensitive to them — `BruteForce` and `Web` at recall 0.000
   remove up to 0.25 from the family macro average — and per-class recall for those
   labels should be read as an anecdote, not an estimate.
4. **Frozen derived features.** `Magnitue`, `Radius`, `Covariance`, `Variance`,
   `Weight` and the flag counts are held constant in the constrained attack. A real
   attacker who measures victim responses could move them, so the constrained results
   understate what is physically possible.
5. **Bidirectional aggregates.** CICIoT2023 features combine both directions; with a
   replying victim, the sender controls less of `Std`, `AVG`, `Rate` and `IAT` than the
   controllability table implies.
6. **Feature space only.** Perturbations are applied to flow statistics, not to emitted
   packets. MTU limits, TCP retransmission, and middlebox rewriting are not modelled,
   and no live testbed was used.
7. **Squeezing is evaluated non-adaptively**, which is generous to the defense; the
   negative result would only get worse under an adaptive attacker.
8. **Single seed per configuration.** Attacks and training use seed 42. Variance across
   seeds was not measured, so differences below roughly one point should not be
   interpreted.

---

## 7. Conclusion

On a 200,000-flow sample of CICIoT2023, all three detectors reach 99%+ accuracy and
only 0.59–0.71 multiclass macro F1, the latter driven by rare classes. Unconstrained
gradient attacks move accuracy a long way — 49.59 points on the multiclass MLP, 82.10
on XGBoost — but the fraction of detected attacks that become benign never exceeds
0.5%, and the binary accuracy loss is dominated by false alarms on benign traffic
rather than by missed attacks. After controlling for random noise, genuine transfer to
the trees is 15.9–38.0 points rather than the 70–90 points the raw numbers imply.
Feature-constrained evasion, the only variant a real IoT device can carry out, achieves
at most 0.13% evasion. Adversarial training restores 24.79–32.68 points of adversarial
accuracy at a cost of 10.29–12.38 clean points and is the only defense that works;
quantile-space feature squeezing preserves clean accuracy at 8 bits but degrades
accuracy on attacked inputs.

### Suggested extensions

- **Capture-level re-split** of the raw release to test whether the clean baselines move.
- **Live packet crafting** (Scapy or an SDN testbed) to verify that the constrained
  feature deltas can be realised on the wire and that the detector sees what the
  feature-space simulation assumes.
- **Detection-preserving metrics**: report "detected vs undetected" separately from
  "correctly attributed", and track benign false-alarm inflation under attack.
- **Adaptive squeezing** and randomized smoothing, to test whether the defense failure
  reported here is intrinsic to tabular flow features.

---

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

---

## Reproducing

```powershell
.\scripts\run_all.ps1        # data → train → attacks → defenses → evaluate → plots
.\.venv\Scripts\python.exe -m pytest
```

Result files quoted above: `baseline_test_summary.csv`, `attack_metrics.csv`,
`attack_worst_case.csv`, `attack_per_class_recall.csv`, `baselines_per_class_recall.csv`,
`transfer_summary.csv`, `defense_metrics.csv`, `defense_matched_epsilon.csv`,
`squeezing_acceptance.csv`, `test_class_counts.csv`, `split_summary.json`,
`attack_manifest.json`, `defense_manifest.json` (all under `results/`).
