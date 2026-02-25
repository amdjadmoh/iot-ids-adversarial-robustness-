# Evaluating the Adversarial Robustness of Machine Learning Intrusion Detectors in IoT Networks

**Author:** amdjadmoh  
**Context:** Final-Year Cybersecurity Engineering Research Project (PFE Portfolio)  
**Dataset:** CICIoT2023 (Sampled 200,000 network flows from 7.8M raw records)  
**Code Repository:** `iot-ids-adversarial`  

---

## 1. Introduction and Motivation

Network Intrusion Detection Systems (NIDS) based on machine learning are widely proposed in academic literature to protect Internet of Things (IoT) ecosystems. Because IoT devices operate under stringent compute and battery constraints, host-based security agents are often infeasible, making centralized or edge-based network traffic monitoring the primary line of defense. Standard benchmarks frequently report near-perfect detection accuracy—often exceeding 99%—for algorithms such as Random Forests, Gradient Boosted Trees, and Deep Neural Networks.

However, in real-world deployments, these headline accuracy metrics can create a false sense of security. First, network datasets are heavily skewed toward high-volume attacks (such as volumetric SYN or UDP floods), meaning that overall accuracy reflects performance on repetitive flood patterns while masking poor recall on stealthy, minority-class intrusions. Second, standard evaluations typically assume a static adversary who transmits attack traffic unchanged. In an adversarial setting, a malicious actor operating a compromised IoT device can modify the traffic characteristics—such as padding packet payloads or altering sending intervals—to evade detection without compromising the attack's operational objective.

In this project, I conduct an empirical study of the adversarial robustness of three representative machine learning classifiers on the recent CICIoT2023 benchmark:
1. A multi-layer perceptron (MLP) trained on scaled continuous features.
2. A Random Forest (RF) classifier operating on raw flow statistics.
3. An Extreme Gradient Boosting (XGBoost) model operating on raw flow statistics.

I investigate three central research questions:
- **RQ1 (Baseline Performance):** How effectively do these detectors classify attack traffic across 34 specific attack classes, 8 high-level attack families, and a binary benign-versus-malicious task?
- **RQ2 (Evasion and Transferability):** How resilient are these models against (a) unconstrained gradient-based attacks on the neural surrogate, (b) realistic feature-constrained evasion adhering to network physics, and (c) black-box transfer attacks targeting the non-differentiable tree ensembles?
- **RQ3 (Defenses and Trade-offs):** Can adversarial training or feature squeezing restore detection robustness, and what cost do these defenses impose on clean-data classification accuracy?

---

## 2. Threat Model and Constraint Formulation

A key critique of adversarial machine learning in network security is that gradient-based attacks often produce "adversarial examples" that are physically impossible in real networks—such as negative packet counts, modified protocol flags that break connection handshakes, or forged victim response metrics. To ensure practical relevance, I formulated a strict, domain-consistent threat model.

### 2.1 Attacker Capabilities and Feature Controllability
The adversary is assumed to control a compromised IoT device within the local network and transmits malicious traffic toward a target. The detector is deployed passively on a network gateway and monitors bidirectional flow statistics.

Under this setup, the 46 features extracted in CICIoT2023 are divided into three categories:

1. **Attacker-Controllable Features (10 features):**
   - *Packet Sizes:* `Min`, `Max`, `AVG`, `Std`, `Tot size`. An attacker can manipulate payload sizes by appending arbitrary dummy bytes (padding).
   - *Traffic Rates and Timing:* `Rate`, `Srate`, `flow_duration`, `IAT`, `Number`. The attacker can slow down transmission, insert idle delays between consecutive packets, or extend the total flow duration.

2. **Fixed Features (25 features):**
   - *Protocol Identifiers:* `Protocol Type`, `TCP`, `UDP`, `ICMP`, `HTTP`, `HTTPS`, `DNS`, `Telnet`, `SMTP`, `SSH`, `IRC`, `DHCP`, `ARP`, `IPv`, `LLC`. The transport and application protocol identities cannot change without altering the core functional behavior of the attack.
   - *Flags and Headers:* `Header_Length`, `Duration` (IP TTL), and flag indicators (`fin_flag_number`, `syn_flag_number`, `rst_flag_number`, `psh_flag_number`, `ack_flag_number`, `ece_flag_number`, `cwr_flag_number`). Altering TCP control flags invalidates stateful connection tracking.
   - *Victim-Side Metrics:* `Drate` represents the destination/victim packet response rate, which is outside the sender's direct control.

3. **Derived and Interaction Features (11 features):**
   - *Recomputed Features:* `Tot sum` represents the total volume of bytes in the flow and is mathematically recomputed during perturbation as $\text{Number} \times \text{AVG}$.
   - *Frozen Derived Features:* Features such as `Magnitue`, `Radius`, `Covariance`, `Variance`, `Weight`, and specific flag counts (`ack_count`, `syn_count`, etc.) depend on complex bidirectional interactions or per-packet rolling windows. In the constrained attack, these features are frozen at their original values, which acts as a conservative bound on attacker capabilities and is documented as a methodological constraint.

### 2.2 Directional Rules and Physical Invariants
Beyond feature classification, network physics impose strict directional bounds on how features may change:
- **Monotonic Size Increases:** An attacker can add padding to packets, but cannot reduce a packet below its minimum functional payload size. Thus, $\text{Size}_{\text{adv}} \ge \text{Size}_{\text{clean}}$.
- **Monotonic Rate Reductions:** Transmission rates are bounded by the device's physical network interface. An attacker can deliberately delay packets to decrease sending rates, but cannot arbitrarily exceed interface bandwidth. Hence, $\text{Rate}_{\text{adv}} \le \text{Rate}_{\text{clean}}$, while inter-arrival time ($\text{IAT}$) and $\text{flow\_duration}$ can only increase.
- **Physical Invariants:** All flow measurements must remain strictly non-negative ($\ge 0$). In addition, size relations must maintain internal mathematical consistency: $\text{Min} \le \text{AVG} \le \text{Max}$ and $\text{Min} \le \text{Tot size} \le \text{Max}$.

---

## 3. Experimental Methodology and Implementation

### 3.1 Dataset Preprocessing and Split Design
The experimental pipeline was evaluated on the official CICIoT2023 dataset. The raw dataset contains 169 capture files totaling over 7.8 million network flow records across 34 categories (33 attack types and 1 benign class).

Processing the complete 13 GB dataset in memory is impractical on standard workstation hardware. I implemented a reservoir-based chunked streaming loader that samples files in fixed-size batches (250,000 rows). To prevent rare attacks (such as `Uploading_Attack`, `Recon-PingSweep`, and `Backdoor_Malware`) from disappearing due to uniform sub-sampling, the loader enforces a stratified retention rule that guarantees representation for minority classes.

A balanced subsample of 200,000 flows was constructed, partitioned into:
- **Training set:** 140,000 flows (70%)
- **Validation set:** 30,000 flows (15%)
- **Test set:** 30,000 flows (15%)

Splits strictly preserve capture-source integrity to prevent data leakage between evaluation partitions.

### 3.2 Model Architectures and Training Setup
I implemented three baseline classifiers:
1. **Random Forest:** Configured with 300 estimators, a minimum leaf sample threshold of 2, and parallel execution across all CPU cores.
2. **XGBoost:** Configured with 400 boosted trees, maximum depth of 6, learning rate of 0.1, sub-sampling ratio of 0.9, and the histogram-based split finder (`tree_method='hist'`).
3. **Multilayer Perceptron (MLP):** A feed-forward architecture with two hidden layers ($128 \to 64$), ReLU activations, dropout regularization ($p=0.2$), and an output linear projection. Features were standardized via training-set medians and standard deviations, with values clipped to $[-8.0, 8.0]$. The network was trained for 30 epochs using AdamW ($\text{lr}=10^{-3}$, weight decay $= 10^{-4}$, batch size $= 512$) on an NVIDIA GeForce RTX 3060 Laptop GPU using PyTorch with CUDA acceleration. The model checkpoint with the highest validation macro F1-score was retained for evaluation.

### 3.3 Attack and Defense Algorithms
- **Unconstrained Attacks:** Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD, 20 steps, step size $\alpha = 0.25\epsilon$) were evaluated across perturbation budgets $\epsilon \in \{0.005, 0.01, 0.02, 0.05, 0.10\}$ in the normalized feature space.
- **Constrained PGD:** At each iteration, the gradient update was applied in the scaled space, inverted back to raw feature units, projected onto the valid threat-model constraints (clamping fixed features, enforcing monotonic directional changes, and resolving size ordering), and re-normalized.
- **Black-Box Transfer:** Adversarial samples generated against the differentiable MLP were passed directly to the frozen Random Forest and XGBoost models to test cross-model transferability.
- **Defenses:**
  - *Adversarial Training:* The MLP was retrained from scratch on a balanced composite loss of clean and FGSM-perturbed samples ($\epsilon \in \{0.01, 0.02\}$, 20 epochs).
  - *Feature Squeezing:* Tested as an inference-time filter by quantizing each feature into discrete bins corresponding to 8-bit, 6-bit, and 4-bit representations.

---

## 4. Empirical Results and Discussion

### 4.1 RQ1: Baseline Classification Performance

Table 1 summarizes the performance of the three baseline models evaluated on the held-out test split of 30,000 samples across the three target granularities.

**Table 1: Clean Baseline Performance on Held-Out Test Set**

| Target Granularity | Model | Accuracy | Macro F1 | Weighted F1 | Macro Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Multiclass (34 classes)** | Random Forest | 99.01% | 0.7054 | 0.9890 | 0.6992 |
| | XGBoost | **99.06%** | 0.6947 | **0.9897** | 0.6928 |
| | MLP | 91.51% | 0.5937 | 0.9114 | 0.5944 |
| **Family (8 classes)** | Random Forest | **99.31%** | 0.6851 | **0.9925** | 0.6859 |
| | XGBoost | 99.24% | **0.6937** | 0.9919 | 0.6869 |
| | MLP | 98.13% | 0.6328 | 0.9806 | 0.6324 |
| **Binary (Benign vs. Attack)** | Random Forest | **99.62%** | **0.9578** | **0.9962** | 0.9559 |
| | XGBoost | 99.56% | 0.9523 | 0.9956 | **0.9605** |
| | MLP | 99.22% | 0.9152 | 0.9922 | 0.9236 |

#### Analysis:
On the surface, overall accuracy appears exceptional across all models. In the binary detection setting, both Random Forest and XGBoost exceed 99.5% accuracy with macro F1-scores above 0.95. Even in the 34-class problem, both tree ensembles exceed 99.0% accuracy.

However, examining **macro F1** and **macro recall** reveals an important insight: while weighted metrics remain high because dominant attack types (such as ICMP and UDP floods) are classified with near-total accuracy, performance drops sharply on low-volume attack classes. In the multiclass task, macro F1 drops to 0.7054 for Random Forest and 0.5937 for the MLP. Detailed per-class inspection shows that attacks such as `DictionaryBruteForce`, `SqlInjection`, and `CommandInjection` suffer from frequent false negatives because their volumetric footprints are easily overshadowed by high-rate flood traffic.

---

### 4.2 RQ2: Evasion Attacks and Transferability

#### 4.2.1 White-Box Evasion on the MLP
Figure 1 illustrates the degradation in detector accuracy as the perturbation budget $\epsilon$ increases.

In the multiclass setting, the MLP proves highly fragile to gradient-based perturbations:
- At a modest perturbation of $\epsilon = 0.02$, multiclass accuracy drops from 91.51% to **61.32%**.
- At $\epsilon = 0.10$, PGD reduces accuracy to **41.92%** (a 49.59% absolute reduction; Attack Success Rate: 54.20%).
- In the 8-class family task, clean accuracy begins at 98.13% but falls to **46.57%** under PGD ($\epsilon=0.10$).
- Conversely, in the binary setting, accuracy remains comparatively high (dropping from 99.22% to 98.53%). This is because shifting an attack sample entirely across the binary boundary into the benign cluster requires altering core protocol identifiers and flag configurations, which are heavily separated in the decision space.

#### 4.2.2 Realistic Feature-Constrained Evasion
When applying the constrained PGD algorithm—where only packet sizes can increase and sending rates decrease, with all protocol flags and derived metrics strictly clamped:
- Multiclass accuracy on the MLP falls from **91.51% to 42.65%** ($\Delta = 48.86\%$, Attack Success Rate: **53.40%**).
- Family classification accuracy drops from **98.13% to 45.62%** ($\Delta = 52.51\%$, Attack Success Rate: **53.52%**).

This is a notable finding: **an attacker does not need to violate network protocols or forge packet headers to evade detection.** By simply padding packet lengths and slightly delaying transmissions, an adversary can manipulate the flow statistics enough to push the classifier into misclassifying one attack type as another, defeating fine-grained attribution and automated response systems.

#### 4.2.3 Black-Box Transfer to Tree Detectors
To test whether these vulnerabilities are limited to neural networks or represent broader geometric weaknesses in the feature space, I transferred adversarial examples generated against the surrogate MLP to the frozen Random Forest and XGBoost models.

**Table 2: Transferability of Surrogate Adversarial Examples to Tree Detectors ($\epsilon = 0.10$)**

| Target | Model | Attack Variant | Clean Accuracy | Adversarial Accuracy | Accuracy Drop |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Multiclass** | Random Forest | Unconstrained PGD | 99.01% | **29.16%** | **69.85%** |
| | XGBoost | Unconstrained PGD | 99.06% | **16.96%** | **82.10%** |
| | Random Forest | Constrained PGD | 99.01% | **81.87%** | 17.14% |
| | XGBoost | Constrained PGD | 99.06% | **83.88%** | 15.18% |
| **Family** | Random Forest | Unconstrained PGD | 99.31% | **25.88%** | **73.43%** |
| | XGBoost | Unconstrained PGD | 99.24% | **9.41%** | **89.83%** |
| | Random Forest | Constrained PGD | 99.31% | **82.72%** | 16.59% |
| | XGBoost | Constrained PGD | 99.24% | **81.49%** | 17.75% |

The transfer results are striking. Under unconstrained PGD transfer, XGBoost accuracy collapses from **99.06% to 16.96%**, and Random Forest drops from **99.01% to 29.16%**. Even though tree-based algorithms construct orthogonal axis-aligned decision boundaries rather than smooth hyperplanes, the gradient directions computed on the MLP effectively identify shared regions in the continuous statistical space where the training data is sparse.

Under the realistic constrained attack, the transfer effect is more moderate but still causes a significant **15% to 18% accuracy drop** on the tree models without any prior knowledge of the tree architectures.

---

### 4.3 RQ3: Defense Robustness and Trade-Off Analysis

I evaluated two defense strategies against a strong PGD attack ($\epsilon = 0.05$): adversarial retraining and feature squeezing.

**Table 3: Robustness and Clean Accuracy Trade-Offs on the MLP Detector (PGD $\epsilon=0.05$)**

| Target Granularity | Defense Strategy | Configuration | Clean Accuracy | Robust Accuracy | Clean Penalty ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Multiclass** | *Undefended Baseline* | — | 91.51% | 41.92% | — |
| | Adversarial Training | $\epsilon = 0.01$ | 81.21% | 72.36% | -10.30% |
| | Adversarial Training | $\epsilon = 0.02$ | 80.40% | **72.81%** | -11.11% |
| | Feature Squeezing | 8-bit | 77.27% | 42.61% | -14.24% |
| | Feature Squeezing | 6-bit | 82.13% | 43.43% | -9.38% |
| | Feature Squeezing | 4-bit | 58.95% | 49.70% | -32.56% |
| **Family** | *Undefended Baseline* | — | 98.13% | 48.06% | — |
| | Adversarial Training | $\epsilon = 0.01$ | 86.13% | 77.92% | -12.00% |
| | Adversarial Training | $\epsilon = 0.02$ | 85.75% | **81.01%** | -12.38% |
| | Feature Squeezing | 8-bit | 69.70% | 48.13% | -28.43% |
| | Feature Squeezing | 6-bit | 74.88% | 47.84% | -23.25% |
| | Feature Squeezing | 4-bit | 61.15% | 52.59% | -36.98% |

#### Defense Comparison:
1. **Adversarial Training is Effective but Costly:** Incorporating FGSM perturbations into the training loop substantially improves worst-case robustness. On multiclass classification, adversarial training at $\epsilon=0.02$ raises robust accuracy under PGD from **41.92% to 72.81%** (a gain of +30.89 percentage points). However, this robustness comes at a measurable cost: clean test accuracy drops by **11.11%** (from 91.51% to 80.40%). In an operational security center, an 11% penalty in clean classification could translate to thousands of misclassified benign flows per day.
2. **Feature Squeezing Fails on Tabular Network Flows:** Feature squeezing is often proposed in computer vision because natural images tolerate spatial color quantization well. In contrast, for network flow statistics, continuous features such as inter-arrival times, standard deviations, and byte rates carry high dynamic range information. Squeezing continuous values into discrete 4-bit or 6-bit representations degrades clean accuracy significantly (down to 58.95% on 4-bit multiclass) while providing minimal gain in adversarial robustness (~49.70%). Feature squeezing is therefore poorly suited as an evasion defense for flow-based NIDS.

---

## 5. Limitations and Practical Constraints

To maintain research integrity, several methodological limitations must be acknowledged:
1. **Flow-Level vs. Packet-Level Crafting:** This study operates in the feature space of flow statistics rather than the raw packet stream. While I enforced directional and physical constraints, translating these feature deltas into live network packets involves real-world obstacles like MTU limits, OS TCP stack retransmissions, and intermediate network middlebox rewriting.
2. **Frozen Complex Aggregates:** Due to the absence of closed-form formulas relating individual packet padding to bidirectional variance metrics (`Radius`, `Covariance`, `Magnitue`), these features were kept frozen during constrained evasion. A real-world adversary who dynamically measures victim ACKs might alter these features as well, meaning the evasion rates reported here represent a lower bound.
3. **Imbalanced Benchmark Sampling:** Even after stratified sampling, minority classes in CICIoT2023 represent fewer than 100 samples in the test split. While this reflects real-world network traffic distributions, it increases metric variance on the rarest attack types.

---

## 6. Conclusion and Future Directions

This project demonstrates that while tree ensembles and neural networks achieve impressive baseline accuracy on modern IoT intrusion benchmarks, they are fundamentally vulnerable to adversarial evasion. 

Crucially, **evasion does not require forging packet headers or breaking protocol rules.** An attacker operating within legitimate network physics—simply adding dummy payload bytes and spacing out packet transmissions—can reduce neural network multiclass accuracy by over 48 percentage points and trigger transfer degradation in excess of 69 percentage points on Random Forests and 82 percentage points on XGBoost. Furthermore, while adversarial training provides meaningful defensive utility, it imposes an operational tax of approximately 11% on clean classification accuracy, and heuristic defenses like feature squeezing prove ineffective on tabular network features.

### Suggested Extensions for Internship Research:
- **Online Packet Crafting:** Implementing a software-defined networking (SDN) or Scapy-based testbed to inject live adversarial packets and verify if hardware network interface cards (NICs) yield the same evasion rates observed in feature space.
- **Robust Feature Selection:** Investigating whether training solely on protocol invariants (while systematically excluding attacker-controllable packet sizes and rate metrics) can yield an evasion-resistant, low-overhead IoT detector.
- **Certified Robustness:** Exploring randomized smoothing or bound propagation techniques to provide provable robustness guarantees for tabular network intrusion architectures.
