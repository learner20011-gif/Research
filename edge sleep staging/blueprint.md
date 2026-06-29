
Here is the comprehensive pre-implementation Research Blueprint for your project. As requested, this document adopts a brutally critical perspective to ensure the research design is bulletproof before you write a single line of code.

---

# 1. Executive Summary

* **Research Motivation:** Real-time sleep staging is critical for continuous neurological monitoring, but current clinical standards (Polysomnography) are intrusive, and consumer wearables rely on proxy signals (Actigraphy/HRV) that lack neurological depth.
* **Problem:** Single-channel EEG models achieve clinical accuracy but rely on memory-heavy architectures (LSTMs/Transformers). Microcontrollers (MCUs) lack the SRAM (<256KB) to run these models, forcing devices to stream raw data to the cloud, which drains batteries and poses privacy risks.
* **Validated Gap:** There is a lack of fully on-device, temporal-context-aware TinyML models for raw single-channel EEG sleep staging. Furthermore, the neuroscientific impact of 8-bit Quantization-Aware Training (QAT) on low-amplitude sleep biomarkers (e.g., K-complexes, delta waves) remains unexplored.
* **Proposed Solution:** "TinyEEG-Sleep," an ultra-lightweight 1D-CNN + Temporal Convolutional Network (TCN) architecture optimized via QAT. It will be deployed on a bare-metal MCU alongside a concurrent Digital Signal Processing (DSP) buffer to prove real-world embedded viability.
* **Expected Contribution:** (1) A memory-profiled embedded C++ pipeline proving concurrent DSP/inference in <256KB SRAM. (2) A neuro-interpretability analysis of how INT8 quantization affects clinical sleep biomarkers.
* **Practical Significance:** Enables the creation of privacy-preserving, zero-latency, multi-day autonomous neuro-wearables without cloud dependency.

---

# 2. Background

* **Domain Overview:** Sleep staging classifies 30-second epochs into Wake (W), REM, N1, N2, and N3.
* **Existing Methods:** Clinical methods use manual scoring of PSG. Automated methods use Deep Learning (DeepSleepNet, SleepFCN) with Float32 precision on GPUs/cloud. Edge methods use simple CNNs on HRV/Actigraphy.
* **Why this problem matters:** Cloud-dependent wearables fail in disconnected environments, drain batteries via Bluetooth/Wi-Fi, and transmit highly sensitive raw brain data to third-party servers.
* **Historical Evolution:** Hand-engineered features $\rightarrow$ CNNs $\rightarrow$ CNN+LSTMs (DeepSleepNet) $\rightarrow$ Mobile-optimized CNN+LSTMs (TinySleepNet) $\rightarrow$ **[Our Work]** MCU-optimized CNN+TCN+QAT.
* **Current State of the Art (SOTA):** TinySleepNet (Supratak, 2020) reduced parameters but still requires Float32 and MBs of RAM. True TinyML sleep models (Alajlan, 2023) only use non-EEG signals.

---

# 3. Literature Review

| Paper / Author                                   | Objective                     | Method              | Dataset         | Strengths                                | Weaknesses / Limitations                                  | Relation to My Work                            |
| :----------------------------------------------- | :---------------------------- | :------------------ | :-------------- | :--------------------------------------- | :-------------------------------------------------------- | :--------------------------------------------- |
| **DeepSleepNet** (Supratak, 2017)          | Auto sleep staging on raw EEG | 1D-CNN + Bi-LSTM    | Sleep-EDF, MASS | High accuracy, captures transition rules | Massive memory footprint, Float32, offline only           | Baseline for temporal context accuracy.        |
| **TinySleepNet** (Supratak, 2020)          | Parameter reduction           | 1D-CNN + LSTM       | Sleep-EDF       | Fewer parameters, mobile-friendly        | Still too large for <256KB SRAM, ignores MCU DSP overhead | Baseline for "lightweight" EEG models.         |
| **SleepFCN** (Goshtasbi, 2022)             | Fully convolutional staging   | FCN + Dilated Convs | Sleep-EDF, SHHS | Removes dense layers, uses dilated convs | Float32 only, no hardware deployment                      | Inspires our TCN approach.                     |
| **EdgeML Stress/Sleep** (Srivastava, 2024) | Edge AI for sleep/stress      | TinyML (TFLite)     | SWELL, ISRUC    | Real MCU deployment (RPi Pico)           | Uses HRV/Actigraphy, not EEG. Lacks neurological depth.   | Proves MCU viability; we replace HRV with EEG. |

---

# 4. Research Gap

* **What is missing:** A temporal-context-aware single-channel EEG sleep staging model that operates strictly within the memory constraints (<256KB) of a Cortex-M MCU, coupled with an analysis of how INT8 quantization degrades specific neurological biomarkers.
* **Why it exists:** ML researchers optimize for accuracy (ignoring hardware constraints). Embedded engineers optimize for memory (using simple signals like HRV to avoid complex temporal EEG models). Neuroscientists distrust INT8 models because quantization crushes dynamic range.
* **Evidence supporting the gap:** Recent reviews (e.g., Alajlan 2023) show TinyML sleep models rely on non-EEG data. Papers using EEG on edge devices (like Raspberry Pi 4) do not meet strict MCU constraints.
* **Why previous work failed:** LSTMs require dynamic memory allocation and large state buffers, making them hostile to bare-metal MCU deployment. Post-Training Quantization (PTQ) crushes the signal-to-noise ratio of low-amplitude EEG waves.
* **Competing viewpoints:** "Edge AI is solved by sending features to a smartphone via Bluetooth." (Rebuttal: This is not true Edge AI; it's a distributed system vulnerable to latency and battery drain).
* **Confidence Level:** High (8/10). The intersection of QAT, TCNs, and EEG biomarker analysis on bare-metal MCUs is verifiably empty.

---

# 5. Research Question

* **Primary Research Question:** Can a Temporal Convolutional Network (TCN) for single-channel EEG sleep staging be compressed via Quantization-Aware Training (QAT) to fit within <256KB of SRAM without degrading clinical accuracy (>83%)?
* **Secondary Questions:** How does INT8 quantization affect the neural network's ability to represent specific micro-voltage sleep biomarkers (e.g., K-complexes, delta waves)?
* **Evaluation Questions:** Does the model maintain robustness when synthetic dry-electrode noise and motion artifacts are injected into the clinical dataset?
* **Engineering Questions:** Can the MCU concurrently maintain a 30-second rolling DSP buffer and execute the TFLite Micro inference without triggering a hard fault or memory overflow?

---

# 6. Objectives

* **Primary Objective:** Design, train, and deploy an INT8-quantized 1D-CNN+TCN model on an MCU (e.g., ESP32/Arduino Nano 33 BLE) that achieves >83% accuracy on the Sleep-EDF dataset.
* **Secondary Objectives:** Profile exact SRAM and Flash usage. Extract feature maps to prove neuro-interpretability of the quantized model.
* **Stretch Objectives:** Measure active power consumption (mW) during inference to estimate battery life on a standard coin cell.
* **Out-of-scope items:** Designing a custom PCB. Collecting real-world dry-electrode data from human subjects.

---

# 7. Hypotheses

* **H1 (Memory vs. Accuracy):**
  * *Null:* Compressing a temporal EEG model to <256KB SRAM using INT8 QAT will result in a statistically significant drop in Macro F1-score compared to a Float32 baseline.
  * *Alternative:* The INT8 QAT TCN will maintain a Macro F1-score within 2% of the Float32 baseline while reducing memory footprint by $\ge$ 70%.
  * *Testing:* Compare Float32 and INT8 models using Wilcoxon signed-rank test across subjects.
* **H2 (Biomarker Preservation):**
  * *Null:* INT8 quantization destroys the model's ability to activate on low-amplitude sleep spindles and delta waves.
  * *Alternative:* Feature map activations of the INT8 model will show a high correlation (Pearson's $r > 0.85$) with the Float32 model during N2 and N3 stages.
  * *Testing:* Extract intermediate activations from both models during known spindle/delta events and compute correlation.
* **H3 (Hardware Viability):**
  * *Null:* Concurrent 100Hz ADC DSP buffering and TCN inference will exceed 256KB SRAM, causing a memory overflow.
  * *Alternative:* The system will successfully execute the full pipeline using < 200KB of statically allocated SRAM.
  * *Testing:* Deploy via TFLite Micro; monitor heap/arena allocation using hardware serial debugging.

---

# 8. Novelty Assessment

* **Type of Novelty:**
  * Algorithmic: *Low* (TCNs and QAT exist).
  * Methodological: *Moderate* (Applying QAT specifically to preserve neuro-biomarkers).
  * System: *Strong* (Concurrent DSP+Inference SRAM mapping for EEG on MCU).
  * Evaluation: *Strong* (Simulated dry-electrode noise benchmarking for TinyML).
* **Overall Rating:** **Strong**.
* **Why:** While the ML components are off-the-shelf, their system-level integration under extreme constraints, combined with a neuroscientific evaluation of the compression loss, creates a highly novel applied engineering paper.
* **How to strengthen:** *Needs Validation* - Ensure the synthetic noise injection perfectly mimics real-world dry electrode impedance variations to satisfy skeptical clinical reviewers.

---

# 9. Expected Contributions

* **Scientific:** A detailed analysis of how 8-bit integer quantization impacts the deep-learning representations of clinical EEG sleep biomarkers.
* **Engineering:** A memory-profiled C++ template for running continuous physiological DSP buffering concurrently with TFLite Micro inference on a Cortex-M MCU.
* **Industrial:** Proof-of-concept for wearable manufacturers that cloud-free, zero-latency EEG sleep staging is viable on cheap ($5) hardware.
* **Open-source:** Public GitHub repository containing the PyTorch QAT pipeline, noise-injection scripts, and C++ MCU deployment code.

---

# 10. Proposed Methodology

**Overall Workflow:**

```text
[Sleep-EDF Dataset] -> [Synthetic Noise Injection] -> [Preprocessing (100Hz, 30s epochs)] 
       |
       v
[PyTorch: 1D-CNN + TCN Float32 Training] -> [QAT Fine-Tuning (INT8)]
       |
       v
[TFLite Conversion] -> [C++ Hex Dump]
       |
       v
[MCU Deployment (ESP32/Arduino)] <--> [PC Serial Port (Feeds EEG data)]
       |
       v
[Measure: Latency, SRAM, Accuracy, Power]
```

* **Data Pipeline:** Extract Fpz-Cz, segment to 30s. Inject synthetic baseline wander and 50Hz noise.
* **Model Architecture:**
  * *Spatial:* 3 blocks of 1D-CNNs (stride 2, ReLU, BatchNorm).
  * *Temporal:* 2 blocks of Dilated Causal Convolutions (TCN) to capture 5-epoch context.
  * *Classifier:* Fully Connected layer to 5 classes.
* **Training Pipeline:** Train Float32 model using Adam. Implement QAT using PyTorch `torch.quantization`.
* **Deployment Pipeline:** Convert to `.tflite`. Use `xxd` to generate a C byte array. Allocate `tflite::MicroInterpreter` arena in C++.

---

# 11. Mathematical Formulation

* **Variables:** $X \in \mathbb{R}^{T \times C}$ (Input EEG, $T=3000$, $C=1$). $Y \in \{W, N1, N2, N3, R\}$.
* **TCN Dilation:** For a 1D sequence $x$ and filter $f$, the dilated convolution operation $F$ on element $s$ is:
  $F(s) = \sum_{i=0}^{k-1} f(i) \cdot x_{s - d \cdot i}$ (where $d$ is dilation factor, $k$ is kernel size).
* **Quantization:** Real value $r$ mapped to INT8 $q$:
  $r = S(q - Z)$ (where $S$ is scale, $Z$ is zero-point).
* **Loss Function:** Weighted Categorical Cross-Entropy (to handle N1 class imbalance):
  $L = - \sum_{c=1}^{5} w_c y_c \log(\hat{y}_c)$
* **Constraints:** $SRAM_{total} = SRAM_{DSP} + SRAM_{TFLite\_Arena} \le 256 \text{ KB}$.

---

# 12. Experimental Design

* **Independent Variables:** Model precision (Float32 vs. INT8), Noise level (Clean vs. 10dB SNR vs. 5dB SNR).
* **Dependent Variables:** Macro F1-Score, Accuracy, Inference Latency (ms), Peak SRAM (KB).
* **Baselines:** DeepSleepNet (Cloud baseline), TinySleepNet (Mobile baseline).
* **Cross-validation:** Leave-One-Subject-Out (LOSO) or k-fold (k=10) subject-independent CV. *Assumption: k-fold is used due to computational limits on PC.*
* **Hyperparameter search:** Grid search on TCN dilation rates and CNN kernel sizes to find the optimal accuracy-to-memory Pareto frontier.

---

# 13. Dataset Analysis

* **Dataset:** Sleep-EDF Expanded (PhysioBank). 153 whole-night PSG recordings.
* **Why suitable:** Gold standard, heavily benchmarked in literature.
* **Biases:** Uses wet Ag/AgCl electrodes. Patients are mostly healthy or have mild sleep issues.
* **Limitations:** Lacks real-world wearable motion artifacts.
* **Data preprocessing:**
  * *Cleaning:* Remove Wake periods > 30 mins before/after sleep to prevent Wake-class dominance.
  * *Augmentation:* **[CRITICAL]** Add synthetic pink noise, 50Hz interference, and transient motion artifacts to simulate dry electrodes.
* **Ethical issues:** Public dataset, completely de-identified. No IRB required.

---

# 14. Baseline Comparison

| Baseline                    | Architecture  | Precision | Target Hardware | Why Selected                |
| :-------------------------- | :------------ | :-------- | :-------------- | :-------------------------- |
| **DeepSleepNet**      | CNN + Bi-LSTM | Float32   | Cloud / GPU     | Gold standard for accuracy. |
| **TinySleepNet**      | CNN + LSTM    | Float32   | Smartphone      | SOTA for "lightweight" EEG. |
| **Our Model (Clean)** | CNN + TCN     | Float32   | PC / Edge       | Proves TCN matches LSTM.    |
| **Our Model (QAT)**   | CNN + TCN     | INT8      | Microcontroller | The proposed contribution.  |

---

# 15. Evaluation Metrics

* **Claim 1: Clinical Accuracy.**
  * *Metric:* Macro F1-Score.
  * *Reason:* Accuracy is misleading due to N2/Wake class dominance.
  * *Expected:* > 78% Macro F1.
  * *Failure case:* Model collapses on N1 (predicts 0 N1 epochs).
* **Claim 2: Hardware Viability.**
  * *Metric:* Peak SRAM allocation (KB).
  * *Reason:* Must prove it fits on MCU.
  * *Expected:* < 200 KB.
* **Claim 3: Real-Time Capability.**
  * *Metric:* Inference Latency (ms).
  * *Expected:* < 1000 ms per 30s epoch.

---

# 16. Ablation Study Plan

* **Remove QAT (Use Post-Training Quantization - PTQ):** Proves that QAT is strictly necessary for EEG signals because PTQ destroys low-amplitude wave representations.
* **Remove TCN (Use 1D-CNN only):** Proves that temporal context is required for high accuracy (specifically distinguishing N1 from REM).
* **Remove Synthetic Noise:** Shows the performance gap between "ideal lab conditions" and "simulated wearable conditions."

---

# 17. Statistical Validation

* **Tests:** Wilcoxon signed-rank test to compare the F1-scores of the Float32 model vs. INT8 model across all cross-validation folds.
* **Significance thresholds:** $p < 0.05$.
* **Confidence intervals:** 95% CIs reported for all accuracy and latency metrics.

---

# 18. Computational Requirements

* **Hardware (Training):** Standard PC with 1x NVIDIA GPU (e.g., RTX 3060/4090).
* **Hardware (Deployment):** ESP32-S3 or Arduino Nano 33 BLE Sense.
* **Training time:** ~12-24 hours for 10-fold CV on Sleep-EDF.
* **Inference latency:** *TODO* (Estimated <500ms on 240MHz ESP32).
* **SRAM:** *TODO* (Estimated ~150KB).

---

# 19. Risks and Failure Modes

* **Technical Risk (Fatal):** TFLite Micro arena fragmentation causes memory overflow when combined with the DSP rolling buffer.
  * *Mitigation:* Statically allocate the DSP circular buffer outside the TFLite arena.
* **Research Risk (Major):** QAT fails to preserve N3 (delta wave) accuracy, dropping F1-score below publishable limits (<70%).
  * *Mitigation:* Implement custom quantization scales (per-channel quantization) to preserve dynamic range in early CNN layers.
* **Reviewer Risk (Major):** Reviewer #2 rejects the paper because "simulated noise is not a real wearable."
  * *Mitigation:* Clearly state in the introduction that this is a *systems-engineering* and *algorithmic* proof-of-concept, laying the groundwork for future clinical dry-electrode trials.

---

# 20. Threats to Validity

* **Internal Validity:** Data leakage between epochs. (Must ensure strict subject-wise splitting, not epoch-wise splitting).
* **External / Ecological Validity:** *Weakest point.* The model is trained on wet-electrode data but claims to be for dry-electrode wearables. The synthetic noise injection mitigates this, but does not eliminate the threat.
* **Construct Validity:** Does INT8 activation mapping truly represent "neuro-biomarker preservation"? (Needs Validation: Must verify with a neuroscientist or authoritative literature).

---

# 21. Ethics and Reproducibility

* **Privacy:** The core ethical advantage of this paper is *Privacy-by-Design*. Brain data never leaves the device.
* **Reproducibility:** All PyTorch training code, noise-injection algorithms, and C++ Arduino sketches will be open-sourced on GitHub.
* **Licensing:** Sleep-EDF is open access. Code will be MIT licensed.

---

# 22. Implementation Roadmap

* **Milestone 1: Data & Baseline (Weeks 1-2)**
  * *Deliverable:* Sleep-EDF downloaded, preprocessed. Float32 CNN+TCN trained.
* **Milestone 2: Noise & QAT (Weeks 3-4)**
  * *Deliverable:* Synthetic noise injected. QAT implemented. INT8 model achieves >78% Macro F1.
* **Milestone 3: Hardware Deployment (Weeks 5-6)**
  * *Deliverable:* C++ code written. ESP32 successfully classifies epochs sent via Serial. Latency/SRAM profiled.
* **Milestone 4: Analysis & Writing (Weeks 7-8)**
  * *Deliverable:* Feature maps extracted. Draft finalized.

---

# 23. Paper Outline

1. **Abstract:** Focus on the SRAM bottleneck, the QAT+TCN solution, and the neuro-preservation findings.
2. **Introduction:** Why cloud-EEG is bad (privacy/power). The TinyML gap.
3. **Related Work:** DeepSleepNet (too heavy), Edge Actigraphy (too simple).
4. **Methodology:** 1D-CNN+TCN architecture, QAT pipeline, Synthetic Noise model.
5. **Hardware Implementation:** Memory allocation strategy (DSP + TFLite Arena).
6. **Experiments:** Sleep-EDF setup, training details.
7. **Results:** Accuracy vs. Memory trade-off. Latency/Power metrics.
8. **Discussion:** *Crucial Section* - Neuro-interpretability of INT8 feature maps.
9. **Limitations:** Lack of real human dry-electrode testing.
10. **Conclusion.**

---

# 24. Reviewer Simulation

* **Reviewer #1 (Supportive - TinyML Expert):** "Great system-level integration. The memory profiling is excellent. Accept."
* **Reviewer #2 (Skeptical - Clinical Neuroscientist):** "You cannot claim this works for wearables when you used Sleep-EDF. Synthetic noise is not real impedance mismatch."
  * *Pre-implementation fix:* Soften claims. Change "Solves wearable sleep staging" to "Provides a hardware-aware computational framework for future wearable sleep staging."
* **Reviewer #3 (Critical - ML Theorist):** "TCNs and QAT are standard tools. Where is the novelty?"
  * *Pre-implementation fix:* Emphasize Section 8 (Discussion). The novelty is the *analysis* of how QAT affects EEG micro-events, not the invention of QAT itself.

---

# 25. Implementation Readiness Checklist

* **Research & Novelty**

  * [X] Gap validated against literature
  * [X] Contributions clearly defined
* **Methodology & Data**

  * [X] Sleep-EDF dataset accessible

  * [⚠] Synthetic noise parameters defined (Needs literature review on dry-electrode SNR profiles)
* **Code & Hardware**

  * [X] PyTorch / TFLite toolchain verified
  * [X] ESP32 / Arduino hardware on hand

  * [⚠] C++ Serial feeding script written (Needs Work)
* **Evaluation**

  * [X] Metrics defined (Macro F1, SRAM, Latency)

  * [⚠] Feature map extraction code for INT8 models (Needs Work)

---

# 26. Final Verdict

* **Is this research worth pursuing?** Yes. It perfectly matches your resources (PC, GPU, MCU, no lab) while addressing a highly relevant gap in digital health and IoT.
* **Would it likely be publishable?** Yes, if the hardware constraints (SRAM profiling) and neuro-analysis are executed rigorously.
* **Main strengths:** Highly practical, solves a real privacy/power issue, excellent cross-disciplinary appeal.
* **Biggest weaknesses:** Ecological validity (using synthetic noise instead of real dry-electrode data).
* **Highest-risk assumptions:** That QAT will preserve the N1 and REM transition states adequately.
* **Estimated publication potential:** Q1/Q2 Journal.
* **Recommended Target Venue:** *IEEE Journal of Biomedical and Health Informatics (JBHI)* or *Biomedical Signal Processing and Control (BSPC)*.
* **Action Plan:**
  1. Download Sleep-EDF and build the Float32 baseline.
  2. Research exact SNR values for dry-electrode EEG to build a mathematically defensible synthetic noise injector.
  3. Run QAT and verify the F1 score doesn't collapse.
  4. Move to C++ MCU deployment.

