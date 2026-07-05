Here is the comprehensive Research Decomposition for **"TinyEEG-Sleep: An Ultra-Lightweight, Temporal-Context-Aware Edge AI Framework for Real-Time Single-Channel EEG Sleep Staging on Microcontrollers."**

This document breaks the validated blueprint down to the absolute leaf-node tasks.

---

# SECTION 1 — High-Level Problem Tree

```text
TinyEEG-Sleep Project
├── 1. Data & Environment Simulation
│   ├── 1.1 Sleep-EDF Extraction (Fpz-Cz)
│   └── 1.2 Synthetic Dry-Electrode Noise Injection
├── 2. Neural Architecture Design (Float32)
│   ├── 2.1 1D-CNN Spatial Feature Extractor
│   └── 2.2 TCN Temporal Context Extractor
├── 3. TinyML Compression
│   ├── 3.1 Quantization-Aware Training (QAT) to INT8
│   └── 3.2 TFLite Conversion & C-Byte Array Generation
├── 4. Embedded Firmware Engineering
│   ├── 4.1 Concurrent DSP Rolling Buffer (C++)
│   ├── 4.2 TFLite Micro Arena Allocation (C++)
│   └── 4.3 Serial Comms & Benchmarking (Latency/SRAM)
└── 5. Scientific & Neuro-Evaluation
    ├── 5.1 Macro F1 & Clinical Metrics Evaluation
    └── 5.2 INT8 vs Float32 Biomarker Activation Analysis
```

**Component Analysis:**

* **1. Data:** *Why:* Needs clinical baseline + real-world wearable simulation. *Risks:* Synthetic noise doesn't fool clinical reviewers.
* **2. Model:** *Why:* TCN replaces memory-heavy LSTM. *Risks:* TCN receptive field might not capture macro-transition rules.
* **3. Compression:** *Why:* Must fit <256KB. *Risks:* QAT destroys low-amplitude delta waves.
* **4. Firmware:** *Why:* Proves real-world hardware viability. *Risks:* TFLite arena fragmentation crashes the MCU.
* **5. Evaluation:** *Why:* Bridges ML and Neuroscience. *Risks:* Feature maps might be uninterpretable.

---

# SECTION 2 — Research Questions

**Primary Research Question:**

* Can a Temporal Convolutional Network (TCN) for single-channel EEG sleep staging be compressed via QAT to INT8, fitting within <256KB SRAM, without degrading clinical accuracy (Macro F1) below 78%?
  * *Why it matters:* Proves cloud-free neuro-wearables are viable.
  * *Required evidence:* SRAM profiling logs, 10-fold CV F1-scores.

**Secondary Questions:**

* How does INT8 quantization affect the network's internal representation of micro-voltage sleep biomarkers (K-complexes, slow waves)?
  * *Why it matters:* Validates the model learns neuroscience, not noise.

**Hidden Questions:**

* Does static allocation of a 30-second DSP buffer severely limit the available `tensor_arena` size for TFLite Micro, causing tensor allocation failures during inference?
  * *How answered:* Bare-metal memory profiling in C++.

**Reviewer Questions:**

* Why use synthetic noise instead of collecting real dry-electrode data?
  * *How answered:* Justify as a reproducible, controlled system-level benchmark prior to human clinical trials.

**Engineering Questions:**

* Does the Serial communication overhead to feed the MCU test data interfere with the latency measurements of the inference itself?

---

# SECTION 3 — Problem Decomposition

### Problem 1: Synthetic Noise Generation

* **Description:** Create a mathematical model to inject realistic dry-electrode noise into wet-electrode data.
* **Inputs:** Clean Sleep-EDF epochs.
* **Outputs:** Noisy epochs (10dB, 5dB SNR).
* **Constraints:** Must mathematically mimic impedance mismatch and motion.
* **Difficulty:** Moderate. **Effort:** 5 hours. **Required Knowledge:** DSP, Biomedical signal noise profiles.

### Problem 2: TCN Receptive Field Tuning

* **Description:** Tune the dilation rates and kernel sizes of the TCN so the receptive field covers at least 5 epochs (150 seconds) without blowing up parameter count.
* **Dependencies:** Model architecture.
* **Difficulty:** Hard. **Effort:** 10 hours. **Required Knowledge:** PyTorch, TCN math.

### Problem 3: QAT Implementation for 1D Signals

* **Description:** Configure PyTorch/TF QAT to use per-channel symmetric quantization for the CNN filters to preserve dynamic range.
* **Dependencies:** Float32 model trained.
* **Difficulty:** Hard. **Effort:** 15 hours. **Required Knowledge:** FakeQuantization modules.

### Problem 4: MCU Memory Management

* **Description:** Write C++ code that statically allocates a `float32` (or `int16`) circular buffer for DSP, while keeping the TFLite `tensor_arena` strictly separated to prevent heap collisions.
* **Constraints:** `< 256KB` total RAM.
* **Difficulty:** Very Hard. **Effort:** 20 hours. **Required Knowledge:** Embedded C++, RTOS/Bare-metal memory mapping.

---

# SECTION 4 — Unknowns and Assumptions

1. **Assumption:** Synthetic noise accurately reflects real wearable conditions.
   * *Confidence:* Medium. *Risk if false:* High (Reviewer 2 rejects for lack of ecological validity). *Validation:* Compare synthetic noise spectrum to published dry-electrode spectra.
2. **Assumption:** TCNs can learn sleep transition rules as well as LSTMs.
   * *Confidence:* High. *Risk if false:* Low (Just requires more hyperparameter tuning).
3. **Assumption:** Serial feeding latency can be decoupled from inference latency.
   * *Confidence:* High. *Validation:* Use hardware timers (`micros()`) strictly wrapping the `interpreter->Invoke()` call.
4. **Assumption:** PyTorch QAT exports cleanly to TFLite.
   * *Confidence:* Low-Medium. *Risk if false:* High (ONNX/TFLite export of QAT 1D-CNNs often breaks). *Alternative:* Write the model natively in TensorFlow/Keras from day one. **[MAJOR DECISION FLAGGED]**

---

# SECTION 5 — Engineering Tasks

### Module 1: Data Pipeline (Python)

* *Purpose:* Parse EDF files, extract Fpz-Cz, remove excess Wake, inject noise, save as `.npy` or `.h5`.
* *Failure modes:* RAM overflow on PC if loading all 153 patients at once. *Mitigation:* Use PyTorch `Dataset` with lazy loading.

### Module 2: Model & Training (Python)

* *Purpose:* Define CNN+TCN, train Float32, fine-tune QAT, export `.tflite`.
* *Failure modes:* QAT gradients collapse to zero.

### Module 3: Firmware (C++)

* *Purpose:* Receive EEG array via Serial, push to circular buffer, invoke TFLite, print prediction and timing.
* *Interfaces:* `Serial.readBytes()`, `tflite::MicroInterpreter`.
* *Failure modes:* Hard fault due to `tensor_arena` being too small.

---

# SECTION 6 — Coding Task Breakdown

**Python (PC/GPU):**

* `dataset.py`: EDF parser, lazy loader, noise injector. (Complexity: Low)
* `model.py`: PyTorch/Keras definition of 1D-CNN + TCN. (Complexity: Medium)
* `train_float.py`: Cross-validation loop, logging to Weights & Biases. (Complexity: Medium)
* `train_qat.py`: Quantization-aware fine-tuning. (Complexity: High)
* `export.py`: TFLite conversion and `xxd` array generation. (Complexity: Low)
* `evaluate_biomarkers.py`: Extracts feature maps and calculates Pearson correlation between Float32 and INT8. (Complexity: High)

**C++ (MCU):**

* `main.cpp`: Setup, loop, serial handling. (Complexity: Low)
* `dsp_buffer.h`: Circular buffer implementation. (Complexity: Medium)
* `model_data.h`: The exported C-byte array. (Complexity: Low)
* `inference.cpp`: TFLite Micro setup, arena allocation, `Invoke()`. (Complexity: High)

---

# SECTION 7 — Mathematical Tasks

1. **TCN Receptive Field Calculation:** $R = 1 + 2 \cdot (K-1) \cdot \sum d_i$. Must prove $R \ge 150$ seconds (5 epochs).
2. **Weighted Cross-Entropy:** $w_c = \frac{N_{total}}{5 \cdot N_c}$ to balance the N1 class.
3. **Quantization Scaling:** $q = \text{round}(\frac{r}{S} + Z)$. Must document the scaling factors for the first CNN layer to prove delta-wave dynamic range isn't lost to zero.
4. **Signal-to-Noise Ratio (SNR):** $SNR_{dB} = 10 \log_{10}(\frac{P_{signal}}{P_{noise}})$. Used for synthetic noise generation.

---

# SECTION 8 — Literature Tasks

* **[LIT-1]** *Dry-Electrode Noise Profiles:* Search "dry electrode EEG impedance noise spectrum". *Why:* Need exact parameters to justify the synthetic noise generator.
* **[LIT-2]** *TFLite Micro Memory Overheads:* Search "TFLite Micro tensor arena fragmentation". *Why:* Need to know how much padding to add to the theoretical SRAM limit.
* **[LIT-3]** *Sleep Biomarker Frequencies:* Review AASM manual for exact micro-voltage ranges of K-complexes and sleep spindles. *Why:* To analyze the QAT feature maps.

---

# SECTION 9 — Dataset Tasks

* [ ] Download Sleep-EDF Expanded (153 subjects).
* [ ] Write script to extract `Fpz-Cz` and `Hypnogram` annotations.
* [ ] Filter out Wake epochs that occur 30 minutes before/after sleep.
* [ ] Standardize signals (Z-score normalization per subject).
* [ ] Implement Noise Injector (Baseline wander 0.1-0.5Hz, 50Hz mains noise, Gaussian pink noise).
* [ ] Create 10-fold subject-independent split metadata files.

---

# SECTION 10 — Experimental Tasks

* **EXP-1: Float32 Baseline.** Train model without noise. *Metric:* Macro F1. *Goal:* >80%.
* **EXP-2: Noise Robustness.** Test Float32 model on 10dB and 5dB SNR data. *Goal:* Measure degradation.
* **EXP-3: QAT vs PTQ.** Train INT8 model via QAT. Convert Float32 model via PTQ. *Metric:* F1 comparison. *Hypothesis:* QAT beats PTQ by >5% on N1 and N3.
* **EXP-4: MCU Hardware Benchmark.** Flash INT8 to ESP32. Feed 100 epochs via Serial. *Metrics:* Avg Latency (ms), Peak SRAM (KB).

---

# SECTION 11 — Evaluation Tasks

* **Claim:** "Model fits on MCU." -> *Evidence:* Screenshot/log of C++ compiler SRAM usage + runtime `arena_used_bytes()`.
* **Claim:** "Model preserves clinical accuracy." -> *Evidence:* Confusion matrix, Macro F1, Per-class F1 (especially N1 and N3).
* **Claim:** "QAT preserves neurological biomarkers." -> *Evidence:* Visual plot of Float32 vs INT8 feature map activations during a known sleep spindle, accompanied by Pearson's $r$ correlation.

---

# SECTION 12 — Dependency Graph

```text
[LIT-1] -> [Dataset Prep] -> [Synthetic Noise] -> [EXP-1: Float32 Train]
                                                          |
                                                          v
[LIT-2] ----------------------------------------> [EXP-3: QAT/PTQ Train]
                                                          |
                                                          v
                                                  [TFLite Export]
                                                          |
                                                          v
[C++ DSP Buffer] -------------------------------> [EXP-4: MCU Benchmark] -> [Paper Writing]
```

* **Critical Path:** Dataset Prep -> Float32 Train -> QAT Train -> TFLite Export -> MCU Benchmark.
* **Blocking Dependency:** Cannot write C++ inference code until TFLite export is successful and input/output tensor shapes are locked.

---

# SECTION 13 — Milestones

* **M1: Software Baseline (Week 1-2)**
  * *Deliverables:* Clean dataset, Float32 model trained, >80% F1.
* **M2: Compression & Noise (Week 3)**
  * *Deliverables:* Noise injector, INT8 QAT model trained, >78% F1.
* **M3: Hardware Proof (Week 4)**
  * *Deliverables:* ESP32 running inference, SRAM < 200KB, Latency < 500ms.
* **M4: Neuro-Analysis (Week 5)**
  * *Deliverables:* Feature map correlation plots (Float32 vs INT8).
* **M5: Paper Draft (Week 6)**
  * *Deliverables:* Complete manuscript ready for submission.

---

# SECTION 14 — Daily Action Plan (Sample First 5 Days)

* **Day 1:** Download Sleep-EDF. Write `dataset.py` EDF parser. Verify array shapes (N_epochs, 3000, 1). *Difficulty: Low.*
* **Day 2:** Literature review on dry-electrode noise [LIT-1]. Implement `inject_noise()` function. *Difficulty: Medium.*
* **Day 3:** Write `model.py` (1D-CNN + TCN). Calculate receptive field mathematically to ensure it covers 150 seconds. *Difficulty: High.*
* **Day 4:** Write `train_float.py`. Implement Weighted Cross-Entropy. Run 1-fold test to check convergence. *Difficulty: Medium.*
* **Day 5:** **[DECISION]** Decide between PyTorch QAT or Keras QAT based on export stability. Write basic QAT script. *Difficulty: High.*

---

# SECTION 15 — Decision Log

* **Decision 1: PyTorch vs. TensorFlow/Keras for QAT.**
  * *Pros PyTorch:* Better research flexibility. *Cons:* ONNX to TFLite conversion for quantized 1D-CNNs frequently breaks.
  * *Pros Keras:* Native TFLite support. *Cons:* Less flexible QAT API.
  * *Decision:* **Use TensorFlow/Keras.** *Reason:* TFLite Micro deployment is the critical path. Broken exports will kill the project.
* **Decision 2: Microcontroller Choice.**
  * *Options:* ESP32-S3 vs Arduino Nano 33 BLE.
  * *Decision:* **ESP32-S3**. *Reason:* 512KB SRAM provides a safety buffer during development; we can artificially restrict the arena to 256KB for the paper claims.

---

# SECTION 16 — Risk Register

* **Risk:** QAT accuracy collapses on N1/N3.
  * *Probability:* High. *Impact:* Fatal.
  * *Mitigation:* Use INT16xINT8 mixed quantization (INT16 activations, INT8 weights) if strict INT8 fails. TFLite Micro supports this.
* **Risk:** TFLite Micro `tensor_arena` fragmentation.
  * *Probability:* Medium. *Impact:* Major (Hard fault).
  * *Mitigation:* Profile memory using `MicroProfiler`. Avoid dynamic allocations in the C++ `loop()`.

---

# SECTION 17 — Deliverables

1. `dataset_prep.py`
2. `train_qat.ipynb` (Jupyter notebook for reproducibility)
3. `model_int8.tflite`
4. `model_data.cc` (Hex dump)
5. `mcu_inference.ino` (C++ source)
6. Figure: Macro F1 Bar Chart (Clean vs Noise, Float32 vs INT8).
7. Figure: SRAM Allocation Pie Chart.
8. Figure: Biomarker Feature Map Correlation.
9. Final Manuscript.

---

# SECTION 18 — Research Kanban

* **Backlog:** C++ MCU code, Feature map extraction, Paper writing.
* **Next:** TCN architecture definition, QAT implementation.
* **In Progress:** Dataset downloading, EDF parsing.
* **Completed:** [Empty]

---

# SECTION 19 — Final Master Checklist

* **Research Tasks**
  * [ ] [LIT-1] Dry-electrode noise profiles
  * [ ] [LIT-2] Sleep biomarker frequencies
* **Coding Tasks**
  * [ ] `dataset.py`
  * [ ] `model.py` (Keras TCN)
  * [ ] `train.py`
  * [ ] `qat_export.py`
  * [ ] `main.cpp` (MCU)
* **Experimental Tasks**
  * [ ] EXP-1: Float32 Baseline
  * [ ] EXP-2: Noise Robustness
  * [ ] EXP-3: QAT vs PTQ
  * [ ] EXP-4: MCU Hardware Benchmark
* **Validation Tasks**
  * [ ] Extract feature maps
  * [ ] Calculate Pearson $r$ for biomarkers
* **Writing Tasks**
  * [ ] Abstract & Intro
  * [ ] Methodology
  * [ ] Results (Hardware + Clinical)
  * [ ] Discussion (Neuro-interpretability)

---

# FINAL VERDICT

* **Is this research worth pursuing?** Absolutely. It perfectly bridges the gap between ML, Embedded Systems, and Neuroscience.
* **Would it likely be publishable?** Yes. Hardware-backed TinyML papers targeting clinical signals are highly sought after.
* **Main strengths:** Extreme practical utility; avoids the "simulation only" trap of many ML papers; strong interdisciplinary evaluation.
* **Biggest weaknesses:** Lack of real-world dry-electrode testing (mitigated by synthetic noise, but still a vulnerability).
* **Highest-risk assumptions:** That standard QAT won't destroy the low-frequency delta wave representations required for N3 staging.
* **Most important next steps before implementation:** **LOCK IN THE ML FRAMEWORK.** Verify immediately if Keras/TensorFlow QAT supports 1D Dilated Convolutions for TFLite Micro export. If it doesn't, you must pivot the architecture to standard 1D-CNNs immediately.
* **Estimated publication potential:** Q1/Q2 Journal.
* **Recommended target venue:** *IEEE Journal of Biomedical and Health Informatics (JBHI)*.

**Prioritized Action Plan:**

1. Setup Python environment and verify Keras TCN -> TFLite INT8 export works with dummy data (Do this today to retire the highest technical risk).
2. Download Sleep-EDF and write the parser.
3. Train the Float32 baseline.

