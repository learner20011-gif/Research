# 🧠 Ear-EEG Sleep Staging — Plain-English Algorithm & Workflow Guide

> **⚠️ SYNC RULE:** This document describes the exact implementation in [ear_eeg_sleep_staging.py](ear_eeg_sleep_staging.py). Whenever you change any hyperparameter, preprocessing step, model layer, validation split, or output directory in the Python notebook, update the matching section here immediately.

---

## 📋 Quick Overview of What This Pipeline Does

Our goal is to automatically classify **30-second slices of brain waves recorded from the ear (Ear-EEG)** into the **5 clinical sleep stages** (**Wake, N1, N2, N3, REM**) using a neural network small enough to run directly inside an **Arduino Nano 33 BLE Sense** microcontroller (**256 KB SRAM / 1,024 KB Flash**) without losing clinical accuracy.

---

## 🔍 Section-by-Section Algorithm Walkthrough (Plain English)

### Section 0 — Environment Setup & Experiment Configuration
- **What It Does:** Installs/imports libraries, selects the fastest compute device (`Intel Arc XPU > NVIDIA CUDA > CPU`), sets publication-grade **white-background, 300 DPI** chart styling, and defines every experiment setting in one place (`RUN_NAME`, `FS = 128`, `CONTEXT_LEN = 3`, `CLASS_WT_POWER = 0.5`, etc.).
- **How Outputs Are Separated:**
  - Heavy dataset files go into `outputs/cache/` so they are shared across runs.
  - Every training experiment gets its own folder under `outputs/runs/<RUN_NAME>/` and saves its settings to `config.json`, so running a new test never overwrites old results.
- **Why We Do It This Way:** Having all parameters at the top and isolating every run by `RUN_NAME` makes ablation studies reproducible and prevents accidental mixing of old 256 Hz and new 128 Hz results.

---

### Section 1 — Dataset Architecture Overview
- **What It Does:** Summarizes the **EESM19 (`ds005185`)** dataset metadata: **20 healthy subjects**, **214 total nights** (**80 laboratory nights with gold-standard PSG sleep labels** + 134 unlabeled home nights), recorded with dry-contact iridium oxide electrodes.
- **Files Written:** `outputs/runs/<RUN_NAME>/figures/dataset_overview.png`

---

### Section 2 — Fast Lazy Data Loading (`RealEEGSleepLoader`)
- **What It Does:** Reads the raw EEG files (`.set` + `.fdt`) and sleep stage label files (`scoring1_events.tsv`) for all **20 subjects (80 scored nights)**.
- **Step-by-Step Algorithm:**
  1. **Lazy Channel Picking (`preload=False`):** Opens only the file header, throws away all 13 scalp PSG/EOG/EMG channels, and loads **only the 12 Ear-EEG channels** (`ELA..ELI` on Left ear, `ERA..ERI` on Right ear) into RAM.
  2. **Cleanest Channel Selection:** Scores each Left and Right ear electrode by the percentage of valid (non-`NaN`) samples and picks the cleanest Left channel and cleanest Right channel for that night.
  3. **Per-Epoch Cross-Ear Derivation with Fault-Tolerant Fallback:** Slices each labeled 30-second window at the native `500 Hz` **before** resampling (so one bad moment never smears `NaN`s across the whole night):
     - If **both** ears are clean $\to$ computes `Left - Right` (**cross-ear bipolar signal**).
     - If **one** ear drops out (`NaN`) $\to$ automatically falls back to the surviving ear for that epoch.
     - If **both** are bad $\to$ skips that epoch.
  4. **DC Removal & Downsampling to `128 Hz`:** Converts Volts to microvolts ($\mu\text{V}$), subtracts the epoch mean, and downsamples all clean epochs in one fast vectorized call (`scipy.signal.resample_poly`, 500 Hz $\to$ 128 Hz, giving `3,840` samples per 30s epoch).
- **Why `128 Hz` & Cross-Ear?**
  - All sleep EEG waves (`0.5–40 Hz`) sit below `40 Hz`. By Nyquist's rule, `128 Hz` (Nyquist = `64 Hz`) preserves 100% of sleep frequencies while cutting RAM, training time, and microcontroller buffer size in half compared to `256 Hz`.
  - Subtracting `Left - Right` spans across the brain (boosting slow-waves and spindles by 3–5x) while canceling shared noise (50 Hz hum, body movement).
- **Files Written:** `outputs/cache/loaded_cross_ear_128hz.pkl`

---

### Section 3 — Exploratory Data Analysis (EDA)
- **What It Does:** Counts how many 30-second epochs belong to each sleep stage across all 80 nights, plots a full-night hypnogram (sleep stage timeline over 8 hours), and visualizes raw 30s Ear-EEG waveforms for each stage.
- **Why We Do It This Way:** Confirms the severe class imbalance (**N2 ~43%, N1 ~7%**) that drives our weighted loss design in Section 6.
- **Files Written:** `eda_summary.pkl`, `class_distribution.png`, `hypnogram.png`, `raw_eeg_per_stage.png`

---

### Section 4 — Signal Processing & Artifact Removal (`EarEEGPreprocessor`)
- **What It Does:** Cleans every 30-second epoch and flags corrupted/flat-line recordings before training.
- **Step-by-Step Algorithm:**
  1. **50 Hz Notch Filter (`Q=30`):** Removes power-line electrical hum.
  2. **0.5–40 Hz Bandpass Filter (4th-order Butterworth):** Removes slow baseline sweat/drift (`<0.5 Hz`) and high-frequency muscle noise (`>40 Hz`).
  3. **Artifact Flagging:** Marks an epoch as an artifact (excluded from training/testing) if filtered voltage exceeds `±500 µV` (motion spike) or standard deviation drops below `0.5 µV` (detached electrode / flatline).
  4. **SNR Measurement:** Estimates Signal-to-Noise Ratio (`1–30 Hz` EEG power vs. `45–62 Hz` noise band) on the raw signal before filtering.
  5. **Leak-Free Per-Epoch Z-Score Standardization:** Subtracts each single epoch's mean, divides by its own standard deviation (mean `0`, std `1`), and clips extreme outliers to `[-10, +10]`.
- **Why Per-Epoch Standardization?** Computing mean/std strictly inside each individual 30-second window guarantees **zero information leakage across subjects** and keeps activation values well-behaved for INT8 quantization.
- **Files Written:** `outputs/cache/processed_cross_ear_128hz.pkl`, `preprocessing.png`

---

### Section 5 — Time-Frequency Biomarker Analysis
- **What It Does:** Verifies that real physiological sleep markers survive Ear-EEG filtering by computing log-power spectrograms and relative band power (`Delta 0.5–4 Hz`, `Theta 4–8 Hz`, `Alpha 8–13 Hz`, `Beta 13–30 Hz`) for each sleep stage.
- **Files Written:** `spectrograms.png`, `band_powers.png`

---

### Section 6 — Lightweight Deep Learning Architecture (`TinyEEGSleep` v2)
- **What It Does:** Defines our edge-optimized neural network (~**122.7K parameters**, **~120 KB** raw INT8 weights), the multi-epoch context dataset loader, and training utilities.
- **Step-by-Step Architecture:**
  1. **Multi-Epoch Context Input `(batch, 3, 3840)`:** For each target epoch $t$, feeds `[epoch t-1, epoch t, epoch t+1]` clamped strictly within the same recording night (never crossing nights or subjects).
  2. **Shared Epoch Encoder (`encode_single`):** Runs on one 30-second epoch `(1, 3840)` at a time:
     - **3 Strided 1D-Conv Blocks:** `1->32 (k=25, s=4)` $\to$ `32->64 (k=9, s=2)` $\to$ `64->96 (k=7, s=2)`, compressing time `3840 -> 240` steps to keep peak activation memory tiny.
     - **4 Depthwise-Separable Dilated TCN Blocks (`DSTCNBlock`):** `dilation = 1, 2, 4, 8` with `k=5`. Splits each convolution into a per-channel spatial filter (`groups=96`) plus a `1x1` channel mixer, cutting parameters and MACs by ~4x while seeing long-range patterns (spindles, K-complexes, slow-wave trains). Includes quantization-safe `FloatFunctional` residual skips and `BatchNorm` fusion support.
     - **Global Average Pooling:** Summarizes each 30s epoch into a compact **96-value vector** (96 bytes in INT8).
  3. **Context Classifier (`clf`):** Concatenates the 3 neighboring 96-dim epoch embeddings (`288` features) $\to$ `Linear(288 -> 64) -> ReLU -> Dropout -> Linear(64 -> 5)`.
  4. **N1 Over-Prediction Fix + Augmentation:** Uses **dampened inverse-frequency class weights** $w_c \propto (1/\text{count}_c)^{0.5}$ plus `label_smoothing=0.05` so rare **N1** is learned without flooding predictions with false N1s. Training data is randomly augmented on the fly (polarity flip, amplitude scaling `0.85–1.15x`, small time shift `±0.5s`, light noise).

---

### Section 7 — Leak-Free Subject-Independent Evaluation (3-Way LOSO-CV)
- **What It Does:** Tests how well the model generalizes to a **completely unseen person** across all **20 subjects**.
- **Strict 3-Way Split (Per Fold):**
  - **Test Set (`1` subject):** Locked away and evaluated **only once** after training finishes.
  - **Validation Set (`2` other subjects):** Used at each training epoch to pick the best checkpoint and trigger **Early Stopping (`patience=12`, max `50` epochs)**.
  - **Training Set (`17` remaining subjects):** Used to update model weights with `AdamW` + `CosineAnnealingLR`.
- **Why 17 / 2 / 1 Instead of 19 / 1?** Earlier versions picked the best checkpoint using the test subject's own score, which artificially inflated F1 by ~2.8%. Using 2 separate validation subjects makes the test score 100% honest.
- **Files Written:** `checkpoints/loso_fold_<sub_id>.pkl`, `checkpoints/loso_results.pkl`, `figures/loso_results.png`

---

### Section 8 — Real 20-Fold INT8 Quantization (PTQ & QAT)
- **What It Does:** Compresses **every single one of the 20 fold models** from 32-bit floating-point (`FP32`) down to 8-bit integers (`INT8`, ~4x smaller) using two methods and compares them fairly across all 20 subjects (no best-fold cherry-picking):
  1. **INT8 PTQ (Post-Training Quantization):** Fuses `Conv+BN+ReLU`, calibrates activation ranges on 30 batches of the **validation** subjects, and converts to INT8.
  2. **INT8 QAT (Quantization-Aware Training):** Fuses `Conv+BN+ReLU`, inserts fake-quantization observers, fine-tunes for `5` epochs on CPU so weights adapt to 8-bit rounding noise, and converts to true CPU INT8 kernels (`fbgemm`). *(Runs on CPU because PyTorch observers require `float64`, which Intel Arc XPU does not support.)*
- **Files Written:** `checkpoints/qat_fold_<sub_id>.pkl`, `checkpoints/qat_results.pkl`, `models/tiny_eeg_sleep_fp32.pt`, `models/tiny_eeg_sleep_int8.pt`, `figures/quantization_analysis.png`

---

### Section 9 — Analytical Arduino Nano 33 BLE Sense Profiling
- **What It Does:** Measures exact hardware feasibility for the **Arduino Nano 33 BLE Sense** (`Nordic nRF52840`, **256 KB SRAM**, **1,024 KB Flash**) using real PyTorch forward hooks rather than guessed numbers:
  - **Flash ROM (Weights):** `122,725` parameters $\times 1\text{ byte} \approx 119.8\text{ KB}$ (`~125.8 KB` with 5% TFLite FlatBuffer overhead $\to$ **12.3% of 1 MB Flash**). Weights live in Flash ROM, **not** SRAM.
  - **Internal SRAM (Dynamic Runtime Arena):**
    - `30s @ 128 Hz` Float32 DSP input buffer = `15.0 KB`
    - Hook-measured peak simultaneous layer activation buffer $\times 1.25$ TFLite-Micro safety margin $\approx 84.4\text{ KB}$
    - Streaming 3-epoch embedding ring buffer (`3 x 96 B`) = `0.28 KB` (on-device, the MCU encodes only **1 new epoch every 30s** and slides it into this tiny buffer!)
    - RTOS / stack / interpreter overhead = `20.0 KB`
    - **Total SRAM $\approx 119.7\text{ KB} / 256\text{ KB}$** (**>136 KB free headroom** $\to$ genuinely fits!).
  - **Compute:** Computes exact MACs per 30s epoch (`~14.6 M MACs`) and marks physical Arduino ms/epoch as pending real hardware-in-the-loop (HIL) benchmarking.
- **Files Written:** `checkpoints/edge_profile.pkl`

---

### Section 10 — Honest 20-Subject Benchmarks & SOTA Comparison
- **What It Does:** Compares our **20-subject LOSO mean** (`FP32`, `INT8 PTQ`, `INT8 QAT`) against human experts and published baselines (`DeepSleepNet`, `TinySleepNet`, `SeqSleepNet` on EESM19 `73.0%` Macro F1).
- **Files Written:** `figures/benchmark_comparison.png`

---

### Section 11 — Real Statistical Validation, Logging & Dashboard
- **What It Does:**
  1. Runs **real paired Wilcoxon signed-rank tests** across all 20 subjects comparing `FP32 vs INT8 QAT` and `FP32 vs INT8 PTQ` (to verify statistically whether INT8 quantization causes significant accuracy loss).
  2. Evaluates Hypothesis **H1** (`20-fold mean F1 drop < 2%`) and Hypothesis **H3** (`Total SRAM < 256 KB` and `Flash < 1024 KB`).
  3. Automatically appends/updates a summary row in `outputs/experiments_log.csv` and saves `metrics_summary.json` + the 300 DPI `dashboard.png`.
- **Files Written:** `metrics_summary.json`, `outputs/experiments_log.csv`, `figures/dashboard.png`

---

## 📁 How `outputs/` Is Organized Across Iterations

```text
work_dir/outputs/
├── experiments_log.csv                    <-- Master comparison table across ALL runs!
│
├── cache/                                 <-- Shared heavy dataset files (reused across runs)
│   ├── loaded_cross_ear_256hz.pkl
│   ├── processed_cross_ear_256hz.pkl
│   ├── loaded_cross_ear_128hz.pkl
│   └── processed_cross_ear_128hz.pkl
│
└── runs/
    ├── v1_256hz_single_epoch_baseline/    <-- Preserved v1 baseline (never overwritten)
    │   ├── figures/                       (11 PNGs)
    │   ├── checkpoints/                   (20 loso_fold_*.pkl, loso_results.pkl, etc.)
    │   └── models/                        (tiny_eeg_sleep_int8.pt)
    │
    └── v2_128hz_dwtcn_ctx3_leakfree/      <-- Current leak-free 128Hz v2 run
        ├── config.json                    (exact hyperparameters for this run)
        ├── metrics_summary.json           (final FP32, PTQ, QAT, SRAM, Wilcoxon p)
        ├── figures/                       (300 DPI white-background plots)
        ├── checkpoints/                   (loso_fold_*.pkl, qat_fold_*.pkl, edge_profile.pkl)
        └── models/                        (tiny_eeg_sleep_fp32.pt, tiny_eeg_sleep_int8.pt)
```

### How to Run a New Experiment / Ablation Safely
1. Keep `outputs/cache/` untouched so you never have to re-read the 80 raw EEG `.fdt` files again.
2. In **Section 0** of `ear_eeg_sleep_staging.py`, simply change **`RUN_NAME`** (for example, `'v3_128hz_ctx5_focal'`) and tweak any parameter (`CONTEXT_LEN = 5`, `USE_CROSS_EAR = False`, etc.).
3. Run the notebook. All new checkpoints, figures, and models automatically save into `outputs/runs/v3_128hz_ctx5_focal/`, and a new comparison row is added to `outputs/experiments_log.csv`—without losing any previous run!
