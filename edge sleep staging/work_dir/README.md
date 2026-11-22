# Ear-EEG Sleep Staging (`ear_eeg_sleep_staging.py`)

VS Code interactive Python notebook (`#%%` cells) for 5-stage cross-ear EEG sleep classification on OpenNeuro **EESM19 (`ds005185`)**, compressed to INT8 for **Arduino Nano 33 BLE Sense** (<256 KB SRAM, <1024 KB Flash).

For a plain-English explanation of every section, see [ALGORITHM.md](ALGORITHM.md).

---

## 1. Prerequisites & Setup

1. **Interpreter:** Select `d:\Project\Research\.venv\Scripts\python.exe` in VS Code (with PyTorch XPU / CUDA / CPU already installed).
2. **Dataset Path:** Verify your local EESM19 folder path in **Section 2**:
   ```python
   loader = RealEEGSleepLoader(data_root=r'E:\EESM19\ds005185-download', ...)
   ```
3. **Running Cells:** Open `ear_eeg_sleep_staging.py` in VS Code and click **Run Cell** above each `#%%` block (or `Shift+Enter` inside a cell).

---

## 2. Key Settings (Top of Section 0)

| Variable             | Default                            | Purpose                                                                                  |
| :------------------- | :--------------------------------- | :--------------------------------------------------------------------------------------- |
| `RUN_NAME`         | `'v2_128hz_dwtcn_ctx3_leakfree'` | Base folder name for this experiment                                                     |
| `DEV_FOLD_INDICES` | `[7, 8, 13]`                     | `None` = all 20 folds; `[7, 8, 13]` = fast 3-fold dev subset (`Sub 008, 009, 014`) |
| `RUN_QAT`          | `False`                          | `False` = ~30s INT8 PTQ only; `True` = PTQ + validation-guarded CPU QAT              |
| `QAT_LR`           | `5e-5`                           | Conservative QAT fine-tuning learning rate                                               |
| `FS`               | `128`                            | Sampling rate in Hz (`EPOCH_SAMP = 3,840`)                                             |
| `CONTEXT_LEN`      | `3`                              | Odd context window (`1`, `3`, or `5` adjacent epochs)                              |
| `CLASS_WT_POWER`   | `0.5`                            | Class-imbalance dampening exponent                                                       |
| `FORCE_RERUN`      | `False`                          | `True` recomputes current `ACTIVE_RUN_NAME` artifacts                                |

> **Auto-Isolation:** When `DEV_FOLD_INDICES` is set, `_dev3f` is automatically appended to `ACTIVE_RUN_NAME`, so dev runs never overwrite full 20-fold runs.

---

## 3. How to Run (3 Common Workflows)

### A. First Run (or after changing `FS` / `USE_CROSS_EAR`)

Run Sections **`0 → 11`** in order. Sections **2** and **4** populate the heavy shared dataset caches in `outputs/cache/` so future runs can skip them.

### B. Fast Dev Experiment (~5–8 min, reusing 128 Hz cache)

1. In **Section 0**, set a new `RUN_NAME` (e.g. `'v3_128hz_ctx5'`) and adjust your hyperparameters.
2. Keep `DEV_FOLD_INDICES = [7, 8, 13]` and `RUN_QAT = False`.
3. Run Sections **`0 → 6 → 7 → 8 → 9 → 10 → 11`**.
4. Compare your 3-fold F1 against the `v2` dev anchor (**68.4% FP32 / 68.8% PTQ**).

### C. Full 20-Fold Paper Run

1. Once a Tier-1 dev config wins, set `DEV_FOLD_INDICES = None` (and optionally `RUN_QAT = True` to evaluate QAT).
2. Run Sections **`0 → 6 → 7 → 8 → 9 → 10 → 11`**.

---

## 4. Outputs Layout

```text
work_dir/outputs/
├── experiments_log.csv                    # Master comparison table across all runs
├── cache/                                 # Shared raw/preprocessed dataset caches
└── runs/<ACTIVE_RUN_NAME>/
    ├── config.json                        # Exact hyperparameters used
    ├── metrics_summary.json               # Final summary metrics & hypothesis checks
    ├── checkpoints/                       # Per-fold checkpoints (auto-resumes if stopped)
    ├── figures/                           # 300 DPI white-background PNG plots
    └── models/                            # Saved FP32 & INT8 state dicts
```

---

## 5. Quick Troubleshooting

* **`NameError` after kernel restart:** Run **Section 0** and **Section 6** first, then re-run your target cell (completed folds load from `checkpoints/`).
* **Section 8 takes hours:** Ensure `RUN_QAT = False` during experimentation so it runs ~30s Static PTQ instead of CPU QAT.
* **Shape / sample mismatch after editing `FS`:** Re-run **Section 2** and **Section 4** to rebuild `outputs/cache/`.
* **Documentation rule:** Whenever you modify code, hyperparameters, splits, or paths in `ear_eeg_sleep_staging.py`, update [ALGORITHM.md](ALGORITHM.md) to match.




Here are the sections in [ear_eeg_sleep_staging.py](<file:///d:/Project/Research/edge%20sleep%20staging/work_dir/ear_eeg_sleep_staging.py>) ranked by importance for your daily workflow:

---

### 1. Core Pipeline (Must Run for Every Experiment)

If your 128 Hz cache already exists, these **6 sections** are the only ones you need to run (`0 → 6 → 7 → 8 → 9 → 11`):

| Section              | What It Does                                                  | Why It's Essential                                                                                                                                                    |
| :------------------- | :------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Section 0**  | Setup, `RUN_NAME`, `DEV_FOLD_INDICES`, `RUN_QAT`, paths | Sets all hyperparameters, device (`xpu`/`cpu`), and isolates run outputs (`_dev3f` vs full).                                                                    |
| **Section 6**  | `TinyEEGSleep` v2 + `SleepContextDataset` + train helpers | Defines the neural network (`122,725` params) and context loader required by Sections 7 & 8.                                                                        |
| **Section 7**  | Leak-free 3-way LOSO training (`17 train / 2 val / 1 test`) | Trains the FP32 models and saves per-fold checkpoints.                                                                                                                |
| **Section 8**  | INT8 PTQ & (optional) Validation-Guarded QAT                  | Compresses FP32 $\to$ INT8 (~30s when `RUN_QAT=False`).                                                                                                           |
| **Section 9**  | Analytical Arduino Nano 33 BLE Sense profiler                 | Verifies Flash (`125.8 KB`) and SRAM (`119.7 / 256 KB`) fit via PyTorch hooks.                                                                                    |
| **Section 11** | Wilcoxon tests, H1/H3 verdict, CSV logging, dashboard         | Logs final scores to [experiments_log.csv](<file:///d:/Project/Research/edge%20sleep%20staging/work_dir/outputs/experiments_log.csv>) and generates `dashboard.png`. |

---

### 2. One-Time Data Prep (Skip Unless `FS` or `USE_CROSS_EAR` Changes)

| Section             | What It Does                                                      | When to Re-Run                                                                                    |
| :------------------ | :---------------------------------------------------------------- | :------------------------------------------------------------------------------------------------ |
| **Section 2** | Loads raw EESM19 EDFs, cross-ear subtraction, resamples to `FS` | **Only** if you change `FS` or `USE_CROSS_EAR` (writes `outputs/cache/loaded_*.pkl`). |
| **Section 4** | Notch/bandpass filtering, artifact mask, per-epoch z-score        | **Only** after re-running Section 2 (writes `outputs/cache/processed_*.pkl`).             |

---

### 3. Paper Figures & Benchmarks (Optional / Run for Final Paper)

| Section              | What It Produces                                                                                 |
| :------------------- | :----------------------------------------------------------------------------------------------- |
| **Section 1**  | `dataset_overview.png` (EESM19 BIDS metadata summary)                                          |
| **Section 3**  | `class_distribution.png`, `hypnogram.png`, `raw_eeg_per_stage.png`                         |
| **Section 5**  | `spectrograms.png`, `band_powers.png` (verifies Delta/Theta/Alpha/Spindle markers)           |
| **Section 10** | `benchmark_comparison.png` (compares your run against DeepSleepNet, TinySleepNet, SeqSleepNet) |
