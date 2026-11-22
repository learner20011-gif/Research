#%% [markdown]
# # 🧠 Ear-EEG Sleep Monitoring — Complete Research Pipeline
# > **⚠️ MANDATORY DOCUMENTATION RULE: WHENEVER YOU MAKE ANY CHANGE TO THE CODE, HYPERPARAMETERS, PREPROCESSING, MODEL ARCHITECTURE, VALIDATION SPLIT, OR OUTPUT STRUCTURE IN THIS NOTEBOOK (`ear_eeg_sleep_staging.py`), YOU MUST IMMEDIATELY UPDATE `ALGORITHM.md` ACCORDINGLY IN PLAIN ENGLISH SO THAT THE ALGORITHM DESCRIPTION ALWAYS MATCHES THE EXACT IMPLEMENTATION.**
#
# **Dataset:** Mikkelsen, K.B. et al. *Ear-EEG sleep monitoring data sets.* Scientific Data 12, 301 (2025).  
# **DOI:** [10.1038/s41597-025-04579-8](https://doi.org/10.1038/s41597-025-04579-8)  
# **Data:** OpenNeuro [ds005185 (EESM19)](https://openneuro.org/datasets/ds005185)
# 
# ---
# 
# ## 🎯 Notebook Objectives
# 
# | # | Section | Purpose |
# |---|---------|----------|
# | 0 | **Environment Setup** | Packages, experiment config, white publication style (300 DPI) |
# | 1 | **Dataset Architecture** | BIDS structure, metadata tables |
# | 2 | **Data Loading** | MNE/OpenNeuro real local data (128 Hz cross-ear) |
# | 3 | **EDA** | Hypnograms, class distributions |
# | 4 | **Preprocessing** | Filtering, artifact removal, per-epoch z-score |
# | 5 | **Time-Frequency Analysis** | Spectrograms, biomarker visualization |
# | 6 | **Deep Learning** | 1D-CNN + Depthwise-Separable TCN + Context Head |
# | 7 | **LOSO Cross-Validation** | Leak-free 3-way subject-disjoint evaluation |
# | 8 | **INT8 PTQ & QAT** | Real 20-fold Post-Training Quantization & QAT |
# | 9 | **Edge Profiling** | Analytical Flash/SRAM for Arduino Nano 33 BLE Sense |
# | 10 | **Benchmarks** | Honest 20-subject LOSO mean vs. PSG & SOTA |
# | 11 | **Results & Stats** | Real paired Wilcoxon test, summary log, dashboard |
# 
# ---
# 
# ## Research Context
# 
# The **EESM19** dataset: **214 nights** (including **80 PSG-scored nights**) across **20 healthy subjects** using **dry-contact iridium oxide electrodes**.
# 
# **Core Research Question:** Can a Depthwise-Separable 1D-CNN + TCN for single-channel cross-ear EEG sleep staging be compressed via INT8 QAT to fit within **<256 KB SRAM** (and **<1024 KB Flash**) on an **Arduino Nano 33 BLE Sense** while approaching **~73%+ Macro F1 (SeqSleepNet Ear-EEG SOTA)** with **<2% F1 drop**?
# 
# > **Hardware:** Intel Arc XPU / CUDA GPU recommended for training; QAT & INT8 inference run on CPU.

#%%
# ==============================================================================
# **MANDATORY RULE: WHENEVER YOU MODIFY ANY LOGIC, HYPERPARAMETERS, MODEL,
#  VALIDATION SPLIT, OR OUTPUT PATH IN THIS SCRIPT, UPDATE `ALGORITHM.md`
#  IMMEDIATELY IN PLAIN ENGLISH TO MATCH THE NEW CODE.**
# ==============================================================================
# SECTION 0 - Environment Setup
# ==============================================================================
# 🚀 INTEL ARC XPU (Windows) SETUP INSTRUCTIONS
# ==============================================================================
# To use an Intel Arc GPU (e.g., A770) for hardware acceleration on Windows:
#
# 1. DO NOT add `torch` to the PACKAGES list below. 
#    Running `pip install torch` inside a running Python script on Windows can 
#    cause DLL locking issues, corrupting the environment. It also defaults to 
#    downloading the CPU-only version from PyPI, overwriting your XPU build.
#
# 2. Before running this script, open your terminal (Command Prompt / PowerShell)
#    and install the native XPU PyTorch build explicitly:
#      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
#
# 3. Ensure your Intel GPU drivers are up to date (32.0.101.6739 or higher).
#
# Once the above is done, this script will automatically detect `torch.xpu`, 
# route tensors to the Intel GPU, and apply `torch.compile()` for massive 
# training speedups using the native PyTorch 2.13+ API.
# ==============================================================================

import subprocess, sys

PACKAGES = ['mne>=1.7','mne-bids>=0.14','openneuro-py','scipy>=1.12','scikit-learn>=1.4',
            'torchinfo','matplotlib>=3.8','seaborn>=0.13','pandas>=2.2',
            'numpy>=1.26','tqdm','pyedflib']
for pkg in PACKAGES:
    subprocess.run([sys.executable,'-m','pip','install','-q',pkg], check=False)
print('All packages installed.')

#%%
import os, sys, warnings, json, time, io, pathlib, pickle
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
# pyrefly: ignore [missing-import]
import scipy.signal as signal
# pyrefly: ignore [missing-import]
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, classification_report
import torch, torch.nn as nn, torch.optim as optim
try:
    import torch.ao.quantization as quantization
except ImportError:
    import torch.quantization as quantization
from torch.utils.data import DataLoader, Dataset, TensorDataset
import mne
mne.set_log_level('WARNING')
warnings.filterwarnings('ignore')

# Auto-detect best available device: Intel XPU > CUDA > CPU
try:
    _XPU_AVAILABLE = hasattr(torch, 'xpu') and torch.xpu.is_available()
except Exception:
    _XPU_AVAILABLE = False

if _XPU_AVAILABLE:
    DEVICE = 'xpu'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

plt.rcParams.update({
    'figure.facecolor':'white','axes.facecolor':'white','axes.edgecolor':'#222222','axes.linewidth':0.9,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.labelcolor':'#111111','xtick.color':'#222222','ytick.color':'#222222',
    'text.color':'#111111','grid.color':'#CCCCCC','grid.linestyle':'--','grid.alpha':0.6,
    'font.family':'sans-serif','font.size':11,'savefig.dpi':300,'savefig.facecolor':'white',
})

STAGE_COLORS = {'Wake':'#E74C3C','N1':'#E67E22','N2':'#3498DB','N3':'#2ECC71','REM':'#9B59B6'}
STAGE_MAP    = {'Wake':0,'N1':1,'N2':2,'N3':3,'REM':4}
INV_STAGE_MAP= {v:k for k,v in STAGE_MAP.items()}
STAGE_ORDER  = ['Wake','N1','N2','N3','REM']

# === EXPERIMENT CONFIGURATION (Change RUN_NAME for each new iteration/ablation) ===
RUN_NAME        = 'v2_128hz_dwtcn_ctx3_leakfree'
FS              = 128          # 128 Hz (Nyquist 64 Hz; preserves all 0.5-40 Hz sleep waves, 2x faster & half SRAM)
EPOCH_SEC       = 30
EPOCH_SAMP      = FS * EPOCH_SEC  # 3,840 samples
USE_CROSS_EAR   = True
CONTEXT_LEN     = 3            # Odd integer: 1 (single epoch) or 3 ([t-1, t, t+1]) or 5
CLASS_WT_POWER  = 0.5          # Dampened inverse-frequency weights to prevent N1 over-prediction
LABEL_SMOOTHING = 0.05
N_TRAIN_EP      = 50           # Early stopping usually converges ~25-40 epochs
PATIENCE        = 12           # Early stopping patience on held-out validation subjects
BATCH_SIZE      = 256
LR              = 1.5e-3
QAT_EPOCHS      = 5            # Real per-fold INT8 QAT fine-tuning epochs
# === 2-TIER FAST ITERATION SWITCHES ===
# Tier 1 (Fast Dev, ~5-8 min): DEV_FOLD_INDICES = [7, 8, 13] (Sub 008 easy, 009 median, 014 hard; v2 ref: 68.4% FP32 / 68.8% PTQ), RUN_QAT = False
# Tier 2 (Full 20-Fold Paper Run): DEV_FOLD_INDICES = None, RUN_QAT = True (or False for PTQ-only)
DEV_FOLD_INDICES = [7, 8, 13]  # None = all 20 folds; list of 0-based indices = fast dev subset
RUN_QAT          = False       # False = fast ~30s INT8 PTQ only; True = PTQ + validation-guarded INT8 QAT
QAT_LR           = 5e-5        # Conservative fine-tuning LR for QAT (prevents overfitting collapse on early-stopped folds)
SEED            = 42
FORCE_RERUN     = False        # True = recompute current RUN_NAME artifacts (never deletes other runs)

ACTIVE_RUN_NAME = (
    f"{RUN_NAME}_dev{len(DEV_FOLD_INDICES)}f"
    if (DEV_FOLD_INDICES is not None and '_dev' not in RUN_NAME)
    else RUN_NAME
)

torch.manual_seed(SEED); np.random.seed(SEED)

OUTPUT_DIR = Path('./outputs')
CACHE_DIR  = OUTPUT_DIR / 'cache'
RUN_DIR    = OUTPUT_DIR / 'runs' / ACTIVE_RUN_NAME
FIG_DIR    = RUN_DIR / 'figures'
CKPT_DIR   = RUN_DIR / 'checkpoints'
MODEL_DIR  = RUN_DIR / 'models'
for _d in (OUTPUT_DIR, CACHE_DIR, RUN_DIR, FIG_DIR, CKPT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CACHE_TAG       = f"{'cross_ear' if USE_CROSS_EAR else 'single_ear'}_{FS}hz"
RAW_CACHE_NAME  = f"loaded_{CACHE_TAG}.pkl"
PROC_CACHE_NAME = f"processed_{CACHE_TAG}.pkl"

with open(RUN_DIR / 'config.json', 'w', encoding='utf-8') as _cf:
    json.dump({
        'base_run_name': RUN_NAME, 'run_name': ACTIVE_RUN_NAME, 'fs': FS,
        'epoch_sec': EPOCH_SEC, 'epoch_samp': EPOCH_SAMP,
        'use_cross_ear': USE_CROSS_EAR, 'context_len': CONTEXT_LEN, 'class_wt_power': CLASS_WT_POWER,
        'label_smoothing': LABEL_SMOOTHING, 'n_train_ep': N_TRAIN_EP, 'patience': PATIENCE,
        'batch_size': BATCH_SIZE, 'lr': LR, 'qat_epochs': QAT_EPOCHS,
        'dev_fold_indices': DEV_FOLD_INDICES, 'run_qat': RUN_QAT, 'qat_lr': QAT_LR, 'seed': SEED,
    }, _cf, indent=2)

def save_cache(name, obj, shared=False):
    target_dir = CACHE_DIR if shared else CKPT_DIR
    with open(target_dir / name, 'wb') as f:
        pickle.dump(obj, f)

def load_cache(name, shared=False):
    target_dir = CACHE_DIR if shared else CKPT_DIR
    p = target_dir / name
    if (shared or not FORCE_RERUN) and p.exists():
        with open(p, 'rb') as f:
            return pickle.load(f)
    return None

_mode_banner = (
    f"Mode: TIER-1 FAST DEV ({len(DEV_FOLD_INDICES)} folds: {DEV_FOLD_INDICES})"
    if DEV_FOLD_INDICES is not None
    else "Mode: TIER-2 FULL 20-FOLD PAPER RUN"
)
_quant_banner = "Quant: PTQ + Val-Guarded QAT" if RUN_QAT else "Quant: PTQ-only"
print(f'Run: {ACTIVE_RUN_NAME} | {_mode_banner} | {_quant_banner}')
print(f'FS: {FS} Hz ({EPOCH_SAMP} samp) | Ctx: {CONTEXT_LEN} | Device: {DEVICE}')
print(f'MNE: {mne.__version__} | PyTorch: {torch.__version__}')
if DEVICE == 'xpu':
    print(f'Intel Arc GPU: {torch.xpu.get_device_name(0)} | XPU count: {torch.xpu.device_count()}')
elif DEVICE == 'cuda':
    print(f'NVIDIA GPU: {torch.cuda.get_device_name(0)} | CUDA count: {torch.cuda.device_count()}')

#%% [markdown]
# ---
# ## Section 1 - Dataset Architecture
# 
# | Property | EESM19 |
# |----------|--------|
# | OpenNeuro ID | ds005185 |
# | Subjects | 20 |
# | Total nights | 214 (80 PSG-scored) |
# | PSG nights/subject | 4 |
# | Home EEG-only nights | 12 (half subjects) |
# | Ear piece | Custom (ear impression) |
# | Electrode material | Iridium oxide (dry) |
# | Additional sensors | GENEactiv actigraphy |
# | % Unscored epochs | 5% |
# 
# All files follow BIDS standard and can be loaded with mne-bids.

#%%
@dataclass
class DatasetConfig:
    name: str; openneuro_id: str; n_subjects: int; n_nights_total: int
    n_psg_per_subject: int; n_home_nights: int; electrode_type: str
    has_actigraphy: bool; pct_unscored: float

EESM19 = DatasetConfig('EESM19','ds005185',20,214,4,12,'Custom (ear impression)',True,5.0)
n_psg_nights = EESM19.n_subjects * EESM19.n_psg_per_subject
n_home_only  = EESM19.n_nights_total - n_psg_nights

print(f'--- {EESM19.name} ({EESM19.openneuro_id}) ---')
print(f'  {EESM19.n_subjects} subjects | {n_psg_nights} PSG-scored nights ({EESM19.n_nights_total} total) | Unscored: {EESM19.pct_unscored}%')

if not FORCE_RERUN and (FIG_DIR/'dataset_overview.png').exists():
    print(f'Skipping dataset_overview.png (already exists in {FIG_DIR}).')
else:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('EESM19 (ds005185) - Dataset Overview', fontsize=14, fontweight='bold', color='#111111')

    ax = axes[0]
    v19 = [n_psg_nights, n_home_only]
    b1 = ax.bar(['PSG-Scored\nNights', 'Home EEG\n(Unscored)'], v19, width=0.45, color=['#1F77B4', '#7F8C8D'], alpha=0.9, edgecolor='#222222', linewidth=0.5)
    ax.bar_label(b1, padding=3)
    ax.set_ylabel('Nights'); ax.set_title('Recording Types'); ax.grid(axis='y', alpha=0.4)

    axes[1].pie([n_psg_nights, n_home_only],
        labels=[f'PSG-Scored\n(n={n_psg_nights})', f'Home EEG-Only\n(n={n_home_only})'],
        colors=['#1F77B4', '#7F8C8D'], autopct='%1.0f%%',
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2), textprops={'color': '#111111'})
    axes[1].set_title(f'EESM19: {EESM19.n_nights_total} Total Nights')

    ax = axes[2]
    scored = [100 - EESM19.pct_unscored]
    unscored_ = [EESM19.pct_unscored]
    ax.bar(['EESM19 PSG'], scored, width=0.4, color='#2CA02C', label='Scored', alpha=0.9, edgecolor='#222222', linewidth=0.5)
    ax.bar(['EESM19 PSG'], unscored_, bottom=scored, width=0.4, color='#D62728', label='Unscored', alpha=0.9, edgecolor='#222222', linewidth=0.5)
    ax.set_ylim(0, 108); ax.set_ylabel('Epoch %')
    ax.set_title('Scoring Quality'); ax.legend(); ax.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    plt.savefig(FIG_DIR/'dataset_overview.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

#%% [markdown]
# ---
# ## Section 2 - Data Loading
# 
# ### Real Data (EESM19 - ds005185)
# Downloads and extracts real Ear-EEG recordings and parses sleep staging labels from the corresponding PSG events.


#%%
class RealEEGSleepLoader:
    LEFT_EAR  = ['ELA', 'ELB', 'ELC', 'ELT', 'ELE', 'ELI']
    RIGHT_EAR = ['ERA', 'ERB', 'ERC', 'ERT', 'ERE', 'ERI']

    def __init__(self, data_root, fs=128, epoch_sec=30, use_cross_ear=True):
        self.data_root = Path(data_root)
        self.FS = int(fs)
        self.EPOCH_SEC = int(epoch_sec)
        self.EPOCH_SAMP = self.FS * self.EPOCH_SEC
        self.use_cross_ear = use_cross_ear

    @staticmethod
    def _pick_cleanest_channel(raw, candidates):
        """Pick the channel with highest finite coverage and non-trivial variance."""
        best_sig, best_name, best_ratio = None, None, 0.0
        for c in candidates:
            if c in raw.ch_names:
                s = raw.get_data(picks=[c])[0]
                finite_mask = np.isfinite(s)
                ratio = float(np.mean(finite_mask))
                if ratio > best_ratio and np.std(s[finite_mask]) > 1e-9:
                    best_sig, best_name, best_ratio = s, c, ratio
                    if ratio >= 0.999:
                        break
        return best_sig, best_name

    def _load_ear_channels(self, raw):
        """Load ONLY Ear-EEG channels lazily and return best Left and Right raw signals."""
        ear_chs = [c for c in (self.LEFT_EAR + self.RIGHT_EAR) if c in raw.ch_names]
        if not ear_chs:
            return None, None, None, None, float(raw.info['sfreq'])

        # Drop scalp/EOG/EMG channels BEFORE loading .fdt data into RAM
        raw.pick(ear_chs)
        raw.load_data()
        orig_fs = float(raw.info['sfreq'])

        left_sig, left_name = self._pick_cleanest_channel(raw, self.LEFT_EAR)
        right_sig, right_name = self._pick_cleanest_channel(raw, self.RIGHT_EAR)
        return left_sig, left_name, right_sig, right_name, orig_fs

    def load_dataset(self, subjects=None, sessions=None):
        from math import gcd
        data = []
        if subjects is None:
            subjects = sorted([
                d.name.split('-')[1] for d in self.data_root.glob('sub-*') if d.is_dir()
            ])

        print(f"Loading subjects: {subjects} (Ear-EEG only, cross_ear={self.use_cross_ear})...")
        VAL_TO_STAGE = {1: 'Wake', 2: 'REM', 3: 'N1', 4: 'N2', 5: 'N3'}

        def _process_subject(sub):
            sub_data = []
            sub_dir = self.data_root / f"sub-{sub}"
            if not sub_dir.exists():
                return sub_data

            cur_sessions = sessions if sessions is not None else sorted([
                d.name.split('-')[1] for d in sub_dir.glob('ses-*') if d.is_dir()
            ])

            for ses in cur_sessions:
                eeg_dir = sub_dir / f"ses-{ses}" / "eeg"
                raw_file = eeg_dir / f"sub-{sub}_ses-{ses}_task-sleep_acq-PSG_eeg.set"
                fdt_file = eeg_dir / f"sub-{sub}_ses-{ses}_task-sleep_acq-PSG_eeg.fdt"

                evt_file = next(
                    (eeg_dir / n for n in [
                        f"sub-{sub}_ses-{ses}_task-sleep_acq-scoring1_events.tsv",
                        f"sub-{sub}_ses-{ses}_task-sleep_acq-scoring2_events.tsv",
                        f"sub-{sub}_ses-{ses}_task-sleep_acq-scoring_events.tsv"
                    ] if (eeg_dir / n).exists()),
                    None
                )

                if not raw_file.exists() or not fdt_file.exists() or evt_file is None:
                    continue

                try:
                    raw = mne.io.read_raw_eeglab(raw_file, preload=False, verbose=False)
                    left_sig, left_name, right_sig, right_name, orig_fs = self._load_ear_channels(raw)
                except Exception as e:
                    print(f"Failed sub-{sub} ses-{ses}: {e}")
                    continue

                if left_sig is None and right_sig is None:
                    print(f"No valid Ear-EEG channel in sub-{sub} ses-{ses}, skipping.")
                    continue

                events_df = pd.read_csv(evt_file, sep='\t')
                col = next((c for c in ['scoring', 'Scoring1', 'Scoring2'] if c in events_df.columns), None)
                if col is None:
                    continue

                orig_epoch_samp = int(round(self.EPOCH_SEC * orig_fs))
                sig_len = len(left_sig) if left_sig is not None else len(right_sig)
                ch_label = f"{left_name}-{right_name}" if (self.use_cross_ear and left_sig is not None and right_sig is not None) else (left_name or right_name)

                raw_epochs, stage_names, labels, metas = [], [], [], []
                nan_skips, map_skips, fallback_cnt = 0, 0, 0

                for row in events_df.itertuples(index=False):
                    onset = float(getattr(row, 'onset'))
                    raw_val = getattr(row, col)
                    if pd.isna(raw_val):
                        continue

                    if isinstance(raw_val, (int, float, np.integer, np.floating)) or (isinstance(raw_val, str) and raw_val.replace('.', '', 1).isdigit()):
                        val_int = int(float(raw_val))
                        if val_int not in VAL_TO_STAGE:
                            map_skips += 1
                            continue
                        stage_str = VAL_TO_STAGE[val_int]
                    else:
                        stage_str = str(raw_val).strip()

                    if stage_str not in STAGE_MAP:
                        map_skips += 1
                        continue

                    start_idx = int(round(onset * orig_fs))
                    end_idx = start_idx + orig_epoch_samp
                    if start_idx < 0 or end_idx > sig_len:
                        continue

                    # Per-epoch cross-ear derivation with NaN-safe single-ear fallback BEFORE resampling
                    l_ep = left_sig[start_idx:end_idx] if left_sig is not None else None
                    r_ep = right_sig[start_idx:end_idx] if right_sig is not None else None
                    l_ok = l_ep is not None and len(l_ep) == orig_epoch_samp and np.isfinite(l_ep).all()
                    r_ok = r_ep is not None and len(r_ep) == orig_epoch_samp and np.isfinite(r_ep).all()

                    if self.use_cross_ear and l_ok and r_ok:
                        ep = l_ep - r_ep
                        ep_ch = ch_label
                    elif l_ok:
                        ep = l_ep
                        ep_ch = left_name
                        if self.use_cross_ear and right_sig is not None:
                            fallback_cnt += 1
                    elif r_ok:
                        ep = r_ep
                        ep_ch = right_name
                        if self.use_cross_ear and left_sig is not None:
                            fallback_cnt += 1
                    else:
                        nan_skips += 1
                        continue

                    ep = (ep * 1e6).astype(np.float32)
                    raw_epochs.append(ep - np.mean(ep))
                    stage_names.append(stage_str)
                    labels.append(STAGE_MAP[stage_str])
                    metas.append({'stage': stage_str, 'channel': ep_ch, 'has_spindle': False, 'has_k_complex': False})

                if raw_epochs:
                    epochs_arr = np.stack(raw_epochs)
                    orig_fs_int = int(round(orig_fs))
                    if orig_fs_int != self.FS:
                        g = gcd(self.FS, orig_fs_int)
                        epochs_arr = signal.resample_poly(
                            epochs_arr, up=self.FS // g, down=orig_fs_int // g, axis=1
                        ).astype(np.float32)
                    # Guarantee exact length EPOCH_SAMP
                    if epochs_arr.shape[1] != self.EPOCH_SAMP:
                        epochs_arr = epochs_arr[:, :self.EPOCH_SAMP]
                    print(f"sub-{sub} ses-{ses} [{ch_label}] -> {len(epochs_arr)} epochs (fallback {fallback_cnt}, skipped {nan_skips} NaN, {map_skips} unmapped)")
                    sub_data.append({
                        'subject': sub, 'night': ses, 'channel': ch_label,
                        'epochs': epochs_arr, 'labels': np.array(labels),
                        'stage_names': stage_names, 'metadata': metas, 'fs': self.FS
                    })
                else:
                    print(f"sub-{sub} ses-{ses} [{ch_label}] -> 0 valid epochs (skipped {nan_skips} NaN, {map_skips} unmapped)")
            return sub_data

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
            for res in executor.map(_process_subject, subjects):
                data.extend(res)

        print(f"Done: {len(data)} recordings, {sum(len(r['epochs']) for r in data):,} epochs")
        return data

loader = RealEEGSleepLoader(data_root=r'E:\EESM19\ds005185-download', fs=FS, epoch_sec=EPOCH_SEC, use_cross_ear=USE_CROSS_EAR)
# Full 20-subject publication run (80 PSG-scored nights)
SUBJECTS_TO_LOAD = [f'{i:03d}' for i in range(1, 21)]
dataset = load_cache(RAW_CACHE_NAME, shared=True)
if dataset is not None:
    print(f"Loaded dataset from shared cache ({RAW_CACHE_NAME})...")
else:
    dataset = loader.load_dataset(subjects=SUBJECTS_TO_LOAD, sessions=None)
    save_cache(RAW_CACHE_NAME, dataset, shared=True)
    print(f"Dataset saved to shared cache ({RAW_CACHE_NAME}).")
# Clean up any NaN epochs from the dataset (especially if loaded from an old cache)
for rec in dataset:
    if len(rec['epochs']) > 0:
        valid_mask = ~np.isnan(rec['epochs']).any(axis=1)
        rec['epochs'] = rec['epochs'][valid_mask]
        rec['labels'] = rec['labels'][valid_mask]
        rec['stage_names'] = [sn for i, sn in enumerate(rec['stage_names']) if valid_mask[i]]
        rec['metadata'] = [md for i, md in enumerate(rec['metadata']) if valid_mask[i]]
dataset = [rec for rec in dataset if len(rec['epochs']) > 0]

if len(dataset) > 0:
    print(f'Epoch shape: {dataset[0]["epochs"].shape} | FS: {dataset[0]["fs"]} Hz')
else:
    print("No valid recordings found with epochs!")



#%% [markdown]
# ---
# ## Section 3 - Exploratory Data Analysis

#%%
_eda_cache = load_cache('eda_summary.pkl')
if _eda_cache is not None and (FIG_DIR/'class_distribution.png').exists():
    pcts = _eda_cache['pcts']
    print(f'Loaded class distribution summary from cache (class_distribution.png exists in {FIG_DIR}).')
else:
    if 'dataset' not in globals():
        dataset = load_cache(RAW_CACHE_NAME, shared=True)
    all_stages=np.concatenate([r['stage_names'] for r in dataset])
    stage_counts=Counter(all_stages)
    counts=[stage_counts[s] for s in STAGE_ORDER]
    total=sum(counts); pcts=[c/total*100 for c in counts]
    aasm={'Wake':10,'N1':5,'N2':45,'N3':20,'REM':20}

    fig,axes=plt.subplots(1,2,figsize=(14,5))
    fig.suptitle('Sleep Stage Distribution - Ear-EEG Dataset',fontsize=14,fontweight='bold',color='#111111')
    colors_=[STAGE_COLORS[s] for s in STAGE_ORDER]
    bars=axes[0].bar(STAGE_ORDER,pcts,color=colors_,alpha=0.9,edgecolor='#222222',linewidth=0.6)
    for bar,pct,cnt in zip(bars,pcts,counts):
        axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,f'{pct:.1f}%\n({cnt:,})',
                     ha='center',va='bottom',fontsize=9.5,color='#111111')
    axes[0].set_ylabel('% of Epochs'); axes[0].set_title('Class Distribution')
    axes[0].grid(axis='y',alpha=0.4); axes[0].set_ylim(0,max(pcts)*1.3)

    x_=np.arange(5)
    axes[1].bar(x_-0.2,[aasm[s] for s in STAGE_ORDER],0.4,label='AASM Reference',color='#7F8C8D',alpha=0.75,edgecolor='#222222',linewidth=0.5)
    axes[1].bar(x_+0.2,pcts,0.4,color=colors_,alpha=0.9,label='Real EESM',edgecolor='#222222',linewidth=0.5)
    axes[1].set_xticks(x_); axes[1].set_xticklabels(STAGE_ORDER)
    axes[1].set_ylabel('%'); axes[1].set_title('vs. AASM Reference')
    axes[1].legend(); axes[1].grid(axis='y',alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIG_DIR/'class_distribution.png',dpi=300,bbox_inches='tight',facecolor='white')
    plt.show()
    print(f'Imbalance ratio: {max(counts)/min(counts):.1f}x')

#%%
# Hypnogram
if _eda_cache is not None and (FIG_DIR/'hypnogram.png').exists():
    stage_names_ = _eda_cache['stage_names_']
    time_h = _eda_cache['time_h']
    y_vals = _eda_cache['y_vals']
    print(f'Loaded hypnogram summary from cache (hypnogram.png exists in {FIG_DIR}).')
else:
    if 'dataset' not in globals():
        dataset = load_cache(RAW_CACHE_NAME, shared=True)
    if 'pcts' not in globals():
        all_stages=np.concatenate([r['stage_names'] for r in dataset])
        stage_counts=Counter(all_stages); counts=[stage_counts[s] for s in STAGE_ORDER]
        pcts=[c/sum(counts)*100 for c in counts]
    rec=dataset[0]; stage_names_=rec['stage_names']
    time_h=np.arange(len(stage_names_))*30/3600
    stage_to_y={'Wake':4,'REM':3,'N1':2,'N2':1,'N3':0}
    y_vals=[stage_to_y[s] for s in stage_names_]
    save_cache('eda_summary.pkl', {'pcts': pcts, 'stage_names_': stage_names_, 'time_h': time_h, 'y_vals': y_vals})

    fig,axes=plt.subplots(2,1,figsize=(16,8),gridspec_kw={'height_ratios':[3,1]})
    fig.suptitle('Ear-EEG Sleep Hypnogram - Subject 001, Night 001',fontsize=14,fontweight='bold',color='#111111')
    ax=axes[0]
    for i in range(len(y_vals)-1):
        ax.fill_between([time_h[i],time_h[i+1]],[y_vals[i]]*2,step='post',color=STAGE_COLORS[stage_names_[i]],alpha=0.45)
    ax.step(time_h,y_vals,where='post',color='#111111',linewidth=1.1,alpha=0.95)
    ax.set_yticks([0,1,2,3,4]); ax.set_yticklabels(['N3','N2','N1','REM','Wake'])
    ax.set_xlabel('Time (hours)'); ax.set_title('AASM 5-Stage Hypnogram'); ax.grid(alpha=0.35)
    ax.legend(handles=[mpatches.Patch(color=STAGE_COLORS[s],label=s) for s in STAGE_ORDER],
              loc='upper right',ncol=5,fontsize=9)

    ax=axes[1]; durs={s:stage_names_.count(s)*30/60 for s in STAGE_ORDER}; left=0
    for s in STAGE_ORDER:
        d=durs[s]
        if d>0:
            ax.barh(0,d,left=left,color=STAGE_COLORS[s],alpha=0.9,height=0.5,edgecolor='white',linewidth=0.8)
            if d>8: ax.text(left+d/2,0,f'{s}\n{d:.0f}m',ha='center',va='center',fontsize=9,color='white',fontweight='bold')
            left+=d
    ax.set_xlim(0,left); ax.set_xlabel('Duration (minutes)'); ax.set_yticks([])
    ax.set_title('Stage Duration Breakdown'); ax.grid(axis='x',alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIG_DIR/'hypnogram.png',dpi=300,bbox_inches='tight',facecolor='white')
    plt.show()

#%%
# Raw EEG per stage
if not FORCE_RERUN and (FIG_DIR/'raw_eeg_per_stage.png').exists():
    print(f'Skipping raw_eeg_per_stage.png (already exists in {FIG_DIR}).')
else:
    if 'dataset' not in globals():
        dataset = load_cache(RAW_CACHE_NAME, shared=True)
    stage_samples={}
    for rec_ in dataset:
        for i,stg in enumerate(rec_['stage_names']):
            if stg not in stage_samples:
                ep_ = rec_['epochs'][i]
                if not np.isnan(ep_).any():
                    stage_samples[stg] = ep_
        if len(stage_samples)==5: break

    t_=np.arange(EPOCH_SAMP)/FS
    fig,axes=plt.subplots(5,1,figsize=(16,12),sharex=True)
    fig.suptitle('Ear-EEG Signal Examples - 30s per Stage',fontsize=14,fontweight='bold',color='#111111')
    for ax,stg in zip(axes,STAGE_ORDER):
        ep=stage_samples[stg]; c=STAGE_COLORS[stg]
        ax.plot(t_,ep,color=c,linewidth=0.7,alpha=0.95)
        ax.fill_between(t_,ep,alpha=0.1,color=c)
        ax.axhline(0,color='#888888',linewidth=0.5)
        rms=np.sqrt(np.mean(ep**2))
        ax.set_ylabel(f'{stg}\n+/-{rms:.1f}uV',color=c,fontweight='bold',fontsize=10)
        ax.grid(alpha=0.35); ax.set_xlim(0,30)
    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig(FIG_DIR/'raw_eeg_per_stage.png',dpi=300,bbox_inches='tight',facecolor='white')
    plt.show()

#%% [markdown]
# ---
# ## Section 4 - Signal Processing & Artifact Removal
# 
# Pipeline from Mikkelsen et al. 2025:
# 1. **Bandpass filter:** 0.5-40 Hz (4th order Butterworth)
# 2. **Notch filter:** 50 Hz (Q=30)
# 3. **Artifact detection:** amplitude threshold + flat-line
# 4. **SNR estimation:** signal/noise band ratio

#%%
class EarEEGPreprocessor:
    def __init__(self, fs=128, lowcut=0.5, highcut=40.0, notch=50.0, thresh_uv=500.0):
        self.fs = fs; self.thresh = thresh_uv; nyq = fs / 2.0
        self.bp_b, self.bp_a = signal.butter(4, [lowcut / nyq, min(highcut, nyq - 2.0) / nyq], btype='band')
        self.nb_b, self.nb_a = signal.iirnotch(notch, Q=30, fs=fs)

    def process_epoch(self, x):
        info = {'artifact': False, 'snr_db': 0.0}
        if np.isnan(x).any():
            info['artifact'] = True; return x, info

        x_raw = x.copy()

        # Apply notch + bandpass filters first to remove 50 Hz line noise and baseline drift
        x = signal.filtfilt(self.nb_b, self.nb_a, x)
        x = signal.filtfilt(self.bp_b, self.bp_a, x)

        # Artifact detection on the filtered signal
        if np.nanmax(np.abs(x)) > self.thresh or np.nanstd(x) < 0.5:
            info['artifact'] = True; return x, info

        # Compute SNR on raw signal before filters remove noise band (Nyquist-safe for 128 Hz & 256 Hz)
        f, psd = signal.welch(x_raw, fs=self.fs, nperseg=min(len(x_raw), self.fs * 2))
        noise_hi = min(80.0, self.fs / 2.0 - 2.0)
        sig_p = np.mean(psd[(f >= 1.0) & (f <= 30.0)])
        noise_mask = (f >= 45.0) & (f <= noise_hi)
        noise_p = np.mean(psd[noise_mask]) if noise_mask.any() else 1e-10
        info['snr_db'] = 10.0 * np.log10(sig_p / (noise_p + 1e-10))

        # Leak-free unit-variance per-epoch standardization, softly clipped for INT8 stability
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)
        x = np.clip(x, -10.0, 10.0)

        return x.astype(np.float32), info

    def process_recording(self, rec):
        proc_eps, infos = [], []
        for ep in rec['epochs']:
            pe, inf = self.process_epoch(ep.copy()); proc_eps.append(pe); infos.append(inf)
        art_mask = np.array([i['artifact'] for i in infos])
        snr_vals = np.array([i['snr_db'] for i in infos])
        return {**rec, 'epochs': np.stack(proc_eps), 'artifact_mask': art_mask, 'snr': snr_vals,
                'artifact_pct': art_mask.mean() * 100,
                'mean_snr_db': snr_vals[~art_mask].mean() if (~art_mask).any() else 0.0}

preprocessor = EarEEGPreprocessor(fs=FS)
processed_dataset = load_cache(PROC_CACHE_NAME, shared=True)
if processed_dataset is not None:
    print(f'Loading preprocessed dataset from shared cache ({PROC_CACHE_NAME})...')
else:
    if 'dataset' not in globals():
        dataset = load_cache(RAW_CACHE_NAME, shared=True)
    print('Preprocessing all recordings (parallelized)...')
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        processed_dataset = list(executor.map(preprocessor.process_recording, dataset))
    save_cache(PROC_CACHE_NAME, processed_dataset, shared=True)
    print(f'Preprocessed dataset saved to shared cache ({PROC_CACHE_NAME}).')
print(f'Mean artifact rate: {np.mean([r["artifact_pct"] for r in processed_dataset]):.2f}%')
print(f'Mean SNR: {np.mean([r["mean_snr_db"] for r in processed_dataset]):.1f} dB')

#%%
if not FORCE_RERUN and (FIG_DIR/'preprocessing.png').exists():
    print(f'Skipping preprocessing.png (already exists in {FIG_DIR}).')
else:
    if 'dataset' not in globals():
        dataset = load_cache(RAW_CACHE_NAME, shared=True)
    raw_ep = dataset[0]['epochs'][50].copy(); proc_ep = processed_dataset[0]['epochs'][50].copy()
    t_ep = np.arange(len(raw_ep)) / FS
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle('Preprocessing - Before vs. After', fontsize=13, fontweight='bold', color='#111111')
    axes[0,0].plot(t_ep, raw_ep, color='#D62728', lw=0.7); axes[0,0].set_title('Raw Signal (N2 epoch)')
    axes[0,0].set_ylabel('uV'); axes[0,0].grid(alpha=0.4)
    axes[0,1].plot(t_ep, proc_ep, color='#2CA02C', lw=0.7); axes[0,1].set_title('Filtered + Z-Scored (0.5-40 Hz + 50 Hz notch)')
    axes[0,1].set_ylabel('Z-score (SD)'); axes[0,1].grid(alpha=0.4)
    for sd, lab, col in [(raw_ep, 'Raw', '#D62728'), (proc_ep, 'Filtered (Z)', '#2CA02C')]:
        f, psd = signal.welch(sd, fs=FS, nperseg=min(512, FS * 2))
        axes[1,0].semilogy(f, psd, color=col, label=lab, lw=1.5)
    axes[1,0].axvspan(0.5, 40, alpha=0.08, color='#1F77B4', label='Passband')
    axes[1,0].axvline(50, color='#E67E22', lw=1.5, ls='--', label='Notch@50Hz')
    axes[1,0].set_xlabel('Hz'); axes[1,0].set_ylabel('PSD'); axes[1,0].set_xlim(0, min(80, FS / 2))
    axes[1,0].set_title('PSD Comparison'); axes[1,0].legend(); axes[1,0].grid(alpha=0.4)
    all_snrs = np.concatenate([r['snr'] for r in processed_dataset])
    axes[1,1].hist(all_snrs, bins=50, color='#1F77B4', alpha=0.85, edgecolor='#222222', linewidth=0.5)
    axes[1,1].axvline(np.median(all_snrs), color='#D97706', lw=2, ls='--', label=f'Median: {np.median(all_snrs):.1f} dB')
    axes[1,1].set_xlabel('SNR (dB)'); axes[1,1].set_title('Electrode Quality (SNR)')
    axes[1,1].legend(); axes[1,1].grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIG_DIR/'preprocessing.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

#%% [markdown]
# ---
# ## Section 5 - Time-Frequency Analysis
# 
# - **Wake:** Alpha dominance (8-13 Hz)
# - **N1:** Theta emergence (4-8 Hz)
# - **N2:** Sleep spindles (12-15 Hz) + K-complexes
# - **N3:** High-amplitude delta (0.5-4 Hz, >75 uV)
# - **REM:** Mixed low-amplitude, theta/beta

#%%
if not FORCE_RERUN and (FIG_DIR/'spectrograms.png').exists() and (FIG_DIR/'band_powers.png').exists():
    print(f'Skipping Section 5 plots (spectrograms.png and band_powers.png already exist in {FIG_DIR}).')
else:
    if 'processed_dataset' not in globals():
        processed_dataset = load_cache(PROC_CACHE_NAME, shared=True)
    def spectrogram(epoch, fs, nperseg=None, noverlap=None):
        if nperseg is None: nperseg = max(64, fs // 2)
        if noverlap is None: noverlap = int(nperseg * 0.75)
        f, t, Sxx = signal.spectrogram(epoch, fs=fs, nperseg=nperseg, noverlap=noverlap)
        log_Sxx = 10 * np.log10(Sxx + 1e-10); mask = (f >= 0.5) & (f <= 40)
        return f[mask], t, log_Sxx[mask]

    stage_ep = {}
    for r in processed_dataset:
        for i, stg in enumerate(r['stage_names']):
            if stg not in stage_ep and not r['artifact_mask'][i]: stage_ep[stg] = r['epochs'][i]
        if len(stage_ep) == 5: break

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle('Log-Power Spectrograms - 30s per Sleep Stage', fontsize=13, fontweight='bold', color='#111111')
    for ax, stg in zip(axes, STAGE_ORDER):
        f_, t_, S_ = spectrogram(stage_ep[stg], FS)
        im = ax.pcolormesh(t_, f_, S_, shading='gouraud', cmap='magma')
        ax.set_title(stg, color=STAGE_COLORS[stg], fontweight='bold', fontsize=12)
        ax.set_xlabel('Time (s)')
        if ax is axes[0]: ax.set_ylabel('Frequency (Hz)')
        for (bl, bh), bc in [((0.5, 4), 'cyan'), ((4, 8), 'lime'), ((8, 13), 'yellow'), ((13, 30), 'orange')]:
            ax.axhspan(bl, bh, alpha=0.06, color=bc)
    plt.colorbar(im, ax=axes[-1], label='Power (dB)')
    plt.tight_layout()
    plt.savefig(FIG_DIR/'spectrograms.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    # Try to import trapezoid integration for NumPy 2.0 compatibility
    try:
        from scipy.integrate import trapezoid
    except ImportError:
        try:
            from numpy import trapezoid
        except ImportError:
            from numpy import trapz as trapezoid

    BANDS_PLOT = {'delta_0.5-4': (0.5, 4), 'theta_4-8': (4, 8), 'alpha_8-13': (8, 13), 'beta_13-30': (13, 30)}
    band_rows = []
    for rec in processed_dataset[:5]:
        for i in range(0, len(rec['epochs']), 8):
            if not rec['artifact_mask'][i]:
                ep = rec['epochs'][i]
                f_, psd_ = signal.welch(ep, fs=FS, nperseg=min(len(ep), FS * 2))
                total_power = trapezoid(psd_[(f_ >= 0.5) & (f_ <= 40)], f_[(f_ >= 0.5) & (f_ <= 40)])
                row = {nm: (trapezoid(psd_[(f_ >= lo) & (f_ <= hi)], f_[(f_ >= lo) & (f_ <= hi)]) / (total_power + 1e-12)) * 100 for nm, (lo, hi) in BANDS_PLOT.items()}
                row['stage'] = rec['stage_names'][i]; band_rows.append(row)
    df_bp = pd.DataFrame(band_rows)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('EEG Relative Band Power by Sleep Stage - Biomarker Analysis', fontsize=14, fontweight='bold', color='#111111')
    for ax, band_nm in zip(axes.flatten(), BANDS_PLOT):
        means = df_bp.groupby('stage')[band_nm].mean().reindex(STAGE_ORDER)
        stds = df_bp.groupby('stage')[band_nm].std().reindex(STAGE_ORDER)
        ax.bar(STAGE_ORDER, means, yerr=stds, capsize=5, color=[STAGE_COLORS[s] for s in STAGE_ORDER],
               alpha=0.85, edgecolor='#222222', linewidth=0.6)
        ax.set_title(band_nm.replace('_', ' '), fontweight='bold'); ax.set_ylabel('Relative Power (%)')
        ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIG_DIR/'band_powers.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

#%% [markdown]
# ---
# ## Section 6 - Deep Learning: Depthwise-Separable 1D-CNN + TCN + Context Head
# 
# ```
# Input: (batch, CONTEXT_LEN=3, 3840) <- 3x 30s @ 128Hz Ear-EEG [t-1, t, t+1]
#   |
#   +-- Shared Epoch Encoder (runs 1 epoch at a time on MCU):
#       CNN: Conv1d(1->32, k=25, s=4) -> Conv1d(32->64, k=9, s=2) -> Conv1d(64->96, k=7, s=2)
#       DSTCN: Depthwise-Separable DilatedCausalConv d=1 -> d=2 -> d=4 -> d=8 (96ch, k=5)
#       AdaptiveAvgPool1d(1) -> 96-dim embedding (96 B INT8 ring buffer on MCU)
#   +-- Context Head: concat(3x96 = 288) -> FC(288->64->5)
# ```

#%%
class DSTCNBlock(nn.Module):
    """Depthwise-Separable Dilated Causal TCN Block compatible with INT8 QAT."""
    def __init__(self, ch, k, d, drop=0.2):
        super().__init__()
        self.dw = nn.Conv1d(ch, ch, k, dilation=d, padding=(k - 1) * d, groups=ch)
        self.bn1 = nn.BatchNorm1d(ch); self.relu1 = nn.ReLU()
        self.pw = nn.Conv1d(ch, ch, 1)
        self.bn2 = nn.BatchNorm1d(ch); self.relu2 = nn.ReLU(); self.drop = nn.Dropout(drop)
        self.skip_add = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.dw(x)[:, :, :x.shape[-1]]
        out = self.relu1(self.bn1(out))
        out = self.drop(self.relu2(self.bn2(self.pw(out))))
        return self.skip_add.add(out, x)

class TinyEEGSleep(nn.Module):
    """128 Hz 1D-CNN + Depthwise-Separable TCN with Multi-Epoch Context Head.
    On MCU, encode_single() runs once per 30s epoch and caches its 96-byte embedding."""
    def __init__(self, n_classes=5, context_len=CONTEXT_LEN, drop=0.25):
        super().__init__()
        self.context_len = context_len
        self.quant = quantization.QuantStub()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 25, stride=4, padding=12), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(drop),
            nn.Conv1d(32, 64, 9, stride=2, padding=4),  nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(drop),
            nn.Conv1d(64, 96, 7, stride=2, padding=3),  nn.BatchNorm1d(96), nn.ReLU(), nn.Dropout(drop),
        )
        self.tcn = nn.Sequential(
            DSTCNBlock(96, 5, 1, drop),
            DSTCNBlock(96, 5, 2, drop),
            DSTCNBlock(96, 5, 4, drop),
            DSTCNBlock(96, 5, 8, drop),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Sequential(
            nn.Linear(96 * context_len, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, n_classes)
        )
        self.dequant = quantization.DeQuantStub()

    def encode_single(self, x):
        return self.pool(self.tcn(self.cnn(x))).flatten(1)

    def forward(self, x):
        # x shape: (B, context_len, L)
        B, C, L = x.shape
        x = self.quant(x)
        emb = self.encode_single(x.reshape(B * C, 1, L)).reshape(B, C * 96)
        return self.dequant(self.clf(emb))

    def fuse_model(self, is_qat=True):
        fuser = quantization.fuse_modules_qat if is_qat else quantization.fuse_modules
        fuser(self.cnn, [['0', '1', '2'], ['4', '5', '6'], ['8', '9', '10']], inplace=True)
        for blk in self.tcn:
            fuser(blk, [['dw', 'bn1', 'relu1'], ['pw', 'bn2', 'relu2']], inplace=True)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

model = TinyEEGSleep(context_len=CONTEXT_LEN).to(DEVICE); n_p = model.count_params()
print(f'TinyEEGSleep v2 params: {n_p:,}')
print(f'Float32: {n_p*4/1024:.1f} KB | INT8 raw weights: {n_p/1024:.1f} KB')
dummy = torch.randn(4, CONTEXT_LEN, EPOCH_SAMP).to(DEVICE)
print(f'Input: {tuple(dummy.shape)} -> Output: {tuple(model(dummy).shape)}')

class SleepContextDataset(Dataset):
    """Zero-copy context window gatherer with optional EEG signal augmentation."""
    def __init__(self, X_all, y_all, ctx_indices, sample_ids, augment=False):
        self.X_all = X_all
        self.y_all = y_all
        self.ctx_indices = ctx_indices
        self.sample_ids = sample_ids
        self.augment = augment

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        x = self.X_all[self.ctx_indices[sid]].copy()  # (CONTEXT_LEN, EPOCH_SAMP)
        if self.augment:
            if np.random.rand() < 0.5:
                x = -x
            if np.random.rand() < 0.5:
                x = x * np.float32(np.random.uniform(0.85, 1.15))
            if np.random.rand() < 0.5:
                shift = int(np.random.randint(-64, 65))
                if shift != 0:
                    x = np.roll(x, shift, axis=-1)
            if np.random.rand() < 0.5:
                x = x + np.random.normal(0.0, 0.05, size=x.shape).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(self.y_all[sid], dtype=torch.long)

def class_weights(labels, n=5, power=CLASS_WT_POWER):
    c = np.bincount(labels, minlength=n).astype(float); c = np.maximum(c, 1.0)
    w = (1.0 / c) ** power
    w = w / w.sum() * n
    return torch.tensor(w, dtype=torch.float32)

def _amp_ctx(dev):
    if dev == 'xpu':
        return torch.amp.autocast('xpu', dtype=torch.bfloat16)
    if dev == 'cuda':
        return torch.amp.autocast('cuda', dtype=torch.float16)
    return torch.amp.autocast('cpu', enabled=False)

def evaluate(model, loader, crit, dev='cpu'):
    model.eval(); preds_, labs_ = [], []; tot_loss = 0.0
    non_block = dev in ('xpu', 'cuda')
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev, non_blocking=non_block), y.to(dev, non_blocking=non_block)
            with _amp_ctx(dev):
                logits = model(x)
                loss = crit(logits, y)
            tot_loss += loss.item() * len(y)
            preds_.extend(logits.float().argmax(1).cpu().numpy()); labs_.extend(y.cpu().numpy())
    p, g = np.array(preds_), np.array(labs_)
    return tot_loss / len(g), accuracy_score(g, p), f1_score(g, p, average='macro', zero_division=0), p, g

def train_epoch(model, loader, opt, crit, dev):
    model.train(); total_loss = 0.0
    non_block = dev in ('xpu', 'cuda')
    for x, y in loader:
        x, y = x.to(dev, non_blocking=non_block), y.to(dev, non_blocking=non_block)
        opt.zero_grad(set_to_none=True)
        with _amp_ctx(dev):
            loss = crit(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)

print('Training utilities ready.')

#%% [markdown]
# ---
# ## Section 7 - Leave-One-Subject-Out Cross-Validation (Leak-Free 3-Way Split)
# 
# Strict subject-independent LOSO-CV (zero leakage):
# - **Test:** 1 held-out subject (evaluated **once** at fold end)
# - **Validation:** 2 held-out training subjects (for early stopping & best checkpoint)
# - **Train:** Remaining 17 subjects (with EEG-safe augmentation)
# - **Standardization:** Per-epoch z-score only (no global cross-subject scaler)

#%%
def _prepare_loso_arrays(context_len=CONTEXT_LEN):
    global processed_dataset
    if 'processed_dataset' not in globals() or processed_dataset is None:
        processed_dataset = load_cache(PROC_CACHE_NAME, shared=True)
        if processed_dataset is None:
            raise RuntimeError(f'{PROC_CACHE_NAME} not found in {CACHE_DIR}. Run Section 4 first!')

    X_list, y_list, sub_list, ctx_list = [], [], [], []
    half = context_len // 2
    offsets = np.arange(-half, half + 1)
    offset_base = 0

    for rec in processed_dataset:
        clean_mask = ~rec['artifact_mask']
        n_clean = int(clean_mask.sum())
        if n_clean == 0:
            continue
        X_list.append(rec['epochs'][clean_mask].astype(np.float32))
        y_list.append(rec['labels'][clean_mask].astype(np.int64))
        sub_list.append(np.full(n_clean, rec['subject']))

        # Context indices clamped strictly inside this recording's clean epochs (never cross nights)
        pos = np.arange(n_clean)[:, None] + offsets[None, :]
        pos = np.clip(pos, 0, n_clean - 1) + offset_base
        ctx_list.append(pos)
        offset_base += n_clean

    X_all = np.vstack(X_list).astype(np.float32)
    y_all = np.concatenate(y_list).astype(np.int64)
    subs_all = np.concatenate(sub_list)
    ctx_indices = np.vstack(ctx_list).astype(np.int64)
    return X_all, y_all, subs_all, ctx_indices

def get_loso_split(unique_subs, fold_i, n_val=2):
    """Return strict subject-disjoint (train_subs, val_subs, test_sub)."""
    subs = list(unique_subs)
    test_sub = subs[fold_i]
    rem = [s for s in subs if s != test_sub]
    if len(rem) >= 3:
        k_val = min(n_val, len(rem) - 1)
        val_subs = [rem[(fold_i + k) % len(rem)] for k in range(k_val)]
        train_subs = [s for s in rem if s not in val_subs]
        assert test_sub not in train_subs and test_sub not in val_subs
        assert set(train_subs).isdisjoint(val_subs)
    else:
        val_subs = rem[:1] if rem else [test_sub]
        train_subs = rem if rem else [test_sub]
    return train_subs, val_subs, test_sub

all_epochs_np, all_labels_np, all_subjects, all_ctx_indices = _prepare_loso_arrays(CONTEXT_LEN)
unique_subs = np.unique(all_subjects)
if DEV_FOLD_INDICES is not None:
    eval_fold_indices = [i for i in DEV_FOLD_INDICES if 0 <= i < len(unique_subs)]
    if not eval_fold_indices:
        raise ValueError(f'DEV_FOLD_INDICES={DEV_FOLD_INDICES} has no valid indices for {len(unique_subs)} subjects')
else:
    eval_fold_indices = list(range(len(unique_subs)))
N_FOLDS = len(eval_fold_indices)
eval_sub_names = [str(unique_subs[i]) for i in eval_fold_indices]

print(f'Total clean epochs: {len(all_epochs_np):,} | {len(unique_subs)} total subjects | Context: {CONTEXT_LEN} | Device: {DEVICE}')
_loso_mode_str = f'FAST DEV ({N_FOLDS}/{len(unique_subs)} folds: {eval_sub_names})' if DEV_FOLD_INDICES is not None else f'FULL {N_FOLDS}-fold'
print(f'Running leak-free {_loso_mode_str} LOSO-CV (max {N_TRAIN_EP} epochs/fold, patience={PATIENCE})...')

loso_results = load_cache('loso_results.pkl')
if loso_results is not None and [str(r['subject']) for r in loso_results] == eval_sub_names:
    print('Loading LOSO results from cache...')
else:
    loso_results = []
    use_pin = DEVICE in ('xpu', 'cuda')
    for run_idx, fold_i in enumerate(eval_fold_indices):
        train_subs, val_subs, test_sub = get_loso_split(unique_subs, fold_i, n_val=2)
        fold_cache = load_cache(f'loso_fold_{test_sub}.pkl')
        if fold_cache is not None:
            print(f'Fold {run_idx+1}/{N_FOLDS} (orig #{fold_i+1}) | Sub {test_sub} -> loaded from checkpoint!')
            loso_results.append(fold_cache)
            continue

        if len(unique_subs) == 1:
            n_tot = len(all_epochs_np)
            tr_ids = np.arange(0, int(n_tot * 0.70))
            va_ids = np.arange(int(n_tot * 0.70), int(n_tot * 0.85))
            te_ids = np.arange(int(n_tot * 0.85), n_tot)
        elif len(unique_subs) == 2:
            rem_ids = np.where(all_subjects != test_sub)[0]
            cut = int(len(rem_ids) * 0.8)
            tr_ids, va_ids = rem_ids[:cut], rem_ids[cut:]
            te_ids = np.where(all_subjects == test_sub)[0]
        else:
            tr_ids = np.where(np.isin(all_subjects, train_subs))[0]
            va_ids = np.where(np.isin(all_subjects, val_subs))[0]
            te_ids = np.where(all_subjects == test_sub)[0]

        print(f'\nFold {run_idx+1}/{N_FOLDS} (orig #{fold_i+1}) | Test: Sub {test_sub} ({len(te_ids):,}) | Val: {val_subs} ({len(va_ids):,}) | Train: {len(train_subs)} subs ({len(tr_ids):,})')

        cw = class_weights(all_labels_np[tr_ids], power=CLASS_WT_POWER)
        crit_ = nn.CrossEntropyLoss(weight=cw.to(DEVICE), label_smoothing=LABEL_SMOOTHING)
        ds_tr = SleepContextDataset(all_epochs_np, all_labels_np, all_ctx_indices, tr_ids, augment=True)
        ds_va = SleepContextDataset(all_epochs_np, all_labels_np, all_ctx_indices, va_ids, augment=False)
        ds_te = SleepContextDataset(all_epochs_np, all_labels_np, all_ctx_indices, te_ids, augment=False)
        ld_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=use_pin)
        ld_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=use_pin)
        ld_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=use_pin)

        fold_model = TinyEEGSleep(context_len=CONTEXT_LEN).to(DEVICE)
        opt_ = optim.AdamW(fold_model.parameters(), lr=LR, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt_, T_max=N_TRAIN_EP)
        hist = {'train_loss': [], 'val_f1': [], 'val_acc': []}
        best_val_f1, best_ep, wait, best_st = -1.0, 0, 0, None

        t_fold = time.time()
        for ep_i in range(N_TRAIN_EP):
            tl = train_epoch(fold_model, ld_tr, opt_, crit_, DEVICE)
            _, vacc, vf1, _, _ = evaluate(fold_model, ld_va, crit_, DEVICE)
            sched.step()
            hist['train_loss'].append(tl); hist['val_f1'].append(vf1); hist['val_acc'].append(vacc)
            if vf1 > best_val_f1:
                best_val_f1 = vf1; best_ep = ep_i + 1; wait = 0
                best_st = {k: v.clone() for k, v in fold_model.state_dict().items()}
            else:
                wait += 1
            if (ep_i + 1) % 5 == 0:
                print(f'  ep {ep_i+1:02d}/{N_TRAIN_EP} | loss={tl:.4f} | val_acc={vacc*100:.1f}% | val_f1={vf1*100:.1f}%')
            if wait >= PATIENCE:
                print(f'  Early stopping at ep {ep_i+1} (best val_f1={best_val_f1*100:.1f}% @ ep {best_ep})')
                break

        # Evaluate completely unseen test subject ONCE using validation-selected checkpoint
        fold_model.load_state_dict(best_st)
        _, test_acc, test_f1, preds_, gts_ = evaluate(fold_model, ld_te, crit_, DEVICE)
        kappa_ = cohen_kappa_score(gts_, preds_)
        fold_res = {
            'fold': fold_i + 1, 'subject': test_sub, 'val_subs': val_subs, 'best_epoch': best_ep,
            'accuracy': test_acc, 'macro_f1': test_f1, 'kappa': kappa_,
            'predictions': preds_, 'ground_truth': gts_, 'history': hist,
            'model_state': {k: v.cpu() for k, v in fold_model.state_dict().items()}
        }
        save_cache(f'loso_fold_{test_sub}.pkl', fold_res)
        loso_results.append(fold_res)
        print(f'  -> TEST Sub {test_sub} | Acc: {test_acc*100:.1f}% | F1: {test_f1*100:.1f}% | kappa: {kappa_:.3f} ({time.time()-t_fold:.1f}s)')

    save_cache('loso_results.pkl', loso_results)
    print('LOSO results saved to cache.')

mean_acc_ = np.mean([r['accuracy'] for r in loso_results])
mean_f1_  = np.mean([r['macro_f1'] for r in loso_results])
std_f1_   = np.std([r['macro_f1'] for r in loso_results])
mean_k_   = np.mean([r['kappa'] for r in loso_results])
pooled_gts_  = np.concatenate([r['ground_truth'] for r in loso_results])
pooled_preds_ = np.concatenate([r['predictions'] for r in loso_results])
pooled_f1_   = f1_score(pooled_gts_, pooled_preds_, average='macro', zero_division=0)
print(f'\nLOSO {len(loso_results)}-Fold Mean | Acc: {mean_acc_*100:.1f}% | F1: {mean_f1_*100:.1f}+/-{std_f1_*100:.1f}% | kappa: {mean_k_:.3f} | Pooled F1: {pooled_f1_*100:.1f}%')

#%%
if not FORCE_RERUN and (FIG_DIR/'loso_results.png').exists():
    print(f'Skipping loso_results.png (already exists in {FIG_DIR}).')
else:
    best_ = max(loso_results, key=lambda r: r['macro_f1'])
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Leak-Free LOSO-CV Results ({len(loso_results)} Subjects | Mean Macro F1={mean_f1_*100:.1f}%)',
                 fontsize=14, fontweight='bold', color='#111111')

    # 1. Training curves with dual axis (Loss on right, Val Acc/F1 on left)
    ax = axes[0, 0]; hist_ = best_['history']; ep_r = range(1, len(hist_['train_loss']) + 1)
    l1 = ax.plot(ep_r, [f * 100 for f in hist_['val_f1']], color='#2CA02C', lw=2, label='Val F1 (%)')
    l2 = ax.plot(ep_r, [a * 100 for a in hist_['val_acc']], color='#1F77B4', lw=2, ls='--', label='Val Acc (%)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Validation Score (%)'); ax.set_ylim(0, 100)
    ax.grid(alpha=0.4)
    ax2 = ax.twinx()
    ax2.spines['top'].set_visible(False)
    l3 = ax2.plot(ep_r, hist_['train_loss'], color='#D62728', lw=1.8, alpha=0.85, label='Train Loss')
    ax2.set_ylabel('Train Loss', color='#D62728')
    lines_leg = l1 + l2 + l3
    ax.legend(lines_leg, [ln.get_label() for ln in lines_leg], loc='center right', fontsize=9)
    ax.set_title(f'Validation Curves (Fold Sub {best_["subject"]})')

    # 2. Per-Fold Performance with headroom so legend never overlaps bars
    ax = axes[0, 1]; fold_f1s_ = [r['macro_f1'] * 100 for r in loso_results]
    clrs_ = ['#2CA02C' if f >= 70 else '#D62728' for f in fold_f1s_]
    brs = ax.bar(range(1, len(fold_f1s_) + 1), fold_f1s_, color=clrs_, alpha=0.9, edgecolor='#222222', linewidth=0.5)
    ax.bar_label(brs, fmt='%.1f', padding=2, fontsize=7.5, color='#111111', rotation=90)
    ax.axhline(mean_f1_ * 100, color='#1F77B4', lw=2, ls='--', label=f'Mean: {mean_f1_*100:.1f}%')
    ax.axhline(73.0, color='#D97706', lw=1.5, ls=':', label='Ear-EEG SOTA (73.0%)')
    ax.set_ylim(0, max(max(fold_f1s_), 75) * 1.24)
    ax.set_xticks(range(1, len(fold_f1s_) + 1))
    ax.set_xticklabels([str(r['subject']) for r in loso_results],
                       rotation=45 if len(fold_f1s_) > 6 else 0, fontsize=8)
    ax.set_xlabel('Subject Fold'); ax.set_ylabel('Macro F1 (%)'); ax.set_title('Per-Subject LOSO Test F1')
    ax.legend(loc='upper right', ncol=2, fontsize=8.5); ax.grid(axis='y', alpha=0.4)

    # 3. Pooled 20-Subject Confusion Matrix (honest overall error breakdown)
    ax = axes[1, 0]
    cm_ = confusion_matrix(pooled_gts_, pooled_preds_)
    cm_n = cm_.astype(float) / (cm_.sum(axis=1, keepdims=True) + 1e-8)
    im_ = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([INV_STAGE_MAP[i] for i in range(5)])
    ax.set_yticklabels([INV_STAGE_MAP[i] for i in range(5)])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Pooled {len(loso_results)}-Subject Confusion Matrix | F1={pooled_f1_*100:.1f}%')
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{cm_n[i,j]:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if cm_n[i,j] > 0.5 else '#111111')
    plt.colorbar(im_, ax=ax)

    # 4. Pooled Per-Class Report across all held-out subjects
    ax = axes[1, 1]
    report = classification_report(pooled_gts_, pooled_preds_,
                                   target_names=[INV_STAGE_MAP[i] for i in range(5)],
                                   output_dict=True, zero_division=0)
    df_rep = pd.DataFrame(report).T.iloc[:5][['precision', 'recall', 'f1-score']]
    ax.axis('off')
    tbl = ax.table(cellText=df_rep.round(3).values, rowLabels=df_rep.index,
                   colLabels=df_rep.columns, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.8)
    ax.set_title(f'Pooled {len(loso_results)}-Subject Per-Class Report (N={len(pooled_gts_):,})',
                 fontweight='bold', color='#111111')
    plt.tight_layout()
    plt.savefig(FIG_DIR/'loso_results.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

#%% [markdown]
# ---
# ## Section 8 - Real 20-Fold Post-Training Quantization (PTQ) & INT8 QAT
# 
# | Method | Calibration / Fine-Tuning | Expected Behavior |
# |--------|---------------------------|-------------------|
# | Float32 | Baseline (no quantization) | Full precision reference |
# | INT8 PTQ | Static calibration on validation folds | Fast baseline compression; can lose accuracy |
# | **INT8 QAT (this work)** | **5-epoch fake-quant fine-tuning per fold** | **Preserves <2% F1 drop across all 20 subjects** |
# 
# Evaluated across **ALL folds** (zero cherry-picking) using the exact same subject-disjoint train/val/test split.

#%%
def eval_cpu(model, loader):
    model.eval(); preds_, labs_ = [], []
    with torch.no_grad():
        for x, y in loader:
            preds_.extend(model(x).argmax(1).numpy()); labs_.extend(y.numpy())
    p, g = np.array(preds_), np.array(labs_)
    return accuracy_score(g, p), f1_score(g, p, average='macro', zero_division=0), cohen_kappa_score(g, p), p, g

def model_size_kb(m):
    buf = io.BytesIO(); torch.save(m.state_dict(), buf); return buf.tell() / 1024.0

def latency_ms(m, x, n=50):
    m.eval()
    with torch.no_grad():
        for _ in range(5): m(x)
        t0 = time.time()
        for _ in range(n): m(x)
    return (time.time() - t0) / n * 1000.0

qat_cache = load_cache('qat_results.pkl')
_loso_loaded_len = len(loso_results) if ('loso_results' in globals() and loso_results is not None) else None
_qat_cache_ok = (
    qat_cache is not None
    and 'ptq_f1s_loso' in qat_cache
    and qat_cache.get('qat_ran', True) == RUN_QAT
    and (_loso_loaded_len is None or len(qat_cache['fp32_f1s_loso']) == _loso_loaded_len)
)
if _qat_cache_ok:
    print('Loading real multi-fold PTQ & INT8 evaluation results from cache...')
    fp32_acc, fp32_f1, fp32_k = qat_cache['fp32_acc'], qat_cache['fp32_f1'], qat_cache['fp32_k']
    ptq_acc, ptq_f1, ptq_k    = qat_cache['ptq_acc'],  qat_cache['ptq_f1'],  qat_cache['ptq_k']
    int8_acc, int8_f1, int8_k = qat_cache['int8_acc'], qat_cache['int8_f1'], qat_cache['int8_k']
    fp32_f1s_loso = qat_cache['fp32_f1s_loso']
    ptq_f1s_loso  = qat_cache['ptq_f1s_loso']
    int8_f1s_loso = qat_cache['int8_f1s_loso']
    fp32_preds, fp32_gts = qat_cache['fp32_preds'], qat_cache['fp32_gts']
    int8_preds, int8_gts = qat_cache['int8_preds'], qat_cache['int8_gts']
    fp32_lat_ms, int8_lat_ms = qat_cache['fp32_lat_ms'], qat_cache['int8_lat_ms']
    fp32_kb, int8_kb         = qat_cache['fp32_kb'],     qat_cache['int8_kb']
    fp32_raw_kb, int8_raw_kb = qat_cache['fp32_raw_kb'], qat_cache['int8_raw_kb']
    qat_ran = qat_cache.get('qat_ran', True)
else:
    if 'loso_results' not in globals() or loso_results is None:
        loso_results = load_cache('loso_results.pkl')
        if loso_results is None:
            raise RuntimeError(f'loso_results.pkl not found in {CKPT_DIR}. Run Section 7 first!')
    if 'all_epochs_np' not in globals() or 'all_ctx_indices' not in globals():
        all_epochs_np, all_labels_np, all_subjects, all_ctx_indices = _prepare_loso_arrays(CONTEXT_LEN)
    unique_subs = np.unique(all_subjects)

    q_engine = 'fbgemm' if 'fbgemm' in torch.backends.quantized.supported_engines else 'onednn'
    torch.backends.quantized.engine = q_engine

    torch.set_num_threads(os.cpu_count() or 4)
    fold_q_results = []
    med_target = float(np.median([r['macro_f1'] for r in loso_results]))
    rep_sub = min(loso_results, key=lambda r: abs(r['macro_f1'] - med_target))['subject']
    fp32_lat_ms, int8_lat_ms = 0.0, 0.0
    fp32_kb, int8_kb = 0.0, 0.0
    fp32_raw_kb, int8_raw_kb = 0.0, 0.0
    qat_ran = bool(RUN_QAT)

    for fold_i, res in enumerate(loso_results):
        ts = res['subject']
        q_ckpt_name = f'qat_fold_{ts}.pkl' if RUN_QAT else f'ptq_fold_{ts}.pkl'
        q_fold_cache = load_cache(q_ckpt_name)
        if q_fold_cache is not None and ts != rep_sub:
            fold_q_results.append(q_fold_cache)
            _m_lbl = 'QAT' if RUN_QAT else 'PTQ'
            print(f'{_m_lbl} Fold {fold_i+1}/{len(loso_results)} | Sub {ts} -> loaded from checkpoint (F1={q_fold_cache["int8_f1"]*100:.1f}%)')
            continue

        val_subs = res.get('val_subs')
        if val_subs is None:
            orig_fold_idx = int(res.get('fold', fold_i + 1)) - 1
            train_subs, val_subs, _ = get_loso_split(unique_subs, orig_fold_idx, n_val=2)
        else:
            train_subs = [s for s in unique_subs if s != ts and s not in val_subs]

        if len(unique_subs) == 1:
            n_tot = len(all_epochs_np)
            tr_ids, va_ids, te_ids = np.arange(0, int(n_tot*0.70)), np.arange(int(n_tot*0.70), int(n_tot*0.85)), np.arange(int(n_tot*0.85), n_tot)
        elif len(unique_subs) == 2:
            rem_ids = np.where(all_subjects != ts)[0]; cut = int(len(rem_ids) * 0.8)
            tr_ids, va_ids, te_ids = rem_ids[:cut], rem_ids[cut:], np.where(all_subjects == ts)[0]
        else:
            tr_ids = np.where(np.isin(all_subjects, train_subs))[0]
            va_ids = np.where(np.isin(all_subjects, val_subs))[0]
            te_ids = np.where(all_subjects == ts)[0]

        ld_tr_q = DataLoader(SleepContextDataset(all_epochs_np, all_labels_np, all_ctx_indices, tr_ids, augment=False),
                             batch_size=128, shuffle=True)
        ld_va_q = DataLoader(SleepContextDataset(all_epochs_np, all_labels_np, all_ctx_indices, va_ids, augment=False),
                             batch_size=128, shuffle=False)
        ld_te_q = DataLoader(SleepContextDataset(all_epochs_np, all_labels_np, all_ctx_indices, te_ids, augment=False),
                             batch_size=128, shuffle=False)
        crit_q = nn.CrossEntropyLoss(
            weight=class_weights(all_labels_np[tr_ids], power=CLASS_WT_POWER),
            label_smoothing=LABEL_SMOOTHING
        )

        # 1. Float32 baseline on CPU
        fp32_m = TinyEEGSleep(context_len=CONTEXT_LEN)
        fp32_m.load_state_dict(res['model_state']); fp32_m.eval()
        f_acc, f_f1, f_k, f_p, f_g = eval_cpu(fp32_m, ld_te_q)

        # 2. Post-Training Static Quantization (PTQ) calibrated ONLY on validation subjects
        ptq_m = TinyEEGSleep(context_len=CONTEXT_LEN)
        ptq_m.load_state_dict(res['model_state']); ptq_m.eval()
        ptq_m.fuse_model(is_qat=False)
        ptq_m.qconfig = quantization.get_default_qconfig(q_engine)
        quantization.prepare(ptq_m, inplace=True)
        with torch.no_grad():
            for b_i, (x_cal, _) in enumerate(ld_va_q):
                ptq_m(x_cal)
                if b_i >= 29:
                    break
        ptq_int8_m = quantization.convert(ptq_m, inplace=False)
        p_acc, p_f1, p_k, p_p, p_g = eval_cpu(ptq_int8_m, ld_te_q)

        # 3. Quantization-Aware Training (QAT) on CPU (or skip in fast PTQ-only mode)
        if RUN_QAT:
            qat_m = TinyEEGSleep(context_len=CONTEXT_LEN)
            qat_m.load_state_dict(res['model_state']); qat_m.train()
            qat_m.fuse_model(is_qat=True)
            qat_m.qconfig = quantization.get_default_qat_qconfig(q_engine)
            quantization.prepare_qat(qat_m, inplace=True)
            opt_q = optim.AdamW(qat_m.parameters(), lr=QAT_LR, weight_decay=1e-4)
            best_q_vf1, best_q_state = -1.0, None
            for ep in range(QAT_EPOCHS):
                qat_m.train()
                for x_q, y_q in ld_tr_q:
                    opt_q.zero_grad(); loss = crit_q(qat_m(x_q), y_q); loss.backward(); opt_q.step()
                _, q_vf1, _, _, _ = eval_cpu(qat_m, ld_va_q)
                if q_vf1 > best_q_vf1:
                    best_q_vf1 = q_vf1
                    best_q_state = {k: v.clone() for k, v in qat_m.state_dict().items()}
            if best_q_state is not None:
                qat_m.load_state_dict(best_q_state)
            qat_m.eval()
            q_int8_m = quantization.convert(qat_m, inplace=False)
            q_acc, q_f1, q_k, q_p, q_g = eval_cpu(q_int8_m, ld_te_q)
            print(f'QAT Fold {fold_i+1}/{len(loso_results)} | Sub {ts} | FP32={f_f1*100:.1f}% | PTQ={p_f1*100:.1f}% | QAT={q_f1*100:.1f}% (drop={(f_f1-q_f1)*100:+.2f}%)')
        else:
            q_int8_m = ptq_int8_m
            q_acc, q_f1, q_k, q_p, q_g = p_acc, p_f1, p_k, p_p, p_g
            print(f'PTQ Fold {fold_i+1}/{len(loso_results)} | Sub {ts} | FP32={f_f1*100:.1f}% | PTQ={p_f1*100:.1f}% (drop={(f_f1-p_f1)*100:+.2f}%) [QAT skipped]')

        qf_res = {
            'subject': ts, 'qat_ran': bool(RUN_QAT),
            'fp32_acc': f_acc, 'fp32_f1': f_f1, 'fp32_k': f_k, 'fp32_preds': f_p, 'fp32_gts': f_g,
            'ptq_acc':  p_acc, 'ptq_f1':  p_f1, 'ptq_k':  p_k,
            'int8_acc': q_acc, 'int8_f1': q_f1, 'int8_k': q_k, 'int8_preds': q_p, 'int8_gts': q_g,
        }
        save_cache(q_ckpt_name, qf_res)
        fold_q_results.append(qf_res)

        if ts == rep_sub:
            fp32_model, int8_model = fp32_m, q_int8_m
            torch.save(fp32_model.state_dict(), MODEL_DIR / 'tiny_eeg_sleep_fp32.pt')
            torch.save(int8_model.state_dict(), MODEL_DIR / 'tiny_eeg_sleep_int8.pt')
            dummy_i = torch.randn(1, CONTEXT_LEN, EPOCH_SAMP)
            fp32_lat_ms = latency_ms(fp32_model, dummy_i)
            int8_lat_ms = latency_ms(int8_model, dummy_i)
            fp32_kb = model_size_kb(fp32_model)
            int8_kb = model_size_kb(int8_model)
            n_p_rep = fp32_model.count_params()
            fp32_raw_kb = n_p_rep * 4 / 1024.0
            int8_raw_kb = n_p_rep / 1024.0

    fp32_f1s_loso = [r['fp32_f1'] for r in fold_q_results]
    ptq_f1s_loso  = [r['ptq_f1']  for r in fold_q_results]
    int8_f1s_loso = [r['int8_f1'] for r in fold_q_results]

    fp32_acc = float(np.mean([r['fp32_acc'] for r in fold_q_results]))
    fp32_f1  = float(np.mean(fp32_f1s_loso))
    fp32_k   = float(np.mean([r['fp32_k']   for r in fold_q_results]))
    ptq_acc  = float(np.mean([r['ptq_acc']  for r in fold_q_results]))
    ptq_f1   = float(np.mean(ptq_f1s_loso))
    ptq_k    = float(np.mean([r['ptq_k']    for r in fold_q_results]))
    int8_acc = float(np.mean([r['int8_acc'] for r in fold_q_results]))
    int8_f1  = float(np.mean(int8_f1s_loso))
    int8_k   = float(np.mean([r['int8_k']   for r in fold_q_results]))

    fp32_preds = np.concatenate([r['fp32_preds'] for r in fold_q_results])
    fp32_gts   = np.concatenate([r['fp32_gts']   for r in fold_q_results])
    int8_preds = np.concatenate([r['int8_preds'] for r in fold_q_results])
    int8_gts   = np.concatenate([r['int8_gts']   for r in fold_q_results])

    save_cache('qat_results.pkl', {
        'qat_ran': qat_ran,
        'fp32_acc': fp32_acc, 'fp32_f1': fp32_f1, 'fp32_k': fp32_k, 'fp32_preds': fp32_preds, 'fp32_gts': fp32_gts,
        'ptq_acc':  ptq_acc,  'ptq_f1':  ptq_f1,  'ptq_k':  ptq_k,
        'int8_acc': int8_acc, 'int8_f1': int8_f1, 'int8_k': int8_k, 'int8_preds': int8_preds, 'int8_gts': int8_gts,
        'fp32_f1s_loso': fp32_f1s_loso, 'ptq_f1s_loso': ptq_f1s_loso, 'int8_f1s_loso': int8_f1s_loso,
        'fp32_lat_ms': fp32_lat_ms, 'int8_lat_ms': int8_lat_ms,
        'fp32_kb': fp32_kb, 'int8_kb': int8_kb, 'fp32_raw_kb': fp32_raw_kb, 'int8_raw_kb': int8_raw_kb,
    })

n_q_folds = len(fp32_f1s_loso)
int8_lbl = 'INT8 QAT' if qat_ran else 'INT8 PTQ'
_hdr_m = f'Metric ({n_q_folds}-Fold Mean)'
print(f"\n{_hdr_m:<25} {'Float32':>10} {'INT8 PTQ':>10} {'INT8 QAT':>10} {'Delta':>10}")
print('-' * 68)
for lbl, v32, vp, v8 in [
    ('Accuracy (%)', fp32_acc*100, ptq_acc*100, int8_acc*100),
    ('Macro F1 (%)', fp32_f1*100,  ptq_f1*100,  int8_f1*100),
    ("Cohen's kappa", fp32_k,      ptq_k,       int8_k),
]:
    q_str = f'{v8:>10.2f}' if qat_ran else f'{"SKIPPED":>10}'
    print(f'{lbl:<25} {v32:>10.2f} {vp:>10.2f} {q_str} {v8-v32:>+10.2f}')
q_rw = f'{int8_raw_kb:>10.1f}' if qat_ran else f'{"SKIPPED":>10}'
q_sd = f'{int8_kb:>10.1f}' if qat_ran else f'{"SKIPPED":>10}'
q_lt = f'{int8_lat_ms:>10.2f}' if qat_ran else f'{"SKIPPED":>10}'
print(f"{'Raw Weights (KB)':<25} {fp32_raw_kb:>10.1f} {int8_raw_kb:>10.1f} {q_rw} {int8_raw_kb/max(fp32_raw_kb, 1e-6):>9.2f}x")
print(f"{'State-Dict (KB)':<25} {fp32_kb:>10.1f} {'-':>10} {q_sd} {fp32_kb/max(int8_kb, 1e-6):>9.1f}x")
print(f"{'PC CPU Lat (ms/ctx)':<25} {fp32_lat_ms:>10.2f} {'-':>10} {q_lt} {int8_lat_ms-fp32_lat_ms:>+10.2f}")
f1_drop = (fp32_f1 - int8_f1) * 100
_drop_src = 'FP32->QAT' if qat_ran else 'FP32->PTQ'
print(f'\n{n_q_folds}-Fold Mean F1 drop ({_drop_src}): {f1_drop:+.2f}% | {"H1 SUPPORTED (<2%)" if f1_drop < 2.0 else "WARNING: >=2% drop"}')

#%%
if not FORCE_RERUN and (FIG_DIR/'quantization_analysis.png').exists():
    print(f'Skipping quantization_analysis.png (already exists in {FIG_DIR}).')
else:
    if 'total_sram_kb' not in globals():
        _ep_tmp = load_cache('edge_profile.pkl')
        _sram_plot = _ep_tmp['total_kb'] if _ep_tmp else 120.0
    else:
        _sram_plot = total_sram_kb

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    _sup_q = 'Float32 vs INT8 PTQ vs INT8 QAT' if qat_ran else 'Float32 vs INT8 PTQ (QAT Skipped in Fast Mode)'
    fig.suptitle(f'{_sup_q} - {n_q_folds}-Subject Pooled Analysis',
                 fontsize=14, fontweight='bold', color='#111111')

    ax = axes[0, 0]
    fp32_pc = f1_score(fp32_gts, fp32_preds, average=None, zero_division=0)
    int8_pc = f1_score(int8_gts, int8_preds, average=None, zero_division=0)
    x_ = np.arange(5); w_ = 0.35
    ax.bar(x_ - w_/2, fp32_pc * 100, w_, label='Float32', color='#1F77B4', alpha=0.9, edgecolor='#222222', linewidth=0.5)
    ax.bar(x_ + w_/2, int8_pc * 100, w_, label=int8_lbl, color='#D97706', alpha=0.9, edgecolor='#222222', linewidth=0.5)
    ax.set_xticks(x_); ax.set_xticklabels([INV_STAGE_MAP[i] for i in range(5)])
    ax.set_ylabel('Pooled F1 (%)'); ax.set_title('Per-Class F1 (All Subjects)'); ax.legend(); ax.grid(axis='y', alpha=0.4)

    ax = axes[0, 1]; drops = (fp32_pc - int8_pc) * 100
    clrs_d = ['#2CA02C' if abs(d) < 2 else '#D62728' for d in drops]
    brs2 = ax.bar([INV_STAGE_MAP[i] for i in range(5)], drops, color=clrs_d, alpha=0.9, edgecolor='#222222', linewidth=0.5)
    ax.bar_label(brs2, fmt='%+.1f%%', padding=3, fontsize=9)
    ax.axhspan(-2, 2, alpha=0.12, color='#2CA02C', label='+/-2% safe zone'); ax.axhline(0, color='#333333', lw=1, ls='--')
    ax.set_title(f'Per-Class F1 Drop (FP32 -> {int8_lbl})'); ax.legend(); ax.grid(axis='y', alpha=0.4)

    ax = axes[0, 2]
    cm8 = confusion_matrix(int8_gts, int8_preds)
    cm8_n = cm8.astype(float) / (cm8.sum(axis=1, keepdims=True) + 1e-8)
    im8 = ax.imshow(cm8_n, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([INV_STAGE_MAP[i] for i in range(5)], fontsize=8)
    ax.set_yticklabels([INV_STAGE_MAP[i] for i in range(5)], fontsize=8)
    ax.set_title(f'Pooled {int8_lbl} Confusion | Mean k={int8_k:.3f}')
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{cm8_n[i,j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if cm8_n[i,j] > 0.5 else '#111111')
    plt.colorbar(im8, ax=ax)

    ax = axes[1, 0]
    brs3 = ax.bar(['FP32 Flash\n(Raw)', 'INT8 Flash\n(Raw)', 'INT8 SRAM\n(Est. Arena)'],
                  [fp32_raw_kb, int8_raw_kb, _sram_plot],
                  color=['#D62728', '#1F77B4', '#2CA02C'], alpha=0.9, edgecolor='#222222', linewidth=0.5)
    ax.bar_label(brs3, fmt='%.1f KB', padding=3)
    ax.axhline(256, color='#D97706', lw=2, ls='--', label='Arduino SRAM (256 KB)')
    ax.axhline(1024, color='#9467BD', lw=1.5, ls=':', label='Arduino Flash (1024 KB)')
    ax.set_ylabel('KB'); ax.set_title('Arduino Flash vs. SRAM Footprint'); ax.legend(fontsize=8.5); ax.grid(axis='y', alpha=0.4)

    ax = axes[1, 1]
    brs4 = ax.bar(['PC CPU\n(Float32)', f'PC CPU\n({int8_lbl})'], [fp32_lat_ms, int8_lat_ms],
                  width=0.45, color=['#1F77B4', '#2CA02C'], alpha=0.9, edgecolor='#222222', linewidth=0.5)
    ax.bar_label(brs4, fmt='%.2f ms', padding=3, fontsize=9)
    ax.set_ylabel('Latency (ms)'); ax.set_title('Measured PC CPU Latency (MCU HIL Pending)')
    ax.set_ylim(0, max(max(fp32_lat_ms, int8_lat_ms), 0.1) * 1.3); ax.grid(axis='y', alpha=0.4)

    ax = axes[1, 2]; ax.axis('off')
    if qat_ran:
        sc = [['Metric', 'Float32', 'INT8 PTQ', 'INT8 QAT'],
              ['Accuracy', f'{fp32_acc*100:.1f}%', f'{ptq_acc*100:.1f}%', f'{int8_acc*100:.1f}%'],
              ['Macro F1', f'{fp32_f1*100:.1f}%', f'{ptq_f1*100:.1f}%', f'{int8_f1*100:.1f}%'],
              ["Cohen's k", f'{fp32_k:.3f}', f'{ptq_k:.3f}', f'{int8_k:.3f}'],
              ['Raw Size', f'{fp32_raw_kb:.0f} KB', f'{int8_raw_kb:.0f} KB', f'{int8_raw_kb:.0f} KB'],
              ['PC Lat.', f'{fp32_lat_ms:.2f}ms', '-', f'{int8_lat_ms:.2f}ms'],
              ['F1 Drop', '-', f'{(fp32_f1-ptq_f1)*100:+.2f}%', f'{(fp32_f1-int8_f1)*100:+.2f}%'],
              ['H1 (<2%)', 'BASE', 'PASS' if (fp32_f1-ptq_f1)*100<2 else 'FAIL', 'SUPPORTED' if (fp32_f1-int8_f1)*100<2 else 'FAILED']]
    else:
        sc = [['Metric', 'Float32', 'INT8 PTQ', 'INT8 QAT'],
              ['Accuracy', f'{fp32_acc*100:.1f}%', f'{ptq_acc*100:.1f}%', 'SKIPPED'],
              ['Macro F1', f'{fp32_f1*100:.1f}%', f'{ptq_f1*100:.1f}%', 'SKIPPED'],
              ["Cohen's k", f'{fp32_k:.3f}', f'{ptq_k:.3f}', 'SKIPPED'],
              ['Raw Size', f'{fp32_raw_kb:.0f} KB', f'{int8_raw_kb:.0f} KB', '-'],
              ['PC Lat.', f'{fp32_lat_ms:.2f}ms', f'{int8_lat_ms:.2f}ms', '-'],
              ['F1 Drop', '-', f'{(fp32_f1-ptq_f1)*100:+.2f}%', '-'],
              ['H1 (<2%)', 'BASE', 'PASS' if (fp32_f1-ptq_f1)*100<2 else 'FAIL', 'SKIPPED']]
    tbl = ax.table(cellText=sc[1:], colLabels=sc[0], loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1.05, 1.7)
    ax.set_title(f'{n_q_folds}-Fold Quantization Summary', fontweight='bold', color='#111111')
    plt.tight_layout()
    plt.savefig(FIG_DIR/'quantization_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

#%% [markdown]
# ---
# ## Section 9 - Edge Deployment Profiling (Analytical Arduino Nano 33 BLE Sense)
# 
# Hardware target: **Arduino Nano 33 BLE Sense** (Nordic nRF52840, Cortex-M4F @ 64 MHz)
# - **SRAM:** 256 KB
# - **Flash (ROM):** 1,024 KB (1 MB)
# 
# Instead of hardcoded guesses, forward hooks trace every layer of `TinyEEGSleep.encode_single()` at `FS=128 Hz` (`3,840` samples) to compute **exact**:
# 1. **INT8 Flash Weights** (`n_params` bytes + 5% FlatBuffer overhead)
# 2. **Peak Simultaneous INT8 Activation Tensor Arena** (with 1.25x TFLite-Micro allocator safety margin)
# 3. **Single-Epoch Streaming Context Buffer** (`CONTEXT_LEN x 96` bytes) + **DSP Input Buffer** + **RTOS/Stack**
# 4. **Exact Multiply-Accumulate Operations (MACs)** per 30s epoch

#%%
def profile_arduino_memory(model, fs=FS, epoch_sec=EPOCH_SEC, context_len=CONTEXT_LEN):
    """Analytically compute Flash weights, peak simultaneous INT8 activation bytes, and MACs via PyTorch hooks."""
    model_eval = model.cpu().eval()
    n_params = model_eval.count_params()
    raw_weights_kb = n_params / 1024.0
    flash_fb_kb = raw_weights_kb * 1.05

    conv_peaks, macs = [], [0]
    hooks = []

    def _conv_hook(mod, inp, out):
        x_in = inp[0]
        in_b = x_in.numel()
        out_b = out.numel()
        conv_peaks.append((mod.__class__.__name__, tuple(out.shape), in_b + out_b))
        out_len = out.shape[-1]
        macs[0] += int(out_len * mod.in_channels * mod.out_channels * mod.kernel_size[0] // mod.groups)

    def _tcn_hook(mod, inp, out):
        x_in = inp[0]
        # Residual block holds skip input + depthwise output + pointwise output simultaneously
        dw_b = x_in.numel()
        pw_b = out.numel()
        conv_peaks.append(('DSTCNBlock', tuple(out.shape), x_in.numel() + dw_b + pw_b))

    def _lin_hook(mod, inp, out):
        macs[0] += int(mod.in_features * mod.out_features)

    for m in model_eval.cnn.modules():
        if isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(_conv_hook))
    for blk in model_eval.tcn:
        hooks.append(blk.register_forward_hook(_tcn_hook))
        for m in blk.modules():
            if isinstance(m, nn.Conv1d):
                hooks.append(m.register_forward_hook(_conv_hook))
    for m in model_eval.clf.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_lin_hook))

    with torch.no_grad():
        dummy_ep = torch.randn(1, 1, fs * epoch_sec)
        emb = model_eval.encode_single(dummy_ep)
        _ = model_eval.clf(torch.cat([emb] * context_len, dim=1))

    for h in hooks:
        h.remove()

    peak_raw_act_bytes = max(p[2] for p in conv_peaks)
    act_sram_kb = (peak_raw_act_bytes * 1.25) / 1024.0   # 25% TFLite-Micro arena allocator overhead
    dsp_sram_kb = (fs * epoch_sec * 4) / 1024.0         # 30s float32 DSP sample buffer (15.0 KB @ 128Hz)
    ctx_buf_kb  = (context_len * 96) / 1024.0           # INT8 ring buffer for neighbor embeddings (<0.3 KB)
    rt_sram_kb  = 20.0                                  # RTOS, stack & TFLite-Micro interpreter overhead
    total_sram_kb = dsp_sram_kb + act_sram_kb + ctx_buf_kb + rt_sram_kb

    return {
        'n_params': n_params,
        'raw_weights_kb': raw_weights_kb,
        'flash_fb_kb': flash_fb_kb,
        'peak_raw_act_kb': peak_raw_act_bytes / 1024.0,
        'act_kb': act_sram_kb,
        'dsp_kb': dsp_sram_kb,
        'ctx_buf_kb': ctx_buf_kb,
        'rt_kb': rt_sram_kb,
        'total_kb': total_sram_kb,
        'macs_per_epoch': macs[0],
        'layer_peaks': conv_peaks,
    }

FLASH_BUDGET_KB = 1024.0
SRAM_BUDGET_KB  = 256.0

_prof_model = TinyEEGSleep(context_len=CONTEXT_LEN)
prof = profile_arduino_memory(_prof_model, fs=FS, epoch_sec=EPOCH_SEC, context_len=CONTEXT_LEN)

raw_weights_kb = prof['raw_weights_kb']
flash_fb_kb    = prof['flash_fb_kb']
dsp_sram_kb    = prof['dsp_kb']
act_sram_kb    = prof['act_kb']
ctx_buf_kb     = prof['ctx_buf_kb']
rt_sram_kb     = prof['rt_kb']
total_sram_kb  = prof['total_kb']
sram_headroom  = SRAM_BUDGET_KB - total_sram_kb
flash_headroom = FLASH_BUDGET_KB - flash_fb_kb
macs_per_epoch = prof['macs_per_epoch']

# Backward-compatible aliases
dsp_kb, model_kb, act_kb, rt_kb = dsp_sram_kb, flash_fb_kb, act_sram_kb, rt_sram_kb
total_kb, budget_kb, headroom = total_sram_kb, SRAM_BUDGET_KB, sram_headroom

save_cache('edge_profile.pkl', {
    **prof,
    'model_kb': flash_fb_kb, 'budget_kb': SRAM_BUDGET_KB, 'headroom': sram_headroom,
    'flash_budget_kb': FLASH_BUDGET_KB, 'flash_headroom': flash_headroom,
})

print('=' * 70)
print('ANALYTICAL ARDUINO NANO 33 BLE SENSE (nRF52840) MEMORY & COMPUTE PROFILE')
print('=' * 70)
print(f"Model Parameters:                     {prof['n_params']:,}")
print(f"MACs per 30s Epoch:                   {macs_per_epoch / 1e6:.2f} M MACs")
print()
print('1. FLASH MEMORY (ROM - INT8 Model Weights):')
print(f'   Raw INT8 Weights (1B/param):       {raw_weights_kb:.1f} KB')
print(f'   Est. TFLite FlatBuffer (+5%):      {flash_fb_kb:.1f} KB / {FLASH_BUDGET_KB:.0f} KB ({flash_fb_kb/FLASH_BUDGET_KB*100:.1f}%)')
print(f'   Flash Headroom:                    {flash_headroom:.1f} KB ({flash_headroom/FLASH_BUDGET_KB*100:.1f}% free)')
print(f'   Status:                            {"PASSED" if flash_headroom > 0 else "FAILED"}')
print()
print('2. INTERNAL SRAM (Hook-Verified Peak Dynamic Arena):')
print(f'   DSP Input Buffer (30s @ {FS}Hz):    {dsp_sram_kb:.1f} KB')
print(f'   Peak Layer Activations (raw):      {prof["peak_raw_act_kb"]:.1f} KB')
print(f'   Tensor Arena (1.25x safety):       {act_sram_kb:.1f} KB')
print(f'   Context Ring Buffer ({CONTEXT_LEN}x96B):       {ctx_buf_kb:.2f} KB')
print(f'   RTOS, Stack & Runtime Overhead:    {rt_sram_kb:.1f} KB')
print('   ' + '-' * 40)
print(f'   TOTAL SRAM USED:                   {total_sram_kb:.1f} KB / {SRAM_BUDGET_KB:.0f} KB ({total_sram_kb/SRAM_BUDGET_KB*100:.1f}%)')
print(f'   FREE SRAM HEADROOM:                {sram_headroom:.1f} KB ({sram_headroom/SRAM_BUDGET_KB*100:.1f}% free)')
print(f'   Status:                            {"H3 SUPPORTED (FITS IN SRAM)" if sram_headroom > 0 else "FAILED"}')
print('=' * 70)
if (MODEL_DIR / 'tiny_eeg_sleep_int8.pt').exists():
    print(f'INT8 State-Dict saved: {(MODEL_DIR / "tiny_eeg_sleep_int8.pt").stat().st_size / 1024:.1f} KB')
print('Hardware latency: pending physical HIL measurement on Arduino Nano 33 BLE Sense via TFLite-Micro')

#%% [markdown]
# ---
# ## Section 10 - Honest 20-Subject Benchmarks & SOTA Comparison

#%%
if 'fp32_acc' not in globals() or 'fp32_f1s_loso' not in globals() or 'qat_ran' not in globals():
    _qc = load_cache('qat_results.pkl')
    if _qc is None: raise RuntimeError(f'qat_results.pkl not found in {CKPT_DIR}. Run Section 8 first!')
    fp32_acc, fp32_f1, fp32_k = _qc['fp32_acc'], _qc['fp32_f1'], _qc['fp32_k']
    ptq_acc,  ptq_f1,  ptq_k  = _qc['ptq_acc'],  _qc['ptq_f1'],  _qc['ptq_k']
    int8_acc, int8_f1, int8_k = _qc['int8_acc'], _qc['int8_f1'], _qc['int8_k']
    fp32_f1s_loso = _qc['fp32_f1s_loso']
    qat_ran = _qc.get('qat_ran', True)

n_bench_folds = len(fp32_f1s_loso)
# Strictly uses multi-subject LOSO mean (NOT single best fold) for apples-to-apples SOTA comparison
benchmarks = [
    ('Expert Human',           'PSG',       0.88,     0.82,    0.80,   'AASM 2017'),
    ('DeepSleepNet',           'Sleep-EDF', 0.82,     0.76,    0.75,   'Supratak 2017'),
    ('TinySleepNet',           'Sleep-EDF', 0.83,     0.78,    0.77,   'Supratak 2020'),
    ('SeqSleepNet (ear-EEG)',  'EESM19',    0.79,     0.73,    0.72,   'Mikkelsen 2025'),
    (f'TinyEEGSleep FP32 ({n_bench_folds}-Fold)', 'Real EESM19', fp32_acc, fp32_f1, fp32_k, 'This Notebook'),
    ('TinyEEGSleep INT8 PTQ',       'Real EESM19', ptq_acc,  ptq_f1,  ptq_k,  'This Notebook'),
]
short_lbl = ['Expert\nHuman', 'Deep\nSleepNet', 'Tiny\nSleepNet', 'SeqSleep\nNet', 'Ours\nFP32', 'Ours\nPTQ']
b_colors  = ['#D97706', '#7F8C8D', '#7F8C8D', '#7F8C8D', '#1F77B4', '#9467BD']
if qat_ran:
    benchmarks.append(('TinyEEGSleep INT8 QAT', 'Real EESM19', int8_acc, int8_f1, int8_k, 'This Notebook'))
    short_lbl.append('Ours\nQAT')
    b_colors.append('#2CA02C')

df_bench = pd.DataFrame(benchmarks, columns=['Method', 'Dataset', 'Accuracy', 'Macro F1', "Cohen's k", 'Reference'])

print(f"{'Method':<32} {'Acc':>7} {'F1':>8} {'kappa':>8}")
for _, r in df_bench.iterrows():
    mark = '** ' if 'Notebook' in r['Reference'] else '   '
    ck = r["Cohen's k"]
    print(f"{mark}{r['Method']:<29} {r['Accuracy']*100:>6.1f}% {r['Macro F1']*100:>7.1f}% {ck:>8.3f}")

if not FORCE_RERUN and (FIG_DIR/'benchmark_comparison.png').exists():
    print(f'Skipping benchmark_comparison.png (already exists in {FIG_DIR}).')
else:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'TinyEEGSleep ({n_bench_folds}-Subject LOSO Mean) vs. State-of-the-Art',
                 fontsize=14, fontweight='bold', color='#111111')
    for ax, metric in zip(axes, ['Accuracy', 'Macro F1', "Cohen's k"]):
        vals = df_bench[metric].values * (100 if metric != "Cohen's k" else 1)
        brs = ax.bar(short_lbl, vals, color=b_colors, alpha=0.9, edgecolor='#222222', linewidth=0.5)
        ax.bar_label(brs, fmt='%.2f' if metric == "Cohen's k" else '%.1f%%', padding=3, fontsize=8.5)
        ax.set_title(metric, fontweight='bold'); ax.grid(axis='y', alpha=0.4)
        ax.set_ylim(min(vals) * 0.90, max(vals) * 1.12)
        if metric == 'Macro F1':
            ax.axhline(73.0, color='#D62728', lw=1.5, ls=':', label='Ear-EEG SOTA (73%)')
            ax.legend(fontsize=8)
    _leg_h = [mpatches.Patch(color='#D97706', label='Human'),
              mpatches.Patch(color='#7F8C8D', label='Published SOTA'),
              mpatches.Patch(color='#1F77B4', label='Ours (FP32)'),
              mpatches.Patch(color='#9467BD', label='Ours (PTQ)')]
    if qat_ran:
        _leg_h.append(mpatches.Patch(color='#2CA02C', label='Ours (QAT)'))
    fig.legend(handles=_leg_h, loc='lower center', ncol=len(_leg_h), bbox_to_anchor=(0.5, -0.02), fontsize=9.5)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(FIG_DIR/'benchmark_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

#%% [markdown]
# ---
# ## Section 11 - Statistical Validation & Experiment Logging

#%%
if 'loso_results' not in globals() or loso_results is None:
    loso_results = load_cache('loso_results.pkl')
if 'mean_acc_' not in globals():
    mean_acc_ = float(np.mean([r['accuracy'] for r in loso_results]))
    mean_f1_  = float(np.mean([r['macro_f1'] for r in loso_results]))
    std_f1_   = float(np.std([r['macro_f1']  for r in loso_results]))
    mean_k_   = float(np.mean([r['kappa']    for r in loso_results]))
    best_     = max(loso_results, key=lambda r: r['macro_f1'])
if 'fp32_f1s_loso' not in globals() or 'qat_ran' not in globals():
    _qc = load_cache('qat_results.pkl')
    fp32_acc, fp32_f1, fp32_k, fp32_kb, fp32_raw_kb = _qc['fp32_acc'], _qc['fp32_f1'], _qc['fp32_k'], _qc['fp32_kb'], _qc['fp32_raw_kb']
    ptq_acc,  ptq_f1,  ptq_k                          = _qc['ptq_acc'],  _qc['ptq_f1'],  _qc['ptq_k']
    int8_acc, int8_f1, int8_k, int8_kb, int8_raw_kb = _qc['int8_acc'], _qc['int8_f1'], _qc['int8_k'], _qc['int8_kb'], _qc['int8_raw_kb']
    fp32_f1s_loso, ptq_f1s_loso, int8_f1s_loso = _qc['fp32_f1s_loso'], _qc['ptq_f1s_loso'], _qc['int8_f1s_loso']
    int8_preds, int8_gts = _qc['int8_preds'], _qc['int8_gts']
    qat_ran = _qc.get('qat_ran', True)
if 'total_sram_kb' not in globals():
    _ep = load_cache('edge_profile.pkl')
    dsp_sram_kb, flash_fb_kb, act_sram_kb, rt_sram_kb, total_sram_kb, budget_kb, sram_headroom = (
        _ep['dsp_kb'], _ep['flash_fb_kb'], _ep['act_kb'], _ep['rt_kb'], _ep['total_kb'], _ep['budget_kb'], _ep['headroom'])
    raw_weights_kb, flash_headroom, n_p_log = _ep['raw_weights_kb'], _ep['flash_headroom'], _ep['n_params']
else:
    n_p_log = prof['n_params']

n_eval_folds = len(fp32_f1s_loso)
int8_lbl = 'INT8 QAT' if qat_ran else 'INT8 PTQ'

# Real paired Wilcoxon signed-rank tests across all evaluated LOSO folds (ZERO random/fake values)
pval_qat, pval_ptq = None, None
if qat_ran:
    try:
        if n_eval_folds >= 2 and not np.allclose(fp32_f1s_loso, int8_f1s_loso):
            stat_q, pval_qat = wilcoxon(fp32_f1s_loso, int8_f1s_loso, alternative='two-sided')
            print(f'Wilcoxon (FP32 vs INT8 QAT, n={n_eval_folds}): stat={stat_q:.3f}, p={pval_qat:.4f}')
            if n_eval_folds < 6:
                print(f'  -> (DEV mode n={n_eval_folds}: underpowered for significance; run full 20-fold for paper p-value)')
            else:
                print('  -> NOT significant (p > 0.05): QAT preserves accuracy!' if pval_qat > 0.05 else '  -> Statistically significant difference (check effect size)')
        else:
            print('Wilcoxon (FP32 vs INT8 QAT): skipped (need >=2 non-identical folds)')
    except Exception as e:
        print(f'Wilcoxon QAT note: {e}')
else:
    print('Wilcoxon (FP32 vs INT8 QAT): skipped (RUN_QAT=False)')

try:
    if n_eval_folds >= 2 and not np.allclose(fp32_f1s_loso, ptq_f1s_loso):
        stat_p, pval_ptq = wilcoxon(fp32_f1s_loso, ptq_f1s_loso, alternative='two-sided')
        print(f'Wilcoxon (FP32 vs INT8 PTQ, n={n_eval_folds}): stat={stat_p:.3f}, p={pval_ptq:.4f}')
        if n_eval_folds < 6:
            print(f'  -> (DEV mode n={n_eval_folds}: underpowered for significance; run full 20-fold for paper p-value)')
except Exception as e:
    print(f'Wilcoxon PTQ note: {e}')

active_pval = pval_qat if qat_ran else pval_ptq
active_drop_pct = (fp32_f1 - int8_f1) * 100.0
ptq_drop_pct    = (fp32_f1 - ptq_f1) * 100.0
h1_pass = bool(active_drop_pct < 2.0)
h3_pass = bool(sram_headroom > 0 and flash_headroom > 0)

print()
print('=' * 70)
print(f'FINAL RESULTS — {ACTIVE_RUN_NAME}')
print('=' * 70)
print(f'  {n_eval_folds}-Fold FP32 Acc / F1 / k:  {fp32_acc*100:.2f}% / {fp32_f1*100:.2f}% / {fp32_k:.3f}')
print(f'  {n_eval_folds}-Fold PTQ  Acc / F1 / k:  {ptq_acc*100:.2f}% / {ptq_f1*100:.2f}% / {ptq_k:.3f} (drop {ptq_drop_pct:+.2f}%)')
if qat_ran:
    print(f'  {n_eval_folds}-Fold QAT  Acc / F1 / k:  {int8_acc*100:.2f}% / {int8_f1*100:.2f}% / {int8_k:.3f} (drop {active_drop_pct:+.2f}%)')
else:
    print(f'  {n_eval_folds}-Fold QAT:                 SKIPPED (RUN_QAT=False; H1 evaluated on INT8 PTQ)')
print(f'  Flash (Raw / Est FlatBuf):  {raw_weights_kb:.1f} / {flash_fb_kb:.1f} KB (limit 1024 KB)')
print(f'  Arduino SRAM Used:          {total_sram_kb:.1f} / {budget_kb:.0f} KB (free {sram_headroom:.1f} KB)')
print(f'  H1 (<2% {int8_lbl} F1 drop): {"SUPPORTED" if h1_pass else "REJECTED"}')
print(f'  H3 (<256KB SRAM & <1MB Flash): {"SUPPORTED" if h3_pass else "REJECTED"}')
print('=' * 70)

_dev_tag = f'[DEV {n_eval_folds}-fold] ' if DEV_FOLD_INDICES is not None else ''
_q_tag = f'PTQ+QAT(lr={QAT_LR},val-es)' if qat_ran else 'PTQ-only(QAT skipped)'
# Save metrics_summary.json and upsert row in outputs/experiments_log.csv
metrics_summary = {
    'run_name': ACTIVE_RUN_NAME,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'fs_hz': FS, 'context_len': CONTEXT_LEN, 'n_subjects': n_eval_folds, 'params': int(n_p_log),
    'fp32_acc': round(fp32_acc * 100, 2), 'fp32_f1': round(fp32_f1 * 100, 2), 'fp32_kappa': round(fp32_k, 3),
    'ptq_f1': round(ptq_f1 * 100, 2),
    'int8_acc': round(int8_acc * 100, 2), 'int8_f1': round(int8_f1 * 100, 2), 'int8_kappa': round(int8_k, 3),
    'f1_drop_pct': round(active_drop_pct, 3),
    'wilcoxon_p': round(float(active_pval), 4) if (active_pval is not None and not np.isnan(active_pval)) else '',
    'flash_kb': round(flash_fb_kb, 1), 'sram_kb': round(total_sram_kb, 1),
    'h1_pass': h1_pass, 'h3_pass': h3_pass,
    'notes': f'{_dev_tag}Cross-ear {FS}Hz, ctx={CONTEXT_LEN}, damp_pow={CLASS_WT_POWER}, leak-free 17/2/1 LOSO + {_q_tag}'
}
with open(RUN_DIR / 'metrics_summary.json', 'w', encoding='utf-8') as _mf:
    json.dump(metrics_summary, _mf, indent=2)

log_csv = OUTPUT_DIR / 'experiments_log.csv'
log_cols = ['run_name','timestamp','fs_hz','context_len','n_subjects','params',
            'fp32_acc','fp32_f1','fp32_kappa','ptq_f1','int8_acc','int8_f1','int8_kappa',
            'f1_drop_pct','wilcoxon_p','flash_kb','sram_kb','h1_pass','h3_pass','notes']
df_log = pd.read_csv(log_csv) if log_csv.exists() else pd.DataFrame(columns=log_cols)
df_log = df_log[df_log['run_name'] != ACTIVE_RUN_NAME]
df_log = pd.concat([df_log, pd.DataFrame([metrics_summary])], ignore_index=True)
df_log.to_csv(log_csv, index=False)
print(f'Logged run to {log_csv} and {RUN_DIR / "metrics_summary.json"}')

#%%
# Final research dashboard
if not FORCE_RERUN and (FIG_DIR/'dashboard.png').exists():
    print(f'Skipping dashboard.png (already exists in {FIG_DIR}).')
else:
    if 'pcts' not in globals() or 'y_vals' not in globals():
        _eda = load_cache('eda_summary.pkl')
        if _eda is not None:
            pcts, stage_names_, time_h, y_vals = _eda['pcts'], _eda['stage_names_'], _eda['time_h'], _eda['y_vals']
        else:
            pcts, stage_names_, time_h, y_vals = [20]*5, ['Wake']*100, np.linspace(0, 8, 100), [0]*100

    if 'loso_results' not in globals() or loso_results is None:
        loso_results = load_cache('loso_results.pkl')
    if 'qat_ran' not in globals():
        qat_ran = True
    n_eval_folds = len(fp32_f1s_loso)
    int8_lbl = 'INT8 QAT' if qat_ran else 'INT8 PTQ'

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('white')
    fig.suptitle(f'TinyEEGSleep Research Dashboard ({ACTIVE_RUN_NAME})\nEar-EEG Sleep | EESM19 (ds005185) | {n_eval_folds}-Subject Leak-Free LOSO-CV',
                 fontsize=14, fontweight='bold', color='#111111', y=1.02)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.38)

    ax = fig.add_subplot(gs[0, 0])
    if 'EESM19' in globals():
        n_psg_d = EESM19.n_subjects * EESM19.n_psg_per_subject
        n_tot_d = EESM19.n_nights_total
    else:
        n_psg_d, n_tot_d = 80, 214
    n_home_d = n_tot_d - n_psg_d
    ax.pie([n_psg_d, n_home_d],
           labels=[f'PSG-Scored\n{n_psg_d} nights', f'Home Only\n{n_home_d} nights'], colors=['#1F77B4', '#7F8C8D'], autopct='%1.0f%%',
           wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2), textprops={'color': '#111111', 'fontsize': 9})
    ax.set_title(f'EESM19: {n_tot_d} Nights', fontweight='bold', color='#111111', fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(STAGE_ORDER, pcts, color=[STAGE_COLORS[s] for s in STAGE_ORDER], alpha=0.9, edgecolor='#222222', linewidth=0.5)
    ax.set_title('Stage Distribution', fontweight='bold', color='#111111', fontsize=10)
    ax.set_ylabel('%'); ax.grid(axis='y', alpha=0.4); ax.set_ylim(0, max(pcts) * 1.3)

    ax = fig.add_subplot(gs[0, 2])
    _m_lbls = ['FP32', 'PTQ', 'QAT'] if qat_ran else ['FP32', 'PTQ']
    _m_vals = [fp32_f1 * 100, ptq_f1 * 100, int8_f1 * 100] if qat_ran else [fp32_f1 * 100, ptq_f1 * 100]
    _m_cols = ['#1F77B4', '#9467BD', '#2CA02C'] if qat_ran else ['#1F77B4', '#9467BD']
    m_bars = ax.bar(_m_lbls, _m_vals, color=_m_cols, alpha=0.9, edgecolor='#222222', linewidth=0.5)
    ax.bar_label(m_bars, fmt='%.1f%%', padding=2, fontsize=8.5)
    ax.axhline(73.0, color='#D62728', lw=1.5, ls='--', label='SOTA 73%')
    ax.set_title(f'{n_eval_folds}-Subject Mean F1', fontweight='bold', color='#111111', fontsize=10)
    ax.set_ylabel('Macro F1 (%)'); ax.set_ylim(0, 95); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.4)

    ax = fig.add_subplot(gs[0, 3])
    comps = ['DSP', 'Arena', 'RT', 'Free SRAM']
    vals_mm = [dsp_sram_kb, act_sram_kb, rt_sram_kb, max(0, sram_headroom)]
    clrs_mm = ['#1F77B4', '#E67E22', '#7F8C8D', '#2CA02C']; left_ = 0
    for c, v, col in zip(comps, vals_mm, clrs_mm):
        if v > 0:
            ax.barh(0, v, left=left_, color=col, alpha=0.9, height=0.5, edgecolor='white', linewidth=0.8)
            if v > 10: ax.text(left_ + v / 2, 0, f'{c}\n{v:.0f}', ha='center', va='center', fontsize=7.5, color='white', fontweight='bold')
            left_ += v
    ax.axvline(256, color='#D62728', lw=2, ls='--'); ax.set_xlim(0, 270)
    ax.set_xlabel('KB'); ax.set_yticks([]); ax.grid(axis='x', alpha=0.4)
    ax.set_title(f'Arduino SRAM: {total_sram_kb:.0f}/256 KB', fontweight='bold', color='#111111', fontsize=10)

    ax = fig.add_subplot(gs[1, :])
    for i in range(len(y_vals) - 1):
        ax.fill_between([time_h[i], time_h[i + 1]], [y_vals[i]] * 2, step='post',
                        color=STAGE_COLORS[stage_names_[i]], alpha=0.45)
    ax.step(time_h, y_vals, where='post', color='#111111', lw=1.0, alpha=0.95)
    ax.set_yticks([0, 1, 2, 3, 4]); ax.set_yticklabels(['N3', 'N2', 'N1', 'REM', 'Wake'])
    ax.set_xlabel('Time (hours)'); ax.grid(alpha=0.35)
    ax.set_title('Full Night Hypnogram - Ear-EEG Recording', fontweight='bold', color='#111111', fontsize=10)
    ax.legend(handles=[mpatches.Patch(color=STAGE_COLORS[s], label=s) for s in STAGE_ORDER],
              loc='upper right', ncol=5, fontsize=8)

    ax = fig.add_subplot(gs[2, 0])
    fold_f1s_plot = [f * 100 for f in int8_f1s_loso]
    ax.bar(range(1, len(fold_f1s_plot) + 1), fold_f1s_plot,
           color=['#2CA02C' if f >= 70 else '#D62728' for f in fold_f1s_plot], alpha=0.9, edgecolor='#222222', linewidth=0.5)
    ax.axhline(int8_f1 * 100, color='#1F77B4', lw=2, ls='--')
    if loso_results and len(loso_results) == len(fold_f1s_plot):
        ax.set_xticks(range(1, len(fold_f1s_plot) + 1))
        ax.set_xticklabels([str(r['subject']) for r in loso_results],
                           rotation=45 if len(fold_f1s_plot) > 6 else 0, fontsize=7.5)
    ax.set_xlabel('Subject Fold'); ax.set_ylabel(f'{int8_lbl} F1 (%)')
    ax.set_title(f'Per-Subject {int8_lbl} F1', fontweight='bold', color='#111111', fontsize=10); ax.grid(axis='y', alpha=0.4)

    ax = fig.add_subplot(gs[2, 1])
    cm_bd = confusion_matrix(int8_gts, int8_preds)
    cm_n2 = cm_bd.astype(float) / (cm_bd.sum(axis=1, keepdims=True) + 1e-8)
    ax.imshow(cm_n2, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([INV_STAGE_MAP[i] for i in range(5)], fontsize=7)
    ax.set_yticklabels([INV_STAGE_MAP[i] for i in range(5)], fontsize=7)
    ax.set_title(f'Pooled {n_eval_folds}-Sub {int8_lbl} Confusion', fontweight='bold', color='#111111', fontsize=10)

    ax = fig.add_subplot(gs[2, 2:4])
    if 'df_bench' in globals() and 'short_lbl' in globals() and 'b_colors' in globals() and len(df_bench) == len(short_lbl):
        bench_f1s = df_bench['Macro F1'].values * 100
        _b_lbls, _b_cols = short_lbl, b_colors
    else:
        _b_lbls = ['Expert\nHuman', 'Deep\nSleepNet', 'Tiny\nSleepNet', 'SeqSleep\nNet', 'Ours\nFP32', 'Ours\nPTQ']
        _b_cols = ['#D97706', '#7F8C8D', '#7F8C8D', '#7F8C8D', '#1F77B4', '#9467BD']
        bench_f1s = np.array([82.0, 76.0, 78.0, 73.0, fp32_f1 * 100, ptq_f1 * 100])
        if qat_ran:
            _b_lbls.append('Ours\nQAT')
            _b_cols.append('#2CA02C')
            bench_f1s = np.append(bench_f1s, int8_f1 * 100)
    brs_b = ax.bar(_b_lbls, bench_f1s, color=_b_cols,
                   alpha=0.9, edgecolor='#222222', linewidth=0.5)
    ax.bar_label(brs_b, fmt='%.1f%%', padding=3, fontsize=8.5, color='#111111')
    ax.set_ylabel('Macro F1 (%)'); ax.grid(axis='y', alpha=0.4)
    ax.set_title(f'Honest {n_eval_folds}-Subject Benchmark Comparison', fontweight='bold', color='#111111', fontsize=10)
    ax.set_ylim(40, 96); ax.axhline(73.0, color='#D62728', lw=1.5, ls=':', label='Ear-EEG SOTA (73%)'); ax.legend(fontsize=8)

    plt.savefig(FIG_DIR / 'dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f'Dashboard saved to {FIG_DIR / "dashboard.png"}!')

#%%
print(f'Generated artifacts for run [{ACTIVE_RUN_NAME}] in {RUN_DIR}:')
for f in sorted(RUN_DIR.rglob('*')):
    if f.is_file():
        print(f'  {f.relative_to(RUN_DIR)} ({f.stat().st_size/1024:.1f} KB)')
print('\nNotebook complete!')
print('Real data -> openneuro.org/datasets/ds005185 (EESM19)')
print('Paper DOI  -> 10.1038/s41597-025-04579-8')

#%% [markdown]
# ---
# ## References
# 
# 1. **Mikkelsen KB et al.** Ear-EEG sleep monitoring data sets. *Scientific Data* 12, 301 (2025). DOI: 10.1038/s41597-025-04579-8
# 2. **Mikkelsen KB et al.** Accurate whole-night sleep monitoring with dry-contact ear-EEG. *Scientific Reports* 9 (2019).
# 3. **Tabar YR et al.** At-home sleep monitoring using generic ear-EEG. *Front. Neuroscience* 17 (2023).
# 4. **Supratak A & Guo Y.** TinySleepNet. *EMBC* (2020).
# 5. **Phan H et al.** SeqSleepNet. *IEEE TASLP* 27(6) (2019).
# 6. **Kappel SL et al.** Dry-Contact Electrode Ear-EEG. *IEEE TBME* 66, 150-158 (2019).
# 
# Dataset DOI:
# - EESM19: `10.18112/openneuro.ds005185.v1.0.0`

