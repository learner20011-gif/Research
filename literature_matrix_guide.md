# 📚 Comprehensive Literature Review Matrix — Template Guide

**File:** [Literature_Review_Matrix.xlsx](file:///d:/Research/Literature_Review_Matrix.xlsx)  
**Domain Focus:** Engineering · Electronics · EE · ML · Neuroscience Data Analysis

---

## 📂 Sheet Overview (8 Sheets)

| Sheet | Purpose |
|-------|---------|
| 📖 How to Use | Step-by-step instructions & heuristics |
| 📚 Literature Matrix | **Main table** — one row per paper, 38 columns |
| 🔍 Gap Tracker | Synthesize & score identified research gaps |
| 🗺️ Domain Coverage Map | Heatmap: domains × techniques (zeros = gaps!) |
| ⚙️ Methodology Comparison | Side-by-side model/method benchmarking |
| 📈 Temporal Trends | Publication volume by topic & year |
| 💾 Dataset Inventory | Key datasets with limitations & access info |
| 📋 Reading Tracker | Manage reading workflow & status |

---

## 🎯 How to Find Research Gaps

### Quick-Start Steps
1. **Add papers** → `📚 Literature Matrix` (one row per paper, assign P001, P002…)
2. **Tag domains** → Mark Y/N for EE, Electronics, Signal Processing, etc.
3. **Fill gap columns** → "Identified Gaps", "Limitations Stated", "Future Work"
4. **Synthesize** → Move recurring gaps to `🔍 Gap Tracker`
5. **Score gaps** → Priority = Novelty × Feasibility × Impact (max 125)
6. **Update heatmap** → `🗺️ Domain Coverage Map` — zeros = unexplored intersections

### Gap Heuristics
- **Zero cells in Domain Map** → No paper combines that domain + technique
- **≥3 papers share same Future Work theme** → High-priority validated gap
- **Cross-domain absences** → e.g., FPGA + EEG/ECG bio-signals = almost zero
- **Lab-only papers** → Real-time / embedded / federated deployment is almost always a gap
- **Single-dataset papers** → Validation gap; multi-site or cross-population needed

---

## 📊 Pre-filled Sample Data (5 Real-World Papers)

| ID | Paper | Domain | Key Gap Identified |
|----|-------|--------|-------------------|
| P001 | CNN-LSTM EEG Motor Imagery | Neuro + ML | No cross-subject transfer learning |
| P002 | Wavelet-LSTM Grid Fault Detection | EE + ML | Single topology; no multi-fault |
| P003 | Neuromorphic Chip (SNN) | Electronics + HW | Image-only; no bio-signal benchmark |
| P004 | ViT-1D ECG Arrhythmia | ML + Cardiac | No wearable deployment or federated |
| P005 | FPGA CNN Acceleration | Embedded + AI | Bio-signal edge inference unexplored |

---

## 🔑 Priority Score Formula

```
Priority = Novelty (1-5) × Feasibility (1-5) × Impact (1-5)
```

| Score Range | Interpretation |
|-------------|---------------|
| 75–125 | 🔴 High Priority — Strong research gap, pursue first |
| 30–74  | 🟡 Medium Priority — Worth investigating |
| 1–29   | 🟢 Low Priority — Incremental or difficult |

---

## 💾 Pre-loaded Datasets in Inventory

| Dataset | Domain | Access |
|---------|--------|--------|
| BCI Competition IV | EEG / BCI | Free |
| PhysioNet / MIMIC-III | ECG / Clinical | Free |
| PTB-XL | ECG 12-lead | CC BY 4.0 |
| IEEE 13-bus Test Feeder | Power Systems | Free |
| SEED (SJTU) | EEG / Emotion | Conditional |
| CWRU Bearing | EE / Fault Detection | Free |
| Chapman ECG | ECG / Arrhythmia | CC BY 4.0 |

---

> [!TIP]
> Use Excel's **AutoFilter** (`Data → Filter`) on the main matrix to slice by domain, year, or method type for targeted gap mining.

> [!IMPORTANT]
> The **Domain Coverage Map** is your most powerful gap-finding tool. Any cell showing **0** means that domain + technique combination has **not been studied** — that's your research opportunity.
