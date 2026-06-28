import openpyxl
from openpyxl.styles import (
    PatternFill, Font, Alignment, Border, Side, GradientFill
)
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.chart import BarChart, Reference
from openpyxl.formatting.rule import ColorScaleRule, DataBarRule
import datetime

# ─── Color Palette ────────────────────────────────────────────────────────────
CLR = {
    "navy":       "1B2A4A",
    "teal":       "0D7377",
    "teal_light": "14BDAC",
    "gold":       "F0A500",
    "amber":      "E07B00",
    "slate":      "4A5568",
    "steel":      "718096",
    "ice":        "EBF4F5",
    "sky":        "D4EFF0",
    "mint":       "D4F5E9",
    "peach":      "FFF3CD",
    "rose":       "FCE4EC",
    "lavender":   "EDE7F6",
    "white":      "FFFFFF",
    "light_gray": "F7F8FA",
    "mid_gray":   "E2E8F0",
    "dark_gray":  "2D3748",
}

def fill(hex_col):
    return PatternFill("solid", fgColor=hex_col)

def font(bold=False, size=11, color="000000", italic=False):
    return Font(bold=bold, size=size, color=color, italic=italic, name="Calibri")

def border(style="thin"):
    s = Side(style=style, color="CBD5E0")
    return Border(left=s, right=s, top=s, bottom=s)

def center():
    return Alignment(horizontal="center", vertical="center", wrap_text=True)

def left():
    return Alignment(horizontal="left", vertical="center", wrap_text=True)

def style_header(ws, row, col, value, bg=CLR["navy"], fg=CLR["white"],
                 sz=11, bold=True, center_align=True):
    c = ws.cell(row=row, column=col, value=value)
    c.fill = fill(bg)
    c.font = font(bold=bold, size=sz, color=fg)
    c.alignment = center() if center_align else left()
    c.border = border()
    return c

def style_subheader(ws, row, col, value):
    c = ws.cell(row=row, column=col, value=value)
    c.fill = fill(CLR["teal"])
    c.font = font(bold=True, size=10, color=CLR["white"])
    c.alignment = center()
    c.border = border()
    return c

def style_cell(ws, row, col, value="", bg=CLR["white"]):
    c = ws.cell(row=row, column=col, value=value)
    c.fill = fill(bg)
    c.font = font(size=10)
    c.alignment = left()
    c.border = border()
    return c

def merge_header(ws, text, row, col_start, col_end, bg=CLR["navy"], sz=12):
    ws.merge_cells(start_row=row, start_column=col_start,
                   end_row=row, end_column=col_end)
    c = ws.cell(row=row, column=col_start, value=text)
    c.fill = fill(bg)
    c.font = font(bold=True, size=sz, color=CLR["white"])
    c.alignment = center()
    c.border = border()

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 1 — MAIN LITERATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════
def build_main_matrix(wb):
    ws = wb.create_sheet("📚 Literature Matrix", 0)
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "C4"

    # ── Title banner ──
    ws.merge_cells("A1:BH2") # Extended for 60 columns
    c = ws.cell(row=1, column=1,
                value="📚 COMPREHENSIVE LITERATURE REVIEW MATRIX — Research Gap Analysis")
    c.fill = fill(CLR["navy"])
    c.font = Font(bold=True, size=16, color=CLR["white"], name="Calibri")
    c.alignment = center()
    ws.row_dimensions[1].height = 35
    ws.row_dimensions[2].height = 5

    # ── Section group headers (row 3) ──
    sections = [
        (1,  5,  "1. METADATA & IDENTIFIERS",   CLR["navy"]),
        (6,  10, "2. THEORY & CONCEPTS",        CLR["teal"]),
        (11, 16, "3. METHODOLOGY & DESIGN",     CLR["slate"]),
        (17, 20, "4. SAMPLE & CONTEXT",         CLR["navy"]),
        (21, 26, "5. DOMAIN COVERAGE",          CLR["teal"]),
        (27, 31, "6. DATA & DATASETS",          CLR["slate"]),
        (32, 36, "7. ML / AI TECHNIQUES",       CLR["navy"]),
        (37, 39, "8. NEURO / BIO SIGNAL",       CLR["teal"]),
        (40, 47, "9. FINDINGS & OUTCOMES",      CLR["slate"]),
        (48, 56, "10. GAP-FINDING & CRITIQUE",  CLR["gold"]),
        (57, 60, "11. SYNTHESIS",               CLR["navy"]),
    ]
    for col_s, col_e, label, bg in sections:
        merge_header(ws, label, 3, col_s, col_e, bg=bg, sz=10)
    ws.row_dimensions[3].height = 30

    # ── Column headers (row 4) ──
    headers = [
        # 1. METADATA
        "Paper ID\n(P001…)",
        "Full Citation\n(Author, Year, Journal/Conf)",
        "Publication\nType",
        "Year",
        "Funding Source\n/ Sponsor",
        # 2. THEORY & CONCEPTS
        "Research\nQuestion(s)",
        "Theoretical\nFramework",
        "Definitions of\nCore Concepts",
        "Key\nAssumptions",
        "Epistemological\nStance",
        # 3. METHODOLOGY
        "Methodology\n/ Study Design",
        "Level of\nAnalysis",
        "Data Collection\nTimeline",
        "Analytical /\nStatistical Tools",
        "Intervention\nDetails",
        "Validation\nMethod",
        # 4. SAMPLE & CONTEXT
        "Sample Size\n/ Participants",
        "Geographic /\nCultural Context",
        "Time Period\n/ Era",
        "Stakeholder\nPerspective",
        # 5. DOMAIN COVERAGE
        "Electrical\nEngineering",
        "Electronics\n/ Circuits",
        "Signal\nProcessing",
        "Power\nSystems",
        "IoT /\nEmbedded",
        "Control\nSystems",
        # 6. DATA & DATASETS
        "Dataset\nName",
        "Data\nType",
        "Data\nSource",
        "Preprocessing\nSteps",
        "Open\nAccess?",
        # 7. ML / AI TECHNIQUES
        "ML Model\nUsed",
        "Deep Learning\n(Y/N)",
        "Architecture\nType",
        "Feature\nEngineering",
        "Optimizer /\nLoss Fn",
        # 8. NEURO / BIO SIGNAL
        "EEG / EMG\n/ ECG",
        "Brain Region\n/ Signal Band",
        "Neuro\nApplication",
        # 9. FINDINGS
        "Key\nVariables",
        "Key Findings /\nContributions",
        "Accuracy\n(%)",
        "F1 / AUC",
        "RMSE /\nMAE",
        "Comparison\nBaseline",
        "Effect Size\n/ Power",
        "Practical\nImplications",
        # 10. GAP-FINDING
        "Identified\nGaps",
        "Limitations\nStated",
        "Authors'\nFuture Directions",
        "Excluded Demographics\n/ Contexts",
        "Neglected\nVariables",
        "Methodological\nWeaknesses",
        "Alternative\nExplanations",
        "Ecological\nValidity",
        "Risk of\nBias",
        # 11. SYNTHESIS
        "Relevance\n(1-5 ★)",
        "Conflicts &\nTensions",
        "The Unanswered\nQuestion",
        "Personal\nNotes",
    ]

    col_widths = [
        # Metadata
        10, 40, 15, 7, 25,
        # Theory
        35, 30, 30, 30, 20,
        # Methodology
        25, 20, 20, 25, 25, 18,
        # Sample
        20, 20, 15, 20,
        # Domain
        12, 12, 14, 12, 12, 14,
        # Data
        20, 15, 18, 22, 10,
        # ML
        20, 10, 18, 20, 18,
        # Neuro
        15, 18, 20,
        # Findings
        25, 35, 10, 10, 12, 20, 15, 30,
        # Gaps
        35, 30, 30, 25, 25, 30, 30, 20, 20,
        # Synthesis
        12, 35, 35, 35,
    ]

    row4_bgs = [
        # Metadata
        CLR["navy"], CLR["navy"], CLR["navy"], CLR["navy"], CLR["navy"],
        # Theory
        CLR["teal"], CLR["teal"], CLR["teal"], CLR["teal"], CLR["teal"],
        # Methodology
        CLR["slate"], CLR["slate"], CLR["slate"], CLR["slate"], CLR["slate"], CLR["slate"],
        # Sample
        CLR["navy"], CLR["navy"], CLR["navy"], CLR["navy"],
        # Domain
        CLR["teal"], CLR["teal"], CLR["teal"], CLR["teal"], CLR["teal"], CLR["teal"],
        # Data
        CLR["slate"], CLR["slate"], CLR["slate"], CLR["slate"], CLR["slate"],
        # ML
        CLR["navy"], CLR["navy"], CLR["navy"], CLR["navy"], CLR["navy"],
        # Neuro
        CLR["teal"], CLR["teal"], CLR["teal"],
        # Findings
        CLR["slate"], CLR["slate"], CLR["slate"], CLR["slate"], CLR["slate"], CLR["slate"], CLR["slate"], CLR["slate"],
        # Gaps
        CLR["gold"], CLR["gold"], CLR["gold"], CLR["gold"], CLR["gold"], CLR["gold"], CLR["gold"], CLR["gold"], CLR["gold"],
        # Synthesis
        CLR["navy"], CLR["navy"], CLR["navy"], CLR["navy"],
    ]

    for i, (h, w, bg) in enumerate(zip(headers, col_widths, row4_bgs), start=1):
        style_header(ws, 4, i, h, bg=bg)
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.row_dimensions[4].height = 40

    # ── Sample data rows (5-6) ──
    sample_rows = [
        ["P001", "Smith et al. (2023). Hybrid CNN-LSTM for EEG Motor Imagery.",
         "Journal", 2023, "NSF Grant",
         "Can CNN-LSTM hybrids improve motor imagery BCI accuracy?", "Deep Learning feature extraction", "Motor imagery: mental execution of movement", "EEG noise can be ICA removed", "Positivism",
         "Controlled Lab", "Individual", "Cross-sectional", "ANOVA, PyTorch", "N/A", "10-fold CV",
         "109 subjects", "USA", "2020-2022", "Clinical/Researcher",
         "N", "N", "Y", "N", "N", "N",
         "BCI Competition IV", "EEG (64-ch)", "PhysioNet", "Bandpass + ICA", "Yes",
         "CNN-LSTM", "Y", "Hybrid CNN-LSTM", "CSP + PSD", "Adam / CrossEntropy",
         "EEG", "Mu/Beta band", "Motor Imagery BCI",
         "IV: CNN-LSTM model, DV: Classification accuracy", "Novel attention mechanism; 94% acc", "94.2", "0.941", "—", "EEGNet", "Large (Cohen's d=1.2)", "Deployable for severe motor impairment",
         "Small subject pool", "Lab conditions only", "Real-time deployment", "Non-western populations", "Fatigue over time", "No real-time test", "Overfitting to dataset", "Low", "Low",
         "5", "Contradicts paper P003 on latency", "How does it perform on mobile edge?", "Strong baseline for neuro-ML work"],
         
         ["P002", "Kumar & Lee (2022). Fault Detection in Power Grids via LSTM.",
         "Journal", 2022, "DOE GridMod",
         "How effectively can LSTM classify distribution network faults?", "Time-series anomaly detection", "Grid fault: deviation >10% nominal V", "Simulated faults represent real faults", "Pragmatism",
         "Simulation + Real Grid", "System level", "Longitudinal (1 yr SCADA)", "TensorFlow, GridLab-D", "N/A", "Train/Test 80/20",
         "5000 fault events", "Europe", "2021", "Utility Operators",
         "Y", "N", "Y", "Y", "N", "Y",
         "IEEE 13-bus", "Time-series voltage", "Simulated + SCADA", "Wavelet decomposition", "Partial",
         "LSTM", "Y", "Stacked LSTM", "Wavelet features", "RMSprop / BCE",
         "N/A", "N/A", "N/A",
         "IV: Wavelet features, DV: Fault type/location", "First wavelet-LSTM integration for grids", "97.8", "0.975", "0.032", "SVM", "High", "Can reduce utility downtime by 20%",
         "No multi-fault scenario", "Idealized simulation data", "Adversarial fault injection", "Microgrids with high renewable penetration", "Weather correlates", "Single feeder topology", "Sensors miscalibrated", "Medium", "Low",
         "4", "None yet", "Does it scale to 10k bus systems?", "Relevant to smart grid EE gap"]
    ]

    alt_bgs = [CLR["white"], CLR["light_gray"]]
    for r_idx, row_data in enumerate(sample_rows):
        r = r_idx + 5
        bg = alt_bgs[r_idx % 2]
        for c_idx, val in enumerate(row_data, start=1):
            style_cell(ws, r, c_idx, val, bg=bg)
        ws.row_dimensions[r].height = 55

    # ── Empty rows for user data ──
    for r_idx in range(10):
        r = 7 + r_idx
        bg = alt_bgs[r_idx % 2]
        style_cell(ws, r, 1, f"P{r_idx+3:03d}", bg=bg)
        for c_idx in range(2, 61):
            style_cell(ws, r, c_idx, "", bg=bg)
        ws.row_dimensions[r].height = 50

    # ── Data validations ──
    pub_types = '"Journal,Conference,Review,Book Chapter,Thesis,Preprint,Patent"'
    dv_pub = DataValidation(type="list", formula1=pub_types, showDropDown=False)
    dv_pub.sqref = "C5:C100"
    ws.add_data_validation(dv_pub)

    yn_list = '"Y,N,Partial"'
    # Domain coverage and Deep learning (cols 21-26, 33)
    for col_letter in ["U", "V", "W", "X", "Y", "Z", "AG"]:
        dv = DataValidation(type="list", formula1=yn_list, showDropDown=False)
        dv.sqref = f"{col_letter}5:{col_letter}100"
        ws.add_data_validation(dv)

    rel_list = '"1,2,3,4,5"'
    dv_rel = DataValidation(type="list", formula1=rel_list, showDropDown=False)
    dv_rel.sqref = "BE5:BE100" # Relevance column is BE (57)
    ws.add_data_validation(dv_rel)

    # ── Conditional formatting — relevance column (col 57) ──
    color_scale = ColorScaleRule(
        start_type="num", start_value=1, start_color="FCE4EC",
        mid_type="num", mid_value=3, mid_color="FFF3CD",
        end_type="num", end_value=5, end_color="D4F5E9"
    )
    ws.conditional_formatting.add("BE5:BE100", color_scale)

    return ws


# ══════════════════════════════════════════════════════════════════════════════
# SHEET 2 — RESEARCH GAP TRACKER
# ══════════════════════════════════════════════════════════════════════════════
def build_gap_tracker(wb):
    ws = wb.create_sheet("🔍 Gap Tracker")
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "A4"

    ws.merge_cells("A1:N2")
    c = ws.cell(row=1, column=1,
                value="🔍 RESEARCH GAP TRACKER — Structured Opportunity Analysis")
    c.fill = fill(CLR["navy"])
    c.font = Font(bold=True, size=15, color=CLR["white"], name="Calibri")
    c.alignment = center()
    ws.row_dimensions[1].height = 35

    headers = [
        "Gap ID", "Gap Description", "Related Papers\n(P001, P002…)",
        "Domain", "Sub-Domain / Topic",
        "Gap Type", "Evidence\nStrength", "Novelty\nScore (1-5)",
        "Feasibility\n(1-5)", "Impact\n(1-5)", "Priority\nScore",
        "Your Hypothesis /\nProposed Solution", "Resources\nNeeded", "Status"
    ]
    widths = [10, 45, 25, 18, 25, 20, 15, 12, 12, 12, 12, 40, 25, 15]

    for i, (h, w) in enumerate(zip(headers, widths), 1):
        style_header(ws, 3, i, h)
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.row_dimensions[3].height = 40

    gap_types = [
        "Methodological Gap", "Dataset/Data Gap", "Theoretical Gap",
        "Validation Gap", "Application Gap", "Cross-Domain Gap",
        "Hardware-SW Integration Gap", "Real-World Deployment Gap"
    ]

    sample_gaps = [
        ["G001",
         "No cross-subject transfer learning evaluated for EEG motor imagery BCI systems in real-world noisy conditions",
         "P001, P004",
         "Neuroscience / ML",
         "EEG-BCI, Transfer Learning",
         "Methodological Gap",
         "High (4 papers agree)", "5", "4", "5", "=H4*I4*J4",
         "Propose subject-agnostic transformer with domain adaptation pre-training on multi-subject EEG corpora",
         "Multi-subject EEG dataset, GPU cluster, signal processing toolbox",
         "Active Research"],

        ["G002",
         "FPGA-based real-time inference for bio-signal data (EEG/ECG) remains unexplored vs. image-only FPGA acceleration",
         "P003, P005",
         "Electronics / Embedded AI",
         "FPGA, Edge AI, Bio-signals",
         "Cross-Domain Gap",
         "Medium (2 papers)", "5", "4", "5", "=H5*I5*J5",
         "Design HLS-based FPGA pipeline for streaming ECG arrhythmia detection achieving <10ms latency",
         "Xilinx Zynq FPGA board, Vivado HLS, PTB-XL dataset",
         "Proposed"]
    ]

    alt_bgs = [CLR["white"], CLR["light_gray"]]
    for r_idx, row in enumerate(sample_gaps):
        r = r_idx + 4
        bg = alt_bgs[r_idx % 2]
        for c_idx, val in enumerate(row, 1):
            style_cell(ws, r, c_idx, val, bg=bg)
        ws.row_dimensions[r].height = 60

    # empty rows
    for r_idx in range(15):
        r = 6 + r_idx
        bg = alt_bgs[r_idx % 2]
        style_cell(ws, r, 1, f"G{r_idx+3:03d}", bg=bg)
        for c in range(2, 15):
            style_cell(ws, r, c, "", bg=bg)
        ws.row_dimensions[r].height = 50

    # dropdown validations
    dv_gap = DataValidation(type="list",
        formula1='"Methodological Gap,Dataset/Data Gap,Theoretical Gap,Validation Gap,Application Gap,Cross-Domain Gap,Hardware-SW Integration Gap,Real-World Deployment Gap"')
    dv_gap.sqref = "F4:F100"
    ws.add_data_validation(dv_gap)

    dv_status = DataValidation(type="list",
        formula1='"Idea Stage,Proposed,Active Research,In Progress,Completed,Abandoned"')
    dv_status.sqref = "N4:N100"
    ws.add_data_validation(dv_status)

    # color-scale on priority (col 11)
    ws.conditional_formatting.add("K4:K100",
        ColorScaleRule(start_type="min", start_color="FCE4EC",
                       mid_type="percentile", mid_value=50, mid_color="FFF3CD",
                       end_type="max", end_color="D4F5E9"))

    return ws


# ══════════════════════════════════════════════════════════════════════════════
# SHEET 3 — DOMAIN COVERAGE MAP
# ══════════════════════════════════════════════════════════════════════════════
def build_domain_map(wb):
    ws = wb.create_sheet("🗺️ Domain Coverage Map")
    ws.sheet_view.showGridLines = False

    ws.merge_cells("A1:L2")
    c = ws.cell(row=1, column=1, value="🗺️ DOMAIN COVERAGE MAP — Research Landscape Heatmap")
    c.fill = fill(CLR["navy"])
    c.font = Font(bold=True, size=15, color=CLR["white"], name="Calibri")
    c.alignment = center()
    ws.row_dimensions[1].height = 35

    # Instructions
    ws.merge_cells("A3:L3")
    c = ws.cell(row=3, column=1,
                value="📋 Enter a count of papers covering each intersection. Use 0 if none. The heatmap colours update automatically.")
    c.fill = fill(CLR["ice"])
    c.font = font(italic=True, size=10, color=CLR["slate"])
    c.alignment = left()
    ws.row_dimensions[3].height = 22

    domains = [
        "Electrical Eng.", "Electronics", "Signal Processing",
        "Power Systems", "IoT / Embedded", "Control Systems",
        "ML / Deep Learning", "SNN / Neuromorphic", "EEG / BCIs",
        "ECG / Cardiac", "EMG / Motion", "Edge AI / FPGA"
    ]
    techniques = [
        "CNN", "RNN / LSTM", "Transformer", "GNN",
        "SVM / Classical ML", "Wavelet / DSP",
        "Federated Learning", "Transfer Learning",
        "Reinforcement Learning", "SNN / Spike"
    ]

    # Row labels
    style_header(ws, 4, 1, "Domain \\ Technique", bg=CLR["dark_gray"], sz=10)
    for j, t in enumerate(techniques, 2):
        style_header(ws, 4, j, t, bg=CLR["teal"], sz=9)
        ws.column_dimensions[get_column_letter(j)].width = 16
    ws.column_dimensions["A"].width = 22
    ws.row_dimensions[4].height = 35

    seed_data = [
        [5, 4, 3, 1, 2, 1, 0, 3, 1, 0],  # Electrical Eng
        [4, 2, 2, 0, 1, 2, 0, 1, 0, 3],  # Electronics
        [3, 5, 3, 1, 3, 6, 0, 2, 1, 1],  # Signal Processing
        [2, 4, 1, 0, 3, 3, 0, 1, 2, 0],  # Power Systems
        [3, 1, 1, 0, 2, 1, 1, 2, 1, 2],  # IoT/Embedded
        [1, 3, 1, 0, 4, 2, 0, 1, 4, 0],  # Control Systems
        [6, 5, 6, 3, 4, 2, 4, 5, 3, 2],  # ML / DL
        [1, 0, 1, 0, 0, 0, 0, 1, 0, 5],  # SNN
        [2, 4, 4, 1, 1, 2, 1, 4, 0, 2],  # EEG/BCI
        [3, 3, 5, 1, 2, 2, 3, 3, 0, 0],  # ECG
        [2, 2, 2, 0, 1, 2, 0, 2, 1, 1],  # EMG
        [4, 1, 2, 0, 1, 1, 0, 2, 0, 3],  # Edge AI
    ]

    for r_idx, (domain, row_data) in enumerate(zip(domains, seed_data)):
        r = r_idx + 5
        style_header(ws, r, 1, domain, bg=CLR["slate"], sz=10)
        for c_idx, val in enumerate(row_data, 2):
            cell = ws.cell(row=r, column=c_idx, value=val)
            cell.font = font(size=11, bold=True)
            cell.alignment = center()
            cell.border = border()
        ws.row_dimensions[r].height = 25

    # Conditional formatting color scale on data area
    ws.conditional_formatting.add("B5:K16",
        ColorScaleRule(
            start_type="num", start_value=0, start_color="FFFFFF",
            mid_type="num", mid_value=3, mid_color="AED6F1",
            end_type="num", end_value=6, end_color="1B4F72"
        ))

    # Legend
    ws.merge_cells("A18:L18")
    c = ws.cell(row=18, column=1,
                value="💡 Legend: 0 (White) = No coverage → Dark Blue = High coverage | Cells with 0 = RESEARCH GAP opportunities!")
    c.fill = fill(CLR["peach"])
    c.font = font(bold=True, size=11, color=CLR["amber"])
    c.alignment = center()
    ws.row_dimensions[18].height = 28

    return ws


# ══════════════════════════════════════════════════════════════════════════════
# SHEET 4 — METHODOLOGY COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def build_methodology_comparison(wb):
    ws = wb.create_sheet("⚙️ Methodology Comparison")
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "C4"

    ws.merge_cells("A1:R2")
    c = ws.cell(row=1, column=1, value="⚙️ METHODOLOGY COMPARISON TABLE")
    c.fill = fill(CLR["navy"])
    c.font = Font(bold=True, size=15, color=CLR["white"], name="Calibri")
    c.alignment = center()
    ws.row_dimensions[1].height = 35

    headers = [
        "Paper ID", "Method / Model", "Category",
        "Input\nModality", "Parameters\n(M)", "Training\nTime",
        "Inference\nLatency (ms)", "Accuracy\n(%)", "F1-Score",
        "Hardware\nUsed", "Energy\n(W)", "Memory\n(MB)",
        "Scalable?", "Real-Time\nCapable?", "Open\nSource?",
        "Reproducible?", "Strength", "Weakness"
    ]
    widths = [10, 25, 20, 15, 12, 15, 16, 12, 12,
              18, 10, 12, 12, 14, 12, 14, 35, 35]

    for i, (h, w) in enumerate(zip(headers, widths), 1):
        style_header(ws, 3, i, h)
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.row_dimensions[3].height = 40

    data = [
        ["P001", "CNN-LSTM Hybrid", "Deep Learning", "EEG (64-ch)",
         "2.3", "~4h (GPU)", "N/A (offline)", "94.2", "0.941",
         "NVIDIA RTX 3090", "250W (train)", "1,200",
         "Partial", "No", "No", "Partial",
         "Strong temporal + spatial feature extraction; attention enhances class discrimination",
         "High compute; offline only; no subject-independent evaluation"]
    ]

    alt_bgs = [CLR["white"], CLR["light_gray"]]
    for r_idx, row in enumerate(data):
        r = r_idx + 4
        bg = alt_bgs[r_idx % 2]
        for c_idx, val in enumerate(row, 1):
            style_cell(ws, r, c_idx, val, bg=bg)
        ws.row_dimensions[r].height = 55

    for r_idx in range(15):
        r = 5 + r_idx
        bg = alt_bgs[r_idx % 2]
        style_cell(ws, r, 1, "", bg=bg)
        for c in range(2, 19):
            style_cell(ws, r, c, "", bg=bg)
        ws.row_dimensions[r].height = 45

    return ws


# ══════════════════════════════════════════════════════════════════════════════
# SHEET 5 — TEMPORAL TREND ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def build_temporal_trends(wb):
    ws = wb.create_sheet("📈 Temporal Trends")
    ws.sheet_view.showGridLines = False

    ws.merge_cells("A1:N2")
    c = ws.cell(row=1, column=1,
                value="📈 TEMPORAL TREND ANALYSIS — Publication Volume & Topic Evolution")
    c.fill = fill(CLR["navy"])
    c.font = Font(bold=True, size=15, color=CLR["white"], name="Calibri")
    c.alignment = center()
    ws.row_dimensions[1].height = 35

    # Year-by-topic table
    topics = [
        "EEG/BCI + ML", "ECG + DL", "FPGA + AI", "Power Grid + ML",
        "Neuromorphic", "Federated Learning (EE)", "SNN", "IoT + ML",
        "Transfer Learning (Neuro)", "Explainable AI (EE/Medical)"
    ]
    years = list(range(2018, 2026))

    style_header(ws, 4, 1, "Topic / Year", bg=CLR["dark_gray"])
    ws.column_dimensions["A"].width = 28
    for j, yr in enumerate(years, 2):
        style_header(ws, 4, j, str(yr), bg=CLR["teal"])
        ws.column_dimensions[get_column_letter(j)].width = 10
    ws.row_dimensions[4].height = 30

    trend_data = [
        [4, 6, 9, 12, 18, 24, 31, 38],
        [3, 5, 8, 11, 16, 22, 29, 35],
        [2, 3, 5, 7,  9, 12, 15, 19],
        [3, 4, 6, 8, 11, 14, 16, 20],
        [1, 2, 2, 3,  5,  8, 13, 18],
        [0, 0, 1, 2,  5, 10, 17, 25],
        [1, 1, 2, 3,  4,  6,  9, 14],
        [2, 4, 7, 11, 16, 21, 26, 32],
        [0, 1, 2, 4,  7, 12, 18, 24],
        [0, 0, 1, 2,  4,  8, 14, 20],
    ]

    alt_bgs = [CLR["white"], CLR["light_gray"]]
    for r_idx, (topic, row_data) in enumerate(zip(topics, trend_data)):
        r = r_idx + 5
        bg = alt_bgs[r_idx % 2]
        style_header(ws, r, 1, topic, bg=CLR["slate"], sz=10)
        for c_idx, val in enumerate(row_data, 2):
            cell = ws.cell(row=r, column=c_idx, value=val)
            cell.font = font(size=11)
            cell.alignment = center()
            cell.border = border()
            cell.fill = fill(bg)
        ws.row_dimensions[r].height = 22

    # Data bar conditional formatting on each row
    data_bar = DataBarRule(start_type="num", start_value=0,
                           end_type="num", end_value=40,
                           color="0D7377")
    ws.conditional_formatting.add("B5:I14", data_bar)

    # Growth trend notes
    ws.merge_cells("A17:N17")
    c = ws.cell(row=17, column=1,
                value="📌 Key Trend Insights (fill in as you review more papers):")
    c.fill = fill(CLR["navy"])
    c.font = font(bold=True, size=11, color=CLR["white"])
    c.alignment = left()
    ws.row_dimensions[17].height = 25

    insights = [
        "• Federated Learning applied to EE/medical domains is rapidly growing but still underexplored pre-2021 → HIGH GAP opportunity",
        "• SNN / Neuromorphic research is accelerating but bio-signal applications remain sparse → cross-domain gap",
        "• Explainable AI for EE/medical systems is emerging but lacks standardized benchmarks → methodological gap",
        "• FPGA + AI hardware acceleration is growing but mostly for vision — bio-signal edge inference is largely untouched",
        "• Transfer Learning for neural signals (EEG/EMG) is high-growth → subject-independent BCI is a key open problem",
    ]
    for r_idx, text in enumerate(insights):
        r = 18 + r_idx
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=14)
        c = ws.cell(row=r, column=1, value=text)
        c.fill = fill(CLR["light_gray"] if r_idx % 2 == 0 else CLR["white"])
        c.font = font(size=10, color=CLR["dark_gray"])
        c.alignment = left()
        ws.row_dimensions[r].height = 22

    return ws


# ══════════════════════════════════════════════════════════════════════════════
# SHEET 6 — DATASET INVENTORY
# ══════════════════════════════════════════════════════════════════════════════
def build_dataset_inventory(wb):
    ws = wb.create_sheet("💾 Dataset Inventory")
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "A4"

    ws.merge_cells("A1:M2")
    c = ws.cell(row=1, column=1, value="💾 DATASET INVENTORY — Key Datasets in Your Research Domain")
    c.fill = fill(CLR["navy"])
    c.font = Font(bold=True, size=15, color=CLR["white"], name="Calibri")
    c.alignment = center()
    ws.row_dimensions[1].height = 35

    headers = [
        "Dataset Name", "Domain", "Signal / Data Type",
        "Size / #Samples", "# Subjects/Channels",
        "Open Access?", "License", "URL / DOI",
        "Used In Papers", "Preprocessing Needed",
        "Limitation / Gap", "Your Notes", "Priority"
    ]
    widths = [25, 20, 18, 18, 18, 12, 15, 35, 18, 25, 35, 25, 12]

    for i, (h, w) in enumerate(zip(headers, widths), 1):
        style_header(ws, 3, i, h)
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.row_dimensions[3].height = 38

    datasets = [
        ["BCI Competition IV (2a/2b)", "EEG/BCI", "EEG (22/3-ch)",
         "9 subjects × sessions", "9 subjects, 22 ch",
         "Yes", "Public", "https://www.bbci.de/competition/iv/",
         "P001", "Bandpass, ICA, CSP",
         "Small N; lab conditions; limited session count; single paradigm",
         "Go-to EEG benchmark — well studied", "High"]
    ]

    alt_bgs = [CLR["white"], CLR["light_gray"]]
    for r_idx, row in enumerate(datasets):
        r = r_idx + 4
        bg = alt_bgs[r_idx % 2]
        for c_idx, val in enumerate(row, 1):
            style_cell(ws, r, c_idx, val, bg=bg)
        ws.row_dimensions[r].height = 50

    for r_idx in range(10):
        r = 5 + r_idx
        bg = alt_bgs[r_idx % 2]
        for c in range(1, 14):
            style_cell(ws, r, c, "", bg=bg)
        ws.row_dimensions[r].height = 40

    dv_prio = DataValidation(type="list", formula1='"High,Medium,Low"')
    dv_prio.sqref = "M4:M100"
    ws.add_data_validation(dv_prio)

    return ws


# ══════════════════════════════════════════════════════════════════════════════
# SHEET 7 — READING PROGRESS TRACKER
# ══════════════════════════════════════════════════════════════════════════════
def build_reading_tracker(wb):
    ws = wb.create_sheet("📋 Reading Tracker")
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "A4"

    ws.merge_cells("A1:L2")
    c = ws.cell(row=1, column=1, value="📋 READING PROGRESS TRACKER")
    c.fill = fill(CLR["navy"])
    c.font = Font(bold=True, size=15, color=CLR["white"], name="Calibri")
    c.alignment = center()
    ws.row_dimensions[1].height = 35

    headers = [
        "Paper ID", "Title (Short)", "Authors", "Year",
        "Reading Status", "Date Added", "Date Read",
        "Abstract\nRead?", "Full Text\nRead?", "Notes\nAdded?",
        "Matrix\nFilled?", "Priority"
    ]
    widths = [10, 40, 30, 8, 18, 14, 14, 12, 14, 12, 12, 12]

    for i, (h, w) in enumerate(zip(headers, widths), 1):
        style_header(ws, 3, i, h)
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.row_dimensions[3].height = 40

    statuses = ["P001", "P002"]
    titles = [
        "Hybrid CNN-LSTM for EEG Motor Imagery",
        "Fault Detection in Power Grids via LSTM"
    ]
    authors_list = [
        "Smith et al.", "Kumar & Lee"
    ]
    read_statuses = ["Fully Read", "Fully Read"]
    dates_added = ["2026-01-10", "2026-01-15"]
    dates_read =  ["2026-01-12", "2026-01-18"]
    yn_data = [
        ["Y", "Y", "Y", "Y"],
        ["Y", "Y", "Y", "Y"]
    ]
    priorities = ["High", "High"]

    alt_bgs = [CLR["white"], CLR["light_gray"]]
    for r_idx in range(2):
        r = r_idx + 4
        bg = alt_bgs[r_idx % 2]
        row_vals = [statuses[r_idx], titles[r_idx], authors_list[r_idx], 2023 - (r_idx % 2),
                    read_statuses[r_idx], dates_added[r_idx], dates_read[r_idx]] + yn_data[r_idx] + [priorities[r_idx]]
        for c_idx, val in enumerate(row_vals, 1):
            style_cell(ws, r, c_idx, val, bg=bg)
        ws.row_dimensions[r].height = 30

    for r_idx in range(20):
        r = 6 + r_idx
        bg = alt_bgs[r_idx % 2]
        style_cell(ws, r, 1, f"P{r_idx+3:03d}", bg=bg)
        for c in range(2, 13):
            style_cell(ws, r, c, "", bg=bg)
        ws.row_dimensions[r].height = 28

    dv_status = DataValidation(type="list",
        formula1='"Not Started,Queued,Skimmed,Abstract Only,Notes Only,Partially Read,Fully Read"')
    dv_status.sqref = "E4:E100"
    ws.add_data_validation(dv_status)

    dv_yn = DataValidation(type="list", formula1='"Y,N"')
    for col in ["H", "I", "J", "K"]:
        dv = DataValidation(type="list", formula1='"Y,N"')
        dv.sqref = f"{col}4:{col}100"
        ws.add_data_validation(dv)

    dv_p = DataValidation(type="list", formula1='"High,Medium,Low"')
    dv_p.sqref = "L4:L100"
    ws.add_data_validation(dv_p)

    return ws


# ══════════════════════════════════════════════════════════════════════════════
# SHEET 8 — INSTRUCTIONS & GUIDE
# ══════════════════════════════════════════════════════════════════════════════
def build_instructions(wb):
    ws = wb.create_sheet("📖 How to Use", 0)
    ws.sheet_view.showGridLines = False

    ws.column_dimensions["A"].width = 5
    ws.column_dimensions["B"].width = 22
    ws.column_dimensions["C"].width = 70

    ws.merge_cells("A1:C2")
    c = ws.cell(row=1, column=1,
                value="📖  COMPREHENSIVE LITERATURE REVIEW MATRIX — USER GUIDE")
    c.fill = fill(CLR["navy"])
    c.font = Font(bold=True, size=17, color=CLR["white"], name="Calibri")
    c.alignment = center()
    ws.row_dimensions[1].height = 40
    ws.row_dimensions[2].height = 5

    ws.merge_cells("A3:C3")
    c = ws.cell(row=3, column=1,
                value="Designed for: Engineering · Electronics · EE · ML · Neuroscience Data Analysis")
    c.fill = fill(CLR["teal"])
    c.font = font(bold=True, size=12, color=CLR["white"])
    c.alignment = center()
    ws.row_dimensions[3].height = 28

    sections = [
        (5,  CLR["gold"],    "🗂️  SHEET OVERVIEW", ""),
        (6,  CLR["ice"],     "📚 Literature Matrix",
             "Main table — one row per paper. Covers reference info, theory, methods, context, ML techniques, findings, and identified gaps (60 cols)."),
        (7,  CLR["ice"],     "🔍 Gap Tracker",
             "Synthesize identified gaps with evidence, novelty & feasibility scores, proposed hypotheses, and research status tracking."),
        (8,  CLR["ice"],     "🗺️ Domain Coverage Map",
             "Heatmap matrix — domains (rows) vs techniques (columns). Zeros highlight unexplored intersections = research gaps."),
        (9,  CLR["ice"],     "⚙️ Methodology Comparison",
             "Side-by-side comparison of ML/AI methods across papers: accuracy, latency, energy, hardware, reproducibility."),
        (10, CLR["ice"],     "📈 Temporal Trends",
             "Track publication volume by topic and year to identify fast-growing areas and underexplored niches."),
        (11, CLR["ice"],     "💾 Dataset Inventory",
             "Catalog all datasets used across your literature; note limitations and open-access status — dataset gaps = research gaps."),
        (12, CLR["ice"],     "📋 Reading Tracker",
             "Manage your reading workflow — track reading status, dates, and matrix completion for each paper."),

        (14, CLR["gold"],    "✅  HOW TO START", ""),
        (15, CLR["mint"],    "Step 1 — Import Papers",
             "In '📚 Literature Matrix', add one paper per row. Fill Paper ID (P001, P002…), full citation, and all fields you can."),
        (16, CLR["mint"],    "Step 2 — Identify Theory & Method",
             "Log their research questions, definitions, epistemological stance, level of analysis, and sample details."),
        (17, CLR["mint"],    "Step 3 — Identify Gaps",
             "For each paper, fill the 'Identified Gaps', 'Neglected Variables', 'Excluded Demographics', and 'Methodological Weaknesses' columns."),
        (18, CLR["mint"],    "Step 4 — Synthesize in Gap Tracker",
             "Distill recurring gaps into '🔍 Gap Tracker'. Score novelty, feasibility, and impact to calculate priority."),
        (19, CLR["mint"],    "Step 5 — Update Domain Map",
             "Update counts in '🗺️ Domain Coverage Map' — zeros indicate unexplored intersections between domains and methods."),
        (20, CLR["mint"],    "Step 6 — Trend Analysis",
             "Fill '📈 Temporal Trends' with paper counts by year per topic to see which research areas are saturating vs. emerging."),
        (21, CLR["mint"],    "Step 7 — Mine Your Gaps",
             "Review top-priority gaps from Gap Tracker. Form your research hypothesis in column 'Your Hypothesis / Proposed Solution'."),

        (23, CLR["gold"],    "🎯  GAP FINDING HEURISTICS", ""),
        (24, CLR["sky"],     "Zero-count cells in Domain Map",
             "Any cell = 0 means NO paper combines that domain + technique. That's a potential research gap."),
        (25, CLR["sky"],     "Recurring 'Future Work' themes",
             "If ≥3 papers suggest the same future direction, that's a validated, high-priority gap."),
        (26, CLR["sky"],     "Methodological Weaknesses",
             "Look for unproven 'Key Assumptions' or issues with 'Ecological Validity' (e.g. lab vs real-world)."),
        (27, CLR["sky"],     "Excluded Demographics/Contexts",
             "If all studies on an algorithm are trained on Western demographics, apply it elsewhere."),
        (28, CLR["sky"],     "Datasets with known limitations",
             "Small N, single-site, single-paradigm datasets signal need for broader validation studies."),
        (29, CLR["sky"],     "Real-world deployment gap",
             "Many papers are lab/offline only — real-time, embedded, or federated deployment is almost always a gap."),

        (31, CLR["gold"],    "🔑  PRIORITY SCORE FORMULA", ""),
        (32, CLR["lavender"],"Priority = Novelty × Feasibility × Impact",
             "In '🔍 Gap Tracker': rate each 1–5 → Priority = H×I×J (max 125). Higher = stronger research gap target."),
        (33, CLR["lavender"],"Novelty (1-5)",
             "1 = well-studied; 5 = completely unexplored combination of ideas"),
        (34, CLR["lavender"],"Feasibility (1-5)",
             "1 = requires breakthrough; 5 = achievable with available tools/datasets in 1-2 years"),
        (35, CLR["lavender"],"Impact (1-5)",
             "1 = incremental; 5 = could shift the field, enable new applications, or address clinical need"),

        (37, CLR["gold"],    "📌  TIPS", ""),
        (38, CLR["peach"],   "Use Paper IDs consistently",
             "Always reference papers by ID (P001, P002…) across all sheets for cross-referencing."),
        (39, CLR["peach"],   "Use filters",
             "Add Excel AutoFilter (Data → Filter) to the main matrix to slice by domain, year, or method."),
        (40, CLR["peach"],   "Colour conventions",
             "Gold headers = Gap Analysis columns | Teal/Slate = Methods & Details | Navy = Core Info"),
        (41, CLR["peach"],   "Export summaries",
             "Copy the Gap Tracker rows to a Word/LaTeX table to draft your Related Work / Research Gap section."),
        (42, CLR["peach"],   "Created",
             f"Template generated: {datetime.date.today().strftime('%B %d, %Y')} | Version 2.0"),
    ]

    for row_num, bg, label, desc in sections:
        c_label = ws.cell(row=row_num, column=2, value=label)
        c_label.fill = fill(bg)
        c_label.font = font(bold=True, size=10, color=CLR["dark_gray"])
        c_label.alignment = left()
        c_label.border = border()

        c_desc = ws.cell(row=row_num, column=3, value=desc)
        c_desc.fill = fill(CLR["white"] if bg == CLR["gold"] else bg)
        c_desc.font = font(size=10, color=CLR["dark_gray"])
        c_desc.alignment = left()
        c_desc.border = border()

        ws.row_dimensions[row_num].height = 32

    return ws


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    build_instructions(wb)
    build_main_matrix(wb)
    build_gap_tracker(wb)
    build_domain_map(wb)
    build_methodology_comparison(wb)
    build_temporal_trends(wb)
    build_dataset_inventory(wb)
    build_reading_tracker(wb)

    # Set active sheet to instructions
    wb.active = wb["📖 How to Use"]

    out_path = r"d:\Research\Literature_Review_Matrix.xlsx"
    wb.save(out_path)
    print("Excel matrix generated.")

if __name__ == "__main__":
    main()
