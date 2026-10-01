"""Draws docs/architecture.pdf and docs/architecture.png."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEAL, INK, MUTED, PURPLE, AMBER, RED = "#0F766E", "#1C1917", "#57534E", "#6D28D9", "#B45309", "#B91C1C"

fig, ax = plt.subplots(figsize=(16, 9.6))
ax.set_xlim(0, 160)
ax.set_ylim(0, 96)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, title, lines, edge, fill="#FFFFFF", tsize=11.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                                linewidth=1.8, edgecolor=edge, facecolor=fill))
    ax.text(x + 1.6, y + h - 2.4, title, fontsize=tsize, fontweight="bold", color=edge, va="top")
    for k, ln in enumerate(lines):
        ax.text(x + 1.6, y + h - 6.6 - k * 3.1, ln, fontsize=8.9, color=INK, va="top")


def arrow(x0, y0, x1, y1, label=None, color=MUTED, lx=0, ly=1.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.5, color=color, shrinkA=2, shrinkB=2))
    if label:
        ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label, fontsize=8.2, color=color, ha="center")


ax.text(2, 93.5, "Antaral — architecture", fontsize=19, fontweight="bold", color=INK, va="top")
ax.text(2, 88.6, "A digital twin for the gap between dialysis sessions: patient twin → risk layer → unit twin → doctor's dashboard",
        fontsize=11, color=MUTED, va="top")

# --- data sources column
ax.text(2, 84, "1  DATA STREAMS", fontsize=10, fontweight="bold", color=MUTED)
ax.text(2, 81.2, "synthetic, sandbox-compliant", fontsize=8.5, color=MUTED)
box(2, 58, 34, 21, "Static EHR", ["Age, dry weight, diagnosis",
                                  "Drugs: ACE-i/ARB, beta-blocker", "Residual urine, bicarbonate",
                                  "Dialysis pattern (2x or 3x / week)"], MUTED)
box(2, 31, 34, 25, "Wearables & home devices", ["Smartwatch single-lead ECG (2x/day)",
                                                  "Smart-scale morning weight",
                                                  "Steps, resting HR, HRV, sleep, BP",
                                                  "Food diary: coconut water, banana..."], PURPLE)
box(2, 15, 34, 14, "Dialysis unit records", ["Sessions: start, length, pre-weight",
                                               "Potassium lab (~monthly)"], MUTED)
box(2, 1, 34, 12, "Simulator (ground truth)", ["Hidden physiology, illness, unlogged",
                                                "food. Used only for evaluation."], "#A8A29E", "#F5F5F4", 10.5)

# --- patient twin
ax.text(48, 84, "2  PATIENT TWIN", fontsize=10, fontweight="bold", color=MUTED)
ax.text(48, 81.2, "one per patient, updated hourly", fontsize=8.5, color=MUTED)
box(48, 56, 46, 23, "Mechanistic potassium model + Kalman filter",
    ["dK/dt = (absorbed intake - renal + release) / V", "Personalised from the EHR",
     "Dialysis: decay to 2.0 bath, then rebound", "State: potassium K and personal drift b"], TEAL, tsize=10.8)
box(48, 31, 32, 21, "ECG potassium sensor", ["Median beat: T height, width,", "slope, PR, QRS, QT vs own",
                                              "post-dialysis beat -> LightGBM", "K estimate +/- sd (MAE 0.5)"], PURPLE)
box(48, 4, 46, 23, "What-if forecaster (Monte Carlo)", ["Sample (K, b), roll forward with noise",
                                                         "Scenarios: session time, food, binder,",
                                                         "diet call -> P(K >= 6 before session)",
                                                         "and danger-hours over next two gaps"], TEAL)

# --- decisions
ax.text(108, 84, "3  DECISIONS", fontsize=10, fontweight="bold", color=MUTED)
ax.text(108, 81.2, "every change needs a clinician's approval", fontsize=8.5, color=MUTED)
box(108, 58, 50, 21, "Risk layer: LightGBM + calibration", ["Twin forecast + EHR + wearables + labs",
                                                             "-> calibrated risk of K >= 6.0 before session",
                                                             "SHAP: why this patient, why tonight",
                                                             "Test AUROC 0.966 (last lab: 0.899)"], AMBER)
box(108, 30, 50, 24, "Unit twin: nightly scheduler (MILP)", ["Tomorrow's chairs at 9 PM: maximise",
                                                              "danger-hours avoided, same sessions/week,",
                                                              "chairs, min/max gaps; stability penalty",
                                                              "Riskiest patients -> earliest shift",
                                                              "Sim: ~15% fewer arrive with K >= 6"], TEAL)
box(108, 4, 50, 22, "Doctor's dashboard (Streamlit)", ["Unit tonight: suggestions + chair board",
                                                        "Patient twin: potassium trajectory, forecast",
                                                        "fan, ECG vs baseline, food log, reasons",
                                                        "What-if panel for the clinician"], RED)

arrow(36, 69, 48, 69, "EHR priors", ly=1.3)
arrow(36, 48, 48, 61, "food, weight", PURPLE, lx=-1, ly=2.5)
arrow(36, 40, 48, 40, "ECG beats", PURPLE, ly=1.3)
arrow(36, 22, 48, 58, "sessions,\nlabs", MUTED, lx=-3.2, ly=-15)
arrow(64, 52, 64, 56, None, PURPLE)
ax.text(65.5, 53.2, "K +/- sd", fontsize=8.2, color=PURPLE)
arrow(88, 56, 88, 27, None, TEAL)
ax.text(89.2, 42, "belief", fontsize=8.5, color=TEAL, rotation=90, va="center")
arrow(94, 20, 108, 62, "forecast", TEAL, lx=-3, ly=6)
arrow(94, 12, 108, 38, "what-if", TEAL, lx=2, ly=-5)
arrow(133, 58, 133, 54, None, AMBER)
arrow(133, 30, 133, 26, None, TEAL)

fig.savefig(ROOT / "docs" / "architecture.pdf", bbox_inches="tight")
fig.savefig(ROOT / "docs" / "architecture.png", dpi=150, bbox_inches="tight")
print("saved")
