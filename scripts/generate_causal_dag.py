"""
generate_causal_dag.py
======================
Generates a publication-quality Causal DAG figure for the simulation
data-generating process (SuppNote 3).

Outputs:
  SuppFig_CausalDAG.png  (300 dpi, ~12x9 inches)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe

# ─── output path ─────────────────────────────────────────────────────────────
OUTDIR = os.path.dirname(os.path.abspath(__file__))
OUTFILE = os.path.join(OUTDIR, "SuppFig_CausalDAG.png")

# ─── colour palette ──────────────────────────────────────────────────────────
COL = {
    "preinjury":   "#4E79A7",   # blue  — pre-injury factors
    "severity":    "#E15759",   # red   — injury severity
    "acutecare":   "#F28E2B",   # orange — acute care / ICU
    "latent":      "#76B7B2",   # teal  — latent/trajectory
    "qsp":         "#59A14F",   # green — QSP-ODE features
    "outcome":     "#B07AA1",   # purple — outcomes
    "mortality":   "#9C755F",   # brown  — mortality (special)
    "edge_pos":    "#1F77B4",   # positive effect
    "edge_neg":    "#D62728",   # negative effect
    "edge_der":    "#7F7F7F",   # derived/transformational
    "edge_str":    "#E377C2",   # strong psychiatric path
}

# ─── node definitions: (label, x, y, colour, fontsize) ──────────────────────
#  x: 0 (leftmost) → 10 (rightmost)
#  y: 0 (bottom) → 10 (top)
nodes = {
    # ── L0: Pre-injury factors (collapsed supernode + key individuals) ─────
    "preinjury":   ("Pre-injury\nfactors\n(age, sex, education,\nHTN, DM, CVD)",
                     0.5, 7.5, COL["preinjury"], 7.5),
    "psych_hx":    ("Prior\nPsych Hx",        0.5, 5.0, COL["preinjury"], 8),
    "alcohol":     ("Alcohol\nMisuse",         0.5, 3.0, COL["preinjury"], 8),
    "marital":     ("Marital\nStatus",         0.5, 1.5, COL["preinjury"], 8),

    # ── L1: Injury severity ───────────────────────────────────────────────
    "diagnosis":   ("Diagnosis\n(TBI/SAH/\nStroke/ICH)", 2.5, 9.0, COL["severity"], 8),
    "gcs":         ("GCS\nAdmission",          2.5, 6.5, COL["severity"], 8.5),
    "apache":      ("APACHE II",               2.5, 5.0, COL["severity"], 8.5),

    # ── L2: Acute care ───────────────────────────────────────────────────
    "icu_los":     ("ICU LOS",                 4.3, 9.5, COL["acutecare"], 8),
    "mechvent":    ("Mech.\nVentilation",      4.3, 8.2, COL["acutecare"], 8),
    "icp":         ("ICP\nMonitoring",         4.3, 7.0, COL["acutecare"], 8),
    "early_mob":   ("Early\nMobilisation",     4.3, 5.5, COL["acutecare"], 8),
    "delirium":    ("Delirium",                4.3, 4.1, COL["acutecare"], 8.5),
    "icu_anx":     ("ICU\nAnxiety",            4.3, 2.8, COL["acutecare"], 8),
    "surgery":     ("Surgery",                 4.3, 1.5, COL["acutecare"], 8),

    # ── L3a: Latent / Trajectory ──────────────────────────────────────────
    "latent":      ("Latent\nSeverity",        6.0, 6.5, COL["latent"], 8),
    "trajectory":  ("Trajectory\nClass",       6.0, 5.0, COL["latent"], 8.5),

    # ── L3b: QSP features (collapsed) ────────────────────────────────────
    "qsp":         ("QSP-ODE\nMechanistic\nFeatures\n(21 variables)",
                     6.0, 2.2, COL["qsp"], 7.5),

    # ── L4: Outcomes (grouped) ────────────────────────────────────────────
    "gose":        ("GOSE / FIM /\nBarthel / mRS\n(Functional)",
                     8.5, 9.2, COL["outcome"], 7.5),
    "cognition":   ("Cog Composite\nMoCA\n(Cognitive)",
                     8.5, 7.3, COL["outcome"], 7.5),
    "hads_anx":    ("HADS-Anxiety\nGAD-7",     8.5, 5.8, COL["outcome"], 8),
    "hads_dep":    ("HADS-Dep\nPHQ-9",         8.5, 4.5, COL["outcome"], 8),
    "ptsd":        ("PCL-5\n(PTSD)",            8.5, 3.3, COL["outcome"], 8),
    "qol":         ("SF-36 / QOLIBRI\nMPAI-4\n(QoL)",
                     8.5, 2.0, COL["outcome"], 7.5),
    "rtw":         ("Return to\nWork /\nSocial Partic.",
                     8.5, 0.7, COL["outcome"], 7.5),
    "mortality":   ("Mortality\n(12m)",         6.0, 0.5, COL["mortality"], 8),
}

# ─── edge definitions: (from_key, to_key, colour, lw, style, label) ─────────
# style: 'solid', 'dashed'
# label: short annotation for key paths ('' for unlabelled)
edges = [
    # L0 → L1
    ("preinjury",  "gcs",       COL["edge_pos"], 1.2, "solid", ""),
    ("preinjury",  "apache",    COL["edge_pos"], 1.2, "solid", ""),
    ("diagnosis",  "gcs",       COL["edge_der"], 1.2, "solid", ""),
    ("gcs",        "apache",    COL["edge_neg"], 1.5, "solid", "β=−1.2"),

    # L0 → L2
    ("psych_hx",   "delirium",  COL["edge_pos"], 1.5, "solid", ""),
    ("psych_hx",   "icu_anx",   COL["edge_pos"], 2.0, "solid", "β=+2.0"),
    ("alcohol",    "icu_anx",   COL["edge_pos"], 1.2, "solid", ""),

    # L1 → L2
    ("gcs",        "icu_los",   COL["edge_neg"], 1.2, "solid", ""),
    ("gcs",        "mechvent",  COL["edge_neg"], 1.2, "solid", ""),
    ("gcs",        "icp",       COL["edge_neg"], 1.2, "solid", ""),
    ("gcs",        "delirium",  COL["edge_neg"], 1.5, "solid", "β=+0.15\non (15−GCS)"),
    ("apache",     "delirium",  COL["edge_pos"], 1.0, "dashed", ""),
    ("diagnosis",  "surgery",   COL["edge_der"], 1.2, "solid", ""),
    ("mechvent",   "delirium",  COL["edge_pos"], 1.5, "solid", "β=+0.5"),
    ("early_mob",  "delirium",  COL["edge_neg"], 1.5, "solid", "β=−0.4"),
    ("delirium",   "icu_anx",   COL["edge_pos"], 1.2, "solid", ""),
    ("early_mob",  "icu_anx",   COL["edge_neg"], 1.2, "solid", ""),
    ("preinjury",  "early_mob", COL["edge_der"], 0.8, "dashed", ""),

    # L1/L2 → L3a (latent/trajectory)
    ("gcs",        "latent",    COL["edge_neg"], 1.5, "solid", "β=−0.15"),
    ("apache",     "latent",    COL["edge_pos"], 1.2, "solid", "β=+0.08"),
    ("preinjury",  "latent",    COL["edge_pos"], 1.0, "dashed", "age β=+0.02"),
    ("psych_hx",   "latent",    COL["edge_pos"], 2.0, "solid", "β=+0.50"),
    ("delirium",   "latent",    COL["edge_pos"], 1.8, "solid", "β=+0.40"),
    ("early_mob",  "latent",    COL["edge_neg"], 1.5, "solid", "β=−0.30"),
    ("latent",     "trajectory",COL["edge_der"], 2.0, "solid", "logistic\nshift"),

    # L1/L2 → L3b (QSP)
    ("gcs",        "qsp",       COL["edge_neg"], 1.5, "solid", "GCS → AR,\nODE params"),
    ("apache",     "qsp",       COL["edge_pos"], 1.2, "solid", ""),
    ("preinjury",  "qsp",       COL["edge_pos"], 1.0, "dashed", "age → NI\nrate consts"),
    ("psych_hx",   "qsp",       COL["edge_pos"], 1.0, "dashed", "microglial\npriming"),
    ("alcohol",    "qsp",       COL["edge_pos"], 1.0, "dashed", ""),
    ("delirium",   "qsp",       COL["edge_pos"], 1.0, "dashed", "DAMP\nS₀ +0.3×del"),

    # L3a → L4 (trajectory → all outcomes) — single thick edge to each
    ("trajectory", "gose",      COL["edge_pos"], 2.5, "solid", "traj_mod\n×1.5"),
    ("trajectory", "cognition", COL["edge_pos"], 2.0, "solid", ""),
    ("trajectory", "hads_anx",  COL["edge_neg"], 2.0, "solid", "β=−2.0×mod"),
    ("trajectory", "hads_dep",  COL["edge_neg"], 2.0, "solid", ""),
    ("trajectory", "ptsd",      COL["edge_neg"], 1.5, "solid", ""),
    ("trajectory", "qol",       COL["edge_pos"], 2.0, "solid", ""),
    ("trajectory", "rtw",       COL["edge_pos"], 1.5, "solid", ""),

    # L0/L1/L2 → L4 (direct paths)
    ("gcs",        "gose",      COL["edge_pos"], 1.5, "solid", "β=+0.20"),
    ("preinjury",  "gose",      COL["edge_neg"], 1.0, "dashed", "age β=−0.02"),
    ("early_mob",  "gose",      COL["edge_pos"], 1.2, "solid", "β=+0.20"),
    ("delirium",   "gose",      COL["edge_neg"], 1.2, "solid", "β=−0.30"),
    ("gcs",        "cognition", COL["edge_pos"], 1.5, "solid", ""),
    ("delirium",   "cognition", COL["edge_neg"], 1.5, "solid", "β=−0.5"),

    # Key psychiatric direct paths
    ("icu_anx",    "hads_anx",  COL["edge_str"], 2.5, "solid", "β=+0.60\n★ strongest"),
    ("psych_hx",   "hads_anx",  COL["edge_str"], 2.2, "solid", "β=+2.5"),
    ("delirium",   "hads_anx",  COL["edge_pos"], 1.5, "solid", "β=+1.5"),
    ("alcohol",    "hads_anx",  COL["edge_pos"], 1.2, "solid", "β=+0.5"),
    ("psych_hx",   "hads_dep",  COL["edge_str"], 2.2, "solid", "β=+2.5"),
    ("delirium",   "hads_dep",  COL["edge_pos"], 1.2, "solid", "β=+1.0"),
    ("marital",    "hads_dep",  COL["edge_neg"], 1.2, "solid", "β=−0.4"),
    ("alcohol",    "hads_dep",  COL["edge_pos"], 1.2, "solid", "β=+0.4"),
    ("psych_hx",   "ptsd",      COL["edge_str"], 2.5, "solid", "β=+3.0\nstrongest PTSD"),
    ("alcohol",    "ptsd",      COL["edge_pos"], 1.8, "solid", "β=+2.0"),
    ("hads_anx",   "ptsd",      COL["edge_pos"], 1.5, "solid", "β=+1.5"),
    ("hads_dep",   "ptsd",      COL["edge_pos"], 1.2, "solid", "β=+1.0"),
    ("hads_dep",   "qol",       COL["edge_neg"], 1.2, "solid", ""),
    ("hads_anx",   "qol",       COL["edge_neg"], 1.2, "solid", ""),
    ("cognition",  "qol",       COL["edge_pos"], 1.5, "solid", "β=+5.0"),
    ("gose",       "qol",       COL["edge_pos"], 1.2, "solid", ""),
    ("gose",       "rtw",       COL["edge_pos"], 1.5, "solid", "β=+0.30"),
    ("cognition",  "rtw",       COL["edge_pos"], 1.2, "solid", "β=+0.30"),
    ("hads_dep",   "rtw",       COL["edge_neg"], 1.2, "solid", "β=−0.10"),
    ("marital",    "rtw",       COL["edge_pos"], 0.8, "dashed", "social\nsupport"),

    # QSP → (no outcome path — key structural absence)
    # shown as a 'dead end' — no edges drawn from qsp to L4

    # Mortality
    ("gcs",        "mortality", COL["edge_neg"], 1.5, "solid", "β=+0.30\non (15−GCS)"),
    ("apache",     "mortality", COL["edge_pos"], 1.2, "solid", "β=+0.02"),
    ("preinjury",  "mortality", COL["edge_pos"], 1.0, "dashed", "age β=+0.03"),
    ("diagnosis",  "mortality", COL["edge_pos"], 1.2, "solid", "ICH β=+0.5"),
    ("early_mob",  "mortality", COL["edge_neg"], 1.0, "dashed", "β=−0.30"),
]


def draw_node(ax, x, y, label, color, fontsize=8, width=1.6, height=0.7):
    """Draw a rounded-rectangle node."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.08",
        linewidth=1.2,
        edgecolor="white",
        facecolor=color,
        alpha=0.92,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x, y, label,
        ha="center", va="center",
        fontsize=fontsize, fontweight="bold",
        color="white",
        zorder=4,
        multialignment="center",
    )


def draw_edge(ax, x0, y0, x1, y1, color, lw=1.2, style="solid", label="",
              connectionstyle="arc3,rad=0.0"):
    """Draw an arrow between two nodes."""
    arrowprops = dict(
        arrowstyle="-|>",
        color=color,
        lw=lw,
        linestyle=style,
        mutation_scale=10,
        connectionstyle=connectionstyle,
    )
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=arrowprops,
        zorder=2,
    )
    if label:
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2
        ax.text(mx, my, label,
                fontsize=5.5, color=color, ha="center", va="center",
                fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))


def get_node_centre(key):
    label, x, y, color, fs = nodes[key]
    return x, y


# ─── Build node size lookup for edge connection offsets ──────────────────────
NODE_W = {"preinjury": 1.9, "qsp": 1.7, "gose": 1.85, "cognition": 1.7,
          "qol": 1.85, "rtw": 1.7}
NODE_H = {"preinjury": 1.0, "qsp": 1.0}


def clamp_to_node_edge(x0, y0, x1, y1, key0, key1, nodes):
    """Approximate clamping to box edge (simplified — just shorten arrow)."""
    dx = x1 - x0
    dy = y1 - y0
    dist = np.sqrt(dx**2 + dy**2) + 1e-9
    # offset from each node centre by half-width + small margin
    w0 = 0.85
    w1 = 0.85
    ux, uy = dx / dist, dy / dist
    xs = x0 + ux * w0
    ys = y0 + uy * w0
    xe = x1 - ux * w1
    ye = y1 - uy * w1
    return xs, ys, xe, ye


# ─── MAIN FIGURE ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.3, 10.8)
ax.set_aspect("equal")
ax.axis("off")

fig.patch.set_facecolor("#F8F8F8")
ax.set_facecolor("#F8F8F8")

# Title
ax.text(5.0, 10.5,
        "Causal DAG: Simulation Data-Generating Process",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#2c2c2c")

# Layer labels
for lx, ltxt in [(0.5, "L0\nPre-injury"), (2.5, "L1\nSeverity"),
                  (4.3, "L2\nAcute care"), (6.0, "L3\nLatent/QSP"),
                  (8.5, "L4\nOutcomes")]:
    ax.text(lx, 10.2, ltxt, ha="center", va="center",
            fontsize=8, color="#555555", style="italic")

# Vertical dividers
for lx in [1.55, 3.45, 5.2, 7.3]:
    ax.axvline(lx, color="#cccccc", lw=0.8, ls="--", zorder=1)

# Draw edges first (under nodes)
# To keep it clean, use curvature for overlapping edges
curve_map = {
    # (from, to): arc rad
    ("psych_hx", "hads_anx"): "arc3,rad=-0.15",
    ("psych_hx", "hads_dep"): "arc3,rad=-0.20",
    ("psych_hx", "ptsd"):     "arc3,rad=-0.25",
    ("psych_hx", "latent"):   "arc3,rad=0.15",
    ("psych_hx", "qsp"):      "arc3,rad=0.25",
    ("alcohol",  "hads_anx"): "arc3,rad=-0.12",
    ("alcohol",  "hads_dep"): "arc3,rad=-0.18",
    ("alcohol",  "ptsd"):     "arc3,rad=-0.22",
    ("alcohol",  "qsp"):      "arc3,rad=0.20",
    ("delirium", "hads_dep"): "arc3,rad=0.10",
    ("delirium", "latent"):   "arc3,rad=0.12",
    ("delirium", "cognition"):"arc3,rad=0.10",
    ("hads_anx", "ptsd"):     "arc3,rad=0.08",
    ("hads_dep", "ptsd"):     "arc3,rad=-0.08",
    ("hads_dep", "qol"):      "arc3,rad=0.08",
    ("hads_anx", "qol"):      "arc3,rad=-0.08",
    ("gose",     "qol"):      "arc3,rad=0.10",
    ("gose",     "rtw"):      "arc3,rad=0.10",
    ("preinjury","latent"):   "arc3,rad=0.18",
    ("preinjury","qsp"):      "arc3,rad=0.30",
    ("preinjury","gose"):     "arc3,rad=-0.10",
    ("preinjury","mortality"):"arc3,rad=0.35",
    ("early_mob","mortality"):"arc3,rad=0.15",
    ("diagnosis","mortality"):"arc3,rad=-0.15",
    ("marital",  "hads_dep"): "arc3,rad=-0.12",
    ("marital",  "rtw"):      "arc3,rad=-0.15",
    ("cognition","qol"):      "arc3,rad=0.08",
    ("cognition","rtw"):      "arc3,rad=0.08",
    ("apache",   "delirium"): "arc3,rad=-0.10",
    ("apache",   "mortality"):"arc3,rad=-0.12",
}

for (src, dst, color, lw, style, label) in edges:
    x0, y0 = get_node_centre(src)
    x1, y1 = get_node_centre(dst)
    xs, ys, xe, ye = clamp_to_node_edge(x0, y0, x1, y1, src, dst, nodes)
    cs = curve_map.get((src, dst), "arc3,rad=0.0")
    draw_edge(ax, xs, ys, xe, ye, color=color, lw=lw, style=style,
              label=label, connectionstyle=cs)

# Draw nodes on top
node_widths  = {"preinjury": 1.9, "qsp": 1.75, "gose": 1.85,
                "cognition": 1.75, "qol": 1.85, "rtw": 1.75, "diagnosis": 1.6}
node_heights = {"preinjury": 1.05, "qsp": 1.05}

for key, (label, x, y, color, fs) in nodes.items():
    w = node_widths.get(key, 1.55)
    h = node_heights.get(key, 0.72)
    draw_node(ax, x, y, label, color, fontsize=fs, width=w, height=h)

# ── "No direct path" annotation on QSP node ──────────────────────────────────
ax.annotate(
    "★ No direct path\nto outcomes\n(collinear with GCS)",
    xy=(6.0, 2.2), xytext=(6.8, 0.1),
    arrowprops=dict(arrowstyle="->", color="#59A14F", lw=1.0),
    fontsize=6.5, color="#59A14F", ha="center", va="bottom",
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#59A14F", alpha=0.85),
    zorder=6,
)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=COL["preinjury"], label="Pre-injury factors (L0)"),
    mpatches.Patch(color=COL["severity"],  label="Injury severity (L1)"),
    mpatches.Patch(color=COL["acutecare"], label="Acute care / ICU (L2)"),
    mpatches.Patch(color=COL["latent"],    label="Latent / trajectory (L3a)"),
    mpatches.Patch(color=COL["qsp"],       label="QSP-ODE features (L3b)"),
    mpatches.Patch(color=COL["outcome"],   label="12-month outcomes (L4)"),
    mpatches.Patch(color=COL["mortality"], label="Mortality"),
    mpatches.Patch(color=COL["edge_pos"],  label="Positive causal effect"),
    mpatches.Patch(color=COL["edge_neg"],  label="Negative causal effect"),
    mpatches.Patch(color=COL["edge_str"],  label="Strong psychiatric path"),
    mpatches.Patch(color=COL["edge_der"],  label="Derived / conditional"),
]
ax.legend(
    handles=legend_items,
    loc="lower left",
    bbox_to_anchor=(-0.02, -0.03),
    fontsize=6.5,
    framealpha=0.92,
    ncol=2,
    title="Node category / Edge type",
    title_fontsize=7,
)

# Caption
ax.text(5.0, -0.25,
        "Dashed edges = indirect / weak paths. Edge width ∝ |β|. "
        "QSP features have no direct outcome edges (collinear with GCS/APACHE II).",
        ha="center", va="center", fontsize=6.5, color="#555555", style="italic")

plt.tight_layout(pad=0.5)
plt.savefig(OUTFILE, dpi=300, bbox_inches="tight", facecolor="#F8F8F8")
plt.close()

sz = os.path.getsize(OUTFILE) // 1024
print(f"Saved: {OUTFILE} ({sz} KB)")
