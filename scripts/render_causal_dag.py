#!/usr/bin/env python3
"""
Render Causal DAG (Supplementary Figure 28) from Mermaid source
using matplotlib + networkx for high-quality PNG output.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(36, 24), dpi=200)
ax.set_xlim(-1, 51)
ax.set_ylim(-1, 25)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')

# ============================================================
# COLOUR PALETTE
# ============================================================
C_PRE   = '#4A6FA5'   # L0 pre-injury (blue)
C_SEV   = '#E74C3C'   # L1 severity (red)
C_ICU   = '#E67E22'   # L2 acute care (orange)
C_LAT   = '#27AE60'   # L3a latent (green)
C_QSP   = '#16A085'   # L3b QSP (teal)
C_OUT   = '#8E44AD'   # L4 outcomes (purple)
C_MORT  = '#7F8C8D'   # mortality (grey)

BG_PRE  = '#D6E4F0'
BG_SEV  = '#FADBD8'
BG_ICU  = '#FDEBD0'
BG_LAT  = '#D5F5E3'
BG_QSP  = '#D1F2EB'
BG_OUT  = '#E8DAEF'

# ============================================================
# NODE POSITIONS  (x, y)   — hand-placed in 6 columns
# ============================================================
# Column X centres:  L0=3, L1=11, L2=21, L3a=31, L3b=31(lower), L4=43

nodes = {
    # --- L0: Pre-injury ---
    'Age':         (3, 22, C_PRE, 'Age\nN(55,18)'),
    'Sex':         (3, 20, C_PRE, 'Sex\nB(0.60)'),
    'Edu':         (3, 18, C_PRE, 'Education\nN(12,3)'),
    'Marital':     (3, 16, C_PRE, 'Marital\nStatus'),
    'Employ':      (3, 14, C_PRE, 'Employment'),
    'Psych':       (3, 11, C_PRE, 'Prior Psych Hx\nB(0.18)'),
    'BrainInj':    (3, 9,  C_PRE, 'Prior Brain\nInjury B(0.08)'),
    'HTN':         (3, 7,  C_PRE, 'HTN'),
    'DM':          (3, 6,  C_PRE, 'DM'),
    'CVD':         (3, 5,  C_PRE, 'CVD'),
    'Anticoag':    (3, 4,  C_PRE, 'Anticoag'),
    'Smoking':     (3, 3,  C_PRE, 'Smoking\nB(0.35)'),
    'Alcohol':     (3, 1,  C_PRE, 'Alcohol\nMisuse B(0.15)'),

    # --- L1: Severity ---
    'Diag':        (11, 22, C_SEV, 'Diagnosis\nTBI/SAH/Stroke/ICH'),
    'GCS':         (11, 18, C_SEV, 'GCS Admission\n3–15'),
    'APACHE':      (11, 14, C_SEV, 'APACHE II\n5–45'),

    # --- L2: Acute care ---
    'ICU_LOS':     (19, 22, C_ICU, 'ICU LOS\nLogN'),
    'MechVent':    (19, 20, C_ICU, 'Mech Vent\ndays'),
    'ICP_Mon':     (19, 18, C_ICU, 'ICP\nMonitoring'),
    'ICP_Mean':    (22, 18, C_ICU, 'ICP Mean\nN(12,4)'),
    'Mobil':       (19, 15, C_ICU, 'Early Mobilisation\nB(0.35)'),
    'Delirium':    (22, 13, C_ICU, 'Delirium\nlogistic'),
    'ICDSC':       (25, 13, C_ICU, 'ICDSC\nScore'),
    'AnxICU':      (22, 10, C_ICU, 'ICU Anxiety\nN(5,3)'),
    'Surgery':     (19, 8,  C_ICU, 'Surgery'),
    'DVT':         (22, 7,  C_ICU, 'DVT'),
    'Pneum':       (22, 5,  C_ICU, 'Pneumonia'),
    'UTI':         (25, 7,  C_ICU, 'UTI'),

    # --- L3a: Latent / Trajectory ---
    'Latent':      (31, 12, C_LAT, 'Latent\nSeverity Score'),
    'Traj':        (35, 12, C_LAT, 'Trajectory Class\n4 classes'),

    # --- L3b: QSP-ODE ---
    'QSP':         (31, 6,  C_QSP, 'QSP-ODE\nMechanistic Features\n(21 variables)'),

    # --- L4: Outcomes ---
    'GOSE':        (43, 22, C_OUT, 'GOSE / FIM\nBarthel / mRS / DRS'),
    'COG':         (43, 19, C_OUT, 'Cog Composite\nMoCA'),
    'HADS_A':      (43, 16, C_OUT, 'HADS-Anxiety\nGAD-7'),
    'HADS_D':      (43, 13, C_OUT, 'HADS-Depression\nPHQ-9'),
    'PCL5':        (47, 14.5, C_OUT, 'PCL-5\n(PTSD)'),
    'QOL':         (47, 20, C_OUT, 'SF-36 / QOLIBRI\nMPAI-4'),
    'RTW':         (47, 10, C_OUT, 'Return to Work\nSocial Partic.'),
    'MORT':        (43, 24, C_MORT, 'Mortality\n12-month'),
}

# ============================================================
# DRAW LAYER BACKGROUNDS
# ============================================================
layer_bgs = [
    ((-0.5, -0.5, 7, 25.5), BG_PRE,  'L0: Pre-Injury'),
    ((8, 11.5, 6, 14),      BG_SEV,  'L1: Injury Severity'),
    ((16, 3, 12, 21.5),     BG_ICU,  'L2: Acute Care / ICU'),
    ((28.5, 9.5, 9, 5.5),   BG_LAT,  'L3a: Latent / Trajectory'),
    ((28.5, 3.5, 5, 5),     BG_QSP,  'L3b: QSP-ODE'),
    ((40.5, 8, 9, 18),      BG_OUT,  'L4: 12-Month Outcomes'),
]

for (x, y, w, h), col, label in layer_bgs:
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3',
                                    facecolor=col, edgecolor='grey',
                                    linewidth=2, alpha=0.35, zorder=0)
    ax.add_patch(rect)
    ax.text(x + 0.3, y + h - 0.3, label, fontsize=11, fontweight='bold',
            color='#333', va='top', ha='left', zorder=1)

# ============================================================
# DRAW NODES
# ============================================================
node_centres = {}
for nid, (x, y, col, label) in nodes.items():
    node_centres[nid] = (x, y)
    box = mpatches.FancyBboxPatch((x - 1.4, y - 0.7), 2.8, 1.4,
                                   boxstyle='round,pad=0.15',
                                   facecolor=col, edgecolor='#222',
                                   linewidth=1.2, alpha=0.92, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, label, fontsize=7, color='white', fontweight='bold',
            ha='center', va='center', zorder=3, linespacing=1.1)

# ============================================================
# EDGES  (from, to, label, colour, linewidth)
# ============================================================
POS_COL = '#2E86C1'   # positive (blue)
NEG_COL = '#C0392B'   # negative (red)
DER_COL = '#7F8C8D'   # derived (grey)
PSY_COL = '#AF7AC5'   # psychiatric (pink)

edges = [
    # L0 → L1
    ('Age',    'APACHE', '+0.15',        POS_COL, 1.0),
    ('Diag',   'GCS',    'varies',       DER_COL, 1.0),
    ('GCS',    'APACHE', '−1.2',         NEG_COL, 1.5),

    # L0/L1 → L2
    ('GCS',    'ICU_LOS',  '',           NEG_COL, 0.8),
    ('GCS',    'MechVent', '',           NEG_COL, 0.8),
    ('GCS',    'ICP_Mon',  'GCS≤8',     NEG_COL, 0.8),
    ('ICP_Mon','ICP_Mean', '',           DER_COL, 0.8),
    ('GCS',    'Delirium', '+0.15',      POS_COL, 1.0),
    ('Age',    'Delirium', '+0.01',      POS_COL, 0.6),
    ('MechVent','Delirium','+0.5',       POS_COL, 1.0),
    ('Mobil',  'Delirium', '−0.4',       NEG_COL, 1.0),
    ('Psych',  'Delirium', '+0.3',       PSY_COL, 1.0),
    ('Delirium','ICDSC',   '',           DER_COL, 0.8),
    ('Psych',  'AnxICU',   '+2.0',       PSY_COL, 2.0),
    ('Delirium','AnxICU',  '+0.3',       POS_COL, 0.8),
    ('Mobil',  'AnxICU',   '−0.5',       NEG_COL, 0.8),
    ('Age',    'AnxICU',   '+0.02',      POS_COL, 0.5),
    ('Diag',   'Surgery',  'varies',     DER_COL, 0.8),
    ('MechVent','Pneum',   '+0.10',      POS_COL, 0.7),
    ('Mobil',  'DVT',      '−0.04',      NEG_COL, 0.7),

    # → L3a  (Latent severity)
    ('GCS',     'Latent', '−0.15',       NEG_COL, 1.2),
    ('APACHE',  'Latent', '+0.08',       POS_COL, 1.0),
    ('Age',     'Latent', '+0.02',       POS_COL, 0.6),
    ('Edu',     'Latent', '−0.05',       NEG_COL, 0.7),
    ('Psych',   'Latent', '+0.50',       PSY_COL, 1.8),
    ('BrainInj','Latent', '+0.30',       POS_COL, 1.0),
    ('Delirium','Latent', '+0.40',       POS_COL, 1.2),
    ('Mobil',   'Latent', '−0.30',       NEG_COL, 1.0),
    ('MechVent','Latent', '+0.20',       POS_COL, 0.8),
    ('DVT',     'Latent', '+0.30',       POS_COL, 0.8),
    ('Pneum',   'Latent', '+0.20',       POS_COL, 0.8),
    ('Latent',  'Traj',   'logistic',    DER_COL, 2.0),

    # → L3b  (QSP)
    ('GCS',     'QSP',    'AR, C₀',     POS_COL, 1.0),
    ('APACHE',  'QSP',    'MAP',         POS_COL, 0.8),
    ('Age',     'QSP',    'rates',       POS_COL, 0.6),
    ('ICP_Mean','QSP',    'ICP_init',    POS_COL, 0.8),
    ('Delirium','QSP',    'DAMP',        POS_COL, 0.8),
    ('Psych',   'QSP',    'priming',     PSY_COL, 1.0),
    ('Alcohol', 'QSP',    'NI priming',  POS_COL, 0.8),

    # → L4  (Outcomes)
    # Functional
    ('GCS',     'GOSE',   '+0.20',       POS_COL, 1.2),
    ('Age',     'GOSE',   '−0.02',       NEG_COL, 0.5),
    ('Edu',     'GOSE',   '+0.05',       POS_COL, 0.5),
    ('Traj',    'GOSE',   '×1.5',        POS_COL, 2.5),
    ('Delirium','GOSE',   '−0.30',       NEG_COL, 1.0),
    ('Mobil',   'GOSE',   '+0.20',       POS_COL, 0.8),
    ('APACHE',  'GOSE',   '−0.015',      NEG_COL, 0.5),

    # Cognitive
    ('Traj',    'COG',    '×0.4',        POS_COL, 2.0),
    ('GCS',     'COG',    '+0.06',       POS_COL, 0.8),
    ('Delirium','COG',    '−0.4',        NEG_COL, 1.0),

    # HADS-Anxiety
    ('AnxICU',  'HADS_A', '+0.60',       PSY_COL, 2.5),
    ('Psych',   'HADS_A', '+2.5',        PSY_COL, 2.5),
    ('Delirium','HADS_A', '+1.5',        POS_COL, 1.5),
    ('Alcohol', 'HADS_A', '+0.5',        POS_COL, 0.8),
    ('Traj',    'HADS_A', '−2.0×mod',    NEG_COL, 2.0),

    # HADS-Depression
    ('Psych',   'HADS_D', '+2.5',        PSY_COL, 2.5),
    ('Delirium','HADS_D', '+1.0',        POS_COL, 1.0),
    ('Marital', 'HADS_D', '−0.4',        NEG_COL, 0.7),
    ('Alcohol', 'HADS_D', '+0.4',        POS_COL, 0.8),
    ('Traj',    'HADS_D', '−2.5×mod',    NEG_COL, 2.0),

    # PCL-5
    ('HADS_A',  'PCL5',   '+1.5',        POS_COL, 1.5),
    ('HADS_D',  'PCL5',   '+1.0',        POS_COL, 1.0),
    ('Psych',   'PCL5',   '+3.0',        PSY_COL, 2.5),
    ('Alcohol', 'PCL5',   '+2.0',        POS_COL, 1.5),

    # QoL
    ('GOSE',    'QOL',    '+3.0',        POS_COL, 1.5),
    ('HADS_A',  'QOL',    '−1.5',        NEG_COL, 1.2),
    ('HADS_D',  'QOL',    '−1.5',        NEG_COL, 1.2),
    ('COG',     'QOL',    '+5.0',        POS_COL, 1.8),
    ('Traj',    'QOL',    '×8',          POS_COL, 2.0),

    # RTW / Social
    ('GOSE',    'RTW',    '+0.3',        POS_COL, 1.0),
    ('HADS_D',  'RTW',    '−0.1',        NEG_COL, 0.5),
    ('COG',     'RTW',    '+0.3',        POS_COL, 0.8),
    ('Traj',    'RTW',    '×1.5',        POS_COL, 1.5),
    ('Marital', 'RTW',    '+3.0',        POS_COL, 1.0),

    # Mortality
    ('GCS',     'MORT',   '+0.3',        POS_COL, 1.5),
    ('Age',     'MORT',   '+0.03',       POS_COL, 0.7),
    ('APACHE',  'MORT',   '+0.02',       POS_COL, 0.7),
    ('Diag',    'MORT',   '+0.5 ICH',    POS_COL, 1.0),
    ('Mobil',   'MORT',   '−0.3',        NEG_COL, 1.0),
]

# ============================================================
# DRAW EDGES
# ============================================================
from matplotlib.patches import FancyArrowPatch

for src, dst, label, col, lw in edges:
    x1, y1 = node_centres[src]
    x2, y2 = node_centres[dst]

    # Offset start/end to node borders
    dx = x2 - x1
    dy = y2 - y1
    dist = max(np.sqrt(dx**2 + dy**2), 0.01)
    # Start/end offset from centre
    off = 1.5
    sx = x1 + dx / dist * off
    sy = y1 + dy / dist * off
    ex = x2 - dx / dist * off
    ey = y2 - dy / dist * off

    arrow = FancyArrowPatch(
        (sx, sy), (ex, ey),
        arrowstyle='->,head_width=3,head_length=3',
        color=col, linewidth=lw, alpha=0.6,
        connectionstyle='arc3,rad=0.08',
        zorder=1
    )
    ax.add_patch(arrow)

    # Label at midpoint
    if label:
        mx = (sx + ex) / 2 + 0.15
        my = (sy + ey) / 2 + 0.15
        ax.text(mx, my, label, fontsize=5.5, color=col, fontweight='bold',
                ha='center', va='center', zorder=4,
                bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.75))

# ============================================================
# LEGEND
# ============================================================
legend_items = [
    (C_PRE,  'Pre-injury factors (L0)'),
    (C_SEV,  'Injury severity (L1)'),
    (C_ICU,  'Acute care / ICU (L2)'),
    (C_LAT,  'Latent / Trajectory (L3a)'),
    (C_QSP,  'QSP-ODE features (L3b)'),
    (C_OUT,  '12-month outcomes (L4)'),
    (C_MORT, 'Mortality'),
]
edge_legend = [
    (POS_COL, 'Positive causal effect'),
    (NEG_COL, 'Negative causal effect'),
    (PSY_COL, 'Strong psychiatric path'),
    (DER_COL, 'Derived / conditional'),
]

ly = 2.5
ax.text(40.5, ly + 1.2, 'Node Categories', fontsize=9, fontweight='bold', color='#333')
for i, (col, lab) in enumerate(legend_items):
    rect = mpatches.FancyBboxPatch((40.5, ly - i * 0.65), 0.5, 0.4,
                                    boxstyle='round,pad=0.05', facecolor=col, zorder=5)
    ax.add_patch(rect)
    ax.text(41.2, ly - i * 0.65 + 0.2, lab, fontsize=7, va='center', zorder=5)

ly2 = ly - len(legend_items) * 0.65 - 0.5
ax.text(40.5, ly2 + 0.5, 'Edge Types', fontsize=9, fontweight='bold', color='#333')
for i, (col, lab) in enumerate(edge_legend):
    ax.plot([40.5, 41.0], [ly2 - i * 0.55, ly2 - i * 0.55],
            color=col, linewidth=2.5, zorder=5)
    ax.text(41.2, ly2 - i * 0.55, lab, fontsize=7, va='center', zorder=5)

# ============================================================
# TITLE
# ============================================================
ax.set_title('Causal DAG: Simulation Data-Generating Process\n(Supplementary Figure 28)',
             fontsize=20, fontweight='bold', pad=20, color='#222')

# Note about QSP dead-end
ax.text(31, 3.2,
        '⚠ QSP features have NO direct edges to outcomes\n'
        '(parallel branch — explains modest ΔR² ≈ +0.002)',
        fontsize=7, ha='center', va='top', style='italic', color='#117A65',
        bbox=dict(boxstyle='round,pad=0.3', fc='#D1F2EB', ec='#16A085', alpha=0.8))

plt.tight_layout()
plt.savefig('causal_dag_mermaid_hq.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved causal_dag_mermaid_hq.png")
plt.close()
