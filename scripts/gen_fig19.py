"""
gen_fig19.py — Generate fig19: mechanistic feature × outcome correlation heatmap
Uses pre-computed CSV files; does not re-run ODE simulations.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = r'C:\Users\indan\OneDrive - aiimsbhubaneswar.edu.in\Brain_Injury_AI_Nitasha'

# ── Load data ────────────────────────────────────────────────────────────────
cohort   = pd.read_csv(f'{OUTDIR}/simulated_neurocritical_cohort_n2000.csv')
mech_raw = pd.read_csv(f'{OUTDIR}/mechanistic_features_n2000.csv')

# Align on patient_id
cohort  = cohort.set_index('patient_id')
mech_df = mech_raw.set_index('patient_id') if 'patient_id' in mech_raw.columns \
          else mech_raw.set_index(mech_raw.columns[0])
df = cohort.join(mech_df, how='inner')

mech_cols = mech_df.columns.tolist()

# ── Outcome groups ────────────────────────────────────────────────────────────
outcomes = {
    'Functional':  ['gose_12m', 'fim_total_12m', 'barthel_12m', 'mrs_12m', 'drs_12m'],
    'Cognitive':   ['cog_composite_12m', 'moca_12m'],
    'Psychiatric': ['hads_anxiety_12m', 'hads_depression_12m', 'phq9_12m', 'pcl5_12m'],
    'QoL':         ['sf36_pcs_12m', 'sf36_mcs_12m', 'qolibri_os_12m'],
    'Employment':  ['return_to_work_12m'],
}
all_outcomes = [o for grp in outcomes.values() for o in grp if o in df.columns]
avail_mch    = [m for m in mech_cols if m in df.columns]

# ── Build correlation matrix ──────────────────────────────────────────────────
corr = pd.DataFrame(index=avail_mch, columns=all_outcomes, dtype=float)
for mf in avail_mch:
    for oc in all_outcomes:
        pair = df[[mf, oc]].dropna()
        if len(pair) > 10:
            corr.loc[mf, oc] = pair.corr().iloc[0, 1]
corr = corr.astype(float)

# Save for reference
corr.to_csv(f'{OUTDIR}/mech_feature_correlations.csv')

# ── Pretty labels ─────────────────────────────────────────────────────────────
MECH_LABELS = {
    'mech_icp_peak':            'ICP Peak',
    'mech_icp_mean_72h':        'ICP Mean 72h',
    'mech_icp_auc_7d':          'ICP AUC 7d',
    'mech_icp_time_above_20':   'ICP Time >20',
    'mech_icp_at_day7':         'ICP at Day 7',
    'mech_cpp_min':             'CPP Min',
    'mech_cpp_mean':            'CPP Mean',
    'mech_cpp_time_below_60':   'CPP Time <60',
    'mech_ar_index':            'AR Index',
    'mech_cpp_optimal_time':    'CPP Optimal Time',
    'mech_m1_peak':             'M1 Peak',
    'mech_ni_peak':             'NI Peak',
    'mech_ni_auc_7d':           'NI AUC 7d',
    'mech_m1_m2_ratio_72h':     'M1/M2 Ratio 72h',
    'mech_np_steady_state':     'NP Steady State',
    'mech_np_auc':              'NP AUC',
    'mech_m2_m1_dominance':     'M2/M1 Dominance',
    'mech_secondary_injury_index': 'Secondary Injury Idx',
    'mech_recovery_potential':  'Recovery Potential',
}
OUT_LABELS = {
    'gose_12m': 'GOSE', 'fim_total_12m': 'FIM', 'barthel_12m': 'Barthel',
    'mrs_12m': 'mRS', 'drs_12m': 'DRS',
    'cog_composite_12m': 'Cog Composite', 'moca_12m': 'MoCA',
    'hads_anxiety_12m': 'HADS-Anx', 'hads_depression_12m': 'HADS-Dep',
    'phq9_12m': 'PHQ-9', 'pcl5_12m': 'PCL-5',
    'sf36_pcs_12m': 'SF-36 PCS', 'sf36_mcs_12m': 'SF-36 MCS',
    'qolibri_os_12m': 'QOLIBRI-OS', 'return_to_work_12m': 'RTW',
}

y_labels = [MECH_LABELS.get(m, m.replace('mech_','').replace('_',' ')) for m in avail_mch]
x_labels = [OUT_LABELS.get(o, o) for o in all_outcomes]

# ── Group separator positions ─────────────────────────────────────────────────
group_sizes = [len([o for o in v if o in df.columns]) for v in outcomes.values()]
group_names = list(outcomes.keys())
sep_positions = []
pos = 0
for gs in group_sizes[:-1]:
    pos += gs
    sep_positions.append(pos - 0.5)

# Mech feature group separators (y-axis)
mech_groups = {
    'ICP':        [m for m in avail_mch if 'icp' in m],
    'CPP / AR':   [m for m in avail_mch if 'cpp' in m or 'ar_' in m],
    'Neuro-\nInflammation': [m for m in avail_mch if any(k in m for k in ['m1','m2','ni','dam'])],
    'Neuro-\nProtection':  [m for m in avail_mch if any(k in m for k in ['np','recovery','secondary'])],
}
mech_sep = []
pos = 0
for grp_mch in list(mech_groups.values())[:-1]:
    pos += len(grp_mch)
    mech_sep.append(pos - 0.5)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 9))
im = ax.imshow(corr.values, cmap='RdBu_r', aspect='auto', vmin=-0.7, vmax=0.7)

# Colorbar
cbar = plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.75, pad=0.02)
cbar.ax.tick_params(labelsize=9)

# Axis ticks
ax.set_xticks(range(len(all_outcomes)))
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(avail_mch)))
ax.set_yticklabels(y_labels, fontsize=9)

# Annotate cells
for i in range(len(avail_mch)):
    for j in range(len(all_outcomes)):
        val = corr.iloc[i, j]
        if not np.isnan(val):
            color = 'white' if abs(val) > 0.38 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6.5, color=color)

# Vertical separators (outcome groups)
for sp in sep_positions:
    ax.axvline(sp, color='white', linewidth=2.5)

# Horizontal separators (mech groups)
for sp in mech_sep:
    ax.axhline(sp, color='white', linewidth=2.5)

# Outcome group labels on top
ax.xaxis.set_label_position('top')
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
grp_centers = []
pos = 0
for gs in group_sizes:
    grp_centers.append(pos + gs / 2 - 0.5)
    pos += gs
ax2.set_xticks(grp_centers)
ax2.set_xticklabels(group_names, fontsize=10, fontweight='bold', color='#333333')
ax2.tick_params(length=0)

# Mech group labels on right
ax3 = ax.twinx()
ax3.set_ylim(ax.get_ylim())
mch_centers = []
pos = 0
for grp_mch in mech_groups.values():
    mch_centers.append(pos + len(grp_mch) / 2 - 0.5)
    pos += len(grp_mch)
ax3.set_yticks(mch_centers)
ax3.set_yticklabels(list(mech_groups.keys()), fontsize=9, fontweight='bold',
                    color='#333333', va='center')
ax3.tick_params(length=0)

ax.set_title('QSP Mechanistic Features × Clinical Outcome Correlations (Pearson r)',
             fontsize=12, fontweight='bold', pad=30)
ax.set_xlabel('12-Month Outcomes', labelpad=8, fontsize=10)
ax.set_ylabel('ODE-Derived Mechanistic Features', labelpad=8, fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUTDIR}/Figure1_mech_outcome_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved Figure1_mech_outcome_heatmap.png")
