"""
tbi_severity_comparison.py
─────────────────────────────────────────────────────────────────────────────
Moderate vs Severe TBI outcome comparison analysis.

Compares outcomes (functional, cognitive, psychiatric, QoL, binary) at 3, 6,
and 12 months between:
  - Severe TBI  : GCS 3–8   (n varies)
  - Moderate TBI: GCS 9–12  (n varies)
  - Mild TBI    : GCS 13–15 (shown for reference)

Statistical tests  : Mann-Whitney U (continuous), Chi-squared / Fisher's exact
                     (binary); Benjamini-Hochberg FDR correction.
Effect sizes       : rank-biserial correlation r for Mann-Whitney;
                     odds ratio for binary outcomes.
Figures produced   : SuppFig30–SuppFig33
CSV output         : tbi_severity_comparison_results.csv

Author : Brain Injury AI Pipeline (automated)
Date   : 2026-03-11
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import scipy.stats as stats

warnings.filterwarnings('ignore')

OUTDIR = os.path.dirname(os.path.abspath(__file__))
CSV    = os.path.join(OUTDIR, 'simulated_neurocritical_cohort_n2000.csv')

# ── Colour palette ────────────────────────────────────────────────────────────
COL = {'Severe': '#C0392B', 'Moderate': '#2980B9', 'Mild': '#27AE60'}

# ── Outcome definitions ───────────────────────────────────────────────────────
FUNCTIONAL_OUTCOMES = [
    ('gose_12m',       'GOSE',        'higher = better',  True),
    ('fim_total_12m',  'FIM Total',   'higher = better',  True),
    ('barthel_12m',    'Barthel',     'higher = better',  True),
    ('mrs_12m',        'mRS',         'lower = better',   False),
    ('drs_12m',        'DRS',         'lower = better',   False),
]
COGNITIVE_OUTCOMES = [
    ('moca_12m',          'MoCA',           'higher = better', True),
    ('cog_composite_12m', 'Cog. Composite', 'higher = better', True),
]
PSYCHIATRIC_OUTCOMES = [
    ('hads_anxiety_12m',    'HADS-Anxiety',    'lower = better', False),
    ('hads_depression_12m', 'HADS-Depression', 'lower = better', False),
    ('phq9_12m',            'PHQ-9',           'lower = better', False),
    ('pcl5_12m',            'PCL-5',           'lower = better', False),
    ('gad7_12m',            'GAD-7',           'lower = better', False),
]
QOL_OUTCOMES = [
    ('qolibri_os_12m',   'QOLIBRI-OS',  'higher = better', True),
    ('sf36_pcs_12m',     'SF-36 PCS',   'higher = better', True),
    ('sf36_mcs_12m',     'SF-36 MCS',   'higher = better', True),
    ('mpai4_tscore_12m', 'MPAI-4',      'lower = better',  False),
]
BINARY_OUTCOMES = [
    ('return_to_work_12m', 'Return to Work'),
    ('mortality_12m',      'Mortality'),
]

# Panel outcomes for violin figure (most clinically important)
VIOLIN_PANEL = [
    ('gose_12m',          'GOSE 12m',         True),
    ('fim_total_12m',     'FIM Total 12m',    True),
    ('moca_12m',          'MoCA 12m',         True),
    ('hads_anxiety_12m',  'HADS-Anxiety 12m', False),
    ('pcl5_12m',          'PCL-5 12m',        False),
    ('qolibri_os_12m',    'QOLIBRI-OS 12m',   True),
]

# Longitudinal outcomes for trajectory figure
LONGIT_OUTCOMES = [
    ('gose',     'GOSE',         True),
    ('fim_total','FIM Total',    True),
    ('moca',     'MoCA',         True),
    ('hads_anxiety','HADS-Anxiety', False),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def rank_biserial(u_stat, n1, n2):
    """Rank-biserial correlation from Mann-Whitney U statistic."""
    return 1 - (2 * u_stat) / (n1 * n2)

def bh_correction(pvals):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    adj = np.minimum(1.0, pvals * n / ranks)
    # Enforce monotonicity from the largest to smallest rank
    for i in range(n - 2, -1, -1):
        adj[order[i]] = min(adj[order[i]], adj[order[i + 1]])
    return adj

def mannwhitney_compare(a, b, higher_better=True):
    """Return (U, p, r_rb, direction_label)."""
    a, b = a.dropna(), b.dropna()
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan, np.nan, ''
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    r = rank_biserial(u, len(a), len(b))
    med_a, med_b = a.median(), b.median()
    if higher_better:
        label = 'Moderate > Severe' if med_a > med_b else 'Severe > Moderate'
    else:
        label = 'Severe worse' if med_b > med_a else 'Moderate worse'
    return u, p, r, label

def chi2_compare(a, b):
    """Chi-squared test on binary outcomes. Returns (stat, p, OR)."""
    a, b = a.dropna(), b.dropna()
    ct = np.array([[a.sum(), len(a) - a.sum()],
                   [b.sum(), len(b) - b.sum()]])
    if ct.min() < 5:
        _, p = stats.fisher_exact(ct)
        stat = np.nan
    else:
        stat, p, _, _ = stats.chi2_contingency(ct, correction=False)
    try:
        or_ = (ct[0,0] * ct[1,1]) / (ct[0,1] * ct[1,0])
    except ZeroDivisionError:
        or_ = np.nan
    return stat, p, or_


# ─────────────────────────────────────────────────────────────────────────────
# Load & stratify
# ─────────────────────────────────────────────────────────────────────────────
print("Loading cohort …")
df_all = pd.read_csv(CSV)
tbi = df_all[df_all['diagnosis'] == 'TBI'].copy()

sev  = tbi[tbi['gcs_admission'] <= 8].copy()
mod  = tbi[(tbi['gcs_admission'] >= 9) & (tbi['gcs_admission'] <= 12)].copy()
mild = tbi[tbi['gcs_admission'] >= 13].copy()

print(f"  Total TBI  : n={len(tbi)}")
print(f"  Severe     : n={len(sev)}  (GCS 3–8)")
print(f"  Moderate   : n={len(mod)}  (GCS 9–12)")
print(f"  Mild       : n={len(mild)} (GCS 13–15)")

# Exclude deceased from outcome comparisons (outcomes set to NaN post-mortem)
sev_alive  = sev[sev['mortality_12m'] == 0]
mod_alive  = mod[mod['mortality_12m'] == 0]
mild_alive = mild[mild['mortality_12m'] == 0]

print(f"\n  Alive at 12m — Severe: n={len(sev_alive)}, "
      f"Moderate: n={len(mod_alive)}, Mild: n={len(mild_alive)}")


# ─────────────────────────────────────────────────────────────────────────────
# Table 1: Baseline & ICU characteristics by severity
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing baseline characteristics …")

char_rows = []
char_vars = [
    ('age',              'Age (years)',          'continuous'),
    ('sex',              'Male sex (%)',          'binary_pct', 1),
    ('apache_ii',        'APACHE II',            'continuous'),
    ('icu_los_days',     'ICU LOS (days)',        'continuous'),
    ('mech_ventilation_days', 'Mech. vent. (days)', 'continuous'),
    ('icp_monitored',    'ICP monitored (%)',     'binary_pct', 1),
    ('icp_mean_mmhg',    'Mean ICP (mmHg)',       'continuous'),
    ('delirium_present', 'Delirium (%)',          'binary_pct', 1),
    ('anxiety_icu_score','ICU Anxiety Score',     'continuous'),
    ('prior_psych_history','Prior psych. hx (%)', 'binary_pct', 1),
    ('alcohol_misuse',   'Alcohol misuse (%)',    'binary_pct', 1),
    ('mortality_12m',    'Mortality 12m (%)',     'binary_pct', 1),
]

for entry in char_vars:
    col, label, vtype = entry[0], entry[1], entry[2]
    row = {'Variable': label}
    for grp, name in [(sev, 'Severe'), (mod, 'Moderate'), (mild, 'Mild')]:
        d = grp[col].dropna()
        if vtype == 'continuous':
            row[name] = f"{d.median():.1f} ({d.quantile(0.25):.1f}–{d.quantile(0.75):.1f})"
        elif vtype == 'binary_pct':
            val = entry[3]
            row[name] = f"{(d == val).mean()*100:.1f}%"
    # Kruskal-Wallis 3-group or chi2 for binary
    if vtype == 'continuous':
        ks, kp = stats.kruskal(sev[col].dropna(), mod[col].dropna(), mild[col].dropna())
        row['p-value'] = f"{kp:.3f}" if kp >= 0.001 else "<0.001"
    else:
        ct = np.array([[(grp[col] == 1).sum(), (grp[col] == 0).sum()] for grp in [sev, mod, mild]])
        _, cp, _, _ = stats.chi2_contingency(ct)
        row['p-value'] = f"{cp:.3f}" if cp >= 0.001 else "<0.001"
    char_rows.append(row)

df_char = pd.DataFrame(char_rows)
df_char.to_csv(os.path.join(OUTDIR, 'tbi_severity_baseline_table.csv'), index=False)
print("  Saved: tbi_severity_baseline_table.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Statistical comparison: Moderate vs Severe (primary) — all 12m outcomes
# ─────────────────────────────────────────────────────────────────────────────
print("\nRunning Moderate vs Severe 12m outcome comparisons …")

all_outcomes = ([(c, l, 'continuous', hb)
                 for c, l, _, hb in FUNCTIONAL_OUTCOMES + COGNITIVE_OUTCOMES +
                                    PSYCHIATRIC_OUTCOMES + QOL_OUTCOMES] +
                [(c, l, 'binary', None) for c, l in BINARY_OUTCOMES])

results = []
raw_p   = []

for col, label, vtype, hb in all_outcomes:
    row = {'Outcome': label, 'Variable': col}
    if vtype == 'continuous':
        for grp, name in [(sev_alive, 'Severe'), (mod_alive, 'Moderate'), (mild_alive, 'Mild')]:
            d = grp[col].dropna()
            row[f'{name}_median'] = round(d.median(), 1)
            row[f'{name}_iqr']    = f"{d.quantile(0.25):.1f}–{d.quantile(0.75):.1f}"
        u, p, r, direction = mannwhitney_compare(mod_alive[col], sev_alive[col], hb)
        row.update({'U': u, 'p_raw': p, 'r_rb': round(r, 3) if not np.isnan(r) else np.nan,
                    'Direction': direction, 'Test': 'Mann-Whitney U'})
    else:
        for grp, name in [(sev, 'Severe'), (mod, 'Moderate'), (mild, 'Mild')]:
            pct = grp[col].mean() * 100
            row[f'{name}_pct'] = round(pct, 1)
        stat, p, or_ = chi2_compare(mod[col], sev[col])
        row.update({'U': stat, 'p_raw': p, 'r_rb': round(or_, 2) if not np.isnan(or_) else np.nan,
                    'Direction': 'OR (Moderate/Severe)', 'Test': 'Chi-squared'})
    results.append(row)
    raw_p.append(p if not np.isnan(p) else 1.0)

# BH correction
adj_p = bh_correction(raw_p)
for i, row in enumerate(results):
    row['p_adj_BH'] = round(adj_p[i], 4)
    row['Significant'] = adj_p[i] < 0.05

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(OUTDIR, 'tbi_severity_comparison_results.csv'), index=False)
print("  Saved: tbi_severity_comparison_results.csv")
print(f"  Significant after BH correction (p_adj<0.05): "
      f"{df_results['Significant'].sum()}/{len(df_results)}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 (SuppFig30): Violin plots — key outcomes, Severe vs Moderate vs Mild
# ─────────────────────────────────────────────────────────────────────────────
print("\nPlotting SuppFig30 — violin plots …")

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

grp_data = [
    (sev_alive,  'Severe',   COL['Severe']),
    (mod_alive,  'Moderate', COL['Moderate']),
    (mild_alive, 'Mild',     COL['Mild']),
]

for idx, (col, label, higher_better) in enumerate(VIOLIN_PANEL):
    ax = axes[idx]
    parts = ax.violinplot(
        [g[col].dropna().values for g, _, _ in grp_data],
        positions=[1, 2, 3], widths=0.6, showmedians=True,
        showextrema=False
    )
    for pc, (_, _, clr) in zip(parts['bodies'], grp_data):
        pc.set_facecolor(clr)
        pc.set_alpha(0.65)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)

    # Overlay jittered points
    for pos, (grp, _, clr) in enumerate(grp_data, start=1):
        vals = grp[col].dropna().values
        jitter = np.random.uniform(-0.12, 0.12, len(vals))
        ax.scatter(pos + jitter, vals, alpha=0.18, s=5, color=clr, zorder=2)

    # Annotate significance (Moderate vs Severe)
    mod_vals = mod_alive[col].dropna()
    sev_vals = sev_alive[col].dropna()
    if len(mod_vals) > 5 and len(sev_vals) > 5:
        _, p = stats.mannwhitneyu(mod_vals, sev_vals, alternative='two-sided')
        ymax = max(
            sev_alive[col].quantile(0.95),
            mod_alive[col].quantile(0.95)
        ) * 1.05
        sig_str = ('***' if p < 0.001 else '**' if p < 0.01
                   else '*' if p < 0.05 else 'ns')
        ax.annotate('', xy=(2, ymax), xytext=(1, ymax),
                    arrowprops=dict(arrowstyle='-', color='black', lw=1.2))
        ax.text(1.5, ymax * 1.01, sig_str, ha='center', va='bottom', fontsize=11,
                color='black' if sig_str != 'ns' else 'grey')

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Severe\n(GCS 3–8)', 'Moderate\n(GCS 9–12)', 'Mild\n(GCS 13–15)'],
                       fontsize=9)
    ax.set_title(label, fontsize=11, fontweight='bold', pad=6)
    arrow = '↑ better' if higher_better else '↓ better'
    ax.set_ylabel(f"{label} score  ({arrow})", fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', labelsize=8)

fig.suptitle('Outcomes by TBI Severity (Survivors at 12 months)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
out30 = os.path.join(OUTDIR, 'SuppFig30_tbi_severity_outcomes.png')
fig.savefig(out30, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {os.path.basename(out30)}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 (SuppFig31): Longitudinal trajectories — GOSE, FIM, MoCA, HADS-A
# ─────────────────────────────────────────────────────────────────────────────
print("Plotting SuppFig31 — longitudinal trajectories …")

timepoints = [3, 6, 12]
groups_longit = [
    (sev_alive,  'Severe',   COL['Severe'],   'o-'),
    (mod_alive,  'Moderate', COL['Moderate'], 's--'),
    (mild_alive, 'Mild',     COL['Mild'],     '^:'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, (base_col, label, higher_better) in enumerate(LONGIT_OUTCOMES):
    ax = axes[idx]
    for grp, name, clr, fmt in groups_longit:
        means, cis = [], []
        for t in timepoints:
            col = f"{base_col}_{t}m"
            if col not in grp.columns:
                means.append(np.nan); cis.append(np.nan)
                continue
            d = grp[col].dropna()
            m  = d.mean()
            se = d.std() / np.sqrt(len(d))
            means.append(m)
            cis.append(1.96 * se)
        means, cis = np.array(means), np.array(cis)
        ax.plot(timepoints, means, fmt, color=clr, label=name, lw=2, ms=7)
        ax.fill_between(timepoints, means - cis, means + cis,
                        color=clr, alpha=0.15)

    ax.set_xticks(timepoints)
    ax.set_xticklabels(['3m', '6m', '12m'], fontsize=10)
    ax.set_xlabel('Follow-up', fontsize=9)
    arrow = '(↑ better)' if higher_better else '(↓ better)'
    ax.set_ylabel(f"Mean ± 95% CI  {arrow}", fontsize=8)
    ax.set_title(f"{label} — Longitudinal Trajectory", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('Longitudinal Recovery Trajectories by TBI Severity (Survivors)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
out31 = os.path.join(OUTDIR, 'SuppFig31_tbi_severity_trajectories.png')
fig.savefig(out31, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {os.path.basename(out31)}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 (SuppFig32): Effect-size heatmap — rank-biserial r (Moderate vs Severe)
# ─────────────────────────────────────────────────────────────────────────────
print("Plotting SuppFig32 — effect-size heatmap …")

# Compute r for all continuous outcomes at 3m, 6m, 12m
heatmap_outcomes = [
    ('gose',             'GOSE',          True),
    ('fim_total',        'FIM Total',     True),
    ('barthel',          'Barthel',       True),
    ('mrs',              'mRS',           False),
    ('moca',             'MoCA',          True),
    ('cog_composite',    'Cog. Composite',True),
    ('hads_anxiety',     'HADS-Anxiety',  False),
    ('hads_depression',  'HADS-Dep.',     False),
    ('phq9',             'PHQ-9',         False),
    ('pcl5',             'PCL-5',         False),
    ('qolibri_os',       'QOLIBRI-OS',    True),
    ('sf36_pcs',         'SF-36 PCS',     True),
    ('sf36_mcs',         'SF-36 MCS',     True),
]

heatmap_data  = []
heatmap_sig   = []
outcome_labels = []

for base, label, hb in heatmap_outcomes:
    row_r   = []
    row_sig = []
    for t in [3, 6, 12]:
        col = f"{base}_{t}m"
        if col not in mod_alive.columns or col not in sev_alive.columns:
            row_r.append(np.nan); row_sig.append(False); continue
        a = mod_alive[col].dropna()
        b = sev_alive[col].dropna()
        if len(a) < 5 or len(b) < 5:
            row_r.append(np.nan); row_sig.append(False); continue
        u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        r = rank_biserial(u, len(a), len(b))
        # Flip sign so positive r always means Moderate better than Severe
        if not hb:
            r = -r
        row_r.append(round(r, 3))
        row_sig.append(p < 0.05)
    heatmap_data.append(row_r)
    heatmap_sig.append(row_sig)
    outcome_labels.append(label)

heatmap_arr = np.array(heatmap_data, dtype=float)

fig, ax = plt.subplots(figsize=(7, 9))
im = ax.imshow(heatmap_arr, cmap='RdYlGn', vmin=-0.6, vmax=0.6, aspect='auto')

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['3 months', '6 months', '12 months'], fontsize=11)
ax.set_yticks(range(len(outcome_labels)))
ax.set_yticklabels(outcome_labels, fontsize=10)

# Annotate with r values and * for significance
for i in range(len(outcome_labels)):
    for j, t_label in enumerate(['3m', '6m', '12m']):
        val = heatmap_arr[i, j]
        sig = heatmap_sig[i][j]
        if np.isnan(val):
            continue
        txt = f"{val:+.2f}"
        if sig:
            txt += '*'
        txt_color = 'white' if abs(val) > 0.35 else 'black'
        ax.text(j, i, txt, ha='center', va='center', fontsize=8.5,
                color=txt_color, fontweight='bold' if sig else 'normal')

cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
cbar.set_label('Rank-biserial r\n(positive = Moderate better than Severe)',
               fontsize=9, labelpad=8)
cbar.ax.tick_params(labelsize=8)

ax.set_title('Effect Sizes: Moderate vs Severe TBI\n(* p<0.05 uncorrected)',
             fontsize=12, fontweight='bold', pad=10)
ax.spines[:].set_visible(False)

plt.tight_layout()
out32 = os.path.join(OUTDIR, 'SuppFig32_tbi_severity_effectsizes.png')
fig.savefig(out32, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {os.path.basename(out32)}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 (SuppFig33): Binary outcomes + baseline ICU characteristics
# ─────────────────────────────────────────────────────────────────────────────
print("Plotting SuppFig33 — binary outcomes and ICU characteristics …")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# ── Panel A: Binary outcomes (mortality, RTW) ─────────────────────────────
ax = axes[0]
binary_vars = [
    ('mortality_12m',      'Mortality\n(12m)'),
    ('return_to_work_12m', 'Return to\nWork 12m'),
    ('delirium_present',   'Delirium\n(ICU)'),
    ('icp_monitored',      'ICP\nMonitored'),
]

x     = np.arange(len(binary_vars))
width = 0.25

for offset, (grp, name, clr) in zip([-width, 0, width], grp_data):
    pcts  = [grp[c].mean() * 100 for c, _ in binary_vars]
    ns    = [len(grp[c].dropna()) for c, _ in binary_vars]
    # Wilson 95% CI
    lo, hi = [], []
    for pct, n_i in zip(pcts, ns):
        p_ = pct / 100
        z  = 1.96
        ci = z * np.sqrt(p_ * (1 - p_) / n_i) * 100
        lo.append(ci)
        hi.append(ci)
    bars = ax.bar(x + offset, pcts, width=width, color=clr, alpha=0.80,
                  label=name, yerr=[lo, hi], capsize=4,
                  error_kw=dict(elinewidth=1, ecolor='black'))

ax.set_xticks(x)
ax.set_xticklabels([lbl for _, lbl in binary_vars], fontsize=10)
ax.set_ylabel('Percentage (%)', fontsize=10)
ax.set_title('Binary Outcomes & ICU Markers\nby TBI Severity (95% CI)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim(0, 100)

# ── Panel B: ICU continuous characteristics ───────────────────────────────
ax = axes[1]
icu_vars = [
    ('apache_ii',          'APACHE II'),
    ('icu_los_days',       'ICU LOS\n(days)'),
    ('anxiety_icu_score',  'ICU Anxiety\nScore'),
    ('icp_mean_mmhg',      'Mean ICP\n(mmHg)'),
]

# Normalize each variable 0–1 across all TBI for radar-style bar comparison
for iv_idx, (var, var_label) in enumerate(icu_vars):
    col_all = tbi[var].dropna()
    vmin, vmax = col_all.min(), col_all.max()
    positions = np.array([iv_idx - width, iv_idx, iv_idx + width])
    for offset, (grp, name, clr) in zip([-width, 0, width], grp_data):
        mn  = grp[var].mean()
        sem = grp[var].std() / np.sqrt(len(grp[var].dropna()))
        # Plot raw mean with SE
        ax.bar(iv_idx + offset, mn, width=width, color=clr, alpha=0.80,
               yerr=1.96 * sem, capsize=4,
               error_kw=dict(elinewidth=1, ecolor='black'))

ax.set_xticks(range(len(icu_vars)))
ax.set_xticklabels([lbl for _, lbl in icu_vars], fontsize=10)
ax.set_ylabel('Mean ± 95% CI', fontsize=10)
ax.set_title('ICU Characteristics by TBI Severity',
             fontsize=11, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)

# Shared legend
handles = [mpatches.Patch(color=COL[name], label=name)
           for name in ['Severe', 'Moderate', 'Mild']]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
out33 = os.path.join(OUTDIR, 'SuppFig33_tbi_severity_binary_icu.png')
fig.savefig(out33, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {os.path.basename(out33)}")


# ─────────────────────────────────────────────────────────────────────────────
# Print summary table to console
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*75)
print("MODERATE vs SEVERE TBI — 12-MONTH OUTCOME SUMMARY")
print("="*75)
print(f"{'Outcome':<22} {'Severe median':>14} {'Moderate median':>16} "
      f"{'r_rb':>7} {'p_adj':>8} {'Sig':>4}")
print("-"*75)
for _, row in df_results[df_results['Test'] == 'Mann-Whitney U'].iterrows():
    sv = row.get('Severe_median', '—')
    mv = row.get('Moderate_median', '—')
    r  = row.get('r_rb', np.nan)
    p  = row.get('p_adj_BH', np.nan)
    sig = '*' if row.get('Significant', False) else ''
    print(f"{row['Outcome']:<22} {str(sv):>14} {str(mv):>16} "
          f"{r:>7.3f} {p:>8.4f} {sig:>4}")

print("\nBinary outcomes:")
for _, row in df_results[df_results['Test'] == 'Chi-squared'].iterrows():
    sv  = row.get('Severe_pct', '—')
    mv  = row.get('Moderate_pct', '—')
    or_ = row.get('r_rb', np.nan)
    p   = row.get('p_adj_BH', np.nan)
    sig = '*' if row.get('Significant', False) else ''
    print(f"  {row['Outcome']:<20}  Severe={sv}%  Moderate={mv}%  "
          f"OR={or_:.2f}  p_adj={p:.4f} {sig}")

print("\nDone. Figures: SuppFig30–SuppFig33")
print("CSVs : tbi_severity_comparison_results.csv, tbi_severity_baseline_table.csv")
