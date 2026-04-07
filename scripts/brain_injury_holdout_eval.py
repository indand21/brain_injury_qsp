#!/usr/bin/env python3
"""
Brain Injury AI — Hold-out Test Set Evaluation
================================================
Trains models on n=2000 cohort (seed=42), evaluates on independent n=500
hold-out cohort (seed=99). Quantifies optimism bias in 5-fold CV estimates.

Outputs:
  holdout_performance_comparison.csv
  fig34_cv_vs_holdout.png
  fig35_optimism_bias_scatter.png
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, r2_score

# ── Constants ────────────────────────────────────────────────────────────────
TRAIN_SEED   = 42
HOLDOUT_SEED = 99
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))

# Add project dir to path so we can import the simulation function
sys.path.insert(0, SCRIPT_DIR)
from brain_injury_ai_pipeline import simulate_neurocritical_cohort

# ── Feature and outcome definitions ──────────────────────────────────────────
BASELINE_FEATURES = [
    'age', 'sex', 'education_years', 'marital_status', 'employment_pre',
    'gcs_admission', 'apache_ii',
    'hypertension', 'diabetes', 'cardiovascular_disease',
    'prior_psych_history', 'prior_brain_injury', 'anticoagulation',
    'smoking', 'alcohol_misuse',
    'icu_los_days', 'mech_ventilation_days',
    'early_mobilization', 'delirium_present', 'icdsc_score',
    'anxiety_icu_score', 'surgery', 'dvt', 'pneumonia', 'uti',
]

REGRESSION_TARGETS = [
    'gose_12m', 'fim_total_12m', 'barthel_12m',
    'hads_anxiety_12m', 'hads_depression_12m',
    'phq9_12m', 'pcl5_12m',
    'moca_12m', 'sf36_mcs_12m', 'qolibri_os_12m',
]

CLASSIFICATION_TARGETS = [
    'return_to_work_12m', 'mortality_12m',
]

# Domain colour map for figures
DOMAIN_COLORS = {
    'gose_12m':          '#2196F3',  # Functional — blue
    'fim_total_12m':     '#2196F3',
    'barthel_12m':       '#2196F3',
    'hads_anxiety_12m':  '#9C27B0',  # Psychiatric — purple
    'hads_depression_12m': '#9C27B0',
    'phq9_12m':          '#9C27B0',
    'pcl5_12m':          '#9C27B0',
    'moca_12m':          '#FF9800',  # Cognitive — orange
    'sf36_mcs_12m':      '#9C27B0',  # Mental health QoL — purple
    'qolibri_os_12m':    '#4CAF50',  # QoL — green
    'return_to_work_12m':'#F44336',  # Classification — red
    'mortality_12m':     '#F44336',
}

DOMAIN_LABELS = {
    'gose_12m':          'Functional',
    'fim_total_12m':     'Functional',
    'barthel_12m':       'Functional',
    'hads_anxiety_12m':  'Psychiatric',
    'hads_depression_12m': 'Psychiatric',
    'phq9_12m':          'Psychiatric',
    'pcl5_12m':          'Psychiatric',
    'moca_12m':          'Cognitive',
    'sf36_mcs_12m':      'Psychiatric',
    'qolibri_os_12m':    'QoL',
    'return_to_work_12m':'Classification',
    'mortality_12m':     'Classification',
}

METRIC_LABELS = {
    'gose_12m':          'R²',
    'fim_total_12m':     'R²',
    'barthel_12m':       'R²',
    'hads_anxiety_12m':  'R²',
    'hads_depression_12m': 'R²',
    'phq9_12m':          'R²',
    'pcl5_12m':          'R²',
    'moca_12m':          'R²',
    'sf36_mcs_12m':      'R²',
    'qolibri_os_12m':    'R²',
    'return_to_work_12m':'AUC',
    'mortality_12m':     'AUC',
}

OUTCOME_SHORT = {
    'gose_12m':           'GOSE',
    'fim_total_12m':      'FIM',
    'barthel_12m':        'Barthel',
    'hads_anxiety_12m':   'HADS-A',
    'hads_depression_12m':'HADS-D',
    'phq9_12m':           'PHQ-9',
    'pcl5_12m':           'PCL-5',
    'moca_12m':           'MoCA',
    'sf36_mcs_12m':       'SF-36 MCS',
    'qolibri_os_12m':     'QOLIBRI-OS',
    'return_to_work_12m': 'RTW',
    'mortality_12m':      'Mortality',
}


# ── Helper: build feature matrix from a dataframe ────────────────────────────
def prepare_X(df):
    le = LabelEncoder()
    df = df.copy()
    df['diagnosis_encoded'] = le.fit_transform(df['diagnosis'])
    df['icp_filled'] = df['icp_mean_mmhg'].fillna(df['icp_mean_mmhg'].median())
    cols = BASELINE_FEATURES + ['diagnosis_encoded', 'icp_filled']
    return df[cols].copy()


# ── Part 1: Generate / load cohorts ──────────────────────────────────────────
def get_cohorts():
    print("=" * 70)
    print("PART 1: GENERATING COHORTS")
    print("=" * 70)

    print("  Simulating training cohort  (n=2000, seed=42) …")
    df_train = simulate_neurocritical_cohort(n=2000, random_state=TRAIN_SEED)
    print(f"  Train shape: {df_train.shape}")

    holdout_csv = os.path.join(SCRIPT_DIR, 'holdout_cohort_n500.csv')
    if os.path.exists(holdout_csv):
        print("  Loading existing hold-out cohort from holdout_cohort_n500.csv …")
        df_hold = pd.read_csv(holdout_csv)
    else:
        print("  Simulating hold-out cohort  (n=500,  seed=99) …")
        df_hold = simulate_neurocritical_cohort(n=500, random_state=HOLDOUT_SEED)
        df_hold.to_csv(holdout_csv, index=False)
        print(f"  Hold-out saved: holdout_cohort_n500.csv")
    print(f"  Hold-out shape: {df_hold.shape}")
    return df_train, df_hold


# ── Part 2: Train → Holdout evaluation ───────────────────────────────────────
def evaluate_all(df_train, df_hold):
    print("\n" + "=" * 70)
    print("PART 2: TRAIN → HOLD-OUT EVALUATION")
    print("=" * 70)

    X_train = prepare_X(df_train)
    X_hold  = prepare_X(df_hold)

    records = []

    # ── Regression outcomes (LassoCV) ────────────────────────────────────────
    for target in REGRESSION_TARGETS:
        if target not in df_train.columns:
            continue
        y_train_full = df_train[target].values
        y_hold_full  = df_hold[target].values

        # Drop NaN rows (e.g., deceased patients with no functional score)
        mask_train = ~np.isnan(y_train_full)
        mask_hold  = ~np.isnan(y_hold_full)
        X_tr = X_train[mask_train]
        y_train = y_train_full[mask_train]
        X_ho = X_hold[mask_hold]
        y_hold = y_hold_full[mask_hold]

        if len(y_train) < 50 or len(y_hold) < 10:
            print(f"  {OUTCOME_SHORT.get(target, target):14s}  skipped (too few non-NaN rows)")
            continue

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso',  LassoCV(cv=5, max_iter=5000, random_state=TRAIN_SEED, n_jobs=-1)),
        ])

        # 5-fold CV score on training set
        cv_scores = cross_val_score(
            model, X_tr, y_train, cv=5, scoring='r2', n_jobs=-1
        )

        # Fit on all training data, predict on hold-out
        model.fit(X_tr, y_train)
        y_pred   = model.predict(X_ho)
        holdout_r2 = r2_score(y_hold, y_pred)

        cv_mean = float(cv_scores.mean())
        cv_std  = float(cv_scores.std())
        bias    = cv_mean - holdout_r2

        records.append({
            'outcome':       target,
            'short_name':    OUTCOME_SHORT[target],
            'domain':        DOMAIN_LABELS[target],
            'metric':        'R²',
            'model':         'LassoCV',
            'cv_mean':       round(cv_mean, 4),
            'cv_std':        round(cv_std, 4),
            'holdout_score': round(holdout_r2, 4),
            'optimism_bias': round(bias, 4),
        })
        print(f"  {OUTCOME_SHORT[target]:14s}  CV={cv_mean:.3f}±{cv_std:.3f}  "
              f"Hold-out={holdout_r2:.3f}  Bias={bias:+.3f}")

    # ── Classification outcomes (RandomForest) ────────────────────────────────
    for target in CLASSIFICATION_TARGETS:
        if target not in df_train.columns:
            continue
        y_train_full = df_train[target].values
        y_hold_full  = df_hold[target].values

        # Drop NaN rows
        mask_train = ~np.isnan(y_train_full.astype(float))
        mask_hold  = ~np.isnan(y_hold_full.astype(float))
        X_tr = X_train[mask_train]
        y_train = y_train_full[mask_train].astype(int)
        X_ho = X_hold[mask_hold]
        y_hold = y_hold_full[mask_hold].astype(int)

        # Skip if only one class present in either split
        if len(np.unique(y_hold)) < 2 or len(np.unique(y_train)) < 2:
            print(f"  {OUTCOME_SHORT.get(target, target):14s}  skipped (single class)")
            continue

        model = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            random_state=TRAIN_SEED, n_jobs=-1
        )

        cv_scores = cross_val_score(
            model, X_tr, y_train, cv=5, scoring='roc_auc', n_jobs=-1
        )

        model.fit(X_tr, y_train)
        y_prob    = model.predict_proba(X_ho)[:, 1]
        holdout_auc = roc_auc_score(y_hold, y_prob)

        cv_mean = float(cv_scores.mean())
        cv_std  = float(cv_scores.std())
        bias    = cv_mean - holdout_auc

        records.append({
            'outcome':       target,
            'short_name':    OUTCOME_SHORT[target],
            'domain':        DOMAIN_LABELS[target],
            'metric':        'AUC',
            'model':         'RandomForest',
            'cv_mean':       round(cv_mean, 4),
            'cv_std':        round(cv_std, 4),
            'holdout_score': round(holdout_auc, 4),
            'optimism_bias': round(bias, 4),
        })
        print(f"  {OUTCOME_SHORT[target]:14s}  CV={cv_mean:.3f}±{cv_std:.3f}  "
              f"Hold-out={holdout_auc:.3f}  Bias={bias:+.3f}")

    df_res = pd.DataFrame(records)
    return df_res


# ── Part 3: Figures ───────────────────────────────────────────────────────────
def make_fig34(df_res):
    """Side-by-side bar chart: 5-fold CV vs hold-out per outcome."""
    outcomes = df_res['short_name'].tolist()
    x        = np.arange(len(outcomes))
    width    = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    colors_cv   = [DOMAIN_COLORS[o] for o in df_res['outcome']]
    colors_hold = [DOMAIN_COLORS[o] for o in df_res['outcome']]

    bars1 = ax.bar(x - width/2, df_res['cv_mean'], width,
                   color=colors_cv, alpha=0.85, label='5-fold CV (train)', edgecolor='white')
    bars2 = ax.bar(x + width/2, df_res['holdout_score'], width,
                   color=colors_hold, alpha=0.45, label='Hold-out (n=500)', edgecolor='black',
                   linewidth=0.8, linestyle='--')

    # Error bars for CV std
    ax.errorbar(x - width/2, df_res['cv_mean'], yerr=df_res['cv_std'],
                fmt='none', color='black', capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(outcomes, rotation=30, ha='right', fontsize=11)
    ax.set_ylabel('R²  /  AUC', fontsize=12)
    ax.set_title('fig34 — 5-fold CV vs Independent Hold-out Performance\n'
                 '(Train n=2000, seed=42 → Hold-out n=500, seed=99)', fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='grey', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)

    # Domain legend
    from matplotlib.patches import Patch
    domain_handles = [
        Patch(facecolor='#2196F3', label='Functional'),
        Patch(facecolor='#9C27B0', label='Psychiatric / Mental QoL'),
        Patch(facecolor='#FF9800', label='Cognitive'),
        Patch(facecolor='#4CAF50', label='QoL'),
        Patch(facecolor='#F44336', label='Classification'),
    ]
    ax.legend(handles=domain_handles + [
        plt.Rectangle((0,0),1,1, fc='grey', alpha=0.85, label='5-fold CV'),
        plt.Rectangle((0,0),1,1, fc='grey', alpha=0.35,
                      ec='black', lw=0.8, ls='--', label='Hold-out'),
    ], loc='upper right', fontsize=9, ncol=2)

    fig.tight_layout()
    path = os.path.join(SCRIPT_DIR, 'SuppFig04_cv_vs_holdout.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: SuppFig04_cv_vs_holdout.png")


def make_fig35(df_res):
    """Optimism bias scatter: x=CV R²/AUC, y=Hold-out, diagonal=no bias."""
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = [DOMAIN_COLORS[o] for o in df_res['outcome']]

    ax.scatter(df_res['cv_mean'], df_res['holdout_score'],
               c=colors, s=120, zorder=3, edgecolors='black', linewidths=0.6)

    # Diagonal (no optimism bias)
    lim_max = max(df_res['cv_mean'].max(), df_res['holdout_score'].max()) + 0.05
    lim_min = min(df_res['cv_mean'].min(), df_res['holdout_score'].min()) - 0.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            'k--', linewidth=1.2, alpha=0.5, label='No optimism bias (diagonal)')

    # Labels
    for _, row in df_res.iterrows():
        ax.annotate(row['short_name'],
                    xy=(row['cv_mean'], row['holdout_score']),
                    xytext=(5, 4), textcoords='offset points',
                    fontsize=9, color='#333333')

    ax.set_xlabel('5-fold CV score (train, n=2000)', fontsize=12)
    ax.set_ylabel('Independent hold-out score (n=500)', fontsize=12)
    ax.set_title('fig35 — Optimism Bias: CV vs Hold-out\n'
                 '(Points above diagonal = pessimistic; below = overfitting)', fontsize=12)
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

    # Domain legend
    from matplotlib.patches import Patch
    domain_handles = [
        Patch(facecolor='#2196F3', label='Functional'),
        Patch(facecolor='#9C27B0', label='Psychiatric / Mental QoL'),
        Patch(facecolor='#FF9800', label='Cognitive'),
        Patch(facecolor='#4CAF50', label='QoL'),
        Patch(facecolor='#F44336', label='Classification'),
    ]
    ax.legend(handles=domain_handles + [
        plt.Line2D([0],[0], color='k', linestyle='--', label='No bias (diagonal)')
    ], loc='lower right', fontsize=9)

    fig.tight_layout()
    path = os.path.join(SCRIPT_DIR, 'SuppFig05_optimism_bias_scatter.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: SuppFig05_optimism_bias_scatter.png")


# ── Part 4: Console summary and CSV output ───────────────────────────────────
def print_summary(df_res):
    print("\n" + "=" * 70)
    print("HOLD-OUT PERFORMANCE SUMMARY")
    print("=" * 70)
    header = f"{'Outcome':<18} {'Metric':<5} {'Model':<14} {'CV Mean':>8} {'CV Std':>7} {'Hold-out':>9} {'Bias':>7}"
    print(header)
    print("-" * len(header))
    for _, row in df_res.iterrows():
        print(f"{row['short_name']:<18} {row['metric']:<5} {row['model']:<14} "
              f"{row['cv_mean']:>8.3f} {row['cv_std']:>7.3f} "
              f"{row['holdout_score']:>9.3f} {row['optimism_bias']:>+7.3f}")
    print("-" * len(header))
    avg_bias = df_res['optimism_bias'].mean()
    print(f"{'Mean optimism bias':<18} {'':>5} {'':>14} {'':>8} {'':>7} {'':>9} {avg_bias:>+7.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("BRAIN INJURY AI — HOLD-OUT EVALUATION")
    print("Train n=2000 (seed=42)  →  Hold-out n=500 (seed=99)")
    print("=" * 70)

    df_train, df_hold = get_cohorts()
    df_res = evaluate_all(df_train, df_hold)

    # Save CSV
    csv_path = os.path.join(SCRIPT_DIR, 'holdout_performance_comparison.csv')
    df_res.to_csv(csv_path, index=False)
    print(f"\n  Results saved: holdout_performance_comparison.csv")

    # Figures
    make_fig34(df_res)
    make_fig35(df_res)

    print_summary(df_res)
    print("\nDone.\n")
