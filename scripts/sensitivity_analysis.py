"""
sensitivity_analysis.py
=======================
Three sensitivity analyses for the QSP-ML neurocritical care prediction framework.

SA1 — Sample size (learning curves): n = 250, 500, 750, 1 000, 1 500, 2 000
SA2 — Feature reduction: top-K SHAP features, K = 5, 10, 15, 20, 27
SA3 — Missing data (MCAR): missingness rate = 0%, 5%, 10%, 15%, 20%

Primary outcomes evaluated:
  Regression   : GOSE 12m, FIM Total 12m, HADS-Anxiety 12m, MoCA 12m
  Classification: Mortality 12m, Return-to-Work 12m (AUC)

Outputs:
  sensitivity_sa1_sample_size.csv
  sensitivity_sa2_feature_reduction.csv
  sensitivity_sa3_missing_data.csv
  SuppFig29_sensitivity_analysis.png
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

np.random.seed(42)

# ─── Data paths ───────────────────────────────────────────────────────────────
TRAIN_CSV   = "simulated_neurocritical_cohort_n2000.csv"
HOLDOUT_CSV = "holdout_cohort_n500.csv"
SHAP_CSV    = "shap_importance_matrix.csv"

# ─── Feature definition (matches brain_injury_ai_pipeline.py) ─────────────────
BASE_FEATURES = [
    'age', 'sex', 'education_years', 'marital_status', 'employment_pre',
    'gcs_admission', 'apache_ii',
    'hypertension', 'diabetes', 'cardiovascular_disease',
    'prior_psych_history', 'prior_brain_injury', 'anticoagulation',
    'smoking', 'alcohol_misuse',
    'icu_los_days', 'mech_ventilation_days',
    'early_mobilization', 'delirium_present', 'icdsc_score',
    'anxiety_icu_score', 'surgery', 'dvt', 'pneumonia', 'uti',
    'diagnosis_encoded', 'icp_filled',
]

REG_OUTCOMES  = ['gose_12m', 'fim_total_12m', 'hads_anxiety_12m', 'moca_12m']
CLF_OUTCOMES  = ['mortality_12m', 'return_to_work_12m']

REG_LABELS = {
    'gose_12m': 'GOSE 12m',
    'fim_total_12m': 'FIM Total 12m',
    'hads_anxiety_12m': 'HADS-Anxiety 12m',
    'moca_12m': 'MoCA 12m',
}
CLF_LABELS = {
    'mortality_12m': 'Mortality',
    'return_to_work_12m': 'Return-to-Work',
}

COLORS = {
    'gose_12m':         '#1f77b4',
    'fim_total_12m':    '#ff7f0e',
    'hads_anxiety_12m': '#d62728',
    'moca_12m':         '#2ca02c',
    'mortality_12m':    '#9467bd',
    'return_to_work_12m': '#8c564b',
}


# ─── Helper: encode & clean a dataframe ───────────────────────────────────────
def encode_df(df):
    df = df.copy()
    le = LabelEncoder()
    df['diagnosis_encoded'] = le.fit_transform(df['diagnosis'].astype(str))
    median_icp = df['icp_mean_mmhg'].median()
    df['icp_filled'] = df['icp_mean_mmhg'].fillna(median_icp)
    return df


# ─── Helper: build LASSO pipeline ─────────────────────────────────────────────
def lasso_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   LassoCV(cv=5, max_iter=5000, random_state=42, n_alphas=50)),
    ])


# ─── Helper: build RF pipeline ────────────────────────────────────────────────
def rf_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model',   RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=10,
            random_state=42, n_jobs=-1)),
    ])


# ─── Helper: evaluate on holdout ──────────────────────────────────────────────
def evaluate(X_tr, y_tr, X_ho, y_ho, outcome, is_clf):
    if is_clf:
        pipe = rf_pipeline()
        pipe.fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_ho)[:, 1]
        return roc_auc_score(y_ho, prob)
    else:
        pipe = lasso_pipeline()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_ho)
        return r2_score(y_ho, pred)


# ─── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
train_raw   = pd.read_csv(TRAIN_CSV)
holdout_raw = pd.read_csv(HOLDOUT_CSV)
shap_raw    = pd.read_csv(SHAP_CSV, index_col=0)

train   = encode_df(train_raw)
holdout = encode_df(holdout_raw)

# Ordered feature names from SHAP (mean |SHAP| across all outcomes)
shap_mean = shap_raw.abs().mean(axis=1).sort_values(ascending=False)

# Map SHAP index labels → feature column names
SHAP_LABEL_TO_COL = {
    'Age':                          'age',
    'Sex (Male)':                   'sex',
    'Education (years)':            'education_years',
    'Marital Status':               'marital_status',
    'Pre-injury Employment':        'employment_pre',
    'GCS at Admission':             'gcs_admission',
    'APACHE II':                    'apache_ii',
    'Hypertension':                 'hypertension',
    'Diabetes':                     'diabetes',
    'Cardiovascular Disease':       'cardiovascular_disease',
    'Prior Psychiatric History':    'prior_psych_history',
    'Prior Brain Injury':           'prior_brain_injury',
    'Anticoagulation':              'anticoagulation',
    'Smoking':                      'smoking',
    'Alcohol Misuse':               'alcohol_misuse',
    'ICU LOS (days)':               'icu_los_days',
    'Mechanical Ventilation (days)':'mech_ventilation_days',
    'Early Mobilization':           'early_mobilization',
    'Delirium':                     'delirium_present',
    'ICDSC Score':                  'icdsc_score',
    'ICU Anxiety Score':            'anxiety_icu_score',
    'Surgery':                      'surgery',
    'DVT':                          'dvt',
    'Pneumonia':                    'pneumonia',
    'UTI':                          'uti',
    'Diagnosis':                    'diagnosis_encoded',
    'ICP (mean mmHg)':              'icp_filled',
}

shap_ranked_cols = []
for label in shap_mean.index:
    col = SHAP_LABEL_TO_COL.get(label)
    if col and col in BASE_FEATURES:
        shap_ranked_cols.append(col)
# append any missing
for col in BASE_FEATURES:
    if col not in shap_ranked_cols:
        shap_ranked_cols.append(col)

print(f"Feature ranking (top 10): {shap_ranked_cols[:10]}")


# ═══════════════════════════════════════════════════════════════════════════════
# SA1 — Sample Size Sensitivity
# ═══════════════════════════════════════════════════════════════════════════════
print("\n-- SA1: Sample size sensitivity --")

sample_sizes = [250, 500, 750, 1000, 1500, 2000]
n_repeats    = 5

X_ho = holdout[BASE_FEATURES].values

sa1_rows = []
for outcome in REG_OUTCOMES + CLF_OUTCOMES:
    is_clf = outcome in CLF_OUTCOMES
    y_ho = holdout[outcome].dropna()
    valid_ho = holdout[outcome].notna()
    Xh = holdout.loc[valid_ho, BASE_FEATURES].values
    yh = holdout.loc[valid_ho, outcome].values

    for n in sample_sizes:
        scores = []
        for rep in range(n_repeats):
            rng = np.random.default_rng(rep * 100)
            avail = train[outcome].notna()
            pool  = train[avail].reset_index(drop=True)
            if len(pool) < n:
                n_sample = len(pool)
            else:
                n_sample = n
            idx = rng.choice(len(pool), size=n_sample, replace=False)
            sub = pool.iloc[idx]
            Xtr = sub[BASE_FEATURES].values
            ytr = sub[outcome].values
            try:
                sc = evaluate(Xtr, ytr, Xh, yh, outcome, is_clf)
                scores.append(sc)
            except Exception:
                pass

        metric = 'AUC' if is_clf else 'R²'
        label  = CLF_LABELS.get(outcome, REG_LABELS.get(outcome, outcome))
        sa1_rows.append({
            'outcome': label,
            'metric':  metric,
            'n_train': n,
            'mean':    np.mean(scores),
            'sd':      np.std(scores),
            'n_reps':  len(scores),
        })
        print(f"  {label:25s}  n={n:5d}  {metric}={np.mean(scores):.3f} ± {np.std(scores):.3f}")

sa1_df = pd.DataFrame(sa1_rows)
sa1_df.to_csv("sensitivity_sa1_sample_size.csv", index=False)
print("Saved sensitivity_sa1_sample_size.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# SA2 — Feature Reduction Sensitivity
# ═══════════════════════════════════════════════════════════════════════════════
print("\n-- SA2: Feature reduction sensitivity --")

k_values = [5, 10, 15, 20, 27]

sa2_rows = []
for outcome in REG_OUTCOMES + CLF_OUTCOMES:
    is_clf = outcome in CLF_OUTCOMES
    avail_tr = train[outcome].notna()
    Xtr_full = train.loc[avail_tr, BASE_FEATURES].values
    ytr      = train.loc[avail_tr, outcome].values

    avail_ho = holdout[outcome].notna()
    yh       = holdout.loc[avail_ho, outcome].values

    for k in k_values:
        top_cols = shap_ranked_cols[:k]
        Xtr_k = train.loc[avail_tr, top_cols].values
        Xho_k = holdout.loc[avail_ho, top_cols].values
        try:
            sc = evaluate(Xtr_k, ytr, Xho_k, yh, outcome, is_clf)
        except Exception:
            sc = np.nan

        metric = 'AUC' if is_clf else 'R²'
        label  = CLF_LABELS.get(outcome, REG_LABELS.get(outcome, outcome))
        sa2_rows.append({
            'outcome': label,
            'metric':  metric,
            'k_features': k,
            'score': sc,
        })
        print(f"  {label:25s}  k={k:2d}  {metric}={sc:.3f}")

sa2_df = pd.DataFrame(sa2_rows)
sa2_df.to_csv("sensitivity_sa2_feature_reduction.csv", index=False)
print("Saved sensitivity_sa2_feature_reduction.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# SA3 — Missing Data Sensitivity (MCAR)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n-- SA3: Missing data (MCAR) sensitivity --")

miss_rates = [0.0, 0.05, 0.10, 0.15, 0.20]
n_repeats3 = 5

sa3_rows = []
for outcome in REG_OUTCOMES + CLF_OUTCOMES:
    is_clf = outcome in CLF_OUTCOMES

    avail_tr = train[outcome].notna()
    Xtr_full = train.loc[avail_tr, BASE_FEATURES].values.astype(float)
    ytr      = train.loc[avail_tr, outcome].values

    avail_ho = holdout[outcome].notna()
    Xho      = holdout.loc[avail_ho, BASE_FEATURES].values.astype(float)
    yh       = holdout.loc[avail_ho, outcome].values

    for rate in miss_rates:
        scores = []
        for rep in range(n_repeats3):
            rng = np.random.default_rng(rep * 7 + 13)
            Xtr_miss = Xtr_full.copy()
            if rate > 0:
                mask = rng.random(Xtr_miss.shape) < rate
                Xtr_miss[mask] = np.nan
            try:
                sc = evaluate(Xtr_miss, ytr, Xho, yh, outcome, is_clf)
                scores.append(sc)
            except Exception:
                pass

        metric = 'AUC' if is_clf else 'R²'
        label  = CLF_LABELS.get(outcome, REG_LABELS.get(outcome, outcome))
        sa3_rows.append({
            'outcome':     label,
            'metric':      metric,
            'miss_rate':   rate,
            'mean':        np.mean(scores),
            'sd':          np.std(scores),
        })
        print(f"  {label:25s}  miss={int(rate*100):2d}%  {metric}={np.mean(scores):.3f} ± {np.std(scores):.3f}")

sa3_df = pd.DataFrame(sa3_rows)
sa3_df.to_csv("sensitivity_sa3_missing_data.csv", index=False)
print("Saved sensitivity_sa3_missing_data.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure — 3-panel sensitivity plot
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating figure …")

fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

ax1a = fig.add_subplot(gs[0, 0])   # SA1 regression
ax1b = fig.add_subplot(gs[0, 1])   # SA1 classification
ax2  = fig.add_subplot(gs[1, 0])   # SA2 feature reduction
ax3  = fig.add_subplot(gs[1, 1])   # SA3 missing data

panel_label_kw = dict(fontsize=13, fontweight='bold', va='top')

# ── SA1a: Learning curves — regression ──────────────────────────────────────
for outcome in REG_OUTCOMES:
    sub = sa1_df[sa1_df['outcome'] == REG_LABELS[outcome]]
    ax1a.plot(sub['n_train'], sub['mean'], marker='o', linewidth=2,
              color=COLORS[outcome], label=REG_LABELS[outcome])
    ax1a.fill_between(sub['n_train'],
                      sub['mean'] - sub['sd'],
                      sub['mean'] + sub['sd'],
                      alpha=0.12, color=COLORS[outcome])

ax1a.set_xlabel('Training sample size (n)', fontsize=11)
ax1a.set_ylabel('R² (hold-out)', fontsize=11)
ax1a.set_title('SA1: Sample Size — Regression Outcomes', fontsize=12, fontweight='bold')
ax1a.legend(fontsize=9, loc='lower right')
ax1a.set_xticks(sample_sizes)
ax1a.set_xticklabels([str(n) for n in sample_sizes], rotation=30)
ax1a.axhline(0, color='grey', linewidth=0.7, linestyle='--')
ax1a.text(-0.12, 1.04, 'A', transform=ax1a.transAxes, **panel_label_kw)
ax1a.grid(True, alpha=0.3)

# ── SA1b: Learning curves — classification ───────────────────────────────────
for outcome in CLF_OUTCOMES:
    sub = sa1_df[sa1_df['outcome'] == CLF_LABELS[outcome]]
    ax1b.plot(sub['n_train'], sub['mean'], marker='s', linewidth=2,
              color=COLORS[outcome], label=CLF_LABELS[outcome])
    ax1b.fill_between(sub['n_train'],
                      sub['mean'] - sub['sd'],
                      sub['mean'] + sub['sd'],
                      alpha=0.12, color=COLORS[outcome])

ax1b.set_xlabel('Training sample size (n)', fontsize=11)
ax1b.set_ylabel('AUC (hold-out)', fontsize=11)
ax1b.set_title('SA1: Sample Size — Classification Outcomes', fontsize=12, fontweight='bold')
ax1b.legend(fontsize=9, loc='lower right')
ax1b.set_xticks(sample_sizes)
ax1b.set_xticklabels([str(n) for n in sample_sizes], rotation=30)
ax1b.axhline(0.5, color='grey', linewidth=0.7, linestyle='--', label='Chance')
ax1b.text(-0.12, 1.04, 'B', transform=ax1b.transAxes, **panel_label_kw)
ax1b.grid(True, alpha=0.3)

# ── SA2: Feature reduction ───────────────────────────────────────────────────
for outcome in REG_OUTCOMES:
    sub = sa2_df[(sa2_df['outcome'] == REG_LABELS[outcome]) & (sa2_df['metric'] == 'R²')]
    ax2.plot(sub['k_features'], sub['score'], marker='o', linewidth=2,
             color=COLORS[outcome], label=REG_LABELS[outcome])

for outcome in CLF_OUTCOMES:
    sub = sa2_df[(sa2_df['outcome'] == CLF_LABELS[outcome]) & (sa2_df['metric'] == 'AUC')]
    ax2.plot(sub['k_features'], sub['score'], marker='s', linewidth=2,
             linestyle='--', color=COLORS[outcome], label=CLF_LABELS[outcome])

ax2.set_xlabel('Number of features (top-K by SHAP importance)', fontsize=11)
ax2.set_ylabel('R² / AUC (hold-out)', fontsize=11)
ax2.set_title('SA2: Feature Reduction Sensitivity', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8, loc='lower right')
ax2.set_xticks(k_values)
ax2.axhline(0.5, color='grey', linewidth=0.7, linestyle=':', alpha=0.5)
ax2.text(-0.12, 1.04, 'C', transform=ax2.transAxes, **panel_label_kw)
ax2.grid(True, alpha=0.3)

# ── SA3: Missing data ────────────────────────────────────────────────────────
miss_pct = [int(r * 100) for r in miss_rates]
for outcome in REG_OUTCOMES:
    sub = sa3_df[(sa3_df['outcome'] == REG_LABELS[outcome]) & (sa3_df['metric'] == 'R²')]
    ax3.plot(miss_pct, sub['mean'], marker='o', linewidth=2,
             color=COLORS[outcome], label=REG_LABELS[outcome])
    ax3.fill_between(miss_pct,
                     sub['mean'] - sub['sd'],
                     sub['mean'] + sub['sd'],
                     alpha=0.12, color=COLORS[outcome])

for outcome in CLF_OUTCOMES:
    sub = sa3_df[(sa3_df['outcome'] == CLF_LABELS[outcome]) & (sa3_df['metric'] == 'AUC')]
    ax3.plot(miss_pct, sub['mean'], marker='s', linewidth=2, linestyle='--',
             color=COLORS[outcome], label=CLF_LABELS[outcome])
    ax3.fill_between(miss_pct,
                     sub['mean'] - sub['sd'],
                     sub['mean'] + sub['sd'],
                     alpha=0.12, color=COLORS[outcome])

ax3.set_xlabel('MCAR missingness rate (%)', fontsize=11)
ax3.set_ylabel('R² / AUC (hold-out)', fontsize=11)
ax3.set_title('SA3: Missing Data (MCAR) Sensitivity', fontsize=12, fontweight='bold')
ax3.legend(fontsize=8, loc='lower left')
ax3.set_xticks(miss_pct)
ax3.set_xticklabels([f'{p}%' for p in miss_pct])
ax3.axhline(0.5, color='grey', linewidth=0.7, linestyle=':', alpha=0.5)
ax3.text(-0.12, 1.04, 'D', transform=ax3.transAxes, **panel_label_kw)
ax3.grid(True, alpha=0.3)

fig.suptitle(
    'Sensitivity Analyses: Sample Size, Feature Reduction, and Missing Data\n'
    'Solid lines = R² (regression); dashed lines = AUC (classification)',
    fontsize=13, fontweight='bold', y=1.01
)

plt.savefig('SuppFig29_sensitivity_analysis.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved SuppFig29_sensitivity_analysis.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary table — print for manuscript reporting
# ═══════════════════════════════════════════════════════════════════════════════
print("\n== SA1: Performance at full n=2000 vs n=250 ==")
for outcome in REG_OUTCOMES + CLF_OUTCOMES:
    label = CLF_LABELS.get(outcome, REG_LABELS.get(outcome, outcome))
    metric = 'AUC' if outcome in CLF_OUTCOMES else 'R²'
    sub = sa1_df[sa1_df['outcome'] == label]
    r_250  = sub[sub['n_train'] == 250]['mean'].values[0]
    r_2000 = sub[sub['n_train'] == 2000]['mean'].values[0]
    print(f"  {label:25s}  n=250: {metric}={r_250:.3f}  n=2000: {metric}={r_2000:.3f}  delta={r_2000-r_250:+.3f}")

print("\n== SA2: Full-feature vs top-5 only ==")
for outcome in REG_OUTCOMES + CLF_OUTCOMES:
    label = CLF_LABELS.get(outcome, REG_LABELS.get(outcome, outcome))
    metric = 'AUC' if outcome in CLF_OUTCOMES else 'R²'
    sub = sa2_df[sa2_df['outcome'] == label]
    r_5  = sub[sub['k_features'] == 5]['score'].values[0]
    r_27 = sub[sub['k_features'] == 27]['score'].values[0]
    print(f"  {label:25s}  k=5: {metric}={r_5:.3f}  k=27: {metric}={r_27:.3f}  delta={r_27-r_5:+.3f}")

print("\n== SA3: Baseline vs 20% MCAR ==")
for outcome in REG_OUTCOMES + CLF_OUTCOMES:
    label = CLF_LABELS.get(outcome, REG_LABELS.get(outcome, outcome))
    metric = 'AUC' if outcome in CLF_OUTCOMES else 'R²'
    sub = sa3_df[sa3_df['outcome'] == label]
    r_0  = sub[sub['miss_rate'] == 0.0]['mean'].values[0]
    r_20 = sub[sub['miss_rate'] == 0.2]['mean'].values[0]
    print(f"  {label:25s}  0%: {metric}={r_0:.3f}  20%: {metric}={r_20:.3f}  delta={r_20-r_0:+.3f}")

print("\nAll done.")
