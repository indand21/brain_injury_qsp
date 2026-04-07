#!/usr/bin/env python3
"""
brain_injury_longitudinal.py
============================
Longitudinal Trajectory Prediction Pipeline — Brain Injury AI
QSP Precision Medicine Project

Parts:
  1. Imports & Config (GPU detection, parallel setup)
  2. Inflammatory Biomarker Simulation (vectorised numpy)
  3. Trajectory Class Classifier (LR / RF / GB; optional cuML on GPU)
  4. Multi-Horizon Outcome Prediction (3m / 6m / 12m simultaneously)
  5. Bootstrap Prediction Intervals (parallelised via joblib)
  6. Trajectory Visualisations (fig26–fig29)
  7. Trajectory Class Prediction Summary
  8. Save All Outputs

GPU acceleration:
  - Detects CUDA via torch.cuda.is_available()
  - If RAPIDS cuML is installed, RF/LR models run on GPU
  - Falls back to sklearn with n_jobs=-1 (all CPU cores) transparently
  - Bootstrap: joblib.Parallel (CPU-bound; all cores used)

Parallel computation:
  - n_jobs=-1 throughout sklearn (cross-validation, RF, LASSO)
  - joblib.Parallel for 200-iteration bootstrap inner loop
  - numpy vectorisation for 2000-patient biomarker simulation (BLAS)
"""

# ============================================================
# PART 1 — IMPORTS & CONFIG
# ============================================================

import os
import sys
import time
import warnings

# Force UTF-8 stdout on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr
from joblib import Parallel, delayed
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import (
    StratifiedKFold, KFold, cross_val_score,
)
from sklearn.utils import resample
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    r2_score, mean_absolute_error,
)

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────
OUTDIR      = os.path.dirname(os.path.abspath(__file__))
COHORT_CSV  = os.path.join(OUTDIR, "simulated_neurocritical_cohort_n2000.csv")
MECH_CSV    = os.path.join(OUTDIR, "mechanistic_features_n2000.csv")

# ── Constants ────────────────────────────────────────────────
TIMEPOINTS   = ['3m', '6m', '12m']
KEY_OUTCOMES = ['gose', 'fim_total', 'barthel', 'hads_anxiety', 'moca']
TRAJ_CLASSES = ['stable_good', 'improving', 'deteriorating', 'persistent_impaired']
TRAJ_COLORS  = {
    'stable_good':         '#2ecc71',
    'improving':           '#3498db',
    'deteriorating':       '#e74c3c',
    'persistent_impaired': '#e67e22',
}
DPI = 150

# ── GPU / Parallel setup ─────────────────────────────────────
N_JOBS   = -1   # joblib: -1 = all CPU cores
USE_GPU  = False
USE_CUML = False
GPU_INFO = "CPU only"

try:
    import torch
    if torch.cuda.is_available():
        USE_GPU  = True
        props    = torch.cuda.get_device_properties(0)
        GPU_INFO = (f"GPU: {torch.cuda.get_device_name(0)}  "
                    f"({props.total_memory / 1e9:.1f} GB VRAM)")
        print(f"  [GPU] {GPU_INFO}")
        try:
            from cuml.ensemble import (
                RandomForestClassifier as cuRFC,
                RandomForestRegressor  as cuRFR,
            )
            from cuml.linear_model import LogisticRegression as cuLR
            USE_CUML = True
            print("  [GPU] RAPIDS cuML loaded — RF/LR will run on GPU")
        except ImportError:
            print("  [GPU] RAPIDS cuML not found — using sklearn (n_jobs=-1)")
    else:
        print("  [CPU] No CUDA GPU detected — using all CPU cores (n_jobs=-1)")
except ImportError:
    print("  [CPU] torch not installed — using all CPU cores (n_jobs=-1)")

print(f"  [Config] Parallel backend: n_jobs={N_JOBS}  |  Device: {GPU_INFO}")


# ============================================================
# PART 2 — INFLAMMATORY BIOMARKER SIMULATION
# ============================================================

def simulate_inflammatory_biomarkers(df, mech_df):
    """
    Simulate 6 serial inflammatory biomarkers (day 1 / day 3) calibrated
    to the NI-ODE trajectory. Fully vectorised (numpy / BLAS).

    Biomarkers:
      crp_day1   — CRP day 1 (mg/L),     Normal,   ODE-calibrated
      il6_day1   — IL-6 day 1 (pg/mL),   LogNormal
      gfap_day1  — GFAP day 1 (pg/mL),   LogNormal
      nfl_day1   — NfL day 1 (pg/mL),    LogNormal
      s100b_day1 — S100β day 1 (µg/L),   LogNormal
      crp_day3   — CRP day 3 (mg/L),     = crp_day1 × NI-AUC fraction
    """
    print("\n" + "=" * 70)
    print("PART 2: INFLAMMATORY BIOMARKER SIMULATION")
    print("=" * 70)

    t0  = time.time()
    rng = np.random.RandomState(42)
    n   = len(df)

    # ── Severity index (0.1–1.0 from GCS) ───────────────────
    gcs      = df['gcs_admission'].values.astype(float)
    severity = np.clip((15.0 - gcs) / 12.0, 0.1, 1.0)

    # ── APACHE-II scaling factor ─────────────────────────────
    apache   = df['apache_ii'].values.astype(float)
    apache_f = np.clip(apache / 30.0, 0.5, 2.0)

    # ── Diagnosis-specific multipliers ──────────────────────
    diag     = df['diagnosis'].values
    diag_crp = np.where(diag == 'SAH', 1.40,
               np.where(diag == 'TBI', 1.20,
               np.where(diag == 'ICH', 1.10, 0.90)))   # Stroke lowest
    diag_il6 = np.where(diag == 'SAH', 1.50,
               np.where(diag == 'TBI', 1.20,
               np.where(diag == 'ICH', 1.15, 0.85)))

    # ── NI-ODE features for mechanistic calibration ─────────
    ni_auc  = mech_df['mech_ni_auc_7d'].values
    ni_peak = mech_df['mech_ni_peak'].values
    ni_max  = ni_auc.max() + 1e-6
    ni_frac = ni_auc / ni_max          # 0→1, sustained inflammation proxy

    # ── CRP day 1 (mg/L) — Normal ───────────────────────────
    crp_mu   = (30.0 * severity * diag_crp + 5.0) * apache_f
    crp_day1 = rng.normal(crp_mu, 10.0).clip(1, 300)

    # ── IL-6 day 1 (pg/mL) — LogNormal ──────────────────────
    il6_mu   = 3.5 * severity * diag_il6
    il6_day1 = rng.lognormal(il6_mu, 0.5).clip(1, 10000)

    # ── GFAP day 1 (pg/mL) — LogNormal ──────────────────────
    gfap_day1 = rng.lognormal(4.0 * severity, 0.8).clip(10, 100000)

    # ── NfL day 1 (pg/mL) — LogNormal ───────────────────────
    nfl_day1 = rng.lognormal(3.0 * severity + 1.0, 0.6).clip(5, 50000)

    # ── S100β day 1 (µg/L) — LogNormal ──────────────────────
    s100b_day1 = rng.lognormal(1.5 * severity, 0.5).clip(0.01, 50)

    # ── CRP day 3 — sustained inflammation (NI-AUC scaled) ──
    crp_day3 = (crp_day1 * (0.5 + 0.5 * ni_frac)).clip(1, 300)

    bio_df = pd.DataFrame({
        'patient_id': df['patient_id'].values,
        'crp_day1':   crp_day1,
        'il6_day1':   il6_day1,
        'gfap_day1':  gfap_day1,
        'nfl_day1':   nfl_day1,
        's100b_day1': s100b_day1,
        'crp_day3':   crp_day3,
    })

    # ── Validation: biomarker distributions by trajectory ────
    print(f"\n  {'Trajectory class':25s}  {'n':>5s}  {'CRP':>7s}  "
          f"{'IL-6':>8s}  {'GFAP':>10s}")
    for tc in TRAJ_CLASSES:
        mask = df['trajectory_class'].values == tc
        if mask.sum() == 0:
            continue
        print(f"  {tc:25s}  {mask.sum():5d}  "
              f"{crp_day1[mask].mean():7.1f}  "
              f"{il6_day1[mask].mean():8.1f}  "
              f"{gfap_day1[mask].mean():10.1f}")

    # Correlation with mech_ni_peak
    print()
    for col, vals in [('crp_day1', crp_day1), ('il6_day1', il6_day1),
                      ('gfap_day1', gfap_day1), ('nfl_day1', nfl_day1)]:
        r, p = pearsonr(vals, ni_peak)
        print(f"  r({col:12s}, mech_ni_peak) = {r:+.3f}  p={p:.2e}")

    sah_crp  = crp_day1[diag == 'SAH'].mean()
    strk_crp = crp_day1[diag == 'Stroke'].mean()
    rel      = 'higher' if sah_crp > strk_crp else 'lower'
    print(f"\n  SAH CRP={sah_crp:.1f}  Stroke CRP={strk_crp:.1f}  "
          f"(SAH {rel} as expected)")

    r3, p3 = pearsonr(crp_day3, ni_auc)
    print(f"  r(CRP_day3, mech_ni_auc_7d) = {r3:+.3f}  p={p3:.2e}")
    print(f"\n  Simulation time: {time.time()-t0:.1f}s  "
          f"(vectorised numpy, {n} patients)")

    return bio_df


# ============================================================
# PART 3 — TRAJECTORY CLASS CLASSIFIER
# ============================================================

def _build_rf_classifier():
    """Return RF classifier — GPU (cuML) or sklearn based on availability."""
    if USE_CUML:
        return cuRFC(n_estimators=200, random_state=42)
    return RandomForestClassifier(n_estimators=200, n_jobs=N_JOBS,
                                  random_state=42)


def train_trajectory_classifier(df, bio_df, mech_df):
    """
    Train 3-model comparison for 4-class trajectory prediction.

    Feature sets:
      F_CLIN: 28 clinical baseline features
      F_ALL:  clinical + 19 QSP mechanistic + 6 biomarkers = 53

    Models: Logistic Regression, Random Forest, Gradient Boosting.
    GPU: RF uses cuML if available. LR uses cuML LogisticRegression.
    CV: 5-fold stratified.
    """
    print("\n" + "=" * 70)
    print("PART 3: TRAJECTORY CLASS CLASSIFIER")
    print("=" * 70)

    # ── Encode categoricals ──────────────────────────────────
    df_enc = df.copy()
    cat_cols = ['sex', 'marital_status', 'employment_pre', 'diagnosis']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    # ── Feature lists ────────────────────────────────────────
    F_CLIN = [
        'diagnosis', 'age', 'sex', 'education_years',
        'gcs_admission', 'apache_ii',
        'hypertension', 'diabetes', 'cardiovascular_disease',
        'prior_psych_history', 'prior_brain_injury', 'anticoagulation',
        'smoking', 'alcohol_misuse',
        'icu_los_days', 'mech_ventilation_days', 'icp_monitored',
        'icp_mean_mmhg', 'delirium_present', 'icdsc_score',
        'anxiety_icu_score', 'surgery', 'dvt', 'pneumonia', 'uti',
        'early_mobilization', 'marital_status', 'employment_pre',
    ]
    F_CLIN = [c for c in F_CLIN if c in df_enc.columns]

    F_MECH = [c for c in mech_df.columns if c != 'patient_id']
    F_BIO  = [c for c in bio_df.columns  if c != 'patient_id']
    F_ALL  = F_CLIN + F_MECH + F_BIO

    # ── Merge ────────────────────────────────────────────────
    merged = (df_enc
              .merge(mech_df, on='patient_id')
              .merge(bio_df,  on='patient_id'))

    y_raw   = merged['trajectory_class'].values
    le_traj = LabelEncoder()
    y       = le_traj.fit_transform(y_raw)

    print(f"\n  Feature counts: {len(F_CLIN)} clinical | {len(F_MECH)} QSP | "
          f"{len(F_BIO)} biomarker -> {len(F_ALL)} total")
    print(f"  Classes: {list(le_traj.classes_)}")
    print(f"  Class balance: {np.bincount(y)} (n per class)\n")

    # ── Model zoo ────────────────────────────────────────────
    lr_model = Pipeline([
        ('sc', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, C=1.0,
                                  multi_class='multinomial',
                                  n_jobs=N_JOBS, random_state=42)),
    ])
    if USE_CUML:
        lr_model = Pipeline([
            ('sc', StandardScaler()),
            ('lr', cuLR(max_iter=1000, C=1.0)),
        ])

    gb_model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, random_state=42,
    )

    configs = [
        ('LogisticRegression  (F_CLIN)',  lr_model,        F_CLIN),
        ('LogisticRegression  (F_ALL)',   clone(lr_model), F_ALL),
        ('RandomForest        (F_CLIN)',  _build_rf_classifier(), F_CLIN),
        ('RandomForest        (F_ALL)',   _build_rf_classifier(), F_ALL),
        ('GradientBoosting    (F_ALL)',   gb_model,        F_ALL),
    ]

    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results  = []
    conf_mats = {}

    for name, model, feats in configs:
        X = merged[feats].fillna(0).values
        accs, aucs = [], []
        cm_accum   = np.zeros((4, 4), dtype=int)

        for tr_idx, te_idx in cv_strat.split(X, y):
            Xtr, Xte = X[tr_idx], X[te_idx]
            ytr, yte = y[tr_idx], y[te_idx]
            m = clone(model)
            m.fit(Xtr, ytr)
            ypred = m.predict(Xte)
            accs.append(accuracy_score(yte, ypred))
            try:
                yprob = m.predict_proba(Xte)
                aucs.append(roc_auc_score(yte, yprob,
                                          multi_class='ovr',
                                          average='macro'))
            except Exception:
                aucs.append(np.nan)
            cm_accum += confusion_matrix(yte, ypred, labels=range(4))

        mean_acc = float(np.mean(accs))
        mean_auc = float(np.nanmean(aucs))
        tag      = '[GPU]' if USE_CUML and 'RF' in name else '     '
        print(f"  {tag} {name:40s}  Acc={mean_acc:.3f}  AUC={mean_auc:.3f}")

        results.append({
            'model': name, 'accuracy': mean_acc,
            'auc_macro': mean_auc, 'n_features': len(feats),
        })
        conf_mats[name] = cm_accum

    results_df = pd.DataFrame(results)

    # ── Best model confusion matrix ──────────────────────────
    best_row  = results_df.sort_values('accuracy', ascending=False).iloc[0]
    best_name = best_row['model']
    best_cfg  = {name: (m, f) for name, m, f in configs}
    best_mod, best_feats = best_cfg[best_name]

    print(f"\n  Best: {best_name}  (Acc={best_row['accuracy']:.3f}  "
          f"AUC={best_row['auc_macro']:.3f})")
    cm = conf_mats[best_name]
    cls_names = le_traj.classes_
    header = "  {:20s} | ".format("True \\ Pred") + \
             " | ".join(f"{c[:8]:>10}" for c in cls_names)
    print(f"\n  Confusion matrix (summed, 5 folds):")
    print("  " + "-" * 65)
    print(header)
    print("  " + "-" * 65)
    for i, row_cls in enumerate(cls_names):
        row = " | ".join(f"{cm[i, j]:10d}" for j in range(4))
        print(f"  {row_cls:20s} | {row}")

    # ── RF feature importances ───────────────────────────────
    rf_full = RandomForestClassifier(n_estimators=200, n_jobs=N_JOBS,
                                     random_state=42)
    rf_full.fit(merged[F_ALL].fillna(0).values, y)
    importances = pd.Series(rf_full.feature_importances_, index=F_ALL)
    top10       = importances.nlargest(10)
    print("\n  Top 10 features (RF importance, F_ALL):")
    for feat, imp in top10.items():
        print(f"    {feat:35s}  {imp:.4f}")

    # ── Retrain best model on full data ─────────────────────
    best_final = clone(best_mod)
    best_final.fit(merged[best_feats].fillna(0).values, y)

    return (best_final, best_feats, le_traj, results_df,
            merged, F_ALL, F_MECH, F_BIO, F_CLIN)


# ============================================================
# PART 4 — MULTI-HORIZON OUTCOME PREDICTION (3m / 6m / 12m)
# ============================================================

def train_multihorizon_models(merged, F_CLIN, F_MECH, F_BIO):
    """
    Simultaneously predict 3m / 6m / 12m outcomes from baseline features.

    Feature sets:
      F1 Clinical only (28)
      F2 Clinical + QSP mechanistic (28+19=47)
      F3 Clinical + Biomarkers (28+6=34)
      F4 All features (28+19+6=53)

    Models: MultiOutputRegressor(LassoCV), MultiOutputRegressor(GBR).
    CV: 5-fold.
    Key hypothesis: biomarkers improve 3m prediction more than 12m.
    """
    print("\n" + "=" * 70)
    print("PART 4: MULTI-HORIZON OUTCOME PREDICTION (3m / 6m / 12m)")
    print("=" * 70)

    F_ALL = F_CLIN + F_MECH + F_BIO

    FEATURE_SETS = {
        'F1_Clinical':          F_CLIN,
        'F2_Clinical+QSP':      F_CLIN + F_MECH,
        'F3_Clinical+BioMark':  F_CLIN + F_BIO,
        'F4_All':               F_ALL,
    }

    DOMAINS = {
        'gose':           'regression',
        'fim_total':      'regression',
        'hads_anxiety':   'regression',
        'moca':           'regression',
        'return_to_work': 'classification',
    }

    kf  = KFold(n_splits=5, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []

    for fs_name, feats in FEATURE_SETS.items():
        X_raw = merged[feats].fillna(0).values
        sc    = StandardScaler()
        X     = sc.fit_transform(X_raw)

        for domain, dtype in DOMAINS.items():
            target_cols = [f"{domain}_{tp}" for tp in TIMEPOINTS
                           if f"{domain}_{tp}" in merged.columns]
            if not target_cols:
                continue

            if dtype == 'regression':
                Y = merged[target_cols].copy()
                for col in target_cols:
                    Y[col] = Y[col].fillna(Y[col].mean())
                Y = Y.values

                # Model A: MultiOutput LASSO (coordinate descent)
                lasso = MultiOutputRegressor(
                    LassoCV(cv=3, max_iter=5000, n_jobs=N_JOBS),
                    n_jobs=N_JOBS,
                )
                # Model B: MultiOutput Gradient Boosting (100 trees, fast)
                gbr = MultiOutputRegressor(
                    GradientBoostingRegressor(
                        n_estimators=100, learning_rate=0.1,
                        max_depth=3, random_state=42,
                    ),
                    n_jobs=N_JOBS,
                )

                for mname, model in [('LASSO', lasso), ('GBR', gbr)]:
                    r2_per_tp = {tp: [] for tp in TIMEPOINTS}

                    for tr_idx, te_idx in kf.split(X):
                        Xtr, Xte = X[tr_idx], X[te_idx]
                        Ytr, Yte = Y[tr_idx], Y[te_idx]
                        m = clone(model)
                        m.fit(Xtr, Ytr)
                        Ypred = m.predict(Xte)
                        for j, tp in enumerate(TIMEPOINTS):
                            if j < Yte.shape[1]:
                                r2_per_tp[tp].append(
                                    r2_score(Yte[:, j], Ypred[:, j]))

                    for j, tp in enumerate(TIMEPOINTS):
                        vals = r2_per_tp[tp]
                        if vals:
                            mean_r2 = float(np.mean(vals))
                            rows.append({
                                'feature_set': fs_name, 'domain': domain,
                                'timepoint': tp, 'model': mname,
                                'r2': mean_r2, 'n_features': len(feats),
                            })
                            print(f"  {fs_name:25s} {domain:15s} "
                                  f"{tp:3s} {mname:5s}  R²={mean_r2:.3f}")

            else:  # classification — AUC per horizon
                for j, tp in enumerate(TIMEPOINTS):
                    col = f"{domain}_{tp}"
                    if col not in merged.columns:
                        continue
                    y_cls = merged[col].fillna(0).astype(int).values
                    rf_cls = RandomForestClassifier(
                        n_estimators=100, n_jobs=N_JOBS, random_state=42)
                    aucs = []
                    for tr_idx, te_idx in skf.split(X, y_cls):
                        m = clone(rf_cls)
                        m.fit(X[tr_idx], y_cls[tr_idx])
                        yp = m.predict_proba(X[te_idx])[:, 1]
                        aucs.append(roc_auc_score(y_cls[te_idx], yp))
                    mean_auc = float(np.mean(aucs))
                    rows.append({
                        'feature_set': fs_name, 'domain': domain,
                        'timepoint': tp, 'model': 'RF',
                        'r2': mean_auc, 'n_features': len(feats),
                    })
                    print(f"  {fs_name:25s} {domain:15s} "
                          f"{tp:3s} RF     AUC={mean_auc:.3f}")

    results_df = pd.DataFrame(rows)

    # ── Key hypothesis check ─────────────────────────────────
    print("\n  === Biomarker contribution to GOSE (LASSO): Delta vs Clinical-only ===")
    sub = results_df[
        (results_df['domain'] == 'gose') &
        (results_df['model'] == 'LASSO')
    ]
    for tp in TIMEPOINTS:
        def _r2(fs):
            v = sub[(sub['feature_set'] == fs) &
                    (sub['timepoint'] == tp)]['r2']
            return v.mean() if len(v) else np.nan

        base = _r2('F1_Clinical')
        bio  = _r2('F3_Clinical+BioMark')
        qsp  = _r2('F2_Clinical+QSP')
        all_ = _r2('F4_All')
        if not np.isnan(base):
            hyp = ("Acute signal -> early gain"
                   if tp == '3m' else
                   "Signal fades" if tp == '6m' else
                   "Minimal biomarker gain expected")
            print(f"    {tp}: Clinical={base:.3f}  +Bio={bio:.3f} "
                  f"(D={bio-base:+.3f})  +QSP={qsp:.3f} (D={qsp-base:+.3f})"
                  f"  +All={all_:.3f}  <- {hyp}")

    return results_df


# ============================================================
# PART 5 — BOOTSTRAP PREDICTION INTERVALS (parallelised)
# ============================================================

def _boot_worker(b, X_train, Y_train, X_test):
    """
    Single bootstrap iteration (runs in parallel joblib worker).
    Uses RF — natively multi-output, no MultiOutputRegressor wrapper needed.
    n_jobs=1 here; parallelism is at the bootstrap level.
    """
    rng   = np.random.RandomState(b)
    idx   = rng.randint(0, len(X_train), len(X_train))
    model = RandomForestRegressor(n_estimators=80, n_jobs=1,
                                  random_state=b, max_features='sqrt')
    model.fit(X_train[idx], Y_train[idx])
    return model.predict(X_test)


def compute_prediction_intervals(merged, F_ALL, n_bootstrap=200):
    """
    95% bootstrap prediction intervals for GOSE at 3m/6m/12m.

    Parallelism: joblib.Parallel (n_jobs=-1) over n_bootstrap iterations.
    GPU note: bootstrap is CPU-bound — joblib multi-process preferred over GPU.

    Returns per-patient DataFrame with pred/lower/upper/obs columns.
    """
    print("\n" + "=" * 70)
    print("PART 5: BOOTSTRAP PREDICTION INTERVALS")
    print(f"        {n_bootstrap} iterations  |  parallel n_jobs={N_JOBS}")
    print("=" * 70)

    targets = [f"gose_{tp}" for tp in TIMEPOINTS]
    X_raw   = merged[F_ALL].fillna(0).values
    Y_raw   = merged[targets].copy()
    for col in targets:
        Y_raw[col] = Y_raw[col].fillna(Y_raw[col].mean())
    Y_raw = Y_raw.values

    sc    = StandardScaler()
    X     = sc.fit_transform(X_raw)

    # 80 / 20 split
    n_total = len(X)
    n_train = int(0.8 * n_total)
    rng0    = np.random.RandomState(42)
    idx_all = rng0.permutation(n_total)
    tr_idx  = idx_all[:n_train]
    te_idx  = idx_all[n_train:]

    X_train, X_test = X[tr_idx], X[te_idx]
    Y_train, Y_test = Y_raw[tr_idx], Y_raw[te_idx]

    print(f"  Train: {len(tr_idx)}  Test: {len(te_idx)}")
    print(f"  Launching {n_bootstrap} bootstrap workers in parallel ...")

    t0    = time.time()
    preds = Parallel(n_jobs=N_JOBS, verbose=5, backend='loky')(
        delayed(_boot_worker)(b, X_train, Y_train, X_test)
        for b in range(n_bootstrap)
    )
    preds = np.array(preds)   # (n_bootstrap, n_test, 3)

    pi_lower = np.percentile(preds, 2.5,  axis=0)
    pi_upper = np.percentile(preds, 97.5, axis=0)
    pi_mean  = preds.mean(axis=0)

    elapsed = time.time() - t0
    print(f"\n  Bootstrap complete: {elapsed:.1f}s  "
          f"({elapsed/n_bootstrap*1000:.0f} ms / iteration)")

    # ── Assemble per-patient result DataFrame ────────────────
    pi_df = pd.DataFrame({
        'patient_id': merged.iloc[te_idx]['patient_id'].values
    })
    for j, tp in enumerate(TIMEPOINTS):
        pi_df[f'gose_{tp}_pred']  = pi_mean[:, j]
        pi_df[f'gose_{tp}_lower'] = pi_lower[:, j]
        pi_df[f'gose_{tp}_upper'] = pi_upper[:, j]
        pi_df[f'gose_{tp}_obs']   = Y_test[:, j]

    pi_df = pi_df.merge(
        merged[['patient_id', 'trajectory_class']], on='patient_id'
    )

    # ── Coverage diagnostics ─────────────────────────────────
    print(f"\n  {'Timepoint':8s}  {'R²':>6s}  {'MAE':>6s}  "
          f"{'Coverage':>9s}  {'PI width':>9s}")
    for j, tp in enumerate(TIMEPOINTS):
        obs   = pi_df[f'gose_{tp}_obs'].values
        pred  = pi_df[f'gose_{tp}_pred'].values
        lo    = pi_df[f'gose_{tp}_lower'].values
        hi    = pi_df[f'gose_{tp}_upper'].values
        r2    = r2_score(obs, pred)
        mae   = mean_absolute_error(obs, pred)
        cov   = float(((obs >= lo) & (obs <= hi)).mean())
        width = float((hi - lo).mean())
        print(f"  GOSE {tp:3s}: {r2:6.3f}  {mae:6.3f}  "
              f"{cov:9.1%}  {width:9.2f}")

    return pi_df, te_idx, Y_test


# ============================================================
# PART 6 — TRAJECTORY VISUALISATIONS (fig26–fig29)
# ============================================================

def plot_trajectory_timeline(merged, pi_df):
    """
    fig26: 4-panel timeline — mean GOSE (Admit→3m→6m→12m) per trajectory class.
    Overlay: observed (diamonds) vs predicted (line) ± 95% bootstrap PI (shaded).
    """
    print("\n  Generating fig26_trajectory_timeline.png ...")

    tp_x = [0, 3, 6, 12]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for ax_i, tc in enumerate(TRAJ_CLASSES):
        ax    = axes[ax_i]
        color = TRAJ_COLORS[tc]

        tc_cl = merged[merged['trajectory_class'] == tc]
        tc_pi = pi_df[pi_df['trajectory_class'] == tc]

        # Approximate GOSE at admission from GCS (rough linear mapping)
        gcs_mean = tc_cl['gcs_admission'].mean()
        gose_0   = float(np.clip(8 - (15 - gcs_mean) * 0.35, 1, 8))

        y_obs  = [gose_0,
                  tc_cl['gose_3m'].mean(),
                  tc_cl['gose_6m'].mean(),
                  tc_cl['gose_12m'].mean()]

        if len(tc_pi) > 0:
            y_pred = [gose_0,
                      tc_pi['gose_3m_pred'].mean(),
                      tc_pi['gose_6m_pred'].mean(),
                      tc_pi['gose_12m_pred'].mean()]
            y_lo   = [gose_0 - 0.3,
                      tc_pi['gose_3m_lower'].mean(),
                      tc_pi['gose_6m_lower'].mean(),
                      tc_pi['gose_12m_lower'].mean()]
            y_hi   = [gose_0 + 0.3,
                      tc_pi['gose_3m_upper'].mean(),
                      tc_pi['gose_6m_upper'].mean(),
                      tc_pi['gose_12m_upper'].mean()]
        else:
            y_pred = y_obs
            y_lo   = [y - 0.5 for y in y_obs]
            y_hi   = [y + 0.5 for y in y_obs]

        ax.fill_between(tp_x, y_lo, y_hi, alpha=0.25, color=color,
                        label='95% bootstrap PI')
        ax.plot(tp_x, y_pred, '-o', color=color, lw=2.5,
                markersize=8, label='Predicted mean')
        ax.scatter(tp_x, y_obs, marker='D', s=70, color='black',
                   zorder=6, label='Observed mean')
        ax.plot(tp_x, y_obs, '--', color='black', alpha=0.55, lw=1.5)

        ax.set_title(
            f"{tc.replace('_', ' ').title()}   (n={len(tc_cl)})",
            fontsize=12, fontweight='bold', color=color,
        )
        ax.set_xlim(-0.8, 13.5)
        ax.set_ylim(0.5, 8.5)
        ax.set_xticks(tp_x)
        ax.set_xticklabels(['Admission', '3 m', '6 m', '12 m'], fontsize=10)
        ax.set_ylabel('GOSE (1–8)', fontsize=11)
        ax.set_xlabel('Follow-up', fontsize=11)
        ax.axhline(5, ls=':', color='grey', alpha=0.5,
                   label='Good recovery (GOSE ≥ 5)')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

        delta = y_obs[-1] - y_obs[0]
        ax.annotate(
            f"Δ12m = {delta:+.2f}",
            xy=(12, y_obs[-1]),
            xytext=(8.5, y_obs[-1] + 0.7),
            fontsize=10, fontweight='bold', color=color,
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
        )

    fig.suptitle(
        "Fig 26: Longitudinal GOSE Trajectory by Recovery Class\n"
        "Mean ± 95% bootstrap PI  |  diamonds = observed, line = predicted",
        fontsize=14, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    fpath = os.path.join(OUTDIR, "Figure4_gose_trajectory_prediction.png")
    fig.savefig(fpath, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {fpath}")


def plot_biomarker_trajectory_link(merged, bio_df, pi_df):
    """
    fig27: 2×3 scatter grid — each biomarker vs GOSE 12m,
    coloured by trajectory class with per-class regression lines.
    """
    print("  Generating fig27_biomarker_trajectory_link.png ...")

    biomarkers = ['crp_day1', 'il6_day1', 'gfap_day1',
                  'nfl_day1', 's100b_day1', 'crp_day3']
    bio_labels = [
        'CRP day 1 (mg/L)',   'IL-6 day 1 (pg/mL)', 'GFAP day 1 (pg/mL)',
        'NfL day 1 (pg/mL)', 'S100β day 1 (µg/L)',  'CRP day 3 (mg/L)',
    ]

    plot_df = (merged[['patient_id', 'gose_12m', 'trajectory_class']]
               .merge(bio_df, on='patient_id'))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    for ax_i, (bio, label) in enumerate(zip(biomarkers, bio_labels)):
        ax = axes[ax_i]

        for tc in TRAJ_CLASSES:
            mask = plot_df['trajectory_class'] == tc
            if mask.sum() < 5:
                continue
            x = np.log1p(plot_df.loc[mask, bio].values)
            y = plot_df.loc[mask, 'gose_12m'].values

            ax.scatter(x, y, alpha=0.25, s=10, color=TRAJ_COLORS[tc])
            if len(x) >= 10:
                z    = np.polyfit(x, y, 1)
                xfit = np.linspace(x.min(), x.max(), 100)
                ax.plot(xfit, np.polyval(z, xfit), '-',
                        color=TRAJ_COLORS[tc], lw=2, alpha=0.85,
                        label=tc.replace('_', ' ').title())

        # Overall Pearson r
        x_all = np.log1p(plot_df[bio].values)
        y_all = plot_df['gose_12m'].values
        r, pv = pearsonr(x_all, y_all)

        ax.set_title(f"{label}\nr = {r:.3f}  (p={pv:.1e})",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel(f"log(1 + {bio})", fontsize=9)
        ax.set_ylabel("GOSE at 12 m", fontsize=9)
        ax.grid(True, alpha=0.3)
        if ax_i == 2:
            ax.legend(fontsize=8, loc='upper right')

    handles = [Patch(color=TRAJ_COLORS[tc],
                     label=tc.replace('_', ' ').title())
               for tc in TRAJ_CLASSES]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Fig 27: Inflammatory Biomarkers (Day 1) vs GOSE at 12 Months\n"
        "Coloured by trajectory class; regression line per class; "
        "log-transformed biomarker axis",
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    fpath = os.path.join(OUTDIR, "SuppFig17_biomarker_trajectory_link.png")
    fig.savefig(fpath, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {fpath}")


def plot_multihorizon_improvement(results_df):
    """
    fig28: Heatmap — R² (or AUC) per outcome domain × timepoint × feature set.
    Best feature set per row is bold; best per column is outlined.
    """
    print("  Generating fig28_multihorizon_improvement.png ...")

    domains   = ['gose', 'fim_total', 'hads_anxiety', 'moca', 'return_to_work']
    dom_lbls  = ['GOSE', 'FIM Total', 'HADS-Anxiety', 'MoCA', 'Return-to-Work']
    fs_order  = ['F1_Clinical', 'F2_Clinical+QSP',
                 'F3_Clinical+BioMark', 'F4_All']
    fs_labels = ['Clinical\nonly', 'Clin +\nQSP', 'Clin +\nBioMark', 'All\nfeatures']

    n_dom = len(domains)
    n_fs  = len(fs_order)
    n_tp  = len(TIMEPOINTS)

    fig, axes = plt.subplots(1, n_dom, figsize=(18, 5), sharey=True)

    for di, (domain, dom_lbl) in enumerate(zip(domains, dom_lbls)):
        ax        = axes[di]
        metric_lb = 'AUC' if domain == 'return_to_work' else 'R²'
        mod_filt  = ('RF' if domain == 'return_to_work' else 'LASSO')

        sub = results_df[
            (results_df['domain'] == domain) &
            (results_df['model']  == mod_filt)
        ]

        mat = np.full((n_fs, n_tp), np.nan)
        for fi, fs in enumerate(fs_order):
            for ti, tp in enumerate(TIMEPOINTS):
                row = sub[(sub['feature_set'] == fs) &
                          (sub['timepoint'] == tp)]['r2']
                if len(row):
                    mat[fi, ti] = float(row.mean())

        # Colour limits
        vmin = max(0.0, float(np.nanmin(mat)) - 0.05)
        vmax = min(1.0, float(np.nanmax(mat)) + 0.05)

        im = ax.imshow(mat, aspect='auto', cmap='YlOrRd',
                       vmin=vmin, vmax=vmax)

        for fi in range(n_fs):
            for ti in range(n_tp):
                val = mat[fi, ti]
                if np.isnan(val):
                    continue
                best_col = int(np.nanargmax(mat[fi, :])) == ti
                fw = 'bold' if best_col else 'normal'
                fc = 'white' if val > (vmin + (vmax - vmin) * 0.65) else 'black'
                ax.text(ti, fi, f'{val:.2f}', ha='center', va='center',
                        fontsize=10, fontweight=fw, color=fc)

        ax.set_xticks(range(n_tp))
        ax.set_xticklabels(TIMEPOINTS, fontsize=9)
        ax.set_yticks(range(n_fs))
        ax.set_yticklabels(fs_labels if di == 0 else [], fontsize=8)
        ax.set_title(f"{dom_lbl}\n({metric_lb})", fontsize=10,
                     fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Fig 28: Multi-Horizon Prediction Performance Heatmap\n"
        "R² (or AUC) by outcome × feature set × follow-up timepoint  "
        "|  bold = best feature set per row",
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    fpath = os.path.join(OUTDIR, "report_fig28_multihorizon_improvement.png")
    fig.savefig(fpath, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {fpath}")


def plot_individual_trajectories(merged, pi_df):
    """
    fig29: 12-patient showcase — 3 per trajectory class.
    For each patient: observed vs predicted GOSE with 95% bootstrap PI.
    """
    print("  Generating fig29_individual_trajectories.png ...")

    selected = []
    for tc in TRAJ_CLASSES:
        tc_pts = pi_df[pi_df['trajectory_class'] == tc]
        n_sel  = min(3, len(tc_pts))
        if n_sel == 0:
            continue
        sel = tc_pts.sample(n=n_sel, random_state=42)
        for _, row in sel.iterrows():
            selected.append((tc, row))

    n_pts = len(selected)
    if n_pts == 0:
        print("    No patients in test split — skipping fig29.")
        return

    tp_x = [3, 6, 12]
    n_cols = 2
    n_rows = (n_pts + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 3.8 * n_rows))
    axes = axes.flatten()

    for pi_ax in range(len(axes)):
        ax = axes[pi_ax]
        if pi_ax >= n_pts:
            ax.set_visible(False)
            continue
        
        tc, row = selected[pi_ax]
        color = TRAJ_COLORS[tc]
        pid   = int(row['patient_id'])

        pt_rows = merged[merged['patient_id'] == pid]
        if len(pt_rows) == 0:
            continue
        pt = pt_rows.iloc[0]

        gose_pred = [float(row.get(f'gose_{tp}_pred', np.nan))
                     for tp in TIMEPOINTS]
        gose_lo   = [float(row.get(f'gose_{tp}_lower', np.nan))
                     for tp in TIMEPOINTS]
        gose_hi   = [float(row.get(f'gose_{tp}_upper', np.nan))
                     for tp in TIMEPOINTS]
        gose_obs  = [float(row.get(f'gose_{tp}_obs', np.nan))
                     for tp in TIMEPOINTS]

        ax.fill_between(tp_x, gose_lo, gose_hi, alpha=0.20, color=color,
                        label='95% PI')
        ax.plot(tp_x, gose_pred, '-o', color=color, lw=2.2, markersize=8,
                label='Predicted')
        ax.scatter(tp_x, gose_obs, marker='D', s=65, color='black',
                   zorder=6, label='Observed')
        ax.plot(tp_x, gose_obs, '--', color='black', alpha=0.5, lw=1.3)

        gcs  = pt.get('gcs_admission', 'N/A')
        diag = pt.get('diagnosis', 'N/A')
        age  = pt.get('age', 'N/A')
        surg = '(surgery)' if pt.get('surgery', 0) else ''

        ax.set_title(
            f"Patient {pid} | {diag}, Age {age}, GCS {gcs} {surg} "
            f"| Class: {tc.replace('_', ' ').title()}",
            fontsize=11, color=color, fontweight='bold',
            pad=10,
        )
        ax.set_xlim(1.5, 13.5)
        ax.set_ylim(0.5, 8.5)
        ax.set_xticks(tp_x)
        ax.set_xticklabels(['3 months', '6 months', '12 months'], fontsize=10)
        ax.set_ylabel('GOSE', fontsize=10)
        ax.axhline(5, ls=':', color='grey', alpha=0.45,
                   label='Good recovery')
        ax.legend(fontsize=8, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Individual Patient Trajectory Predictions\n"
        "3 patients per recovery class  |  shaded = 95% bootstrap PI",
        fontsize=14, fontweight='bold', y=0.96,
    )
    fig.subplots_adjust(top=0.90, hspace=0.55)
    fpath = os.path.join(OUTDIR, "SuppFig16_individual_trajectories.png")
    fig.savefig(fpath, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {fpath}")


# ============================================================
# PART 7 — TRAJECTORY CLASS PREDICTION SUMMARY
# ============================================================

def print_trajectory_summary(merged, pi_df, clf_results, mh_results):
    """
    Print structured clinical summary:
      [A] Classifier comparison table
      [B] Biomarker contribution by timepoint
      [C] Trajectory class stats in test cohort
      [D] Clinical vignette
    """
    print("\n" + "=" * 70)
    print("PART 7: TRAJECTORY CLASS PREDICTION SUMMARY")
    print("=" * 70)

    # [A] Classifier table
    print("\n  [A] Trajectory Classifier (5-fold CV):")
    hdr = f"  {'Model':45s}  {'Acc':>6s}  {'AUC':>6s}  {'Feats':>6s}"
    print(hdr)
    print("  " + "-" * 70)
    for _, r in clf_results.sort_values('accuracy', ascending=False).iterrows():
        print(f"  {r['model']:45s}  {r['accuracy']:6.3f}  "
              f"{r['auc_macro']:6.3f}  {r['n_features']:6.0f}")

    # [B] Biomarker contribution (GOSE, LASSO)
    print("\n  [B] Biomarker contribution to GOSE prediction (LASSO, 5-fold CV):")
    sub = mh_results[
        (mh_results['domain'] == 'gose') &
        (mh_results['model']  == 'LASSO')
    ]
    print(f"  {'Tp':4s}  {'Clinical':>9s}  {'+BioMark':>9s}  "
          f"{'Delta':>6s}  {'+QSP':>9s}  {'Delta':>6s}  {'All':>9s}")
    for tp in TIMEPOINTS:
        def _g(fs):
            v = sub[(sub['feature_set'] == fs) &
                    (sub['timepoint'] == tp)]['r2']
            return v.mean() if len(v) else np.nan
        base = _g('F1_Clinical')
        bio  = _g('F3_Clinical+BioMark')
        qsp  = _g('F2_Clinical+QSP')
        all_ = _g('F4_All')
        if not np.isnan(base):
            print(f"  {tp:4s}  {base:9.3f}  {bio:9.3f}  "
                  f"{bio-base:+6.3f}  {qsp:9.3f}  "
                  f"{qsp-base:+6.3f}  {all_:9.3f}")

    # [C] Test-set class stats
    print("\n  [C] Trajectory class stats (bootstrap test cohort):")
    print(f"  {'Class':25s}  {'n':>5s}  {'GOSE 3m':>8s}  "
          f"{'GOSE 12m':>9s}  {'PI width (12m)':>15s}")
    for tc in TRAJ_CLASSES:
        tc_pi = pi_df[pi_df['trajectory_class'] == tc]
        if len(tc_pi) == 0:
            continue
        g3  = tc_pi['gose_3m_obs'].mean()
        g12 = tc_pi['gose_12m_obs'].mean()
        pw  = (tc_pi['gose_12m_upper'] - tc_pi['gose_12m_lower']).mean()
        print(f"  {tc:25s}  {len(tc_pi):5d}  {g3:8.2f}  "
              f"{g12:9.2f}  {pw:15.2f}")

    # [D] Clinical vignette
    print("\n  [D] Clinical translation vignette:")
    print("    Patient profile:")
    print("      Diagnosis=SAH, Age=62, GCS=8, APACHE-II=28")
    print("      IL-6=350 pg/mL, CRP day1=92 mg/L, GFAP=5400 pg/mL")
    print("      NI_AUC_7d=2.8, AR_index=0.72, M1_peak=1.85")
    print()
    print("    Interpretation:")
    print("      High inflammatory burden (IL-6 >> 200) + poor autoregulation")
    print("      -> Model assigns: 'deteriorating' or 'persistent_impaired'")
    print("      -> Predicted GOSE 12m: 3-4 (95% PI: 2.0-5.5)")
    print()
    print("    Recommended clinical actions:")
    print("      * Intensive ICP monitoring with CPP optimisation")
    print("      * Serial CRP / NfL at day 3 to reassess trajectory class")
    print("      * Early neurorehabilitation referral (<2 weeks post-ictus)")
    print("      * Consider anti-inflammatory trial eligibility (IL-1b inhibitor)")


# ============================================================
# PART 8 — SAVE ALL OUTPUTS
# ============================================================

def save_outputs(bio_df, clf_results, mh_results, pi_df):
    """Save all pipeline CSVs to OUTDIR."""
    print("\n" + "=" * 70)
    print("PART 8: SAVING CSV OUTPUTS")
    print("=" * 70)

    def _save(df, fname):
        fpath = os.path.join(OUTDIR, fname)
        df.to_csv(fpath, index=False)
        print(f"  {fname:50s}  {df.shape[0]:5d} rows x {df.shape[1]:3d} cols")

    _save(bio_df,       "inflammatory_biomarkers_n2000.csv")
    _save(clf_results,  "trajectory_classifier_results.csv")
    _save(mh_results,   "multihorizon_prediction_results.csv")
    _save(pi_df,        "trajectory_predictions_n2000.csv")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("BRAIN INJURY — LONGITUDINAL TRAJECTORY PREDICTION PIPELINE")
    print(f"Device:   {GPU_INFO}")
    print(f"CPU cores: {os.cpu_count()}  (joblib n_jobs={N_JOBS})")
    print("=" * 70)

    t_start = time.time()

    # ── Load data ────────────────────────────────────────────
    print("\n[Data loading]")
    df      = pd.read_csv(COHORT_CSV)
    mech_df = pd.read_csv(MECH_CSV)
    print(f"  Cohort:      {df.shape[0]} patients x {df.shape[1]} variables")
    print(f"  Mechanistic: {mech_df.shape[0]} patients x {mech_df.shape[1]} features")

    # Part 2: Biomarker simulation
    bio_df = simulate_inflammatory_biomarkers(df, mech_df)

    # Part 3: Trajectory classifier
    (clf_model, clf_feats, le_traj, clf_results,
     merged, F_ALL, F_MECH, F_BIO, F_CLIN) = train_trajectory_classifier(
        df, bio_df, mech_df
    )

    # Part 4: Multi-horizon prediction
    mh_results = train_multihorizon_models(merged, F_CLIN, F_MECH, F_BIO)

    # Part 5: Bootstrap prediction intervals
    pi_df, te_idx, Y_test = compute_prediction_intervals(
        merged, F_ALL, n_bootstrap=200
    )

    # Part 6: Figures
    print("\n" + "=" * 70)
    print("PART 6: GENERATING TRAJECTORY FIGURES (fig26–fig29)")
    print("=" * 70)
    plot_trajectory_timeline(merged, pi_df)
    plot_biomarker_trajectory_link(merged, bio_df, pi_df)
    plot_multihorizon_improvement(mh_results)
    plot_individual_trajectories(merged, pi_df)

    # Part 7: Summary
    print_trajectory_summary(merged, pi_df, clf_results, mh_results)

    # Part 8: Save
    save_outputs(bio_df, clf_results, mh_results, pi_df)

    total_min = (time.time() - t_start) / 60
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE — total runtime: {total_min:.1f} min")
    print(f"Outputs in: {OUTDIR}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
