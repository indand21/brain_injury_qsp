"""
brain_injury_qsp_hybrid.py
=============================================================================
Hybrid QSP-ML pipeline for brain injury outcome prediction
Combines:
  1. Monroe-Kellie ICP/CPP ODE model with cerebrovascular autoregulation
  2. Neuroinflammation cascade ODE (DAMP → microglial activation → resolution)
  3. Neuroplasticity/recovery index ODE
  --> Extracts ~22 mechanistic features per patient
  --> Augments ML feature set and compares performance vs clinical-only models

Author: Clinical Pharmacology / QSP-AI precision medicine framework
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from scipy.integrate import solve_ivp
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
import xgboost as xgb

# ─────────────────────────────────────────────────────────────────────────────
# 1. ODE SYSTEM DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def icp_cpp_ode(t, y, params):
    """
    Monroe-Kellie ICP/CPP ODE with cerebrovascular autoregulation.

    State variables:
        y[0] = ICP (mmHg)
        y[1] = V_edema (mL) — cerebral oedema volume
        y[2] = V_csf (mL)  — CSF compartment volume

    Mechanistic basis:
        - Monroe-Kellie: V_brain + V_blood + V_csf = constant (rigid skull)
        - CSF formation/absorption: Davson's equation
        - Cerebrovascular autoregulation: sigmoidal CPP-CBF curve
        - Brain compliance: C_brain = dV/dICP (impaired in severe injury)
        - Vasogenic oedema driven by BBB disruption
    """
    ICP, V_edema, V_csf = y

    MAP = params['MAP']
    CPP = max(MAP - ICP, 0)

    # Cerebrovascular autoregulation (sigmoidal Lassen curve)
    # Autoregulation index: 0 = intact, 1 = fully impaired (pressure-passive)
    AR_index = params['AR_index']  # derived from GCS/injury severity
    CPP_opt = 70.0
    CPP_range = 20.0
    CBF_normal = params['CBF_normal']

    # Intact autoregulation: flat CBF in 50-150 mmHg
    # Impaired: pressure-passive (linear with CPP)
    CBF_autoregulated = CBF_normal / (1 + np.exp(-(CPP - CPP_opt) / CPP_range))
    CBF_pressure_passive = CBF_normal * CPP / 80.0
    CBF = (1 - AR_index) * CBF_autoregulated + AR_index * CBF_pressure_passive
    CBF = max(CBF, 0)

    # Brain compliance (inverse relationship with ICP)
    ICP_ref = 10.0
    C_brain = params['C0_brain'] / (1 + params['k_elastance'] * max(ICP - ICP_ref, 0))

    # CSF dynamics (Davson's equation)
    # Formation rate ~0.35 mL/min = constant; absorption proportional to (ICP - P_venous)
    I_formation = 0.35        # mL/min
    P_venous = 6.0            # mmHg
    R_abs = params['R_csf']   # CSF outflow resistance (increased in injury)
    I_absorption = max(ICP - P_venous, 0) / R_abs

    dV_csf_dt = I_formation - I_absorption

    # Vasogenic oedema driven by BBB disruption (peaks at ~12-24h, resolves over days)
    k_bbb = params['k_bbb']
    k_edema_clear = params['k_edema_clear']
    dV_edema_dt = k_bbb * np.exp(-k_bbb * t / 24.0) - k_edema_clear * V_edema

    # ICP dynamics (via Monroe-Kellie volume conservation)
    # Net volume change drives ICP change via compliance
    CBF_in = CBF / 60.0  # convert to mL/min (approximate cerebral blood volume change)
    dV_total = dV_edema_dt + dV_csf_dt

    dICP_dt = dV_total / C_brain

    return [dICP_dt, dV_edema_dt, dV_csf_dt]


def neuroinflammation_ode(t, y, params):
    """
    Neuroinflammation cascade ODE.

    Based on simplified DAMP-driven microglial activation framework
    (Cheng et al. 2018, Hou et al. 2020 adapted for TBI/SAH/stroke).

    State variables:
        y[0] = DAM  — Damage-associated molecular patterns (DAMP signal)
        y[1] = M1   — Pro-inflammatory microglia/macrophage activation
        y[2] = M2   — Anti-inflammatory / reparative microglia
        y[3] = NI   — Neuroinflammation index (downstream cytokine burden: IL-6, TNF-α)
        y[4] = NP   — Neuroprotection/plasticity index (BDNF, VEGF, anti-apoptotic)

    ODE structure:
        dDAM/dt   = S0 * exp(-k_dam_clear * t) - k_dam_deg * DAM
        dM1/dt    = k_m1 * DAM * (1 - M1/M1_max) - k_m1_res * M2 * M1
        dM2/dt    = k_m2 * M1 - k_m2_deg * M2
        dNI/dt    = k_ni * M1 - k_ni_deg * NI - k_ni_m2 * M2 * NI
        dNP/dt    = k_np * M2 - k_np_deg * NP + k_np_base * (age_factor)
    """
    DAM, M1, M2, NI, NP = y

    S0            = params['S0']            # Initial DAMP signal (severity-dependent)
    k_dam_clear   = params['k_dam_clear']   # DAMP clearance rate
    k_dam_deg     = params['k_dam_deg']
    k_m1          = params['k_m1']          # M1 activation rate
    M1_max        = params['M1_max']
    k_m1_res      = params['k_m1_res']      # M2-mediated M1 resolution
    k_m2          = params['k_m2']          # M2 activation rate
    k_m2_deg      = params['k_m2_deg']
    k_ni          = params['k_ni']          # NI production by M1
    k_ni_deg      = params['k_ni_deg']
    k_ni_m2       = params['k_ni_m2']       # M2-mediated NI suppression
    k_np          = params['k_np']          # NP production by M2
    k_np_deg      = params['k_np_deg']
    k_np_base     = params['k_np_base']     # Basal NP (age-dependent)

    dDAM_dt = S0 * np.exp(-k_dam_clear * t) - k_dam_deg * DAM
    dM1_dt  = k_m1 * DAM * max(1 - M1/M1_max, 0) - k_m1_res * M2 * M1
    dM2_dt  = k_m2 * M1 - k_m2_deg * M2
    dNI_dt  = k_ni * M1 - k_ni_deg * NI - k_ni_m2 * M2 * NI
    dNP_dt  = k_np * M2 - k_np_deg * NP + k_np_base

    return [dDAM_dt, dM1_dt, dM2_dt, dNI_dt, dNP_dt]


# ─────────────────────────────────────────────────────────────────────────────
# 2. PARAMETER PERSONALISATION
# ─────────────────────────────────────────────────────────────────────────────

def get_icp_params(row):
    """
    Personalise ICP/CPP ODE parameters from patient clinical variables.
    """
    gcs   = float(row.get('gcs_admission', 10))
    apache = float(row.get('apache_ii', 20))
    diag  = str(row.get('diagnosis', 'TBI'))
    icp_m = float(row.get('icp_mean_mmhg', 15))
    age   = float(row.get('age', 55))

    # MAP estimate from APACHE II (higher APACHE → more haemodynamic instability)
    MAP = max(70 - (apache - 20) * 0.4, 50)

    # Autoregulation index: GCS 3-8 → severely impaired (~0.8), 9-12 → moderate, 13-15 → intact
    AR_index = np.clip(1.0 - (gcs - 3) / 12.0, 0.1, 0.95)

    # Brain compliance: lower in severe injury / older patients
    severity_factor = np.clip((15 - gcs) / 12.0, 0.1, 1.0)
    C0_brain = 0.8 * (1 - 0.4 * severity_factor) * (1 - 0.003 * max(age - 40, 0))

    # CSF outflow resistance (increased in TBI/SAH)
    diag_factor = {'TBI': 1.3, 'SAH': 1.5, 'Stroke': 1.1, 'ICH': 1.4}.get(diag, 1.2)
    R_csf = 8.0 * diag_factor

    # BBB disruption: drives oedema (more severe injury → higher k_bbb)
    k_bbb = 0.05 * severity_factor * diag_factor

    # Oedema clearance (slower in older/diabetic patients)
    k_edema_clear = 0.02 * (1 - 0.003 * max(age - 40, 0))

    return {
        'MAP': MAP,
        'AR_index': AR_index,
        'CBF_normal': 50.0,
        'C0_brain': C0_brain,
        'k_elastance': 0.12,
        'R_csf': R_csf,
        'k_bbb': k_bbb,
        'k_edema_clear': k_edema_clear,
        'ICP_init': icp_m
    }


def get_ni_params(row):
    """
    Personalise neuroinflammation ODE parameters from patient clinical variables.
    """
    gcs         = float(row.get('gcs_admission', 10))
    age         = float(row.get('age', 55))
    psych_hx    = float(row.get('prior_psych_history', 0))
    alcohol     = float(row.get('alcohol_misuse', 0))
    apache      = float(row.get('apache_ii', 20))
    anxiety_icu = float(row.get('anxiety_icu_score', 5))
    delirium    = float(row.get('delirium_present', 0))

    # Initial DAMP signal proportional to injury severity + systemic inflammation
    severity = np.clip((15 - gcs) / 12.0, 0.1, 1.0)
    apache_factor = np.clip(apache / 30.0, 0.3, 1.5)
    S0 = 2.0 * severity * apache_factor * (1 + 0.3 * delirium)

    # Age slows resolution (inflammaging)
    age_factor = 1.0 + 0.015 * max(age - 40, 0)

    # Prior psych history / alcohol → baseline neuroinflammation (primed microglia)
    priming = 1.0 + 0.3 * psych_hx + 0.2 * alcohol

    return {
        'S0': S0,
        'k_dam_clear': 0.15 / age_factor,
        'k_dam_deg': 0.08,
        'k_m1': 0.5 * priming,
        'M1_max': 2.0,
        'k_m1_res': 0.4 / age_factor,       # slower M2-mediated resolution in elderly
        'k_m2': 0.3 / age_factor,
        'k_m2_deg': 0.15 * age_factor,
        'k_ni': 0.6 * priming,
        'k_ni_deg': 0.2 / age_factor,
        'k_ni_m2': 0.3 / age_factor,
        'k_np': 0.25 / age_factor,
        'k_np_deg': 0.1,
        'k_np_base': 0.02 / age_factor,     # lower in elderly/psych
        'anxiety_icu': anxiety_icu
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. SIMULATE PATIENT TRAJECTORIES & EXTRACT MECHANISTIC FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def simulate_patient(row):
    """
    Run both ODE systems for a single patient and extract mechanistic features.
    Time units: hours (ICP model: 0-168h = 7 days; NI model: 0-720h = 30 days)
    """
    feats = {}

    # ── ICP/CPP simulation (0-168 h) ────────────────────────────────────────
    icp_params = get_icp_params(row)
    t_icp = np.linspace(0, 168, 500)  # 7 days, hourly resolution
    icp_init = icp_params['ICP_init']

    y0_icp = [icp_init, 0.01, 100.0]   # ICP, V_edema, V_csf

    try:
        sol_icp = solve_ivp(
            fun=lambda t, y: icp_cpp_ode(t, y, icp_params),
            t_span=(0, 168), y0=y0_icp,
            t_eval=t_icp,
            method='RK45', rtol=1e-4, atol=1e-6,
            max_step=1.0
        )
        ICP_traj = np.clip(sol_icp.y[0], 0, 80)
        MAP       = icp_params['MAP']
        CPP_traj  = np.clip(MAP - ICP_traj, 0, 150)

    except Exception:
        ICP_traj = np.full(len(t_icp), icp_init)
        CPP_traj = np.full(len(t_icp), max(icp_params['MAP'] - icp_init, 0))

    # ICP features
    feats['mech_icp_peak']           = float(np.max(ICP_traj))
    feats['mech_icp_mean_72h']       = float(np.mean(ICP_traj[t_icp <= 72]))
    feats['mech_icp_auc_7d']         = float(np.trapz(ICP_traj, t_icp))
    feats['mech_icp_time_above_20']  = float(np.sum(ICP_traj > 20) / len(ICP_traj) * 168)  # hours
    feats['mech_icp_time_above_25']  = float(np.sum(ICP_traj > 25) / len(ICP_traj) * 168)
    feats['mech_icp_at_day7']        = float(ICP_traj[-1])

    # CPP features
    feats['mech_cpp_min']            = float(np.min(CPP_traj))
    feats['mech_cpp_mean']           = float(np.mean(CPP_traj))
    feats['mech_cpp_time_below_60']  = float(np.sum(CPP_traj < 60) / len(CPP_traj) * 168)  # hours
    feats['mech_ar_index']           = float(icp_params['AR_index'])
    feats['mech_cpp_optimal_time']   = float(np.sum((CPP_traj >= 60) & (CPP_traj <= 100)) / len(CPP_traj))

    # ── Neuroinflammation simulation (0-720 h = 30 days) ────────────────────
    ni_params = get_ni_params(row)
    t_ni = np.linspace(0, 720, 800)
    y0_ni = [ni_params['S0'], 0.01, 0.01, 0.01, 0.1]  # DAM, M1, M2, NI, NP

    try:
        sol_ni = solve_ivp(
            fun=lambda t, y: neuroinflammation_ode(t, y, ni_params),
            t_span=(0, 720), y0=y0_ni,
            t_eval=t_ni,
            method='RK45', rtol=1e-5, atol=1e-7,
            max_step=2.0
        )
        M1_traj  = np.clip(sol_ni.y[1], 0, 10)
        M2_traj  = np.clip(sol_ni.y[2], 0, 10)
        NI_traj  = np.clip(sol_ni.y[3], 0, 20)
        NP_traj  = np.clip(sol_ni.y[4], 0, 10)

    except Exception:
        M1_traj = NI_traj = NP_traj = np.ones(len(t_ni)) * 0.5
        M2_traj = np.ones(len(t_ni)) * 0.2

    # Neuroinflammation features
    feats['mech_m1_peak']            = float(np.max(M1_traj))
    feats['mech_ni_peak']            = float(np.max(NI_traj))
    feats['mech_ni_auc_7d']          = float(np.trapz(NI_traj[t_ni <= 168], t_ni[t_ni <= 168]))
    feats['mech_m1_m2_ratio_72h']    = float(np.mean(M1_traj[t_ni <= 72]) /
                                             (np.mean(M2_traj[t_ni <= 72]) + 1e-6))
    # Resolution time: when NI falls below 10% of peak
    ni_threshold = 0.1 * np.max(NI_traj) + 1e-6
    resolution_mask = NI_traj < ni_threshold
    if np.any(resolution_mask):
        feats['mech_ni_resolution_time'] = float(t_ni[np.argmax(resolution_mask)])
    else:
        feats['mech_ni_resolution_time'] = 720.0

    feats['mech_np_steady_state']    = float(NP_traj[-1])   # Neuroprotection at 30d
    feats['mech_np_auc']             = float(np.trapz(NP_traj, t_ni))
    feats['mech_m2_m1_dominance']    = float(np.mean(M2_traj[-100:]) /
                                             (np.mean(M1_traj[-100:]) + 1e-6))  # M2 dominance at steady state

    # Composite QSP indices (clinically interpretable)
    feats['mech_secondary_injury_index'] = (
        feats['mech_icp_time_above_20'] / 168 +     # ICP burden (normalised)
        feats['mech_cpp_time_below_60'] / 168 +     # CPP deficit
        feats['mech_ni_auc_7d'] / 500               # Neuroinflammation burden
    )
    feats['mech_recovery_potential'] = (
        feats['mech_np_steady_state'] *
        feats['mech_m2_m1_dominance'] *
        (1 - feats['mech_ar_index'])                # Better autoregulation → more recovery
    )

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# 4. BATCH SIMULATION ACROSS COHORT
# ─────────────────────────────────────────────────────────────────────────────

def run_qsp_simulation(df, n_sample=None):
    """Run ODE simulations for all (or n_sample) patients."""
    if n_sample:
        df_sim = df.sample(n=n_sample, random_state=42)
    else:
        df_sim = df.copy()

    print(f"Running QSP simulations for {len(df_sim)} patients...")
    records = []
    for i, (idx, row) in enumerate(df_sim.iterrows()):
        feats = simulate_patient(row)
        feats['patient_id'] = row.get('patient_id', idx)
        records.append(feats)
        if (i + 1) % 200 == 0:
            print(f"  Simulated {i+1}/{len(df_sim)} patients")

    mech_df = pd.DataFrame(records).set_index('patient_id')
    print(f"  Done. Generated {mech_df.shape[1]} mechanistic features.")
    return mech_df


# ─────────────────────────────────────────────────────────────────────────────
# 5. ML COMPARISON: CLINICAL-ONLY vs HYBRID QSP-ML
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(df_merged, mech_feature_names, key_reg, key_cls):
    """Compare clinical-only vs QSP-hybrid ML models with 5-fold CV."""

    # Clinical features
    clinical_cols = [
        'age', 'sex', 'education_years', 'marital_status', 'employment_pre',
        'diagnosis', 'gcs_admission', 'apache_ii',
        'hypertension', 'diabetes', 'cardiovascular_disease', 'prior_psych_history',
        'prior_brain_injury', 'anticoagulation', 'smoking', 'alcohol_misuse',
        'icu_los_days', 'mech_ventilation_days', 'icp_monitored', 'icp_mean_mmhg',
        'early_mobilization', 'delirium_present', 'icdsc_score', 'anxiety_icu_score',
        'surgery', 'dvt', 'pneumonia', 'uti'
    ]
    avail_clin = [c for c in clinical_cols if c in df_merged.columns]
    hybrid_cols = avail_clin + mech_feature_names

    def prep_X(df, cols):
        X = df[cols].copy()
        for c in X.select_dtypes(include='object').columns:
            X[c] = pd.Categorical(X[c]).codes
        X = X.fillna(X.median(numeric_only=True))
        return StandardScaler().fit_transform(X)

    X_clin   = prep_X(df_merged, avail_clin)
    X_hybrid = prep_X(df_merged, hybrid_cols)

    print(f"\nClinical features: {len(avail_clin)}")
    print(f"Hybrid features:   {len(hybrid_cols)}")

    models_reg = {
        'LASSO':         LassoCV(cv=5, max_iter=5000),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
        'XGBoost':       xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                                          random_state=42, verbosity=0),
    }
    models_cls = {
        'Logistic-L1':   LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear', max_iter=2000),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
        'XGBoost':       xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                           random_state=42, verbosity=0, eval_metric='logloss'),
    }

    cv5  = KFold(n_splits=5, shuffle=True, random_state=42)
    cv5s = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    print("\nRunning cross-validation comparisons...")

    # Build positional index maps (df_merged uses patient_id as index)
    idx_map = {pid: pos for pos, pid in enumerate(df_merged.index)}

    for tgt in key_reg:
        if tgt not in df_merged.columns:
            continue
        y = df_merged[tgt].dropna()
        pos_idx = np.array([idx_map[i] for i in y.index])
        Xc = X_clin[pos_idx]
        Xh = X_hybrid[pos_idx]

        for mname, model in models_reg.items():
            sc_c = cross_val_score(model, Xc, y.values, cv=cv5, scoring='r2', n_jobs=1)
            sc_h = cross_val_score(model, Xh, y.values, cv=cv5, scoring='r2', n_jobs=1)
            results.append({
                'Outcome': tgt, 'Model': mname, 'Metric': 'R²',
                'Clinical_mean': sc_c.mean(), 'Clinical_sd': sc_c.std(),
                'Hybrid_mean':   sc_h.mean(), 'Hybrid_sd':   sc_h.std(),
                'Delta_R2':      sc_h.mean() - sc_c.mean(),
                'Pct_improvement': (sc_h.mean() - sc_c.mean()) / (abs(sc_c.mean()) + 1e-6) * 100
            })
        print(f"  ✓ {tgt}")

    for tgt in key_cls:
        if tgt not in df_merged.columns:
            continue
        y = df_merged[tgt].dropna().astype(int)
        pos_idx = np.array([idx_map[i] for i in y.index])
        Xc = X_clin[pos_idx]
        Xh = X_hybrid[pos_idx]

        for mname, model in models_cls.items():
            sc_c = cross_val_score(model, Xc, y.values, cv=cv5s, scoring='roc_auc', n_jobs=1)
            sc_h = cross_val_score(model, Xh, y.values, cv=cv5s, scoring='roc_auc', n_jobs=1)
            results.append({
                'Outcome': tgt, 'Model': mname, 'Metric': 'AUC-ROC',
                'Clinical_mean': sc_c.mean(), 'Clinical_sd': sc_c.std(),
                'Hybrid_mean':   sc_h.mean(), 'Hybrid_sd':   sc_h.std(),
                'Delta_R2':      sc_h.mean() - sc_c.mean(),
                'Pct_improvement': (sc_h.mean() - sc_c.mean()) / (abs(sc_c.mean()) + 1e-6) * 100
            })
        print(f"  ✓ {tgt}")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 6. MECHANISTIC FEATURE IMPORTANCE (LASSO PATH)
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_importance(df_merged, mech_feature_names, outcome='gose_12m'):
    """LASSO coefficient path to see which mechanistic features enter the model."""
    clinical_cols = [
        'age', 'sex', 'education_years', 'marital_status', 'employment_pre',
        'diagnosis', 'gcs_admission', 'apache_ii',
        'hypertension', 'diabetes', 'cardiovascular_disease', 'prior_psych_history',
        'prior_brain_injury', 'anticoagulation', 'smoking', 'alcohol_misuse',
        'icu_los_days', 'mech_ventilation_days', 'icp_monitored', 'icp_mean_mmhg',
        'early_mobilization', 'delirium_present', 'icdsc_score', 'anxiety_icu_score',
        'surgery', 'dvt', 'pneumonia', 'uti'
    ]
    avail_clin = [c for c in clinical_cols if c in df_merged.columns]
    all_feats = avail_clin + mech_feature_names

    X = df_merged[all_feats].copy()
    for c in X.select_dtypes(include='object').columns:
        X[c] = pd.Categorical(X[c]).codes
    X = X.fillna(X.median(numeric_only=True))
    y = df_merged[outcome].dropna()
    X = X.loc[y.index]
    y = y.loc[X.index]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    lasso = LassoCV(cv=5, max_iter=5000)
    lasso.fit(X_s, y)

    coef_df = pd.DataFrame({
        'Feature': all_feats,
        'LASSO_coef': lasso.coef_,
        'Is_Mechanistic': [f in mech_feature_names for f in all_feats]
    }).sort_values('LASSO_coef', key=abs, ascending=False)

    return coef_df


# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample_trajectories(df, outdir):
    """Plot ICP/CPP and neuroinflammation trajectories for representative patients."""
    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    fig.suptitle('QSP Model Trajectories — Representative Brain Injury Patients',
                 fontsize=14, fontweight='bold', y=0.98)

    diagnoses = ['TBI', 'SAH', 'Stroke', 'ICH']
    t_icp = np.linspace(0, 168, 500)
    t_ni  = np.linspace(0, 720, 800)

    colors = {'TBI': '#E74C3C', 'SAH': '#9B59B6', 'Stroke': '#3498DB', 'ICH': '#F39C12'}

    for col, diag in enumerate(diagnoses):
        subset = df[df['diagnosis'] == diag]
        if len(subset) == 0:
            continue
        # Pick mild/moderate/severe representative
        gcs_vals = subset['gcs_admission'].values
        mild_idx   = subset.index[np.argmax(gcs_vals)]
        severe_idx = subset.index[np.argmin(gcs_vals)]
        mid_val    = np.median(gcs_vals)
        med_idx    = subset.index[np.argmin(np.abs(gcs_vals - mid_val))]

        ax_icp = axes[0, col]
        ax_ni  = axes[1, col]
        ax_np  = axes[2, col]

        for label, pidx, ls in [('Severe', severe_idx, '-'), ('Moderate', med_idx, '--'), ('Mild', mild_idx, ':')]:
            row = df.loc[pidx]
            icp_p = get_icp_params(row)
            ni_p  = get_ni_params(row)

            y0_icp = [icp_p['ICP_init'], 0.01, 100.0]
            try:
                sol = solve_ivp(lambda t, y: icp_cpp_ode(t, y, icp_p), (0, 168), y0_icp,
                                t_eval=t_icp, method='RK45', rtol=1e-4, atol=1e-6, max_step=1.0)
                ICP_t = np.clip(sol.y[0], 0, 80)
                CPP_t = np.clip(icp_p['MAP'] - ICP_t, 0, 150)
            except:
                ICP_t = CPP_t = np.full(len(t_icp), 15)

            y0_ni = [ni_p['S0'], 0.01, 0.01, 0.01, 0.1]
            try:
                sol_ni = solve_ivp(lambda t, y: neuroinflammation_ode(t, y, ni_p), (0, 720), y0_ni,
                                   t_eval=t_ni, method='RK45', rtol=1e-5, atol=1e-7, max_step=2.0)
                NI_t = np.clip(sol_ni.y[3], 0, 20)
                NP_t = np.clip(sol_ni.y[4], 0, 10)
            except:
                NI_t = NP_t = np.full(len(t_ni), 0.5)

            clr = colors[diag]
            alpha_map = {'Severe': 1.0, 'Moderate': 0.7, 'Mild': 0.4}
            ax_icp.plot(t_icp, ICP_t, color=clr, ls=ls, lw=1.5,
                        alpha=alpha_map[label], label=f'{label} (GCS={int(row["gcs_admission"])})')
            ax_ni.plot(t_ni / 24, NI_t, color=clr, ls=ls, lw=1.5, alpha=alpha_map[label])
            ax_np.plot(t_ni / 24, NP_t, color=clr, ls=ls, lw=1.5, alpha=alpha_map[label])

        # Formatting
        ax_icp.axhline(20, color='red', ls='--', lw=0.8, alpha=0.5, label='ICP=20 mmHg')
        ax_icp.axhline(25, color='darkred', ls='--', lw=0.8, alpha=0.5, label='ICP=25 mmHg')
        ax_icp.set_title(f'{diag} — ICP Trajectory', fontsize=10, fontweight='bold')
        ax_icp.set_xlabel('Time (hours)', fontsize=8)
        ax_icp.set_ylabel('ICP (mmHg)', fontsize=8)
        ax_icp.legend(fontsize=6, loc='upper right')
        ax_icp.set_ylim(0, 50)
        ax_icp.set_facecolor('#F8F9FA')

        ax_ni.set_title(f'{diag} — Neuroinflammation (NI Index)', fontsize=10, fontweight='bold')
        ax_ni.set_xlabel('Time (days)', fontsize=8)
        ax_ni.set_ylabel('Neuroinflammation Index', fontsize=8)
        ax_ni.set_facecolor('#F8F9FA')

        ax_np.set_title(f'{diag} — Neuroprotection / Plasticity', fontsize=10, fontweight='bold')
        ax_np.set_xlabel('Time (days)', fontsize=8)
        ax_np.set_ylabel('Neuroprotection Index', fontsize=8)
        ax_np.set_facecolor('#F8F9FA')

    plt.tight_layout()
    plt.savefig(f'{outdir}/SuppFig08_qsp_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved SuppFig08_qsp_trajectories.png")


def plot_hybrid_improvement(results_df, outdir):
    """
    Panel figure showing improvement in R²/AUC from QSP features across
    all outcomes and models.
    """
    outcomes_reg = ['gose_12m', 'fim_total_12m', 'barthel_12m', 'mrs_12m', 'drs_12m',
                    'cog_composite_12m', 'moca_12m', 'hads_anxiety_12m',
                    'hads_depression_12m', 'phq9_12m', 'pcl5_12m',
                    'sf36_pcs_12m', 'sf36_mcs_12m', 'qolibri_os_12m']
    outcomes_cls = ['return_to_work_12m', 'mortality_12m']

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('QSP-Hybrid vs Clinical-Only Models — Performance Improvement',
                 fontsize=14, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    models = ['LASSO', 'Random Forest', 'XGBoost']
    palette = {'LASSO': '#2ECC71', 'Random Forest': '#3498DB', 'XGBoost': '#E74C3C'}

    # ── Panel 1: ΔR² per regression outcome (best model) ──────────────────
    ax1 = fig.add_subplot(gs[0, :])
    reg_res = results_df[results_df['Metric'] == 'R²'].copy()
    # Best model per outcome
    best_delta = reg_res.loc[reg_res.groupby('Outcome')['Delta_R2'].idxmax()]
    best_delta = best_delta[best_delta['Outcome'].isin(outcomes_reg)]
    best_delta = best_delta.set_index('Outcome').reindex(
        [o for o in outcomes_reg if o in best_delta.index])

    x = np.arange(len(best_delta))
    colors_bar = [palette.get(m, '#95A5A6') for m in best_delta['Model']]
    bars = ax1.bar(x, best_delta['Delta_R2'], color=colors_bar, edgecolor='white', linewidth=0.5, width=0.6)

    for bar, (_, row2) in zip(bars, best_delta.iterrows()):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                 f"+{h:.3f}" if h >= 0 else f"{h:.3f}",
                 ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax1.axhline(0, color='black', lw=0.8)
    ax1.set_xticks(x)
    xlabels = [o.replace('_12m', '').replace('_', ' ').upper() for o in best_delta.index]
    ax1.set_xticklabels(xlabels, rotation=35, ha='right', fontsize=8)
    ax1.set_ylabel('ΔR² (Hybrid − Clinical)', fontsize=10)
    ax1.set_title('Regression Outcomes: Best-Model ΔR² from QSP Feature Augmentation', fontsize=11)
    ax1.set_facecolor('#F8F9FA')

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, label=k) for k, v in palette.items()]
    ax1.legend(handles=legend_elements, fontsize=8, loc='upper right', title='Best Model')

    # ── Panel 2: Clinical vs Hybrid R² side-by-side for key outcomes ──────
    ax2 = fig.add_subplot(gs[1, 0])
    key_show = ['gose_12m', 'fim_total_12m', 'hads_anxiety_12m', 'cog_composite_12m',
                'pcl5_12m', 'qolibri_os_12m']
    lasso_res = reg_res[reg_res['Model'] == 'LASSO'].set_index('Outcome')
    avail_key = [k for k in key_show if k in lasso_res.index]

    x2 = np.arange(len(avail_key))
    w = 0.35
    clin_vals = [lasso_res.loc[o, 'Clinical_mean'] for o in avail_key]
    hyb_vals  = [lasso_res.loc[o, 'Hybrid_mean']   for o in avail_key]
    bars1 = ax2.bar(x2 - w/2, clin_vals, w, label='Clinical Only', color='#95A5A6', edgecolor='white')
    bars2 = ax2.bar(x2 + w/2, hyb_vals,  w, label='QSP Hybrid',    color='#2ECC71', edgecolor='white')
    ax2.set_xticks(x2)
    ax2.set_xticklabels([o.replace('_12m', '').replace('_', '\n').upper() for o in avail_key], fontsize=8)
    ax2.set_ylabel('5-Fold CV R²', fontsize=9)
    ax2.set_title('LASSO: Clinical vs QSP-Hybrid (Key Outcomes)', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.set_facecolor('#F8F9FA')

    # ── Panel 3: AUC classification ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    cls_res = results_df[results_df['Metric'] == 'AUC-ROC'].copy()
    if len(cls_res) > 0:
        cls_pivot = cls_res.groupby(['Outcome', 'Model'])[['Clinical_mean', 'Hybrid_mean']].first().reset_index()
        tgts_cls = cls_pivot['Outcome'].unique()
        x3 = np.arange(len(tgts_cls))
        for i, m in enumerate(['Logistic-L1', 'Random Forest', 'XGBoost']):
            sub = cls_pivot[cls_pivot['Model'] == m]
            if len(sub) == 0: continue
            ax3.bar(x3 + i * 0.25 - 0.25, sub['Hybrid_mean'].values, 0.25,
                    label=f'{m} (Hybrid)', color=list(palette.values())[i], alpha=0.85)
            ax3.bar(x3 + i * 0.25 - 0.25, sub['Clinical_mean'].values, 0.25,
                    color='none', edgecolor=list(palette.values())[i], linewidth=2, ls='--')
        ax3.set_xticks(x3)
        ax3.set_xticklabels([o.replace('_12m', '').replace('_', ' ').upper() for o in tgts_cls], fontsize=9)
        ax3.set_ylabel('AUC-ROC', fontsize=9)
        ax3.set_title('Classification: AUC-ROC (Solid=Hybrid, Dashed=Clinical)', fontsize=10)
        ax3.legend(fontsize=7)
        ax3.set_facecolor('#F8F9FA')

    # ── Panel 4: Domain-level improvement summary ──────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    domains = {
        'Functional':  ['gose_12m', 'fim_total_12m', 'barthel_12m', 'mrs_12m', 'drs_12m'],
        'Cognitive':   ['cog_composite_12m', 'moca_12m'],
        'Psychiatric': ['hads_anxiety_12m', 'hads_depression_12m', 'phq9_12m', 'pcl5_12m'],
        'QoL':         ['sf36_pcs_12m', 'sf36_mcs_12m', 'qolibri_os_12m']
    }
    domain_names = list(domains.keys())
    domain_clin = []
    domain_hyb  = []
    for dom, tgts in domains.items():
        sub = reg_res[(reg_res['Model'] == 'LASSO') & (reg_res['Outcome'].isin(tgts))]
        domain_clin.append(sub['Clinical_mean'].mean() if len(sub) else 0)
        domain_hyb.append(sub['Hybrid_mean'].mean()   if len(sub) else 0)

    x4 = np.arange(len(domain_names))
    ax4.bar(x4 - 0.2, domain_clin, 0.38, label='Clinical Only', color='#95A5A6')
    ax4.bar(x4 + 0.2, domain_hyb,  0.38, label='QSP Hybrid',    color='#2ECC71')
    for i, (c, h) in enumerate(zip(domain_clin, domain_hyb)):
        delta = h - c
        ax4.text(i + 0.2, h + 0.005, f'+{delta:.3f}' if delta >= 0 else f'{delta:.3f}',
                 ha='center', fontsize=9, fontweight='bold', color='#2ECC71' if delta > 0 else 'red')
    ax4.set_xticks(x4)
    ax4.set_xticklabels(domain_names, fontsize=10)
    ax4.set_ylabel('Mean Domain R² (LASSO)', fontsize=9)
    ax4.set_title('Domain-Level QSP Improvement (LASSO)', fontsize=10)
    ax4.legend(fontsize=9)
    ax4.set_facecolor('#F8F9FA')

    # ── Panel 5: Mechanistic feature correlation with outcomes ─────────────
    ax5 = fig.add_subplot(gs[2, 1])
    mech_cols_plot = ['mech_icp_peak', 'mech_icp_time_above_20', 'mech_cpp_min',
                      'mech_m1_peak', 'mech_ni_auc_7d', 'mech_ni_resolution_time',
                      'mech_np_steady_state', 'mech_recovery_potential',
                      'mech_secondary_injury_index', 'mech_m2_m1_dominance']
    outcomes_corr = ['gose_12m', 'hads_anxiety_12m', 'pcl5_12m', 'qolibri_os_12m']

    # Build correlation matrix from merged data (passed indirectly via closure)
    ax5.text(0.5, 0.5,
             'See fig16 for\nmechanistic trajectories\nand model architecture',
             ha='center', va='center', fontsize=12, transform=ax5.transAxes,
             fontweight='bold', color='#566573',
             bbox=dict(boxstyle='round', facecolor='#EBF5FB', alpha=0.8))
    ax5.axis('off')
    ax5.set_title('QSP Model Architecture Overview', fontsize=10)

    plt.savefig(f'{outdir}/SuppFig06_hybrid_improvement.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved SuppFig06_hybrid_improvement.png")


def plot_mech_feature_importance(coef_df, outdir):
    """LASSO coefficient plot distinguishing clinical vs mechanistic features."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('LASSO Feature Coefficients: Clinical vs Mechanistic Features (GOSE 12m)',
                 fontsize=13, fontweight='bold')

    top_n = 20
    top_feats = coef_df.head(top_n)

    colors = ['#E74C3C' if m else '#3498DB' for m in top_feats['Is_Mechanistic']]
    axes[0].barh(range(len(top_feats)), top_feats['LASSO_coef'],
                 color=colors, edgecolor='white', linewidth=0.5)
    axes[0].set_yticks(range(len(top_feats)))
    axes[0].set_yticklabels(
        [f.replace('mech_', '⚙ ').replace('_', ' ') for f in top_feats['Feature']],
        fontsize=9)
    axes[0].axvline(0, color='black', lw=0.8)
    axes[0].set_xlabel('LASSO Coefficient (standardised)', fontsize=10)
    axes[0].set_title(f'Top {top_n} Features by |Coefficient|', fontsize=10)
    axes[0].set_facecolor('#F8F9FA')

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor='#E74C3C', label='Mechanistic (QSP)'),
                    Patch(facecolor='#3498DB', label='Clinical/Acute')]
    axes[0].legend(handles=legend_elems, fontsize=9)

    # Pie chart: fraction of |coef| explained by mechanistic vs clinical
    mech_total  = coef_df[coef_df['Is_Mechanistic']]['LASSO_coef'].abs().sum()
    clin_total  = coef_df[~coef_df['Is_Mechanistic']]['LASSO_coef'].abs().sum()
    total       = mech_total + clin_total + 1e-9
    axes[1].pie([mech_total/total, clin_total/total],
                labels=['Mechanistic\n(QSP features)', 'Clinical\n(Acute features)'],
                colors=['#E74C3C', '#3498DB'],
                autopct='%1.1f%%', startangle=140,
                textprops={'fontsize': 11})
    axes[1].set_title('Share of Predictive Signal\n(LASSO |coefficients| for GOSE)', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{outdir}/SuppFig07_mech_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved SuppFig07_mech_feature_importance.png")


def plot_mech_feature_heatmap(df_merged, mech_feature_names, outdir):
    """Heatmap of mechanistic feature correlations with all outcome domains."""
    reg_outcomes = ['gose_12m', 'fim_total_12m', 'cog_composite_12m', 'moca_12m',
                    'hads_anxiety_12m', 'hads_depression_12m', 'pcl5_12m',
                    'sf36_pcs_12m', 'sf36_mcs_12m', 'qolibri_os_12m']

    avail_out = [o for o in reg_outcomes if o in df_merged.columns]
    avail_mch = [m for m in mech_feature_names if m in df_merged.columns]

    corr_matrix = pd.DataFrame(index=avail_mch, columns=avail_out, dtype=float)
    for mf in avail_mch:
        for oc in avail_out:
            pair = df_merged[[mf, oc]].dropna()
            if len(pair) > 10:
                corr_matrix.loc[mf, oc] = pair.corr().iloc[0, 1]

    corr_matrix = corr_matrix.astype(float)

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto',
                   vmin=-0.7, vmax=0.7)
    plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)

    ax.set_xticks(range(len(avail_out)))
    ax.set_xticklabels([o.replace('_12m', '').replace('_', ' ').upper()
                        for o in avail_out], rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(avail_mch)))
    ax.set_yticklabels([m.replace('mech_', '').replace('_', ' ') for m in avail_mch], fontsize=9)

    for i in range(len(avail_mch)):
        for j in range(len(avail_out)):
            val = corr_matrix.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color='white' if abs(val) > 0.4 else 'black')

    ax.set_title('Mechanistic Feature — Outcome Correlation Heatmap\n(Pearson r)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{outdir}/Figure1_mech_outcome_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved Figure1_mech_outcome_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUTDIR = os.path.dirname(os.path.abspath(__file__))
    DATA   = f'{OUTDIR}/simulated_neurocritical_cohort_n2000.csv'

    print("=" * 70)
    print("QSP-HYBRID ML PIPELINE FOR BRAIN INJURY OUTCOME PREDICTION")
    print("=" * 70)

    df = pd.read_csv(DATA)
    print(f"\nLoaded dataset: {df.shape[0]} patients × {df.shape[1]} variables")

    # ── Step 1: Run QSP simulations ──────────────────────────────────────────
    print("\n[STEP 1] Running patient-specific ODE simulations...")
    mech_df = run_qsp_simulation(df)

    # Align on patient_id
    df_indexed = df.set_index('patient_id')
    df_merged  = df_indexed.join(mech_df, how='inner')
    mech_feature_names = mech_df.columns.tolist()

    print(f"\nMerged dataset: {df_merged.shape[0]} patients × {df_merged.shape[1]} variables")
    print(f"Mechanistic features added: {len(mech_feature_names)}")
    print(f"  ICP/CPP features: {[f for f in mech_feature_names if 'icp' in f or 'cpp' in f or 'ar_' in f]}")
    print(f"  Neuro-inflammation: {[f for f in mech_feature_names if 'm1' in f or 'm2' in f or 'ni' in f or 'dam' in f]}")
    print(f"  Neuroprotection:    {[f for f in mech_feature_names if 'np' in f or 'recovery' in f or 'secondary' in f]}")

    # ── Step 2: Plot ODE trajectories ────────────────────────────────────────
    print("\n[STEP 2] Plotting QSP trajectories...")
    plot_sample_trajectories(df, OUTDIR)

    # ── Step 3: ML comparison ────────────────────────────────────────────────
    print("\n[STEP 3] Running Clinical-Only vs QSP-Hybrid ML comparison...")
    key_reg = ['gose_12m', 'fim_total_12m', 'barthel_12m', 'mrs_12m', 'drs_12m',
               'cog_composite_12m', 'moca_12m',
               'hads_anxiety_12m', 'hads_depression_12m', 'phq9_12m', 'pcl5_12m',
               'sf36_pcs_12m', 'sf36_mcs_12m', 'qolibri_os_12m']
    key_cls = ['return_to_work_12m', 'mortality_12m']

    results_df = run_comparison(df_merged, mech_feature_names, key_reg, key_cls)

    # ── Step 4: Mechanistic feature importance ───────────────────────────────
    print("\n[STEP 4] Computing LASSO feature importance with mechanistic features...")
    coef_df = get_feature_importance(df_merged, mech_feature_names, outcome='gose_12m')

    # ── Step 5: Visualisations ───────────────────────────────────────────────
    print("\n[STEP 5] Generating visualisations...")
    plot_hybrid_improvement(results_df, OUTDIR)
    plot_mech_feature_importance(coef_df, OUTDIR)
    plot_mech_feature_heatmap(df_merged, mech_feature_names, OUTDIR)

    # ── Step 6: Print results ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS: Clinical-Only vs QSP-Hybrid (5-Fold CV, LASSO)")
    print("=" * 80)
    lasso_res = results_df[results_df['Model'] == 'LASSO'].sort_values('Delta_R2', ascending=False)
    print(f"\n{'Outcome':<26} {'Clinical R²':>12} {'Hybrid R²':>12} {'ΔR²':>10} {'Δ%':>8}")
    print("-" * 72)
    for _, row2 in lasso_res.iterrows():
        met = row2['Metric']
        print(f"{row2['Outcome']:<26} "
              f"{row2['Clinical_mean']:>+12.3f} "
              f"{row2['Hybrid_mean']:>+12.3f} "
              f"{row2['Delta_R2']:>+10.3f} "
              f"{row2['Pct_improvement']:>+7.1f}%")

    # Domain summary
    domains = {
        'Functional':  ['gose_12m','fim_total_12m','barthel_12m','mrs_12m','drs_12m'],
        'Cognitive':   ['cog_composite_12m','moca_12m'],
        'Psychiatric': ['hads_anxiety_12m','hads_depression_12m','phq9_12m','pcl5_12m'],
        'QoL':         ['sf36_pcs_12m','sf36_mcs_12m','qolibri_os_12m']
    }
    print("\n" + "=" * 60)
    print("DOMAIN-LEVEL SUMMARY (LASSO)")
    print("=" * 60)
    print(f"{'Domain':<18} {'Clinical':>10} {'Hybrid':>10} {'ΔR²':>8}")
    print("-" * 50)
    for dom, tgts in domains.items():
        sub = lasso_res[lasso_res['Outcome'].isin(tgts) & (lasso_res['Metric'] == 'R²')]
        if len(sub):
            c = sub['Clinical_mean'].mean()
            h = sub['Hybrid_mean'].mean()
            print(f"{dom:<18} {c:>+10.3f} {h:>+10.3f} {h-c:>+8.3f}")

    # ── Step 7: Save outputs ─────────────────────────────────────────────────
    results_df.to_csv(f'{OUTDIR}/qsp_hybrid_comparison.csv', index=False)
    coef_df.to_csv(f'{OUTDIR}/qsp_lasso_coefficients.csv', index=False)

    # Save augmented dataset
    mech_save = df_merged[mech_feature_names].reset_index()
    mech_save.to_csv(f'{OUTDIR}/mechanistic_features_n2000.csv', index=False)

    print("\n── Saved files ──────────────────────────────────────────")
    print("  qsp_hybrid_comparison.csv")
    print("  qsp_lasso_coefficients.csv")
    print("  mechanistic_features_n2000.csv")
    print("  fig16_qsp_trajectories.png")
    print("  fig17_hybrid_improvement.png")
    print("  fig18_mech_feature_importance.png")
    print("  fig19_mech_outcome_heatmap.png")
    print("\nQSP-HYBRID PIPELINE COMPLETE.")


if __name__ == '__main__':
    main()
