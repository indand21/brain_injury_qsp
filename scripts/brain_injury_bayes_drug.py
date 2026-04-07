"""
brain_injury_bayes_drug.py
=============================================================================
Extension of the Brain Injury QSP Pipeline — Bayesian ODE Parameter
Estimation + Drug PK/PD Osmotherapy Submodel.

Two major extensions:
  1. Bayesian ODE parameter estimation (MAP + Laplace approximation)
       - Generates synthetic ICP waveforms per patient (respiration/cardiac noise)
       - Estimates 5 ODE parameters via MAP with physiological priors
       - Computes posterior uncertainty via numerical Hessian (Laplace approx)
       - Produces 19 Bayesian mechanistic features per patient

  2. Drug PK/PD submodel (osmotherapy)
       - Monroe-Kellie ODE extended to 7 state variables (+ osmolality, drugs)
       - Mannitol + HSS 1-compartment pharmacokinetics
       - 3 scenarios per patient: baseline | mannitol 0.5 g/kg | HSS 3%
       - Extracts 38 drug-response features per patient

Combined into a 5-way ML comparison:
  Clinical | + Det-QSP | + Bayes-ODE | + Drug-Response | + All

Outputs (OUTDIR):
  icp_summaries_n2000.npy
  bayes_ode_parameters_n2000.csv
  drug_response_features_n2000.csv
  extended_model_comparison.csv
  fig20_posterior_parameters.png
  fig21_bayesian_outcome_heatmap.png
  fig22_extended_ml_comparison.png
  fig23_icp_drug_trajectories.png
  fig24_osmolality_icp_coupling.png
  fig25_drug_response_phenotypes.png

Runtime: ~1-2 hours (Bayesian MAP+Laplace and drug scenarios are parallelised)
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
from scipy.optimize import minimize, approx_fprime
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

OUTDIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)
N_JOBS      = -1    # use all available cores
DPI         = 150   # figure resolution

# 5 ODE parameters estimated by Bayesian MAP
PARAM_NAMES  = ['AR_index', 'C0_brain', 'R_csf', 'k_bbb', 'k_edema_clear']
PARAM_BOUNDS = [(0.05, 0.99), (0.10, 2.50), (0.50, 50.0), (0.001, 0.60), (0.001, 0.25)]

# Prior information for each parameter
PARAM_PRIORS = {
    'AR_index':      {'type': 'uniform', 'mean': 0.50, 'sd': 0.25},
    'C0_brain':      {'type': 'normal',  'mean': 0.80, 'sd': 0.25},
    'R_csf':         {'type': 'halfnormal', 'mean': 0.0, 'sd': 8.00},
    'k_bbb':         {'type': 'halfnormal', 'mean': 0.0, 'sd': 0.05},
    'k_edema_clear': {'type': 'halfnormal', 'mean': 0.0, 'sd': 0.02},
}

# Time points for MAP likelihood (28 × 6-hour window midpoints, hours)
T_EVAL_BAYES = np.array([3.0 + 6.0 * w for w in range(28)])

# Diagnosis colours
DIAG_COLORS = {'TBI': '#E74C3C', 'SAH': '#9B59B6', 'Stroke': '#3498DB', 'ICH': '#E67E22'}

# ML outcome definitions
REG_OUTCOMES = [
    'gose_12m', 'fim_total_12m', 'barthel_12m',
    'hads_anxiety_12m', 'moca_12m', 'qolibri_os_12m', 'drs_12m',
]
CLF_OUTCOMES = ['return_to_work_12m', 'mortality_12m']

# All longitudinal outcome columns to exclude from clinical feature set
_ALL_OUTCOME_COLS = {
    'gose_3m', 'gose_6m', 'gose_12m',
    'mrs_3m', 'mrs_6m', 'mrs_12m',
    'fim_total_3m', 'fim_total_6m', 'fim_total_12m',
    'barthel_3m', 'barthel_6m', 'barthel_12m',
    'drs_3m', 'drs_6m', 'drs_12m',
    'moca_3m', 'moca_6m', 'moca_12m',
    'hads_anxiety_3m', 'hads_anxiety_6m', 'hads_anxiety_12m',
    'hads_depression_3m', 'hads_depression_6m', 'hads_depression_12m',
    'phq9_3m', 'phq9_6m', 'phq9_12m',
    'gad7_3m', 'gad7_6m', 'gad7_12m',
    'pcl5_3m', 'pcl5_6m', 'pcl5_12m',
    'sf36_pcs_3m', 'sf36_pcs_6m', 'sf36_pcs_12m',
    'sf36_mcs_3m', 'sf36_mcs_6m', 'sf36_mcs_12m',
    'qolibri_os_3m', 'qolibri_os_6m', 'qolibri_os_12m',
    'mpai4_tscore_3m', 'mpai4_tscore_6m', 'mpai4_tscore_12m',
    'return_to_work_3m', 'return_to_work_6m', 'return_to_work_12m',
    'social_participation_3m', 'social_participation_6m', 'social_participation_12m',
    'cog_memory_3m', 'cog_memory_6m', 'cog_memory_12m',
    'cog_executive_3m', 'cog_executive_6m', 'cog_executive_12m',
    'cog_attention_3m', 'cog_attention_6m', 'cog_attention_12m',
    'cog_visuoconst_3m', 'cog_visuoconst_6m', 'cog_visuoconst_12m',
    'cog_composite_3m', 'cog_composite_6m', 'cog_composite_12m',
    'mortality_12m', 'trajectory_class', 'patient_id',
}

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — ODE HELPERS (verbatim from brain_injury_qsp_hybrid.py lines 39–193)
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
    AR_index  = params['AR_index']
    CPP_opt   = 70.0
    CPP_range = 20.0
    CBF_normal = params['CBF_normal']

    CBF_autoregulated  = CBF_normal / (1 + np.exp(-(CPP - CPP_opt) / CPP_range))
    CBF_pressure_passive = CBF_normal * CPP / 80.0
    CBF = (1 - AR_index) * CBF_autoregulated + AR_index * CBF_pressure_passive
    CBF = max(CBF, 0)

    # Brain compliance (inverse relationship with ICP)
    ICP_ref  = 10.0
    C_brain  = params['C0_brain'] / (1 + params['k_elastance'] * max(ICP - ICP_ref, 0))
    C_brain  = max(C_brain, 0.01)

    # CSF dynamics (Davson's equation)
    I_formation  = 0.35
    P_venous     = 6.0
    R_abs        = params['R_csf']
    I_absorption = max(ICP - P_venous, 0) / R_abs

    dV_csf_dt = I_formation - I_absorption

    # Vasogenic oedema (BBB disruption; peaks ~12-24h, resolves over days)
    k_bbb         = params['k_bbb']
    k_edema_clear = params['k_edema_clear']
    dV_edema_dt   = k_bbb * np.exp(-k_bbb * t / 24.0) - k_edema_clear * V_edema

    dV_total  = dV_edema_dt + dV_csf_dt
    dICP_dt   = dV_total / C_brain

    return [dICP_dt, dV_edema_dt, dV_csf_dt]


def get_icp_params(row):
    """Personalise ICP/CPP ODE parameters from patient clinical variables."""
    gcs    = float(row.get('gcs_admission', 10))
    apache = float(row.get('apache_ii', 20))
    diag   = str(row.get('diagnosis', 'TBI'))
    icp_m  = float(row.get('icp_mean_mmhg', 15))
    age    = float(row.get('age', 55))

    MAP = max(70 - (apache - 20) * 0.4, 50)

    AR_index        = np.clip(1.0 - (gcs - 3) / 12.0, 0.1, 0.95)
    severity_factor = np.clip((15 - gcs) / 12.0, 0.1, 1.0)
    C0_brain        = 0.8 * (1 - 0.4 * severity_factor) * (1 - 0.003 * max(age - 40, 0))

    diag_factor  = {'TBI': 1.3, 'SAH': 1.5, 'Stroke': 1.1, 'ICH': 1.4}.get(diag, 1.2)
    R_csf        = 8.0 * diag_factor
    k_bbb        = 0.05 * severity_factor * diag_factor
    k_edema_clear = 0.02 * (1 - 0.003 * max(age - 40, 0))

    return {
        'MAP':          MAP,
        'AR_index':     AR_index,
        'CBF_normal':   50.0,
        'C0_brain':     C0_brain,
        'k_elastance':  0.12,
        'R_csf':        R_csf,
        'k_bbb':        k_bbb,
        'k_edema_clear': k_edema_clear,
        'ICP_init':     icp_m,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — SYNTHETIC ICP WAVEFORM GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_icp_summary(row):
    """
    Generate a 28-point ICP waveform summary for one patient.

    Solves Monroe-Kellie ODE at ~1-min resolution (0–168 h), adds
    physiological noise (respiratory, cardiac pulsatility, Gaussian
    measurement noise), then aggregates into 28 × 6-hour window means.

    Returns: np.ndarray of shape (28,)
    """
    params   = get_icp_params(row)
    ICP_init = params['ICP_init']
    t_eval   = np.linspace(0, 168, 10081)  # 1 point per ~1 min

    y0 = [ICP_init, 0.01, 100.0]
    try:
        sol = solve_ivp(icp_cpp_ode, (0, 168), y0, args=(params,),
                        method='RK45', t_eval=t_eval,
                        rtol=1e-4, atol=1e-6, max_step=0.2)
        if not sol.success:
            return np.full(28, ICP_init)
        ICP_ode = np.clip(sol.y[0], 0, 80)
        t       = sol.t
    except Exception:
        return np.full(28, ICP_init)

    # Physiological noise
    anxiety     = float(row.get('anxiety_icu_score', 5))
    HR          = max(60.0 + anxiety * 5.0, 40.0)             # bpm
    noise_resp  = 1.0 * np.sin(2 * np.pi * 0.25 * t)          # respiratory (~0.25 Hz)
    noise_card  = (2.0 + 0.1 * ICP_ode) * np.sin(2 * np.pi * (HR / 60.0) * t)
    noise_meas  = np.random.normal(0, 0.5, len(t))             # Gaussian measurement

    ICP_total = np.clip(ICP_ode + noise_resp + noise_card + noise_meas, 0, 80)

    # Aggregate to 28 × 6-hour windows
    n = len(t_eval)
    ppw = n // 28
    icp_summary = np.zeros(28)
    for w in range(28):
        s = w * ppw
        e = s + ppw if w < 27 else n
        icp_summary[w] = np.mean(ICP_total[s:e])

    return icp_summary


def generate_all_waveforms(df):
    """Generate and cache 28-point ICP summaries for all patients."""
    npy_path = os.path.join(OUTDIR, 'icp_summaries_n2000.npy')
    if os.path.exists(npy_path):
        print("  Loading cached ICP waveform summaries...")
        return np.load(npy_path)

    print(f"  Generating ICP waveform summaries for {len(df)} patients...")
    summaries = np.zeros((len(df), 28))
    for i, (_, row) in enumerate(df.iterrows()):
        summaries[i] = generate_icp_summary(row)
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(df)}")

    np.save(npy_path, summaries)
    print(f"  Saved: {npy_path}")
    return summaries


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — MAP + LAPLACE ODE PARAMETER ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def _solve_icp_at_bayes_times(theta, base_params):
    """Solve ICP ODE at T_EVAL_BAYES for a given parameter vector theta."""
    AR_index, C0_brain, R_csf, k_bbb, k_edema_clear = theta
    params = dict(base_params)
    params.update({
        'AR_index':      AR_index,
        'C0_brain':      max(C0_brain, 0.01),
        'R_csf':         max(R_csf, 0.1),
        'k_bbb':         max(k_bbb, 1e-4),
        'k_edema_clear': max(k_edema_clear, 1e-4),
    })
    y0 = [params['ICP_init'], 0.01, 100.0]
    try:
        sol = solve_ivp(icp_cpp_ode, (0, 168), y0, args=(params,),
                        method='RK45', t_eval=T_EVAL_BAYES,
                        rtol=1e-3, atol=1e-5, max_step=2.0)
        if sol.success:
            return np.clip(sol.y[0], 0, 80)
    except Exception:
        pass
    return None


def build_nlp(icp_obs, base_params):
    """
    Return the negative log-posterior function for MAP optimisation.

    Log-posterior = log-likelihood + log-prior
      Likelihood : Normal(icp_obs | ICP_ODE(θ), σ=2 mmHg)
      Priors     : see PARAM_PRIORS (AR_index uniform, others Normal/HalfNormal)
    """
    obs_sigma = 2.0   # mmHg

    def nlp(theta):
        AR_index, C0_brain, R_csf, k_bbb, k_edema_clear = theta
        # Hard bounds (Powell ignores bounds — enforce via penalty)
        if not (0.05 <= AR_index <= 0.99 and
                0.05 <= C0_brain <= 2.50 and
                0.50 <= R_csf    <= 50.0 and
                0.001 <= k_bbb   <= 0.60 and
                0.001 <= k_edema_clear <= 0.25):
            return 1e10

        icp_pred = _solve_icp_at_bayes_times(theta, base_params)
        if icp_pred is None:
            return 1e10

        residuals   = icp_obs - icp_pred
        log_lik     = -0.5 * np.sum(residuals ** 2) / (obs_sigma ** 2)

        # Priors: AR_index uniform (no contribution), others regularise
        log_prior_C0  = -0.5 * ((C0_brain - 0.80) / 0.25) ** 2
        log_prior_Rcsf = -0.5 * (R_csf / 8.0) ** 2
        log_prior_kbbb = -0.5 * (k_bbb / 0.05) ** 2
        log_prior_kec  = -0.5 * (k_edema_clear / 0.02) ** 2

        return -(log_lik + log_prior_C0 + log_prior_Rcsf + log_prior_kbbb + log_prior_kec)

    return nlp


def numerical_hessian(f, x, eps=2e-4):
    """Symmetric numerical Hessian via central differences of gradient."""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        gp = approx_fprime(xp, f, eps)
        gm = approx_fprime(xm, f, eps)
        H[i] = (gp - gm) / (2.0 * eps)
    return (H + H.T) / 2.0   # symmetrise


def _estimate_one_patient(i, row_dict, icp_obs):
    """MAP + Laplace estimation for one patient. Called in parallel."""
    base_params = get_icp_params(row_dict)
    x0 = [
        base_params['AR_index'],
        base_params['C0_brain'],
        base_params['R_csf'],
        base_params['k_bbb'],
        base_params['k_edema_clear'],
    ]

    nlp = build_nlp(icp_obs, base_params)

    # --- MAP via Powell ---
    try:
        result   = minimize(nlp, x0, method='Powell',
                            options={'maxiter': 500, 'ftol': 1e-5, 'xtol': 1e-5})
        theta_map = np.array(result.x)
        for j, (lo, hi) in enumerate(PARAM_BOUNDS):
            theta_map[j] = np.clip(theta_map[j], lo, hi)
    except Exception:
        theta_map = np.array(x0)

    # --- Laplace approximation (posterior covariance via Hessian) ---
    try:
        H   = numerical_hessian(nlp, theta_map)
        H  += 1e-4 * np.eye(5)                  # ridge for numerical stability
        cov = np.linalg.inv(H)
        variances = np.abs(np.diag(cov))
    except Exception:
        # Fallback: small default variances matching prior SD²
        variances = np.array([0.01, 0.004, 1.0, 2e-5, 8e-6])

    sds = np.sqrt(variances)

    # Cap SDs at physiologically meaningful bounds
    # (Hessian can be near-singular for non-identifiable parameters)
    max_sds = np.array([0.45, 0.50, 10.0, 0.15, 0.05])   # AR, C0, Rcsf, kbbb, kec
    min_sds = np.array([0.002, 0.005, 0.10, 5e-4, 2e-4])
    sds     = np.clip(sds, min_sds, max_sds)

    return i, theta_map, sds


def run_bayesian_estimation(df, icp_summaries):
    """
    Run MAP + Laplace for all n=2000 patients in parallel.
    Returns DataFrame with 19 Bayesian features per patient.
    """
    csv_path = os.path.join(OUTDIR, 'bayes_ode_parameters_n2000.csv')
    if os.path.exists(csv_path):
        print("  Loading cached Bayesian ODE parameters...")
        return pd.read_csv(csv_path)

    print(f"  Running MAP + Laplace for {len(df)} patients (parallel, ~15-30 min)...")
    rows    = [r.to_dict() for _, r in df.iterrows()]
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(_estimate_one_patient)(i, rows[i], icp_summaries[i])
        for i in range(len(df))
    )
    results.sort(key=lambda x: x[0])

    theta_maps = np.array([r[1] for r in results])
    theta_sds  = np.array([r[2] for r in results])

    # --- Build feature DataFrame ---
    prior_means = dict(zip(PARAM_NAMES, [0.50, 0.80, 8.0, 0.05, 0.02]))
    prior_sds   = dict(zip(PARAM_NAMES, [0.25, 0.25, 8.0, 0.05, 0.02]))

    records = []
    for i in range(len(df)):
        rec = {}
        for j, pname in enumerate(PARAM_NAMES):
            rec[f'bayes_{pname}_mean']      = theta_maps[i, j]
            rec[f'bayes_{pname}_sd']        = theta_sds[i, j]
            rec[f'bayes_{pname}_ci95_width'] = 3.92 * theta_sds[i, j]   # 2 × 1.96

        # --- Composite Bayesian features ---
        rec['bayes_param_uncertainty_index']  = float(np.mean(theta_sds[i]))
        rec['bayes_ar_deviation_from_prior']  = (
            abs(theta_maps[i, 0] - prior_means['AR_index']) / prior_sds['AR_index']
        )
        rec['bayes_compliance_recovery_ratio'] = (
            theta_maps[i, 1] / max(theta_maps[i, 2], 0.01)    # C0_brain / R_csf
        )
        rec['bayes_bbb_edema_ratio'] = (
            theta_maps[i, 3] / max(theta_maps[i, 4], 1e-5)    # k_bbb / k_edema_clear
        )
        records.append(rec)

    bayes_df = pd.DataFrame(records)
    bayes_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    return bayes_df


# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — DRUG PK/PD ODE SYSTEM (extended Monroe-Kellie, 7 state variables)
# ─────────────────────────────────────────────────────────────────────────────

def get_drug_input(t, drug_events, drug_type):
    """Return drug infusion rate at time t.
    Mannitol: mg/h   |   HSS: mEq Na⁺/h
    """
    total = 0.0
    for ev in drug_events.get(drug_type, []):
        t0 = ev['time']
        t1 = t0 + ev['infusion_time']
        if t0 <= t <= t1:
            total += ev['dose'] / ev['infusion_time']
    return total


def icp_cpp_osmotic_ode(t, y, params, drug_events):
    """
    Extended Monroe-Kellie ODE with osmotherapy PK/PD.

    State variables (7):
        y[0] = ICP (mmHg)
        y[1] = V_edema (mL)
        y[2] = V_csf (mL)
        y[3] = Osm_total (mOsm/kg)   — serum osmolality
        y[4] = Osm_brain (mOsm/kg)   — brain osmolality (slow BBB equilibration)
        y[5] = A_mannitol (mg)        — plasma mannitol amount
        y[6] = C_sodium (mEq/L)       — serum sodium concentration
    """
    ICP, V_edema, V_csf, Osm_total, Osm_brain, A_mannitol, C_sodium = y

    # ── Haemodynamics and autoregulation ──────────────────────────────────
    MAP          = params['MAP']
    CPP          = max(MAP - ICP, 0)
    AR_index     = params['AR_index']
    CBF_normal   = params['CBF_normal']

    CBF_ar = CBF_normal / (1 + np.exp(-(CPP - 70.0) / 20.0))
    CBF_pp = CBF_normal * CPP / 80.0
    CBF    = max((1 - AR_index) * CBF_ar + AR_index * CBF_pp, 0)

    C_brain = params['C0_brain'] / (1 + params['k_elastance'] * max(ICP - 10.0, 0))
    C_brain = max(C_brain, 0.01)

    # ── CSF dynamics (Davson) ─────────────────────────────────────────────
    I_formation  = 0.35
    P_venous     = 6.0
    I_absorption = max(ICP - P_venous, 0) / max(params['R_csf'], 0.01)
    dV_csf_dt    = I_formation - I_absorption

    # ── Vasogenic oedema (modified: osmotic reabsorption term) ───────────
    k_bbb         = params['k_bbb']
    k_edema_clear = params['k_edema_clear']
    osm_gradient  = max(Osm_total - Osm_brain, 0.0)
    dV_edema_dt   = (k_bbb * np.exp(-k_bbb * t / 24.0)
                     - k_edema_clear * V_edema
                     - 0.01 * osm_gradient)

    # ── ICP ───────────────────────────────────────────────────────────────
    dICP_dt = (dV_edema_dt + dV_csf_dt) / C_brain

    # ── Brain osmolality (slow BBB equilibration ~24 h) ───────────────────
    dOsm_brain_dt = 0.2 * (Osm_total - Osm_brain) / 24.0

    # ── Mannitol PK (1-compartment, renal clearance CL=5 L/h) ───────────
    weight_kg      = params.get('weight_kg', 70.0)
    V_plasma_L     = 0.2 * weight_kg           # L  (Vd ≈ 0.2 L/kg)
    CL_man_L_h     = 5.0                        # L/h renal CL
    A_mannitol     = max(A_mannitol, 0.0)
    C_man_mg_L     = A_mannitol / V_plasma_L    # mg/L
    C_man_g_L      = C_man_mg_L / 1000.0        # g/L

    Input_man_mg_h  = get_drug_input(t, drug_events, 'mannitol')
    dA_mannitol_dt  = Input_man_mg_h - CL_man_L_h * C_man_mg_L
    dC_man_g_L_dt   = dA_mannitol_dt / (1000.0 * V_plasma_L)   # g/L/h

    # ── Sodium PK (return-to-baseline model) ─────────────────────────────
    Vd_Na_L        = 0.6 * weight_kg           # L
    k_Na_return    = 0.10                       # h⁻¹ (returns to 140 over ~10h)
    Input_Na_mEq_h = get_drug_input(t, drug_events, 'hss')
    dC_sodium_dt   = Input_Na_mEq_h / Vd_Na_L - k_Na_return * (C_sodium - 140.0)

    # ── Serum osmolality ──────────────────────────────────────────────────
    # 5.46 ≈ 1000/182 (mannitol MW g/mol) → g/L to mOsm/kg
    # 1.86 = Van't Hoff for Na⁺ + counterion pair (osmolality per mEq/L Na⁺)
    dOsm_total_dt = (-0.15 * (Osm_total - 290.0)
                     + 5.46 * dC_man_g_L_dt
                     + 1.86 * dC_sodium_dt)

    return [dICP_dt, dV_edema_dt, dV_csf_dt,
            dOsm_total_dt, dOsm_brain_dt,
            dA_mannitol_dt, dC_sodium_dt]


# ─────────────────────────────────────────────────────────────────────────────
# PART 5 — DRUG SCENARIO SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

_DRUG_SCENARIOS = {
    'baseline': {'mannitol': [], 'hss': []},
    'mannitol': {
        'mannitol': [{'time': 1.0, 'dose': 35000.0, 'infusion_time': 0.5}],   # 35 g over 30 min
        'hss':      [],
    },
    'hss': {
        'mannitol': [],
        'hss':      [{'time': 1.0, 'dose': 77.0, 'infusion_time': 0.33}],      # 77 mEq over 20 min
    },
}

_T_DRUG_EVAL = np.linspace(0, 168, 500)


def _sim_one_patient_drug(i, row_dict, bayes_row_dict):
    """Simulate 3 drug scenarios for one patient. Called in parallel."""
    params = get_icp_params(row_dict)

    # Override with Bayesian MAP estimates where available
    for pname in PARAM_NAMES:
        col = f'bayes_{pname}_mean'
        val = bayes_row_dict.get(col, np.nan)
        if val is not None and not np.isnan(float(val)):
            params[pname] = float(val)

    icp_init = params['ICP_init']
    y0       = [icp_init, 0.01, 100.0, 290.0, 290.0, 0.0, 140.0]

    sols = {}
    for scen, drug_ev in _DRUG_SCENARIOS.items():
        try:
            sol = solve_ivp(
                icp_cpp_osmotic_ode, (0, 168), y0,
                args=(params, drug_ev),
                method='RK45', t_eval=_T_DRUG_EVAL,
                rtol=1e-4, atol=1e-6, max_step=1.0,
            )
            sols[scen] = sol if sol.success else None
        except Exception:
            sols[scen] = None

    return i, sols, _T_DRUG_EVAL


# ─────────────────────────────────────────────────────────────────────────────
# PART 6 — DRUG RESPONSE FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

_DRUG_FEAT_KEYS = [
    'baseline_icp_pre_drug', 'icp_nadir_post_drug',
    'icp_reduction_absolute', 'icp_reduction_pct',
    'time_to_icp_control_20', 'icp_at_6h', 'icp_at_24h',
    'rebound_magnitude', 'has_rebound',
    'osmotic_gradient_peak', 'osmotic_gradient_auc',
    'icp_reduction_per_osm',
]


def extract_drug_features(sol, t_eval, drug_time=1.0):
    """Extract 12 physiological response features from one ODE solution."""
    nan_dict = {k: np.nan for k in _DRUG_FEAT_KEYS}
    if sol is None or not sol.success:
        return nan_dict

    t         = t_eval
    ICP       = np.clip(sol.y[0], 0, 80)
    Osm_total = sol.y[3]
    Osm_brain = sol.y[4]

    # Pre-drug baseline ICP
    pre_mask    = t < drug_time
    baseline_icp = float(np.mean(ICP[pre_mask])) if np.any(pre_mask) else float(ICP[0])

    # Nadir within 2 h of drug administration
    nadir_mask = (t >= drug_time) & (t <= drug_time + 2.0)
    icp_nadir  = float(np.min(ICP[nadir_mask])) if np.any(nadir_mask) else baseline_icp

    icp_red_abs = baseline_icp - icp_nadir
    icp_red_pct = icp_red_abs / max(baseline_icp, 1.0) * 100.0

    # Time until ICP < 20 mmHg
    below20      = t[ICP < 20.0]
    time_to_ctrl = float(below20[0]) if len(below20) > 0 else 168.0

    # ICP at 6 h and 24 h
    icp_at_6h  = float(ICP[np.argmin(np.abs(t - 6.0))])
    icp_at_24h = float(ICP[np.argmin(np.abs(t - 24.0))])

    # Rebound (4–8 h after drug)
    reb_mask   = (t >= drug_time + 4.0) & (t <= drug_time + 8.0)
    rebound_m  = float(max(np.max(ICP[reb_mask]) - icp_nadir, 0.0)) if np.any(reb_mask) else 0.0
    has_rebound = float(rebound_m > 5.0)

    # Osmotic gradient features
    osm_grad  = np.maximum(Osm_total - Osm_brain, 0.0)
    osm_peak  = float(np.max(osm_grad))
    osm_auc   = float(np.trapz(osm_grad, t))
    icp_per_osm = icp_red_abs / max(osm_peak, 0.1)

    return {
        'baseline_icp_pre_drug':   baseline_icp,
        'icp_nadir_post_drug':     icp_nadir,
        'icp_reduction_absolute':  icp_red_abs,
        'icp_reduction_pct':       icp_red_pct,
        'time_to_icp_control_20':  time_to_ctrl,
        'icp_at_6h':               icp_at_6h,
        'icp_at_24h':              icp_at_24h,
        'rebound_magnitude':       rebound_m,
        'has_rebound':             has_rebound,
        'osmotic_gradient_peak':   osm_peak,
        'osmotic_gradient_auc':    osm_auc,
        'icp_reduction_per_osm':   icp_per_osm,
    }


def simulate_and_extract_drug_features(df, bayes_df):
    """Run all drug scenarios, extract features. Returns (drug_df, sample_sols)."""
    csv_path = os.path.join(OUTDIR, 'drug_response_features_n2000.csv')
    if os.path.exists(csv_path):
        print("  Loading cached drug response features...")
        return pd.read_csv(csv_path), {}

    print(f"  Simulating drug scenarios for {len(df)} patients (parallel, ~10-20 min)...")
    rows       = [r.to_dict() for _, r in df.iterrows()]
    bayes_rows = [r.to_dict() for _, r in bayes_df.iterrows()]

    all_results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(_sim_one_patient_drug)(i, rows[i], bayes_rows[i])
        for i in range(len(df))
    )
    all_results.sort(key=lambda x: x[0])

    records     = []
    sample_sols = {}   # store first 9 patients for figures

    for i, sols, t_eval in all_results:
        rec = {}
        for scen in ['baseline', 'mannitol', 'hss']:
            feats = extract_drug_features(sols.get(scen), t_eval)
            for k, v in feats.items():
                rec[f'{scen}_{k}'] = v

        # Composite: preferred drug
        man_red = rec.get('mannitol_icp_reduction_absolute') or 0.0
        hss_red = rec.get('hss_icp_reduction_absolute') or 0.0
        rec['drug_response_optimal'] = max(man_red, hss_red)
        if   man_red > hss_red and man_red > 2.0:
            rec['drug_type_preferred'] = 'mannitol'
        elif hss_red > man_red and hss_red > 2.0:
            rec['drug_type_preferred'] = 'hss'
        else:
            rec['drug_type_preferred'] = 'neither'
        records.append(rec)

        if i < 9:
            sample_sols[i] = {'sols': sols, 't_eval': t_eval}

    drug_df = pd.DataFrame(records)
    drug_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    return drug_df, sample_sols


# ─────────────────────────────────────────────────────────────────────────────
# PART 7 — EXTENDED 5-WAY ML COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def get_clinical_features(df):
    """Return list of numerical clinical baseline feature column names."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in _ALL_OUTCOME_COLS]


def _cv_regression(X, y, n_splits=5):
    """5-fold CV with LASSO and RF. Returns (mean_lasso_R², mean_rf_R²)."""
    kf     = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()
    lasso_scores, rf_scores = [], []

    for tr, te in kf.split(X):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        Xs_tr = scaler.fit_transform(X_tr)
        Xs_te = scaler.transform(X_te)
        m = LassoCV(cv=3, max_iter=5000, random_state=42, n_jobs=1)
        m.fit(Xs_tr, y_tr)
        lasso_scores.append(float(m.score(Xs_te, y_te)))

        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        rf.fit(X_tr, y_tr)
        rf_scores.append(float(rf.score(X_te, y_te)))

    return float(np.mean(lasso_scores)), float(np.mean(rf_scores))


def _cv_classification(X, y, n_splits=5):
    """5-fold stratified CV. Returns (mean_logreg_AUC, mean_rf_AUC)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()
    lr_aucs, rf_aucs = [], []

    for tr, te in skf.split(X, y):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        if len(np.unique(y_te)) < 2:
            continue

        Xs_tr = scaler.fit_transform(X_tr)
        Xs_te = scaler.transform(X_te)
        lr = LogisticRegressionCV(cv=3, max_iter=1000, random_state=42, n_jobs=1)
        lr.fit(Xs_tr, y_tr)
        lr_aucs.append(roc_auc_score(y_te, lr.predict_proba(Xs_te)[:, 1]))

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        rf.fit(X_tr, y_tr)
        rf_aucs.append(roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1]))

    return (float(np.mean(lr_aucs)) if lr_aucs else np.nan,
            float(np.mean(rf_aucs)) if rf_aucs else np.nan)


def run_extended_comparison(df, bayes_df, drug_df, mech_df):
    """5-way ML comparison across all outcomes. Returns results DataFrame."""
    csv_path = os.path.join(OUTDIR, 'extended_model_comparison.csv')
    if os.path.exists(csv_path):
        print("  Loading cached extended ML comparison...")
        return pd.read_csv(csv_path)

    print("  Running 5-way extended ML comparison (~20-40 min)...")

    # --- Define feature pools ---
    clin_cols  = get_clinical_features(df)
    mech_cols  = [c for c in mech_df.columns if c.startswith('mech_')] if mech_df is not None else []
    bayes_cols = [c for c in bayes_df.columns]
    drug_cols  = [c for c in drug_df.select_dtypes(include=[np.number]).columns]

    # Merge all features into one wide DataFrame
    dm = df.reset_index(drop=True).copy()
    if mech_df is not None:
        dm = pd.concat([dm, mech_df.reset_index(drop=True)[mech_cols]], axis=1)
    dm = pd.concat([dm, bayes_df.reset_index(drop=True)], axis=1)
    dm = pd.concat([dm, drug_df.reset_index(drop=True)[drug_cols]], axis=1)

    feature_sets = {
        'Clinical':          clin_cols,
        'Clinical+DetQSP':   clin_cols + mech_cols,
        'Clinical+BayesODE': clin_cols + bayes_cols,
        'Clinical+Drug':     clin_cols + drug_cols,
        'Clinical+All':      list(dict.fromkeys(clin_cols + mech_cols + bayes_cols + drug_cols)),
    }

    records = []
    for outcome in REG_OUTCOMES + CLF_OUTCOMES:
        if outcome not in dm.columns:
            continue
        y_raw = dm[outcome].values
        valid  = np.isfinite(y_raw)
        if valid.sum() < 100:
            continue
        y_cv = y_raw[valid]

        for fs_name, fs_cols in feature_sets.items():
            avail = [c for c in fs_cols if c in dm.columns]
            X_raw = dm[avail].values[valid]
            X_cv  = np.nan_to_num(X_raw, nan=0.0)

            if outcome in CLF_OUTCOMES:
                y_bin = (y_cv > 0).astype(int)
                lr_auc, rf_auc = _cv_classification(X_cv, y_bin)
                records.append({'outcome': outcome, 'feature_set': fs_name,
                                 'model': 'LogReg', 'metric': 'AUC', 'value': lr_auc})
                records.append({'outcome': outcome, 'feature_set': fs_name,
                                 'model': 'RF',    'metric': 'AUC', 'value': rf_auc})
            else:
                lasso_r2, rf_r2 = _cv_regression(X_cv, y_cv)
                records.append({'outcome': outcome, 'feature_set': fs_name,
                                 'model': 'LASSO', 'metric': 'R2', 'value': lasso_r2})
                records.append({'outcome': outcome, 'feature_set': fs_name,
                                 'model': 'RF',    'metric': 'R2', 'value': rf_r2})
        print(f"    Done: {outcome}")

    results_df = pd.DataFrame(records)
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# PART 8 — VISUALISATIONS (fig20–fig25)
# ─────────────────────────────────────────────────────────────────────────────

def plot_posterior_parameters(df, bayes_df, outdir=OUTDIR):
    """
    fig20: 5-panel violin plot of Bayesian MAP estimates by diagnosis.
    Shows MAP mean, 95% CI width, and dashed prior mean for each parameter.
    """
    diags = ['TBI', 'SAH', 'Stroke', 'ICH']
    diag_col = df['diagnosis'].values

    param_labels = {
        'AR_index':      'AR Index\n(0=intact, 1=impaired)',
        'C0_brain':      'C₀ Brain\n(mL/mmHg)',
        'R_csf':         'R_CSF\n(mmHg·h/mL)',
        'k_bbb':         'k_BBB\n(h⁻¹)',
        'k_edema_clear': 'k_edema_clear\n(h⁻¹)',
    }
    prior_means = {'AR_index': 0.50, 'C0_brain': 0.80,
                   'R_csf': 8.0, 'k_bbb': 0.05, 'k_edema_clear': 0.02}

    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
    fig.suptitle(
        'Fig 20: Bayesian ODE Parameter Estimates by Diagnosis\n'
        'MAP via Powell optimisation + Laplace approximation (n=2,000)',
        fontsize=13, fontweight='bold', y=1.02
    )

    for ax, pname in zip(axes, PARAM_NAMES):
        col = f'bayes_{pname}_mean'
        data = [bayes_df.loc[diag_col == d, col].dropna().values for d in diags]

        vp = ax.violinplot(data, positions=range(4), showmedians=True,
                           showextrema=False, widths=0.7)
        for pc, d in zip(vp['bodies'], diags):
            pc.set_facecolor(DIAG_COLORS[d])
            pc.set_alpha(0.65)
        vp['cmedians'].set_color('black')
        vp['cmedians'].set_linewidth(2)

        ax.axhline(prior_means[pname], color='#555555', ls='--', lw=1.5,
                   label='Prior mean', alpha=0.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(diags, rotation=25, ha='right', fontsize=9)
        ax.set_title(param_labels[pname], fontsize=9, fontweight='bold')
        ax.set_ylabel('MAP Estimate', fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[0].legend(loc='upper right', fontsize=7)

    # Colour legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=DIAG_COLORS[d], label=d, alpha=0.7) for d in diags]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    path = os.path.join(outdir, 'SuppFig22_posterior_parameters.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_bayesian_feature_outcome_heatmap(df, bayes_df, outdir=OUTDIR):
    """
    fig21: Pearson r heatmap — 19 Bayesian features × 9 outcomes.
    """
    outcomes = REG_OUTCOMES + CLF_OUTCOMES
    bayes_feat_cols = [c for c in bayes_df.columns]

    dm = pd.concat([df.reset_index(drop=True), bayes_df.reset_index(drop=True)], axis=1)

    avail_outcomes = [o for o in outcomes if o in dm.columns]
    n_feat, n_out  = len(bayes_feat_cols), len(avail_outcomes)

    corr_matrix = np.zeros((n_feat, n_out))
    for i, fc in enumerate(bayes_feat_cols):
        for j, oc in enumerate(avail_outcomes):
            valid = dm[[fc, oc]].dropna()
            if len(valid) > 30:
                corr_matrix[i, j] = valid[fc].corr(valid[oc])

    fig, ax = plt.subplots(figsize=(max(n_out * 0.9 + 2, 10), max(n_feat * 0.4 + 2, 9)))
    im = ax.imshow(corr_matrix, aspect='auto', cmap='RdBu_r', vmin=-0.6, vmax=0.6)
    plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)

    ax.set_xticks(range(n_out))
    ax.set_xticklabels(avail_outcomes, rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(bayes_feat_cols, fontsize=7)

    # Annotate cells with |r| > 0.1
    for i in range(n_feat):
        for j in range(n_out):
            r = corr_matrix[i, j]
            if abs(r) > 0.10:
                ax.text(j, i, f'{r:.2f}', ha='center', va='center',
                        fontsize=5.5, color='white' if abs(r) > 0.35 else 'black')

    ax.set_title('Fig 21: Bayesian ODE Features vs Clinical Outcomes\n'
                 'Pearson correlation (n=2,000)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(outdir, 'SuppFig23_bayesian_outcome_heatmap.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_extended_ml_comparison(results_df, outdir=OUTDIR):
    """
    fig22: Grouped bar chart — 5 feature sets × 4 key outcomes.
    Shows R² (regression) or AUC (classification) with LASSO/LogReg model only.
    """
    key_outcomes = ['gose_12m', 'fim_total_12m', 'mortality_12m', 'return_to_work_12m']
    fs_order     = ['Clinical', 'Clinical+DetQSP', 'Clinical+BayesODE',
                    'Clinical+Drug', 'Clinical+All']
    fs_short     = ['Clin', 'Clin+QSP', 'Clin+Bayes', 'Clin+Drug', 'All']
    colors       = ['#BDC3C7', '#3498DB', '#9B59B6', '#E67E22', '#2ECC71']

    models = {'gose_12m': 'LASSO', 'fim_total_12m': 'LASSO',
              'mortality_12m': 'LogReg', 'return_to_work_12m': 'LogReg'}
    metrics = {'gose_12m': 'R2', 'fim_total_12m': 'R2',
               'mortality_12m': 'AUC', 'return_to_work_12m': 'AUC'}
    titles  = {'gose_12m': 'GOSE 12m (R²)',
               'fim_total_12m': 'FIM Total 12m (R²)',
               'mortality_12m': 'Mortality (AUC)',
               'return_to_work_12m': 'Return-to-Work (AUC)'}

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)
    fig.suptitle('Fig 22: Extended 5-Way ML Comparison\n'
                 'LASSO (regression) | Logistic Regression (classification) | 5-fold CV',
                 fontsize=12, fontweight='bold')

    for ax, outcome in zip(axes, key_outcomes):
        sub = results_df[
            (results_df['outcome'] == outcome) &
            (results_df['model']   == models[outcome]) &
            (results_df['metric']  == metrics[outcome])
        ]
        vals = []
        for fs in fs_order:
            row = sub[sub['feature_set'] == fs]
            vals.append(float(row['value'].values[0]) if len(row) > 0 else 0.0)

        bars = ax.bar(range(len(fs_order)), vals, color=colors, edgecolor='white',
                      linewidth=0.5, width=0.75)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

        ax.set_xticks(range(len(fs_order)))
        ax.set_xticklabels(fs_short, rotation=30, ha='right', fontsize=8)
        ax.set_title(titles[outcome], fontsize=9, fontweight='bold')
        ax.set_ylabel(metrics[outcome], fontsize=8)
        ax.set_ylim(0, min(max(vals) * 1.2 + 0.05, 1.05) if max(vals) > 0 else 1.0)
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    path = os.path.join(outdir, 'SuppFig24_extended_ml_comparison.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _select_representative_patients(df):
    """Select up to 3 patients per diagnosis × severity combination."""
    diags = ['TBI', 'SAH', 'Stroke', 'ICH']
    selected = []
    for diag in diags:
        sub = df[df['diagnosis'] == diag]
        for gcs_min, gcs_max, label in [(3, 8, 'Severe'), (9, 12, 'Moderate'), (13, 15, 'Mild')]:
            mask = (sub['gcs_admission'] >= gcs_min) & (sub['gcs_admission'] <= gcs_max)
            pts  = sub[mask]
            if len(pts) > 0:
                selected.append((diag, label, pts.iloc[0]))
    return selected


def plot_icp_drug_trajectories(df, bayes_df, outdir=OUTDIR):
    """
    fig23: ICP trajectories for 3 drug scenarios across 4 diagnoses.
    Each row = one diagnosis. Lines: baseline (grey), mannitol (blue), HSS (orange).
    """
    selected = _select_representative_patients(df)
    diags    = ['TBI', 'SAH', 'Stroke', 'ICH']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes      = axes.flatten()

    t_eval = np.linspace(0, 48, 200)   # show first 48 h for clarity

    for ax, diag in zip(axes, diags):
        ax.set_title(f'{diag}', fontsize=10, fontweight='bold',
                     color=DIAG_COLORS[diag])
        ax.axhspan(0, 20, alpha=0.08, color='green', label='Normal ICP (≤20)')
        ax.axhline(20, color='green', ls=':', lw=1.0, alpha=0.6)

        diag_pts = [(d, sev, row) for d, sev, row in selected if d == diag]
        ls_styles = ['-', '--', ':']

        for (_, sev, row), ls in zip(diag_pts, ls_styles):
            row_dict = row.to_dict()
            # Merge Bayesian params
            bi = df.index[df['patient_id'] == row['patient_id']].tolist()
            params = get_icp_params(row_dict)
            if bi:
                brow = bayes_df.iloc[bi[0]].to_dict()
                for pn in PARAM_NAMES:
                    col = f'bayes_{pn}_mean'
                    if col in brow and not np.isnan(float(brow[col])):
                        params[pn] = float(brow[col])

            icp0 = params['ICP_init']
            y0   = [icp0, 0.01, 100.0, 290.0, 290.0, 0.0, 140.0]

            scenario_styles = [
                ('baseline',  '#888888', 1.0, f'{sev}: No drug'),
                ('mannitol',  '#2980B9', 1.8, f'{sev}: Mannitol'),
                ('hss',       '#E67E22', 1.8, f'{sev}: HSS'),
            ]
            for scen, color, lw, label in scenario_styles:
                try:
                    sol = solve_ivp(icp_cpp_osmotic_ode, (0, 48), y0,
                                    args=(params, _DRUG_SCENARIOS[scen]),
                                    method='RK45', t_eval=t_eval,
                                    rtol=1e-4, atol=1e-6, max_step=1.0)
                    if sol.success:
                        ax.plot(t_eval, np.clip(sol.y[0], 0, 80),
                                ls=ls, color=color, lw=lw, alpha=0.85,
                                label=label if ls == '-' else '_nolegend_')
                except Exception:
                    pass

        ax.set_xlabel('Time (h)', fontsize=8)
        ax.set_ylabel('ICP (mmHg)', fontsize=8)
        ax.set_xlim(0, 48)
        ax.set_ylim(0, 55)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)

        # Custom legend for scenarios
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color='#888888', lw=1.5, label='No drug'),
            Line2D([0], [0], color='#2980B9', lw=2.0, label='Mannitol 0.5 g/kg'),
            Line2D([0], [0], color='#E67E22', lw=2.0, label='HSS 3%'),
            Line2D([0], [0], color='grey', ls='-',  lw=1.5, label='Severe GCS'),
            Line2D([0], [0], color='grey', ls='--', lw=1.5, label='Moderate GCS'),
            Line2D([0], [0], color='grey', ls=':',  lw=1.5, label='Mild GCS'),
        ]
        ax.legend(handles=legend_handles, fontsize=6.5, loc='upper right', ncol=2)

    fig.suptitle('Fig 23: ICP Trajectories under Drug Scenarios\n'
                 'Baseline | Mannitol 0.5 g/kg IV | HSS 3% (250 mL) — first 48 h',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(outdir, 'SuppFig25_icp_drug_trajectories.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_osmolality_icp_coupling(df, bayes_df, drug_df, outdir=OUTDIR):
    """
    fig24: Osmolality–ICP coupling for 3 drug-response phenotypes.
    3 rows × 3 columns: ICP | Osm_total | Osmotic gradient
    Phenotypes: rapid responder | partial | non-responder (by ICP reduction)
    """
    # Classify phenotypes by mannitol ICP reduction
    icp_red_col = 'mannitol_icp_reduction_absolute'
    if icp_red_col not in drug_df.columns:
        print("  fig24: drug features not available, skipping.")
        return

    red = drug_df[icp_red_col].fillna(0)
    p66 = np.percentile(red, 66)
    p33 = np.percentile(red, 33)

    phenotype_masks = {
        'Rapid Responder\n(ICP reduction >p66)': red >= p66,
        'Partial Responder\n(p33–p66)':          (red >= p33) & (red < p66),
        'Non-Responder\n(reduction <p33)':        red < p33,
    }

    t_eval = np.linspace(0, 48, 300)
    fig    = plt.figure(figsize=(15, 10))
    gs     = gridspec.GridSpec(3, 3, hspace=0.5, wspace=0.4)

    row_labels = list(phenotype_masks.keys())
    col_labels = ['ICP (mmHg)', 'Serum Osm (mOsm/kg)', 'Osmotic Gradient (mOsm/kg)']
    col_idx    = [0, 3, 4]   # indices into sol.y for ICP, Osm_total, Osm_brain

    for ri, (phenotype, mask) in enumerate(phenotype_masks.items()):
        # Pick up to 5 representative patients
        idx_list = np.where(mask.values)[0][:5]
        for ax_ci, (col_label, y_idx) in enumerate(zip(col_labels, col_idx)):
            ax = fig.add_subplot(gs[ri, ax_ci])
            plotted = 0

            for i in idx_list:
                row_dict = df.iloc[i].to_dict()
                params   = get_icp_params(row_dict)
                brow     = bayes_df.iloc[i].to_dict()
                for pn in PARAM_NAMES:
                    col = f'bayes_{pn}_mean'
                    if col in brow and not np.isnan(float(brow[col])):
                        params[pn] = float(brow[col])

                y0 = [params['ICP_init'], 0.01, 100.0, 290.0, 290.0, 0.0, 140.0]
                try:
                    sol = solve_ivp(
                        icp_cpp_osmotic_ode, (0, 48), y0,
                        args=(params, _DRUG_SCENARIOS['mannitol']),
                        method='RK45', t_eval=t_eval,
                        rtol=1e-4, atol=1e-6, max_step=1.0,
                    )
                    if sol.success:
                        if y_idx == 4:   # osmotic gradient
                            osm_grad = np.maximum(sol.y[3] - sol.y[4], 0)
                            ax.plot(t_eval, osm_grad, alpha=0.6, lw=1.2,
                                    color=f'C{plotted}')
                        else:
                            ax.plot(t_eval, np.clip(sol.y[y_idx], 0, None),
                                    alpha=0.6, lw=1.2, color=f'C{plotted}')
                        plotted += 1
                except Exception:
                    pass

            ax.axvline(1.0, color='red', ls=':', lw=1, alpha=0.6, label='Drug at t=1h')
            ax.set_xlabel('Time (h)', fontsize=7)
            ax.set_ylabel(col_label, fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25)
            if ax_ci == 0:
                ax.set_ylabel(f'{phenotype}\n\n{col_label}', fontsize=7)
            if ri == 0:
                ax.set_title(col_label, fontsize=8, fontweight='bold')

    fig.suptitle('Fig 24: Osmolality–ICP Coupling under Mannitol (0.5 g/kg)\n'
                 '3 Response Phenotypes: rapid / partial / non-responder',
                 fontsize=12, fontweight='bold', y=1.01)
    path = os.path.join(outdir, 'SuppFig26_osmolality_icp_coupling.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_drug_response_phenotypes(df, drug_df, outdir=OUTDIR):
    """
    fig25: PCA (2 PC) of drug-response features + k-means (k=3) clusters.
    Scatter coloured by cluster; overlay median ICP reduction per cluster.
    """
    num_drug_cols = drug_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_low_var = ['has_rebound', 'drug_response_optimal']
    feat_cols = [c for c in num_drug_cols
                 if not any(c.endswith(s) for s in ['_per_osm'])  # avoid inf
                 and c not in exclude_low_var]

    X_raw = drug_df[feat_cols].values
    valid = np.all(np.isfinite(X_raw), axis=1)
    X_raw = X_raw[valid]

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X_raw)

    pca    = PCA(n_components=2, random_state=42)
    X_pca  = pca.fit_transform(X_s)

    km     = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(X_s)

    # Sort clusters by mannitol ICP reduction (descending → 0=rapid, 1=partial, 2=non)
    man_col = 'mannitol_icp_reduction_absolute'
    if man_col in drug_df.columns:
        med_red = [float(np.nanmedian(drug_df.loc[valid][man_col].values[labels == k]))
                   for k in range(3)]
        rank    = np.argsort(med_red)[::-1]
        label_map = {rank[0]: 'Rapid Responder', rank[1]: 'Partial Responder',
                     rank[2]: 'Non-Responder'}
    else:
        label_map = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2'}

    cluster_colors = ['#2ECC71', '#F39C12', '#E74C3C']
    fig, ax = plt.subplots(figsize=(9, 7))

    for k in range(3):
        mask = labels == k
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=cluster_colors[k], label=label_map[k],
                   alpha=0.5, s=18, edgecolors='none')

        # Centroid annotation
        cx, cy = X_pca[mask, 0].mean(), X_pca[mask, 1].mean()
        if man_col in drug_df.columns:
            mr = np.nanmedian(drug_df.loc[valid][man_col].values[mask])
            ax.annotate(f'{label_map[k]}\nMedian ΔICP={mr:.1f} mmHg',
                        (cx, cy), fontsize=8, ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', fc=cluster_colors[k],
                                  alpha=0.25, ec='none'))

    var_exp = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var_exp[0]:.1%} variance)', fontsize=10)
    ax.set_ylabel(f'PC2 ({var_exp[1]:.1%} variance)', fontsize=10)
    ax.set_title('Fig 25: Drug Response Phenotypes\n'
                 'PCA of 38 drug-response features | k-means (k=3)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, frameon=True, loc='best')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    path = os.path.join(outdir, 'SuppFig27_drug_response_phenotypes.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 9 — MAIN ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df, bayes_df, drug_df, results_df):
    """Print a concise summary of results."""
    print("\n" + "=" * 70)
    print("BRAIN INJURY QSP EXTENSION — RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nDataset       : n={len(df)} patients")
    print(f"Bayesian feats: {len(bayes_df.columns)} columns ({len(PARAM_NAMES)} params × 3 + 4 composite)")
    print(f"Drug feats    : {len(drug_df.select_dtypes(include=[np.number]).columns)} numeric columns")

    # Bayesian parameter check
    print("\n--- Bayesian MAP vs Deterministic (mean ± SD across patients) ---")
    for pname in PARAM_NAMES:
        col = f'bayes_{pname}_mean'
        if col in bayes_df.columns:
            print(f"  {pname:20s}: MAP={bayes_df[col].mean():.4f} ± {bayes_df[col].std():.4f}")

    # Uncertainty by diagnosis
    unc_col = 'bayes_param_uncertainty_index'
    if unc_col in bayes_df.columns:
        print(f"\n  Uncertainty index by diagnosis (higher = more uncertain):")
        for diag in ['TBI', 'SAH', 'Stroke', 'ICH']:
            mask = df['diagnosis'] == diag
            u    = bayes_df.loc[mask.values, unc_col].mean()
            print(f"    {diag:8s}: {u:.4f}")

    # Drug response
    man_col = 'mannitol_icp_reduction_absolute'
    hss_col = 'hss_icp_reduction_absolute'
    if man_col in drug_df.columns:
        print(f"\n--- Drug Response (ICP reduction, mmHg) ---")
        print(f"  Mannitol: {drug_df[man_col].mean():.2f} ± {drug_df[man_col].std():.2f}")
        print(f"  HSS     : {drug_df[hss_col].mean():.2f} ± {drug_df[hss_col].std():.2f}")
        pref = drug_df['drug_type_preferred'].value_counts()
        print(f"  Preferred drug: {pref.to_dict()}")

    # ML comparison
    if results_df is not None and len(results_df) > 0:
        print("\n--- 5-Way ML Comparison (key outcomes) ---")
        key = ['gose_12m', 'fim_total_12m', 'mortality_12m', 'return_to_work_12m']
        for outcome in key:
            sub = results_df[results_df['outcome'] == outcome]
            if len(sub) == 0:
                continue
            is_clf = outcome in CLF_OUTCOMES
            model  = 'LogReg' if is_clf else 'LASSO'
            metric = 'AUC' if is_clf else 'R2'
            row    = sub[(sub['model'] == model) & (sub['metric'] == metric)]
            if len(row) == 0:
                continue
            print(f"\n  {outcome} ({metric}):")
            for _, r in row.iterrows():
                v = r['value']
                print(f"    {r['feature_set']:22s}: {v:.4f}")

    print("\n" + "=" * 70)
    print("All outputs written to:", OUTDIR)
    print("=" * 70 + "\n")


def main():
    print("\n" + "=" * 70)
    print("BRAIN INJURY QSP EXTENSION — Bayesian ODE + Drug PK/PD")
    print("=" * 70 + "\n")

    # ── Load data ────────────────────────────────────────────────────────
    print("[1/9] Loading datasets...")
    cohort_path = os.path.join(OUTDIR, 'simulated_neurocritical_cohort_n2000.csv')
    mech_path   = os.path.join(OUTDIR, 'mechanistic_features_n2000.csv')

    df      = pd.read_csv(cohort_path)
    mech_df = pd.read_csv(mech_path) if os.path.exists(mech_path) else None
    print(f"  Cohort: {df.shape[0]} patients × {df.shape[1]} columns")
    if mech_df is not None:
        print(f"  Mech features: {mech_df.shape[0]} × {mech_df.shape[1]} columns")

    # ── Part 2: Waveform generation ──────────────────────────────────────
    print("\n[2/9] Generating synthetic ICP waveforms...")
    icp_summaries = generate_all_waveforms(df)
    print(f"  ICP summaries shape: {icp_summaries.shape}")

    # ── Part 3: Bayesian ODE estimation ─────────────────────────────────
    print("\n[3/9] MAP + Laplace ODE parameter estimation...")
    bayes_df = run_bayesian_estimation(df, icp_summaries)
    print(f"  Bayesian features: {bayes_df.shape}")

    # Save mechanistic Bayesian features separately
    bayes_mech_path = os.path.join(OUTDIR, 'bayes_mechanistic_features_n2000.csv')
    if not os.path.exists(bayes_mech_path):
        bayes_df.to_csv(bayes_mech_path, index=False)
        print(f"  Saved: {bayes_mech_path}")

    # ── Parts 5–6: Drug simulation + feature extraction ──────────────────
    print("\n[4/9] Drug PK/PD scenario simulation + feature extraction...")
    drug_df, sample_sols = simulate_and_extract_drug_features(df, bayes_df)
    print(f"  Drug features: {drug_df.shape}")

    # ── Part 7: Extended ML comparison ───────────────────────────────────
    print("\n[5/9] 5-way extended ML comparison...")
    results_df = run_extended_comparison(df, bayes_df, drug_df, mech_df)
    print(f"  Comparison results: {results_df.shape}")

    # ── Part 8: Visualisations ────────────────────────────────────────────
    print("\n[6/9] fig20 — Posterior parameter violin plots...")
    plot_posterior_parameters(df, bayes_df)

    print("[7/9] fig21 — Bayesian feature–outcome heatmap...")
    plot_bayesian_feature_outcome_heatmap(df, bayes_df)

    print("[8/9] fig22 — Extended ML comparison bar chart...")
    plot_extended_ml_comparison(results_df)

    print("[8b/9] fig23 — ICP drug trajectories...")
    plot_icp_drug_trajectories(df, bayes_df)

    print("[8c/9] fig24 — Osmolality–ICP coupling...")
    plot_osmolality_icp_coupling(df, bayes_df, drug_df)

    print("[8d/9] fig25 — Drug response phenotypes (PCA + k-means)...")
    plot_drug_response_phenotypes(df, drug_df)

    # ── Part 9: Summary ──────────────────────────────────────────────────
    print("\n[9/9] Summary...")
    print_summary(df, bayes_df, drug_df, results_df)


if __name__ == '__main__':
    main()
