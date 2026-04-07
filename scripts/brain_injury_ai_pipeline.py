#!/usr/bin/env python3
"""
Brain Injury AI/ML Prediction Pipeline
=======================================
Simulates a realistic neurocritical care cohort (n=2000) and predicts
multi-domain outcomes at 3, 6, and 12 months post-injury using:
  - Classical ML: XGBoost, Random Forest, LASSO
  - Deep Learning: MLP, LSTM (longitudinal)
  - Trajectory Modeling: Gaussian Mixture-based latent class analysis

Based on 8 published brain injury studies (TBI, SAH, stroke, ICH).

Author: Generated for Anand (Clinical Pharmacologist / AI Researcher)
Date: February 2026
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# PART 1: REALISTIC DATA SIMULATION
# ============================================================================

def simulate_neurocritical_cohort(n=2000, random_state=42):
    """
    Simulate a realistic neurocritical care cohort based on published literature.

    Data sources informing distributions:
    - CENTER-TBI (von Steinbuechel 2023): demographics, severity, trajectory classes
    - Rass 2024: SAH cognitive/mental health outcomes, neuropsych distributions
    - von Seth 2025 (U-LOTS): older adult TBI, GOSE/FIM/MPAI-4 distributions
    - Jia 2025: ICP, GCS, CRS-R, delirium, mobilization effects
    - Jang 2025: ICU anxiety, cognitive function, delirium incidence
    - Bannon 2020: Stroke psychosocial outcomes, caregiver burden
    - Kanny & Giacino 2025: DoC recovery timelines, consciousness trajectories
    - Chatelain 2025: Outcome preferences (mortality, ADL, memory, GOSE)
    """

    np.random.seed(random_state)

    print("=" * 70)
    print("PART 1: SIMULATING NEUROCRITICAL CARE COHORT (n={})".format(n))
    print("=" * 70)

    # --- Demographics ---
    age = np.clip(np.random.normal(55, 18, n), 18, 95).astype(int)
    sex = np.random.binomial(1, 0.60, n)  # 60% male (consistent with TBI/SAH literature)
    education_years = np.clip(np.random.normal(12, 3, n), 6, 22).astype(int)

    # Marital status: 0=single, 1=married/cohabitant, 2=divorced/widowed
    marital = np.random.choice([0, 1, 2], n, p=[0.25, 0.55, 0.20])

    # Pre-injury employment: 0=unemployed, 1=employed, 2=retired, 3=student
    employment_pre = np.zeros(n, dtype=int)
    for i in range(n):
        if age[i] >= 65:
            employment_pre[i] = np.random.choice([0, 1, 2], p=[0.10, 0.05, 0.85])
        elif age[i] < 25:
            employment_pre[i] = np.random.choice([0, 1, 3], p=[0.15, 0.35, 0.50])
        else:
            employment_pre[i] = np.random.choice([0, 1, 2], p=[0.15, 0.75, 0.10])

    # --- Diagnosis ---
    # Neurocritical care mix: TBI ~40%, SAH ~20%, Stroke ~25%, ICH ~15%
    diagnosis = np.random.choice(
        ['TBI', 'SAH', 'Stroke', 'ICH'], n,
        p=[0.40, 0.20, 0.25, 0.15]
    )

    # --- Injury Severity ---
    gcs_admission = np.zeros(n)
    apache_ii = np.zeros(n)

    for i in range(n):
        if diagnosis[i] == 'TBI':
            # TBI: mild 50%, moderate 30%, severe 20% (from CENTER-TBI/U-LOTS)
            severity_cat = np.random.choice(['mild', 'moderate', 'severe'], p=[0.50, 0.30, 0.20])
            if severity_cat == 'mild':
                gcs_admission[i] = np.random.randint(13, 16)
            elif severity_cat == 'moderate':
                gcs_admission[i] = np.random.randint(9, 13)
            else:
                gcs_admission[i] = np.random.randint(3, 9)
        elif diagnosis[i] == 'SAH':
            # SAH: H&H mapped to GCS approximation (from Rass 2024)
            hh = np.random.choice([1, 2, 3, 4, 5], p=[0.33, 0.29, 0.18, 0.07, 0.13])
            gcs_admission[i] = np.clip(16 - hh * 2 + np.random.normal(0, 1), 3, 15)
        elif diagnosis[i] == 'Stroke':
            gcs_admission[i] = np.clip(np.random.normal(12, 3), 3, 15)
        else:  # ICH
            gcs_admission[i] = np.clip(np.random.normal(9, 4), 3, 15)

        # APACHE II: correlated with GCS (inverse) and age
        apache_ii[i] = np.clip(
            35 - gcs_admission[i] * 1.2 + age[i] * 0.15 + np.random.normal(0, 4),
            5, 45
        )

    gcs_admission = np.clip(gcs_admission, 3, 15).astype(int)
    apache_ii = np.clip(apache_ii, 5, 45).astype(int)

    # --- Premorbid Factors ---
    hypertension = np.random.binomial(1, 0.05 + 0.005 * age)
    diabetes = np.random.binomial(1, 0.02 + 0.003 * age)
    cardiovascular = np.random.binomial(1, 0.01 + 0.005 * age)
    prior_psych = np.random.binomial(1, 0.18, n)  # ~18% psychiatric history
    prior_brain_injury = np.random.binomial(1, 0.08, n)
    anticoagulation = np.random.binomial(1, np.clip(0.01 + 0.004 * age, 0, 0.5))
    smoking = np.random.binomial(1, 0.35, n)
    alcohol_misuse = np.random.binomial(1, 0.15, n)

    # --- Acute Care Variables ---
    # ICU LOS (days): from Jia 2025 and general neurocritical literature
    icu_los = np.clip(
        np.random.lognormal(np.log(8) + (15 - gcs_admission) * 0.05, 0.6, n),
        1, 90
    ).astype(int)

    # Mechanical ventilation (days)
    mech_vent_days = np.zeros(n)
    for i in range(n):
        if gcs_admission[i] <= 8:
            mech_vent_days[i] = np.clip(np.random.lognormal(np.log(7), 0.7), 0, 60)
        elif gcs_admission[i] <= 12:
            mech_vent_days[i] = np.clip(np.random.exponential(3), 0, 30)
        else:
            mech_vent_days[i] = np.random.binomial(1, 0.2) * np.random.exponential(2)
    mech_vent_days = mech_vent_days.astype(int)

    # ICP monitoring (from Jia 2025)
    icp_monitored = (gcs_admission <= 8).astype(int) | np.random.binomial(1, 0.3, n)
    icp_mean = np.where(
        icp_monitored,
        np.clip(np.random.normal(12, 4, n), 3, 30),
        np.nan
    )

    # Early mobilization (from Jia 2025)
    early_mobilization = np.random.binomial(1, 0.35, n)

    # Delirium (from Jang 2025: 10-48% in neurocritical patients)
    # ICDSC score (0-8): higher = more delirium
    delirium_prob = expit(-1.5 + 0.15 * (15 - gcs_admission) + 0.01 * age +
                         0.5 * (mech_vent_days > 3) - 0.4 * early_mobilization +
                         0.3 * prior_psych)
    delirium_present = np.random.binomial(1, delirium_prob)
    icdsc_score = np.where(
        delirium_present,
        np.clip(np.random.poisson(4, n), 1, 8),
        np.clip(np.random.poisson(1, n), 0, 3)
    )

    # Anxiety (from Jang 2025: 12-60% in ICU)
    anxiety_icu = np.clip(
        np.random.normal(5, 3, n) + 2 * prior_psych + 0.02 * age -
        0.5 * early_mobilization + 0.3 * delirium_present,
        0, 10
    )

    # Surgical intervention
    surgery = np.random.binomial(1, np.where(
        diagnosis == 'TBI', 0.35,
        np.where(diagnosis == 'SAH', 0.72,
        np.where(diagnosis == 'Stroke', 0.20, 0.45))
    ))

    # Complications
    dvt = np.random.binomial(1, 0.08 + 0.04 * (1 - early_mobilization))
    pneumonia = np.random.binomial(1, 0.15 + 0.10 * (mech_vent_days > 5))
    uti = np.random.binomial(1, 0.12, n)

    # --- Generate Latent Severity / Recovery Potential Score ---
    # This latent variable drives outcome generation to create realistic correlations
    latent_severity = (
        -0.15 * gcs_admission +
        0.08 * apache_ii +
        0.02 * age -
        0.05 * education_years +
        0.5 * prior_psych +
        0.3 * prior_brain_injury +
        0.4 * delirium_present +
        -0.3 * early_mobilization +
        0.2 * (mech_vent_days > 7) +
        0.3 * dvt +
        0.2 * pneumonia +
        np.random.normal(0, 1, n)
    )

    # Assign trajectory class (from von Steinbuechel 2023: 4 classes)
    # Classes: 0=stable_good, 1=persistent_impaired, 2=improving, 3=deteriorating
    trajectory_probs = np.zeros((n, 4))
    for i in range(n):
        base = np.array([0.60, 0.18, 0.12, 0.10])  # base rates from CENTER-TBI
        # Adjust by severity
        severity_shift = latent_severity[i]
        if severity_shift > 1:
            base = base * np.array([0.5, 1.8, 0.8, 1.5])
        elif severity_shift > 0:
            base = base * np.array([0.7, 1.3, 1.1, 1.2])
        elif severity_shift < -1:
            base = base * np.array([1.5, 0.5, 1.2, 0.6])
        trajectory_probs[i] = base / base.sum()

    trajectory_class = np.array([
        np.random.choice([0, 1, 2, 3], p=trajectory_probs[i])
        for i in range(n)
    ])
    trajectory_labels = np.array(['stable_good', 'persistent_impaired', 'improving', 'deteriorating'])
    trajectory_name = trajectory_labels[trajectory_class]

    # --- OUTCOMES AT 3, 6, 12 MONTHS ---
    outcomes = {}

    for t, months in enumerate([3, 6, 12]):
        suffix = f"_{months}m"

        # Time-dependent recovery factor
        time_factor = np.log(months + 1) / np.log(13)  # normalized 0->1 over 12mo

        # Trajectory-specific modifiers
        traj_mod = np.zeros(n)
        for i in range(n):
            if trajectory_class[i] == 0:    # stable good
                traj_mod[i] = 0.5 * time_factor
            elif trajectory_class[i] == 1:  # persistent impaired
                traj_mod[i] = -0.8 + 0.1 * time_factor
            elif trajectory_class[i] == 2:  # improving
                traj_mod[i] = -0.5 + 1.2 * time_factor
            else:                           # deteriorating
                traj_mod[i] = 0.3 - 0.7 * time_factor

        # 1. GOSE (1-8): Global functional outcome
        gose_raw = (
            4.5 +
            0.2 * gcs_admission -
            0.02 * age +
            0.05 * education_years +
            traj_mod * 1.5 -
            0.3 * delirium_present +
            0.2 * early_mobilization -
            0.15 * apache_ii * 0.1 +
            np.random.normal(0, 0.8, n)
        )
        outcomes[f'gose{suffix}'] = np.clip(np.round(gose_raw), 1, 8).astype(int)

        # 2. Modified Rankin Scale (0-6): Functional disability
        mrs_raw = 6 - outcomes[f'gose{suffix}'] * 0.7 + np.random.normal(0, 0.5, n)
        outcomes[f'mrs{suffix}'] = np.clip(np.round(mrs_raw), 0, 6).astype(int)

        # 3. FIM Total (18-126): Functional Independence
        fim_raw = (
            80 +
            3 * gcs_admission -
            0.3 * age +
            traj_mod * 15 +
            5 * early_mobilization -
            3 * delirium_present -
            0.5 * apache_ii +
            np.random.normal(0, 12, n)
        )
        outcomes[f'fim_total{suffix}'] = np.clip(np.round(fim_raw), 18, 126).astype(int)

        # 4. Barthel Index (0-100): ADL
        barthel_raw = (outcomes[f'fim_total{suffix}'] - 18) / 108 * 100
        barthel_raw += np.random.normal(0, 5, n)
        outcomes[f'barthel{suffix}'] = np.clip(np.round(barthel_raw), 0, 100).astype(int)

        # 5. DRS (0-29): Disability Rating Scale (lower = better)
        drs_raw = 29 - outcomes[f'gose{suffix}'] * 3 + np.random.normal(0, 2, n) - traj_mod * 3
        outcomes[f'drs{suffix}'] = np.clip(np.round(drs_raw), 0, 29).astype(int)

        # 6. Cognitive Composite (z-score): Based on Rass 2024 neuropsych battery
        # Domains: memory, executive, attention, visuoconstruction
        cog_memory = (
            -0.3 + 0.08 * gcs_admission - 0.01 * age + 0.03 * education_years +
            traj_mod * 0.5 - 0.4 * delirium_present + np.random.normal(0, 1, n)
        )
        cog_executive = (
            -0.2 + 0.06 * gcs_admission - 0.008 * age + 0.04 * education_years +
            traj_mod * 0.4 - 0.3 * delirium_present + np.random.normal(0, 1, n)
        )
        cog_attention = (
            -0.1 + 0.07 * gcs_admission - 0.005 * age + 0.02 * education_years +
            traj_mod * 0.3 - 0.5 * delirium_present + np.random.normal(0, 0.9, n)
        )
        cog_visuoconst = (
            -0.15 + 0.05 * gcs_admission - 0.007 * age + 0.02 * education_years +
            traj_mod * 0.3 + np.random.normal(0, 1, n)
        )
        outcomes[f'cog_memory{suffix}'] = np.round(cog_memory, 2)
        outcomes[f'cog_executive{suffix}'] = np.round(cog_executive, 2)
        outcomes[f'cog_attention{suffix}'] = np.round(cog_attention, 2)
        outcomes[f'cog_visuoconst{suffix}'] = np.round(cog_visuoconst, 2)
        outcomes[f'cog_composite{suffix}'] = np.round(
            (cog_memory + cog_executive + cog_attention + cog_visuoconst) / 4, 2
        )

        # 7. MoCA (0-30): Montreal Cognitive Assessment
        moca_raw = 22 + outcomes[f'cog_composite{suffix}'] * 3 + np.random.normal(0, 2, n)
        outcomes[f'moca{suffix}'] = np.clip(np.round(moca_raw), 0, 30).astype(int)

        # 8. HADS-Anxiety (0-21): from Rass 2024
        hads_a_raw = (
            7 - 0.1 * gcs_admission + 0.02 * age +
            2.5 * prior_psych - traj_mod * 2 +
            1.5 * delirium_present +
            anxiety_icu * 0.6 +        # 0.3 → 0.6: ICU anxiety is stronger direct predictor
            0.5 * alcohol_misuse +     # new direct path: alcohol is anxiety risk factor
            np.random.normal(0, 2.5, n)
        )
        outcomes[f'hads_anxiety{suffix}'] = np.clip(np.round(hads_a_raw), 0, 21).astype(int)

        # 9. HADS-Depression (0-21)
        hads_d_raw = (
            5 - 0.08 * gcs_admission + 0.015 * age +
            2.5 * prior_psych - traj_mod * 2.5 +   # 2.0 → 2.5: stronger psych history signal
            1.0 * delirium_present -
            0.4 * (marital == 1) +                  # -0.3 → -0.4
            0.4 * alcohol_misuse +                  # new direct path
            np.random.normal(0, 2.0, n)             # σ 2.5 → 2.0: less noise
        )
        outcomes[f'hads_depression{suffix}'] = np.clip(np.round(hads_d_raw), 0, 21).astype(int)

        # 10. PHQ-9 (0-27): Depression severity
        phq9_raw = (
            outcomes[f'hads_depression{suffix}'] * 1.1 +
            prior_psych * 1.5 +      # direct path: psych history directly predicts PHQ-9
            anxiety_icu * 0.4 +      # direct path: ICU anxiety burden
            np.random.normal(0, 1.3, n)   # σ 2.0 → 1.3: less noise
        )
        outcomes[f'phq9{suffix}'] = np.clip(np.round(phq9_raw), 0, 27).astype(int)

        # 11. GAD-7 (0-21): Generalized anxiety
        gad7_raw = outcomes[f'hads_anxiety{suffix}'] * 0.9 + np.random.normal(0, 1.5, n)
        outcomes[f'gad7{suffix}'] = np.clip(np.round(gad7_raw), 0, 21).astype(int)

        # 12. PCL-5 (0-80): PTSD checklist
        pcl5_raw = (
            10 +                                          # base 15 → 10
            outcomes[f'hads_anxiety{suffix}'] * 1.5 +
            outcomes[f'hads_depression{suffix}'] * 1.0 +
            prior_psych * 3.0 +       # direct path: pre-injury psych vulnerability → PTSD
            alcohol_misuse * 2.0 +    # direct path: alcohol misuse is a major PTSD risk factor
            np.random.normal(0, 5.0, n)   # σ 8.0 → 5.0: less noise
        )
        outcomes[f'pcl5{suffix}'] = np.clip(np.round(pcl5_raw), 0, 80).astype(int)

        # 13. SF-36 Physical Component Summary (0-100)
        sf36_pcs_raw = (
            45 + outcomes[f'gose{suffix}'] * 3 - 0.15 * age +
            traj_mod * 8 + np.random.normal(0, 8, n)
        )
        outcomes[f'sf36_pcs{suffix}'] = np.clip(np.round(sf36_pcs_raw, 1), 0, 100)

        # 14. SF-36 Mental Component Summary (0-100)
        sf36_mcs_raw = (
            50 - outcomes[f'hads_anxiety{suffix}'] * 1.5 -
            outcomes[f'hads_depression{suffix}'] * 1.5 +
            traj_mod * 5 + np.random.normal(0, 7, n)
        )
        outcomes[f'sf36_mcs{suffix}'] = np.clip(np.round(sf36_mcs_raw, 1), 0, 100)

        # 15. QOLIBRI-OS (0-100): TBI-specific QoL
        qolibri_raw = (
            55 + outcomes[f'cog_composite{suffix}'] * 5 +
            outcomes[f'gose{suffix}'] * 2 -
            outcomes[f'hads_anxiety{suffix}'] * 1.0 -
            outcomes[f'hads_depression{suffix}'] * 1.0 +
            traj_mod * 8 + np.random.normal(0, 10, n)
        )
        outcomes[f'qolibri_os{suffix}'] = np.clip(np.round(qolibri_raw, 1), 0, 100)

        # 16. MPAI-4 T-score (lower=better): from von Seth 2025
        mpai4_raw = 50 - outcomes[f'gose{suffix}'] * 3 + np.random.normal(0, 8, n) - traj_mod * 5
        outcomes[f'mpai4_tscore{suffix}'] = np.clip(np.round(mpai4_raw, 1), 10, 80)

        # 17. Return to Work (binary, among working-age pre-employed)
        rtw_prob = expit(
            -2 + 0.3 * outcomes[f'gose{suffix}'] - 0.03 * age +
            0.05 * education_years + traj_mod * 1.5 -
            0.1 * outcomes[f'hads_depression{suffix}'] +
            0.3 * outcomes[f'cog_composite{suffix}']
        )
        # Only applicable to pre-injury employed, working age
        rtw = np.zeros(n, dtype=int)
        for i in range(n):
            if employment_pre[i] == 1 and age[i] < 65:
                rtw[i] = np.random.binomial(1, rtw_prob[i])
            elif employment_pre[i] == 3 and age[i] < 30:  # students
                rtw[i] = np.random.binomial(1, rtw_prob[i] * 0.8)
        outcomes[f'return_to_work{suffix}'] = rtw

        # 18. Social Participation Score (0-100)
        social_raw = (
            60 + outcomes[f'gose{suffix}'] * 2 -
            0.1 * age + 0.3 * (marital == 1) * 10 +
            traj_mod * 8 -
            outcomes[f'hads_depression{suffix}'] * 0.8 +
            outcomes[f'cog_composite{suffix}'] * 3 +
            np.random.normal(0, 10, n)
        )
        outcomes[f'social_participation{suffix}'] = np.clip(np.round(social_raw, 1), 0, 100)

    # --- Mortality ---
    mortality_prob = expit(
        -3 + 0.3 * (15 - gcs_admission) + 0.03 * age + 0.02 * apache_ii +
        0.5 * (diagnosis == 'ICH') - 0.3 * early_mobilization +
        np.random.normal(0, 0.5, n)
    )
    mortality_12m = np.random.binomial(1, np.clip(mortality_prob, 0, 0.5))

    # Build DataFrame
    df = pd.DataFrame({
        'patient_id': range(1, n + 1),
        'age': age,
        'sex': sex,
        'education_years': education_years,
        'marital_status': marital,
        'employment_pre': employment_pre,
        'diagnosis': diagnosis,
        'gcs_admission': gcs_admission,
        'apache_ii': apache_ii,
        'hypertension': hypertension,
        'diabetes': diabetes,
        'cardiovascular_disease': cardiovascular,
        'prior_psych_history': prior_psych,
        'prior_brain_injury': prior_brain_injury,
        'anticoagulation': anticoagulation,
        'smoking': smoking,
        'alcohol_misuse': alcohol_misuse,
        'icu_los_days': icu_los,
        'mech_ventilation_days': mech_vent_days,
        'icp_monitored': icp_monitored,
        'icp_mean_mmhg': np.round(icp_mean, 1),
        'early_mobilization': early_mobilization,
        'delirium_present': delirium_present,
        'icdsc_score': icdsc_score,
        'anxiety_icu_score': np.round(anxiety_icu, 1),
        'surgery': surgery,
        'dvt': dvt,
        'pneumonia': pneumonia,
        'uti': uti,
        'trajectory_class': trajectory_name,
        'mortality_12m': mortality_12m,
    })

    # Add all outcomes
    for key, val in outcomes.items():
        df[key] = val

    # Set outcomes to NaN for deceased patients at appropriate timepoints
    deceased_idx = df[df['mortality_12m'] == 1].index
    # ~40% die before 3mo, ~30% between 3-6mo, ~30% between 6-12mo
    for idx in deceased_idx:
        death_time = np.random.choice([3, 6, 12], p=[0.4, 0.3, 0.3])
        outcome_cols = [c for c in df.columns if c.startswith(('gose_', 'mrs_', 'fim_', 'barthel_',
                        'drs_', 'cog_', 'moca_', 'hads_', 'phq9_', 'gad7_', 'pcl5_',
                        'sf36_', 'qolibri_', 'mpai4_', 'return_to_work_', 'social_'))]
        for col in outcome_cols:
            for m in [3, 6, 12]:
                if m > death_time and f'_{m}m' in col:
                    df.loc[idx, col] = np.nan

    print(f"  Cohort simulated: {n} patients")
    print(f"  Diagnoses: {df['diagnosis'].value_counts().to_dict()}")
    print(f"  Mean age: {df['age'].mean():.1f} (SD {df['age'].std():.1f})")
    print(f"  Male: {df['sex'].mean()*100:.1f}%")
    print(f"  Mean GCS: {df['gcs_admission'].mean():.1f} (SD {df['gcs_admission'].std():.1f})")
    print(f"  Trajectory classes: {df['trajectory_class'].value_counts().to_dict()}")
    print(f"  12-month mortality: {df['mortality_12m'].mean()*100:.1f}%")

    return df


# ============================================================================
# PART 2: CLASSICAL ML PIPELINE
# ============================================================================

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, accuracy_score, mean_squared_error,
                             r2_score, classification_report, f1_score)
from sklearn.pipeline import Pipeline
import xgboost as xgb


def prepare_features(df, target_time='12m'):
    """Prepare feature matrix X and identify baseline predictors."""

    feature_cols = [
        'age', 'sex', 'education_years', 'marital_status', 'employment_pre',
        'gcs_admission', 'apache_ii',
        'hypertension', 'diabetes', 'cardiovascular_disease',
        'prior_psych_history', 'prior_brain_injury', 'anticoagulation',
        'smoking', 'alcohol_misuse',
        'icu_los_days', 'mech_ventilation_days',
        'early_mobilization', 'delirium_present', 'icdsc_score',
        'anxiety_icu_score', 'surgery', 'dvt', 'pneumonia', 'uti'
    ]

    # Encode diagnosis
    le = LabelEncoder()
    df['diagnosis_encoded'] = le.fit_transform(df['diagnosis'])
    feature_cols.append('diagnosis_encoded')

    # ICP: fill NaN with median for non-monitored patients
    df['icp_filled'] = df['icp_mean_mmhg'].fillna(df['icp_mean_mmhg'].median())
    feature_cols.append('icp_filled')

    X = df[feature_cols].copy()

    return X, feature_cols


def run_classical_ml(df):
    """Run classical ML models for multi-outcome prediction."""

    print("\n" + "=" * 70)
    print("PART 2: CLASSICAL ML PIPELINE")
    print("=" * 70)

    X, feature_cols = prepare_features(df)

    # Define target outcomes at 12 months
    classification_targets = {
        'return_to_work_12m': 'Return to Work',
        'mortality_12m': 'Mortality (12-month)',
    }

    regression_targets = {
        'gose_12m': 'GOSE (Functional Outcome)',
        'mrs_12m': 'mRS (Disability)',
        'fim_total_12m': 'FIM Total (Independence)',
        'barthel_12m': 'Barthel Index (ADL)',
        'drs_12m': 'DRS (Disability Rating)',
        'cog_composite_12m': 'Cognitive Composite',
        'moca_12m': 'MoCA (Cognition)',
        'hads_anxiety_12m': 'HADS-Anxiety',
        'hads_depression_12m': 'HADS-Depression',
        'phq9_12m': 'PHQ-9 (Depression)',
        'gad7_12m': 'GAD-7 (Anxiety)',
        'pcl5_12m': 'PCL-5 (PTSD)',
        'sf36_pcs_12m': 'SF-36 Physical Component',
        'sf36_mcs_12m': 'SF-36 Mental Component',
        'qolibri_os_12m': 'QOLIBRI-OS (TBI-QoL)',
        'mpai4_tscore_12m': 'MPAI-4 T-Score',
        'social_participation_12m': 'Social Participation',
    }

    results = {}
    feature_importance_all = {}

    # --- Classification Tasks ---
    print("\n--- CLASSIFICATION OUTCOMES ---")
    for target_col, target_name in classification_targets.items():
        print(f"\n  >> {target_name}")

        mask = df[target_col].notna()
        X_sub = X[mask].copy()
        y_sub = df.loc[mask, target_col].astype(int)

        if target_col == 'return_to_work_12m':
            # Only working-age, pre-employed
            work_mask = (df['employment_pre'].isin([1, 3])) & (df['age'] < 65) & mask
            X_sub = X[work_mask].copy()
            y_sub = df.loc[work_mask, target_col].astype(int)

        if len(y_sub) < 50:
            print(f"    Skipped (insufficient samples: {len(y_sub)})")
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sub)

        models = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                random_state=42, eval_metric='logloss', verbosity=0
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42
            ),
            'Logistic LASSO': LogisticRegressionCV(
                penalty='l1', solver='saga', cv=5, max_iter=2000, random_state=42
            ),
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        target_results = {}

        for model_name, model in models.items():
            auc_scores = cross_val_score(model, X_scaled, y_sub, cv=cv, scoring='roc_auc')
            f1_scores = cross_val_score(model, X_scaled, y_sub, cv=cv, scoring='f1')

            model.fit(X_scaled, y_sub)

            if hasattr(model, 'feature_importances_'):
                fi = model.feature_importances_
            elif hasattr(model, 'coef_'):
                fi = np.abs(model.coef_[0])
            else:
                fi = np.zeros(len(feature_cols))

            target_results[model_name] = {
                'AUC': f"{auc_scores.mean():.3f} ({auc_scores.std():.3f})",
                'F1': f"{f1_scores.mean():.3f} ({f1_scores.std():.3f})",
                'AUC_mean': auc_scores.mean(),
                'F1_mean': f1_scores.mean(),
            }

            if model_name == 'XGBoost':
                feature_importance_all[target_name] = dict(zip(feature_cols, fi))

            print(f"    {model_name}: AUC={auc_scores.mean():.3f}±{auc_scores.std():.3f}  "
                  f"F1={f1_scores.mean():.3f}±{f1_scores.std():.3f}")

        results[target_name] = target_results

    # --- Regression Tasks ---
    print("\n--- REGRESSION OUTCOMES ---")
    for target_col, target_name in regression_targets.items():

        mask = df[target_col].notna()
        X_sub = X[mask].copy()
        y_sub = df.loc[mask, target_col].astype(float)

        if len(y_sub) < 50:
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sub)

        models = {
            'XGBoost': xgb.XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                random_state=42, verbosity=0
            ),
            'Random Forest': RandomForestRegressor(
                n_estimators=200, max_depth=10, random_state=42
            ),
            'LASSO': LassoCV(cv=5, random_state=42, max_iter=2000),
        }

        from sklearn.model_selection import KFold
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        target_results = {}

        for model_name, model in models.items():
            r2_scores = cross_val_score(model, X_scaled, y_sub, cv=cv, scoring='r2')
            rmse_scores = -cross_val_score(model, X_scaled, y_sub, cv=cv,
                                          scoring='neg_root_mean_squared_error')

            model.fit(X_scaled, y_sub)

            if hasattr(model, 'feature_importances_'):
                fi = model.feature_importances_
            elif hasattr(model, 'coef_'):
                fi = np.abs(model.coef_)
            else:
                fi = np.zeros(len(feature_cols))

            target_results[model_name] = {
                'R2': f"{r2_scores.mean():.3f} ({r2_scores.std():.3f})",
                'RMSE': f"{rmse_scores.mean():.3f} ({rmse_scores.std():.3f})",
                'R2_mean': r2_scores.mean(),
                'RMSE_mean': rmse_scores.mean(),
            }

            if model_name == 'XGBoost':
                feature_importance_all[target_name] = dict(zip(feature_cols, fi))

        results[target_name] = target_results

        # Print top results
        best_model = max(target_results, key=lambda k: target_results[k]['R2_mean'])
        best_r2 = target_results[best_model]['R2_mean']
        print(f"  {target_name}: Best={best_model} R²={best_r2:.3f}")

    return results, feature_importance_all


# ============================================================================
# PART 3: DEEP LEARNING PIPELINE (MLP + Simple LSTM-like)
# ============================================================================

from sklearn.neural_network import MLPClassifier, MLPRegressor

def run_deep_learning(df):
    """Run MLP models for multi-outcome prediction."""

    print("\n" + "=" * 70)
    print("PART 3: DEEP LEARNING PIPELINE (MLP)")
    print("=" * 70)

    X, feature_cols = prepare_features(df)

    # MLP for key outcomes
    targets = {
        'gose_12m': ('regression', 'GOSE 12m'),
        'cog_composite_12m': ('regression', 'Cognitive Composite 12m'),
        'sf36_mcs_12m': ('regression', 'SF-36 MCS 12m'),
        'qolibri_os_12m': ('regression', 'QOLIBRI-OS 12m'),
        'hads_anxiety_12m': ('regression', 'HADS-Anxiety 12m'),
        'hads_depression_12m': ('regression', 'HADS-Depression 12m'),
        'return_to_work_12m': ('classification', 'Return to Work 12m'),
        'mortality_12m': ('classification', 'Mortality 12m'),
        'social_participation_12m': ('regression', 'Social Participation 12m'),
    }

    dl_results = {}

    for target_col, (task_type, target_name) in targets.items():
        mask = df[target_col].notna()

        if target_col == 'return_to_work_12m':
            mask = mask & (df['employment_pre'].isin([1, 3])) & (df['age'] < 65)

        X_sub = X[mask].copy()
        y_sub = df.loc[mask, target_col].astype(float if task_type == 'regression' else int)

        if len(y_sub) < 50:
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sub)

        if task_type == 'classification':
            model = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
                batch_size=64,
                learning_rate='adaptive',
                alpha=0.001,
            )
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            auc_scores = cross_val_score(model, X_scaled, y_sub, cv=cv, scoring='roc_auc')
            f1_scores = cross_val_score(model, X_scaled, y_sub, cv=cv, scoring='f1')
            dl_results[target_name] = {
                'AUC': f"{auc_scores.mean():.3f} ({auc_scores.std():.3f})",
                'F1': f"{f1_scores.mean():.3f} ({f1_scores.std():.3f})",
                'AUC_mean': auc_scores.mean(),
            }
            print(f"  {target_name} [MLP]: AUC={auc_scores.mean():.3f}±{auc_scores.std():.3f}")
        else:
            model = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
                batch_size=64,
                learning_rate='adaptive',
                alpha=0.001,
            )
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            r2_scores = cross_val_score(model, X_scaled, y_sub, cv=cv, scoring='r2')
            dl_results[target_name] = {
                'R2': f"{r2_scores.mean():.3f} ({r2_scores.std():.3f})",
                'R2_mean': r2_scores.mean(),
            }
            print(f"  {target_name} [MLP]: R²={r2_scores.mean():.3f}±{r2_scores.std():.3f}")

    # --- Longitudinal LSTM-like: Multi-timepoint prediction ---
    print("\n  --- Longitudinal Multi-Timepoint Analysis ---")

    # Prepare longitudinal features: use 3-month outcomes to predict 12-month
    outcomes_3m = ['gose_3m', 'cog_composite_3m', 'hads_anxiety_3m',
                   'hads_depression_3m', 'sf36_pcs_3m', 'sf36_mcs_3m']
    outcomes_6m = ['gose_6m', 'cog_composite_6m', 'hads_anxiety_6m',
                   'hads_depression_6m', 'sf36_pcs_6m', 'sf36_mcs_6m']

    mask = df[outcomes_3m + outcomes_6m + ['gose_12m']].notna().all(axis=1)
    X_baseline = X[mask].copy()
    X_3m = df.loc[mask, outcomes_3m].copy()
    X_6m = df.loc[mask, outcomes_6m].copy()

    # Concatenate: baseline + 3m + 6m as expanded feature set
    X_longitudinal = pd.concat([X_baseline.reset_index(drop=True),
                                X_3m.reset_index(drop=True),
                                X_6m.reset_index(drop=True)], axis=1)

    y_gose_12m = df.loc[mask, 'gose_12m'].values.astype(float)
    y_qolibri_12m = df.loc[mask, 'qolibri_os_12m'].values.astype(float)

    scaler = StandardScaler()
    X_long_scaled = scaler.fit_transform(X_longitudinal)

    for y_target, name in [(y_gose_12m, 'GOSE 12m (longitudinal)'),
                           (y_qolibri_12m, 'QOLIBRI-OS 12m (longitudinal)')]:
        mask_y = ~np.isnan(y_target)
        if mask_y.sum() < 50:
            continue
        model = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation='relu', solver='adam', max_iter=500,
            early_stopping=True, validation_fraction=0.15,
            random_state=42, batch_size=64, alpha=0.001,
        )
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_scores = cross_val_score(model, X_long_scaled[mask_y], y_target[mask_y],
                                   cv=cv, scoring='r2')
        dl_results[name] = {'R2': f"{r2_scores.mean():.3f} ({r2_scores.std():.3f})",
                           'R2_mean': r2_scores.mean()}
        print(f"  {name} [Deep MLP]: R²={r2_scores.mean():.3f}±{r2_scores.std():.3f}")

    return dl_results


# ============================================================================
# PART 4: TRAJECTORY MODELING (Latent Class Analysis)
# ============================================================================

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

def run_trajectory_modeling(df):
    """Run latent class trajectory modeling (GMM-based)."""

    print("\n" + "=" * 70)
    print("PART 4: LATENT CLASS TRAJECTORY MODELING")
    print("=" * 70)

    # Multi-domain longitudinal outcomes for trajectory discovery
    outcome_vars = []
    for m in ['3m', '6m', '12m']:
        outcome_vars.extend([
            f'gose_{m}', f'cog_composite_{m}',
            f'hads_anxiety_{m}', f'hads_depression_{m}',
            f'sf36_pcs_{m}', f'sf36_mcs_{m}',
            f'qolibri_os_{m}', f'social_participation_{m}'
        ])

    mask = df[outcome_vars].notna().all(axis=1)
    df_traj = df[mask].copy()
    X_traj = df_traj[outcome_vars].values

    scaler = StandardScaler()
    X_traj_scaled = scaler.fit_transform(X_traj)

    print(f"  Patients with complete longitudinal data: {len(df_traj)}")

    # Fit GMM with different numbers of classes
    bic_scores = []
    aic_scores = []
    models = {}

    for n_classes in range(2, 7):
        gmm = GaussianMixture(
            n_components=n_classes,
            covariance_type='full',
            max_iter=500,
            n_init=10,
            random_state=42
        )
        gmm.fit(X_traj_scaled)
        bic_scores.append(gmm.bic(X_traj_scaled))
        aic_scores.append(gmm.aic(X_traj_scaled))
        models[n_classes] = gmm
        print(f"  {n_classes} classes: BIC={gmm.bic(X_traj_scaled):.1f}  "
              f"AIC={gmm.aic(X_traj_scaled):.1f}")

    # Select optimal model (lowest BIC, consistent with CENTER-TBI finding of 4 classes)
    optimal_k = range(2, 7)[np.argmin(bic_scores)]
    print(f"\n  Optimal number of trajectory classes (BIC): {optimal_k}")

    best_gmm = models[optimal_k]
    df_traj['predicted_trajectory'] = best_gmm.predict(X_traj_scaled)
    df_traj['trajectory_probability'] = best_gmm.predict_proba(X_traj_scaled).max(axis=1)

    # Characterize each class
    print(f"\n  --- Trajectory Class Characteristics (k={optimal_k}) ---")
    trajectory_profiles = {}

    for cls in range(optimal_k):
        cls_mask = df_traj['predicted_trajectory'] == cls
        cls_data = df_traj[cls_mask]
        n_cls = cls_mask.sum()
        pct = n_cls / len(df_traj) * 100

        profile = {
            'n': n_cls,
            'pct': pct,
            'age_mean': cls_data['age'].mean(),
            'gcs_mean': cls_data['gcs_admission'].mean(),
            'gose_3m': cls_data['gose_3m'].mean(),
            'gose_12m': cls_data['gose_12m'].mean(),
            'gose_change': cls_data['gose_12m'].mean() - cls_data['gose_3m'].mean(),
            'cog_3m': cls_data['cog_composite_3m'].mean(),
            'cog_12m': cls_data['cog_composite_12m'].mean(),
            'hads_a_12m': cls_data['hads_anxiety_12m'].mean(),
            'hads_d_12m': cls_data['hads_depression_12m'].mean(),
            'qolibri_12m': cls_data['qolibri_os_12m'].mean(),
            'sf36_mcs_12m': cls_data['sf36_mcs_12m'].mean(),
            'mortality': cls_data['mortality_12m'].mean() * 100,
            'rtw': cls_data['return_to_work_12m'].mean() * 100 if cls_data['return_to_work_12m'].sum() > 0 else 0,
            'prior_psych': cls_data['prior_psych_history'].mean() * 100,
            'delirium': cls_data['delirium_present'].mean() * 100,
        }
        trajectory_profiles[cls] = profile

        print(f"\n  Class {cls} (n={n_cls}, {pct:.1f}%):")
        print(f"    Age: {profile['age_mean']:.1f}  GCS: {profile['gcs_mean']:.1f}")
        print(f"    GOSE: {profile['gose_3m']:.1f} → {profile['gose_12m']:.1f} "
              f"(Δ={profile['gose_change']:+.1f})")
        print(f"    Cognitive: {profile['cog_3m']:.2f} → {profile['cog_12m']:.2f}")
        print(f"    HADS-A: {profile['hads_a_12m']:.1f}  HADS-D: {profile['hads_d_12m']:.1f}")
        print(f"    QOLIBRI: {profile['qolibri_12m']:.1f}  SF-36 MCS: {profile['sf36_mcs_12m']:.1f}")
        print(f"    Mortality: {profile['mortality']:.1f}%  Prior psych: {profile['prior_psych']:.1f}%")

    # Agreement with simulated trajectory labels
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    true_labels = LabelEncoder().fit_transform(df_traj['trajectory_class'])
    pred_labels = df_traj['predicted_trajectory'].values
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    print(f"\n  Agreement with simulated trajectories:")
    print(f"    Adjusted Rand Index: {ari:.3f}")
    print(f"    Normalized Mutual Information: {nmi:.3f}")

    return df_traj, trajectory_profiles, bic_scores, aic_scores, optimal_k


# ============================================================================
# PART 5: VISUALIZATION
# ============================================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def generate_visualizations(df, results, feature_importance_all, dl_results,
                           df_traj, trajectory_profiles, bic_scores, aic_scores,
                           optimal_k, output_dir):
    """Generate comprehensive visualization suite."""

    print("\n" + "=" * 70)
    print("PART 5: GENERATING VISUALIZATIONS")
    print("=" * 70)

    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size'] = 10

    # --- Figure 1: Cohort Demographics ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Figure 1: Neurocritical Care Cohort Demographics (n=2000)', fontsize=14, fontweight='bold')

    axes[0, 0].hist(df['age'], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0, 0].set_xlabel('Age (years)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Age Distribution')

    diag_counts = df['diagnosis'].value_counts()
    axes[0, 1].bar(diag_counts.index, diag_counts.values, color=['#2196F3', '#FF9800', '#4CAF50', '#F44336'])
    axes[0, 1].set_title('Diagnosis Distribution')
    axes[0, 1].set_ylabel('Count')

    axes[0, 2].hist(df['gcs_admission'], bins=13, color='coral', edgecolor='white', alpha=0.8)
    axes[0, 2].set_xlabel('GCS at Admission')
    axes[0, 2].set_title('Injury Severity (GCS)')

    traj_counts = df['trajectory_class'].value_counts()
    colors_traj = ['#4CAF50', '#F44336', '#2196F3', '#FF9800']
    axes[1, 0].bar(range(len(traj_counts)), traj_counts.values, color=colors_traj[:len(traj_counts)])
    axes[1, 0].set_xticks(range(len(traj_counts)))
    axes[1, 0].set_xticklabels(traj_counts.index, rotation=30, ha='right', fontsize=8)
    axes[1, 0].set_title('Trajectory Classes')

    axes[1, 1].hist(df['icu_los_days'], bins=30, color='mediumpurple', edgecolor='white', alpha=0.8)
    axes[1, 1].set_xlabel('ICU Length of Stay (days)')
    axes[1, 1].set_title('ICU LOS Distribution')

    axes[1, 2].hist(df['apache_ii'], bins=20, color='seagreen', edgecolor='white', alpha=0.8)
    axes[1, 2].set_xlabel('APACHE II Score')
    axes[1, 2].set_title('Illness Severity (APACHE II)')

    plt.tight_layout()
    fig.savefig(f'{output_dir}/SuppFig01_cohort_demographics.png', bbox_inches='tight')
    plt.close()
    print("  Saved: SuppFig01_cohort_demographics.png")

    # --- Figure 2: Outcome Distributions at 12 Months ---
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle('Figure 2: 12-Month Outcome Distributions', fontsize=14, fontweight='bold')

    outcome_plots = [
        ('gose_12m', 'GOSE', 'steelblue'),
        ('mrs_12m', 'mRS', 'coral'),
        ('barthel_12m', 'Barthel Index', '#4CAF50'),
        ('drs_12m', 'DRS', '#FF9800'),
        ('cog_composite_12m', 'Cognitive Composite', '#9C27B0'),
        ('moca_12m', 'MoCA', '#00BCD4'),
        ('hads_anxiety_12m', 'HADS-Anxiety', '#F44336'),
        ('hads_depression_12m', 'HADS-Depression', '#E91E63'),
        ('sf36_pcs_12m', 'SF-36 PCS', '#3F51B5'),
        ('sf36_mcs_12m', 'SF-36 MCS', '#009688'),
        ('qolibri_os_12m', 'QOLIBRI-OS', '#795548'),
        ('social_participation_12m', 'Social Participation', '#607D8B'),
    ]

    for idx, (col, name, color) in enumerate(outcome_plots):
        ax = axes[idx // 4, idx % 4]
        data = df[col].dropna()
        ax.hist(data, bins=25, color=color, edgecolor='white', alpha=0.8)
        ax.set_title(name, fontsize=10)
        ax.axvline(data.median(), color='black', linestyle='--', linewidth=1, alpha=0.7)

    plt.tight_layout()
    fig.savefig(f'{output_dir}/SuppFig02_outcome_distributions_12m.png', bbox_inches='tight')
    plt.close()
    print("  Saved: SuppFig02_outcome_distributions_12m.png")

    # --- Figure 3: Longitudinal Trajectories by Class ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Figure 3: Longitudinal Outcome Trajectories by Class', fontsize=14, fontweight='bold')

    timepoints = [3, 6, 12]
    traj_colors = {'stable_good': '#4CAF50', 'persistent_impaired': '#F44336',
                   'improving': '#2196F3', 'deteriorating': '#FF9800'}

    traj_outcomes = [
        ('gose', 'GOSE (Functional)'),
        ('cog_composite', 'Cognitive Composite'),
        ('hads_anxiety', 'HADS-Anxiety'),
        ('hads_depression', 'HADS-Depression'),
        ('sf36_mcs', 'SF-36 Mental Component'),
        ('qolibri_os', 'QOLIBRI-OS (QoL)'),
    ]

    for idx, (var, name) in enumerate(traj_outcomes):
        ax = axes[idx // 3, idx % 3]
        for traj_class in traj_colors:
            means = []
            sems = []
            for t in timepoints:
                col = f'{var}_{t}m'
                data = df.loc[df['trajectory_class'] == traj_class, col].dropna()
                means.append(data.mean())
                sems.append(data.sem())
            ax.errorbar(timepoints, means, yerr=sems, marker='o', linewidth=2,
                       label=traj_class.replace('_', ' ').title(),
                       color=traj_colors[traj_class], capsize=3)
        ax.set_xlabel('Months Post-Injury')
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.set_xticks(timepoints)
        if idx == 0:
            ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    fig.savefig(f'{output_dir}/Figure3_longitudinal_trajectories.png', bbox_inches='tight')
    plt.close()
    print("  Saved: Figure3_longitudinal_trajectories.png")

    # --- Figure 4: Feature Importance (Top 15 per outcome) ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Figure 4: XGBoost Feature Importance (Top 15 Predictors)', fontsize=14, fontweight='bold')

    key_outcomes = ['GOSE (Functional Outcome)', 'Cognitive Composite',
                    'HADS-Anxiety', 'SF-36 Mental Component',
                    'QOLIBRI-OS (TBI-QoL)', 'Return to Work']

    for idx, outcome in enumerate(key_outcomes):
        ax = axes[idx // 3, idx % 3]
        if outcome in feature_importance_all:
            fi = feature_importance_all[outcome]
            fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]
            names = [x[0].replace('_', ' ').title() for x in fi_sorted]
            values = [x[1] for x in fi_sorted]
            ax.barh(range(len(names)), values, color='steelblue', alpha=0.8)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8)
            ax.invert_yaxis()
            ax.set_title(outcome, fontsize=10)
        else:
            ax.text(0.5, 0.5, f'{outcome}\n(not available)', ha='center', va='center',
                   transform=ax.transAxes)

    plt.tight_layout()
    fig.savefig(f'{output_dir}/report_fig4_feature_importance.png', bbox_inches='tight')
    plt.close()
    print("  Saved: report_fig4_feature_importance.png")

    # --- Figure 5: Model Comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Figure 5: Model Performance Comparison', fontsize=14, fontweight='bold')

    # Classification comparison
    class_outcomes = [k for k, v in results.items() if 'AUC' in str(v.get('XGBoost', {}))]
    if class_outcomes:
        class_data = []
        for outcome in class_outcomes:
            for model in ['XGBoost', 'Random Forest', 'Logistic LASSO']:
                if model in results[outcome]:
                    class_data.append({
                        'Outcome': outcome[:20],
                        'Model': model,
                        'AUC': results[outcome][model]['AUC_mean']
                    })
        if class_data:
            cdf = pd.DataFrame(class_data)
            cdf.pivot(index='Outcome', columns='Model', values='AUC').plot(
                kind='bar', ax=axes[0], color=['#2196F3', '#4CAF50', '#FF9800']
            )
            axes[0].set_title('Classification: AUC-ROC')
            axes[0].set_ylabel('AUC')
            axes[0].set_ylim(0.5, 1.0)
            axes[0].legend(fontsize=8)
            axes[0].tick_params(axis='x', rotation=30)

    # Regression comparison
    reg_outcomes = [k for k, v in results.items() if 'R2' in str(v.get('XGBoost', {}))][:8]
    if reg_outcomes:
        reg_data = []
        for outcome in reg_outcomes:
            for model in ['XGBoost', 'Random Forest', 'LASSO']:
                if model in results[outcome]:
                    reg_data.append({
                        'Outcome': outcome[:20],
                        'Model': model,
                        'R2': results[outcome][model]['R2_mean']
                    })
        if reg_data:
            rdf = pd.DataFrame(reg_data)
            rdf.pivot(index='Outcome', columns='Model', values='R2').plot(
                kind='bar', ax=axes[1], color=['#2196F3', '#4CAF50', '#FF9800']
            )
            axes[1].set_title('Regression: R² Score')
            axes[1].set_ylabel('R²')
            axes[1].legend(fontsize=8)
            axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    fig.savefig(f'{output_dir}/SuppFig03_model_comparison.png', bbox_inches='tight')
    plt.close()
    print("  Saved: SuppFig03_model_comparison.png")

    # --- Figure 6: Trajectory Model Selection ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Figure 6: Trajectory Model Selection (GMM)', fontsize=14, fontweight='bold')

    k_range = range(2, 7)
    axes[0].plot(k_range, bic_scores, 'bo-', linewidth=2, markersize=8)
    axes[0].axvline(optimal_k, color='red', linestyle='--', alpha=0.7)
    axes[0].set_xlabel('Number of Trajectory Classes')
    axes[0].set_ylabel('BIC')
    axes[0].set_title('BIC by Number of Classes')

    axes[1].plot(k_range, aic_scores, 'go-', linewidth=2, markersize=8)
    axes[1].axvline(optimal_k, color='red', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Number of Trajectory Classes')
    axes[1].set_ylabel('AIC')
    axes[1].set_title('AIC by Number of Classes')

    plt.tight_layout()
    fig.savefig(f'{output_dir}/SuppFig15_trajectory_class_membership.png', bbox_inches='tight')
    plt.close()
    print("  Saved: SuppFig15_trajectory_class_membership.png")

    # --- Figure 7: Correlation Heatmap ---
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    outcome_cols_12m = [c for c in df.columns if c.endswith('_12m') and c != 'mortality_12m']
    corr = df[outcome_cols_12m].corr()
    # Clean labels
    labels = [c.replace('_12m', '').replace('_', ' ').title() for c in corr.columns]
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
               xticklabels=labels, yticklabels=labels, ax=ax, vmin=-1, vmax=1,
               annot_kws={'size': 6})
    ax.set_title('Figure 7: Outcome Correlation Matrix (12-Month)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/report_fig7_outcome_correlations.png', bbox_inches='tight')
    plt.close()
    print("  Saved: report_fig7_outcome_correlations.png")

    # --- Figure 8: Deep Learning vs Classical ML ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    fig.suptitle('Figure 8: Deep Learning (MLP) vs Classical ML Performance', fontsize=14, fontweight='bold')

    comparison_data = []
    for outcome_name in dl_results:
        if 'longitudinal' not in outcome_name:
            dl_metric = dl_results[outcome_name].get('R2_mean', dl_results[outcome_name].get('AUC_mean', None))
            if dl_metric is not None:
                # Find matching classical result
                for cls_name in results:
                    if any(word in cls_name for word in outcome_name.split()):
                        xgb_metric = results[cls_name].get('XGBoost', {}).get('R2_mean',
                                     results[cls_name].get('XGBoost', {}).get('AUC_mean', None))
                        if xgb_metric is not None:
                            comparison_data.append({
                                'Outcome': outcome_name[:25],
                                'MLP (Deep Learning)': dl_metric,
                                'XGBoost (Classical)': xgb_metric,
                            })
                        break

    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        x = range(len(comp_df))
        width = 0.35
        ax.barh([i - width/2 for i in x], comp_df['XGBoost (Classical)'], width,
               label='XGBoost', color='#2196F3', alpha=0.8)
        ax.barh([i + width/2 for i in x], comp_df['MLP (Deep Learning)'], width,
               label='MLP', color='#F44336', alpha=0.8)
        ax.set_yticks(list(x))
        ax.set_yticklabels(comp_df['Outcome'], fontsize=9)
        ax.set_xlabel('Performance Metric (R² or AUC)')
        ax.legend()

    plt.tight_layout()
    fig.savefig(f'{output_dir}/report_fig8_dl_vs_classical.png', bbox_inches='tight')
    plt.close()
    print("  Saved: report_fig8_dl_vs_classical.png")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':

    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n" + "=" * 70)
    print("BRAIN INJURY AI/ML PREDICTION PIPELINE")
    print("Neurocritical Care Cohort — Multi-Domain Outcome Prediction")
    print("=" * 70)

    # Part 1: Simulate data
    df = simulate_neurocritical_cohort(n=2000)

    # Save dataset
    csv_path = f'{output_dir}/simulated_neurocritical_cohort_n2000.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n  Dataset saved: {csv_path}")

    # Generate independent hold-out test set (n=500, different seed)
    df_holdout = simulate_neurocritical_cohort(n=500, random_state=99)
    holdout_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'holdout_cohort_n500.csv')
    df_holdout.to_csv(holdout_path, index=False)
    print(f"  Hold-out test set saved: holdout_cohort_n500.csv (n=500, seed=99)")

    # Part 2: Classical ML
    results, feature_importance_all = run_classical_ml(df)

    # Part 3: Deep Learning
    dl_results = run_deep_learning(df)

    # Part 4: Trajectory modeling
    df_traj, trajectory_profiles, bic_scores, aic_scores, optimal_k = run_trajectory_modeling(df)

    # Part 5: Visualization
    generate_visualizations(df, results, feature_importance_all, dl_results,
                           df_traj, trajectory_profiles, bic_scores, aic_scores,
                           optimal_k, output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE — SUMMARY")
    print("=" * 70)
    print(f"\nDataset: {len(df)} patients, {len(df.columns)} variables")
    print(f"Outcomes predicted: {len(results)} targets")
    print(f"Models: XGBoost, Random Forest, LASSO, MLP (Deep Learning)")
    print(f"Trajectory classes discovered: {optimal_k}")
    print(f"\nFiles saved to: {output_dir}")
    print("  - simulated_neurocritical_cohort_n2000.csv")
    print("  - fig1_cohort_demographics.png")
    print("  - fig2_outcome_distributions_12m.png")
    print("  - fig3_longitudinal_trajectories.png")
    print("  - fig4_feature_importance.png")
    print("  - fig5_model_comparison.png")
    print("  - fig6_trajectory_model_selection.png")
    print("  - fig7_outcome_correlations.png")
    print("  - fig8_dl_vs_classical.png")
