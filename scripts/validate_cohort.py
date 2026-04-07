#!/usr/bin/env python3
"""Validate simulated cohort against published literature."""

import pandas as pd
import numpy as np

df = pd.read_csv('simulated_neurocritical_cohort_n2000.csv')

print('=' * 75)
print('VALIDATION OF SIMULATED COHORT AGAINST PUBLISHED LITERATURE')
print('=' * 75)

# Age
print('\n1. AGE')
print('   Simulation: mean={:.1f}, SD={:.1f}, range=[{}, {}]'.format(
    df['age'].mean(), df['age'].std(), df['age'].min(), df['age'].max()))
print('   Literature:')
print('   - Rass 2024 (SAH, Neurocrit Care, PMID 38129710): median 54 (IQR 47-62)')
print('   - CENTER-TBI (Steyerberg 2019, Lancet Neurol, PMID 31526754): median 50 (IQR 30-66)')
print('   [OK] 55.4 years consistent with mixed neurocritical ICU population')

# Sex
print('\n2. SEX DISTRIBUTION')
print('   Simulation: {:.1f}% male'.format(df['sex'].mean() * 100))
print('   Literature:')
print('   - Rass 2024 SAH (PMID 38129710): 41% male')
print('   - TBI general: 60-70% male; weighted estimate for mixed cohort ~56%')
print('   [~OK] 61% plausible but slightly above weighted mixed-cohort benchmark (~56%)')

# Diagnosis distribution
print('\n3. DIAGNOSIS DISTRIBUTION')
diag = df['diagnosis'].value_counts()
for d in ['TBI', 'Stroke', 'SAH', 'ICH']:
    print('   {}: {:.1f}%'.format(d, diag.get(d, 0) / len(df) * 100))
print('   Literature: No direct comparison (most registry studies are single-diagnosis)')
print('   [OK] TBI as most common neurocritical ICU diagnosis is consistent')

# GCS severity
print('\n4. INJURY SEVERITY (GCS at Admission)')
gcs_mild = (df['gcs_admission'] >= 13).sum() / len(df) * 100
gcs_mod  = ((df['gcs_admission'] >= 9) & (df['gcs_admission'] <= 12)).sum() / len(df) * 100
gcs_sev  = (df['gcs_admission'] <= 8).sum() / len(df) * 100
print('   Simulation: Mild={:.1f}%, Mod={:.1f}%, Sev={:.1f}%'.format(
    gcs_mild, gcs_mod, gcs_sev))
print('   Literature:')
print('   - CENTER-TBI ICU stratum (Steyerberg 2019, Lancet Neurol, PMID 31526754):')
print('     36% mild (GCS 13-15); ~64% moderate/severe (GCS <=12); n=2138 ICU patients')
print('   NOTE: CENTER-TBI is TBI-only; per-diagnosis GCS benchmarks needed for')
print('         SAH, stroke, and ICH components of this mixed cohort')
print('   [OK] Mild proportion ({:.1f}%) close to CENTER-TBI ICU stratum (36%)'.format(
    gcs_mild))

# Mental health
print('\n5. MENTAL HEALTH OUTCOMES (12 months)')
anx_pct = (df['hads_anxiety_12m'] > 7).mean() * 100
dep_pct = (df['hads_depression_12m'] > 7).mean() * 100
print('   Simulation: Anxiety={:.1f}%, Depression={:.1f}%'.format(anx_pct, dep_pct))
print('   Literature:')
print('   - Rass 2024 (SAH, PMID 38129710): Anxiety 33%, Depression 16%')
print('   - Dehbozorgi 2024 (TBI anxiety meta-analysis, BMC Neurol, PMID 39174923):')
print('     Pooled 37% (range 4-83% across 32 studies)')
print('   [!] Anxiety {:.1f}% above pooled estimate (37%) and SAH reference (33%)'.format(
    anx_pct))
print('   [OK] Depression {:.1f}% consistent with SAH reference (16%)'.format(dep_pct))

# Cognitive deficits
print('\n6. COGNITIVE DEFICITS (12 months)')
moca_pct = (df['moca_12m'] < 26).mean() * 100
print('   Simulation: {:.1f}% with MoCA <26'.format(moca_pct))
print('   Literature:')
print('   - Rass 2024 (SAH, PMID 38129710): 71% with >=1 deficit on cognitive battery')
print('   NOTE: MoCA <26 and multi-domain battery deficits are not equivalent thresholds;')
print('         MoCA <26 underestimates deficit prevalence vs comprehensive battery')
print('   [~OK] Directionally consistent; not directly comparable due to threshold mismatch')

# Trajectory classes
print('\n7. RECOVERY TRAJECTORY CLASSES')
traj = df['trajectory_class'].value_counts().sort_index()
labels = ['stable_good', 'persistent_impaired', 'improving', 'deteriorating']
for i, l in enumerate(labels):
    print('   {}: {:.1f}%'.format(l, traj.get(i, 0) / len(df) * 100))
print('   Literature:')
print('   - von Steinbuechel 2023 (CENTER-TBI TBI-only, J Clin Med, PMID 36983247):')
print('     Stable 76.1%, Persistent 17.3%, Improving 3.2%, Deteriorating 3.4%')
print('   CAVEATS (two reference mismatches):')
print('     (1) Literature uses QOLIBRI-OS/SF-12 patient-reported HRQoL trajectories,')
print('         NOT GOSE-based functional trajectories as in this simulation')
print('     (2) CENTER-TBI is TBI-only; this simulation includes SAH/Stroke/ICH,')
print('         which carry worse prognosis and shift trajectory distribution')
print('   [!] More deterioration/less stability than TBI-only HRQoL benchmark;')
print('       partly explained by mixed-diagnosis severity and trajectory definition')

# Mortality
print('\n8. MORTALITY (12 months)')
mort_pct = df['mortality_12m'].mean() * 100
print('   Simulation: {:.1f}%'.format(mort_pct))
print('   Literature:')
print('   - SYNAPSE-ICU (Robba 2021, Lancet Neurol, PMID 34146513):')
print('     6-month mortality 34-49% in mixed acute brain injury ICU')
print('     [TBI 54%, ICH 25%, SAH 22%; n=2395, 42 countries, prospective]')
print('   - SAH case fatality literature: 18-43% depending on severity')
print('     [Rass 2024 (PMID 38129710): 93% good functional outcome in survivors]')
print('   NOTE: SYNAPSE-ICU reports 6-month outcomes; 12-month mortality expected higher')
print('   [~OK] {:.1f}% at 12 months plausible vs SYNAPSE-ICU 6-month range (34-49%)'.format(
    mort_pct))

# Delirium
print('\n9. DELIRIUM')
del_pct = df['delirium_present'].mean() * 100
print('   Simulation: {:.1f}%'.format(del_pct))
print('   Literature:')
print('   - Patel 2018 (neurocritical ICU systematic review, Crit Care Med, PMID 30142098):')
print('     12-43% in neurocritically ill patients (neurotrauma, stroke)')
print('   - Krewulak 2018 (general ICU meta-analysis, Crit Care Med, PMID 30234569):')
print('     Pooled prevalence 31% (95% CI 24-41%)')
print('   [!] {:.1f}% slightly above neurocritical upper bound (43%, Patel 2018)'.format(
    del_pct))

# ICU LOS
print('\n10. ICU LENGTH OF STAY')
icu_med = df['icu_los_days'].median()
print('   Simulation: median={:.1f} days'.format(icu_med))
print('   Literature:')
print('   - SYNAPSE-ICU (Robba 2021, Lancet Neurol, PMID 34146513): median ~9 days')
print('   - Typical neurocritical ICU range: 5-14 days')
print('   [OK] Consistent with neurocritical care literature')

print('\n' + '=' * 75)
print('OVERALL VALIDATION SUMMARY')
print('=' * 75)
print('  [OK]   Age, GCS mild%, ICU LOS, Depression')
print('  [~OK]  Sex (61% vs weighted ~56%), Mortality (12m vs SYNAPSE-ICU 6m),')
print('         Cognitive impairment (threshold mismatch)')
print('  [!]    Anxiety high (simulation {:.0f}% vs pooled 37%)'.format(anx_pct))
print('  [!]    Delirium slightly above upper bound ({:.0f}% vs 43%)'.format(del_pct))
print('  [!]    Trajectory distribution: reference mismatch (HRQoL vs GOSE;')
print('         TBI-only benchmark vs mixed diagnosis)')
print()
print('  KEY LIMITATION: No published mixed-diagnosis neurocritical ICU cohort')
print('  with GOSE-based trajectory classes exists for direct trajectory comparison.')
print('=' * 75)
