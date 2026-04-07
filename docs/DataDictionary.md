---

# Data Dictionary: Simulated Neurocritical Care Cohort

**Manuscript**: A Quantitative Systems Pharmacology–Machine Learning Framework for Multi-Domain Outcome Prediction in Neurocritical Care: A Computational Simulation Study

**Files**: `simulated_neurocritical_cohort_n2000.csv` (training, n=2,000) + `simulated_holdout_n500.csv` (hold-out, n=500); `mechanistic_features_n2000.csv` (QSP-ODE features, n=2,000)

**Total variables**: 97 (main CSV) + 21 (mechanistic features CSV) = 118

---

## Section A — Patient Identifiers

| # | Variable | Type | Range / Categories | Description |
|---|---|---|---|---|
| 1 | patient_id | Integer | 1–2000 (training); 1–500 (hold-out) | Sequential patient identifier. No real patient identity — simulation only. |

---

## Section B — Demographics (5 variables)

| # | Variable | Type | Range / Categories | Units | Description | Simulation source |
|---|---|---|---|---|---|---|
| 2 | age | Integer | 18–95 | years | Age at admission. Drawn from Normal(55, 18), clipped to adult range. | CENTER-TBI, SYNAPSE-ICU (median 50–54 y) |
| 3 | sex | Binary | 0 = female, 1 = male | — | Biological sex. P(male) = 0.60 applied uniformly. See validation note on sex ratio. | TBI literature (60–70% male) |
| 4 | education_years | Integer | 6–22 | years | Completed years of formal education. Normal(12, 3). | General population proxy |
| 5 | marital_status | Ordinal | 0 = single, 1 = married/cohabitant, 2 = divorced/widowed | — | Marital/partnership status at admission. | P=[0.25, 0.55, 0.20] |
| 6 | employment_pre | Ordinal | 0 = unemployed, 1 = employed, 2 = retired, 3 = student | — | Pre-injury employment status. Age-stratified: retired P=0.85 for age ≥65; student P=0.50 for age <25. | General population |

---

## Section C — Diagnosis and Injury Severity (3 variables)

| # | Variable | Type | Range / Categories | Units | Description | Simulation source |
|---|---|---|---|---|---|---|
| 7 | diagnosis | Categorical | TBI, SAH, Stroke, ICH | — | Primary diagnosis. P=[0.40, 0.20, 0.25, 0.15]. | SYNAPSE-ICU mixed neurocritical case mix |
| 8 | gcs_admission | Integer | 3–15 | — | Glasgow Coma Scale total score at admission. Diagnosis-specific distributions: TBI mild/moderate/severe 50%/30%/20%; SAH derived from Hunt-Hess grade; Stroke Normal(12,3); ICH Normal(9,4). | CENTER-TBI, Rass 2024 |
| 9 | apache_ii | Integer | 5–45 | — | Acute Physiology and Chronic Health Evaluation II score. Derived as: 35 − GCS×1.2 + age×0.15 + N(0,4), clipped. Higher = greater severity. | SYNAPSE-ICU |

---

## Section D — Premorbid Comorbidities (8 variables)

All binary (0 = absent, 1 = present).

| # | Variable | Type | Prevalence formula | Description |
|---|---|---|---|---|
| 10 | hypertension | Binary | P = clip(0.05 + 0.005 × age, 0, 1) | Pre-existing arterial hypertension (treated or untreated at admission). |
| 11 | diabetes | Binary | P = clip(0.02 + 0.003 × age, 0, 1) | Pre-existing diabetes mellitus (type 1 or 2). |
| 12 | cardiovascular_disease | Binary | P = clip(0.01 + 0.005 × age, 0, 1) | Pre-existing ischaemic heart disease, heart failure, or peripheral arterial disease. |
| 13 | prior_psych_history | Binary | P = 0.18 | Prior diagnosis of any psychiatric disorder (mood, anxiety, psychosis, PTSD). Elevated baseline — major SHAP predictor for psychiatric outcomes. |
| 14 | prior_brain_injury | Binary | P = 0.08 | History of prior TBI, stroke, or other structural brain injury. |
| 15 | anticoagulation | Binary | P = clip(0.01 + 0.004 × age, 0, 0.5) | Prescribed anticoagulant therapy (warfarin, DOAC) at admission. |
| 16 | smoking | Binary | P = 0.35 | Current or recent (within 12 months) tobacco smoking. |
| 17 | alcohol_misuse | Binary | P = 0.15 | Alcohol misuse (AUDIT-C suggestive) or alcohol use disorder. |

---

## Section E — Acute Care and ICU Variables (12 variables)

| # | Variable | Type | Range | Units | Description | Simulation source |
|---|---|---|---|---|---|---|
| 18 | icu_los_days | Integer | 1–90 | days | Total ICU length of stay. Lognormal(log(8) + (15−GCS)×0.05, σ=0.6). Median 9 days consistent with SYNAPSE-ICU. | SYNAPSE-ICU |
| 19 | mech_ventilation_days | Integer | 0–60 | days | Days of invasive mechanical ventilation. Severe GCS: Lognormal(log(7),0.7); Moderate: Exponential(3); Mild: 20% probability of brief ventilation. | General neurocritical literature |
| 20 | icp_monitored | Binary | 0, 1 | — | ICP monitoring in situ during ICU stay. 1 if GCS ≤ 8 (mandatory) or with 30% probability (others). | Brain Trauma Foundation guidelines |
| 21 | icp_mean_mmhg | Continuous / NaN | 3–30 | mmHg | Mean ICP from invasive monitoring. NaN for non-monitored patients (icp_monitored = 0). Normal(12, 4) for monitored patients. | Clinical target <20 mmHg |
| 22 | early_mobilization | Binary | 0, 1 | — | Early physiotherapy/mobilisation initiated within first 72 hours of ICU admission. P = 0.35. | Jia 2025 (35% early mobilisation rate) |
| 23 | delirium_present | Binary | 0, 1 | — | Delirium present at any point during ICU stay (ICDSC ≥ 4 or CAM-ICU positive). Generated from logistic model incorporating GCS, APACHE II, age, mech. ventilation, early mobilisation, prior_psych_history. Prevalence 45.1%. | Patel 2018; Jang 2025 |
| 24 | icdsc_score | Integer | 0–8 | — | Intensive Care Delirium Screening Checklist total score. ≥4 = delirium. Conditional on delirium_present: Poisson(4) clipped 1–8 if delirious; Poisson(1) clipped 0–3 if not. | Bergeron et al. 2001 (ICDSC validation) |
| 25 | anxiety_icu_score | Continuous | 0–10 | 0–10 VAS | ICU anxiety rating (visual analogue or nursing assessment 0–10). Higher = greater anxiety during ICU stay. Influenced by age, prior_psych_history, delirium, early_mobilization. This variable is the strongest direct predictor of 12-month HADS-Anxiety (SHAP weight 0.6). | Jang 2025 |
| 26 | surgery | Binary | 0, 1 | — | Neurosurgical intervention during admission (craniotomy, haematoma evacuation, EVD insertion, coil/clip). Diagnosis-specific: SAH 72%, ICH 45%, TBI 35%, Stroke 20%. | Diagnosis-specific literature |
| 27 | dvt | Binary | 0, 1 | — | Deep vein thrombosis diagnosed during ICU stay. P = 0.08 + 0.04 × (1 − early_mobilization). | General ICU literature |
| 28 | pneumonia | Binary | 0, 1 | — | Hospital-acquired pneumonia (VAP or non-VAP). P = 0.15 + 0.10 × (mech_ventilation_days > 5). | General ICU literature |
| 29 | uti | Binary | 0, 1 | — | Urinary tract infection diagnosed during ICU stay. P = 0.12. | General ICU literature |

---

## Section F — Trajectory and Mortality (2 variables)

| # | Variable | Type | Categories | Description |
|---|---|---|---|---|
| 30 | trajectory_class | Categorical | stable_good, persistent_impaired, improving, deteriorating | GOSE-based longitudinal recovery trajectory class assigned by latent severity model. Base rates from CENTER-TBI (von Steinbuechel 2023) adjusted by latent severity score. Used as a generated grouping variable, not an input predictor. |
| 31 | mortality_12m | Binary | 0 = alive, 1 = deceased | All-cause mortality at 12 months post-injury. Logistic model: −3 + 0.3×(15−GCS) + 0.03×age + 0.02×APACHE II + 0.5×(ICH) − 0.3×early_mobilization + N(0,0.5). Prevalence 41.4%. |

---

## Section G — Functional Outcome Variables (15 variables × 3 timepoints = 45)

All assessed at 3, 6, and 12 months. Variable names use suffix `_3m`, `_6m`, `_12m`. Set to NaN for deceased patients at and beyond their simulated time of death.

| # | Variable root | Type | Range | Instrument | Description |
|---|---|---|---|---|---|
| 32–34 | gose | Integer | 1–8 | Glasgow Outcome Scale Extended | Primary functional outcome. 1 = vegetative state / death, 8 = upper good recovery. |
| 35–37 | mrs | Integer | 0–6 | modified Rankin Scale | Functional disability. 0 = no symptoms, 6 = death. Derived from GOSE with noise. |
| 38–40 | fim_total | Integer | 18–126 | Functional Independence Measure (total) | ADL independence. 18 = total dependence, 126 = complete independence. |
| 41–43 | barthel | Integer | 0–100 | Barthel Index | ADL performance. 0 = total dependence, 100 = independence. Derived from FIM. |
| 44–46 | drs | Integer | 0–29 | Disability Rating Scale | Disability severity. 0 = no disability, 29 = extreme vegetative state. Inverse of GOSE. |

---

## Section H — Cognitive Outcome Variables (18 variables × 3 timepoints = 18 + 3 MoCA = 21 unique roots)

| # | Variable root | Type | Range / Units | Instrument | Description |
|---|---|---|---|---|---|
| 47–49 | cog_memory | Continuous | z-score | Simulated neuropsych battery | Memory domain z-score. Mean ~−0.3 (impaired post-injury). |
| 50–52 | cog_executive | Continuous | z-score | Simulated neuropsych battery | Executive function domain z-score. |
| 53–55 | cog_attention | Continuous | z-score | Simulated neuropsych battery | Sustained attention domain z-score. |
| 56–58 | cog_visuoconst | Continuous | z-score | Simulated neuropsych battery | Visuoconstructive ability domain z-score. |
| 59–61 | cog_composite | Continuous | z-score | Composite | Mean of 4 domain z-scores. Primary cognitive outcome for ML modelling. |
| 62–64 | moca | Integer | 0–30 | Montreal Cognitive Assessment | Screening cognitive assessment. <26 = mild impairment. Derived from cog_composite with noise. Prevalence MoCA <26 = 60.1%. |

---

## Section I — Psychiatric Outcome Variables (5 roots × 3 timepoints = 15 variables)

| # | Variable root | Type | Range | Instrument | Cut-off | Description |
|---|---|---|---|---|---|---|
| 65–67 | hads_anxiety | Integer | 0–21 | Hospital Anxiety and Depression Scale – Anxiety subscale | ≥8 = probable anxiety | Anxiety symptoms. Prevalence >7 = 63.3% (above benchmark; see validation note). |
| 68–70 | hads_depression | Integer | 0–21 | Hospital Anxiety and Depression Scale – Depression subscale | ≥8 = probable depression | Depressive symptoms. Prevalence >7 = 17.5% (consistent with Rass 2024 benchmark 16%). |
| 71–73 | phq9 | Integer | 0–27 | Patient Health Questionnaire-9 | ≥10 = moderate depression | PHQ-9 depression severity. Derived from hads_depression + direct psych history and ICU anxiety paths. |
| 74–76 | gad7 | Integer | 0–21 | Generalized Anxiety Disorder-7 | ≥10 = moderate anxiety | GAD-7 anxiety severity. Derived from hads_anxiety with noise. |
| 77–79 | pcl5 | Integer | 0–80 | PTSD Checklist DSM-5 | ≥33 = probable PTSD | PTSD symptom severity. Driven by HADS-Anxiety, HADS-Depression, prior psych history, alcohol misuse. |

---

## Section J — Quality of Life Outcome Variables (4 roots × 3 timepoints = 12 variables)

| # | Variable root | Type | Range | Instrument | Description |
|---|---|---|---|---|---|
| 80–82 | sf36_pcs | Continuous | 0–100 | SF-36 Physical Component Summary | HRQoL physical domain. Higher = better. Driven by GOSE and age. |
| 83–85 | sf36_mcs | Continuous | 0–100 | SF-36 Mental Component Summary | HRQoL mental domain. Higher = better. Driven by psychiatric outcomes. |
| 86–88 | qolibri_os | Continuous | 0–100 | QOLIBRI-OS (Overall Scale) | TBI-specific quality of life. 0 = worst, 100 = best. Driven by cognition, GOSE, and psychiatric outcomes. |
| 89–91 | mpai4_tscore | Continuous | 10–80 | Mayo-Portland Adaptability Inventory-4 (T-score) | Neurorehabilitation outcome. Lower T-score = better. |

---

## Section K — Participation and Social Outcomes (2 roots × 3 timepoints = 6 variables)

| # | Variable root | Type | Range | Instrument | Description |
|---|---|---|---|---|---|
| 92–94 | return_to_work | Binary | 0, 1 | — | Return to pre-injury employment. Only applicable to pre-injury employed (employment_pre = 1) and working-age (age < 65) patients; NaN for others. Logistic model incorporating GOSE, age, education, trajectory, depression, cognition. |
| 95–97 | social_participation | Continuous | 0–100 | Composite score | Social re-integration score. Driven by GOSE, marital status, depression, cognition, age. |

---

## Section L — Mechanistic Features (separate file: `mechanistic_features_n2000.csv`)

21 features extracted from QSP-ODE simulations (see Supplementary Note 1 for mathematical specification). All continuous. Joined to main cohort by `patient_id`.

### L1 — ICP/CPP Features (11)

| # | Variable | Units | Range (approx.) | Description |
|---|---|---|---|---|
| 1 | mech_icp_peak | mmHg | 8–60 | Maximum simulated ICP over 7 days |
| 2 | mech_icp_mean_72h | mmHg | 5–35 | Mean ICP in first 72 hours |
| 3 | mech_icp_auc_7d | mmHg·h | 800–8,000 | Area under ICP–time curve over 7 days |
| 4 | mech_icp_time_above_20 | hours | 0–168 | Cumulative hours ICP > 20 mmHg |
| 5 | mech_icp_time_above_25 | hours | 0–168 | Cumulative hours ICP > 25 mmHg |
| 6 | mech_icp_at_day7 | mmHg | 5–40 | ICP at end of 7-day window |
| 7 | mech_cpp_min | mmHg | 10–80 | Minimum CPP over 7 days |
| 8 | mech_cpp_mean | mmHg | 40–90 | Mean CPP over 7 days |
| 9 | mech_cpp_time_below_60 | hours | 0–168 | Cumulative hours CPP < 60 mmHg |
| 10 | mech_ar_index | 0–1 | 0.10–0.95 | Autoregulation index (0 = intact, 1 = pressure-passive) |
| 11 | mech_cpp_optimal_time | fraction | 0–1 | Proportion of time CPP in 60–100 mmHg therapeutic range |

### L2 — Neuroinflammation Features (8)

| # | Variable | Units | Description |
|---|---|---|---|
| 12 | mech_m1_peak | arb. units | Maximum M1 (pro-inflammatory microglial) activation |
| 13 | mech_ni_peak | arb. units | Maximum neuroinflammation index (cytokine burden proxy) |
| 14 | mech_ni_auc_7d | arb. units·h | NI area under curve over first 7 days |
| 15 | mech_m1_m2_ratio_72h | ratio | M1/M2 ratio in first 72 hours (higher = more pro-inflammatory) |
| 16 | mech_ni_resolution_time | hours | Time for NI to fall below 10% of peak |
| 17 | mech_np_steady_state | arb. units | Neuroprotection index at 30-day steady state |
| 18 | mech_np_auc | arb. units·h | Total neuroprotective exposure over 30 days |
| 19 | mech_m2_m1_dominance | ratio | M2/M1 ratio at steady state (higher = more reparative) |

### L3 — Composite Indices (2)

| # | Variable | Range | Description |
|---|---|---|---|
| 20 | mech_secondary_injury_index | 0–3 | Composite ICP burden + CPP deficit + neuroinflammation |
| 21 | mech_recovery_potential | 0–∞ | Composite neuroprotection × reparative balance × autoregulation |

---

## Notes on Missing Data

| Variable | Missing data pattern | Reason |
|---|---|---|
| icp_mean_mmhg | ~65% missing (NaN) | Only measured in monitored patients (icp_monitored = 1). Imputed to median for ML modelling. |
| All outcome variables | Missing for deceased patients beyond time of death | Set to NaN. Death time distributed as: 40% before 3m, 30% between 3–6m, 30% between 6–12m. |
| return_to_work | NaN for non-employed and retired patients | Only applicable to pre-injury employed (employment_pre = 1) and working-age (age < 65). |

---

## Coding Reference

| Marital status | Code |
|---|---|
| Single / never married | 0 |
| Married / cohabitant | 1 |
| Divorced / widowed | 2 |

| Employment status | Code |
|---|---|
| Unemployed | 0 |
| Employed (full or part-time) | 1 |
| Retired | 2 |
| Student | 3 |

| Trajectory class | Code | Description |
|---|---|---|
| stable_good | 0 | Stable, favourable functional status |
| persistent_impaired | 1 | Persistent disability without recovery |
| improving | 2 | Progressive functional improvement |
| deteriorating | 3 | Secondary decline after initial stabilisation |

---

*All data are computationally simulated. No real patient data were used. The simulation was designed for methodological pipeline development and does not replicate any specific real-world dataset.*
