# Supplementary Note 3: Causal Directed Acyclic Graph for the Simulation Data-Generating Process

**Manuscript**: A Quantitative Systems Pharmacology–Machine Learning Framework for Multi-Domain Outcome Prediction in Neurocritical Care: A Computational Simulation Study

---

## 1. Purpose

This note documents the complete causal structure of the simulation data-generating process (DGP) as a directed acyclic graph (DAG). The DAG makes explicit all assumed causal pathways, the direction and approximate magnitude of each effect, and the absence of causal paths (missing edges, representing assumed independence). Transparency about the DGP is critical for interpreting model performance, since a simulation-trained ML model can only discover associations that are encoded in the DGP; any association absent from the DAG cannot be recovered by any model.

**Relationship to real-world validity**: The DAG was constructed to reflect established causal relationships in the neurocritical care literature. Key literature sources are cited for each major pathway. Where the simulation simplified or approximated real-world causal structure, this is noted.

---

## 2. Node Taxonomy

Nodes are grouped into six layers reflecting temporal ordering and causal hierarchy:

| Layer | Category | Variables |
|---|---|---|
| L0 | Pre-injury / exogenous | Age, Sex, Education, Marital status, Employment, Prior psych history, Prior brain injury, Hypertension, Diabetes, Cardiovascular disease, Anticoagulation, Smoking, Alcohol misuse |
| L1 | Injury characterisation | Diagnosis (TBI/SAH/Stroke/ICH), GCS at admission, APACHE II |
| L2 | Acute care / ICU | ICU LOS, Mechanical ventilation days, ICP monitoring, ICP mean, Early mobilisation, Delirium, ICDSC score, ICU Anxiety score, Surgery, DVT, Pneumonia, UTI |
| L3a | Latent / trajectory | Latent severity score, Trajectory class (stable-good / persistent-impaired / improving / deteriorating) |
| L3b | QSP-ODE mechanistic | ICP/CPP features (11), Neuroinflammation features (8), Composite indices (2) |
| L4 | Outcomes (12-month) | GOSE, mRS, FIM, Barthel, DRS, Cog composite, MoCA, HADS-A, HADS-D, PHQ-9, GAD-7, PCL-5, SF-36 PCS, SF-36 MCS, QOLIBRI-OS, MPAI-4, Return-to-Work, Social Participation, Mortality |

The DAG figure (Supplementary Figure — see manuscript) shows a simplified version with node grouping for readability. The complete edge list is provided in Table 1 below.

---

## 3. Complete Edge List

**Notation**: β denotes the approximate signed linear coefficient or direction of effect in the simulation DGP. Positive β = higher value of parent increases outcome. Source: equation extracted from `brain_injury_ai_pipeline.py` and `brain_injury_qsp_hybrid.py`.

### 3.1 L0 → L1 Edges (Pre-injury → Injury characterisation)

| From | To | β | Mechanism |
|---|---|---|---|
| Age | GCS (via SAH stratum) | — | SAH Hunt-Hess grade ~ GCS; higher age → higher H&H grade in some studies |
| Age | APACHE II | +0.15 | Age is a direct APACHE II component |
| Diagnosis | GCS | varies | Diagnosis-specific GCS distributions (SAH: H&H mapped; TBI: mild/mod/severe) |
| GCS | APACHE II | −1.2 | Higher GCS (less impaired) → lower APACHE II |

### 3.2 L0/L1 → L2 Edges (Demographics/Severity → Acute care)

| From | To | β | Mechanism |
|---|---|---|---|
| GCS | ICU LOS | − | Lognormal model: log(LOS) = log(8) + (15−GCS)×0.05; severe GCS → longer stay |
| GCS | Mech ventilation | − | GCS ≤8: Lognormal(log(7),0.7); GCS 9-12: Exponential(3); GCS 13-15: 20% short course |
| GCS | ICP monitored | − | All GCS ≤8 automatically monitored; +30% probability for others |
| GCS | Delirium | − | Delirium logistic: β_GCS = +0.15 on (15−GCS) term |
| APACHE II | Delirium | implicit | Via GCS correlation |
| Age | Delirium | +0.01 | Older age → higher delirium risk |
| Age | ICU Anxiety | +0.02 | Older age → slightly elevated ICU anxiety |
| Mech vent (>3d) | Delirium | +0.5 | Ventilator-associated delirium (Jang 2025) |
| Early mobilisation | Delirium | −0.4 | Protective effect on delirium (Jia 2025) |
| Prior psych history | Delirium | +0.3 | Primed neuroinflammatory response increases delirium risk |
| Prior psych history | ICU Anxiety | +2.0 | Pre-existing psychiatric vulnerability → elevated ICU anxiety |
| Delirium | ICU Anxiety | +0.3 | Delirium-associated agitation and anxiety |
| Early mobilisation | ICU Anxiety | −0.5 | Mobilisation reduces anxiety (Jang 2025) |
| Delirium | ICDSC score | + | ICDSC conditioned on delirium_present |
| GCS (≤8) | ICP mean | − | Lower GCS → higher ICP monitoring probability; ICP Normal(12,4) |
| Diagnosis | Surgery | varies | SAH: P=0.72, ICH: P=0.45, TBI: P=0.35, Stroke: P=0.20 |
| Mech vent (>5d) | Pneumonia | +0.10 | Ventilator-associated pneumonia |
| Early mobilisation | DVT | −0.04 | Mobilisation prevents DVT |

### 3.3 L0/L1/L2 → L3a Edges (→ Latent severity / Trajectory class)

The latent severity score is a linear combination used to determine trajectory class probabilities. It has no direct effect on outcomes (all effects are mediated through trajectory class).

| From | To | β | Description |
|---|---|---|---|
| GCS | Latent severity | −0.15 | Better GCS → lower severity |
| APACHE II | Latent severity | +0.08 | Higher APACHE → higher severity |
| Age | Latent severity | +0.02 | Older age → worse severity |
| Education years | Latent severity | −0.05 | Cognitive reserve reduces effective severity |
| Prior psych history | Latent severity | +0.50 | Psychiatric vulnerability markedly worsens trajectory |
| Prior brain injury | Latent severity | +0.30 | Prior structural injury reduces resilience |
| Delirium | Latent severity | +0.40 | Delirium is a major adverse prognostic indicator |
| Early mobilisation | Latent severity | −0.30 | Early mobilisation improves trajectory |
| Mech vent (>7d) | Latent severity | +0.20 | Prolonged ventilation reflects / causes worse outcome |
| DVT | Latent severity | +0.30 | Complication → worse trajectory |
| Pneumonia | Latent severity | +0.20 | Complication → worse trajectory |
| Latent severity | Trajectory class | varies | Logistic-based probability shift: high severity → deteriorating/persistent-impaired; low severity → stable-good/improving |

**Trajectory class → all L4 outcomes**: The trajectory modifier (traj_mod, time-dependent) enters all outcome equations. This represents the dominant unmeasured confounder in the simulation — the trajectory class captures correlated outcome variance driven by unobserved patient factors.

### 3.4 L1/L2 → L3b Edges (→ QSP-ODE mechanistic features)

| From | To | Mechanism |
|---|---|---|
| GCS | AR_index | Direct: AR_index = clip(1 − (GCS−3)/12, 0.1, 0.95) |
| GCS | C₀_brain | Via severity_factor: C₀ = 0.8 × (1 − 0.4 × sev) × (1 − 0.003 × age_term) |
| GCS, Diagnosis | R_CSF | Via severity_factor × diag_factor |
| GCS, Diagnosis | k_BBB | Via severity_factor × diag_factor |
| Age | C₀_brain, k_BBB, k_edema_clear | Age reduces brain compliance and slows oedema clearance |
| APACHE II | MAP | MAP = max(70 − (APACHE−20) × 0.4, 50) |
| ICP mean | ICP_init | ODE initial condition = patient ICP at admission |
| GCS | S₀ (DAMP signal) | S₀ = 2.0 × severity × apache_factor × (1 + 0.3 × delirium) |
| APACHE II | S₀ | apache_factor = clip(APACHE/30, 0.3, 1.5) |
| Delirium | S₀ | +0.3 × delirium amplifies initial DAMP signal |
| Age | All NI ODE rate constants | Via age_factor: slows DAMP clearance, M1 resolution, M2 activation, NP production |
| Prior psych history | k_M1, k_NI (priming) | Primed microglia: priming = 1 + 0.3×psych + 0.2×alcohol |
| Alcohol misuse | k_M1, k_NI (priming) | Alcohol-associated neuroinflammatory priming |
| ICU Anxiety | k_NP_base | anxiety_icu stored in NI params (indirect neuroprotective modulation) |

All 21 mechanistic features (of which 19 have non-zero variance in the simulated cohort; mech_icp_time_above_25 and mech_ni_resolution_time are near-constant) are fully determined by the above ODE system parameters; see Supplementary Note 1 for the ODE equations.

### 3.5 L0/L1/L2/L3 → L4 Edges (→ Outcomes)

**Functional outcomes:**

| From | To | β | Notes |
|---|---|---|---|
| GCS | GOSE | +0.20 | Higher GCS → better function |
| Age | GOSE | −0.02 | Older age → worse function |
| Education years | GOSE | +0.05 | Cognitive reserve |
| Trajectory class | GOSE | ±1.5×traj_mod | Dominant driver; time-dependent |
| Delirium | GOSE | −0.30 | Delirium → worse functional outcome |
| Early mobilisation | GOSE | +0.20 | Functional benefit of early mobility (Jia 2025) |
| APACHE II | GOSE | −0.015 | Severity effect |
| GOSE | mRS | −0.70 | Derived: mRS = 6 − GOSE×0.70 + ε |
| GCS, Age, Traj | FIM Total | similar | β_GCS=+3, β_age=−0.3, β_traj=+15 |
| FIM Total | Barthel | — | Derived: linear rescaling |
| GOSE | DRS | −3.0 | Derived: DRS = 29 − GOSE×3 + ε |

**Cognitive outcomes:**

| From | To | β | Notes |
|---|---|---|---|
| GCS | Cog composite | +0.055–0.080 by domain | Via all 4 cognitive domain equations |
| Age | Cog composite | −0.005–0.010 | Age-related cognitive decline |
| Education years | Cog composite | +0.02–0.04 | Cognitive reserve |
| Trajectory class | Cog composite | +0.3–0.5×traj_mod | |
| Delirium | Cog composite | −0.3 to −0.5 | Strongest deleterious effect on cognition |
| Cog composite | MoCA | +3.0 | MoCA = 22 + 3×cog_composite + ε |

**Psychiatric outcomes (key paths — most complex):**

| From | To | β | Notes |
|---|---|---|---|
| ICU Anxiety | HADS-Anxiety | +0.60 | **Primary driver** — direct ICU anxiety → 12m anxiety (Jang 2025) |
| Prior psych history | HADS-Anxiety | +2.5 | **Strong** — pre-existing vulnerability |
| Delirium | HADS-Anxiety | +1.5 | Post-ICU delirium → anxiety |
| Alcohol misuse | HADS-Anxiety | +0.5 | Risk factor for anxiety disorder |
| Trajectory class | HADS-Anxiety | −2.0×traj_mod | Improving trajectory reduces anxiety |
| GCS | HADS-Anxiety | −0.10 | Mild injury → slightly more anxiety (awareness of injury) |
| Age | HADS-Anxiety | +0.02 | |
| Prior psych history | HADS-Depression | +2.5 | **Strong** |
| Delirium | HADS-Depression | +1.0 | |
| Marital status (married) | HADS-Depression | −0.4 | Social support protective |
| Alcohol misuse | HADS-Depression | +0.4 | |
| Trajectory class | HADS-Depression | −2.5×traj_mod | |
| HADS-Depression | PHQ-9 | +1.1 | Derived: PHQ-9 = HADS-D × 1.1 + ... |
| Prior psych history | PHQ-9 | +1.5 | Direct path (beyond HADS-D) |
| ICU Anxiety | PHQ-9 | +0.4 | Direct path |
| HADS-Anxiety | GAD-7 | +0.9 | Derived |
| HADS-Anxiety | PCL-5 | +1.5 | |
| HADS-Depression | PCL-5 | +1.0 | |
| Prior psych history | PCL-5 | +3.0 | **Strongest direct PTSD predictor** |
| Alcohol misuse | PCL-5 | +2.0 | Major PTSD risk factor (direct path) |

**Quality of life and participation outcomes:**

| From | To | β | Notes |
|---|---|---|---|
| GOSE | SF-36 PCS | +3.0 | Physical functioning mirrors functional outcome |
| Age | SF-36 PCS | −0.15 | |
| Trajectory class | SF-36 PCS | +8×traj_mod | |
| HADS-Anxiety | SF-36 MCS | −1.5 | Mental QoL driven by anxiety |
| HADS-Depression | SF-36 MCS | −1.5 | Mental QoL driven by depression |
| Trajectory class | SF-36 MCS | +5×traj_mod | |
| Cog composite | QOLIBRI-OS | +5.0 | Cognition strongly drives TBI-specific QoL |
| GOSE | QOLIBRI-OS | +2.0 | |
| HADS-Anxiety | QOLIBRI-OS | −1.0 | |
| HADS-Depression | QOLIBRI-OS | −1.0 | |
| Trajectory class | QOLIBRI-OS | +8×traj_mod | |
| GOSE | MPAI-4 | −3.0 | Inverse: higher GOSE → lower (better) MPAI-4 T-score |
| GOSE | Return-to-Work | +0.3 | Logistic model |
| Age | Return-to-Work | −0.03 | |
| Education years | Return-to-Work | +0.05 | |
| HADS-Depression | Return-to-Work | −0.1 | Depression barrier to re-employment |
| Cog composite | Return-to-Work | +0.3 | Cognitive function critical for work |
| Trajectory class | Return-to-Work | +1.5×traj_mod | |
| GOSE | Social Participation | +2.0 | |
| Marital status (married) | Social Participation | +3.0 | Social support enables participation |
| HADS-Depression | Social Participation | −0.8 | |
| Cog composite | Social Participation | +3.0 | |
| Age | Social Participation | −0.1 | |
| Trajectory class | Social Participation | +8×traj_mod | |

**Mortality:**

| From | To | β | Notes |
|---|---|---|---|
| GCS | Mortality | +0.3 (on 15−GCS term) | Severe GCS → higher mortality |
| Age | Mortality | +0.03 | |
| APACHE II | Mortality | +0.02 | |
| Diagnosis (ICH) | Mortality | +0.5 | ICH carries highest case fatality |
| Early mobilisation | Mortality | −0.3 | Protective |

---

## 4. Missing Edges (Independence Assumptions)

The following causal paths were **not** encoded in the simulation, representing explicit independence assumptions:

| Assumed absent path | Justification |
|---|---|
| Mechanistic features (L3b) → Outcomes (L4) | QSP features are predictors available to the ML model but are NOT in the simulation DGP outcome equations. This is the **key design choice**: QSP features carry predictive signal because they are correlated with GCS, APACHE II, and age (their shared causes), not because they have independent causal effects on outcomes in the simulation. |
| Sex → Psychiatric outcomes | Sex was not given a direct path to psychiatric outcomes in this simulation (sex distribution affects group composition but not individual outcome generation). |
| Diagnosis → Outcomes (beyond GCS/trajectory) | Diagnosis affects GCS distribution and surgery probability but has no direct effect on 12-month outcomes beyond these mediators. This simplification may not hold in real data (e.g., SAH-specific subarachnoid vasospasm effects). |
| Comorbidities (HTN, DM, CVD) → Outcomes | Hypertension, diabetes, and cardiovascular disease appear in the predictor set but have no direct causal path to outcomes in the DGP. They are correlated with age and therefore appear as weak predictors via confounding. |
| ICP mean → Outcomes | Mean ICP is in the predictor set and is correlated with GCS/severity (shared cause), but has no independent direct effect on outcomes in the DGP. This is a deliberate simplification — real data would show independent ICP effects on mortality. |
| Surgery → Outcomes | Surgery probability is diagnosis-dependent but has no direct outcome effect — only an indirect effect via its correlation with diagnosis. |

The absence of direct QSP feature → outcome edges is the critical design choice that explains the **modest QSP hybrid improvement** (ΔR² ≈ +0.002 for GOSE). In real-world data, QSP features may have independent biological effects not captured by clinical proxies — which is the scientific hypothesis the framework is designed to test.

---

## 5. Confounding and Mediation Structure

### 5.1 Unmeasured confounders

| Confounder | Variables it affects | Implication |
|---|---|---|
| Latent severity score | All outcomes (via trajectory class) | Trajectory class is an unmeasured latent variable driving correlated outcomes. Models trained without access to trajectory class face residual confounding that limits R². |
| Genetic/biological variation | All outcomes | Captured only by the residual noise terms (σ = 0.8–12 depending on outcome). |

### 5.2 Key mediational paths

| Path | Mediation type |
|---|---|
| GCS → Delirium → HADS-Anxiety | Partial mediation (GCS also has direct path to HADS-A) |
| Prior psych → ICU Anxiety → HADS-Anxiety | Partial mediation (prior psych also has direct path) |
| GCS → Trajectory → GOSE | Full mediation through trajectory class |
| Prior psych → Latent severity → Trajectory → Psychiatric outcomes | Indirect path (strong) |
| Alcohol → HADS-D → PHQ-9 | Partial mediation (alcohol also has direct PHQ-9 path) |
| GCS → QSP features → (no direct outcome path) | QSP features do NOT mediate GCS → outcome; they are parallel descendants of GCS |

### 5.3 Collider variables

Variables that are common effects of two causes (colliders). Conditioning on colliders opens spurious paths:

| Collider | Parents | Risk if conditioned |
|---|---|---|
| Delirium | GCS + Age + Mech vent + Prior psych | Conditioning on delirium_present in a subgroup analysis would open spurious paths between GCS and prior psych |
| Trajectory class | Latent severity (which aggregates multiple variables) | Conditioning on trajectory class in outcome analyses would induce spurious associations among its causes |
| Mortality | GCS + Age + APACHE II + Diagnosis | Survivor-only analyses introduce selection bias: among survivors, GCS and APACHE II are artificially “better” than in the full cohort |

---

## 6. Implications for Model Performance Interpretation

The DAG structure explains several observed modelling results:

1. **LASSO dominance**: LASSO’s L1 penalty performs feature selection appropriate to the sparse DGP (most edges are zero in the simulation). The true model is approximately sparse-linear, which LASSO is optimal for.

2. **QSP feature modest gain**: QSP mechanistic features are descendants of GCS and APACHE II in the DAG (L1 → L3b). Since GCS is already in the feature set, the QSP features add redundant signal. Their marginal ΔR² ≈ +0.002 for GOSE reflects this. In real data, QSP features would add independent biological signal not captured by GCS.

3. **Anxiety R² inflation**: HADS-Anxiety has three direct strong predictors (ICU_anxiety β=0.6, prior_psych β=2.5, delirium β=1.5) that are all in the feature set — this creates high R² (0.435 CV). The DGP makes HADS-Anxiety more predictable than it would be clinically because the ICU anxiety score is perfectly available.

4. **Psychiatric outcome correlations**: HADS-A, HADS-D, PHQ-9, GAD-7, PCL-5 share common causes (prior_psych, delirium, trajectory class). The MTL gain for psychiatric outcomes reflects exploitation of these shared-cause correlations by the chained XGBoost.

5. **Survivor selection bias**: With 41.4% 12-month mortality, psychiatric and QoL outcomes are assessed on a non-random survivor subset. The DAG shows that mortality and these outcomes share causes (GCS, APACHE II, age, trajectory), creating survivor selection bias that reduces apparent R² values.

6. **Trajectory classifier near-chance AUC**: The trajectory class is generated from a latent severity score that is a linear combination of admission variables — however, the stochastic component is large (noise added at each step). The admission feature set cannot perfectly recover the latent severity score, explaining the near-chance trajectory AUC (~0.52).

---

## 7. DAG Figure Description

The accompanying DAG figure (Supplementary Figure — Causal DAG) presents the simulation data-generating process with nodes colour-coded by layer and edges colour-coded by direction of effect (positive: blue; negative: red; derived/transformational: grey). Node size reflects the number of outgoing edges (causal influence breadth). For clarity, the figure groups the 13 pre-injury variables into a single “Pre-injury factors” supernode and the 21 QSP features into a “QSP-ODE features” supernode. The full edge list is provided in Table 1 (Section 3 above).

Key structural features visible in the figure:
- **Trajectory class** acts as a central hub mediating injury severity effects on all 12-month outcomes
- **Prior psychiatric history** has unusually broad downstream effects spanning cognitive, psychiatric, and functional domains via latent severity
- **ICU Anxiety score** is the only acute-care variable with a direct (non-trajectory-mediated) path to a 12-month psychiatric outcome (HADS-Anxiety)
- **QSP features** form a parallel branch from injury severity variables that does not directly connect to outcomes — visible as a “dead-end” branch in the DAG that explains the modest QSP hybrid gain

---

## References

- Jia X et al. (early mobilisation in neurocritical care). *Crit Care Med*. 2025. (Early mobilisation → delirium, outcomes)
- Jang MH et al. (ICU anxiety and cognitive outcomes). *J Intensive Care*. 2025. (ICU anxiety → psychiatric outcomes)
- Von Steinbuechel N et al. (CENTER-TBI trajectory analysis). *J Clin Med*. 2023;12(6):2246. PMID 36983247. (Trajectory class rates)
- Rass V et al. (SAH cognitive/mental health). *Neurocrit Care*. 2024;40(2):620–631. PMID 38129710. (Psychiatric outcome distributions)
- Dehbozorgi M et al. (TBI anxiety meta-analysis). *BMC Neurol*. 2024;24(1):330. PMID 39174923. (Anxiety prevalence benchmark)
- Pearl J. *Causality: Models, Reasoning, and Inference*. 2nd ed. Cambridge University Press; 2009. (DAG methodology)
- Hernán MA, Robins JM. *Causal Inference: What If*. Chapman & Hall/CRC; 2020. (Collider and mediation framework)
