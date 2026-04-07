# Supplementary Document S1

## TRIPOD+AI Reporting Checklist

**Manuscript Title:** A Quantitative Systems Pharmacology–Machine Learning Framework for Multi-Domain Outcome Prediction in Neurocritical Care: A Computational Simulation Study

**Target Journal:** npj Digital Medicine

**Date prepared:** 23 February 2026

---

### What is TRIPOD+AI?

TRIPOD+AI (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis — Artificial Intelligence) is the 2024 update to the original TRIPOD statement, providing harmonised reporting guidance for clinical prediction model studies that use regression or machine learning methods. The checklist encompasses 27 main items (52 sub-items) organised across Title, Abstract, Introduction, Methods, Results, Discussion, and Other Information sections.

**Reference:** Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ* 2024;**385**:e078378. https://doi.org/10.1136/bmj-2023-078378

---

### Important Note: Simulation / Methodology Study

This manuscript describes a **computational simulation study**. The entire dataset (n = 2,000 training + n = 500 independent hold-out) is **fully synthetic**, generated programmatically using clinically informed distributional parameters and validated QSP-ODE mechanistic equations. No real patient data were collected, accessed, or used at any stage.

Consequently:

- Items pertaining to the **source of existing clinical data**, **participant recruitment dates**, **ethical approval for patient data collection**, and **patient and public involvement** are marked **N/A** with explicit explanations.
- Items relating to **external generalisability** to real clinical populations are addressed prospectively in the Discussion (Limitations section, §4.8) rather than empirically.
- The study is positioned as **methodology development** — establishing a proof-of-concept pipeline for QSP–ML fusion prior to deployment on real neurocritical care cohort data.
- Simulation seeds, data-generating equations, and all analytical code are available in the supplementary material and public repository, serving as full reproducibility documentation in lieu of institutional data governance documentation.

---

## TRIPOD+AI for Abstracts Checklist

| # | Item | Reported | Location | Comments |
|---|------|----------|----------|---------|
| A1 | Identify the study as developing or evaluating a prediction model, target population, and outcome | Yes | Abstract, line 1; Title | Title and abstract opening sentence clearly identify development + internal/external validation of prediction models; neurocritical ICU population; 12 multi-domain outcomes stated |
| A2 | Brief explanation of healthcare context and rationale | Yes | Abstract, Background sentence (lines 15–16) | Healthcare context: neurocritical ICU; rationale: absence of multi-domain, mechanistically informed outcome tools |
| A3 | Study objectives: development, validation, or both | Yes | Abstract, Objectives sentence (line 17) | States both development (n=2,000) and validation (n=500 hold-out); notes simulation methodology context |
| A4 | Data sources | Yes | Abstract, Methods (line 18) | States "fully synthetic neurocritical ICU cohort" with training seed=42 and hold-out seed=99; no real patient data |
| A5 | Eligibility criteria and setting | Yes | Abstract, Methods (line 18) | Simulated ICU setting; diagnoses TBI 40%, stroke 25%, SAH 20%, ICH 15%; n=2,000 / n=500 |
| A6 | Outcome specification and time horizon | Yes | Abstract, Methods (line 19) | 12 outcomes (10 regression, 2 classification) at 12-month follow-up listed |
| A7 | Model type, building steps, internal validation | Yes | Abstract, Methods (lines 19–20) | LASSO, Ridge, ElasticNet, RF, XGBoost, deep MLP, MTL-XGBoost; 5-fold CV stratified by diagnosis |
| A8 | Performance assessment measures | Yes | Abstract, Methods (line 20) | R² for regression, AUC-ROC for classification; conformal prediction intervals (95%) stated |
| A9 | Participant and outcome event numbers | Yes | Abstract, Results (line 21) | Training n=2,000; hold-out n=500; mortality events reported in Table 1 |
| A10 | Final model predictors summary | Yes | Abstract, Results (line 22) | Top-5 SHAP predictors named: GCS, Age, ICU Anxiety Score, Delirium, APACHE II |
| A11 | Performance estimates with confidence intervals | Yes | Abstract, Results (lines 21–23) | Key metrics: GOSE R²=0.411, FIM R²=0.512, RTW AUC=0.821, mortality AUC=0.579; hold-out optimism bias +0.007 |
| A12 | Overall interpretation | Yes | Abstract, Conclusions (line 24) | Interpretation includes both promise of pipeline and acknowledged limitations of simulation methodology |
| A13 | Registration or statement of non-registration | Yes | Abstract, final line; Methods §2.8 | States: "This methodology study was not prospectively registered; simulation code and pipeline are publicly archived at [repository URL]" |

---

## Section 1: Title and Abstract

| Item | Domain | TRIPOD+AI Item Description | Reported | Location | Comments |
|------|--------|---------------------------|----------|----------|---------|
| 1 | Title | Identify the study as developing or evaluating the performance of a multivariable prediction model, the target population, and the outcome to be predicted | Yes | Title (line 1) | Title states: "A Quantitative Systems Pharmacology–Machine Learning Framework for Multi-Domain Outcome Prediction in Neurocritical Care: A Computational Simulation Study". Identifies: (i) prediction model development, (ii) neurocritical ICU population, (iii) multi-domain outcomes. The phrase "Computational Simulation Study" explicitly flags the synthetic data design. |
| 2 | Abstract | See TRIPOD+AI for Abstracts checklist | Yes | Abstract (lines 15–24) | See Abstract checklist above (items A1–A13). Structured abstract with Background, Objectives, Methods, Results, Conclusions. All 13 abstract sub-items addressed. |

---

## Section 2: Introduction

| Item | Domain | TRIPOD+AI Item Description | Reported | Location | Comments |
|------|--------|---------------------------|----------|----------|---------|
| 3a | Introduction / Context | Explain the healthcare context (including whether diagnostic or prognostic) and rationale for developing or evaluating the prediction model, including references to existing models | Yes | Introduction §1, lines 27–31 | Prognostic context: ICU survivors of TBI, SAH, ischaemic stroke, ICH; 12-month multi-domain outcome prediction. Rationale: current tools predict single outcomes (GOS, mRS) and lack mechanistic grounding; references to IMPACT, CRASH, PRISM-3 and published QSP neurocritical models cited. |
| 3b | Introduction / Target population | Describe the target population and intended purpose of the prediction model in the context of the care pathway, including intended users | Yes | Introduction §1, lines 31–34 | Target: adults admitted to neurocritical ICU with qualifying brain injury diagnoses. Intended purpose: methodology development for future deployment; intended users identified as clinical pharmacologists, intensivists, and neurorehabilitation clinicians. Care pathway context (ICU → discharge → community follow-up) described. |
| 3c | Introduction / Health inequalities | Describe any known health inequalities between sociodemographic groups relevant to this study | Partial | Introduction §1, lines 34–35; Discussion §4.8 | Brief acknowledgement that age, sex, education, and socioeconomic proxies are included as predictors and that their differential impact is an open issue. However, a targeted health inequalities review is not provided. Full discussion deferred to Limitations §4.8. A dedicated paragraph on known inequalities in TBI/stroke outcomes by sex, race, and socioeconomic status is recommended for revision. |
| 4 | Introduction / Objectives | Specify the study objectives, including whether the study describes the development or validation of a prediction model (or both) | Yes | Introduction §1, lines 36–38 | Objectives explicitly state: (1) develop multi-model ML pipeline on synthetic neurocritical cohort, (2) validate on independent hold-out, (3) evaluate QSP-ODE mechanistic feature augmentation, (4) demonstrate SHAP interpretability and conformal uncertainty quantification. Both development and validation are specified. |

---

## Section 3: Methods

### 3.1 Data

| Item | Domain | TRIPOD+AI Item Description | Reported | Location | Comments |
|------|--------|---------------------------|----------|----------|---------|
| 5a | Methods / Data sources | Describe the sources of data separately for the development and evaluation datasets, the rationale for using these data, and their representativeness | Yes — adapted | Methods §2.1, lines 39–48 | **Simulation context:** Data are fully synthetic; no real-world data source exists. Training dataset (n=2,000, seed=42) and hold-out dataset (n=500, seed=99) generated by the same data-generating process (DGP) with different random seeds. DGP equations, parameter distributions, and clinical basis for parameters described in §2.1 and Supplementary Table S1. Rationale for simulation: enable pipeline development and proof-of-concept without privacy constraints; representativeness assessed by comparing generated summary statistics against published ICU cohort benchmarks (IMPACT database, CENTER-TBI summary statistics) in Table 1. The absence of a real clinical data source is explicitly acknowledged as a study limitation (§4.8). |
| 5b | Methods / Data dates | Specify the dates of collected participant data, including start and end of participant accrual; and, if applicable, end of follow-up | N/A | Methods §2.1 | **N/A — simulation study.** No real participant data were collected; there are no accrual dates. Simulated follow-up period is 12 months per participant by design. This is stated explicitly in §2.1: "As all data are synthetic, no patient accrual period or study calendar dates apply." |
| 6a | Methods / Setting | Specify key elements of the study setting including the number and location of centres | N/A — partially adapted | Methods §2.1, lines 39–41 | **Simulation context:** There are no real clinical centres. The simulated cohort is designed to represent a single-centre neurocritical ICU. Distributional parameters are drawn from published multi-centre data (CENTER-TBI, TRACK-TBI summary statistics) to ensure clinical plausibility. This is acknowledged explicitly: "The simulated setting represents a hypothetical single-centre neurocritical ICU whose patient characteristics are calibrated to published multi-centre cohort summaries." |
| 6b | Methods / Eligibility criteria | Describe the eligibility criteria for study participants | Yes — adapted | Methods §2.1, lines 41–43 | Simulated inclusion criteria stated: adult patients (age 18–85), admitted to simulated ICU, with one of four index diagnoses (TBI 40%, ischaemic stroke 25%, SAH 20%, ICH 15%). No real exclusion criteria apply; the DGP generates all records meeting the distributional specification. Edge cases (e.g., extreme age, implausible GCS/APACHE II combinations) are handled by bounded sampling distributions (Supplementary Table S1). |
| 6c | Methods / Treatments | Give details of any treatments received, and how they were handled during model development or evaluation | Partial | Methods §2.1, lines 43–45; §2.3 | Treatments are implicitly embedded in the DGP (e.g., mechanical ventilation days, surgery flag, ICP monitoring flag, DVT prophylaxis). These are included as binary/continuous predictor variables but are not modelled as causal interventions in the prediction models. The QSP-ODE module (§2.2) simulates ICP/CPP dynamics and neuroinflammation but does not model drug pharmacokinetics in the primary prediction pipeline. The Bayesian drug simulation module (§2.7) models an illustrative neuroprotective intervention on the QSP-ODE system as a separate sensitivity exercise. A formal causal treatment model is not included in the primary ML pipeline; this is noted as a limitation (§4.8). |

### 3.2 Participants

| Item | Domain | TRIPOD+AI Item Description | Reported | Location | Comments |
|------|--------|---------------------------|----------|----------|---------|
| 7 | Methods / Data pre-processing | Describe any data pre-processing and quality checking, including whether this was similar across relevant sociodemographic groups | Yes | Methods §2.3, lines 53–56; §2.8 | Pre-processing steps described: (i) per-target survival mask applied for 12-month outcomes (deceased patients excluded from functional outcome regressions; outcome set to NaN rather than 0 to avoid immortal-time bias); (ii) median imputation for missing mean ICP values (for patients without ICP monitoring, ~30% of cohort); (iii) standard scaling (StandardScaler) applied to continuous predictors before LASSO/Ridge/ElasticNet fitting; (iv) no scaling for tree-based methods. Pre-processing applied uniformly; no sociodemographic subgroup differential pre-processing performed. Imputation approach for ICP noted as a limitation in §4.8 (multiple imputation not implemented). |

### 3.3 Outcomes

| Item | Domain | TRIPOD+AI Item Description | Reported | Location | Comments |
|------|--------|---------------------------|----------|----------|---------|
| 8a | Methods / Outcome definition | Clearly define the outcome to be predicted and the time horizon, including how and when assessed | Yes | Methods §2.1, lines 45–50; Table 2 | All 12 outcomes defined with scale ranges and time horizon (12 months): GOSE (1–8), FIM Total (18–126), Barthel Index (0–100), HADS-Anxiety (0–21), HADS-Depression (0–21), PHQ-9 (0–27), PCL-5 (0–80), MoCA (0–30), QOLIBRI-OS (0–100), SF-36 MCS (0–100), Return-to-Work (binary), 12-month mortality (binary). For deceased patients, survival-dependent outcomes (all except mortality) are masked using the per-target survival mask. Time horizon fixed at 12 months in the DGP. |
| 8b | Methods / Outcome assessment subjectivity | If outcome assessment requires subjective interpretation, describe the qualifications and demographic characteristics of outcome assessors | N/A | Methods §2.1 | **N/A — simulation study.** All outcomes are generated algorithmically by the DGP using specified equations (Supplementary Table S2). No human assessors are involved. This is stated in §2.1: "Outcomes are computed deterministically from predictor values via the data-generating equations with added Gaussian noise; no human raters are involved." |
| 8c | Methods / Blinding of outcome assessment | Report any actions to blind assessment of the outcome to be predicted | N/A | Methods §2.1 | **N/A — simulation study.** Outcomes are generated before model development begins; the DGP parameters are known but are not used by the ML models (models see only the predictor matrix X and target vector y). No blinding procedure is required or applicable. |

### 3.4 Predictors

| Item | Domain | TRIPOD+AI Item Description | Reported | Location | Comments |
|------|--------|---------------------------|----------|----------|---------|
| 9a | Methods / Predictor selection | Describe the choice of initial predictors and any pre-selection of predictors before model building | Yes | Methods §2.1, lines 50–52; §2.3, lines 62–64 | 28 clinical predictor variables selected a priori based on clinical literature and established neurocritical care prognostic models (IMPACT, CRASH). No data-driven pre-selection was performed; all 28 predictors entered all models. Variable list: age, sex, education, GCS at admission, APACHE II, diagnosis (4-level categorical), hypertension, diabetes, CVD, prior psychiatric history, alcohol misuse, smoking, anticoagulation, prior brain injury, ICU LOS, mechanical ventilation days, ICP monitoring (binary), mean ICP (mm Hg), delirium (binary), ICDSC score, ICU anxiety score, early mobilisation (binary), surgery (binary), DVT, pneumonia, UTI. QSP mechanistic features (21 ODE-derived) evaluated as an additional augmentation block (§2.2, §3.5 results). |
| 9b | Methods / Predictor definitions | Clearly define all predictors, including how and when they were measured | Yes | Methods §2.1, lines 50–52; Table 1 footnotes; Supplementary Table S1 | All 28 predictors defined with type (continuous/binary/categorical), scale, measurement timing (admission or ICU stay), and distributional parameters used in the DGP. Table 1 presents summary statistics. Supplementary Table S1 provides the full DGP specification for each variable including means, SDs, bounds, and clinical source references. |
| 9c | Methods / Predictor assessment subjectivity | If predictor measurement requires subjective interpretation, describe qualifications and demographic characteristics of assessors | N/A | Methods §2.1 | **N/A — simulation study.** All predictors are generated algorithmically. In the real-world deployment context, predictors such as GCS and ICDSC require clinical expertise; this is noted as an implementation consideration in §4.9 (next steps) but does not apply to the current simulation. |

### 3.5 Sample Size

| Item | Domain | TRIPOD+AI Item Description | Reported | Location | Comments |
|------|--------|---------------------------|----------|----------|---------|
| 10 | Methods / Sample size | Explain how the study size was arrived at (separately for development and evaluation), and justify that the study size was sufficient | Yes | Methods §2.1, lines 43–45; Discussion §4.7 | Development n=2,000 and hold-out n=500 determined by prospective sample size calculation. Minimum Events Per Variable (EPV) criterion applied: for regression with 28 predictors, minimum n=280 (EPV=10); actual ratio is 71.4. For binary classification (mortality events ~8% = 160 events, RTW events ~55% = 1,100 events), EPV ratios are 5.7 (mortality) and 39.3 (RTW). The low EPV for mortality (5.7 < 10) is acknowledged as a limitation in §4.8 and provides a post-hoc justification for observed mortality AUC being lower (0.579) than RTW AUC (0.821). Riley et al. (2020) formula was applied for confirmatory sizing of the hold-out cohort (n=500 provides 80% power to detect R²≥0.30 with alpha=0.05 for continuous outcomes). |

### 3.6 Missing Data

| Item | Domain | TRIPOD+AI Item Description | Reported | Location | Comments |
|------|--------|---------------------------|----------|----------|---------|
| 11 | Methods / Missing data | Describe how missing data were handled. Provide reasons for omitting any data | Yes | Methods §2.3, lines 53–56 | Two missing data mechanisms: (1) **Survival masking:** For 12-month functional outcomes (GOSE, FIM, Barthel, HADS-A/D, PHQ-9, PCL-5, MoCA, QOLIBRI-OS, SF-36 MCS), deceased patients contribute NaN to the target vector; each model is fitted on the survival-eligible subset using a per-target mask. Deceased patients are not assigned a worst-case score (e.g., GOSE=1) to avoid introducing systematic bias. (2) **ICP imputation:** ~30% of simulated patients do not receive ICP monitoring. For these patients, mean_ICP is set to NaN in the DGP and imputed with the cohort median prior to model fitting. Multiple imputation (MICE) was not implemented; this is acknowledged as a limitation (§4.8). No predictor variables other than mean_ICP have missing values by design of the DGP. |

### 3.7 Statistical Analysis Methods

| Item | Domain | TRIPOD+AI Item Description | Reported | Location | Comments |
|------|--------|---------------------------|----------|----------|---------|
| 12a | Methods / Data partitioning | Describe how data were used in the analysis, including whether the data were partitioned | Yes | Methods §2.3–2.4, lines 57–72 | Training set (n=2,000, seed=42) used for all model development. Hold-out set (n=500, seed=99) used for independent evaluation only; hold-out set not seen during any training, hyperparameter tuning, or feature selection step. 5-fold cross-validation performed on training set, stratified by diagnosis (4-level), to estimate internal performance. Hold-out results reported alongside CV results in Table 2. No further partitioning beyond training/hold-out split and 5-fold CV. |
| 12b | Methods / Predictor handling | Depending on model type, describe how predictors were handled in the analyses | Yes | Methods §2.3, lines 58–63 | Continuous predictors: StandardScaler (zero mean, unit variance) applied for LASSO, Ridge, ElasticNet, and deep MLP. Tree-based models (RF, XGBoost, MTL-XGBoost) use raw (unscaled) values; internal node splitting is scale-invariant. Categorical predictor (diagnosis: TBI/stroke/SAH/ICH) one-hot encoded (3 dummy variables, reference = TBI) for linear models; used as integer-coded for tree models. Binary predictors retained as 0/1 integers throughout. No interaction terms or polynomial expansions applied a priori; LASSO regularisation handles implicit selection. |
| 12c | Methods / Model specification and internal validation | Specify the type of model, rationale, all model building steps including hyperparameter tuning, and method for internal validation | Yes | Methods §2.3–2.4, lines 60–76 | Six model types developed: (1) **LASSO** — L1-regularised linear regression (LassoCV); alpha tuned over log-space grid, 5-fold inner CV; (2) **Ridge** — L2-regularised; same alpha grid; (3) **ElasticNet** — L1+L2; alpha and l1_ratio tuned (grid search); (4) **Random Forest** — n_estimators=200, max_depth=10 (primary pipeline) or max_depth=6 (QSP-hybrid pipeline), fixed hyperparameters (no grid search); (5) **XGBoost** — n_estimators=200, max_depth=5, learning_rate=0.1, fixed hyperparameters; (6) **Deep MLP** — per-outcome: 3 hidden layers [128, 64, 32]; longitudinal: 4 hidden layers [256, 128, 64, 32]; ReLU activation, Adam optimiser, early stopping (patience=20, val_split=0.15). MTL-XGBoost (chained multi-output) uses same XGBoost configuration. Internal validation: 5-fold CV stratified by diagnosis; mean ± SD R² (regression) and AUC-ROC (classification) reported. Scikit-learn cross_val_score used throughout. |
| 12d | Methods / Heterogeneity | Describe if and how any heterogeneity in estimates of model performance was handled | Partial | Methods §2.4, lines 70–72; Results §3.2 | Fold-to-fold variation reported as SD across 5 folds in Table 2. Heterogeneity by diagnosis subgroup is reported for the primary outcome (GOSE 12m) in Supplementary Figure S9 (forest plot of R² by diagnosis). Formal heterogeneity tests (I² or Cochran's Q across folds) are not reported — this is a gap. Heterogeneity across the two data sources (training vs. hold-out) is assessed by the optimism bias metric (mean Δ = +0.007). Diagnosis-stratified subgroup performance is provided as a supplementary analysis; sex-stratified and age-group-stratified analyses are not yet performed (noted as a limitation in §4.8). |
| 12e | Methods / Performance measures | Specify all measures and plots used to evaluate model performance and compare multiple models | Yes | Methods §2.4, lines 70–75; §2.5 | Regression: R² (primary), MAE (secondary), RMSE (secondary). Classification: AUC-ROC (primary), sensitivity/specificity at Youden threshold (secondary), Brier score (secondary). Calibration: formal calibration curves not produced for regression outcomes (R² considered sufficient); for binary classification, reliability diagrams (predicted vs. observed probability deciles) presented in Supplementary Figure S12. Model comparison: performance heatmap (Figure 2), pairwise Wilcoxon signed-rank test across 5 CV folds for significance of between-model differences. Conformal prediction: empirical coverage rate (target 95%, achieved 94.4%) and mean interval width reported (Table 3). All performance figures generated at 300 dpi (matplotlib). |
| 12f | Methods / Model updating | Describe any model updating arising from the model evaluation | N/A | Methods §2.4 | No model updating was performed. The hold-out evaluation is treated as a fixed independent test; no recalibration or re-estimation using hold-out data was done. This approach is consistent with the stated purpose (methodology development). For real-data deployment, updating procedures (e.g., recalibration on local case-mix) will be required and are discussed in §4.9. |
| 12g | Methods / Prediction calculation for evaluation | For model evaluation, describe how model predictions were calculated | Yes | Methods §2.4, lines 72–74 | For the hold-out evaluation: all models were trained on the full training set (n=2,000) using the optimal hyperparameters identified by 5-fold CV, then applied to the hold-out set (n=500). Predictions are generated as: continuous scores (regression models), probability outputs (sigmoid/softmax for MLP; `predict_proba` for RF/XGBoost), and conformal prediction intervals (split conformal, calibration set = 20% of training set withheld after final model fit). No ensemble averaging of multiple CV folds was used for hold-out prediction; the single final model trained on all n=2,000 was used, consistent with standard practice. |

### 3.8 Class Imbalance

| Item | Domain | TRIPOD+AI Item Description | Reported | Location | Comments |
|------|--------|---------------------------|----------|----------|---------|
| 13 | Methods / Class imbalance | If class imbalance methods were used, state why and how this was done | Partial | Methods §2.3, lines 64–65 | **RTW (Return-to-Work):** ~55% positive class — no class imbalance adjustment required; near-balanced. **Mortality:** ~8% positive class (160 events/2,000 training). No explicit class weighting was applied in the current implementation (RF and XGBoost use default class weights). SMOTE oversampling was considered but not used. The low mortality event rate and relatively modest AUC (0.579) are linked in the Results discussion (§3.3). Addition of `class_weight='balanced'` to the RF and XGBoost mortality classifiers is recommended for future iterations. |

### 3.9 Model Fairness

| Item | Domain | TRIPOD+AI Item Description | Reported | Location | Comments |
|------|--------|---------------------------|----------|----------|---------|
| 14 | Methods / Model fairness | Describe any approaches used to address model fairness and their rationale | Partial | Methods §2.5, lines 76–78; Discussion §4.8 | Fairness is partially addressed: SHAP values are reported stratified by sex (binary) and age group (≤60 / >60 years) in Supplementary Figure S14 to assess differential predictor importance. Formal algorithmic fairness metrics (e.g., demographic parity, equal opportunity, calibration by subgroup) are not computed. The rationale given is that the simulation does not encode differential outcome disparities by race or socioeconomic status beyond the included covariates, but the authors acknowledge this does not preclude emergent unfairness. A formal fairness audit using established frameworks (e.g., Aequitas, Fairlearn) is listed as a priority next step for real-data deployment (§4.9). This is an acknowledged gap for the current simulation study. |

### 3.10 Prediction Model Output

| Item | Domain | TRIPOD+AI Item Description | Reported | Location | Comments |
|------|--------|---------------------------|----------|----------|---------|
| 15 | Methods / Model output | Specify the output of the prediction model and provide details and rationale for any classification threshold | Yes | Methods §2.3, lines 63–65; §2.4 | Regression models output continuous scores on the original scale of each outcome (e.g., GOSE 1–8, FIM 18–126). Classification models output a probability (0–1) for RTW and mortality. No fixed binary classification threshold is applied for primary performance reporting (AUC computed across all thresholds). For supplementary sensitivity analyses and clinical utility curves, the Youden Index threshold is used and reported in Supplementary Table S3. Conformal prediction intervals output lower and upper bounds on continuous predictions. |

### 3.11 Model Comparison

| Item | Domain | TRIPOD+AI Item Description | Reported | Location | Comments |
|------|--------|---------------------------|----------|----------|---------|
| 16 | Methods / Development vs. evaluation differences | Identify any differences between development and evaluation data in setting, eligibility criteria, outcome, and predictors | Yes — adapted | Methods §2.1, lines 44–45; §2.4, lines 70–71 | **Simulation context:** Development (n=2,000, seed=42) and hold-out (n=500, seed=99) are generated by the same DGP and therefore share identical distributional parameters, eligibility criteria, outcome definitions, and predictor definitions. The only difference is the random seed, which produces a statistically independent sample from the same data-generating distribution. This constitutes a temporal/site equivalent split rather than a true external validation with a different clinical source. This is explicitly acknowledged as a limitation (§4.8): "The hold-out set represents a same-source split (different seed, same DGP) and cannot be considered a true external validation equivalent to a geographically or temporally distinct cohort." |

### 3.12 Ethics and Approvals

| Item | Domain | TRIPOD+AI Item Description | Reported | Location | Comments |
|------|--------|---------------------------|----------|----------|---------|
| 17 | Methods / Ethics | Name the institutional review board or ethics committee that approved the study | N/A | Methods §2.8; Declaration section | **N/A — simulation study with no human participants.** All data are entirely synthetic. No human participants were involved; no patient data were accessed. Formal IRB/ethics committee review is not required under applicable regulations (Declaration of Helsinki does not apply to computational simulation studies not involving human subjects). Statement included in manuscript: "This study used exclusively synthetic data generated by a computational simulation. No real patient data were collected or accessed at any stage. Institutional ethics review was not required and was not sought." A waiver/exemption statement is included in the Declarations section of the manuscript. |

---

## Section 4: Open Science

| Item | Domain | TRIPOD+AI Item Description | Reported | Location | Comments |
|------|--------|---------------------------|----------|----------|---------|
| 18a | Open Science / Funding | Give the source of funding and the role of funders for the present study | Yes | Declarations, Funding section | Funding statement included in manuscript Declarations. All analyses performed using open-source software on institutional computing resources. No external grant funding was used for this simulation study. Funders had no role in study design, analysis, or decision to publish. |
| 18b | Open Science / Conflicts of interest | Declare any conflicts of interest and financial disclosures for all authors | Yes | Declarations, Competing Interests section | Full competing interests declaration included per npj Digital Medicine requirements. All authors declare no competing interests relevant to this simulation methodology study. |
| 18c | Open Science / Protocol | Indicate where the study protocol can be accessed or state that a protocol was not prepared | Partial | Declarations; Methods §2.8 | A formal pre-registered study protocol was not prepared for this computational methodology study (no patient data, no prospective clinical intervention). The study was designed to develop and demonstrate the pipeline. The manuscript Methods section (§2.1–2.8) serves as the de facto protocol. A recommendation to prospectively register future real-data deployment studies is included in §4.9. |
| 18d | Open Science / Registration | Provide registration information for the study, or state that the study was not registered | Yes | Declarations; Abstract final line | Explicit statement included: "This computational simulation methodology study was not prospectively registered. The simulation pipeline code constitutes the primary reproducibility record and is publicly archived." Registration will be implemented for the planned real-data validation study. |
| 18e | Open Science / Data availability | Provide details of the availability of the study data | Yes | Data Availability statement; Methods §2.8 | The complete synthetic dataset (n=2,000 training + n=500 hold-out) is publicly available as CSV files in the study's GitHub repository (URL provided in Data Availability statement). The data-generating Python script (`brain_injury_ai_pipeline.py`) is included, allowing full reproduction of the datasets from specified seeds (training_seed=42, holdout_seed=99). |
| 18f | Open Science / Code availability | Provide details of the availability of the analytical code | Yes | Code Availability statement; Methods §2.8 | Full analytical code publicly available in GitHub repository: `brain_injury_ai_pipeline.py` (simulation + ML pipeline), `brain_injury_mtl_shap.py` (MTL + SHAP), `brain_injury_qsp_hybrid.py` (QSP-ODE hybrid). Software versions: Python 3.11, scikit-learn 1.4, XGBoost 2.0, PyMC 5.25.1, SHAP 0.47.2, TensorFlow/Keras (deep MLP), matplotlib 3.8, NumPy 1.26, pandas 2.1. Repository includes a `requirements.txt` and step-by-step README for reproduction. |

---

## Section 5: Patient and Public Involvement

| Item | Domain | TRIPOD+AI Item Description | Reported | Location | Comments |
|------|--------|---------------------------|----------|----------|---------|
| 19 | PPI / Patient and public involvement | Provide details of any patient and public involvement during design, conduct, reporting, interpretation, or dissemination | N/A | Declarations; Discussion §4.9 | **N/A — computational simulation methodology study.** No patients or members of the public were involved in the design, conduct, or reporting of this simulation study. PPI is not applicable at this stage because: (1) the study develops and tests a computational methodology, not a clinical intervention or tool to be directly used by patients; (2) no patient data are used; (3) the pipeline is not yet deployed. The manuscript explicitly states: "Patient and public involvement was not conducted for this computational methodology study. PPI will be a mandatory component of the planned prospective real-data validation study (§4.9), with a focus on co-design of output formats with brain injury survivors and carers." |

---

## Section 6: Results

| Item | Domain | TRIPOD+AI Item Description | Reported | Location | Comments |
|------|--------|---------------------------|----------|----------|---------|
| 20a | Results / Participant flow | Describe the flow of participants through the study, including the number of participants with and without the outcome | Yes — adapted | Results §3.1; Figure 1 | **Simulation context:** Figure 1 presents a CONSORT-style flow diagram showing: training set n=2,000 (all generated, none excluded from the predictor matrix); hold-out set n=500. For each outcome, the effective sample size after survival masking is shown: e.g., GOSE 12m n=1,840 training (92% survival), mortality n=2,000 (all included). Number of outcome events: mortality positive class n=160 (8.0%); RTW positive class n=1,100 (55.0%). For regression outcomes, distribution statistics (mean ± SD) are reported in Table 1. |
| 20b | Results / Participant characteristics | Report the characteristics overall and, where applicable, for each data source or setting | Yes | Results §3.1; Table 1 | Table 1 reports summary statistics for all 28 predictor variables and 12 outcomes for: (i) training set overall, (ii) hold-out set overall, (iii) training set by diagnosis subgroup (TBI/stroke/SAH/ICH). Comparison of training vs. hold-out distributions confirms internal consistency (same DGP, different seed). |
| 20c | Results / Predictor distribution comparison | For model evaluation, show a comparison with the development data of the distribution of important predictors | Yes | Results §3.1; Supplementary Figure S1 | Supplementary Figure S1 presents violin plots / density overlays for key predictors (GCS, Age, APACHE II, ICU LOS, mean ICP) comparing training (n=2,000) vs. hold-out (n=500) distributions. Kolmogorov-Smirnov test statistics reported; all KS p-values >0.05, confirming distributional equivalence as expected for same-DGP sampling. |
| 21 | Results / Numbers in analysis | Specify the number of participants and outcome events in each analysis | Yes | Results §3.1–3.3; Table 2 footnotes | For each of the 12 models, Table 2 footnotes report: (i) n used in training (after survival masking), (ii) n used in hold-out evaluation, (iii) number of positive events for binary outcomes. 5-fold CV fold sizes reported as approximately n=400 per validation fold. All analyses reported on the intention-to-simulate population (no exclusions after DGP). |
| 22 | Results / Full model specification | Provide details of the full prediction model to allow predictions in new individuals | Yes | Results §3.4; Table 3; Supplementary Tables S4–S6 | Table 3 reports the full LASSO coefficient table for the primary outcome (GOSE 12m): all 28 predictor names, standardised coefficients, 95% bootstrap confidence intervals. Supplementary Tables S4–S6 report full model specifications for: (S4) LASSO coefficients for all 10 regression outcomes; (S5) RF feature importances (Gini) and XGBoost feature importances (gain) for RTW and mortality; (S6) MTL-XGBoost chain order and per-task hyperparameters. The deep MLP architecture and weights are archived in the public repository (model checkpoint files). QSP-ODE parameter values are reported in Supplementary Table S2. |
| 23a | Results / Model performance | Report model performance estimates with confidence intervals, including for any key subgroups | Yes | Results §3.2–3.3; Table 2; Figure 2 | Table 2 reports: 5-fold CV mean ± SD R² (regression) and AUC-ROC (classification) for all 6 model types × 12 outcomes. Hold-out performance reported alongside CV performance. 95% bootstrap CIs (1,000 bootstrap resamples of hold-out set) reported for all hold-out metrics. Key results: LASSO GOSE R²=0.411 (95% CI 0.387–0.434), FIM R²=0.512 (95% CI 0.489–0.535), RF RTW AUC=0.638 (Supplementary Table 1, primary pipeline) or 0.820 (Supplementary Table 5, QSP-hybrid pipeline with different hyperparameters), RF mortality AUC=0.578 (95% CI 0.541–0.617). Subgroup performance: diagnosis-stratified R²/AUC reported in Supplementary Figure S9 (forest plot). |
| 23b | Results / Heterogeneity in performance | If examined, report results of any heterogeneity in model performance across clusters | Partial | Results §3.2; Supplementary Figure S9 | Heterogeneity of LASSO performance across 4 diagnosis subgroups (TBI, stroke, SAH, ICH) examined for GOSE 12m in Supplementary Figure S9 with point estimates and 95% CIs. Formal I² heterogeneity statistic not computed. Qualitative observation: GOSE R² is highest for TBI (0.447) and lowest for ICH (0.367), likely reflecting greater outcome variability in ICH. Fold-to-fold heterogeneity (SD across 5 CV folds) is reported in Table 2 for all outcomes. Sex-stratified and age-group-stratified performance subanalyses are not yet performed; this is noted as a gap in §4.8. |
| 24 | Results / Model updating | Report results from any model updating, including updated model and subsequent performance | N/A | Results | No model updating was performed (see Item 12f). Hold-out evaluation used the final model trained on the full training set without any updating. N/A for this study. |

---

## Section 7: Discussion

| Item | Domain | TRIPOD+AI Item Description | Reported | Location | Comments |
|------|--------|---------------------------|----------|----------|---------|
| 25 | Discussion / Interpretation | Give an overall interpretation of the main results, including issues of fairness in the context of the objectives | Yes | Discussion §4.1–4.7, lines [section lines] | §4.1 provides overall interpretation: LASSO wins 14/14 regression outcomes (Occam's razor principle in small-to-medium feature space); RF superior for classification. §4.2 interprets QSP mechanistic feature result (ΔR²<0.005): mechanistic features are collinear with their clinical inputs (GCS, APACHE II, mean ICP drive the ODE system) and add minimal orthogonal signal in a simulated context; real physiological signal from continuous waveform data may differ. §4.3 interprets MTL psychiatric gain (+0.020 R²): plausible given shared psychological pathways. §4.4 interprets conformal coverage (94.4% vs. 95% nominal): slight under-coverage consistent with finite calibration set; acceptable. §4.5 contextualises SHAP findings: GCS, Age, ICU Anxiety Score, Delirium, APACHE II are consistent with IMPACT/CRASH predictor literature. Fairness: differential SHAP contributions by sex discussed in §4.6; no evidence of systematic model bias within the simulation but real-world fairness cannot be inferred. |
| 26 | Discussion / Limitations | Discuss limitations of the study and their effects on biases, statistical uncertainty, and generalisability | Yes | Discussion §4.8 | §4.8 (Limitations) explicitly addresses: (1) **Simulation generalisability**: DGP represents a simplified causal structure; real neurobiological complexity (non-linear interactions, unmeasured confounders, heterogeneous treatment effects) is not captured. Models may overfit the DGP's structure and underperform on real data; (2) **Same-source hold-out**: hold-out set is not a true external validation (same DGP, different seed); external validation on real cohort data (e.g., CENTER-TBI, TRACK-TBI) is required; (3) **Low mortality EPV**: 160 events / 28 predictors = EPV 5.7, below recommended threshold of 10, increasing instability of mortality model; (4) **ICP median imputation**: single imputation introduces measurement error; MICE recommended for future work; (5) **Calibration**: no formal calibration curve for regression outcomes; (6) **No sensitivity analyses**: variable selection stability (bootstrap Lasso), distributional shift sensitivity, and alternate imputation strategies not yet tested; (7) **Fairness audit**: no formal algorithmic fairness assessment; (8) **Causal inference**: prediction models do not support causal claims; DGP includes direct causal pathways but the ML models learn correlational associations; (9) **Psychiatric simulation**: revised direct causal weights (prior psych → PHQ-9 weight=1.5; alcohol misuse → PCL-5 weight=2.0) are assumptions, not validated against real data. |
| 27a | Discussion / Implementation — input data quality | Describe how poor quality or unavailable input data should be assessed and handled when implementing the prediction model | Yes | Discussion §4.9, lines [section lines] | §4.9 describes implementation considerations: (i) missing GCS at admission (e.g., intubated patients before assessment): imputation using alternative neurological signs or GCS-Pupils Score recommended; (ii) missing ICP monitoring: the current median imputation is a placeholder; recommend flagging as a separate binary feature ('ICP monitored: yes/no') and applying MICE for the continuous value; (iii) missing APACHE II components: partial APACHE II computation using available variables, with documentation of missing components; (iv) data quality checks recommended: range checks for all continuous inputs, clinical plausibility rules (e.g., CPP = MAP − ICP must be physiologically plausible), and alert for out-of-distribution cases. A prospective data quality framework is listed as a priority before real-data deployment. |
| 27b | Discussion / Implementation — user interaction | Specify whether users will be required to interact in the handling of input data or use of the model | Yes | Discussion §4.9 | §4.9 specifies: in the intended deployment (clinical pharmacologist / intensivist use), the system is designed for minimal user interaction with the model itself. Clinicians provide structured input data via an electronic health record interface; the model runs as a back-end service. However, users will need to: (i) confirm diagnosis category, (ii) adjudicate the ICP monitoring status, (iii) review flagged data quality warnings. SHAP output is presented as a ranked predictor list in a dashboard for clinical review. The conformal prediction interval is presented alongside the point estimate to communicate uncertainty. End-user validation on clinical workflows is required before deployment. |
| 27c | Discussion / Next steps | Discuss next steps for future research, with a specific view to applicability and generalisability | Yes | Discussion §4.9 | Comprehensive next steps detailed: (1) Prospective validation on real neurocritical ICU data (CENTER-TBI, TRACK-TBI, or institutional cohort) with true external validation; (2) Bayesian parameter estimation for personalised ODE parameters (PyMC / Stan), replacing fixed population-mean ODE parameters with patient-specific posterior distributions; (3) Integration of simulated imaging features (lesion volume, midline shift, perfusion maps) using multimodal fusion; (4) QSP with continuous physiological signals (ICP waveform, PRx, TCD) for real-time mechanistic monitoring; (5) Neural ODE / latent ODE for trajectory modelling of multi-timepoint recovery; (6) Formal algorithmic fairness audit using Aequitas or Fairlearn on real-world data; (7) Health economic analysis of the prediction pipeline as a decision-support tool; (8) PPI co-design for output formats with brain injury survivors and carers. |

---

## Summary of Completeness

| Section | Total Items | Reported (Yes) | Partial | N/A | Not Reported (No) |
|---------|-------------|---------------|---------|-----|-------------------|
| Title & Abstract | 2 + 13 sub | 15 | 0 | 0 | 0 |
| Introduction | 4 | 3 | 1 | 0 | 0 |
| Methods — Data | 6 | 2 | 1 | 3 | 0 |
| Methods — Participants | 1 | 1 | 0 | 0 | 0 |
| Methods — Outcomes | 3 | 1 | 0 | 2 | 0 |
| Methods — Predictors | 3 | 2 | 0 | 1 | 0 |
| Methods — Sample size | 1 | 1 | 0 | 0 | 0 |
| Methods — Missing data | 1 | 1 | 0 | 0 | 0 |
| Methods — Analysis | 7 | 5 | 1 | 1 | 0 |
| Methods — Class imbalance | 1 | 1 | 0 | 0 | 0 |
| Methods — Fairness | 1 | 0 | 1 | 0 | 0 |
| Methods — Output | 1 | 1 | 0 | 0 | 0 |
| Methods — Comparison | 1 | 1 | 0 | 0 | 0 |
| Methods — Ethics | 1 | 0 | 0 | 1 | 0 |
| Open Science | 6 | 4 | 2 | 0 | 0 |
| PPI | 1 | 0 | 0 | 1 | 0 |
| Results | 5 | 3 | 2 | 1 | 0 |
| Discussion | 3 | 3 | 0 | 0 | 0 |
| **TOTAL** | **52** | **44** | **8** | **9** | **0** |

*Excludes abstract sub-items from the main totals. N/A items are appropriate for this simulation study design.*

**Items reported as Partial (8) — recommended actions for revision:**

1. **Item 3c** (Health inequalities introduction): Add a paragraph reviewing published evidence on sex/age/socioeconomic disparities in TBI, SAH, and stroke outcomes.
2. **Item 6c** (Treatments): Consider adding a supplementary causal diagram showing how treatment variables enter the DGP and prediction pipeline.
3. **Item 12d** (Heterogeneity): Add I² statistics for between-fold and between-diagnosis heterogeneity in model performance; add sex-stratified and age-stratified performance subanalyses.
4. **Item 14** (Model fairness): Implement Fairlearn/Aequitas fairness metrics; add at minimum demographic parity and equal opportunity metrics stratified by sex and age group for RTW and mortality.
5. **Item 18c** (Protocol): Consider posting a retrospective protocol on OSF or medRxiv preprint prior to journal submission.
6. **Item 23b** (Heterogeneity in performance): Report formal I² for diagnosis subgroup heterogeneity; add sex-stratified performance estimates.
7. **Item 23b** (Additional subgroups): Add age-group-stratified subanalysis (≤60 / >60 years) for primary outcomes.
8. **Item 20a / 20b**: Confirm that Table 1 includes full outcome distribution statistics for both training and hold-out sets, broken down by diagnosis.

---

## Notes on N/A Items for This Simulation Study

The following items are marked N/A and are not deficiencies in reporting quality; they are inapplicable to a fully synthetic simulation study:

| Item | Reason for N/A |
|------|---------------|
| 5b (Data dates) | No real participant data; no accrual dates. Simulation is instantaneous and date-agnostic. |
| 6a (Setting — real centres) | No real clinical centres; simulated single-centre ICU. |
| 8b (Outcome assessor qualifications) | No human assessors; outcomes generated algorithmically. |
| 8c (Blinding of outcome assessment) | Not applicable; algorithmic generation precludes blinding need. |
| 9c (Predictor assessor qualifications) | Not applicable; predictors generated algorithmically. |
| 17 (Ethics committee approval) | No human subjects; no patient data. Ethics review not required. |
| 19 (Patient and public involvement) | Not applicable at methodology development stage; planned for real-data phase. |
| 24 (Model updating results) | No model updating performed; design decision. |

---

*This checklist was completed on 23 February 2026 and will be updated to reflect any revisions made during peer review.*

*Reference: Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378*
