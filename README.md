# Brain Injury QSP — Mechanistic AI Pipeline for Neurocritical Outcomes

A reproducible Quantitative Systems Pharmacology (QSP) + machine-learning pipeline for predicting 12-month functional, cognitive, and psychiatric outcomes after acute brain injury. The pipeline combines a simulated neurocritical-care cohort, ODE-derived mechanistic features (intracranial pressure, cerebral autoregulation, neuroinflammation, neuroprotection), and a portfolio of classical ML, deep learning, multi-task, Bayesian, and explainable models (SHAP).

> **Note.** All cohort data in this repository are **fully simulated**. No real patient data are included. The simulation generators are provided so the entire analysis can be reproduced end-to-end from scripts.

---

## Cohort

| Item | Value |
|---|---|
| Sample size (training) | n = 2,000 |
| Hold-out cohort | n = 500 (independent seed) |
| Diagnoses | TBI 40%, SAH 20%, Stroke 25%, ICH 15% |
| Variables | 97 baseline + ICU + 12-month outcomes |
| Follow-up | 3 / 6 / 12 months |
| Mechanistic features | 21 QSP-ODE features (11 ICP/CPP, 6 neuroinflammation, 4 neuroprotection) |

Outcome instruments: **GOSE**, **mRS**, **FIM**, **Barthel**, **DRS**, **MoCA**, **HADS-A/D**, **PHQ-9**, **GAD-7**, **PCL-5**, **QOLIBRI-OS**, **SF-36 MCS/PCS**, **MPAI-4**, return-to-work, mortality.

---

## Repository structure

```
brain_injury_qsp/
├── scripts/      # 19 Python pipeline scripts (simulation, ML, QSP, SHAP, Bayesian, sensitivity)
├── data/         # Simulated cohorts and feature matrices (CSV + 1 NPY)
├── results/      # Model performance, SHAP, sensitivity, hold-out comparison tables
└── figures/      # 4 main + 33 supplementary + 6 report figures + causal DAG
```

### `scripts/` (12 files)

**Core analysis pipeline (run in this order):**

| Script | Purpose |
|---|---|
| `brain_injury_ai_pipeline.py` | Core simulation + classical ML + deep learning + trajectory modeling |
| `brain_injury_qsp_hybrid.py` | QSP ODE hybrid pipeline (ICP/CPP + neuroinflammation ODEs) |
| `brain_injury_mtl_shap.py` | Multi-task learning + SHAP explainability |
| `brain_injury_longitudinal.py` | Multi-horizon prediction + trajectory classifier + biomarkers |
| `brain_injury_bayes_drug.py` | Bayesian ODE inference + osmotherapy PK/PD |
| `brain_injury_performance.py` | Enhanced feature sets + latent factors + conformal prediction intervals |
| `brain_injury_holdout_eval.py` | Hold-out (n=500) external validation |

**TRIPOD-AI compliance (subgroup + sensitivity):**

| Script | Purpose |
|---|---|
| `sensitivity_analysis.py` | Sample size, feature reduction, missing data sensitivity (SA1–SA3) |
| `tbi_severity_comparison.py` | TBI mild/moderate/severe subgroup analysis with FDR correction |

**Auxiliary:**

| Script | Purpose |
|---|---|
| `generate_causal_dag.py` | Render causal DAG figure (SuppFig28) |
| `validate_cohort.py` | Sanity-check simulated cohort against published literature |
| `check_cohort.py` | Quick inspection of trajectory class distribution |

### `data/`
- `simulated_neurocritical_cohort_n2000.csv` — primary cohort (n=2000, seed 42)
- `holdout_cohort_n500.csv` — independent hold-out (n=500, seed 99)
- `enhanced_cohort_features_n2000.csv` — cohort + simulated latent factors (cognitive reserve, APOE4, IL-6, sleep, physical activity)
- `mechanistic_features_n2000.csv` — 21 QSP-ODE features
- `bayes_mechanistic_features_n2000.csv`, `bayes_ode_parameters_n2000.csv` — Bayesian posterior summaries
- `drug_response_features_n2000.csv` — osmotherapy PK/PD features (mannitol vs HSS)
- `inflammatory_biomarkers_n2000.csv` — CRP, IL-6, NI-index trajectories
- `trajectory_predictions_n2000.csv`, `trajectory_predictions_enhanced_n2000.csv` — longitudinal GOSE predictions
- `icp_summaries_n2000.npy` — ICP waveform summaries (binary)

### `results/`
- `model_performance_summary.csv` — best-model R²/AUC across all 14 outcomes
- `holdout_performance_comparison.csv` — CV vs hold-out optimism bias
- `qsp_hybrid_comparison.csv`, `qsp_lasso_coefficients.csv` — clinical vs hybrid feature contributions
- `mtl_comparison_results.csv` — single-task vs multi-task XGBoost
- `shap_importance_matrix.csv`, `gose_shap_interactions_top30.csv` — SHAP global + interaction values
- `multihorizon_prediction_results.csv` — 3 / 6 / 12 month prediction R²
- `extended_model_comparison.csv` — Clinical vs +DetQSP vs +BayesODE vs +Drug vs +All
- `sensitivity_sa1_sample_size.csv`, `sensitivity_sa2_feature_reduction.csv`, `sensitivity_sa3_missing_data.csv` — sensitivity analyses
- `tbi_severity_comparison_results.csv`, `tbi_severity_baseline_table.csv` — Mild / Moderate / Severe TBI subgroups
- `trajectory_classifier_results.csv` — 4-class trajectory classifier metrics
- `mech_feature_correlations.csv`, `performance_comparison.csv`

### `figures/`
- `Figure1` – `Figure4` — 4 main manuscript figures
- `SuppFig01` – `SuppFig33` + `SuppFig_CausalDAG` — 34 supplementary figures
- `report_fig*` — 6 report-only figures
- `causal_dag_mermaid.{mmd,svg,png,_hq.png}` — causal DAG source + renders

---

## Best-model performance (5-fold CV, seed = 42)

| Outcome | Best model | Metric | CV | Hold-out (n=500) |
|---|---|---|---:|---:|
| GOSE 12m | LASSO | R² | 0.410 | 0.400 |
| FIM Total 12m | LASSO | R² | 0.511 | 0.497 |
| Barthel 12m | LASSO | R² | 0.483 | 0.459 |
| HADS-Anxiety 12m | LASSO | R² | 0.438 | 0.457 |
| HADS-Depression 12m | LASSO | R² | 0.202 | 0.175 |
| PHQ-9 12m | LASSO | R² | 0.377 | 0.354 |
| PCL-5 12m | LASSO | R² | 0.378 | 0.372 |
| MoCA 12m | LASSO | R² | 0.153 | 0.109 |
| SF-36 MCS 12m | LASSO | R² | 0.167 | 0.158 |
| QOLIBRI-OS 12m | LASSO | R² | 0.173 | 0.184 |
| Return-to-Work | Random Forest | AUC | 0.821 | 0.826 |
| Mortality | Random Forest | AUC | 0.576 | 0.619 |
| GOSE longitudinal | Deep MLP | R² | 0.499 | — |

Mean optimism bias = +0.007 → CV estimates are reliable proxies for held-out performance.

**Top 5 universal predictors (mean |SHAP|):** GCS at admission · Age · ICU Anxiety Score · Delirium · APACHE II.

---

## Reproducing the analysis

### Requirements
- Python 3.11+
- Core: `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`
- ODE / Bayesian: `numba`, `pymc`, `arviz`
- Deep learning: `torch`
- Optional: `lifelines`, `statsmodels`

### Run order
```bash
# 1. Generate the cohort and run core ML/DL pipeline
python scripts/brain_injury_ai_pipeline.py

# 2. Build mechanistic ODE features
python scripts/brain_injury_qsp_hybrid.py

# 3. Multi-task learning + SHAP
python scripts/brain_injury_mtl_shap.py

# 4. Longitudinal multi-horizon + biomarker pipeline
python scripts/brain_injury_longitudinal.py

# 5. Bayesian ODE + osmotherapy drug PK/PD
python scripts/brain_injury_bayes_drug.py

# 6. Performance enhancement + conformal PI
python scripts/brain_injury_performance.py

# 7. External hold-out (n=500) validation
python scripts/brain_injury_holdout_eval.py

# 8. Sensitivity and TBI subgroup analyses
python scripts/sensitivity_analysis.py
python scripts/tbi_severity_comparison.py
```

All output figures and CSVs are written next to each script (mirroring this repo's `figures/` and `results/` layout).

---

## Methods notes

- **Mechanistic priors.** ODE features are generated from a coupled ICP / CPP / cerebral-autoregulation / neuroinflammation system. After the psychiatric outcome fix, mechanistic features add neutral-to-marginal signal over clinical features (top QSP predictor for GOSE: `mech_ar_index`, autoregulation index, β = −0.36).
- **Reporting.** The pipeline follows TRIPOD-AI guidance.
- **Uncertainty.** Conformal prediction intervals achieve 94.4% empirical coverage at the 95% nominal level; bootstrap PIs are unreliable (~33% coverage) and are not recommended.

---

## Citation

A manuscript describing this work is in preparation. Please cite the GitHub repository in the meantime:

> Anand. *Brain Injury QSP: a mechanistic AI pipeline for neurocritical outcome prediction.* GitHub repository: https://github.com/indand21/brain_injury_qsp

---

## License

No license has been specified yet. All rights reserved by the author until a license is added. Please contact the author for reuse permissions.

## Contact

Anand — `indand21@gmail.com`
