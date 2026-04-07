# Supplementary Tables

**Manuscript**: A Quantitative Systems Pharmacology–Machine Learning Framework for Multi-Domain Outcome Prediction in Neurocritical Care: A Computational Simulation Study

---

## Supplementary Table 1. Complete Model Performance (5-Fold Cross-Validation)

All models evaluated by 5-fold stratified cross-validation on the training cohort (n=2,000). Values are mean (SD). R²: coefficient of determination (regression outcomes). AUC-ROC: area under receiver operating characteristic curve (classification outcomes). GBM: gradient boosting machine; MLP: multilayer perceptron.

**Panel A — Regression outcomes (R², mean [SD])**

| Outcome | Domain | LASSO | Random Forest | XGBoost | GBM | MLP |
| --- | --- | --- | --- | --- | --- | --- |
| GOSE 12m | Functional | 0.411 (0.046) | 0.367 (0.057) | 0.310 (0.064) | 0.362 (0.068) | -0.212 (0.177) |
| mRS 12m | Functional | 0.331 (0.050) | 0.301 (0.049) | 0.248 (0.064) | 0.276 (0.059) | -0.129 (0.088) |
| FIM Total 12m | Functional | 0.512 (0.044) | 0.466 (0.049) | 0.406 (0.057) | 0.454 (0.056) | 0.199 (0.111) |
| Barthel 12m | Functional | 0.489 (0.048) | 0.442 (0.056) | 0.379 (0.054) | 0.425 (0.060) | 0.118 (0.088) |
| DRS 12m | Functional | 0.257 (0.040) | 0.210 (0.055) | 0.123 (0.072) | 0.183 (0.067) | -0.442 (0.154) |
| Cognitive Composite 12m | Cognitive | 0.336 (0.021) | 0.295 (0.026) | 0.217 (0.029) | 0.271 (0.043) | -0.085 (0.078) |
| MoCA 12m | Cognitive | 0.161 (0.023) | 0.129 (0.030) | 0.060 (0.048) | 0.109 (0.026) | -0.620 (0.090) |
| PCL-5 12m | Psychiatric | 0.373 (0.035) | 0.326 (0.038) | 0.240 (0.048) | 0.293 (0.035) | -0.173 (0.076) |
| gad7_12m | Psychiatric | 0.350 (0.034) | 0.296 (0.034) | 0.189 (0.025) | 0.262 (0.029) | -0.315 (0.063) |
| HADS-Depression 12m | Psychiatric | 0.200 (0.020) | 0.163 (0.033) | 0.062 (0.035) | 0.131 (0.048) | -0.521 (0.121) |
| HADS-Anxiety 12m | Psychiatric | 0.435 (0.019) | 0.384 (0.021) | 0.319 (0.029) | 0.379 (0.020) | -0.162 (0.085) |
| PHQ-9 12m | Psychiatric | 0.373 (0.033) | 0.350 (0.048) | 0.269 (0.076) | 0.308 (0.061) | -0.245 (0.205) |
| mpai4_tscore_12m | QoL | 0.060 (0.022) | 0.042 (0.025) | -0.066 (0.058) | -0.001 (0.044) | -0.758 (0.250) |
| SF-36 PCS 12m | QoL | 0.175 (0.060) | 0.124 (0.063) | 0.008 (0.092) | 0.080 (0.078) | -0.487 (0.146) |
| SF-36 MCS 12m | QoL | 0.161 (0.035) | 0.122 (0.031) | 0.050 (0.037) | 0.082 (0.036) | -0.519 (0.092) |
| QOLIBRI-OS 12m | QoL | 0.179 (0.019) | 0.135 (0.019) | 0.004 (0.028) | 0.075 (0.034) | -0.483 (0.130) |
| social_participation_12m | QoL | 0.123 (0.036) | 0.096 (0.048) | 0.026 (0.048) | 0.068 (0.065) | -0.480 (0.042) |


**Panel B — Classification outcomes (AUC-ROC, mean [SD])**

| Outcome | Domain | Logistic-L1 | Random Forest | XGBoost |
| --- | --- | --- | --- | --- |
| Return-to-Work | Classification | 0.602 (0.086) | 0.638 (0.016) | 0.606 (0.025) |
| 12m Mortality | Classification | 0.535 (0.032) | 0.578 (0.019) | 0.537 (0.018) |


---

## Supplementary Table 2. Full SHAP Feature Importance Matrix

Mean absolute SHAP values (mean |SHAP|) for all 27 features across all 12 outcome domains, computed using SHAP TreeExplainer on Random Forest (classification) and XGBoost surrogate (regression) models. Features ordered by mean |SHAP| averaged across all outcomes (descending). Higher values indicate greater contribution to prediction for that outcome.

| Feature | Mean across outcomes | GOSE (Functional) | Barthel Index (ADL) | Cognitive Composite | MoCA (Cognition) | HADS-Anxiety | HADS-Depression | PCL-5 (PTSD) | SF-36 Mental Component | QOLIBRI-OS (QoL) | Social Participation | Return to Work | 12-Month Mortality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GCS at Admission | 1.2929 | 0.5035 | 7.0572 | 0.1915 | 0.5928 | 0.1482 | 0.2159 | 0.3977 | 0.8333 | 2.5335 | 2.1108 | 0.5485 | 0.3821 |
| Age | 1.1366 | 0.2895 | 4.3738 | 0.1086 | 0.4573 | 0.3220 | 0.3226 | 0.8828 | 1.2435 | 1.8325 | 2.6460 | 0.8056 | 0.3546 |
| ICU Anxiety Score | 1.0429 | 0.1063 | 1.3195 | 0.0573 | 0.2858 | 1.3824 | 0.2582 | 2.2211 | 2.2852 | 2.1889 | 1.4053 | 0.7183 | 0.2867 |
| Delirium | 0.9020 | 0.1414 | 1.2647 | 0.1500 | 0.5082 | 0.8378 | 0.5174 | 1.9221 | 1.8633 | 2.3674 | 0.6854 | 0.5276 | 0.0393 |
| APACHE II | 0.8552 | 0.2432 | 2.9655 | 0.0669 | 0.2365 | 0.3859 | 0.2394 | 1.0550 | 1.3425 | 1.6983 | 1.3554 | 0.3983 | 0.2759 |
| Prior Psychiatric History | 0.7556 | 0.0115 | 0.2351 | 0.0123 | 0.0368 | 0.7699 | 0.6930 | 2.8158 | 2.0158 | 1.5516 | 0.7981 | 0.0704 | 0.0566 |
| ICU LOS (days) | 0.5912 | 0.1188 | 1.4459 | 0.0507 | 0.2448 | 0.2085 | 0.1954 | 0.5709 | 1.0879 | 1.3452 | 1.2100 | 0.3962 | 0.2204 |
| Education (years) | 0.5480 | 0.1647 | 1.1599 | 0.0810 | 0.2892 | 0.1632 | 0.2061 | 0.4792 | 0.9402 | 1.2738 | 1.2148 | 0.4044 | 0.1999 |
| Early Mobilization | 0.4932 | 0.1033 | 2.7112 | 0.0259 | 0.0815 | 0.1375 | 0.1749 | 0.2789 | 0.6877 | 0.8639 | 0.5862 | 0.1531 | 0.1138 |
| ICP (mmHg) | 0.4536 | 0.1115 | 1.0581 | 0.0377 | 0.1904 | 0.2022 | 0.1757 | 0.5539 | 0.8054 | 0.9169 | 0.9374 | 0.2421 | 0.2116 |
| Mech. Ventilation (days) | 0.4145 | 0.0641 | 1.5881 | 0.0381 | 0.1154 | 0.1575 | 0.1305 | 0.3737 | 0.5164 | 1.0337 | 0.6207 | 0.1789 | 0.1566 |
| ICDSC Score | 0.3538 | 0.0698 | 0.5786 | 0.0363 | 0.1758 | 0.1064 | 0.1354 | 0.3999 | 0.4999 | 0.9062 | 0.9678 | 0.2039 | 0.1656 |
| Marital Status | 0.3522 | 0.0523 | 0.4035 | 0.0213 | 0.0836 | 0.0783 | 0.2169 | 0.4640 | 0.3140 | 0.5149 | 1.7663 | 0.2051 | 0.1059 |
| Diagnosis | 0.3158 | 0.0711 | 0.9291 | 0.0313 | 0.1460 | 0.0693 | 0.0871 | 0.2191 | 0.4997 | 0.7330 | 0.6773 | 0.1925 | 0.1335 |
| Sex (Male) | 0.2080 | 0.0225 | 0.5264 | 0.0134 | 0.0709 | 0.0857 | 0.1075 | 0.1691 | 0.2327 | 0.7088 | 0.4446 | 0.0623 | 0.0519 |
| Smoking | 0.2005 | 0.0268 | 0.3142 | 0.0091 | 0.1180 | 0.0983 | 0.0435 | 0.2222 | 0.2786 | 0.4131 | 0.6272 | 0.2086 | 0.0459 |
| Pre-injury Employment | 0.1905 | 0.0293 | 0.2620 | 0.0226 | 0.1531 | 0.0677 | 0.0758 | 0.2513 | 0.3683 | 0.3522 | 0.6348 | 0.0153 | 0.0535 |
| Anticoagulation | 0.1754 | 0.0420 | 0.6272 | 0.0100 | 0.0695 | 0.0507 | 0.0478 | 0.0794 | 0.1288 | 0.3879 | 0.5692 | 0.0455 | 0.0470 |
| Hypertension | 0.1693 | 0.0186 | 0.6121 | 0.0100 | 0.0941 | 0.0417 | 0.0862 | 0.1081 | 0.2837 | 0.2533 | 0.3480 | 0.0754 | 0.1010 |
| Alcohol Misuse | 0.1661 | 0.0291 | 0.2070 | 0.0127 | 0.1114 | 0.0680 | 0.0492 | 0.5413 | 0.2397 | 0.2927 | 0.2761 | 0.0943 | 0.0723 |
| Cardiovascular Disease | 0.1638 | 0.0251 | 0.3403 | 0.0060 | 0.1278 | 0.0583 | 0.0587 | 0.2154 | 0.2949 | 0.4213 | 0.2362 | 0.1273 | 0.0546 |
| Diabetes | 0.1501 | 0.0336 | 0.2440 | 0.0050 | 0.0459 | 0.0382 | 0.0412 | 0.1219 | 0.1720 | 0.2846 | 0.5452 | 0.1936 | 0.0755 |
| Surgery | 0.1243 | 0.0217 | 0.1697 | 0.0120 | 0.0593 | 0.0604 | 0.0405 | 0.2492 | 0.2356 | 0.3030 | 0.2134 | 0.0758 | 0.0505 |
| Pneumonia | 0.1099 | 0.0314 | 0.1969 | 0.0094 | 0.0715 | 0.0686 | 0.0293 | 0.0952 | 0.2157 | 0.3572 | 0.1200 | 0.0831 | 0.0403 |
| UTI | 0.0919 | 0.0228 | 0.1451 | 0.0058 | 0.0303 | 0.0574 | 0.0534 | 0.0664 | 0.2797 | 0.2465 | 0.1152 | 0.0558 | 0.0250 |
| DVT | 0.0881 | 0.0102 | 0.2570 | 0.0076 | 0.0335 | 0.0388 | 0.0370 | 0.1164 | 0.1437 | 0.1893 | 0.1588 | 0.0388 | 0.0259 |
| Prior Brain Injury | 0.0802 | 0.0098 | 0.3233 | 0.0039 | 0.0280 | 0.0416 | 0.0321 | 0.0948 | 0.1057 | 0.1034 | 0.1514 | 0.0360 | 0.0324 |


---

## Supplementary Table 3. Multi-Task Learning vs Independent Models

5-fold cross-validated R² for chained multi-output XGBoost (MTL) versus independent single-outcome XGBoost models. ΔR² = MTL − Independent. Positive values indicate MTL improvement. Domain ordering: Functional → Cognitive → Psychiatric → Quality-of-Life/Participation.

| Domain | Outcome | Independent R² (mean) | Independent R² (SD) | MTL R² (mean) | MTL R² (SD) | ΔR² | Δ% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Functional | GOSE | 0.317 | 0.063 | 0.317 | 0.063 | 0.000 | 0.000 |
| Functional | mRS | 0.235 | 0.048 | 0.235 | 0.048 | 0.000 | 0.000 |
| Functional | FIM Total | 0.400 | 0.060 | 0.400 | 0.060 | 0.000 | 0.000 |
| Functional | Barthel | 0.372 | 0.051 | 0.372 | 0.051 | 0.000 | 0.000 |
| Functional | DRS | 0.103 | 0.068 | 0.103 | 0.068 | 0.000 | 0.000 |
| Cognitive | Cognitive Attention | 0.069 | 0.040 | 0.055 | 0.064 | -0.014 | -20.947 |
| Cognitive | Cognitive Executive | -0.052 | 0.038 | -0.039 | 0.061 | 0.014 | 26.371 |
| Cognitive | MoCA | 0.060 | 0.030 | 0.045 | 0.049 | -0.016 | -25.909 |
| Cognitive | Cognitive Composite | 0.213 | 0.021 | 0.224 | 0.042 | 0.011 | 5.175 |
| Cognitive | Cognitive Memory | 0.057 | 0.060 | 0.063 | 0.038 | 0.007 | 11.784 |
| Psychiatric | HADS-Anxiety | 0.327 | 0.017 | 0.341 | 0.022 | 0.014 | 4.236 |
| Psychiatric | HADS-Depression | 0.063 | 0.045 | 0.055 | 0.040 | -0.007 | -11.906 |
| Psychiatric | PHQ-9 | 0.271 | 0.060 | 0.285 | 0.042 | 0.014 | 5.340 |
| Psychiatric | GAD-7 | 0.188 | 0.042 | 0.222 | 0.074 | 0.034 | 18.064 |
| Psychiatric | PCL-5 | 0.230 | 0.038 | 0.273 | 0.033 | 0.043 | 18.521 |
| QoL/Participation | MPAI-4 T-score | -0.085 | 0.061 | -0.149 | 0.108 | -0.064 | -75.238 |
| QoL/Participation | SF-36 PCS | -0.007 | 0.092 | -0.024 | 0.124 | -0.017 | -231.750 |
| QoL/Participation | SF-36 MCS | 0.053 | 0.049 | -0.016 | 0.089 | -0.069 | -129.484 |
| QoL/Participation | QOLIBRI-OS | 0.018 | 0.033 | -0.023 | 0.057 | -0.041 | -229.048 |
| QoL/Participation | Social Participation | 0.003 | 0.056 | 0.020 | 0.076 | 0.017 | 493.464 |


---

## Supplementary Table 4. Independent Hold-Out Validation and Optimism Bias

Performance on independent hold-out cohort (n=500, seed=99) compared with 5-fold cross-validation on training cohort (n=2,000, seed=42). Optimism bias = CV mean − hold-out score (positive = overestimation by CV). 95% bootstrap CIs for hold-out scores computed from 1,000 resamples.

| Domain | Outcome | Best model | Metric | CV mean | CV SD | Hold-out score | Optimism bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Functional | GOSE 12m | LassoCV | R² | 0.410 | 0.019 | 0.400 | 0.010 |
| Functional | FIM Total 12m | LassoCV | R² | 0.511 | 0.035 | 0.497 | 0.014 |
| Functional | Barthel 12m | LassoCV | R² | 0.483 | 0.044 | 0.459 | 0.024 |
| Cognitive | MoCA 12m | LassoCV | R² | 0.153 | 0.024 | 0.109 | 0.044 |
| Psychiatric | HADS-Anxiety 12m | LassoCV | R² | 0.438 | 0.029 | 0.457 | -0.019 |
| Psychiatric | HADS-Depression 12m | LassoCV | R² | 0.202 | 0.009 | 0.175 | 0.027 |
| Psychiatric | PHQ-9 12m | LassoCV | R² | 0.377 | 0.019 | 0.354 | 0.023 |
| Psychiatric | PCL-5 12m | LassoCV | R² | 0.378 | 0.048 | 0.372 | 0.006 |
| Psychiatric | SF-36 MCS 12m | LassoCV | R² | 0.167 | 0.005 | 0.158 | 0.009 |
| QoL | QOLIBRI-OS 12m | LassoCV | R² | 0.173 | 0.011 | 0.184 | -0.010 |
| Classification | Return-to-Work | RandomForest | AUC | 0.821 | 0.010 | 0.826 | -0.005 |
| Classification | 12m Mortality | RandomForest | AUC | 0.576 | 0.024 | 0.619 | -0.042 |


---

## Supplementary Table 5. QSP-Hybrid vs Clinical-Only Model Performance

5-fold cross-validated performance for clinical-only (28 variables) versus QSP-Hybrid (28 clinical + 21 mechanistic ODE features) models for all outcomes. ΔR² = Hybrid − Clinical (positive = improvement with QSP features). LASSO used for regression; Logistic Regression / Random Forest for classification. **Note:** This table is generated by `brain_injury_qsp_hybrid.py` which uses RF hyperparameters (n_estimators=200, max_depth=6) that differ from the primary pipeline in Table 1 (max_depth=10), resulting in different clinical-only baseline values for classification outcomes (e.g., RTW RF AUC=0.820 here vs 0.638 in Table 1).

| Outcome | Model | Metric | Clinical (mean) | Clinical (SD) | Hybrid (mean) | Hybrid (SD) | ΔR²/ΔAUC | Δ% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12m Mortality | Logistic-L1 | AUC-ROC | 0.521 | 0.026 | 0.500 | 0.000 | -0.021 | -3.946 |
| 12m Mortality | Random Forest | AUC-ROC | 0.579 | 0.022 | 0.580 | 0.018 | 0.000 | 0.065 |
| 12m Mortality | XGBoost | AUC-ROC | 0.537 | 0.015 | 0.554 | 0.010 | 0.017 | 3.155 |
| Barthel 12m | LASSO | R² | 0.489 | 0.048 | 0.483 | 0.047 | -0.006 | -1.286 |
| Barthel 12m | Random Forest | R² | 0.448 | 0.052 | 0.454 | 0.048 | 0.006 | 1.333 |
| Barthel 12m | XGBoost | R² | 0.380 | 0.055 | 0.385 | 0.054 | 0.006 | 1.513 |
| Cognitive Composite 12m | LASSO | R² | 0.336 | 0.021 | 0.334 | 0.023 | -0.002 | -0.526 |
| Cognitive Composite 12m | Random Forest | R² | 0.305 | 0.024 | 0.308 | 0.028 | 0.004 | 1.269 |
| Cognitive Composite 12m | XGBoost | R² | 0.223 | 0.023 | 0.229 | 0.052 | 0.007 | 3.031 |
| DRS 12m | LASSO | R² | 0.256 | 0.041 | 0.254 | 0.042 | -0.002 | -0.778 |
| DRS 12m | Random Forest | R² | 0.224 | 0.053 | 0.221 | 0.047 | -0.003 | -1.337 |
| DRS 12m | XGBoost | R² | 0.120 | 0.069 | 0.100 | 0.062 | -0.020 | -16.836 |
| FIM Total 12m | LASSO | R² | 0.513 | 0.043 | 0.506 | 0.042 | -0.007 | -1.372 |
| FIM Total 12m | Random Forest | R² | 0.470 | 0.045 | 0.480 | 0.041 | 0.010 | 2.099 |
| FIM Total 12m | XGBoost | R² | 0.403 | 0.062 | 0.410 | 0.052 | 0.008 | 1.923 |
| GOSE 12m | LASSO | R² | 0.410 | 0.045 | 0.408 | 0.044 | -0.002 | -0.399 |
| GOSE 12m | Random Forest | R² | 0.377 | 0.053 | 0.381 | 0.048 | 0.003 | 0.887 |
| GOSE 12m | XGBoost | R² | 0.315 | 0.063 | 0.308 | 0.035 | -0.007 | -2.171 |
| HADS-Anxiety 12m | LASSO | R² | 0.435 | 0.019 | 0.436 | 0.020 | 0.002 | 0.370 |
| HADS-Anxiety 12m | Random Forest | R² | 0.404 | 0.020 | 0.397 | 0.021 | -0.007 | -1.755 |
| HADS-Anxiety 12m | XGBoost | R² | 0.324 | 0.028 | 0.326 | 0.022 | 0.003 | 0.886 |
| HADS-Depression 12m | LASSO | R² | 0.200 | 0.020 | 0.198 | 0.020 | -0.002 | -0.987 |
| HADS-Depression 12m | Random Forest | R² | 0.184 | 0.029 | 0.170 | 0.026 | -0.014 | -7.355 |
| HADS-Depression 12m | XGBoost | R² | 0.053 | 0.037 | 0.074 | 0.045 | 0.021 | 39.470 |
| MoCA 12m | LASSO | R² | 0.161 | 0.023 | 0.156 | 0.020 | -0.005 | -3.249 |
| MoCA 12m | Random Forest | R² | 0.144 | 0.024 | 0.144 | 0.022 | -0.001 | -0.507 |
| MoCA 12m | XGBoost | R² | 0.079 | 0.035 | 0.030 | 0.018 | -0.049 | -62.482 |
| PCL-5 12m | LASSO | R² | 0.373 | 0.035 | 0.371 | 0.032 | -0.001 | -0.289 |
| PCL-5 12m | Random Forest | R² | 0.339 | 0.041 | 0.333 | 0.044 | -0.006 | -1.753 |
| PCL-5 12m | XGBoost | R² | 0.245 | 0.044 | 0.217 | 0.055 | -0.027 | -11.125 |
| PHQ-9 12m | LASSO | R² | 0.373 | 0.033 | 0.375 | 0.033 | 0.002 | 0.628 |
| PHQ-9 12m | Random Forest | R² | 0.358 | 0.049 | 0.346 | 0.037 | -0.012 | -3.421 |
| PHQ-9 12m | XGBoost | R² | 0.269 | 0.075 | 0.258 | 0.045 | -0.011 | -4.078 |
| QOLIBRI-OS 12m | LASSO | R² | 0.179 | 0.019 | 0.178 | 0.017 | -0.000 | -0.249 |
| QOLIBRI-OS 12m | Random Forest | R² | 0.147 | 0.023 | 0.131 | 0.019 | -0.016 | -10.885 |
| QOLIBRI-OS 12m | XGBoost | R² | 0.013 | 0.032 | 0.022 | 0.027 | 0.009 | 68.745 |
| Return-to-Work | Logistic-L1 | AUC-ROC | 0.761 | 0.035 | 0.550 | 0.100 | -0.211 | -27.697 |
| Return-to-Work | Random Forest | AUC-ROC | 0.820 | 0.020 | 0.792 | 0.023 | -0.028 | -3.371 |
| Return-to-Work | XGBoost | AUC-ROC | 0.807 | 0.019 | 0.811 | 0.020 | 0.004 | 0.443 |
| SF-36 MCS 12m | LASSO | R² | 0.161 | 0.035 | 0.162 | 0.032 | 0.001 | 0.470 |
| SF-36 MCS 12m | Random Forest | R² | 0.137 | 0.032 | 0.135 | 0.034 | -0.002 | -1.557 |
| SF-36 MCS 12m | XGBoost | R² | 0.044 | 0.039 | 0.008 | 0.058 | -0.036 | -81.889 |
| SF-36 PCS 12m | LASSO | R² | 0.174 | 0.063 | 0.167 | 0.058 | -0.007 | -4.234 |
| SF-36 PCS 12m | Random Forest | R² | 0.143 | 0.065 | 0.145 | 0.070 | 0.002 | 1.273 |
| SF-36 PCS 12m | XGBoost | R² | 0.002 | 0.079 | 0.021 | 0.092 | 0.019 | 1011.885 |
| mRS 12m | LASSO | R² | 0.330 | 0.050 | 0.329 | 0.049 | -0.002 | -0.479 |
| mRS 12m | Random Forest | R² | 0.312 | 0.050 | 0.303 | 0.049 | -0.009 | -2.885 |
| mRS 12m | XGBoost | R² | 0.244 | 0.047 | 0.216 | 0.052 | -0.028 | -11.563 |


---

## Supplementary Table 6. LASSO Coefficients for GOSE 12m Prediction (QSP-Hybrid Model)

LASSO coefficients from the QSP-hybrid model (28 clinical + 21 mechanistic features) predicting 12-month GOSE. Features with coefficient=0 were penalised out. Mechanistic features (from QSP-ODE) are flagged. The sum of |coefficients| for mechanistic vs clinical features provides the signal decomposition reported in the text (33.3% mechanistic, 66.7% clinical).

*Signal decomposition: mechanistic |coef| sum = 0.558 (33.7%); clinical |coef| sum = 1.098 (66.3%).*

| Feature | QSP-ODE mechanistic | LASSO coefficient | |Coefficient| |
| --- | --- | --- | --- |
| mech_ar_index | Yes | -0.370 | 0.370 |
| Age | No | -0.252 | 0.252 |
| APACHE II | No | -0.191 | 0.191 |
| GCS at admission | No | 0.160 | 0.160 |
| Education years | No | 0.135 | 0.135 |
| Early mobilisation | No | 0.115 | 0.115 |
| mech_m1_peak | Yes | -0.103 | 0.103 |
| Delirium | No | -0.092 | 0.092 |
| ICU LOS (days) | No | -0.040 | 0.040 |
| mech_secondary_injury_index | Yes | -0.040 | 0.040 |
| mech_icp_mean_72h | Yes | 0.031 | 0.031 |
| Alcohol misuse | No | 0.027 | 0.027 |
| Pneumonia | No | -0.027 | 0.027 |
| Mean ICP (mmHg) | No | 0.015 | 0.015 |
| mech_ni_auc_7d | Yes | -0.015 | 0.015 |
| Anticoagulation use | No | 0.014 | 0.014 |
| Diabetes | No | 0.010 | 0.010 |
| Pre-injury employment | No | -0.009 | 0.009 |
| Diagnosis | No | -0.007 | 0.007 |
| DVT | No | 0.004 | 0.004 |
| Marital status | No | -0.000 | 0.000 |
| Hypertension | No | -0.000 | 0.000 |
| Cardiovascular disease | No | 0.000 | 0.000 |
| Prior psychiatric history | No | 0.000 | 0.000 |
| mech_icp_auc_7d | Yes | 0.000 | 0.000 |
| Prior brain injury | No | -0.000 | 0.000 |
| Smoking | No | 0.000 | 0.000 |
| Mechanical ventilation (days) | No | -0.000 | 0.000 |
| ICP monitoring | No | -0.000 | 0.000 |
| ICDSC score | No | 0.000 | 0.000 |
| ICU anxiety score | No | -0.000 | 0.000 |
| Sex (Male) | No | -0.000 | 0.000 |
| UTI | No | 0.000 | 0.000 |
| mech_icp_peak | Yes | 0.000 | 0.000 |
| mech_cpp_optimal_time | Yes | -0.000 | 0.000 |
| mech_icp_time_above_20 | Yes | 0.000 | 0.000 |
| Surgery | No | -0.000 | 0.000 |
| mech_icp_at_day7 | Yes | 0.000 | 0.000 |
| mech_np_steady_state | Yes | 0.000 | 0.000 |
| mech_ni_resolution_time | Yes | 0.000 | 0.000 |
| mech_m1_m2_ratio_72h | Yes | -0.000 | 0.000 |
| mech_icp_time_above_25 | Yes | 0.000 | 0.000 |
| mech_ni_peak | Yes | -0.000 | 0.000 |
| mech_np_auc | Yes | -0.000 | 0.000 |
| mech_m2_m1_dominance | Yes | 0.000 | 0.000 |
| mech_cpp_time_below_60 | Yes | -0.000 | 0.000 |
| mech_cpp_mean | Yes | 0.000 | 0.000 |
| mech_cpp_min | Yes | -0.000 | 0.000 |
| mech_recovery_potential | Yes | 0.000 | 0.000 |


---

## Supplementary Table 7. Mechanistic Feature–Outcome Pearson Correlation Matrix

Pearson correlation coefficients (r) between each QSP-ODE derived mechanistic feature (rows) and each 12-month clinical outcome (columns). Values reflect the biological plausibility of mechanistic features as outcome predictors. Bold values (|r| ≥ 0.10) indicate correlations of potential clinical relevance. All correlations were computed on the training cohort (n≈1,170 survivors for survivor-only outcomes; n=2,000 for mortality).

| Mechanistic feature | GOSE 12m | FIM Total 12m | Barthel 12m | mRS 12m | DRS 12m | Cognitive Composite 12m | MoCA 12m | HADS-Anxiety 12m | HADS-Depression 12m | PHQ-9 12m | PCL-5 12m | SF-36 PCS 12m | SF-36 MCS 12m | QOLIBRI-OS 12m | Return-to-Work | Mean |r| |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1 Peak | -0.605 | -0.676 | -0.653 | 0.550 | 0.476 | -0.523 | -0.356 | 0.284 | 0.274 | 0.250 | 0.265 | -0.353 | -0.218 | -0.367 | -0.239 | 0.406 |
| Recovery Potential | 0.551 | 0.620 | 0.604 | -0.490 | -0.442 | 0.473 | 0.326 | -0.238 | -0.224 | -0.207 | -0.204 | 0.399 | 0.172 | 0.314 | 0.304 | 0.371 |
| Ni Peak | -0.450 | -0.514 | -0.493 | 0.406 | 0.358 | -0.381 | -0.255 | 0.367 | 0.310 | 0.348 | 0.365 | -0.378 | -0.252 | -0.317 | -0.288 | 0.365 |
| Cpp Mean | 0.560 | 0.605 | 0.599 | -0.516 | -0.446 | 0.446 | 0.320 | -0.195 | -0.253 | -0.220 | -0.208 | 0.366 | 0.171 | 0.323 | 0.210 | 0.362 |
| Ni Auc 7D | -0.420 | -0.478 | -0.461 | 0.378 | 0.334 | -0.344 | -0.241 | 0.307 | 0.248 | 0.274 | 0.289 | -0.375 | -0.203 | -0.276 | -0.280 | 0.327 |
| Ar Index | -0.552 | -0.597 | -0.588 | 0.499 | 0.425 | -0.462 | -0.321 | 0.161 | 0.191 | 0.144 | 0.143 | -0.246 | -0.137 | -0.313 | -0.121 | 0.327 |
| Secondary Injury Index | -0.424 | -0.455 | -0.453 | 0.380 | 0.338 | -0.328 | -0.207 | 0.185 | 0.160 | 0.151 | 0.150 | -0.283 | -0.140 | -0.245 | -0.153 | 0.270 |
| M2 M1 Dominance | 0.308 | 0.363 | 0.348 | -0.273 | -0.257 | 0.256 | 0.192 | -0.199 | -0.151 | -0.170 | -0.164 | 0.352 | 0.123 | 0.180 | 0.305 | 0.243 |
| Np Steady State | 0.308 | 0.360 | 0.345 | -0.272 | -0.256 | 0.254 | 0.189 | -0.199 | -0.149 | -0.168 | -0.164 | 0.353 | 0.122 | 0.178 | 0.304 | 0.241 |
| M1 M2 Ratio 72H | -0.310 | -0.356 | -0.343 | 0.274 | 0.256 | -0.249 | -0.184 | 0.198 | 0.145 | 0.160 | 0.165 | -0.352 | -0.119 | -0.176 | -0.290 | 0.239 |
| Cpp Min | 0.326 | 0.365 | 0.374 | -0.310 | -0.250 | 0.289 | 0.213 | -0.092 | -0.162 | -0.105 | -0.132 | 0.203 | 0.068 | 0.198 | 0.138 | 0.215 |
| Cpp Time Below 60 | -0.335 | -0.349 | -0.353 | 0.299 | 0.267 | -0.252 | -0.151 | 0.105 | 0.096 | 0.077 | 0.071 | -0.192 | -0.089 | -0.182 | -0.078 | 0.193 |
| Np Auc | -0.197 | -0.201 | -0.200 | 0.182 | 0.144 | -0.190 | -0.108 | 0.041 | 0.082 | 0.044 | 0.060 | 0.056 | -0.061 | -0.131 | 0.094 | 0.119 |
| Cpp Optimal Time | 0.114 | 0.114 | 0.111 | -0.081 | -0.090 | 0.066 | 0.019 | -0.060 | -0.084 | -0.077 | -0.066 | 0.093 | 0.059 | 0.054 | 0.013 | 0.073 |
| Icp Mean 72H | -0.053 | -0.085 | -0.095 | 0.057 | 0.007 | -0.063 | -0.018 | -0.039 | 0.018 | -0.040 | -0.000 | 0.021 | 0.053 | -0.048 | 0.057 | 0.044 |
| Icp Peak | 0.059 | 0.044 | 0.027 | -0.042 | -0.065 | 0.009 | 0.001 | -0.057 | -0.009 | -0.062 | -0.011 | 0.060 | 0.068 | 0.020 | 0.013 | 0.036 |
| Icp Auc 7D | -0.039 | -0.063 | -0.068 | 0.042 | -0.005 | -0.034 | -0.000 | -0.036 | 0.003 | -0.040 | -0.009 | 0.027 | 0.044 | -0.030 | 0.067 | 0.034 |
| Icp At Day7 | 0.009 | -0.001 | -0.001 | -0.006 | -0.040 | 0.025 | 0.037 | -0.039 | -0.025 | -0.046 | -0.026 | 0.046 | 0.038 | 0.009 | 0.079 | 0.028 |
| Icp Time Above 20 | -0.021 | -0.019 | -0.022 | 0.005 | 0.007 | -0.019 | -0.027 | -0.002 | -0.012 | -0.041 | -0.034 | -0.011 | 0.007 | -0.006 | -0.043 | 0.018 |
| Icp Time Above 25 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Ni Resolution Time | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |


---

## Supplementary Table 8. Extended Feature-Set ML Comparison (Bayesian ODE + Drug Features)

5-fold cross-validated performance across five progressive feature sets: Clinical (28 variables), Clinical+DetQSP (+21 deterministic ODE features), Clinical+BayesODE (+19 Bayesian MAP-estimated ODE features), Clinical+Drug (+drug PK/PD features), Clinical+All (all augmented features). LASSO used for regression outcomes; Logistic Regression for classification. ΔR²/ΔAUC reported relative to Clinical baseline.

| Outcome | Clinical | Clinical+DetQSP | Clinical+BayesODE | Clinical+Drug | Clinical+All | Δ vs Clinical (DetQSP) | Δ vs Clinical (BayesODE) | Δ vs Clinical (Drug) | Δ vs Clinical (All) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GOSE 12m | 0.409 | 0.408 | 0.410 | 0.409 | 0.408 | -0.001 | +0.001 | +0.000 | -0.001 |
| FIM Total 12m | 0.514 | 0.509 | 0.515 | 0.513 | 0.509 | -0.005 | +0.001 | -0.001 | -0.005 |
| Barthel 12m | 0.489 | 0.482 | 0.490 | 0.488 | 0.484 | -0.007 | +0.001 | -0.001 | -0.006 |
| HADS-Anxiety 12m | 0.313 | 0.314 | 0.313 | 0.313 | 0.313 | +0.001 | -0.000 | -0.000 | +0.000 |
| MoCA 12m | 0.161 | 0.159 | 0.160 | 0.159 | 0.155 | -0.002 | -0.001 | -0.002 | -0.007 |
| QOLIBRI-OS 12m | 0.163 | 0.162 | 0.163 | 0.163 | 0.162 | -0.001 | -0.000 | -0.000 | -0.001 |
| DRS 12m | 0.255 | 0.253 | 0.256 | 0.255 | 0.253 | -0.002 | +0.001 | -0.000 | -0.002 |
| Return-to-Work | 0.748 | 0.746 | 0.716 | 0.701 | 0.740 | -0.003 | -0.032 | -0.047 | -0.008 |
| 12m Mortality | 0.564 | 0.579 | 0.568 | 0.549 | 0.573 | +0.015 | +0.004 | -0.015 | +0.009 |


---

## Supplementary Table 9. Enhanced Feature-Set Performance (F_BASE to F_ALL)

5-fold cross-validated R² and AUC for five feature set tiers used in the performance enhancement analysis (generated by `brain_injury_performance.py`). **Important: These results use Stacking ensemble models (Ridge + RF + XGBoost meta-learner), not the individual LASSO/RF models reported in Supplementary Table 1.** The higher absolute R² values reflect both the stacking architecture and additional engineered features. F_BASE: 28 clinical variables. F_MECH: +21 QSP-ODE mechanistic features. F_NEW: +simulated enhanced clinical features (cognitive reserve proxy, sleep quality, physical activity, APOE4 carrier status). F_BIO: +inflammatory biomarker features. F_ALL: all features combined. Best model per feature set reported (selected by CV R²/AUC). Values are mean (SD) across 5 folds.

| Outcome | F_BASE | F_MECH | F_NEW | F_BIO | F_ALL | Δ vs F_BASE (F_MECH) | Δ vs F_BASE (F_NEW) | Δ vs F_BASE (F_BIO) | Δ vs F_BASE (F_ALL) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GOSE 12m | 0.670 (0.051) | 0.670 (0.049) | 0.696 (0.048) | 0.670 (0.050) | 0.694 (0.046) | +0.001 | +0.027 | +0.000 | +0.024 |
| FIM Total 12m | 0.653 (0.032) | 0.645 (0.029) | 0.680 (0.032) | 0.650 (0.033) | 0.678 (0.029) | -0.007 | +0.027 | -0.002 | +0.025 |
| Barthel 12m | 0.624 (0.037) | 0.620 (0.033) | 0.641 (0.038) | 0.622 (0.035) | 0.639 (0.034) | -0.003 | +0.017 | -0.002 | +0.015 |
| HADS-Anxiety 12m | 0.522 (0.013) | 0.522 (0.013) | 0.525 (0.017) | 0.523 (0.014) | 0.526 (0.017) | -0.000 | +0.003 | +0.001 | +0.003 |
| MoCA 12m | 0.404 (0.026) | 0.391 (0.031) | 0.447 (0.021) | 0.391 (0.034) | 0.446 (0.021) | -0.013 | +0.043 | -0.013 | +0.042 |
| Return-to-Work | 0.856 (0.019) | 0.848 (0.013) | 0.863 (0.005) | 0.852 (0.015) | 0.866 (0.005) | -0.008 | +0.007 | -0.003 | +0.010 |
| 12m Mortality | 0.587 (0.036) | 0.588 (0.014) | 0.578 (0.029) | 0.582 (0.034) | 0.574 (0.034) | +0.001 | -0.009 | -0.006 | -0.014 |


---

## Supplementary Table 10. Clinically Testable Hypotheses Generated from the Simulation Framework

Each hypothesis arises directly from a quantitative finding in this study and is framed as a falsifiable statement suitable for prospective clinical evaluation. Supporting evidence column references specific results from the current simulation; Suggested Design column proposes the minimal study design required for hypothesis testing in real patient data. SAH: subarachnoid haemorrhage; TBI: traumatic brain injury; ICH: intracerebral haemorrhage; GCS: Glasgow Coma Scale; APACHE II: Acute Physiology and Chronic Health Evaluation II; GOSE: Glasgow Outcome Scale Extended; HADS: Hospital Anxiety and Depression Scale; PCL-5: PTSD Checklist DSM-5; PHQ-9: Patient Health Questionnaire-9; ICP: intracranial pressure; PRx: pressure reactivity index; AR index: autoregulation index; PI: prediction interval; AUC: area under the receiver operating characteristic curve; R²: coefficient of determination.

| ID | Category | Hypothesis | Supporting Evidence (this study) | Suggested Study Design | Primary Endpoint |
| --- | --- | --- | --- | --- | --- |
| H1 | Clinical Triage | A 5-variable parsimonious instrument (GCS, age, APACHE II, delirium, ICU anxiety score) assessed at 48–72 h post-admission prospectively identifies patients at high risk of poor 12-month functional and psychiatric outcomes with ≥90% of the discriminative accuracy of a full 28-variable model | Sensitivity analysis SA2: top-5 SHAP variables retained ≥93% of predictive signal for GOSE (R²=0.371 vs 0.400 with K=27) and FIM Total (R²=0.489 vs 0.497); five variables span both admission severity (GCS, age, APACHE II) and ICU course (delirium, anxiety) | Prospective neurocritical care cohort (n≥500); 5-variable score vs full clinical assessment at 48–72 h; follow-up at 12 months | C-statistic (AUC) for GOSE <5 at 12 months; non-inferiority margin ΔAUC <0.05 vs full model |
| H2 | Psychiatric Sequelae | Psychiatric burden (anxiety, depression, PTSD) is equivalent across moderate and severe TBI severity strata, and pre-injury vulnerability (prior psychiatric history, alcohol misuse) is the dominant driver of psychiatric outcome irrespective of injury severity | TBI subgroup analysis (n=804): HADS-Anxiety p_adj=0.626, PCL-5 p_adj=0.196 across severity strata despite large functional difference (GOSE r_rb=−0.535, p_adj<0.001); SHAP: prior psychiatric history ranked 6th universal predictor (mean |SHAP|=0.756) | Prospective observational TBI cohort; HADS/PHQ-9/PCL-5 at 3, 6, 12 months; stratified by GCS-based severity (mild ≤13, moderate 9–12, severe ≤8); pre-injury psychiatric history collected at admission | Between-group HADS-Anxiety score difference at 12 months (Wilcoxon rank-sum with BH-FDR); secondary: proportion meeting caseness threshold (HADS-A ≥11) by severity group |
| H3 | Psychiatric Screening (SAH) | ICU anxiety score assessed within 48–72 h of admission is an independent predictor of psychiatric outcomes (HADS-Anxiety, PHQ-9, PCL-5) at 12 months above and beyond APACHE II in SAH survivors, supporting routine GAD-7 or HADS integration into SAH ICU nursing assessment | ICU anxiety ranked 3rd universal SHAP predictor (mean |SHAP|=0.588), surpassing APACHE II (0.519), across all 12 outcomes; SAH-specific finding: psychiatric sequelae are the dominant long-term burden, frequently undetected in clinical practice | Prospective SAH cohort (n≥200); GAD-7 or adapted ICU anxiety screen administered at 48–72 h; follow-up at 3, 6, 12 months; multivariable logistic regression adjusting for age, APACHE II, Hunt-Hess grade | Incremental C-statistic for psychiatric caseness (HADS-A ≥11 at 12 months) with vs without ICU anxiety score as predictor; secondary: PHQ-9 ≥10 and PCL-5 ≥33 |
| H4 | QSP / Mechanistic Signal | PRx-derived cerebrovascular autoregulation index (AR index) from continuous bedside ICP monitoring carries incremental predictive signal for 12-month GOSE beyond routine clinical severity scores (GCS, APACHE II) in TBI patients with ICP monitoring | Simulated AR index was the top QSP mechanistic predictor for GOSE (LASSO |coef|=0.36) but mechanistic features overall added ΔR²<0.005 vs clinical-only model — attributed to mathematical collinearity with clinical severity inputs that would not apply to real waveform-derived continuous indices | Prospective TBI cohort with continuous ICP monitoring (n≥150); real-time PRx computation (30-min rolling correlation ICP vs MAP); LASSO models with and without PRx-AR index; evaluate at ICP-monitoring subset only | ΔR² for GOSE 12m with vs without PRx-AR index (LASSO); equivalence null: ΔR² <0.005; superiority threshold: ΔR² ≥0.020 |
| H5 | Multimodal Integration | Addition of automated CT imaging features (lesion volume, perihaematoma oedema extent, midline shift) to the clinical-only model significantly improves 12-month outcome prediction in TBI and ICH, where the current 28-variable clinical model has limited accuracy for mortality | ICH hold-out mortality AUC=0.619 (modest); simulation lacked structural injury data; imaging features are expected to carry independent mechanistic signal (haematoma expansion, mass effect) not captured by physiological scores | Retrospective cohort from CENTER-TBI (n=4,509) or TRACK-TBI (n=2,700) with linked acute CT; automated segmentation (AI-assisted); add lesion volume, oedema index, midline shift to clinical model as additional predictors | ΔAUC for 12-month mortality in ICH; ΔR² for GOSE 12m in TBI; with vs without imaging features (DeLong test for AUC comparison) |
| H6 | Drug Personalisation | Bayesian ODE-derived drug-response phenotype (Rapid Responder, Partial Responder, Non-Responder), estimated from individual patient ICP dynamics, identifies a clinically actionable subgroup in whom hypertonic saline produces superior ICP reduction compared with mannitol | Population-level: mannitol and HSS statistically equivalent (1.08±1.20 vs 0.99±1.17 mmHg ICP reduction, p=NS); individual Bayesian posterior estimates showed SD~1.2 mmHg and three separable phenotypes by PCA; phenotyping ΔR²<0.005 on outcomes due to simulation PK/PD limitations | Prospective observational study or adaptive RCT in ICP-monitored TBI patients (n≥120); Bayesian phenotype assigned at baseline using first 2-h ICP response; primary intervention: mannitol 20% (0.5 g/kg) vs HSS 3% (77 mEq/20 min); 4-h ICP outcome | ICP ≤20 mmHg at 4 h by phenotype group; interaction test: phenotype × drug assignment; secondary: CPP ≥60 mmHg sustained at 4 h |
| H7 | Uncertainty Quantification | Conformal split-inference prediction intervals improve clinician confidence in prognostic decisions and reduce inappropriate care escalation compared with point-estimate-only models in neurocritical care prognostication | Conformal PIs achieved 94.4% empirical coverage vs 25–33% for bootstrap at 95% nominal — a 3-fold miscalibration that would produce clinically significant overconfidence if bootstrap were used in decision support | Clinician vignette-based randomised crossover study; neurocritical care physicians (n≥60) presented identical clinical cases with: (a) point estimate only, (b) point estimate + conformal PI, (c) point estimate + bootstrap PI; decision confidence rated on 7-point Likert scale | Decision confidence Likert score (primary); secondary: appropriateness of care escalation decisions rated by blinded expert panel; proportion of decisions consistent with stated prognosis uncertainty |
| H8 | External Validation | The 5-variable LASSO prediction model trained on this synthetic cohort replicates in real neurocritical care registry data with acceptable calibration (Brier score <0.20) and discrimination (GOSE AUC >0.70) in the CENTER-TBI and TRACK-TBI cohorts | Hold-out optimism bias mean ΔR²=+0.007 across 12 outcomes (excellent internal generalisability); LASSO selected identical top-5 predictors across all regression outcomes; model coefficients are stable and interpretable for prospective application | Apply pre-trained LASSO coefficients (fixed, no retraining) to CENTER-TBI (n=4,509) and TRACK-TBI (n=2,700) 12-month follow-up cohorts; compute calibration and discrimination metrics; subgroup analyses by diagnosis (TBI, SAH, ICH, stroke) and age | C-statistic for GOSE <5 at 12 months; Brier score; calibration slope (Cox recalibration); secondary: Harrell's C across functional, psychiatric, and QoL outcomes |

*Hypotheses are listed in order of clinical immediacy, from bedside triage instruments (H1–H3) requiring modest prospective data collection, to mechanistic validation requiring specialist monitoring (H4), imaging infrastructure (H5), interventional design (H6), methodological evaluation (H7), and large-scale external validation (H8). All hypotheses are directly falsifiable using existing neurocritical care outcome registries or prospective cohort infrastructure.*

---
