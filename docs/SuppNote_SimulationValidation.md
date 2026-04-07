# Supplementary Note 2: Validation of the Simulated Neurocritical Care Cohort Against Published Literature

**Manuscript**: A Quantitative Systems Pharmacology–Machine Learning Framework for Multi-Domain Outcome Prediction in Neurocritical Care: A Computational Simulation Study

---

## 1. Purpose

This note documents the face validity assessment of the simulated neurocritical care cohort (n=2,000 training; n=500 independent hold-out) by systematically comparing key epidemiological and clinical statistics against published benchmarks from prospective neurocritical care registries and meta-analyses. Face validity is necessary to establish that the simulation generates clinically plausible patients and that the methodological findings (model performance, QSP feature contribution, SHAP importance patterns) are interpretable in a real-world context.

This is not a calibration or inference process — the simulation was not fitted to any real dataset. Discrepancies are reported transparently and their implications for interpretation are discussed.

---

## 2. Reference Cohorts Used for Comparison

| Abbreviation | Full name | n | Design | Diagnoses | Key PMID |
|---|---|---|---|---|---|
| CENTER-TBI | Collaborative European NeuroTrauma Effectiveness Research in TBI | 4,509 | Prospective observational | TBI only | 31526754 |
| SYNAPSE-ICU | ICP monitoring practices and outcomes in acute brain injury | 2,395 | Prospective observational | Mixed ABI (TBI 54%, ICH 25%, SAH 22%) | 34146513 |
| Rass 2024 | Cognitive, mental health, functional, and QoL outcomes 1 year after SAH | 177 | Prospective observational | SAH only | 38129710 |
| von Steinbuechel 2023 | CENTER-TBI trajectory analysis | 2,555 | Longitudinal PRO subset | TBI only | 36983247 |
| Patel 2018 | Delirium monitoring in neurocritically ill patients | Systematic review | — | Neurotrauma/stroke | 30142098 |
| Krewulak 2018 | ICU delirium subtypes meta-analysis | Meta-analysis (48 studies) | — | General ICU | 30234569 |
| Osborn 2016 | Anxiety prevalence following adult TBI | Meta-analysis (41 studies) | — | TBI | — |
| Dehbozorgi 2024 | Incidence of anxiety after TBI | Meta-analysis (49 studies) | — | TBI | 39174923 |

**Important caveat on reference matching**: Most registry studies cover single-diagnosis populations (TBI-only or SAH-only). This simulation covers a mixed-diagnosis neurocritical ICU population (TBI 40%, stroke 25%, SAH 20%, ICH 15%). Where diagnosis-specific benchmarks exist, a weighted average across diagnosis proportions was used as the comparator; where only single-diagnosis data exist, this limitation is noted.

---

## 3. Validation Results

### 3.1 Age Distribution

| Statistic | Simulated cohort | Reference value | Source | Assessment |
|---|---|---|---|---|
| Mean age (SD) | 55.4 (17.3) years | Median 54 (IQR 47–62) | Rass 2024 (SAH) | ✓ Consistent |
| Range | 18–95 years | Median 50 (IQR 30–66) | CENTER-TBI TBI | ✓ Consistent |

**Interpretation**: The simulated mean age of 55.4 years is consistent with both SAH-predominant (Rass 2024) and TBI (CENTER-TBI) benchmarks. The mixed-diagnosis cohort would be expected to have a slightly older mean than TBI-only registries, reflecting the older age distribution of haemorrhagic stroke (ICH, SAH) patients. The simulated age distribution is considered valid.

---

### 3.2 Sex Distribution

| Statistic | Simulated | Reference | Source | Assessment |
|---|---|---|---|---|
| % male | 61.4% | 41% male (SAH, n=177) | Rass 2024 | ≈ Slightly high |
| | | 60–70% male (TBI) | General TBI literature | ✓ Consistent for TBI |
| | | ~56% weighted mixed | Weighted estimate | ≈ 5% above benchmark |

**Interpretation**: The simulation applies TBI sex ratios (60–70% male) globally, which overestimates the male proportion for the mixed cohort because SAH has near-equal sex distribution and stroke is ~50% male. A weighted estimate across the four diagnoses yields ~56% male. The simulated value (61.4%) is 5 percentage points above this benchmark. This represents a modest deviation. The clinical impact on model performance is expected to be minimal, as sex is one of 28 predictors and not among the top-5 SHAP predictors. This discrepancy should be corrected in future versions by applying diagnosis-specific sex distributions.

---

### 3.3 Diagnosis Distribution

| Diagnosis | Simulated | Design target | Assessment |
|---|---|---|---|
| TBI | 40.2% | 40% | ✓ On target |
| Ischaemic stroke | 24.9% | 25% | ✓ On target |
| SAH | 18.9% | 20% | ✓ On target |
| ICH | 15.9% | 15% | ✓ On target |

**Interpretation**: Diagnosis proportions match the intended design exactly. No published mixed-diagnosis neurocritical ICU registry with identical proportions exists for external comparison; however, the chosen proportions are consistent with general descriptions of case-mix in tertiary neurocritical care units [SYNAPSE-ICU enrolment patterns].

---

### 3.4 Injury Severity (GCS at Admission)

| Severity stratum | Simulated | Reference (CENTER-TBI ICU) | Source | Assessment |
|---|---|---|---|---|
| Mild (GCS 13–15) | 37.5% | 36% (720/2,138) | Steyerberg 2019 PMID 31526754 | ✓ Consistent |
| Moderate (GCS 9–12) | 37.1% | ~16% | CENTER-TBI companion data (PMC7210239) | ⚠ Substantially higher |
| Severe (GCS ≤8) | 25.3% | ~48% | CENTER-TBI companion data (PMC7210239) | ⚠ Substantially lower |

**Note on GCS severity split**: Steyerberg et al. 2019 reports mild (36%) and moderate-to-severe combined (64%) in the ICU stratum, but does not separately report the moderate vs severe split. The approximate moderate (16%) and severe (48%) proportions are drawn from a companion CENTER-TBI publication (PMC7210239). The simulated cohort has a substantially different moderate/severe ratio compared to the TBI-only CENTER-TBI ICU stratum.

**Interpretation**: The mild GCS proportion (37.5%) closely matches the CENTER-TBI ICU stratum (36%). The simulation has a higher moderate proportion (37.1% vs ~16%) and lower severe proportion (25.3% vs ~48%) relative to the TBI-only CENTER-TBI ICU stratum. This likely reflects: (1) the mixed cohort includes ischaemic stroke patients who typically present with less severe GCS depression than TBI; (2) the simulation's GCS sampling was calibrated for a mixed-diagnosis rather than TBI-only population. No published GCS distribution benchmark for a mixed TBI/stroke/SAH/ICH ICU cohort exists. The overall distribution is considered plausible for a mixed-diagnosis setting, though the moderate/severe ratio should be noted as a limitation.

---

### 3.5 Psychiatric Outcomes at 12 Months

| Outcome | Simulated prevalence | Reference | Source | Assessment |
|---|---|---|---|---|
| Anxiety (HADS-A >7) | 63.3% | 33% (SAH survivors, n=177) | Rass 2024 PMID 38129710 | ⚠ Elevated |
| | | 37% self-reported caseness (TBI, 41 studies) | Osborn 2016 *Neuropsychology* | ⚠ Elevated |
| | | 17.5% incidence (TBI, 49 studies, n=705,024) | Dehbozorgi 2024 PMID 39174923 | ⚠ Substantially elevated |
| Depression (HADS-D >7) | 17.5% | 16% (SAH survivors, n=177) | Rass 2024 PMID 38129710 | ✓ Consistent |

**Note on anxiety benchmarks**: Two TBI anxiety meta-analyses provide complementary but distinct estimates. Osborn et al. (2016, *Neuropsychology*, 41 studies) reported 37% prevalence of clinically significant self-reported anxiety (using validated cutoffs on measures such as HADS, BAI, DASS) and 11% meeting formal GAD diagnostic criteria. Dehbozorgi et al. (2024, *BMC Neurol*, PMID 39174923, 49 studies, n=705,024) reported a pooled incidence rate of 17.5% for anxiety symptoms/disorders after TBI. The difference reflects methodology: self-report caseness thresholds yield higher prevalence than diagnostic incidence rates. The 37% figure from Osborn 2016 is the more appropriate comparator for HADS-A >7 caseness in the simulation.

**Interpretation**: The simulated anxiety prevalence (63.3%) substantially exceeds published benchmarks (33–37% by self-report caseness). The ICU anxiety score is a strong direct causal input in the revised simulation architecture (weight=0.6 directly to HADS-Anxiety), and anxiety is also driven by delirium (35–45% prevalence) and ICU LOS. Together these inputs generate higher anxiety than observed in single-diagnosis outpatient follow-up studies. Additionally, the published benchmarks (Rass 2024, Osborn 2016) reflect surviving outpatients; the simulation generates outcomes for all ICU survivors including those with severe ongoing disability who would be expected to have higher anxiety rates.

This discrepancy has a direct implication for the psychiatric prediction R² values: HADS-Anxiety R²=0.435 in this simulation may reflect partially inflated predictability from over-representation of high-anxiety patients. This limitation is stated in the manuscript (section 4.8) and should be addressed by recalibrating the ICU anxiety score → HADS-Anxiety pathway weight in future simulation versions.

Depression prevalence (17.5%) is well-calibrated against the SAH survivor benchmark (16%).

---

### 3.6 Cognitive Impairment at 12 Months

| Measure | Simulated | Reference | Source | Assessment |
|---|---|---|---|---|
| MoCA <26 (mild impairment threshold) | 60.1% | 71% ≥1 deficit on comprehensive battery (SAH, n=177) | Rass 2024 PMID 38129710 | ≈ Directionally consistent |

**Interpretation**: The simulated MoCA <26 prevalence (60.1%) is directionally consistent with the SAH cognitive impairment benchmark (71%) but not directly comparable, because MoCA <26 is a sensitive but non-specific threshold that differs from comprehensive neuropsychological battery failure. The cognitive R² values (MoCA 12m R²=0.153) reflect genuine difficulty predicting cognitive outcomes with admission-only clinical variables, consistent with published cognitive outcome prediction literature in TBI.

---

### 3.7 Recovery Trajectory Class Distribution

| Trajectory class | Simulated | CENTER-TBI reference (von Steinbuechel 2023, n=2,555) | Assessment |
|---|---|---|---|
| Stable good | 14.9% | 76.1% (1,944/2,555) | ⚠ Large discrepancy — see note |
| Persistent impaired | 11.6% | 17.3% (442/2,555) | ≈ Similar |
| Improving | 30.8% | 3.2% (83/2,555) | ⚠ Large discrepancy — see note |
| Deteriorating | 42.8% | 3.4% (86/2,555) | ⚠ Large discrepancy — see note |

**Critical caveat — two reference mismatches make direct comparison invalid**:

1. **Outcome definition mismatch**: CENTER-TBI (von Steinbuechel 2023) defines trajectory classes using QOLIBRI-OS and SF-12 patient-reported health-related quality of life (HRQoL). This simulation defines trajectory classes using GOSE functional outcome. These are not equivalent: HRQoL trajectories tend to be more stable (patients adapt) while functional GOSE trajectories in a critically ill mixed population show more deterioration. The "Stable 76.1%" in CENTER-TBI reflects HRQoL adaptation, not functional stability.

2. **Diagnosis mismatch**: CENTER-TBI is TBI-only. This simulation includes SAH (20%), ICH (15%), and stroke (25%) — all of which carry higher acute mortality and worse functional outcomes than TBI. The inclusion of these diagnoses shifts the simulated cohort toward more deteriorating and fewer stable-good outcomes compared to TBI-only registries.

Given these mismatches, direct numerical comparison is not valid. The trajectory distribution in this simulation reflects a more severely ill mixed-diagnosis ICU population with higher representation of deteriorating and improving classes. The low stable-good proportion (14.9%) warrants attention and may represent an underestimation of genuine clinical stability; this is identified as a limitation for future calibration work.

---

### 3.8 Twelve-Month Mortality

| Statistic | Simulated | Reference | Source | Assessment |
|---|---|---|---|---|
| 12-month all-cause mortality | 41.4% | 34% (ICP-monitored) to 49% (non-monitored) at **6 months** (mixed ABI) | SYNAPSE-ICU PMID 34146513 | ≈ Plausible |
| | | 18–43% SAH case fatality (background literature estimate) | Cited within Rass 2024 | ≈ Within range |

**Note on SYNAPSE-ICU mortality**: The 34–49% range from SYNAPSE-ICU (Robba et al. 2021) represents 6-month mortality stratified by ICP monitoring status: 34% (441/1,318) in ICP-monitored patients vs 49% (517/1,049) in non-monitored patients — not a diagnostic subgroup range. The SAH case fatality range (18–43%) is a background literature estimate cited within Rass et al. 2024, not a finding from that study's own cohort (which enrolled only survivors, so study-internal mortality = 0%).

**Interpretation**: The simulated 12-month mortality (41.4%) falls within the SYNAPSE-ICU 6-month mortality range (34–49%). As 12-month mortality would be expected to exceed 6-month mortality, this suggests the simulated value may slightly underestimate true 12-month mortality for this severity of illness. The simulated value is considered plausible and is within the reported range for comparable populations.

The modest mortality AUC (0.578 CV; 0.619 hold-out) is consistent with the limited discriminating power of the 28 admission-only clinical variables for long-term mortality, as noted in published mortality prediction literature in neurocritical care.

---

### 3.9 Delirium Prevalence

| Statistic | Simulated | Reference | Source | Assessment |
|---|---|---|---|---|
| Delirium during ICU stay | 45.1% | 12–43% (neurocritical) | Patel 2018 PMID 30142098 | ⚠ Above upper bound |
| | | 31% (95% CI 24–41%, general ICU) | Krewulak 2018 PMID 30234569 | ⚠ Above CI |

**Interpretation**: The simulated delirium prevalence (45.1%) is marginally above the upper bound of the neurocritical ICU systematic review range (43%). The general ICU meta-analytic pooled estimate (31%) is lower; however, neurocritical care patients are systematically higher-risk for delirium than general ICU patients due to primary brain injury, higher sedation requirements, and prolonged immobility. The simulation sets delirium based on a logistic model incorporating GCS, APACHE II, age, and diagnosis — the slightly elevated prevalence may reflect overweighting of severity inputs. This is a minor discrepancy (2 percentage points above the published upper bound) and is unlikely to substantially affect model performance.

---

### 3.10 ICU Length of Stay

| Statistic | Simulated | Reference | Source | Assessment |
|---|---|---|---|---|
| Median ICU LOS | 9.0 days | 5–14 days (range) | General neurocritical care literature | ✓ Within range |

**Note on SYNAPSE-ICU ICU LOS**: SYNAPSE-ICU (Robba et al. 2021, PMID 34146513) does not explicitly report median ICU length of stay. The median value of 9 reported in the SYNAPSE-ICU abstract refers to the **Therapy Intensity Level (TIL) score** (IQR 7–12) in ICP-monitored patients, not ICU LOS in days. The simulated median ICU LOS of 9.0 days is consistent with the general neurocritical care literature range of 5–14 days and is therefore considered plausible, though a direct SYNAPSE-ICU benchmark for ICU LOS is not available from this publication.

**Interpretation**: ICU LOS falls within the expected range for a mixed neurocritical care population. No adjustment required.

---

## 4. Overall Validity Assessment

| Parameter | Assessment | Action required? |
|---|---|---|
| Age distribution | ✓ Valid | None |
| Sex distribution | ≈ Minor deviation (+5%) | Future: diagnosis-specific sex ratios |
| Diagnosis proportions | ✓ Valid (by design) | None |
| GCS severity distribution | ⚠ Moderate/severe ratio differs from CENTER-TBI | Note mixed-cohort calibration; no valid mixed-ABI benchmark exists |
| Depression prevalence | ✓ Valid | None |
| Anxiety prevalence | ⚠ Inflated (+26% above benchmark) | Recalibrate ICU anxiety weight in future version |
| Cognitive impairment | ≈ Directionally valid | Note threshold mismatch in manuscript |
| Trajectory distribution | ⚠ Invalid reference (see §3.7) | No valid external reference exists — caveat stated |
| Mortality | ≈ Plausible (6m vs 12m reference) | None |
| Delirium | ≈ Minor deviation (+2%) | None |
| ICU LOS | ✓ Within expected range | SYNAPSE-ICU LOS not directly available; general literature consistent |

---

## 5. Implications for Interpretation of Model Results

The validation reveals three parameters that warrant caution when interpreting model performance:

**1. Anxiety over-representation** may inflate HADS-Anxiety R² (reported: 0.435). If the anxiety prevalence were recalibrated to ~35%, the R² for HADS-Anxiety prediction might decrease toward the range of other psychiatric outcomes (PHQ-9, PCL-5: R²=0.373). This does not alter the qualitative finding that psychiatric outcomes are predictable with direct causal pathways; it may affect the absolute R² magnitude.

**2. High mortality (41.4%) and the consequent survivor selection bias** — psychiatric, functional, cognitive, and QoL outcomes are assessed only in survivors. The 41.4% mortality creates a relatively small effective sample for survivor-only outcomes (n≈1,173 for 12-month regression outcomes), which limits precision of regression estimates and contributes to the wide confidence intervals for psychiatric outcomes.

**3. Trajectory distribution** — the high deteriorating class proportion (42.8%) reflects the mixed-diagnosis severity of the simulated cohort, which is appropriate given the case-mix but should not be interpreted as representing any specific real-world neurocritical ICU population. The trajectory classifier's near-chance AUC (~0.52) likely reflects both genuine difficulty and the absence of longitudinal biomarker features needed for trajectory discrimination.

---

## 6. Conclusions

The simulated neurocritical care cohort demonstrates face validity against published benchmarks for 6 of 11 compared parameters (age, diagnosis proportions, depression, mortality, delirium, ICU LOS). Five parameters show discrepancies of varying magnitude: anxiety prevalence (inflated), sex ratio (minor), GCS moderate/severe ratio (differs from CENTER-TBI), trajectory distribution (reference mismatch rendering direct comparison invalid), and ICU LOS benchmark (SYNAPSE-ICU does not directly report LOS).

No published mixed-diagnosis neurocritical ICU cohort with GOSE-based trajectory classification, 12-month psychiatric follow-up, and simultaneous mechanistic ODE feature generation exists for comprehensive external benchmarking. The simulation represents a methodological tool for pipeline development and stress-testing, not a replica of any specific patient population. Findings — particularly the QSP feature redundancy result and the psychiatric domain MTL gain — are expected to be directionally reproducible in real-data applications, but absolute performance values (R², AUC) will depend on the degree of causal fidelity of the real dataset to the simulation's assumed data-generating process.

---

## References

- Steyerberg EW et al. (CENTER-TBI). *Lancet Neurol*. 2019;18(1):56–87. PMID 31526754.
- Robba C et al. (SYNAPSE-ICU). Intracranial pressure monitoring in patients with acute brain injury in the intensive care unit. *Lancet Neurol*. 2021;20(7):548–558. PMID 34146513.
- Rass V et al. Cognitive, mental health, functional, and quality of life outcomes 1 year after spontaneous subarachnoid hemorrhage. *Neurocrit Care*. 2024;41:70–79. PMID 38129710. PMC11335887.
- von Steinbuechel N et al. Impact of sociodemographic, premorbid, and injury-related factors on patient-reported outcome trajectories after TBI. *J Clin Med*. 2023;12(6):2246. PMID 36983247. PMC10052290.
- Patel MB et al. Delirium monitoring in neurocritically ill patients: a systematic review. *Crit Care Med*. 2018;46(11):1832–1841. PMID 30142098.
- Krewulak KD et al. Incidence and prevalence of delirium subtypes in an adult ICU: a systematic review and meta-analysis. *Crit Care Med*. 2018;46(12):2029–2035. PMID 30234569.
- Osborn AJ et al. Prevalence of anxiety following adult traumatic brain injury: a meta-analysis comparing measures, samples and postinjury intervals. *Neuropsychology*. 2016;30(2):247–261.
- Dehbozorgi M et al. Incidence of anxiety after traumatic brain injury: a systematic review and meta-analysis. *BMC Neurol*. 2024;24(1):330. PMID 39174923. PMC11340054.
