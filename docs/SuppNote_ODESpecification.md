---

# Supplementary Note 1: Mathematical Specification of the Quantitative Systems Pharmacology ODE Models

**Manuscript**: A Quantitative Systems Pharmacology–Machine Learning Framework for Multi-Domain Outcome Prediction in Neurocritical Care: A Computational Simulation Study

---

## 1. Overview

The QSP component of the pipeline comprises two coupled ordinary differential equation (ODE) systems that simulate the acute-phase pathophysiology of neurocritical injury. Both systems are solved numerically per patient using patient-specific parameters derived from admission clinical variables. The 21 mechanistic features extracted from these simulations are described in full in Section 4.

**Biological rationale**: Mechanistic ODE modelling captures time-integrated physiological processes (ICP burden, inflammatory trajectory, neuroprotection) that are not directly observable from admission snapshots but that drive downstream outcome. By condensing these trajectories into summary features, the QSP layer provides interpretable biological signals to the ML layer.

---

## 2. ODE System I — Intracranial Pressure / Cerebral Perfusion Pressure Dynamics

### 2.1 Biological Basis

The ICP/CPP model extends the Monroe-Kellie doctrine, which states that within the rigid skull the total cranial volume is fixed:

$$V_{\text{brain}} + V_{\text{blood}} + V_{\text{CSF}} = V_{\text{total}} = \text{constant}$$

Any net increase in compartment volumes (oedema, haematoma) is compensated primarily by CSF displacement. Once compensatory reserve is exhausted, ICP rises steeply. Cerebral perfusion pressure is defined as:

$$\text{CPP} = \text{MAP} - \text{ICP}$$

The model incorporates cerebrovascular autoregulation (the capacity to maintain constant cerebral blood flow [CBF] across a range of perfusion pressures), which is variably impaired after brain injury.

### 2.2 State Variables

| Symbol | Description | Units | Initial value |
|---|---|---|---|
| ICP | Intracranial pressure | mmHg | ICP_mean_admission (patient-specific) |
| V_edema | Cerebral oedema volume | mL | 0.01 |
| V_csf | CSF compartment volume | mL | 100.0 |

### 2.3 ODE Equations

**Cerebrovascular autoregulation (Lassen curve adaptation)**:

The autoregulation index (AR_index ∈ [0,1]; 0 = intact, 1 = fully pressure-passive) is derived from GCS:

$$\text{AR\_index} = \text{clip}\!\left(1 - \frac{\text{GCS} - 3}{12},\ 0.1,\ 0.95\right)$$

CBF is modelled as a weighted combination of autoregulated (sigmoidal) and pressure-passive (linear) components:

$$\text{CBF}_{\text{autoregulated}} = \frac{\text{CBF}_{\text{normal}}}{1 + e^{-(\text{CPP} - \text{CPP}_{\text{opt}}) / \Delta_{\text{CPP}}}}$$

$$\text{CBF}_{\text{passive}} = \text{CBF}_{\text{normal}} \cdot \frac{\text{CPP}}{80}$$

$$\text{CBF} = (1 - \text{AR\_index}) \cdot \text{CBF}_{\text{autoregulated}} + \text{AR\_index} \cdot \text{CBF}_{\text{passive}}$$

where CPP_opt = 70 mmHg and Δ_CPP = 20 mmHg.

**Brain compliance (Marmarou exponential approximation)**:

$$C_{\text{brain}} = \frac{C_{0,\text{brain}}}{1 + k_{\text{elastance}} \cdot \max(\text{ICP} - \text{ICP}_{\text{ref}},\ 0)}$$

where ICP_ref = 10 mmHg and k_elastance = 0.12 mmHg⁻¹.

**CSF dynamics (Davson’s equation)**:

$$\frac{dV_{\text{CSF}}}{dt} = I_f - \frac{\max(\text{ICP} - P_{\text{venous}},\ 0)}{R_{\text{CSF}}}$$

where I_f = 0.35 mL/min (constant CSF formation rate), P_venous = 6 mmHg, and R_CSF is the CSF outflow resistance (elevated by injury, Table 1).

**Vasogenic oedema**:

$$\frac{dV_{\text{edema}}}{dt} = k_{\text{BBB}} \cdot e^{-k_{\text{BBB}} \cdot t / 24} - k_{\text{edema\_clear}} \cdot V_{\text{edema}}$$

This models peak BBB disruption at t = 0 (admission) with exponential decay, partially offset by oedema clearance.

**ICP dynamics**:

$$\frac{d\text{ICP}}{dt} = \frac{\dot{V}_{\text{edema}} + \dot{V}_{\text{CSF}}}{C_{\text{brain}}}$$

### 2.4 Parameter Personalisation

| Parameter | Personalisation formula | Biological basis |
|---|---|---|
| MAP | max(70 − (APACHE II − 20) × 0.4, 50) mmHg | APACHE II reflects haemodynamic instability |
| AR_index | clip(1 − (GCS − 3)/12, 0.1, 0.95) | GCS-based autoregulation impairment |
| C₀_brain | 0.8 × (1 − 0.4 × severity_factor) × (1 − 0.003 × max(age − 40, 0)) | Reduced compliance with severity and ageing |
| R_CSF | 8.0 × diag_factor (TBI: 1.3; SAH: 1.5; ICH: 1.4; Stroke: 1.1) | Diagnosis-specific CSF outflow impairment |
| k_BBB | 0.05 × severity_factor × diag_factor | BBB disruption proportional to injury severity and type |
| k_edema_clear | 0.02 × (1 − 0.003 × max(age − 40, 0)) | Slower oedema clearance in older patients |
| CBF_normal | 50.0 mL/100g/min | Population normal value |

severity_factor = clip((15 − GCS)/12, 0.1, 1.0)

### 2.5 Simulation Parameters

| Parameter | Value |
|---|---|
| Time span | 0–168 hours (7 days) |
| Time evaluation points | 500 equally spaced |
| ODE solver | RK45 (Runge-Kutta 4th/5th order) |
| Relative tolerance | 1 × 10⁻⁴ |
| Absolute tolerance | 1 × 10⁻⁶ |
| Maximum step size | 1.0 hour |

### 2.6 ICP/CPP Features Extracted (11 features)

| Feature name | Description | Units |
|---|---|---|
| mech_icp_peak | Maximum ICP over 7 days | mmHg |
| mech_icp_mean_72h | Mean ICP during first 72 hours | mmHg |
| mech_icp_auc_7d | Area under ICP curve over 7 days (trapezoid rule) | mmHg·h |
| mech_icp_time_above_20 | Cumulative hours ICP > 20 mmHg | h |
| mech_icp_time_above_25 | Cumulative hours ICP > 25 mmHg | h |
| mech_icp_at_day7 | ICP at end of 7-day simulation | mmHg |
| mech_cpp_min | Minimum CPP over 7 days | mmHg |
| mech_cpp_mean | Mean CPP over 7 days | mmHg |
| mech_cpp_time_below_60 | Cumulative hours CPP < 60 mmHg | h |
| mech_ar_index | Cerebrovascular autoregulation index (patient-specific) | 0–1 |
| mech_cpp_optimal_time | Proportion of time CPP in therapeutic range 60–100 mmHg | fraction |

---

## 3. ODE System II — Neuroinflammation Cascade

### 3.1 Biological Basis

The neuroinflammation model captures the DAMP (Damage-Associated Molecular Pattern)-driven microglial activation cascade following acute brain injury. The framework is adapted from published mechanistic models of TBI neuroinflammation (Cheng et al. 2018; Hou et al. 2020) and extended to cover mixed ABI diagnoses (TBI, SAH, stroke, ICH).

**Pathophysiological sequence**: Acute injury releases DAMPs → pattern recognition receptors on microglia/macrophages → M1 (pro-inflammatory) activation → downstream cytokine production (IL-6, TNF-α, IL-1β) captured by neuroinflammation index NI → M2 (anti-inflammatory, reparative) counter-activation → NI suppression + neuroprotection/plasticity index NP elevation (BDNF, VEGF) → tissue repair or secondary injury depending on M1/M2 balance.

### 3.2 State Variables

| Symbol | Description | Units | Initial value |
|---|---|---|---|
| DAM | DAMP signal (primary inflammatory trigger) | arb. units | S₀ (patient-specific) |
| M1 | Pro-inflammatory microglial/macrophage activation | arb. units | 0.01 |
| M2 | Anti-inflammatory / reparative microglial activation | arb. units | 0.01 |
| NI | Neuroinflammation index (cytokine burden proxy) | arb. units | 0.01 |
| NP | Neuroprotection/plasticity index (BDNF/VEGF proxy) | arb. units | 0.1 |

### 3.3 ODE Equations

$$\frac{d\text{DAM}}{dt} = S_0 \cdot e^{-k_{\text{dam\_clear}} \cdot t} - k_{\text{dam\_deg}} \cdot \text{DAM}$$

$$\frac{dM1}{dt} = k_{M1} \cdot \text{DAM} \cdot \max\!\left(1 - \frac{M1}{M1_{\max}},\ 0\right) - k_{M1,\text{res}} \cdot M2 \cdot M1$$

$$\frac{dM2}{dt} = k_{M2} \cdot M1 - k_{M2,\text{deg}} \cdot M2$$

$$\frac{dNI}{dt} = k_{NI} \cdot M1 - k_{NI,\text{deg}} \cdot NI - k_{NI,M2} \cdot M2 \cdot NI$$

$$\frac{dNP}{dt} = k_{NP} \cdot M2 - k_{NP,\text{deg}} \cdot NP + k_{NP,\text{base}}$$

**Biological interpretation of terms**:
- S₀ · exp(−k_dam_clear · t): pulse-like DAMP release peaking at t = 0 (injury) and decaying
- k_M1 · DAM · (1 − M1/M1_max): logistic M1 activation with carrying capacity M1_max = 2.0
- k_M1,res · M2 · M1: M2-mediated resolution of M1 (cross-inhibition)
- k_NI,M2 · M2 · NI: M2-mediated cytokine suppression (anti-inflammatory resolution)
- k_NP,base: basal neuroprotective tone (reduced in elderly / pre-existing injury)

### 3.4 Parameter Personalisation

| Parameter | Personalisation formula | Biological basis |
|---|---|---|
| S₀ | 2.0 × severity × apache_factor × (1 + 0.3 × delirium) | DAMP release ∝ injury severity and systemic inflammation |
| k_dam_clear | 0.15 / age_factor | Slower DAMP clearance in elderly (inflammaging) |
| k_M1 | 0.5 × priming | Microglial priming by prior psych history / alcohol |
| k_M1,res | 0.4 / age_factor | Delayed M2-mediated resolution in elderly |
| k_M2 | 0.3 / age_factor | Reduced M2 counter-activation in elderly |
| k_M2,deg | 0.15 × age_factor | Faster M2 degradation in elderly |
| k_NI | 0.6 × priming | Cytokine production elevated by microglial priming |
| k_NI,deg | 0.2 / age_factor | Slower cytokine clearance in elderly |
| k_NI,M2 | 0.3 / age_factor | M2 anti-inflammatory suppression reduced with age |
| k_NP | 0.25 / age_factor | BDNF/VEGF production reduced in elderly |
| k_NP,deg | 0.1 | Fixed degradation |
| k_NP,base | 0.02 / age_factor | Basal neuroprotective tone (age-dependent) |

where:
- severity = clip((15 − GCS)/12, 0.1, 1.0)
- apache_factor = clip(APACHE II / 30, 0.3, 1.5)
- age_factor = 1 + 0.015 × max(age − 40, 0)
- priming = 1 + 0.3 × prior_psych_history + 0.2 × alcohol_misuse

### 3.5 Simulation Parameters

| Parameter | Value |
|---|---|
| Time span | 0–720 hours (30 days) |
| Time evaluation points | 800 equally spaced |
| ODE solver | RK45 |
| Relative tolerance | 1 × 10⁻⁵ |
| Absolute tolerance | 1 × 10⁻⁷ |
| Maximum step size | 2.0 hours |

### 3.6 Neuroinflammation Features Extracted (8 features)

| Feature name | Description | Units |
|---|---|---|
| mech_m1_peak | Maximum M1 activation over 30 days | arb. units |
| mech_ni_peak | Maximum neuroinflammation index over 30 days | arb. units |
| mech_ni_auc_7d | AUC of NI over first 7 days | arb. units·h |
| mech_m1_m2_ratio_72h | Mean M1/M2 ratio in first 72 hours (pro-inflammatory balance) | ratio |
| mech_ni_resolution_time | Time for NI to fall below 10% of peak (inflammatory resolution) | h |
| mech_np_steady_state | NP at 30-day steady state (chronic neuroprotective capacity) | arb. units |
| mech_np_auc | AUC of NP over 30 days (total neuroprotective exposure) | arb. units·h |
| mech_m2_m1_dominance | Mean M2/M1 ratio in final 100 time-points (reparative balance) | ratio |

---

## 4. Composite Mechanistic Indices (2 features)

Two composite indices are computed post-simulation to provide clinically interpretable summaries:

**Secondary Injury Index** (higher = greater secondary injury burden):

$$\text{SII} = \frac{t_{\text{ICP}>20}}{168} + \frac{t_{\text{CPP}<60}}{168} + \frac{\text{NI}_{\text{AUC,7d}}}{500}$$

where time components are normalised by the 7-day simulation window (168 hours) and NI_AUC is normalised to the maximum observed value (500 arb. units·h).

**Recovery Potential Index** (higher = greater endogenous recovery capacity):

$$\text{RPI} = \text{NP}_{\text{steady}} \times \text{M2/M1}_{\text{dominance}} \times (1 - \text{AR\_index})$$

This index captures three complementary recovery mechanisms: sustained neuroprotective signalling (NP_steady), reparative microglial balance (M2/M1 dominance), and preserved cerebrovascular autoregulation (1 − AR_index).

---

## 5. Summary of All 21 Mechanistic Features

| # | Feature name | ODE system | Biological meaning |
|---|---|---|---|
| 1 | mech_icp_peak | ICP/CPP | Peak ICP burden |
| 2 | mech_icp_mean_72h | ICP/CPP | Early-phase ICP control |
| 3 | mech_icp_auc_7d | ICP/CPP | Cumulative ICP exposure |
| 4 | mech_icp_time_above_20 | ICP/CPP | Clinically actionable ICP threshold breach |
| 5 | mech_icp_time_above_25 | ICP/CPP | Severe ICP threshold breach |
| 6 | mech_icp_at_day7 | ICP/CPP | ICP resolution at 1 week |
| 7 | mech_cpp_min | ICP/CPP | Nadir perfusion pressure |
| 8 | mech_cpp_mean | ICP/CPP | Mean perfusion adequacy |
| 9 | mech_cpp_time_below_60 | ICP/CPP | Cumulative ischaemia risk |
| 10 | mech_ar_index | ICP/CPP | Autoregulatory integrity |
| 11 | mech_cpp_optimal_time | ICP/CPP | Time in therapeutic CPP range |
| 12 | mech_m1_peak | Neuroinflammation | Peak pro-inflammatory response |
| 13 | mech_ni_peak | Neuroinflammation | Peak cytokine burden |
| 14 | mech_ni_auc_7d | Neuroinflammation | Acute inflammatory exposure |
| 15 | mech_m1_m2_ratio_72h | Neuroinflammation | Early pro/anti-inflammatory balance |
| 16 | mech_ni_resolution_time | Neuroinflammation | Speed of inflammation resolution |
| 17 | mech_np_steady_state | Neuroinflammation | Chronic neuroprotective capacity |
| 18 | mech_np_auc | Neuroinflammation | Total neuroprotective exposure |
| 19 | mech_m2_m1_dominance | Neuroinflammation | Late reparative microglial balance |
| 20 | mech_secondary_injury_index | Composite | Multi-domain secondary injury burden |
| 21 | mech_recovery_potential | Composite | Multi-domain recovery capacity |

---

## 6. Numerical Integration

All ODEs were solved using the Runge-Kutta 4th/5th order (RK45) adaptive-step solver implemented in `scipy.integrate.solve_ivp` (SciPy ≥1.7.0). RK45 provides error-controlled adaptive step selection with local truncation error bounded by the specified relative and absolute tolerances. For patients where the solver failed to converge (< 0.5% of cases, all with extreme parameter values), fallback constant-value trajectories were substituted (ICP/CPP: admission values; neuroinflammation: unit-amplitude constants). These fallback cases are flagged transparently in the simulation output.

---

## 7. Biological References for Model Structure

| ODE component | Primary references |
|---|---|
| Monroe-Kellie doctrine | Mokri B. *Mayo Clin Proc*. 2001;76(2):190–202 |
| Davson’s equation (CSF) | Davson H, Welch K, Segal MB. *Physiology and Pathophysiology of Cerebrospinal Fluid*. 1987 |
| Lassen autoregulation curve | Lassen NA. *Physiol Rev*. 1959;39(2):183–238 |
| Marmarou compliance model | Marmarou A et al. *J Neurosurg*. 1978;48(4):523–534 |
| DAMP-microglial activation | Cheng P et al. *J Neuroinflammation*. 2018;15(1):275 |
| M1/M2 microglial dynamics | Hou Y et al. *Cell Mol Immunol*. 2020;17(11):1099–1111 |
| Inflammaging (age-dependent resolution) | Franceschi C et al. *Nat Rev Endocrinol*. 2018;14(10):576–590 |
| ICP management thresholds (>20, >25 mmHg) | Carney N et al. (Brain Trauma Foundation Guidelines). *Neurosurgery*. 2017;80(1):6–15 |
| CPP therapeutic range (60–100 mmHg) | Carney N et al. *Neurosurgery*. 2017;80(1):6–15 |

---

*This supplementary note provides a complete mathematical specification sufficient for independent replication of the QSP pipeline. The Python implementation is available in `brain_injury_qsp_hybrid.py` in the associated code repository.*
