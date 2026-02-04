# Optimizing Dual-Drug Therapy in Alzheimer's Disease: A Computational Analysis of Genetic Variants

**Student:** Polina Anfilofyev  
**Project Category:** Computational Biology and Bioinformatics

---

## Abstract
This project investigated the synergistic effects of Lecanemab (an anti-amyloid antibody) and Cromolyn (an anti-inflammatory agent) on neuronal survival in Alzheimer's Disease (AD) across distinct genetic backgrounds. Using an agent-based model (ABM) calibrated to clinical reference data, we simulated 25 dose combinations over a 10-year disease progression. Our results indicate that while high-risk genotypes like **APOE4** exhibit resistance to single-drug therapy due to neuroinflammatory "side effects" (ARIA-like markers), the addition of Cromolyn significantly "rescues" neuronal health. This suggests that personalized, dual-drug regimens are essential for maximizing efficacy in genetically vulnerable populations.

## Research Question
What is the optimal combination and dosage of Lecanemab and Cromolyn across different genetic backgrounds (APOE3, APOE4, and TREM2 R47H) to maximize neuronal survival and minimize neuroinflammation?

## Hypothesis
It is hypothesized that Lecanemab’s efficacy will be limited by its downstream inflammatory effects in APOE4 carriers. Therefore, a synergistic effect is predicted where higher doses of Cromolyn will be required in APOE4 backgrounds to mitigate inflammation and unlock the neuroprotective potential of amyloid clearance.

---

## Methodology

### Computational Simulation
The study utilized a hippocampal-scaled agent-based model (ABM) built in the **Mesa** framework. The model simulates:
- **Neurons:** Individual agents with dynamic "Health" and "Cognitive Reserve," vulnerable to local amyloid toxicity and cytokine levels.
- **Microglia:** Agents transitioning between Resting (M0), Pro-inflammatory (M1), and Senescent (M3) states.
- **Environment:** Spatially distributed amyloid-beta and pro-inflammatory cytokines.

### Technical Rigor: Deterministic Seeding
To ensure high-precision data, a **Deterministic Variable Seeding** strategy was employed. This held the "brain wiring" and "initial plaque distribution" constant for each replicate across all dosage groups, isolating the drug effect as the sole variable and eliminating the 11% statistical noise observed in earlier iterations.

---

## Results and Analysis

### 1. Control Population (APOE 3/3)
In the neutral genetic background, Lecanemab provides a steady benefit. Atrophy is maintained at low levels even with moderate dosing, as the microglial immune system remains balanced.

![APOE 3/3 Heatmap](actual%20stuff/heatmap_apoe3.png)

### 2. High-Risk Population (APOE 4/4)
Results align with **CLARITY-AD trial** findings where APOE4 carriers showed reduced benefit from single-drug therapy. The heatmap reveals a "Success Corner" (Deep Green) ONLY when high doses of Cromolyn are paired with Lecanemab, proving the necessity of dual-drug intervention for this subgroup.

![APOE 4/4 Heatmap](actual%20stuff/heatmap_apoe44.png)

### 3. Loss-of-Function Variant (TREM2 R47H)
The TREM2 variant shows a distinct response profile. Because these microglia are "blind" to plaque, the drug potency must be higher to trigger effective clearance, accompanied by a higher risk of inflammation when clearance finally occurs.

![TREM2 R47H Heatmap](actual%20stuff/heatmap_trem2.png)

---

## Conclusion
The simulation demonstrates that "one-size-fits-all" dosing is suboptimal for Alzheimer's treatment. Specifically:
1. **Synergy is Essential:** For APOE4 homozygotes, Lecanemab alone may trigger inflammatory responses that negate its benefits. Cromolyn acts as a critical stabilizer.
2. **Genetic Stratification:** Patients with TREM2 mutations require specifically tuned anti-inflammatory baselines before commencing antibody therapy.

This work provides a computational framework for "Virtual Clinical Trials," allowing researchers to predict optimal personalized dosing strategies years before they are tested in human subjects.
