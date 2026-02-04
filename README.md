# Hippocampus ABM: Mechanistic Agent-Based Modeling of Alzheimer's Disease Treatment

This project implements a multi-scale Agent-Based Model (ABM) to study the progression of Alzheimer's Disease (AD) in the hippocampus. It focuses on the interplay between amyloid-beta accumulation, neuroinflammation (microglial activation), and neuronal survival across different genetic backgrounds and pharmacological interventions.

## Research Objective
The primary goal is to determine the optimal combination and dosage of **Lecanemab** (anti-amyloid) and **Cromolyn** (anti-inflammatory) to maximize neuronal survival while minimizing side effects (e.g., ARIA-like inflammation) in patients with different genetic profiles:
- **APOE3 (Control)**: Baseline amyloid clearance and microglial function.
- **APOE 3/4 & 4/4**: Characterized by impaired amyloid clearance and increased inflammatory reactivity.
- **TREM2 R47H**: Mutation leading to significantly reduced microglial phagocytosis and impaired plaque clustering.

---

## Key Features
- **Mesa-based Simulation**: Built on the [Mesa 3.x](https://mesa.readthedocs.io/en/stable/) framework for high-performance agent-based modeling.
- **Spatial Hippocampal Architecture**: A multi-layer graph modeling the **DG -> CA3 -> CA1** circuit, with spatial connection probabilities.
- **Microglial Dynamics**: Simulates transitions between Resting (**M0**), Pro-inflammatory (**M1**), and Anti-inflammatory (**M2**) states, including "M1 Burnout" into senescence (**M3**).
- **Genetic Modeling**: Variable parameters for amyloid production, clearance, and microglial efficiency based on real-world clinical data.
- **Interactive Dashboard**: Powered by **Solara**, allowing real-time intervention testing (sliding drug doses and changing genotypes).
- **Data-Driven Calibration**: Model parameters are calibrated against clinical data from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)**.

---

## Installation

### Prerequisites
- Python 3.10+
- `pip` or `conda`

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd hippocampusabm
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Interactive Dashboard (Recommended)
Launch the Solara-based visual simulation tool to interact with the model directly:
```bash
solara run "actual stuff/try_again.py"
```
*Note: This script provides the most advanced version of the hippocampal model with spatial wiring and genetic scaling.*

### CLI Simulation
Run a headless simulation to generate `results.csv`:
```bash
python model.py
```

### Calibration & Analysis
Advanced scripts for model tuning and batch runs are located in the `actual stuff/` directory:
- `calibration.py`: Calibrate ABM trajectories against ADNI reference data.
- `batch_analysis.py`: Run multiple iterations across dose combinations.
- `visualize_final_heatmaps.py`: Generate synergistic dosage heatmaps.

---

## Project Structure
- `model.py`: Core Mesa model definition and basic agent logic.
- `actual stuff/`: Main working directory containing optimized scripts.
    - `try_again.py`: Primary interactive entry point.
    - `data.py`: Data handling and processing utilities.
- `context/`: Scientific background, including the **Literature Review and Research Plan**.
- `raw_data/`: Clinical datasets and ADNI reference curves.
- `requirements.txt`: Project dependencies.

---

## Scientific Background
This model implements findings from several key AD research papers:
1. **Weathered et al. (2023)**: Microglial spatiotemporal response to Beta-Amyloid.
2. **Duchesne et al. (2024)**: APOE-dependent complex multifactorial models of AD.
3. **Van Olst et al. (2025)**: Microglial mechanisms of amyloid clearance and ARIA.

For a full list of primary sources and the detailed research plan, see the documents in [`/context`](file:///Users/maxa/hippocampusabm/context/).

## License
MIT License (or as specified by the user).
