import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ------------------------------
# SYSTEM PARAMETERS (Literature-Informed)
# ------------------------------

# Amyloid (A)
P = 0.01  # production rate (estimate)
C_env_base = 0.005  # baseline clearance (literature-informed concept)
k_phago_base = 0.02  # microglial clearance efficiency (estimate, supported by Kim 2024)

# Cytokines (C)
alpha_base = 0.01  # inflammation production per M1 & Aβ (Combs 2001, Kim 2024)
beta = 0.05  # natural decay of cytokines (estimate)

# Microglia transitions
p0_to_M1_base = 0.02  # transition resting -> M1 (dependent on Aβ, Chen 2025)
p0_to_M2_base = 0.01  # transition resting -> M2 (dependent on Aβ, Chen 2025)
gamma1 = 0.01  # decay M1 -> resting
gamma2 = 0.01  # decay M2 -> resting

# Neurons
k_neuron = 0.05  # neuron death rate due to amyloid + cytokines (Combs 2001)

# ------------------------------
# GENOTYPE MODIFIERS (APOE4/TREM2 Literature-Informed)
# ------------------------------
genotype_modifiers = {
    "APOE3": {"amyloid_clearance": 1.0, "phag_eff": 1.0, "inflam": 1.0},
    "APOE4": {"amyloid_clearance": 0.6, "phag_eff": 1.0, "inflam": 1.3},  # 40% reduced clearance, 30% increased inflammation
    "TREM2": {"amyloid_clearance": 1.0, "phag_eff": 0.35, "inflam": 1.1},  # 65% reduced phagocytosis
    "APOE4+TREM2": {"amyloid_clearance": 0.6, "phag_eff": 0.35, "inflam": 1.5},  # multiplicative assumption
}

# ------------------------------
# DRUG EFFECT FUNCTIONS (Based on Literature)
# ------------------------------
def lecanemab_clearance(dose):
    """Increases Aβ clearance (Swanson 2020, Tucker 2014). Dose in mg/kg."""
    return 1 + 0.8 * dose / 10.0

def lecanemab_inflam(dose):
    """Small pro-inflammatory effect (ARIA-E risk, Swanson 2020)."""
    return 0.075 * dose / 10.0

def cromolyn_reduction(dose):
    """Reduces cytokine production (Wang 2020, Granucci 2019)."""
    return 0.5 * dose / 40.0

def cromolyn_M1_reduction(dose):
    """Reduces microglia transition to M1 (Zhang 2016)."""
    return 0.5 * dose / 40.0

# ------------------------------
# ODE SYSTEM
# ------------------------------
def ad_model(t, y, genotype, lecan_dose, crom_dose):
    A, N, M0, M1, M2, C = y
    
    # Apply genotype modifiers
    gm = genotype_modifiers[genotype]
    C_env = C_env_base * gm["amyloid_clearance"] * lecanemab_clearance(lecan_dose)
    phag_eff = k_phago_base * gm["phag_eff"] * lecanemab_clearance(lecan_dose)
    alpha = alpha_base * gm["inflam"] * (1 - cromolyn_reduction(crom_dose))
    
    # Microglia activation rates
    amy_factor = A  # Aβ-dependent increase
    p0_to_M1 = p0_to_M1_base * (1 + 3 * amy_factor) * (1 - cromolyn_M1_reduction(crom_dose)) * (1 + lecanemab_inflam(lecan_dose))
    p0_to_M2 = p0_to_M2_base * (1 + 1.5 * amy_factor)
    
    # ODEs
    dA = P - C_env * A - phag_eff * M2 * A
    dC = alpha * M1 * A - beta * C
    dM0 = -p0_to_M1 * M0 - p0_to_M2 * M0 + gamma1 * M1 + gamma2 * M2
    dM1 = p0_to_M1 * M0 - gamma1 * M1
    dM2 = p0_to_M2 * M0 - gamma2 * M2
    dN = -k_neuron * N * (A + C)
    
    return [dA, dN, dM0, dM1, dM2, dC]

# ------------------------------
# SIMULATION FUNCTION
# ------------------------------
def run_sim(genotype, lecan, crom, t_end=200):
    """Run ODE solver for specified genotype and drug doses."""
    y0 = [0.1, 1.0, 0.7, 0.2, 0.1, 0.0]  # initial conditions: A, N, M0, M1, M2, C
    sol = solve_ivp(ad_model, [0, t_end], y0, args=(genotype, lecan, crom),
                    t_eval=np.linspace(0, t_end, t_end+1))
    return sol

# ------------------------------
# PLOTTING FUNCTIONS
# ------------------------------
def plot_timecourse(sol, genotype, lecan, crom):
    plt.figure(figsize=(10,6))
    plt.plot(sol.t, sol.y[0], label="Amyloid (Aβ)")
    plt.plot(sol.t, sol.y[1], label="Neuron fraction")
    plt.plot(sol.t, sol.y[5], label="Cytokine (C)")
    plt.plot(sol.t, sol.y[3], label="M1 microglia")
    plt.plot(sol.t, sol.y[4], label="M2 microglia")
    plt.title(f"{genotype}, Lecan={lecan}mg/kg, Crom={crom}mg/kg")
    plt.xlabel("Time")
    plt.ylabel("Level / Fraction")
    plt.legend()
    plt.show()

# ------------------------------
# DRUG SYNERGY HEATMAP
# ------------------------------
def synergy_heatmap(genotype, lecan_doses, crom_doses):
    survival = np.zeros((len(lecan_doses), len(crom_doses)))
    for i, l in enumerate(lecan_doses):
        for j, c in enumerate(crom_doses):
            sol = run_sim(genotype, l, c)
            survival[i, j] = sol.y[1, -1]  # neuron fraction at final time
    plt.figure(figsize=(6,5))
    plt.imshow(survival, origin='lower', aspect='auto',
               extent=[min(crom_doses), max(crom_doses), min(lecan_doses), max(lecan_doses)],
               cmap="viridis")
    plt.colorbar(label="Neuron fraction at t_end")
    plt.xlabel("Cromolyn (mg/kg)")
    plt.ylabel("Lecanemab (mg/kg)")
    plt.title(f"Neuron survival heatmap ({genotype})")
    plt.show()

# ------------------------------
# RUN EXAMPLES
# ------------------------------
if __name__ == "__main__":
    genotype = "APOE4+TREM2"
    lecan_doses = [0, 2.5, 5, 7.5, 10]
    crom_doses = [0, 10, 20, 30, 40]
    
    # Single time-course simulation
    sol = run_sim(genotype, lecan=5, crom=20)
    plot_timecourse(sol, genotype, 5, 20)
    
    # Heatmap for synergy visualization
    synergy_heatmap(genotype, lecan_doses, crom_doses)