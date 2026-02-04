import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import itertools
import pandas as pd

# ------------------------------
# PARAMETERS
# ------------------------------

# Baseline
P = 0.01             # amyloid production
C_env_base = 0.005   # environmental clearance
k_phago_base = 0.02
alpha_base = 0.01    # cytokine production per M1 & amyloid
beta = 0.05          # cytokine decay
p0_to_M1_base = 0.02
p0_to_M2_base = 0.01
gamma1 = 0.01
gamma2 = 0.01
k_neuron = 0.05

# Genotypes
genotype_modifiers = {
    "APOE3": {"amyloid_clearance":1.0, "phag_eff":1.0, "inflam":1.0},
    "APOE4": {"amyloid_clearance":0.6, "phag_eff":1.0, "inflam":1.3},
    "TREM2":{"amyloid_clearance":1.0, "phag_eff":0.35, "inflam":1.1},
    "APOE4+TREM2":{"amyloid_clearance":0.6, "phag_eff":0.35, "inflam":1.5}
}

# Drug effects
def lecanemab_clearance(dose):
    return 1 + 0.8 * dose/10.0
def lecanemab_inflam(dose):
    return 0.075 * dose/10.0
def cromolyn_reduction(dose):
    return 0.5 * dose/40.0
def cromolyn_M1_reduction(dose):
    return 0.5 * dose/40.0

# ------------------------------
# ODE SYSTEM
# ------------------------------
def ad_model(t, y, genotype, lecan_dose, crom_dose):
    A, N, M0, M1, M2, C = y
    
    # Genotype modifiers
    gm = genotype_modifiers[genotype]
    C_env = C_env_base * gm["amyloid_clearance"] * lecanemab_clearance(lecan_dose)
    phag_eff = k_phago_base * gm["phag_eff"] * lecanemab_clearance(lecan_dose)
    alpha = alpha_base * gm["inflam"] * (1 - cromolyn_reduction(crom_dose))
    
    # Microglia transitions
    amy_factor = A
    p0_to_M1 = p0_to_M1_base * (1 + 3*amy_factor) * (1 - cromolyn_M1_reduction(crom_dose)) * (1 + lecanemab_inflam(lecan_dose))
    p0_to_M2 = p0_to_M2_base * (1 + 1.5*amy_factor)
    
    # ODEs
    dA = P - C_env*A - phag_eff*M2*A
    dC = alpha*M1*A - beta*C
    dM0 = -p0_to_M1*M0 - p0_to_M2*M0 + gamma1*M1 + gamma2*M2
    dM1 = p0_to_M1*M0 - gamma1*M1
    dM2 = p0_to_M2*M0 - gamma2*M2
    dN = -k_neuron * N * (A + C)
    
    return [dA, dN, dM0, dM1, dM2, dC]

# ------------------------------
# SIMULATION
# ------------------------------
def run_sim(genotype, lecan, crom, t_end=200):
    y0 = [0.1, 1.0, 0.7, 0.2, 0.1, 0.0] # initial conditions: A, N, M0, M1, M2, C
    sol = solve_ivp(ad_model, [0, t_end], y0, args=(genotype, lecan, crom),
                    t_eval=np.linspace(0, t_end, t_end+1))
    return sol

# ------------------------------
# VISUALIZATION
# ------------------------------
def plot_timecourse(sol, genotype, lecan, crom):
    plt.figure(figsize=(10,6))
    plt.plot(sol.t, sol.y[0], label="Amyloid")
    plt.plot(sol.t, sol.y[1], label="Neuron fraction")
    plt.plot(sol.t, sol.y[5], label="Cytokine")
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
            survival[i,j] = sol.y[1,-1]  # neuron fraction at last time point
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
# RUN EXAMPLE
# ------------------------------
if __name__=="__main__":
    genotype = "APOE4+TREM2"
    lecan_doses = [0, 2.5, 5, 7.5, 10]
    crom_doses = [0, 10, 20, 30, 40]
    
    # Single simulation timecourse
    sol = run_sim(genotype, lecan=5, crom=20)
    plot_timecourse(sol, genotype, 5, 20)
    
    # Heatmap
    synergy_heatmap(genotype, lecan_doses, crom_doses)
