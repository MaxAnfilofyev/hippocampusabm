import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -------------------------------
# PARAMETERS
# -------------------------------

# Baseline rates
P = 0.01        # amyloid production
C_env = 0.005   # environmental clearance
k_phago = 0.02  # microglial clearance efficiency
alpha = 0.05    # cytokine production by M1 microglia
beta = 0.02     # cytokine decay
p0_to_1 = 0.03  # M0->M1 transition rate
p0_to_2 = 0.01  # M0->M2 transition rate
gamma1 = 0.01   # M1->M0 reversion
gamma2 = 0.01   # M2->M0 reversion
k_neuron = 0.1  # neuron death sensitivity

# Genotype modifiers
GENOTYPES = {
    "APOE3":       {"amyloid_clearance":1.0, "cytokine":1.0, "phago":1.0},
    "APOE4":       {"amyloid_clearance":0.6, "cytokine":1.3, "phago":1.0},
    "TREM2":       {"amyloid_clearance":1.0, "cytokine":1.0, "phago":0.35},
    "APOE4_TREM2": {"amyloid_clearance":0.6, "cytokine":1.3, "phago":0.35}
}

# Drug effects
def lecanemab_effect(dose_mgkg):
    EC50 = 5.0; n = 2; max_clearance = 0.8
    clearance = max_clearance * (dose_mgkg**n / (EC50**n + dose_mgkg**n))
    tox = 0.05 * (dose_mgkg / 10.0)**2
    return clearance, tox

def cromolyn_effect(dose_mgkg):
    EC50 = 20.0; n = 2; max_cytokine_red = 0.5
    cytokine_red = max_cytokine_red * (dose_mgkg**n / (EC50**n + dose_mgkg**n))
    tox = 0.02 * (dose_mgkg / 40.0)**2
    return cytokine_red, tox

# -------------------------------
# ODE SYSTEM
# -------------------------------
def ad_model(t, y, genotype, lecanemab_dose, cromolyn_dose):
    A, C, M1, M2, N = y
    # genotype scaling
    g = GENOTYPES[genotype]
    
    # drug effects
    l_clear, l_tox = lecanemab_effect(lecanemab_dose)
    c_red, c_tox = cromolyn_effect(cromolyn_dose)
    
    # Amyloid dynamics
    dA_dt = P - C_env * g["amyloid_clearance"] * (1 + l_clear) * A \
            - k_phago * g["phago"] * (M1+M2) * A
    
    # Cytokine dynamics
    dC_dt = alpha * M1 * g["cytokine"] * (1 - c_red) - beta * C
    
    # Microglia dynamics
    dM1_dt = p0_to_1 * A * (1 - M1 - M2) - gamma1 * M1
    dM2_dt = p0_to_2 * A * (1 - M1 - M2) - gamma2 * M2
    
    # Neuron survival
    exposure = A + C
    death_rate = k_neuron * exposure + l_tox + c_tox
    dN_dt = - death_rate * N
    
    return [dA_dt, dC_dt, dM1_dt, dM2_dt, dN_dt]

# -------------------------------
# SIMULATION
# -------------------------------
def run_sim(genotype="APOE4", lecanemab_dose=5, cromolyn_dose=20, t_max=200):
    y0 = [1.0, 0.0, 0.05, 0.05, 1.0]  # initial [A, C, M1, M2, N]
    sol = solve_ivp(ad_model, [0, t_max], y0, 
                    args=(genotype, lecanemab_dose, cromolyn_dose),
                    dense_output=True)
    return sol

# -------------------------------
# HEATMAP
# -------------------------------
lecan_doses = np.linspace(0, 10, 11)
crom_doses = np.linspace(0, 40, 9)
genotype = "APOE4"

heatmap = np.zeros((len(crom_doses), len(lecan_doses)))

for i, c in enumerate(crom_doses):
    for j, l in enumerate(lecan_doses):
        sol = run_sim(genotype, l, c, t_max=200)
        N_final = sol.y[4,-1]  # neuron fraction at final time
        heatmap[i,j] = N_final

plt.figure(figsize=(8,5))
plt.imshow(heatmap, origin='lower', extent=[0,10,0,40],
           aspect='auto', cmap='viridis')
plt.colorbar(label='Neuron survival fraction')
plt.xlabel('Lecanemab dose (mg/kg)')
plt.ylabel('Cromolyn dose (mg/kg)')
plt.title(f'Neuron survival heatmap ({genotype})')
plt.show()

# -------------------------------
# EXAMPLE: time courses
# -------------------------------
sol = run_sim(genotype="APOE4", lecanemab_dose=5, cromolyn_dose=20, t_max=200)
t = sol.t
A, C, M1, M2, N = sol.y

plt.figure(figsize=(10,5))
plt.plot(t, A, label='Amyloid (A)')
plt.plot(t, C, label='Cytokines (C)')
plt.plot(t, M1, label='M1 Microglia')
plt.plot(t, M2, label='M2 Microglia')
plt.plot(t, N, label='Neurons (N)')
plt.xlabel('Time')
plt.ylabel('Normalized concentration / fraction')
plt.title('AD Dynamics over Time')
plt.legend()
plt.show()
