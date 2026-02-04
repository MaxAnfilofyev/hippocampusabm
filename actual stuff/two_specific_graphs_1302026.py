import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# --- IMPORT HANDLING ---
# We try multiple filenames just in case
try:
    from model import ADModel
except ImportError:
    try:
        from try_again import ADModel
    except ImportError:
        print("❌ CRITICAL ERROR: Could not import ADModel. Ensure 'model.py' is in this folder.")
        exit()

# --- CONFIGURATION ---
MAX_YEARS = 20            
STEPS_PER_YEAR = 50       
TOTAL_STEPS = MAX_YEARS * STEPS_PER_YEAR
TARGET_PARAMS = [100.0, 0.2587, 27.72] # [L, k, t0]

# Optimized Parameters from your code
OPTIMAL_PARAMS = {
    'amyloid_toxicity': 25.0,       
    'cytokine_toxicity': 35.0,      
    'neuron_repair_rate': 0.0035,   
    'neuron_death_threshold': 0.15 
}

# --- HELPER FUNCTIONS ---

def sigmoid(t, L, k, t0):
    """The clinical target curve."""
    return L / (1 + np.exp(-k * (t - t0)))

def run_batch(n_runs, lec_dose=0.0, cro_dose=0.0, genotype="APOE3 (Control)"):
    """
    Runs N simulations and returns a matrix of atrophy history.
    Shape: (n_runs, total_steps)
    """
    all_runs = []
    
    print(f"   Running batch: {genotype} | Lec: {lec_dose} | Cro: {cro_dose}...")
    
    for i in range(n_runs):
        # Initialize model
        model = ADModel(
            width=40, height=40,
            n_neurons=1000, n_microglia=200, 
            genotype=genotype,
            genetic_overrides=None,
            
            # Inject Parameters
            amyloid_toxicity=OPTIMAL_PARAMS['amyloid_toxicity'],
            cytokine_toxicity=OPTIMAL_PARAMS['cytokine_toxicity'],
            neuron_repair_rate=OPTIMAL_PARAMS['neuron_repair_rate'],
            neuron_death_threshold=OPTIMAL_PARAMS['neuron_death_threshold'],
            
            # Doses
            lecanemab_dose=lec_dose,
            cromolyn_dose=cro_dose,
            
            max_steps=TOTAL_STEPS,
            seed=random.randint(1, 9999999) # Random seed for variability
        )
        
        # Run step-by-step
        history = []
        for _ in range(TOTAL_STEPS):
            model.step()
            history.append(model.get_atrophy_percent())
        
        all_runs.append(history)
        
    return np.array(all_runs)

def plot_with_confidence(ax, data_matrix, label, color):
    """
    Plots the mean line and a shaded region for Standard Deviation.
    """
    # Calculate Mean and Std Dev across the N runs (axis 0)
    mean_curve = np.mean(data_matrix, axis=0)
    std_curve = np.std(data_matrix, axis=0)
    
    time_axis = np.linspace(0, MAX_YEARS, len(mean_curve))
    
    # Plot Mean Line
    ax.plot(time_axis, mean_curve, color=color, linewidth=2.5, label=label)
    
    # Plot Shaded Error Band (Mean ± 1 Std Dev)
    ax.fill_between(time_axis, 
                    mean_curve - std_curve, 
                    mean_curve + std_curve, 
                    color=color, alpha=0.2) # alpha is transparency

# --- MAIN GRAPH GENERATION ---

def generate_graphs():
    # 1. PREPARE TARGET CURVE
    t_vals = np.linspace(0, MAX_YEARS, TOTAL_STEPS)
    target_curve = sigmoid(t_vals, *TARGET_PARAMS)

    # ==========================================
    # GRAPH 1: Lecanemab Monotherapy vs Target
    # ==========================================
    print("--- Generating Graph 1: Lecanemab Monotherapy ---")
    
    # Run 10 sims with High Dose Lecanemab on APOE3
    lec_data = run_batch(n_runs=10, lec_dose=10.0, genotype="APOE3 (Control)")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Target
    ax.plot(np.linspace(0, MAX_YEARS, len(target_curve)), target_curve, 
            color='black', linestyle='--', linewidth=2, label="Clinical Target (Severe AD)")
    
    # Plot Simulation
    plot_with_confidence(ax, lec_data, "Lecanemab Monotherapy (10 mg/kg)", "red")
    
    ax.set_title("Lecanemab Monotherapy on APOE3 Baseline (n=10)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Simulation Years", fontsize=12)
    ax.set_ylabel("Hippocampal Atrophy (%)", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100) # Standard scale
    
    plt.savefig("graph_1_lecanemab_toxicity.png", dpi=300)
    print("✅ Saved 'graph_1_lecanemab_toxicity.png'")
    plt.close()

    # ==========================================
    # GRAPH 2: Calibration Zoom (Clean Visual)
    # ==========================================
    print("--- Generating Graph 2: Optimized Calibration ---")
    
    # Run 10 sims with No Drug (Baseline) using Optimized Params
    cal_data = run_batch(n_runs=10, lec_dose=0.0, genotype="APOE3 (Control)")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Target
    ax.plot(np.linspace(0, MAX_YEARS, len(target_curve)), target_curve, 
            color='black', linestyle='--', linewidth=2, label="Clinical Target")
    
    # Plot Simulation
    plot_with_confidence(ax, cal_data, "Optimized Model (Mean of 10 Runs)", "blue")
    
    ax.set_title("Model Calibration: APOE3 Baseline vs Target", fontsize=14, fontweight='bold')
    ax.set_xlabel("Simulation Years", fontsize=12)
    ax.set_ylabel("Hippocampal Atrophy (%)", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # --- Y-AXIS LIMIT SET TO 30 AS REQUESTED ---
    ax.set_ylim(0, 30) 
    
    plt.savefig("graph_2_calibration_zoom.png", dpi=300)
    print("✅ Saved 'graph_2_calibration_zoom.png'")
    plt.close()

if __name__ == "__main__":
    generate_graphs()