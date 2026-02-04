import numpy as np
import matplotlib.pyplot as plt
import random
import os

# --- IMPORT YOUR MODEL ---
try:
    from try_again import ADModel
except ImportError:
    print("❌ CRITICAL ERROR: Could not import 'ADModel' from 'try_again.py'.")
    exit()

# --- 1. CONFIGURATION ---
REPLICATES = 5            # Runs per group
MAX_YEARS = 20            
STEPS_PER_YEAR = 50       
TOTAL_STEPS = MAX_YEARS * STEPS_PER_YEAR

# Visualization Settings
OUTPUT_DIR = "Lecanemab_Experiment"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 2. SIMULATION RUNNER ---

def run_simulation(dose, seed):
    """
    Runs a single simulation instance with a specific Lecanemab dose.
    """
    model = ADModel(
        width=40, height=40,
        n_neurons=1000, n_microglia=200,
        genotype="APOE3 (Control)",  # Both groups are APOE3
        lecanemab_dose=dose,         # Variable Dose
        cromolyn_dose=0.0,
        max_steps=TOTAL_STEPS,
        seed=seed
    )
    
    history = []
    for _ in range(TOTAL_STEPS):
        model.step()
        history.append(model.get_atrophy_percent())
    return history

def run_group(dose_label, dose_value):
    """
    Runs the full batch of replicates for a specific group.
    Returns: (mean_curve, std_curve, time_axis)
    """
    print(f"🧪 Starting Group: {dose_label} ({dose_value} mg/kg)...")
    results = []
    base_seed = random.randint(10000, 99999)

    for i in range(REPLICATES):
        # Use distinct seeds for variance
        current_seed = int(f"{base_seed}{i}")
        print(f"   -> Run {i+1}/{REPLICATES} (Seed {current_seed})", end="\r")
        
        data = run_simulation(dose_value, current_seed)
        results.append(data)
    
    print(f"   ✅ Group {dose_label} Complete.              ")

    # Calculate Statistics
    matrix = np.array(results)
    mean_curve = np.mean(matrix, axis=0)
    std_curve = np.std(matrix, axis=0)
    time_axis = np.linspace(0, MAX_YEARS, TOTAL_STEPS)
    
    return mean_curve, std_curve, time_axis

# --- 3. MAIN EXECUTION ---

if __name__ == "__main__":
    print(f"🚀 STARTING LECANEMAB EFFICACY TEST")
    print("-" * 60)

    # 1. Run Control Group (0 mg/kg)
    ctrl_mean, ctrl_std, time = run_group("APOE3 Control", 0.0)

    # 2. Run Treatment Group (10 mg/kg)
    drug_mean, drug_std, _ = run_group("APOE3 + Lecanemab", 10.0)

    # --- 4. VISUALIZATION ---
    print("\n📊 Generating Comparison Plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Plot Control (Grey/Black)
    ax.plot(time, ctrl_mean, color="#444444", linewidth=2.5, label="APOE3 Control (0 mg/kg)")
    ax.fill_between(time, ctrl_mean - ctrl_std, ctrl_mean + ctrl_std, color="#444444", alpha=0.15)

    # Plot Treatment (Blue/Green)
    ax.plot(time, drug_mean, color="#2ca02c", linewidth=2.5, label="Lecanemab (10 mg/kg)")
    ax.fill_between(time, drug_mean - drug_std, drug_mean + drug_std, color="#2ca02c", alpha=0.15)

    # Formatting
    ax.set_title("Efficacy Analysis: Lecanemab vs. Control (APOE3)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Years", fontsize=12)
    ax.set_ylabel("Neurodegeneration (%)", fontsize=12)
    ax.set_xlim(0, MAX_YEARS)
    ax.set_ylim(0, 30) # Adjusted y-limit to see separation clearly
    
    # 2-Year Ticks
    ax.set_xticks(np.arange(0, MAX_YEARS + 1, 2))

    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=11, loc="upper left", frameon=True, framealpha=0.9)

    # Final Calculation of Difference
    final_diff = ctrl_mean[-1] - drug_mean[-1]
    print(f"\nRESULTS AT YEAR {MAX_YEARS}:")
    print(f"   Control Atrophy:   {ctrl_mean[-1]:.2f}%")
    print(f"   Treatment Atrophy: {drug_mean[-1]:.2f}%")
    print(f"   Absolute Reduction: {final_diff:.2f}%")

    # Save
    save_path = f"{OUTPUT_DIR}/Lecanemab_Comparison_Result.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Graph saved to: {save_path}")