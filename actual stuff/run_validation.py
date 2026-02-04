import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd

# --- IMPORT YOUR MODEL ---
# Replace 'try_again' with the actual name of your python file if different
try:
    from try_again import ADModel
except ImportError:
    print("❌ CRITICAL ERROR: Could not import ADModel. Check your filename.")
    exit()

# --- 1. CONFIGURATION ---
NUM_TRIALS = 10          # Number of times to run the simulation
MAX_YEARS = 20           # Clinical timeline to simulate
STEPS_PER_YEAR = 50      # 1000 steps / 20 years = 50 steps per year

# CLINICAL TARGET PARAMETERS (APOE3 Benchmark)
# L=100 (Max Atrophy), k=0.2587 (Rate), t0=27.72 (Midpoint)
TARGET_PARAMS = [100.0, 0.2587, 27.72]

# --- 2. PASTE YOUR OPTIMIZED PARAMETERS HERE ---
# Use the values you found in the previous optimization step
OPTIMAL_PARAMS = {
    'amyloid_toxicity': 25.0,      # <--- REPLACE WITH YOUR RESULT
    'cytokine_toxicity': 35.0,     # <--- REPLACE WITH YOUR RESULT
    'neuron_repair_rate': 0.0035,  # <--- REPLACE WITH YOUR RESULT
    'neuron_death_threshold': 0.15 # <--- REPLACE WITH YOUR RESULT
}

# --- 3. HELPER FUNCTIONS ---
def sigmoid(t, L, k, t0):
    """The theoretical clinical curve."""
    return L / (1 + np.exp(-k * (t - t0)))

def run_single_trial(trial_id):
    """Runs one complete simulation of 20 years."""
    print(f"   ...Running Trial {trial_id + 1}/{NUM_TRIALS}")
    
    model = ADModel(
        width=40, height=40,            # Using the scaled-up grid
        n_neurons=1000, n_microglia=200, 
        genotype="APOE3 (Control)",
        
        # Inject Optimized Parameters
        amyloid_toxicity=OPTIMAL_PARAMS['amyloid_toxicity'],
        cytokine_toxicity=OPTIMAL_PARAMS['cytokine_toxicity'],
        neuron_repair_rate=OPTIMAL_PARAMS['neuron_repair_rate'],
        neuron_death_threshold=OPTIMAL_PARAMS['neuron_death_threshold'],
        
        max_steps=MAX_YEARS * STEPS_PER_YEAR
    )
    
    atrophy_history = []
    
    # Run year by year
    for _ in range(MAX_YEARS):
        for _ in range(STEPS_PER_YEAR):
            model.step()
        # Record data at the end of each year
        atrophy_history.append(model.get_atrophy_percent())
        
    return atrophy_history

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"🚀 STARTING VALIDATION: {NUM_TRIALS} Trials over {MAX_YEARS} Simulated Years")
    print("-" * 60)

    # A. Collect Data
    all_trials_data = []
    
    for i in range(NUM_TRIALS):
        trial_data = run_single_trial(i)
        all_trials_data.append(trial_data)

    # Convert to NumPy array for easy math (Rows=Trials, Cols=Years)
    results_matrix = np.array(all_trials_data)

    # B. Calculate Statistics
    mean_curve = np.mean(results_matrix, axis=0) # Average across trials
    std_curve = np.std(results_matrix, axis=0)   # Standard Deviation (Error bars)
    
    # C. Generate Target Data for Comparison
    time_points = np.arange(1, MAX_YEARS + 1)
    target_curve = sigmoid(time_points, *TARGET_PARAMS)

    # D. Calculate R-Squared (Goodness of Fit)
    # R^2 = 1 - (Sum of Squared Residuals / Total Sum of Squares)
    ss_res = np.sum((target_curve - mean_curve) ** 2)
    ss_tot = np.sum((target_curve - np.mean(target_curve)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # --- 5. OUTPUT & VISUALIZATION ---
    print("-" * 60)
    print(f"✅ VALIDATION COMPLETE")
    print(f"📊 Final R² Score: {r_squared:.4f}")
    print("-" * 60)
    print("Year | Target % | Sim Mean % | Sim Std Dev")
    for t, tgt, sim, err in zip(time_points, target_curve, mean_curve, std_curve):
        print(f"{t:4d} | {tgt:8.2f} | {sim:10.2f} | {err:11.2f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot individual trials (faint lines)
    for trial in all_trials_data:
        plt.plot(time_points, trial, color='gray', alpha=0.3, linewidth=1)
        
    # Plot Mean Curve (Bold Blue)
    plt.plot(time_points, mean_curve, color='blue', linewidth=3, label='Simulated Average')
    
    # Plot Target Curve (Dashed Red)
    plt.plot(time_points, target_curve, color='red', linestyle='--', linewidth=2, label='Clinical Target (Sigmoid)')
    
    # Formatting
    plt.title(f"Model Validation (n={NUM_TRIALS}): R² = {r_squared:.4f}")
    plt.xlabel("Simulated Years")
    plt.ylabel("Hippocampal Atrophy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 20)
    
    # Save and Show
    plt.savefig("final_validation_curve.png")
    print("\n📈 Plot saved to 'final_validation_curve.png'")
    plt.show()