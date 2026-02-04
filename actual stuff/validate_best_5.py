import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.metrics import r2_score

# --- IMPORT YOUR MODEL ---
try:
    from try_again import ADModel 
except ImportError:
    print("❌ CRITICAL ERROR: Could not import ADModel. Check your filename.")
    exit()

# --- 1. CONFIGURATION ---
NUM_SIMULATIONS = 20     # Total different configurations/genotypes
TRIALS_PER_SIM = 10       # Replicates per simulation to find STDEV
TOP_N = 5                # How many best results to display
MAX_YEARS = 20 
STEPS_PER_YEAR = 50 

# CLINICAL TARGET PARAMETERS (APOE3 Benchmark)
TARGET_PARAMS = [100.0, 0.2587, 27.72]

# --- 2. PARAMETERS ---
OPTIMAL_PARAMS = {
    'amyloid_toxicity': 40.0,    
    'cytokine_toxicity': 10.0, 
    'neuron_repair_rate': 0.004174,    
    'neuron_death_threshold': 0.004174, 
}

# --- 3. HELPER FUNCTIONS ---
def sigmoid(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

def calculate_r2(sim_data, target_data):
    ss_res = np.sum((target_data - sim_data) ** 2)
    ss_tot = np.sum((target_data - np.mean(target_data)) ** 2)
    return 1 - (ss_res / ss_tot)

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"🚀 STARTING: {NUM_SIMULATIONS} Simulations x {TRIALS_PER_SIM} Trials each.")
    
    time_points = np.arange(1, MAX_YEARS + 1)
    target_curve = sigmoid(time_points, *TARGET_PARAMS)
    simulation_results = []

    for sim_idx in range(NUM_SIMULATIONS):
        print(f"\n🧬 Running Simulation {sim_idx + 1}/{NUM_SIMULATIONS}...")
        trial_scores = []
        trial_histories = []
        seeds = []

        for t_idx in range(TRIALS_PER_SIM):
            current_seed = random.randint(10000, 999999)
            
            model = ADModel(
                width=40, height=40,
                n_neurons=1000, n_microglia=200, 
                genotype="APOE3",
                amyloid_toxicity=OPTIMAL_PARAMS['amyloid_toxicity'],
                cytokine_toxicity=OPTIMAL_PARAMS['cytokine_toxicity'],
                neuron_repair_rate=OPTIMAL_PARAMS['neuron_repair_rate'],
                neuron_death_threshold=OPTIMAL_PARAMS['neuron_death_threshold'],
                max_steps=MAX_YEARS * STEPS_PER_YEAR,
                seed=current_seed
            )
            
            atrophy_history = []
            for _ in range(MAX_YEARS):
                for _ in range(STEPS_PER_YEAR):
                    model.step()
                atrophy_history.append(model.get_atrophy_percent())
            
            sim_data = np.array(atrophy_history)
            r2 = calculate_r2(sim_data, target_curve)
            
            trial_scores.append(r2)
            trial_histories.append(sim_data)
            seeds.append(current_seed)
            print(f"   - Trial {t_idx+1}: R² = {r2:.4f}")

        # Calculate Statistics for this simulation group
        avg_r2 = np.mean(trial_scores)
        std_r2 = np.std(trial_scores)
        
        # We store the 'best' trial from this group to plot it later
        best_trial_idx = np.argmax(trial_scores)
        
        simulation_results.append({
            'sim_id': sim_idx + 1,
            'avg_r2': avg_r2,
            'std_r2': std_r2,
            'best_r2': trial_scores[best_trial_idx],
            'best_seed': seeds[best_trial_idx],
            'best_data': trial_histories[best_trial_idx]
        })

    # --- 5. SORTING AND RANKING ---
    # We sort by average R^2 score descending
    simulation_results.sort(key=lambda x: x['avg_r2'], reverse=True)
    top_results = simulation_results[:TOP_N]

    print("\n" + "="*95)
    print(f"🏆 TOP {TOP_N} PERFORMING SIMULATIONS (Sorted by Mean R²)")
    print("="*95)
    print(f"{'Rank':<6} | {'Mean R²':<10} | {'Std Dev':<10} | {'Best Seed':<12} | {'Best R²':<10}")
    print("-" * 95)
    
    for rank, res in enumerate(top_results):
        print(f"{rank+1:<6} | {res['avg_r2']:.5f}  | {res['std_r2']:.5f}  | {res['best_seed']:<12} | {res['best_r2']:.5f}")

    # --- 6. PLOTTING TOP RESULTS ---
    for rank, res in enumerate(top_results):
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, target_curve, 'r--', label='Clinical Target', linewidth=2)
        plt.plot(time_points, res['best_data'], 'b-', label=f"Best Trial (Seed {res['best_seed']})")
        
        plt.title(f"Rank {rank+1}: Mean R² = {res['avg_r2']:.4f} (±{res['std_r2']:.4f})")
        plt.xlabel("Years")
        plt.ylabel("Atrophy %")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = f"rank_{rank+1}_sim_{res['sim_id']}.png"
        plt.savefig(filename)
        plt.close()

    print("\n✅ Analysis complete. Best seeds and stability metrics recorded.")