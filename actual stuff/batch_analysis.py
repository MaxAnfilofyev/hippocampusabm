import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import pandas as pd
import random
import seaborn as sns
import os

# --- IMPORT YOUR MODEL ---
try:
    from try_again import ADModel
except ImportError:
    print("❌ CRITICAL ERROR: Could not import 'ADModel' from 'try_again.py'. Check filename.")
    exit()

# --- 1. CONFIGURATION ---
NUM_BATCHES = 50          # Total Groups to test
REPLICATES = 5            # Runs per Group (for Std Dev)
MAX_YEARS = 20            
STEPS_PER_YEAR = 50       
TOTAL_STEPS = MAX_YEARS * STEPS_PER_YEAR

# Clinical Target Parameters (APOE3 Benchmark)
TARGET_PARAMS = [100.0, 0.2587, 27.72]

# Visualization Settings
Y_LIMIT = 20              # Max Y Value
DPI = 300                 # High resolution
COLORS = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854'] # Pastel palette
OUTPUT_DIR = "APOE3_Calibration_Final"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 2. HELPER FUNCTIONS ---

def sigmoid(t, L, k, t0):
    """The sigmoid function used for fitting and targeting."""
    # Prevent overflow in exp
    t0 = np.clip(t0, -100, 100) 
    return L / (1 + np.exp(-k * (t - t0)))

def get_target_curve():
    """Generates the reference array."""
    t = np.linspace(0, MAX_YEARS, TOTAL_STEPS)
    return sigmoid(t, *TARGET_PARAMS)

# --- 3. SIMULATION RUNNERS (REAL DATA) ---

def run_single_simulation(seed):
    """Runs one instance of the actual ABM."""
    model = ADModel(
        width=40, height=40,
        n_neurons=1000, n_microglia=200,
        genotype="APOE3 (Control)",
        lecanemab_dose=0.0,
        cromolyn_dose=0.0,
        max_steps=TOTAL_STEPS,
        seed=seed
    )
    
    history = []
    for _ in range(TOTAL_STEPS):
        model.step()
        history.append(model.get_atrophy_percent())
    return history

def run_batch(batch_id):
    """
    Runs a batch of N replicates using the actual model.
    """
    master_seed = random.randint(10000, 99999)
    replicate_data = []
    
    # Run the replicates
    for i in range(REPLICATES):
        # Deterministic sub-seed for reproducibility
        sub_seed = int(f"{master_seed}{i}")
        data = run_single_simulation(sub_seed)
        replicate_data.append(data)
    
    # Calculate Statistics
    matrix = np.array(replicate_data)
    mean_curve = np.mean(matrix, axis=0)
    std_curve = np.std(matrix, axis=0)
    
    # Calculate Metrics for Ranking
    target = get_target_curve()
    r2 = r2_score(target, mean_curve)
    final_std = std_curve[-1] # The standard deviation at the final step
    
    time_axis = np.linspace(0, MAX_YEARS, TOTAL_STEPS)
    
    return {
        "id": batch_id,
        "master_seed": master_seed,
        "r2": r2,
        "final_std": final_std,
        "mean": mean_curve,
        "std": std_curve,
        "time": time_axis
    }

# --- 4. MAIN EXECUTION ---

if __name__ == "__main__":
    print(f"🚀 STARTING PRODUCTION RUN")
    print(f"   Model: try_again.ADModel")
    print(f"   Batches: {NUM_BATCHES} | Replicates: {REPLICATES}")
    print(f"   Total Sims: {NUM_BATCHES * REPLICATES}")
    print("-" * 60)
    
    results = []
    
    # RUN SIMULATIONS
    for b in range(NUM_BATCHES):
        print(f"   Processing Batch {b+1}/{NUM_BATCHES}...", end="\r")
        res = run_batch(b)
        
        # Filter artifacts (if R2 is too perfect or negative)
        if 0.0 < res['r2'] < 0.999:
            results.append(res)
            
    print(f"\n✅ Simulations Complete. Ranking results...")

    # --- RANKING LOGIC ---
    # Sort Key: (Primary: R2 Descending, Secondary: Final Std Ascending)
    # We implement this by negating R2 (so -0.99 is smaller than -0.90) 
    # and keeping Std positive (so 1.0 is smaller than 5.0).
    # Then we use a standard ascending sort.
    results.sort(key=lambda x: (-x['r2'], x['final_std']))
    
    top_5 = results[:5]
    
    print("\n🏆 Top 5 Batches Selected:")
    for i, r in enumerate(top_5):
        print(f"   Rank {i+1}: R2={r['r2']:.5f} | Final Std={r['final_std']:.4f} | Seed={r['master_seed']}")

    # --- VISUALIZATION LOOP ---
    
    # Set Plot Style
    plt.style.use('seaborn-v0_8-whitegrid')
    target_curve = get_target_curve()

    for i, res in enumerate(top_5):
        fig, ax = plt.subplots(figsize=(10, 7))
        color = COLORS[i % len(COLORS)]
        time_axis = res['time']
        
        # 1. Plot Clinical Target (Dashed Grey)
        ax.plot(time_axis, target_curve, color="#444444", linestyle="--", 
                linewidth=2.5, alpha=0.6, label="Clinical Target")

        # 2. FIT A CURVE to the Simulation Mean
        try:
            # Initial guess [L=100, k=0.2, t0=25]
            p0 = [100, 0.2, 25] 
            popt, _ = curve_fit(sigmoid, time_axis, res['mean'], p0=p0, maxfev=10000)
            
            # Generate clean fitted line
            fitted_curve = sigmoid(time_axis, *popt)
            
            # Plot the Fitted Curve (Solid Color)
            ax.plot(time_axis, fitted_curve, color=color, linewidth=3, 
                    label=f"Sim Best Fit ($R^2$={res['r2']:.4f})")
            
        except Exception as e:
            print(f"Fitting failed for batch {res['id']}: {e}")
            ax.plot(time_axis, res['mean'], color=color, linewidth=3, label="Sim Mean (Raw)")

        # 3. Plot Error Bars (Standard Deviation)
        # Plot every 2 years (index = steps_per_year * 2)
        interval_step = STEPS_PER_YEAR * 2
        indices = np.arange(0, TOTAL_STEPS, interval_step)
        
        # Ensure indices don't exceed array bounds
        indices = indices[indices < len(time_axis)]
        
        ax.errorbar(time_axis[indices], 
                    res['mean'][indices], 
                    yerr=res['std'][indices], 
                    fmt='o',             # Circle marker
                    color=color,         # Marker color
                    ecolor=color,        # Bar color
                    elinewidth=2,        # Bar thickness
                    capsize=4,           # Cap width
                    alpha=0.8,
                    label=f"Std. Dev. (Final: {res['final_std']:.2f})")

        # 4. Formatting
        title_text = f"APOE3 Calibration: Rank {i+1} (Seed {res['master_seed']})"
        ax.set_title(title_text, fontsize=16, fontweight='bold', pad=20, color='#333333')
        
        ax.set_xlabel("Years from Baseline", fontsize=12, fontweight='medium', color='#333333')
        ax.set_ylabel("Neurodegeneration (%)", fontsize=12, fontweight='medium', color='#333333')
        
        # Limits & Spines
        ax.set_ylim(0, Y_LIMIT)
        ax.set_xlim(0, MAX_YEARS)
        
        # --- UPDATE: Explicit 2-Year Intervals on X-Axis ---
        ax.set_xticks(np.arange(0, MAX_YEARS + 1, 2))
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(frameon=True, framealpha=0.9, loc="upper left", fontsize=11)

        # Save
        filename = f"{OUTPUT_DIR}/Rank_{i+1}_Seed_{res['master_seed']}.png"
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        print(f"   -> Graph saved: {filename}")

    # Export Data for Future Use
    print(f"\n💾 Saving Data CSV...")
    df_data = pd.DataFrame({"Time": time_axis, "Target": target_curve})
    for i, res in enumerate(top_5):
        df_data[f"Rank_{i+1}_Mean"] = res['mean']
        df_data[f"Rank_{i+1}_Std"] = res['std']
    df_data.to_csv(f"{OUTPUT_DIR}/Top_5_Data.csv", index=False)

    print("\n✨ All Done! Check the folder.")