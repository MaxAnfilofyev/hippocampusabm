import pandas as pd
import numpy as np
import time
from itertools import product
import sys

# Import your model class
# Ensure your simulation file is named 'model.py'
try:
    from try_again import ADModel
except ImportError:
    print("Error: Could not import ADModel. Make sure your model file is named 'try_again.py' and is in the same folder.")
    sys.exit(1)

# --- 1. EXPERIMENTAL CONFIGURATION ---

# Simulation Length
MAX_STEPS = 1000  

# Replicates: How many times to repeat EACH scenario?
# For testing, keep this at 1 or 5. 
# For your FINAL Science Fair data, increase this to 10 or 20 for statistical significance.
N_REPLICATES = 5 

# The Dose Matrix (25 Combinations)
lecanemab_doses = [0.0, 2.5, 5.0, 7.5, 10.0]
cromolyn_doses = [0.0, 25.0, 50.0, 75.0, 100.0]

# Genetic Backgrounds & Overrides
# We map your requested names to the model's internal logic.
genetic_scenarios = {
    "APOE3": {
        "base_profile": "APOE3 (Control)",
        "overrides": None
    },
    "APOE4": {
        "base_profile": "APOE 4/4",
        "overrides": None
    },
    "TREM2 R47H": {
        "base_profile": "APOE3 (Control)",
        # TREM2 R47H is a Loss-of-Function mutation. 
        # Microglia are "blind" and "lazy" but not necessarily hyper-toxic like APOE4.
        "overrides": {
            "amyloid_clearance": 0.2,       # Can't eat plaque
            "chemotaxis_efficiency": 0.3,   # Can't find plaque
            "inflammatory_threshold": 0.9,  # Very hard to activate (silent failure)
            "proliferation_rate": 0.5       # Poor replication
        }
    }
}

# --- 2. BATCH RUNNER ENGINE ---

def run_experiments():
    # Calculate total runs just for the progress bar
    total_permutations = len(genetic_scenarios) * len(lecanemab_doses) * len(cromolyn_doses)
    total_runs = total_permutations * N_REPLICATES
    
    print(f"--- SCIENCE FAIR BATCH RUNNER ---")
    print(f"Genotypes: {list(genetic_scenarios.keys())}")
    print(f"Lecanemab Doses: {lecanemab_doses}")
    print(f"Cromolyn Doses: {cromolyn_doses}")
    print(f"Steps per Run: {MAX_STEPS}")
    print(f"Replicates: {N_REPLICATES}")
    print(f"TOTAL SIMULATIONS: {total_runs}")
    print("-" * 50)
    
    all_results = []
    run_count = 0
    start_time = time.time()

    # Loop 1: Genetics
    for gene_label, gene_config in genetic_scenarios.items():
        base = gene_config["base_profile"]
        overrides = gene_config["overrides"]
        
        # Loop 2: Lecanemab
        for lec in lecanemab_doses:
            
            # Loop 3: Cromolyn
            for cro in cromolyn_doses:
                
                # Loop 4: Replicates (The Repeat)
                for rep in range(N_REPLICATES):
                    
                    # Initialize Model
                    model = ADModel(
                        width=40, height=40,
                        n_neurons=1000,
                        n_microglia=200,
                        genotype=base,
                        genetic_overrides=overrides,
                        lecanemab_dose=lec,
                        cromolyn_dose=cro,
                        max_steps=MAX_STEPS
                    )
                    
                    # Run Simulation
                    # We run blindly for speed (no visualization overhead)
                    for _ in range(MAX_STEPS):
                        model.step()
                    
                    # Collect Data
                    # We grab the FINAL state of the model
                    row = {
                        "Run_ID": run_count + 1,
                        "Genotype": gene_label,
                        "Lecanemab_mg_kg": lec,
                        "Cromolyn_mg_kg": cro,
                        "Replicate": rep + 1,
                        "Final_Atrophy_Percent": model.get_atrophy_percent(),
                        "Final_Total_Amyloid": np.sum(model.amyloid),
                        "Final_Total_Cytokine": np.sum(model.cytokine)
                    }
                    all_results.append(row)
                    
                    run_count += 1
                    
                    # Progress Bar Logic
                    if run_count % 10 == 0 or run_count == total_runs:
                        elapsed = time.time() - start_time
                        runs_per_sec = run_count / elapsed
                        remaining = (total_runs - run_count) / runs_per_sec
                        print(f"Progress: {run_count}/{total_runs} ({run_count/total_runs:.1%}) - "
                              f"Est. Remaining: {remaining/60:.1f} min")

    # --- 3. EXPORT ---
    print("-" * 50)
    print("Processing Data...")
    
    df = pd.DataFrame(all_results)
    
    # Save Raw Data (Every single run)
    df.to_csv("science_fair_raw_data.csv", index=False)
    
    # Save Summary Data (Averages per group)
    # This is what you use for your graphs!
    summary = df.groupby(["Genotype", "Lecanemab_mg_kg", "Cromolyn_mg_kg"]).agg({
        "Final_Atrophy_Percent": ["mean", "std"],
        "Final_Total_Amyloid": "mean",
        "Final_Total_Cytokine": "mean"
    }).reset_index()
    
    # Flatten columns for clean Excel import
    summary.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary.columns.values]
    summary.to_csv("science_fair_averages.csv", index=False)

    print("DONE!")
    print("1. 'science_fair_raw_data.csv' created (Use for error bars/stats).")
    print("2. 'science_fair_averages.csv' created (Use for main heatmaps/line charts).")

if __name__ == "__main__":
    run_experiments()