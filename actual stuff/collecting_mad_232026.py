import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
import os
from try_again import ADModel  # Ensure this matches your filename

# --- EXECUTION SCRIPT ---

def run_batch_simulation():
    # 1. Setup Parameter Ranges
    genotypes = ["APOE3 (Control)", "APOE 3/4", "APOE 4/4"]
    lecanemab_doses = [0.0, 2.5, 5.0, 7.5, 10.0]
    cromolyn_doses = [0.0, 25.0, 50.0, 75.0, 100.0]
    iterations = 5
    max_steps = 1200 # Matching the model's default
    
    all_results = []

    print(f"Starting Batch Simulation: {len(genotypes) * len(lecanemab_doses) * len(cromolyn_doses) * iterations} total runs.")

    for genotype in genotypes:
        genotype_data = []
        
        for lec in lecanemab_doses:
            for cro in cromolyn_doses:
                run_atrophies = []
                
                for i in range(iterations):
                    # Use a unique seed for each iteration to get variance
                    run_seed = 42 + i + int(lec * 10) + int(cro)
                    
                    # Initialize Model
                    model = ADModel(
                        genotype=genotype,
                        lecanemab_dose=lec,
                        cromolyn_dose=cro,
                        max_steps=max_steps,
                        seed=run_seed
                    )
                    
                    # Run simulation to completion
                    for _ in range(max_steps):
                        model.step()
                    
                    # Collect final atrophy percentage
                    final_atrophy = model.get_atrophy_percent()
                    run_atrophies.append(final_atrophy)
                
                # 2. Calculate Statistics for the 5 runs
                mean_atrophy = np.mean(run_atrophies)
                std_atrophy = np.std(run_atrophies)
                mad_atrophy = median_abs_deviation(run_atrophies)
                
                # Store result row
                result_row = {
                    "Genotype": genotype,
                    "Lecanemab_mg_kg": lec,
                    "Cromolyn_mg_kg": cro,
                    "Mean_Atrophy_Percent": mean_atrophy,
                    "Std_Dev_Atrophy": std_atrophy,
                    "Median_Abs_Deviation": mad_atrophy
                }
                genotype_data.append(result_row)
                print(f"Completed: {genotype} | Lec: {lec} | Cro: {cro}")

        # 3. Export to separate CSVs per genotype
        df = pd.DataFrame(genotype_data)
        file_name = f"simulation_results_{genotype.replace(' ', '_').replace('/', 'n')}.csv"
        df.to_csv(file_name, index=False)
        print(f"--- Saved results to {file_name} ---")

if __name__ == "__main__":
    # Ensure the model classes from your snippet are in the same scope or imported
    run_batch_simulation()