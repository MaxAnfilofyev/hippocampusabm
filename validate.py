import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extract_input_dynamics(csv_path="ADNI_Integrated_Master.csv"):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: CSV not found.")
        return

    # 1. NORMALIZE DATA (Scale to 0.0 - 1.0 range for the model)
    # Amyloid: 0 = Healthy, 100+ = Alzheimer's. We normalize by 150 (cap).
    df['Amyloid_Norm'] = df['CENTILOIDS'].clip(0, 150) / 150.0
    
    # Inflammation (GFAP): Normalize by the 95th percentile to handle outliers
    gfap_max = df['GFAP_Q'].quantile(0.95)
    df['GFAP_Norm'] = df['GFAP_Q'].clip(0, gfap_max) / gfap_max

    # 2. FIT CURVES
    genotypes = ['3/3', '3/4', '4/4']
    results = {}

    print("\n" + "="*80)
    print(f"{'Genotype':<10} | {'Metric':<15} | {'Equation (at^2 + bt + c)':<30} | {'Sim Input'}")
    print("-" * 80)

    for g in genotypes:
        sub = df[df['GENOTYPE'] == g]
        # Filter: Only visits up to year 20
        sub = sub[(sub['Years'] >= 0) & (sub['Years'] <= 20)]
        sub = sub.dropna(subset=['Years', 'Amyloid_Norm', 'GFAP_Norm'])

        if len(sub) > 20:
            results[g] = {}
            
            for metric, col in [('Amyloid', 'Amyloid_Norm'), ('GFAP', 'GFAP_Norm')]:
                # Fit Polynomial Degree 2
                coeffs = np.polyfit(sub['Years'], sub[col], 2)
                a, b, c = coeffs
                
                results[g][metric] = [a, b, c]
                
                eq = f"{a:.4f}t² + {b:.4f}t + {c:.2f}"
                print(f"{g:<10} | {metric:<15} | {eq:<30} | a={a:.5f}, b={b:.5f}, c={c:.5f}")

    print("-" * 80)
    
    # 3. GENERATE PYTHON DICTIONARY CODE
    print("\n>>> COPY THIS DICTIONARY INTO try_again.py <<<")
    print("DATA_INPUTS = {")
    for g, metrics in results.items():
        print(f"    '{g}': {{")
        # We need the atrophy coeffs we calculated previously (Hardcoded here for example stability, 
        # but you should use your specific values from the previous step)
        print(f"        'amyloid_coeffs': {metrics['Amyloid']},")
        print(f"        'gfap_coeffs': {metrics['GFAP']}")
        print("    },")
    print("}")

if __name__ == "__main__":
    extract_input_dynamics()