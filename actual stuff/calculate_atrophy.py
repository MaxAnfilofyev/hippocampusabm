import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_curved_atrophy(csv_path="ADNI_Integrated_Master.csv"):
    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_path}'")
        return

    # 2. Calculate Total Volume (Left + Right)
    if 'LEFT_HIPP_VOL' in df.columns and 'RIGHT_HIPP_VOL' in df.columns:
        df['Calculated_Total_Vol'] = df['LEFT_HIPP_VOL'] + df['RIGHT_HIPP_VOL']
        print("✓ Successfully calculated Total Volume from Left + Right columns.")
    elif 'Total_Hipp_Vol' in df.columns:
        df['Calculated_Total_Vol'] = df['Total_Hipp_Vol']
        print("⚠ Left/Right columns missing. Using existing 'Total_Hipp_Vol'.")
    else:
        print("Error: Dataset missing Left/Right volume columns.")
        return

    # 3. Calculate % Loss per Patient
    def get_pct_loss(group):
        group = group.sort_values('Years')
        if group.empty: return None
        
        baseline = group.iloc[0]['Calculated_Total_Vol']
        if baseline <= 0 or pd.isna(baseline): return None
        
        group['Pct_Loss'] = ((baseline - group['Calculated_Total_Vol']) / baseline) * 100
        return group

    df_loss = df.groupby('RID').apply(get_pct_loss)
    
    if df_loss is None or df_loss.empty: 
        print("No valid patient volume data found.")
        return
        
    df_loss = df_loss.reset_index(drop=True)
    df_loss = df_loss.dropna(subset=['Years', 'Pct_Loss', 'GENOTYPE'])
    
    # Filter noise
    df_loss = df_loss[(df_loss['Pct_Loss'] > -10) & (df_loss['Pct_Loss'] < 60)]

    # 4. Fit Quadratic Curves (y = ax^2 + bx)
    print("\n" + "="*60)
    print("   HIPPOCAMPAL ATROPHY CURVE COEFFICIENTS")
    print("   Equation: Loss(t) = a*t^2 + b*t")
    print("="*60)
    print(f"{'Genotype':<10} | {'a (Accel)':<12} | {'b (Linear)':<12} | {'Description'}")
    print("-" * 60)

    genotypes = ['3/3', '3/4', '4/4']
    coefficients = {}

    plt.figure(figsize=(10, 6))
    
    for g in genotypes:
        sub = df_loss[df_loss['GENOTYPE'] == g]
        
        if len(sub) > 20: 
            X = np.vstack([sub['Years']**2, sub['Years']]).T
            y = sub['Pct_Loss']
            
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            a, b = coeffs
            
            coefficients[g] = [a, b]
            
            desc = "Accelerating" if a > 0 else "Decelerating"
            print(f"{g:<10} | {a:.5f}      | {b:.5f}      | {desc}")

            # --- CHANGE IS HERE ---
            # Use the max year found in this specific genotype's data
            max_year = sub['Years'].max()
            t = np.linspace(0, max_year, 100) 
            # ----------------------
            
            loss_fit = a*t**2 + b*t
            plt.plot(t, loss_fit, label=f"{g} (Max Year: {max_year:.1f})", linewidth=2.5)
        else:
            print(f"{g:<10} | Insufficient data (n={len(sub)})")

    print("-" * 60)
    plt.title("Atrophy Curves (Total Volume: Left + Right)")
    plt.xlabel("Years")
    plt.ylabel("% Volume Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n>>> PASTE THIS INTO try_again.py (DATA_CALIBRATION) <<<")
    for g, coeffs in coefficients.items():
        print(f"'{g}': {{ ... 'atrophy_coeffs': [{coeffs[0]:.5f}, {coeffs[1]:.5f}] }},")

if __name__ == "__main__":
    calculate_curved_atrophy()