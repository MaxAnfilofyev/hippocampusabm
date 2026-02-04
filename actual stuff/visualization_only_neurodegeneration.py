import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit  # <--- NEW IMPORT

# Load Data
try:
    df = pd.read_csv("ADNI_Integrated_Master.csv")
except FileNotFoundError:
    print("Error: File not found.")
    exit()

# 1. CALCULATE % LOSS (Positive Direction)
def get_pct_loss(group):
    group = group.sort_values('Years')
    if group.empty: return group
    
    # Baseline values
    base_vol = group.iloc[0]['aHV']
    base_wmh = group.iloc[0]['TOTAL_WMH']
    
    # Formula: (Baseline - Current) / Baseline * 100
    if base_vol > 0:
        group['Neuronal_Loss'] = ((base_vol - group['aHV']) / base_vol) * 100
    
    if base_wmh > 0:
         group['Vascular_Injury'] = ((group['TOTAL_WMH'] - base_wmh) / base_wmh) * 100
    else:
        group['Vascular_Injury'] = 0
        
    return group

df = df.groupby('RID').apply(get_pct_loss)

# 2. FILTERING (UPDATED TO 20 YEARS)
plot_df = df[df['Years'] <= 20] 
valid_genotypes = ['3/3', '3/4', '4/4']
plot_df = plot_df[plot_df['GENOTYPE'].isin(valid_genotypes)]

# --- NEW SECTION STARTS HERE ---

# 3. DEFINE THE SIGMOID FUNCTION
def sigmoid(t, L, k, t0):
    """
    t  : Time (years)
    L  : Max loss (saturation point, e.g., 40%)
    k  : Steepness/Growth rate
    t0 : Midpoint (time when loss is half of max)
    """
    return L / (1 + np.exp(-k * (t - t0)))

# 4. VISUALIZATION (Sigmoidal Fit)
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 8))

colors = {"3/3": "green", "3/4": "orange", "4/4": "red"}
genotype_order = ['3/3', '3/4', '4/4']

# Plot the raw scatter points first
sns.scatterplot(data=plot_df, x="Years", y="Neuronal_Loss", hue="GENOTYPE", 
                palette=colors, alpha=0.15, s=15, legend=False)

print("\n" + "="*80)
print("   SIGMOIDAL ATROPHY MODEL COEFFICIENTS")
print("   Equation: Loss(t) = L / (1 + e^(-k*(t - t0)))")
print("="*80)
print(f"{'Genotype':<10} | {'L (Max %)':<12} | {'k (Rate)':<12} | {'t0 (Midpoint)':<15}")
print("-" * 80)

# Generate smooth x-axis for plotting curves
x_range = np.linspace(0, 20, 200)

for g in genotype_order:
    sub_df = plot_df[plot_df['GENOTYPE'] == g].dropna(subset=['Years', 'Neuronal_Loss'])
    
    if len(sub_df) > 20:
        try:
            # Initial Guesses (p0) are CRITICAL for sigmoids
            # L=30 (max loss), k=0.5 (slope), t0=10 (midpoint year)
            p0 = [30, 0.5, 10]
            
            # bounds: L(0-100), k(0-5), t0(0-30)
            popt, pcov = curve_fit(sigmoid, sub_df['Years'], sub_df['Neuronal_Loss'], 
                                   p0=p0, bounds=([0, 0, 0], [100, 5, 30]), maxfev=5000)
            
            L, k, t0 = popt
            
            # Print Params
            print(f"{g:<10} | {L:.4f}       | {k:.4f}       | {t0:.4f}")
            
            # Plot the fitted curve
            y_fit = sigmoid(x_range, *popt)
            plt.plot(x_range, y_fit, color=colors[g], linewidth=3, label=f"{g} (Sigmoid)")
            
        except RuntimeError:
            print(f"{g:<10} | Fit failed (no convergence)")
    else:
        print(f"{g:<10} | Not enough data")

print("-" * 80)

# Graph Styling
plt.title("AD Progression: Neurodegeneration (Sigmoidal Model)", fontsize=20)
plt.ylabel("% Neuronal Volume Loss", fontsize=14)
plt.xlabel("Years from Baseline", fontsize=14)
plt.ylim(-5, 45)  # Adjusted slightly higher for saturation
plt.legend(title="APOE Genotype", fontsize=12)
plt.tight_layout()
plt.show()