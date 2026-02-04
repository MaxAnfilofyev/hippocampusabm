import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

# 2. FILTERING (RESTRICTED TO 14 YEARS)
plot_df = df[df['Years'] <= 14]  # <--- Changed from 20 to 14
valid_genotypes = ['3/3', '3/4', '4/4']
plot_df = plot_df[plot_df['GENOTYPE'].isin(valid_genotypes)]

# 3. VISUALIZATION
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True)

colors = {"3/3": "green", "3/4": "orange", "4/4": "red"}
genotype_order = ['3/3', '3/4', '4/4']

# PANEL 1: Amyloid
for g in genotype_order:
    sns.regplot(data=plot_df[plot_df['GENOTYPE']==g], x="Years", y="CENTILOIDS", 
                scatter_kws={'alpha':0.1, 's':5}, line_kws={'label':g, 'color':colors[g]}, 
                order=2, ax=axes[0])
axes[0].set_title("1. Amyloid Accumulation")
axes[0].set_ylabel("Amyloid Load (Centiloids)")
axes[0].set_ylim(0, 150)

# PANEL 2: Neuroinflammation
for g in genotype_order:
    sns.regplot(data=plot_df[plot_df['GENOTYPE']==g], x="Years", y="GFAP_Q", 
                scatter_kws={'alpha':0.1, 's':5}, line_kws={'label':g, 'color':colors[g]}, 
                order=2, ax=axes[1])
axes[1].set_title("2. Neuroinflammation")
axes[1].set_ylabel("Plasma GFAP (pg/mL)")

# PANEL 3: Neurodegeneration
for g in genotype_order:
    sns.regplot(data=plot_df[plot_df['GENOTYPE']==g], x="Years", y="Neuronal_Loss", 
                scatter_kws={'alpha':0.1, 's':5}, line_kws={'label':g, 'color':colors[g]}, 
                order=1, ax=axes[2])
axes[2].set_title("3. Neurodegeneration (% Volume Lost)")
axes[2].set_ylabel("% Neuronal Loss")
axes[2].set_ylim(-5, 25) # Adjusted scale for 14 years

plt.suptitle("AD Progression Model: Amyloid, Inflammation, & Degeneration (0-14 Years)", fontsize=20)
axes[0].legend(title="APOE Genotype")
plt.tight_layout()

# 4. PRINT SLOPES
print("\n--- ATROPHY RATES (Calculated over 14 Years) ---")
print(f"{'Genotype':<10} | {'Slope (% Loss/Year)':<20} | {'Simulation Value (Decimal)'}")
print("-" * 60)

for g in ['3/3', '3/4', '4/4']:
    sub_df = plot_df[plot_df['GENOTYPE'] == g].dropna(subset=['Years', 'Neuronal_Loss'])
    
    if len(sub_df) > 1:
        slope, intercept = np.polyfit(sub_df['Years'], sub_df['Neuronal_Loss'], 1)
        sim_value = slope / 100.0
        print(f"{g:<10} | {slope:.4f}% / year       | {sim_value:.5f}")
    else:
        print(f"{g:<10} | Not enough data")

print("-" * 60)
plt.show()