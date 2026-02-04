import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import numpy as np
from matplotlib.lines import Line2D

# 1. PASTE YOUR DATA HERE
csv_data = """Genotype,Lecanemab_mg_kg,Cromolyn_mg_kg,Mean_Atrophy_Percent,Std_Dev_Atrophy,Median_Abs_Deviation
APOE3 (Control),0.0,0.0,10.28,5.27,0.60
APOE3 (Control),0.0,25.0,12.54,5.94,3.10
APOE3 (Control),0.0,50.0,10.12,3.61,1.10
APOE3 (Control),0.0,75.0,14.42,5.92,3.50
APOE3 (Control),0.0,100.0,9.68,4.08,2.40
APOE3 (Control),2.5,0.0,11.44,4.71,0.70
APOE3 (Control),2.5,25.0,11.02,2.59,1.60
APOE3 (Control),2.5,50.0,12.38,3.77,0.20
APOE3 (Control),2.5,75.0,11.72,6.74,1.90
APOE3 (Control),2.5,100.0,11.24,6.17,0.50
APOE3 (Control),5.0,0.0,12.72,4.14,2.60
APOE3 (Control),5.0,25.0,14.64,4.12,4.80
APOE3 (Control),5.0,50.0,9.56,3.55,1.50
APOE3 (Control),5.0,75.0,12.88,8.75,0.60
APOE3 (Control),5.0,100.0,9.84,5.09,1.20
APOE3 (Control),7.5,0.0,12.44,4.35,1.30
APOE3 (Control),7.5,25.0,10.56,3.25,3.00
APOE3 (Control),7.5,50.0,14.16,10.24,3.30
APOE3 (Control),7.5,75.0,10.34,5.42,1.50
APOE3 (Control),7.5,100.0,14.62,5.99,6.80
APOE3 (Control),10.0,0.0,10.78,4.10,1.20
APOE3 (Control),10.0,25.0,14.42,10.26,4.70
APOE3 (Control),10.0,50.0,14.00,7.99,2.40
APOE3 (Control),10.0,75.0,13.32,5.59,5.40
APOE3 (Control),10.0,100.0,13.86,7.57,2.30"""

# Load Data
df = pd.read_csv(io.StringIO(csv_data))

# 2. SELECT BEST UNCERTAINTY MEASURE
avg_std = df['Std_Dev_Atrophy'].mean()
avg_mad = df['Median_Abs_Deviation'].mean()

if avg_mad < avg_std:
    uncertainty_col = 'Median_Abs_Deviation'
    label_text = "MAD (Median Abs. Deviation)"
else:
    uncertainty_col = 'Std_Dev_Atrophy'
    label_text = "Standard Deviation"

print(f"Using {label_text} (Lower Average Uncertainty)")

# 3. PIVOT DATA FOR HEATMAP
heatmap_mean = df.pivot(index='Cromolyn_mg_kg', columns='Lecanemab_mg_kg', values='Mean_Atrophy_Percent')
heatmap_err = df.pivot(index='Cromolyn_mg_kg', columns='Lecanemab_mg_kg', values=uncertainty_col)

# Sort index so low doses are at the top-left (optional, standard matrix view)
heatmap_mean.sort_index(ascending=True, inplace=True)
heatmap_err.sort_index(ascending=True, inplace=True)

# 4. PLOT
plt.figure(figsize=(11, 9))

# Draw the main heatmap
ax = sns.heatmap(heatmap_mean, annot=True, fmt=".1f", cmap="viridis", 
                 cbar_kws={'label': 'Mean Atrophy %'}, linewidths=0.5, linecolor='gray')

# 5. OVERLAY "TWO DOT" ERROR BARS
# Determine scaling: Max error should occupy about 80% of the cell width (0.4 * 2)
max_error_val = heatmap_err.max().max()
scaling_factor = 0.4 

rows, cols = heatmap_mean.shape
for i in range(rows):
    for j in range(cols):
        # Get error value for this cell
        # (Note: i corresponds to rows/Cromolyn, j to cols/Lecanemab)
        err = heatmap_err.iloc[i, j]
        
        # Normalize error to pixel distance
        offset = (err / max_error_val) * scaling_factor
        
        # Define cell center
        center_x = j + 0.5
        center_y = i + 0.5
        
        # Plot Two Dots (Shifted slightly down to avoid covering the text number)
        # Dot 1 (Left Bound)
        ax.plot(center_x - offset, center_y + 0.25, 'o', 
                color='white', markeredgecolor='black', markersize=5, alpha=0.9)
        # Dot 2 (Right Bound)
        ax.plot(center_x + offset, center_y + 0.25, 'o', 
                color='white', markeredgecolor='black', markersize=5, alpha=0.9)

# Labels and Titles
plt.title(f"Drug Synergy Heatmap: Mean Atrophy %\n(Dots Separation = {label_text})", fontsize=14)
plt.xlabel("Lecanemab Dose (mg/kg)", fontsize=12)
plt.ylabel("Cromolyn Dose (mg/kg)", fontsize=12)

# Custom Legend
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
                          markeredgecolor='black', label='Uncertainty Interval\n(Wider = Higher Error)')]
plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.35, 1.0))

plt.tight_layout()
plt.show()
# plt.savefig("heatmap_result.png", dpi=300) # Uncomment to save