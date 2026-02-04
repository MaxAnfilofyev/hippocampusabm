import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# 1. CONFIGURATION
# Set to current directory since we run from within 'actual stuff'
DATA_FOLDER = Path(".")


def generate_heatmap(full_df, config):
    try:
        # Filter by Genotype
        df = full_df[full_df["Genotype"] == config["genotype"]]
        
        if df.empty:
            print(f"WARNING: No data found for genotype {config['genotype']}")
            return

        # Pivot Data: Rows=Cromolyn, Cols=Lecanemab, Values=Atrophy
        heatmap_data = df.pivot(
            index="Cromolyn_mg_kg", 
            columns="Lecanemab_mg_kg", 
            values="Final_Atrophy_Percent_mean"
        )
        
        # Sort index so 0.0 is at the bottom (standard graph style)
        heatmap_data.sort_index(ascending=False, inplace=True)
        
        # Create Plot
        plt.figure(figsize=(10, 8))
        
        # Create Heatmap
        ax = sns.heatmap(
            heatmap_data, 
            annot=True, 
            fmt=".1f", 
            cmap="RdYlGn_r", 
            vmin=10, vmax=35, # Adjusted for better contrast
            cbar_kws={'label': 'Neuron Atrophy (%)'}
        )
        
        plt.title(config["title"], fontsize=16, fontweight='bold')
        plt.xlabel("Lecanemab Dose (mg/kg)", fontsize=12)
        plt.ylabel("Cromolyn Dose (mg/kg)", fontsize=12)
        
        # Save
        plt.tight_layout()
        plt.savefig(config["save_as"], dpi=300)
        plt.close()
        
        print(f"Successfully created: {config['save_as']}")
        
    except Exception as e:
        print(f"ERROR generating {config['save_as']}: {e}")

# Run the generator
print("--- Generating Science Fair Heatmaps ---")
sns.set_context("talk") 

# Load the single unified CSV
csv_path = DATA_FOLDER / "science_fair_averages.csv"
if csv_path.exists():
    master_df = pd.read_csv(csv_path)
    
    datasets = [
        {
            "genotype": "APOE3",
            "title": "APOE 3/3 (Control): Atrophy Heatmap",
            "save_as": "heatmap_apoe3.png"
        },
        {
            "genotype": "APOE4",
            "title": "APOE 4/4 (Homozygous): Atrophy Heatmap",
            "save_as": "heatmap_apoe44.png"
        },
        {
            "genotype": "TREM2 R47H",
            "title": "TREM2 R47H: Atrophy Heatmap",
            "save_as": "heatmap_trem2.png"
        }
    ]

    for data in datasets:
        generate_heatmap(master_df, data)
else:
    print(f"ERROR: '{csv_path}' not found. Run the data collect script first.")

print("Done!")