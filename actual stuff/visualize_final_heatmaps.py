import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# 1. CONFIGURATION
# List of your data files and the titles for the graphs
DATA_FOLDER = Path(r"C:\Users\narwh\Downloads\Science Fair\actual stuff")

datasets = [
    {
        "file": DATA_FOLDER /"science_fair_averages_apoe3_edit.csv",
        "title": "APOE 3/3 (Control): Atrophy Heatmap",
        "save_as": "heatmap_apoe3.png"
    },
    {
        "file": DATA_FOLDER /"science_fair_averages_apoe34.csv",
        "title": "APOE 3/4 (Heterozygous): Atrophy Heatmap",
        "save_as": "heatmap_apoe34.png"
    },
    {
        "file": DATA_FOLDER /"science_fair_averages_apoe4.csv",
        "title": "APOE 4/4 (Homozygous): Atrophy Heatmap",
        "save_as": "heatmap_apoe44.png"
    }
]

def generate_heatmap(config):
    try:
        # Load Data
        df = pd.read_csv(config["file"])
        
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
        # cmap="RdYlGn_r" means: Red = High (Bad), Green = Low (Good)
        # annot=True shows the numbers in the boxes
        ax = sns.heatmap(
            heatmap_data, 
            annot=True, 
            fmt=".1f", 
            cmap="RdYlGn_r", 
            vmin=5, vmax=35, # Fix color scale across all 3 charts for fair comparison
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
        
    except FileNotFoundError:
        print(f"ERROR: Could not find file '{config['file']}'. Check your folder.")
    except KeyError:
        print(f"ERROR: Columns missing in '{config['file']}'. Check the CSV format.")

# Run the generator for all 3 files
print("--- Generating Science Fair Heatmaps ---")
sns.set_context("talk") # Makes fonts larger for posters
for data in datasets:
    generate_heatmap(data)
print("Done!")