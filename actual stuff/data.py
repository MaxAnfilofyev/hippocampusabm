import pandas as pd
import numpy as np
from pathlib import Path

# SETUP
DATA_FOLDER = Path(r"C:\Users\narwh\Downloads\Science Fair\raw_data")

# 1. LOAD ALL DATASETS
apoe_df = pd.read_csv(DATA_FOLDER / "APOERES_02Jan2026.csv")
amyloid_df = pd.read_csv(DATA_FOLDER / "UCBERKELEY_AMY_6MM_02Jan2026.csv")
markers_df = pd.read_csv(DATA_FOLDER / "UPENN_PLASMA_FUJIREBIO_QUANTERIX_02Jan2026.csv")
volume_df = pd.read_csv(DATA_FOLDER / "ADNI_PICSLASHS_02Jan2026.csv")
wmh_df = pd.read_csv(DATA_FOLDER / "UCD_WMH_02Jan2026.csv") # Added back

# 2. PRE-PROCESSING FUNCTION
def clean_adni_columns(df):
    if 'RID' in df.columns:
        df['RID'] = df['RID'].astype(int)
    # CRITICAL: Align timelines using VISCODE2
    if 'VISCODE2' in df.columns:
        df = df.rename(columns={'VISCODE': 'VISCODE_RAW', 'VISCODE2': 'VISCODE'})
    return df

# Apply cleaning
apoe_df = clean_adni_columns(apoe_df)
amyloid_df = clean_adni_columns(amyloid_df)
markers_df = clean_adni_columns(markers_df)
volume_df = clean_adni_columns(volume_df)
wmh_df = clean_adni_columns(wmh_df)

# 3. MERGE DATA
# Start with Volume (Neurodegeneration) as base
master_df = volume_df[['RID', 'VISCODE', 'LEFT_HIPP_VOL', 'RIGHT_HIPP_VOL', 'ICV']].copy()

# Merge Amyloid (Centiloids)
master_df = pd.merge(master_df, amyloid_df[['RID', 'VISCODE', 'CENTILOIDS']], on=['RID', 'VISCODE'], how='left')

# Merge Neuroinflammation (GFAP)
master_df = pd.merge(master_df, markers_df[['RID', 'VISCODE', 'GFAP_Q']], on=['RID', 'VISCODE'], how='left')

# Merge Vascular Injury (WMH) - Keeping "TOTAL_WMH"
# Note: WMH is often log-transformed in studies, but we will keep raw for now
master_df = pd.merge(master_df, wmh_df[['RID', 'VISCODE', 'TOTAL_WMH']], on=['RID', 'VISCODE'], how='left')

# Merge Genotype (Static)
master_df = pd.merge(master_df, apoe_df[['RID', 'GENOTYPE']], on='RID', how='left')
master_df['GENOTYPE'] = master_df.groupby('RID')['GENOTYPE'].ffill().bfill()

# 4. CALCULATE METRICS
# aHV (Adjusted Hippocampal Volume)
master_df['Total_Hipp_Vol'] = master_df['LEFT_HIPP_VOL'] + master_df['RIGHT_HIPP_VOL']
master_df['aHV'] = master_df['Total_Hipp_Vol'] / master_df['ICV']

# 5. PARSE TIME (Years from Baseline)
def parse_time(code):
    if code == 'bl' or code == 'sc': return 0.0
    if str(code).startswith('m'):
        try:
            return float(code[1:]) / 12.0
        except:
            return np.nan
    return np.nan

master_df['Years'] = master_df['VISCODE'].apply(parse_time)

# 6. EXPORT
# Remove rows that have NO time data or NO biological data at all
master_df = master_df.dropna(subset=['Years', 'aHV'])
master_df.to_csv("ADNI_Integrated_Master.csv", index=False)
print("Dataset ready. Includes: Amyloid, Hippocampus, GFAP, and WMH.")