import pandas as pd
import numpy as np

# Load datasets
apoe_df = pd.read_csv("raw_data\APOERES_02Jan2026.csv")
amyloid_df = pd.read_csv("raw_data\UCBERKELEY_AMY_6MM_02Jan2026.csv")
fdg_df = pd.read_csv("raw_data\UCBERKELEYFDG_8mm_02_17_23_02Jan2026.csv")
wmh_df = pd.read_csv("raw_data\UCD_WMH_02Jan2026.csv")
volume_df = pd.read_csv("raw_data\ADNI_PICSLASHS_02Jan2026.csv")
markers_df = pd.read_csv("raw_data\UPENN_PLASMA_FUJIREBIO_QUANTERIX_02Jan2026.csv")
anti_amyloid_df = pd.read_csv("raw_data\ANTIAMYTX_02Jan2026.csv")

# Standardize column names
def preprocess_adni(df, name):
    # Ensure RID is integer for consistent matching
    if 'RID' in df.columns:
        df['RID'] = df['RID'].astype(int)
    # Standardize visit code column name if needed
    if 'VISCODE2' in df.columns:
        df = df.rename(columns={'VISCODE2': 'VISCODE'})
    return df

dfs = {
    "apoe": apoe_df, "amyloid": amyloid_df, "fdg": fdg_df, 
    "wmh": wmh_df, "volume": volume_df, "markers": markers_df, 
    "anti_amy": anti_amyloid_df
}

processed_dfs = {k: preprocess_adni(v, k) for k, v in dfs.items()}

# 2. Sequential Outer Merge
# We use an 'outer' join first to keep all possible data points.
# You can switch to 'inner' later if you only want rows with 100% complete data.
master_df = processed_dfs["apoe"]
for name in ["amyloid", "fdg", "wmh", "volume", "markers", "anti_amy"]:
    master_df = pd.merge(
        master_df, 
        processed_dfs[name], 
        on=['RID', 'VISCODE'], 
        how='outer',
        suffixes=('', f'_{name}')
    )

# 3. Handling Static Data (Forward Fill)
# APOE4 status is static. If it's missing for a visit, we copy it from 
# the patient's other visits.
static_columns = ['APGEN1', 'APGEN2'] # Common APOE column names
for col in static_columns:
    if col in master_df.columns:
        master_df[col] = master_df.groupby('RID')[col].ffill().bfill()

# 4. Feature Engineering for your Research Plan
# Calculate the T-tau / A-beta ratio mentioned in Toledo et al.
if 'TAU' in master_df.columns and 'ABETA' in master_df.columns:
    # Ensure numeric conversion
    master_df['TAU'] = pd.to_numeric(master_df['TAU'], errors='coerce')
    master_df['ABETA'] = pd.to_numeric(master_df['ABETA'], errors='coerce')
    master_df['TAU_ABETA_RATIO'] = master_df['TAU'] / master_df['ABETA']

# 5. Final Cleanup
# Sort by Patient and Visit to ensure longitudinal order for your RNN
master_df = master_df.sort_values(by=['RID', 'VISCODE'])

# Export for your RNN model
master_df.to_csv("ADNI_Integrated_Master.csv", index=False)

print(f"Merge Complete. Final shape: {master_df.shape}")
print(f"Total Unique Patients: {master_df['RID'].nunique()}")