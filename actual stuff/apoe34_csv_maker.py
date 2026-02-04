import pandas as pd
import re

# 1. THE RAW DATA STRING
# (Paste your output exactly as provided inside this triple-quoted string)
raw_log_data = """
[Run 1/125] APOE 3/4 | Lec: 0.0 | Cro: 0.0 | Rep: 1 --> Atrophy: 11.30%
[Run 2/125] APOE 3/4 | Lec: 0.0 | Cro: 0.0 | Rep: 2 --> Atrophy: 6.80%
[Run 3/125] APOE 3/4 | Lec: 0.0 | Cro: 0.0 | Rep: 3 --> Atrophy: 12.00%
[Run 4/125] APOE 3/4 | Lec: 0.0 | Cro: 0.0 | Rep: 4 --> Atrophy: 15.20%
[Run 5/125] APOE 3/4 | Lec: 0.0 | Cro: 0.0 | Rep: 5 --> Atrophy: 9.40%
[Run 6/125] APOE 3/4 | Lec: 0.0 | Cro: 10.0 | Rep: 1 --> Atrophy: 9.10%
[Run 7/125] APOE 3/4 | Lec: 0.0 | Cro: 10.0 | Rep: 2 --> Atrophy: 13.90%
[Run 8/125] APOE 3/4 | Lec: 0.0 | Cro: 10.0 | Rep: 3 --> Atrophy: 12.60%
[Run 9/125] APOE 3/4 | Lec: 0.0 | Cro: 10.0 | Rep: 4 --> Atrophy: 13.20%
[Run 10/125] APOE 3/4 | Lec: 0.0 | Cro: 10.0 | Rep: 5 --> Atrophy: 9.80%
[Run 11/125] APOE 3/4 | Lec: 0.0 | Cro: 20.0 | Rep: 1 --> Atrophy: 13.00%
[Run 12/125] APOE 3/4 | Lec: 0.0 | Cro: 20.0 | Rep: 2 --> Atrophy: 5.90%
[Run 13/125] APOE 3/4 | Lec: 0.0 | Cro: 20.0 | Rep: 3 --> Atrophy: 8.20%
[Run 14/125] APOE 3/4 | Lec: 0.0 | Cro: 20.0 | Rep: 4 --> Atrophy: 18.00%
[Run 15/125] APOE 3/4 | Lec: 0.0 | Cro: 20.0 | Rep: 5 --> Atrophy: 14.00%
[Run 16/125] APOE 3/4 | Lec: 0.0 | Cro: 30.0 | Rep: 1 --> Atrophy: 9.40%
[Run 17/125] APOE 3/4 | Lec: 0.0 | Cro: 30.0 | Rep: 2 --> Atrophy: 6.70%
[Run 18/125] APOE 3/4 | Lec: 0.0 | Cro: 30.0 | Rep: 3 --> Atrophy: 8.80%
[Run 19/125] APOE 3/4 | Lec: 0.0 | Cro: 30.0 | Rep: 4 --> Atrophy: 20.00%
[Run 20/125] APOE 3/4 | Lec: 0.0 | Cro: 30.0 | Rep: 5 --> Atrophy: 10.30%
[Run 21/125] APOE 3/4 | Lec: 0.0 | Cro: 40.0 | Rep: 1 --> Atrophy: 9.50%
[Run 22/125] APOE 3/4 | Lec: 0.0 | Cro: 40.0 | Rep: 2 --> Atrophy: 17.10%
[Run 23/125] APOE 3/4 | Lec: 0.0 | Cro: 40.0 | Rep: 3 --> Atrophy: 27.70%
[Run 24/125] APOE 3/4 | Lec: 0.0 | Cro: 40.0 | Rep: 4 --> Atrophy: 13.60%
[Run 25/125] APOE 3/4 | Lec: 0.0 | Cro: 40.0 | Rep: 5 --> Atrophy: 6.50%
[Run 26/125] APOE 3/4 | Lec: 2.5 | Cro: 0.0 | Rep: 1 --> Atrophy: 6.80%
[Run 27/125] APOE 3/4 | Lec: 2.5 | Cro: 0.0 | Rep: 2 --> Atrophy: 7.20%
[Run 28/125] APOE 3/4 | Lec: 2.5 | Cro: 0.0 | Rep: 3 --> Atrophy: 21.40%
[Run 29/125] APOE 3/4 | Lec: 2.5 | Cro: 0.0 | Rep: 4 --> Atrophy: 6.40%
[Run 30/125] APOE 3/4 | Lec: 2.5 | Cro: 0.0 | Rep: 5 --> Atrophy: 16.00%
[Run 31/125] APOE 3/4 | Lec: 2.5 | Cro: 10.0 | Rep: 1 --> Atrophy: 9.70%
[Run 32/125] APOE 3/4 | Lec: 2.5 | Cro: 10.0 | Rep: 2 --> Atrophy: 7.80%
[Run 33/125] APOE 3/4 | Lec: 2.5 | Cro: 10.0 | Rep: 3 --> Atrophy: 8.00%
[Run 34/125] APOE 3/4 | Lec: 2.5 | Cro: 10.0 | Rep: 4 --> Atrophy: 10.80%
[Run 35/125] APOE 3/4 | Lec: 2.5 | Cro: 10.0 | Rep: 5 --> Atrophy: 8.80%
[Run 36/125] APOE 3/4 | Lec: 2.5 | Cro: 20.0 | Rep: 1 --> Atrophy: 15.60%
[Run 37/125] APOE 3/4 | Lec: 2.5 | Cro: 20.0 | Rep: 2 --> Atrophy: 25.50%
[Run 38/125] APOE 3/4 | Lec: 2.5 | Cro: 20.0 | Rep: 3 --> Atrophy: 6.10%
[Run 39/125] APOE 3/4 | Lec: 2.5 | Cro: 20.0 | Rep: 4 --> Atrophy: 6.60%
[Run 40/125] APOE 3/4 | Lec: 2.5 | Cro: 20.0 | Rep: 5 --> Atrophy: 10.40%
[Run 41/125] APOE 3/4 | Lec: 2.5 | Cro: 30.0 | Rep: 1 --> Atrophy: 9.00%
[Run 42/125] APOE 3/4 | Lec: 2.5 | Cro: 30.0 | Rep: 2 --> Atrophy: 7.30%
[Run 43/125] APOE 3/4 | Lec: 2.5 | Cro: 30.0 | Rep: 3 --> Atrophy: 6.50%
[Run 44/125] APOE 3/4 | Lec: 2.5 | Cro: 30.0 | Rep: 4 --> Atrophy: 8.70%
[Run 45/125] APOE 3/4 | Lec: 2.5 | Cro: 30.0 | Rep: 5 --> Atrophy: 14.00%
[Run 46/125] APOE 3/4 | Lec: 2.5 | Cro: 40.0 | Rep: 1 --> Atrophy: 10.40%
[Run 47/125] APOE 3/4 | Lec: 2.5 | Cro: 40.0 | Rep: 2 --> Atrophy: 12.30%
[Run 48/125] APOE 3/4 | Lec: 2.5 | Cro: 40.0 | Rep: 3 --> Atrophy: 8.60%
[Run 49/125] APOE 3/4 | Lec: 2.5 | Cro: 40.0 | Rep: 4 --> Atrophy: 11.10%
[Run 50/125] APOE 3/4 | Lec: 2.5 | Cro: 40.0 | Rep: 5 --> Atrophy: 5.80%
[Run 51/125] APOE 3/4 | Lec: 5.0 | Cro: 0.0 | Rep: 1 --> Atrophy: 6.70%
[Run 52/125] APOE 3/4 | Lec: 5.0 | Cro: 0.0 | Rep: 2 --> Atrophy: 10.00%
[Run 53/125] APOE 3/4 | Lec: 5.0 | Cro: 0.0 | Rep: 3 --> Atrophy: 27.40%
[Run 54/125] APOE 3/4 | Lec: 5.0 | Cro: 0.0 | Rep: 4 --> Atrophy: 7.60%
[Run 55/125] APOE 3/4 | Lec: 5.0 | Cro: 0.0 | Rep: 5 --> Atrophy: 11.70%
[Run 56/125] APOE 3/4 | Lec: 5.0 | Cro: 10.0 | Rep: 1 --> Atrophy: 11.40%
[Run 57/125] APOE 3/4 | Lec: 5.0 | Cro: 10.0 | Rep: 2 --> Atrophy: 29.30%
[Run 58/125] APOE 3/4 | Lec: 5.0 | Cro: 10.0 | Rep: 3 --> Atrophy: 7.90%
[Run 59/125] APOE 3/4 | Lec: 5.0 | Cro: 10.0 | Rep: 4 --> Atrophy: 6.00%
[Run 60/125] APOE 3/4 | Lec: 5.0 | Cro: 10.0 | Rep: 5 --> Atrophy: 14.40%
[Run 61/125] APOE 3/4 | Lec: 5.0 | Cro: 20.0 | Rep: 1 --> Atrophy: 11.60%
[Run 62/125] APOE 3/4 | Lec: 5.0 | Cro: 20.0 | Rep: 2 --> Atrophy: 7.20%
[Run 63/125] APOE 3/4 | Lec: 5.0 | Cro: 20.0 | Rep: 3 --> Atrophy: 18.60%
[Run 64/125] APOE 3/4 | Lec: 5.0 | Cro: 20.0 | Rep: 4 --> Atrophy: 7.80%
[Run 65/125] APOE 3/4 | Lec: 5.0 | Cro: 20.0 | Rep: 5 --> Atrophy: 5.70%
[Run 66/125] APOE 3/4 | Lec: 5.0 | Cro: 30.0 | Rep: 1 --> Atrophy: 4.70%
[Run 67/125] APOE 3/4 | Lec: 5.0 | Cro: 30.0 | Rep: 2 --> Atrophy: 11.70%
[Run 68/125] APOE 3/4 | Lec: 5.0 | Cro: 30.0 | Rep: 3 --> Atrophy: 5.40%
[Run 69/125] APOE 3/4 | Lec: 5.0 | Cro: 30.0 | Rep: 4 --> Atrophy: 14.80%
[Run 70/125] APOE 3/4 | Lec: 5.0 | Cro: 30.0 | Rep: 5 --> Atrophy: 12.60%
[Run 71/125] APOE 3/4 | Lec: 5.0 | Cro: 40.0 | Rep: 1 --> Atrophy: 11.00%
[Run 72/125] APOE 3/4 | Lec: 5.0 | Cro: 40.0 | Rep: 2 --> Atrophy: 17.00%
[Run 73/125] APOE 3/4 | Lec: 5.0 | Cro: 40.0 | Rep: 3 --> Atrophy: 16.20%
[Run 74/125] APOE 3/4 | Lec: 5.0 | Cro: 40.0 | Rep: 4 --> Atrophy: 15.00%
[Run 75/125] APOE 3/4 | Lec: 5.0 | Cro: 40.0 | Rep: 5 --> Atrophy: 14.10%
[Run 76/125] APOE 3/4 | Lec: 7.5 | Cro: 0.0 | Rep: 1 --> Atrophy: 13.90%
[Run 77/125] APOE 3/4 | Lec: 7.5 | Cro: 0.0 | Rep: 2 --> Atrophy: 38.50%
[Run 78/125] APOE 3/4 | Lec: 7.5 | Cro: 0.0 | Rep: 3 --> Atrophy: 6.10%
[Run 79/125] APOE 3/4 | Lec: 7.5 | Cro: 0.0 | Rep: 4 --> Atrophy: 10.80%
[Run 80/125] APOE 3/4 | Lec: 7.5 | Cro: 0.0 | Rep: 5 --> Atrophy: 15.70%
[Run 81/125] APOE 3/4 | Lec: 7.5 | Cro: 10.0 | Rep: 1 --> Atrophy: 11.80%
[Run 82/125] APOE 3/4 | Lec: 7.5 | Cro: 10.0 | Rep: 2 --> Atrophy: 24.30%
[Run 83/125] APOE 3/4 | Lec: 7.5 | Cro: 10.0 | Rep: 3 --> Atrophy: 9.30%
[Run 84/125] APOE 3/4 | Lec: 7.5 | Cro: 10.0 | Rep: 4 --> Atrophy: 9.50%
[Run 85/125] APOE 3/4 | Lec: 7.5 | Cro: 10.0 | Rep: 5 --> Atrophy: 6.40%
[Run 86/125] APOE 3/4 | Lec: 7.5 | Cro: 20.0 | Rep: 1 --> Atrophy: 7.70%
[Run 87/125] APOE 3/4 | Lec: 7.5 | Cro: 20.0 | Rep: 2 --> Atrophy: 6.40%
[Run 88/125] APOE 3/4 | Lec: 7.5 | Cro: 20.0 | Rep: 3 --> Atrophy: 7.20%
[Run 89/125] APOE 3/4 | Lec: 7.5 | Cro: 20.0 | Rep: 4 --> Atrophy: 11.20%
[Run 90/125] APOE 3/4 | Lec: 7.5 | Cro: 20.0 | Rep: 5 --> Atrophy: 12.30%
[Run 91/125] APOE 3/4 | Lec: 7.5 | Cro: 30.0 | Rep: 1 --> Atrophy: 10.80%
[Run 92/125] APOE 3/4 | Lec: 7.5 | Cro: 30.0 | Rep: 2 --> Atrophy: 15.80%
[Run 93/125] APOE 3/4 | Lec: 7.5 | Cro: 30.0 | Rep: 3 --> Atrophy: 8.90%
[Run 94/125] APOE 3/4 | Lec: 7.5 | Cro: 30.0 | Rep: 4 --> Atrophy: 11.50%
[Run 95/125] APOE 3/4 | Lec: 7.5 | Cro: 30.0 | Rep: 5 --> Atrophy: 28.50%
[Run 96/125] APOE 3/4 | Lec: 7.5 | Cro: 40.0 | Rep: 1 --> Atrophy: 6.30%
[Run 97/125] APOE 3/4 | Lec: 7.5 | Cro: 40.0 | Rep: 2 --> Atrophy: 9.80%
[Run 98/125] APOE 3/4 | Lec: 7.5 | Cro: 40.0 | Rep: 3 --> Atrophy: 9.00%
[Run 99/125] APOE 3/4 | Lec: 7.5 | Cro: 40.0 | Rep: 4 --> Atrophy: 9.40%
[Run 100/125] APOE 3/4 | Lec: 7.5 | Cro: 40.0 | Rep: 5 --> Atrophy: 8.60%
[Run 101/125] APOE 3/4 | Lec: 10.0 | Cro: 0.0 | Rep: 1 --> Atrophy: 7.20%
[Run 102/125] APOE 3/4 | Lec: 10.0 | Cro: 0.0 | Rep: 2 --> Atrophy: 29.40%
[Run 103/125] APOE 3/4 | Lec: 10.0 | Cro: 0.0 | Rep: 3 --> Atrophy: 7.70%
[Run 104/125] APOE 3/4 | Lec: 10.0 | Cro: 0.0 | Rep: 4 --> Atrophy: 23.40%
[Run 105/125] APOE 3/4 | Lec: 10.0 | Cro: 0.0 | Rep: 5 --> Atrophy: 25.00%
[Run 106/125] APOE 3/4 | Lec: 10.0 | Cro: 10.0 | Rep: 1 --> Atrophy: 4.90%
[Run 107/125] APOE 3/4 | Lec: 10.0 | Cro: 10.0 | Rep: 2 --> Atrophy: 8.70%
[Run 108/125] APOE 3/4 | Lec: 10.0 | Cro: 10.0 | Rep: 3 --> Atrophy: 10.90%
[Run 109/125] APOE 3/4 | Lec: 10.0 | Cro: 10.0 | Rep: 4 --> Atrophy: 22.70%
[Run 110/125] APOE 3/4 | Lec: 10.0 | Cro: 10.0 | Rep: 5 --> Atrophy: 5.90%
[Run 111/125] APOE 3/4 | Lec: 10.0 | Cro: 20.0 | Rep: 1 --> Atrophy: 9.10%
[Run 112/125] APOE 3/4 | Lec: 10.0 | Cro: 20.0 | Rep: 2 --> Atrophy: 10.10%
[Run 113/125] APOE 3/4 | Lec: 10.0 | Cro: 20.0 | Rep: 3 --> Atrophy: 11.10%
[Run 114/125] APOE 3/4 | Lec: 10.0 | Cro: 20.0 | Rep: 4 --> Atrophy: 13.70%
[Run 115/125] APOE 3/4 | Lec: 10.0 | Cro: 20.0 | Rep: 5 --> Atrophy: 12.40%
[Run 116/125] APOE 3/4 | Lec: 10.0 | Cro: 30.0 | Rep: 1 --> Atrophy: 10.10%
[Run 117/125] APOE 3/4 | Lec: 10.0 | Cro: 30.0 | Rep: 2 --> Atrophy: 7.20%
[Run 118/125] APOE 3/4 | Lec: 10.0 | Cro: 30.0 | Rep: 3 --> Atrophy: 6.80%
[Run 119/125] APOE 3/4 | Lec: 10.0 | Cro: 30.0 | Rep: 4 --> Atrophy: 16.10%
[Run 120/125] APOE 3/4 | Lec: 10.0 | Cro: 30.0 | Rep: 5 --> Atrophy: 9.50%
[Run 121/125] APOE 3/4 | Lec: 10.0 | Cro: 40.0 | Rep: 1 --> Atrophy: 22.80%
[Run 122/125] APOE 3/4 | Lec: 10.0 | Cro: 40.0 | Rep: 2 --> Atrophy: 8.90%
[Run 123/125] APOE 3/4 | Lec: 10.0 | Cro: 40.0 | Rep: 3 --> Atrophy: 10.40%
[Run 124/125] APOE 3/4 | Lec: 10.0 | Cro: 40.0 | Rep: 4 --> Atrophy: 6.70%
[Run 125/125] APOE 3/4 | Lec: 10.0 | Cro: 40.0 | Rep: 5 --> Atrophy: 8.80%
"""

# 2. PARSE THE LOGS
data = []
# This pattern extracts Lec, Cro, Rep, and Atrophy from each line
pattern = r"Lec: ([\d\.]+) \| Cro: ([\d\.]+) \| Rep: (\d+) --> Atrophy: ([\d\.]+)%"

for line in raw_log_data.split('\n'):
    match = re.search(pattern, line)
    if match:
        lec_dose = float(match.group(1))
        cro_dose = float(match.group(2))
        rep_num = int(match.group(3))
        atrophy = float(match.group(4))
        
        data.append({
            "Genotype": "APOE 3/4",  # Hardcoded since your log is all 3/4
            "Lecanemab_mg_kg": lec_dose,
            "Cromolyn_mg_kg": cro_dose,
            "Replicate": rep_num,
            "Final_Atrophy_Percent": atrophy
        })

# 3. CREATE DATAFRAMES
df_raw = pd.DataFrame(data)

# Create Averages Summary
df_summary = df_raw.groupby(["Genotype", "Lecanemab_mg_kg", "Cromolyn_mg_kg"]).agg({
    "Final_Atrophy_Percent": ["mean", "std"]
}).reset_index()

# Flatten MultiIndex columns
df_summary.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_summary.columns.values]

# 4. SAVE TO CSV
raw_filename = "science_fair_raw_apoe34.csv"
summary_filename = "science_fair_averages_apoe34.csv"

df_raw.to_csv(raw_filename, index=False)
df_summary.to_csv(summary_filename, index=False)

print(f"Success! Created {raw_filename} and {summary_filename}")
print("Preview of Averages:")
print(df_summary.head())