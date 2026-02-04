import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

# Load ADNI data
df = pd.read_csv("ADNI_Integrated_Master.csv")

# Clean
df = df.dropna(subset=["CENTILOIDS", "GFAP_Q", "aHV", "Years", "GENOTYPE"])

from statsmodels.nonparametric.smoothers_lowess import lowess

def smooth_signal(x, frac=0.3):
    t = np.arange(len(x))
    return lowess(x, t, frac=frac, return_sorted=False)

calibration = {}

def collapse_by_year(years, values):
    years = np.array(years)
    values = np.array(values)

    unique_years = np.unique(years)
    collapsed = []

    for y in unique_years:
        collapsed.append(values[years == y].mean())

    return unique_years, np.array(collapsed)
# Process each genotype separately

for genotype in df["GENOTYPE"].unique():
    sub = df[df["GENOTYPE"] == genotype].sort_values("Years")

    years = sub["Years"].values

    # ===============================
    # 1️⃣ AMYLOID: dynamic derivative
    # ===============================
    amy_smooth = lowess(
        sub["CENTILOIDS"].values,
        years,
        frac=0.4,
        return_sorted=True
    )

    amy_years = amy_smooth[:, 0]
    yrs, amy = collapse_by_year(years, sub["CENTILOIDS"].values)

    amy_vals = smooth_signal(amy)
    dC_dt = np.gradient(amy_vals, yrs)

    dC_dt = np.clip(dC_dt, 0, None)
    amyloid_rate = (dC_dt / 100.0).tolist()


    # Enforce biological constraint
    dC_dt = np.clip(dC_dt, 0, None)

    # Normalize to ABM space
    amyloid_rate = (dC_dt / 100.0).tolist()

    # ===============================
    # 2️⃣ GFAP: smoothed state (NOT derivative)
    # ===============================
    gfap_smooth = lowess(
        sub["GFAP_Q"].values,
        years,
        frac=0.5,
        return_sorted=True
    )

    gfap_vals = gfap_smooth[:, 1]
    gfap_norm = (gfap_vals / gfap_vals.max()).tolist()

    # ===============================
    # 3️⃣ aHV: atrophy slope (validation only)
    # ===============================
    hv_smooth = lowess(
        sub["aHV"].values,
        years,
        frac=0.5,
        return_sorted=True
    )

    yrs, hv = collapse_by_year(years, sub["aHV"].values)

    hv_vals = smooth_signal(hv)
    atrophy_rate = abs(np.gradient(hv_vals, yrs)).mean()

    # ===============================
    # STORE
    # ===============================
    calibration[genotype] = {
        "years": amy_years.tolist(),
        "amyloid_rate": amyloid_rate,
        "gfap_state": gfap_norm,
        "atrophy_per_year": atrophy_rate
    }

print(calibration)
