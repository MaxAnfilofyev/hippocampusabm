import numpy as np
import time
import warnings
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# --- IMPORT YOUR MODEL ---
try:
    from try_again import ADModel
except ImportError:
    print("❌ CRITICAL ERROR: Could not import ADModel from try_again.py")
    exit()

# Filter out warnings from SKOPT to keep output clean
warnings.filterwarnings("ignore")

# ==========================================
# 1. ESTABLISH GLOBAL BASELINE
# ==========================================
print("\n📊 ESTABLISHING BASELINE (Running Control sim first)...")

def get_baseline_atrophy():
    """Runs APOE3 Control (No Drugs) to get base atrophy."""
    model = ADModel(width=40, height=40, n_neurons=1000, n_microglia=200, 
                    genotype="APOE3 (Control)", seed=374281, max_steps=800)
    for _ in range(800): model.step()
    return model.datacollector.get_model_vars_dataframe().iloc[-1]["Simulated Atrophy %"]

# Execute Baseline once
BASE_ATROPHY = get_baseline_atrophy()

print(f"   ✅ Baseline Atrophy (APOE3):   {BASE_ATROPHY:.2f}%")
print("="*60)

# ==========================================
# 2. DEFINE SEARCH SPACE (LECANEMAB ONLY)
# ==========================================
space = [
    # Search for Lecanemab Potency (multiplier strength)
    # Range: 0.01 (Weak) to 5.0 (Very Strong)
    Real(0.01, 5.0, name='lecanemab_potency')
]

# ==========================================
# 3. OBJECTIVE FUNCTION
# ==========================================
@use_named_args(space)
def objective(**params):
    """
    The 'Loss Function' the AI tries to minimize.
    Loss = Distance from Perfect CLARITY-AD Result (27% Reduction)
    """
    l_potency = params['lecanemab_potency']
    
    # --- SIMULATION: LECANEMAB CHECK ---
    # Goal: 27% Reduction in Atrophy vs Baseline
    # Config: APOE3, Lec Dose 10.0, Cro Dose 0.0 (Ignored)
    model_lec = ADModel(
        width=40, height=40, n_neurons=1000, n_microglia=200,
        genotype="APOE3 (Control)",
        lecanemab_dose=10.0, lecanemab_potency=l_potency,
        cromolyn_dose=0.0, cromolyn_potency=0.0, # Ignored
        seed=374281, max_steps=800
    )
    for _ in range(800): model_lec.step()
    
    final_atrophy = model_lec.datacollector.get_model_vars_dataframe().iloc[-1]["Simulated Atrophy %"]
    
    # Calculate % Reduction
    # Formula: (Baseline - Treated) / Baseline * 100
    if BASE_ATROPHY > 0:
        lec_reduction = ((BASE_ATROPHY - final_atrophy) / BASE_ATROPHY) * 100
    else:
        lec_reduction = 0.0
    
    # Loss: How far off are we from 27%?
    # We square the error to penalize large deviations more
    loss = (lec_reduction - 27.0) ** 2

    print(f"   🔎 Potency: {l_potency:.3f} -> Reduction: {lec_reduction:.1f}% -> Loss: {loss:.2f}")
    
    return loss

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    print("🚀 STARTING LECANEMAB-ONLY OPTIMIZATION...")
    print("   Target: ~27% Atrophy reduction (CLARITY-AD)")
    
    # Gaussian Process Minimization
    res = gp_minimize(
        objective, 
        space, 
        n_calls=20,      # Reduced to 20 since we only have 1 variable to find
        random_state=42, 
        verbose=False 
    )

    print("\n" + "="*60)
    print(f"✅ OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"🏆 Best Loss Score:       {res.fun:.5f}")
    print("-" * 40)
    print(f"1️⃣  Recommended 'lecanemab_potency':  {res.x[0]:.5f}")
    print("="*60)
    print("👉 Update your 'try_again.py' with this value!")