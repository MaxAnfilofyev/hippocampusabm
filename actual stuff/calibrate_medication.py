import numpy as np
import pandas as pd
import time
# IMPORT YOUR MODEL
# Ensure your main model file is named 'ad_model.py'
try:
    from try_again import ADModel
except ImportError:
    print("❌ ERROR: Could not import ADModel.")
    print("Please make sure your main model code is saved in a file named 'ad_model.py'")
    exit()

def run_single_trial(name, genotype, lec_dose, cro_dose, steps=1200):
    """
    Runs a single simulation and returns the final state.
    """
    print(f"   Running {name}...")
    print(f"   [Genotype: {genotype} | Lecanemab: {lec_dose} | Cromolyn: {cro_dose}]")
    
    start_time = time.time()
    
    # Initialize model
    model = ADModel(
        width=40, height=40, 
        n_neurons=1000, n_microglia=200,
        genotype=genotype,
        lecanemab_dose=lec_dose,
        cromolyn_dose=cro_dose,
        max_steps=steps,
        seed=374281 # Fixed seed for consistency
    )
    
    # Run simulation
    for _ in range(steps):
        model.step()
        
    # Extract Data
    data = model.datacollector.get_model_vars_dataframe()
    final_atrophy = data.iloc[-1]["Simulated Atrophy %"]
    final_cytokine = data.iloc[-1]["Total Cytokine"] # Ensure this exists in your DataCollector!
    
    duration = time.time() - start_time
    print(f"   -> Finished in {duration:.2f}s | Atrophy: {final_atrophy:.2f}% | Cytokine: {final_cytokine:.2f}")
    return final_atrophy, final_cytokine

def main():
    print("\n" + "="*70)
    print("🔬  MECHANISTIC LOGIC VERIFICATION PROTOCOL")
    print("="*70)

    # --- PHASE 1: BASELINE (The Control) ---
    print("\n🔹 PHASE 1: ESTABLISHING BASELINE")
    print("Goal: Determine natural disease progression without drugs.")
    
    base_atrophy, base_cytokine = run_single_trial("Baseline Run", "APOE3 (Control)", 0.0, 0.0)
    
    print(f"   ✅ Baseline established: {base_atrophy:.2f}% Atrophy")

    # --- PHASE 2: LECANEMAB CHECK (The Clearance Logic) ---
    print("\n🔹 PHASE 2: LECANEMAB EFFICACY CHECK")
    print("Goal: Verify 10mg/kg dose slows atrophy by ~27% (CLARITY-AD Trial).")
    
    lec_atrophy, _ = run_single_trial("Lecanemab Run", "APOE3 (Control)", 10.0, 0.0)
    
    # Math: % Reduction = (Baseline - Treated) / Baseline
    lec_impact = ((base_atrophy - lec_atrophy) / base_atrophy) * 100
    
    print(f"   RESULTS: Baseline: {base_atrophy:.2f}% -> Treated: {lec_atrophy:.2f}%")
    print(f"   IMPACT:  {lec_impact:.2f}% reduction in neurodegeneration.")
    
    if 24.0 <= lec_impact <= 30.0:
        print("   STATUS:  🟢 PASS (Matches CLARITY-AD data)")
    elif lec_impact < 24.0:
        print("   STATUS:  🔴 FAIL (Too Weak - Increase 'lecanemab_clearance_multiplier' coefficient)")
    else:
        print("   STATUS:  🔴 FAIL (Too Strong - Decrease 'lecanemab_clearance_multiplier' coefficient)")

    # --- PHASE 3: CROMOLYN CHECK (The Inflammation Logic) ---
    print("\n🔹 PHASE 3: CROMOLYN MECHANISM CHECK")
    print("Goal: Verify 100mg/kg dose reduces Total Cytokine by ~50% (Hori et al. Mouse Data).")
    print("Note: Using APOE4 genotype to generate high inflammation for clearer signal.")
    
    # First, get High-Risk Baseline
    print("   ... Establishing High-Inflammation Baseline (APOE 4/4) ...")
    high_risk_atrophy, high_risk_cytokine = run_single_trial("APOE4 Baseline", "APOE 4/4", 0.0, 0.0)
    
    # Now treat with Cromolyn
    cro_atrophy, cro_cytokine = run_single_trial("Cromolyn Run", "APOE 4/4", 0.0, 100.0)
    
    # Math: % Reduction in Cytokine
    cro_impact = ((high_risk_cytokine - cro_cytokine) / high_risk_cytokine) * 100
    
    print(f"   RESULTS: Baseline Cytokine: {high_risk_cytokine:.2f} -> Treated: {cro_cytokine:.2f}")
    print(f"   IMPACT:  {cro_impact:.2f}% reduction in Neuroinflammation.")
    
    if 45.0 <= cro_impact <= 55.0:
        print("   STATUS:  🟢 PASS (Matches Mouse Model data)")
    elif cro_impact < 45.0:
        print("   STATUS:  🔴 FAIL (Too Weak - Check 'cromolyn_cytokine_multiplier' exponential decay)")
    else:
        print("   STATUS:  🔴 FAIL (Too Strong - Check 'cromolyn_cytokine_multiplier' exponential decay)")

    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()