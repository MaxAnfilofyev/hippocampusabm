import numpy as np
import time
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from try_again import ADModel  # Ensure your model file is named try_again.py

# --- 1. SETUP TARGETS (Exact same as Scipy script) ---
def sigmoid(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

# APOE3 Target Parameters
L_3, k_3, t0_3 = 100.0, 0.2587, 27.72
years_to_simulate = np.arange(1, 21)
TARGET_ATROPHY_APOE3 = sigmoid(years_to_simulate, L_3, k_3, t0_3)

print("🎯 Target Curve (APOE3) Loaded.")

# --- 2. DEFINE SEARCH SPACE ---
# We use the same 3 variables as the Scipy script for a fair comparison.
space = [
    Real(0.0, 50.0, name='amyloid_toxicity'),
    Real(0.0, 50.0, name='cytokine_toxicity'),
    Real(0.05, 0.60, name='neuron_death_threshold')
]

# --- 3. OBJECTIVE FUNCTION ---
@use_named_args(space)
def objective(**params):
    
    # 1. Unpack Parameters
    a_tox = params['amyloid_toxicity']
    c_tox = params['cytokine_toxicity']
    d_thresh = params['neuron_death_threshold']
    
    try:
        # 2. Initialize Model
        # Note: All genetic params stay at Baseline (1.0)
        model = ADModel(
            width=20, height=20, 
            n_neurons=200, n_microglia=40, 
            genotype="APOE3 (Control)",
            amyloid_toxicity=a_tox,
            cytokine_toxicity=c_tox,
            neuron_death_threshold=d_thresh,
            max_steps=1000 
        )
        
        simulated_atrophy = []
        
        # 3. Run Simulation (20 Years)
        for year in years_to_simulate:
            for _ in range(50): # 50 steps = 1 year
                model.step()
            
            val = model.get_atrophy_percent()
            simulated_atrophy.append(val)
            
            # Crash protection
            if val > 99.0 and year < 5:
                return 10000.0 # Huge penalty
        
        # 4. Calculate MSE
        simulated_array = np.array(simulated_atrophy)
        error = np.mean((simulated_array - TARGET_ATROPHY_APOE3)**2)
        
        return error

    except Exception:
        return 10000.0

# --- 4. EXECUTION ---
if __name__ == "__main__":
    print(f"🚀 Starting Bayesian Optimization (Targeting Sigmoid Curve)...")
    start_time = time.time()

    # n_calls=30 is usually enough for just 3 variables
    res = gp_minimize(
        objective, 
        space, 
        n_calls=30, 
        random_state=42, 
        verbose=True
    )

    print("\n" + "="*50)
    print(f"✅ SKOPT OPTIMIZATION COMPLETE ({time.time() - start_time:.1f}s)")
    print("="*50)
    print(f"Best Error (MSE):           {res.fun:.5f}")
    print("-" * 30)
    print(f"Optimal Amyloid Toxicity:   {res.x[0]:.4f}")
    print(f"Optimal Cytokine Toxicity:  {res.x[1]:.4f}")
    print(f"Optimal Death Threshold:    {res.x[2]:.4f}")