import numpy as np
from scipy.optimize import minimize, Bounds
from try_again import ADModel 
import sys

# --- 1. SETUP TARGETS (Sigmoid Curve) ---
def sigmoid(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

# APOE3 Target Parameters
L_3, k_3, t0_3 = 100.0, 0.2587, 27.72
years_to_simulate = np.arange(1, 21)
TARGET_ATROPHY_APOE3 = sigmoid(years_to_simulate, L_3, k_3, t0_3)

print("🎯 Target Curve Loaded.")

# --- 2. OPTIMIZATION FUNCTION ---
iteration_count = 0

def run_simulation_and_get_error(params):
    global iteration_count
    iteration_count += 1
    
    # 1. Force Positive Values (The "Bounds" Trick)
    # Even if the optimizer tries -5.0, we force it to 0.0
    a_tox = max(0.0, params[0])
    c_tox = max(0.0, params[1])
    d_thresh = np.clip(params[2], 0.01, 0.99) # Keep threshold between 1% and 99%
    
    try:
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
        
        # Run 20 Years (50 steps per year)
        for year in years_to_simulate:
            for _ in range(50): 
                model.step()
            
            val = model.get_atrophy_percent()
            simulated_atrophy.append(val)
            
            # EMERGENCY BRAKE: If model explodes to 100% loss instantly, stop.
            if val > 99.0 and year < 5:
                return 10000.0 # Return huge error
        
        # Calculate Error
        simulated_array = np.array(simulated_atrophy)
        error = np.mean((simulated_array - TARGET_ATROPHY_APOE3)**2)
        
        # Simple Progress Bar
        print(f"Iter {iteration_count:03d} | Amy: {a_tox:.2f} | Cyto: {c_tox:.2f} | Thresh: {d_thresh:.2f} | Error: {error:.2f}   ", end="\r")
        return error

    except Exception as e:
        return 10000.0

# --- 3. RUN WITH BOUNDS ---
print("\n🚀 Starting Robust Optimization...")

# Initial Guess
x0 = [5.0, 5.0, 0.25]

# Bounds: (min, max) for each parameter
# Amyloid Tox: 0-50, Cytokine Tox: 0-50, Threshold: 0.05-0.5
bounds = ((0.0, 50.0), (0.0, 50.0), (0.05, 0.5))

# We use 'Powell' method because it is very robust for simple curve fitting
result = minimize(
    run_simulation_and_get_error, 
    x0, 
    method='Powell', 
    bounds=bounds, 
    tol=1.0
)

print("\n" + "="*50)
print("✅ OPTIMIZATION COMPLETE")
print(f"Optimal Amyloid Toxicity:   {result.x[0]:.4f}")
print(f"Optimal Cytokine Toxicity:  {result.x[1]:.4f}")
print(f"Optimal Death Threshold:    {result.x[2]:.4f}")
print(f"Final Error:                {result.fun:.4f}")