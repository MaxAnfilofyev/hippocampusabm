import numpy as np
import time
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from pls import ADModel  # Make sure this matches your model filename (pls.py or try_again.py)

# --- 1. SETUP TARGETS ---
def sigmoid(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

# APOE3 Target (Control)
L_3, k_3, t0_3 = 100.0, 0.2587, 27.72
years_to_simulate = np.arange(1, 21)
TARGET_ATROPHY_APOE3 = sigmoid(years_to_simulate, L_3, k_3, t0_3)

print("🎯 Target Curve (APOE3) Loaded.")

# --- 2. EXPANDED SEARCH SPACE (5 Variables) ---
space = [
    # 1. Toxicity Variables
    Real(1.0, 20.0, name='amyloid_toxicity'),
    Real(10.0, 60.0, name='cytokine_toxicity'),
    
    # 2. Biological Sensitivity
    Real(0.15, 0.25, name='neuron_death_threshold'),
    
    # 3. NEW: Plaque Feedback (How fast amyloid spreads)
    Real(0.05, 0.15, name='plaque_feedback_rate'),
    
    # 4. NEW: Time Constant (The "Volume Knob" for death prob)
    # Range: 0.0005 (Conservative) to 0.002 (Aggressive)
    Real(0.0005, 0.002, name='time_constant') 
]

# --- 3. OBJECTIVE FUNCTION ---
@use_named_args(space)
def objective(**params):
    
    try:
        model = ADModel(
            width=20, height=20, 
            n_neurons=200, n_microglia=40, 
            genotype="APOE3 (Control)",
            
            # Pass all 5 optimized parameters
            amyloid_toxicity=params['amyloid_toxicity'],
            cytokine_toxicity=params['cytokine_toxicity'],
            neuron_death_threshold=params['neuron_death_threshold'],
            plaque_feedback_rate=params['plaque_feedback_rate'],
            time_constant=params['time_constant'],
            
            max_steps=1000 
        )
        
        simulated_atrophy = []
        
        for year in years_to_simulate:
            for _ in range(50): 
                model.step()
            
            val = model.get_atrophy_percent()
            simulated_atrophy.append(val)
            
            # Safety Brake: >95% death in 5 years = Explosion
            if val > 95.0 and year < 5:
                return 1000.0 
        
        simulated_array = np.array(simulated_atrophy)
        error = np.mean((simulated_array - TARGET_ATROPHY_APOE3)**2)
        
        # Penalize undershooting (If final atrophy < 10%, punish)
        # This forces the model to use the "Time Constant" knob to raise the curve.
        if simulated_array[-1] < 10.0: 
            return error + 500.0
            
        return error

    except Exception:
        return 1000.0

# --- 4. EXECUTION ---
if __name__ == "__main__":
    print(f"🚀 Starting 5-Variable Optimization...")
    start_time = time.time()

    # n_calls=50 is recommended for 5 variables to find the sweet spot
    res = gp_minimize(
        objective, 
        space, 
        n_calls=50, 
        random_state=42, 
        verbose=True
    )

    print("\n" + "="*50)
    print(f"✅ OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Best Error (MSE):           {res.fun:.5f}")
    print("-" * 30)
    print(f"Optimal Amyloid Toxicity:   {res.x[0]:.4f}")
    print(f"Optimal Cytokine Toxicity:  {res.x[1]:.4f}")
    print(f"Optimal Death Threshold:    {res.x[2]:.4f}")
    print(f"Optimal Plaque Feedback:    {res.x[3]:.4f}")
    print(f"Optimal Time Constant:      {res.x[4]:.6f}")