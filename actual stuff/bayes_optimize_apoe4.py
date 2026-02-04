import numpy as np
import time
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# IMPORTS FROM YOUR MODEL FILE
try:
    # Replace 'try_again' with your actual filename if different
    from try_again import ADModel
except ImportError:
    pass 

# --- 1. SETUP TARGETS (APOE 4/4) ---
def sigmoid(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

# Target Parameters for APOE 4/4
L_4, k_4, t0_4 = 100.0, 0.1539, 26.7703

# Simulate 20 years
years_to_simulate = np.arange(1, 21)
TARGET_ATROPHY_APOE4 = sigmoid(years_to_simulate, L_4, k_4, t0_4)

print("🎯 Target Curve (APOE 4/4) Loaded.")

# --- 2. RESTRICTED SEARCH SPACE ---
# Ranges set strictly according to your specifications
space = [
    # 1. amyloid_production >= 1.0
    # (Setting cap at 2.5 to prevent infinite runaway)
    Real(1.0, 2.5, name='amyloid_production'),
    
    # 2. amyloid_clearance <= 0.6
    # (Setting floor at 0.05 so it doesn't hit 0 and crash)
    Real(0.05, 0.6, name='amyloid_clearance'),
    
    # 3. chemotaxis_efficiency between 0 and 1
    Real(0.0, 1.0, name='chemotaxis_efficiency'),
    
    # 4. inflammatory_threshold < 1
    # (Setting floor at 0.1 to avoid instant inflammation)
    Real(0.1, 0.99, name='inflammatory_threshold'),
    
    # 5. proliferation_rate < 1
    Real(0.1, 0.99, name='proliferation_rate'),
    
    # 6. aggregation_rate > 1.5
    # (Setting cap at 6.0)
    Real(1.51, 6.0, name='aggregation_rate'),
]

# --- 3. OBJECTIVE FUNCTION ---
@use_named_args(space)
def objective(**params):
    
    # Create the override dictionary
    genetic_overrides = {
        "amyloid_production": params['amyloid_production'],
        "amyloid_clearance": params['amyloid_clearance'],
        "chemotaxis_efficiency": params['chemotaxis_efficiency'],
        "inflammatory_threshold": params['inflammatory_threshold'],
        "proliferation_rate": params['proliferation_rate'],
        "aggregation_rate": params['aggregation_rate']
    }

    try:
        # Scale down for speed: 200 neurons, 40 microglia
        model = ADModel(
            width=20, height=20, 
            n_neurons=200, n_microglia=40, 
            genotype="APOE 4/4",
            genetic_overrides=genetic_overrides,
            
            # Mechanism params fixed
            neuron_death_threshold=0.004174, 
            max_steps=1000, # 20 years
            seed=42
        )
        
        simulated_atrophy = []
        
        # Run simulation in year-long chunks
        for year in years_to_simulate:
            # 50 steps = 1 year (based on your 0.02 years/step)
            for _ in range(50): 
                model.step()
            
            val = model.get_atrophy_percent()
            simulated_atrophy.append(val)
            
            # EXPLOSION BRAKE: If >95% die before year 10, punish heavily
            if val > 95.0 and year < 10:
                return 5000.0 
        
        simulated_array = np.array(simulated_atrophy)
        
        # Calculate MSE
        error = np.mean((simulated_array - TARGET_ATROPHY_APOE4)**2)
        
        # DUD BRAKE: If final atrophy < 20%, punish (APOE4 must cause damage)
        if simulated_array[-1] < 20.0: 
            return error + 2000.0
            
        print(f"MSE: {error:.2f} | Prod: {params['amyloid_production']:.2f} | Agg: {params['aggregation_rate']:.2f}")
        return error

    except Exception as e:
        print(f"Failed run: {e}")
        return 5000.0

# --- 4. EXECUTION ---
if __name__ == "__main__":
    print(f"🚀 Starting APOE 4/4 Optimization (Restricted Ranges)...")
    
    start_time = time.time()

    # n_calls=50 allows the optimizer enough attempts to navigate 6 dimensions
    res = gp_minimize(
        objective, 
        space, 
        n_calls=50, 
        n_initial_points=15, 
        random_state=42, 
        verbose=True
    )

    print("\n" + "="*50)
    print(f"✅ OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Best Error (MSE):       {res.fun:.5f}")
    print("-" * 30)
    
    # Extract results
    keys = [
        "amyloid_production", "amyloid_clearance", "chemotaxis_efficiency",
        "inflammatory_threshold", "proliferation_rate", "aggregation_rate"
    ]
    
    print("Paste this into your GENETIC_PROFILES['APOE 4/4']:")
    print("{")
    for i, key in enumerate(keys):
        # Format explicitly to 4 decimal places
        print(f'    "{key}": {res.x[i]:.4f},')
    print("}")