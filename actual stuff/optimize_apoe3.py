import numpy as np
import time
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from try_again import ADModel  # Ensure this matches your filename

# --- 1. DEFINE THE TARGET CURVE ---
# Coefficients provided: [0.01665, -0.02222] -> y = at^2 + bt
COEFFS = [0.01665, -0.02222]

def get_target_atrophy(year):
    """Calculates the target % loss for a specific year."""
    a, b = COEFFS
    val = (a * (year**2)) + (b * year)
    return max(0.0, val)

# --- 2. DEFINE THE SEARCH SPACE ---
space = [
    # Genetic Multipliers (Normal = 1.0)
    Real(0.01, 5.0, name='amyloid_production'),      
    Real(0.1, 6.0, name='amyloid_clearance'),       
    Real(0.1, 2.0, name='chemotaxis_efficiency'),   
    Real(0.1, 3.0, name='inflammatory_threshold'),  
    Real(0.1, 4.0, name='proliferation_rate'),      
    
    # Toxicity Sliders
    Real(0.0, 50.0, name='amyloid_toxicity'),       
    Real(0.0, 50.0, name='cytokine_toxicity'),     

    # Mechanistic Variables
    Real(0.05, 0.40, name='neuron_death_threshold'), 
    Real(0.001, 0.02, name='neuron_repair_rate'),    
    Real(0.01, 0.10, name='plaque_feedback_rate'),   
    Integer(10, 80,  name='microglia_burnout_time') 
]

# --- 3. THE OBJECTIVE FUNCTION ---
@use_named_args(space)
def objective(**params):
    
    # 1. Prepare Genetic Overrides
    genetic_overrides = {
        "amyloid_production": params['amyloid_production'],
        "amyloid_clearance": params['amyloid_clearance'],
        "chemotaxis_efficiency": params['chemotaxis_efficiency'],
        "inflammatory_threshold": params['inflammatory_threshold'],
        "proliferation_rate": params['proliferation_rate']
    }

    # 2. Run Replicates
    mse_scores = []
    
    for _ in range(3): # Run 3 times per parameter set
        model = ADModel(
            width=20, height=20, n_neurons=200, n_microglia=40,
            genotype="APOE3 (Control)",
            
            # Genetic
            genetic_overrides=genetic_overrides, 
            
            # Toxicity
            amyloid_toxicity=params['amyloid_toxicity'],
            cytokine_toxicity=params['cytokine_toxicity'],
            
            # Mechanistic
            neuron_death_threshold=params['neuron_death_threshold'],
            neuron_repair_rate=params['neuron_repair_rate'],
            plaque_feedback_rate=params['plaque_feedback_rate'],
            microglia_burnout_time=int(params['microglia_burnout_time']),
            
            max_steps=1000  # <--- CHANGED: Model initialized for 1000 steps
        )
        
        simulated_curve = []
        target_curve = []
        
        # 3. Step through simulation
        # <--- CHANGED: Loop runs for 1000 steps
        for step in range(1000):
            model.step()
            
            # Sample data every 20 steps (approx same resolution as before)
            if step % 20 == 0:
                sim_val = model.get_atrophy_percent()
                simulated_curve.append(sim_val)
                
                # Calculate target for the model's CURRENT year
                target_val = get_target_atrophy(model.current_year)
                target_curve.append(target_val)
                
                # Penalty Short-Circuit
                if sim_val > 100: 
                    break

        # Pad curves
        max_len = max(len(simulated_curve), len(target_curve))
        while len(simulated_curve) < max_len: simulated_curve.append(100.0)
        while len(target_curve) < max_len: target_curve.append(target_curve[-1])

        # 4. Calculate MSE
        mse = np.mean((np.array(simulated_curve) - np.array(target_curve))**2)
        mse_scores.append(mse)

    return np.mean(mse_scores)

# --- 4. EXECUTION ---
if __name__ == "__main__":
    print(f"Starting Optimization (1000 Steps) against Target Coeffs: {COEFFS}")
    print("This may take a few minutes...")
    start_time = time.time()

    # Run Bayesian Optimization
    # n_calls=50 is a good starting point, increase to 100 if results are poor
    res = gp_minimize(objective, space, n_calls=50, random_state=42, verbose=True)

    print("\n" + "="*50)
    print(f"OPTIMIZATION COMPLETE in {time.time() - start_time:.2f} seconds")
    print(f"Best Error (MSE): {res.fun:.5f}")
    print("="*50)
    print("PASTE THESE INTO GENETIC_PROFILES:")
    print(f'"amyloid_production": {res.x[0]:.3f},')
    print(f'"amyloid_clearance": {res.x[1]:.3f},')
    print(f'"chemotaxis_efficiency": {res.x[2]:.3f},')
    print(f'"inflammatory_threshold": {res.x[3]:.3f},')
    print(f'"proliferation_rate": {res.x[4]:.3f},')
    print("-" * 30)
    print("SLIDER DEFAULTS:")
    print(f'amyloid_toxicity={res.x[5]:.2f}')
    print(f'cytokine_toxicity={res.x[6]:.2f}')
    print("-" * 30)
    print("MECHANISTIC DEFAULTS:")
    print(f'neuron_death_threshold={res.x[7]:.3f}')
    print(f'neuron_repair_rate={res.x[8]:.4f}')
    print(f'plaque_feedback_rate={res.x[9]:.4f}')
    print(f'microglia_burnout_time={res.x[10]}')
    print("="