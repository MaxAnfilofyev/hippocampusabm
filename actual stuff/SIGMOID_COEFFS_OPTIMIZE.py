import numpy as np
import time
import warnings
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# --- IMPORT YOUR MODEL ---
# CHANGE 'try_again' TO THE NAME OF YOUR PYTHON FILE (without .py)
try:
    from try_again import ADModel
except ImportError:
    print("❌ CRITICAL ERROR: Could not import ADModel. Check your filename.")
    exit()

# --- 1. SETUP TARGETS ---
# Target Coefficients for APOE3: L=100 (Max Atrophy), k=0.2587 (Rate), t0=27.72 (Midpoint)
TARGET_COEFFS = np.array([100.0, 0.2587, 27.72]) 

def sigmoid(t, L, k, t0):
    """The clinical Sigmoid curve we want to match."""
    k = np.clip(k, -10, 10) 
    t0 = np.clip(t0, -100, 100)
    return L / (1 + np.exp(-k * (t - t0)))

# --- 2. SMOOTHING & FITTING HELPER ---
def extract_coefficients(time_steps, atrophy_data):
    """
    Fits the noisy simulation data to a sigmoid curve to extract L, k, t0.
    """
    # Smooth data to prevent fitting errors on jagged lines
    try:
        if len(atrophy_data) > 15:
            smoothed_data = savgol_filter(atrophy_data, window_length=15, polyorder=3)
        else:
            smoothed_data = atrophy_data 
    except:
        smoothed_data = atrophy_data

    try:
        # Curve Fit
        popt, _ = curve_fit(
            sigmoid, 
            time_steps, 
            smoothed_data, 
            p0=[100, 0.2, 20], # Initial guesses
            bounds=([0, 0, 0], [105, 5.0, 100]), # Constraints
            maxfev=5000
        )
        return popt 
    except Exception:
        # Return None if fit fails (e.g., flat line)
        return None

# --- 3. SEARCH SPACE (UPDATED) ---
space = [
    # 1. AMYLOID TOXICITY (The Gas Pedal)
    # Controls how fast damage accumulates from plaque.
    Real(5.0, 60.0, name='amyloid_toxicity'),
    
    # 2. CYTOKINE TOXICITY (The Turbo)
    # Controls the severity of the inflammatory feedback loop.
    Real(10.0, 60.0, name='cytokine_toxicity'),
    
    # 3. REPAIR RATE (The Brakes)
    # Controls the duration of the "Silent Phase" (resilience).
    Real(0.001, 0.015, name='neuron_repair_rate'),

    # 4. DEATH THRESHOLD (The Trigger)
    # The health level at which a neuron finally dies.
    Real(0.0, 0.4, name='neuron_death_threshold')
]

# --- 4. HYBRID OBJECTIVE FUNCTION ---
@use_named_args(space)
def objective(**params):
    try:
        # --- A. Initialize Model with Candidate Parameters ---
        model = ADModel(
            width=20, height=20, 
            n_neurons=200, n_microglia=40, # Keeping scale low for speed
            genotype="APOE3 (Control)",
            
            # Optimization Targets
            amyloid_toxicity=params['amyloid_toxicity'],
            cytokine_toxicity=params['cytokine_toxicity'],
            neuron_repair_rate=params['neuron_repair_rate'],
            neuron_death_threshold=params['neuron_death_threshold'],
            
            max_steps=1000 
        )
        
        # --- B. Run Simulation ---
        simulated_atrophy = []
        # We simulate 20 "Years" (assuming ~50 steps = 1 year)
        years = np.arange(1, 21) 
        
        for year in years:
            for _ in range(50): 
                model.step()
            simulated_atrophy.append(model.get_atrophy_percent())

        sim_data = np.array(simulated_atrophy)
        
        # Generate the Perfect Target Curve for these years
        target_curve = sigmoid(years, *TARGET_COEFFS)
        
        # --- C. Scoring Logic ---
        
        # FAILURE CASE 1: Flat Line (Model too resilient)
        # If max atrophy is < 5% after 20 years, penalize heavily.
        if np.max(sim_data) < 5.0:
            mse = np.mean((sim_data - target_curve)**2)
            return mse + 1000.0 # High penalty

        # FAILURE CASE 2: Instant Death (Model too fragile)
        # If atrophy hits 100% in year 1, penalize.
        if sim_data[0] > 10.0:
             return 2000.0

        # Attempt to fit sigmoid coefficients
        sim_coeffs = extract_coefficients(years, sim_data)
        
        # If fit fails, default to simple Mean Squared Error (MSE)
        if sim_coeffs is None:
            return np.mean((sim_data - target_curve)**2) + 50.0 

        # SUCCESS CASE: Compare Coefficients (Shape Matching)
        # We weigh 'k' (slope) and 't0' (timing) highest.
        # [L, k, t0]
        weights = [1.0, 20.0, 5.0] 
        rel_error = 0.0
        
        for i in range(3):
            diff = (sim_coeffs[i] - TARGET_COEFFS[i])
            # Normalize error so L (100) doesn't overpower k (0.2)
            norm_factor = TARGET_COEFFS[i] if TARGET_COEFFS[i] != 0 else 1.0
            rel_error += weights[i] * ((diff / norm_factor) ** 2)
            
        return rel_error

    except Exception as e:
        print(f"⚠️ CRASH IN OBJECTIVE: {e}")
        return 10000.0

# --- 5. EXECUTION ---
if __name__ == "__main__":
    print(f"🚀 Starting Bayesian Optimization...")
    print(f"🎯 Target Curve Parameters: L={TARGET_COEFFS[0]}, k={TARGET_COEFFS[1]}, t0={TARGET_COEFFS[2]}")
    print(f"🔍 Optimizing: Amyloid Tox, Cytokine Tox, Repair Rate, Death Threshold")

    # Run Bayesian Optimization (Gaussian Process)
    res = gp_minimize(
        objective, 
        space, 
        n_calls=50,       # Number of iterations (Higher = Better but slower)
        random_state=42, 
        verbose=True
    )

    print("\n" + "="*60)
    print(f"✅ OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"🏆 Best Loss Score:       {res.fun:.5f}")
    print("-" * 40)
    print(f"1️⃣  Amyloid Toxicity:     {res.x[0]:.4f}")
    print(f"2️⃣  Cytokine Toxicity:    {res.x[1]:.4f}")
    print(f"3️⃣  Neuron Repair Rate:   {res.x[2]:.6f}")
    print(f"4️⃣  Death Threshold:      {res.x[3]:.4f}")
    print("="*60)
    print("👉 Copy these values into your ADModel __init__ defaults!")