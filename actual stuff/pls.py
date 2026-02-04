import numpy as np
import mesa
import random
import networkx as nx
from enum import Enum
import pandas as pd

# VISUALIZATION IMPORTS
from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle
from mesa.discrete_space import OrthogonalMooreGrid, FixedAgent, CellAgent

# --- 1. CONFIGURATION & GENETICS ---

class HippocampalLayer(Enum):
    DG = "DG"
    CA3 = "CA3"
    CA1 = "CA1"

# Added 'atrophy_coeffs' [a, b] for Target Curve: y = at^2 + bt (Quadratic growth of atrophy)
GENETIC_PROFILES = {
    # "APOE3 (Control)": {
    # "amyloid_production": 0.260,
    # "amyloid_clearance": 5.237,
    # "chemotaxis_efficiency": 0.922,
    # "inflammatory_threshold": 0.786,
    # "proliferation_rate": 2.470,
    # "atrophy_coeffs": [0.01665, -0.02222], # Slow progression
    "APOE3 (Control)": {
    "amyloid_production": 1.0,      # Baseline
    "amyloid_clearance": 1.0,       # Baseline
    "chemotaxis_efficiency": 1.0,   # Baseline
    "inflammatory_threshold": 1.0,  # Baseline
    "proliferation_rate": 1.0,      # Baseline
    "target_params": [100.0, 0.2587, 27.72]
    },
    "APOE 3/4": {
        "amyloid_production": 1.5, 
        "amyloid_clearance": 0.5,
        "chemotaxis_efficiency": 1.0,
        "inflammatory_threshold": 0.5,
        "proliferation_rate": 1.5,
        # "atrophy_coeffs": [0.15, 0.8] # Fast progression
        "target_params": [ 25.6069, 0.1778, 20.8447]

    },
    "APOE 4/4": {
        "amyloid_production": 1.0,
        "amyloid_clearance": 1.0, 
        "chemotaxis_efficiency": 0.2,
        "inflammatory_threshold": 1.0,
        "proliferation_rate": 0.5,
        # "atrophy_coeffs": [0.08, 0.4] # Moderate progression
        "target_params":  [100.0000, 0.1539, 26.7703]
    },
}

def build_hippocampal_graph(n_DG, n_CA3, n_CA1, rng):
    G = nx.DiGraph()
    DG_ids = range(0, n_DG)
    CA3_ids = range(n_DG, n_DG + n_CA3)
    CA1_ids = range(n_DG + n_CA3, n_DG + n_CA3 + n_CA1)

    for d in DG_ids:
        if len(CA3_ids) > 0:
            for c in rng.sample(list(CA3_ids), k=min(3, len(CA3_ids))):
                G.add_edge(d, c, weight=rng.uniform(0.4, 0.8))
    for i in CA3_ids:
        for j in CA3_ids:
            if i != j and rng.random() < 0.15:
                G.add_edge(i, j, weight=rng.uniform(0.6, 1.2))
    for c in CA3_ids:
        if len(CA1_ids) > 0:
            for a in rng.sample(list(CA1_ids), k=min(4, len(CA1_ids))):
                G.add_edge(c, a, weight=rng.uniform(0.7, 1.3))
    return G, DG_ids, CA3_ids, CA1_ids

# --- 2. MECHANISTIC AGENTS ---

class Neuron(FixedAgent):
    def __init__(self, model, cell, node_id, layer):
        super().__init__(model)
        self.cell = cell
        self.node_id = node_id
        self.layer = layer
        
        self.health = 1.0
        self.structural_integrity = 1.0
        self.state = "healthy"
        self.resilience = self.random.normalvariate(1.0, 0.1)
        self.vulnerability = {HippocampalLayer.DG: 0.8, HippocampalLayer.CA3: 1.0, HippocampalLayer.CA1: 1.6}[layer]

    def step(self):
        if self.state == "dead": return

        x, y = self.cell.coordinate
        local_amyloid = self.model.amyloid[x, y]
        local_cytokine = self.model.cytokine[x, y]
        
        # Shielding logic
        shield_strength = 0.0
        for agent in self.cell.agents:
            if isinstance(agent, Microglia) and agent.state != "dysfunctional":
                shield_strength = 0.6 if local_amyloid < 0.8 else 0.1
                break
        
        # Damage Calculation
        dam_amyloid = local_amyloid * self.model.amyloid_toxicity * (1.0 - shield_strength)
        dam_cytokine = 0.0
        if local_cytokine > 0.15:
            dam_cytokine = (local_cytokine ** 2) * (self.model.cytokine_toxicity * 0.05)
            
        total_damage = (dam_amyloid + dam_cytokine) * self.vulnerability
        
        # Transient Health Update (with REPAIR variable)
        self.health -= (total_damage * 0.5)
        # --- VARIABLE 1: Tunable Repair Rate ---
        self.health += self.model.neuron_repair_rate 
        self.health = min(self.health, 1.0)
        
        # Death Probability (with TIME CONSTANT variable)
        damage_load = 1.0 - self.health 
        base_risk = 1 / (1 + np.exp(-15 * (damage_load - self.model.neuron_death_threshold)))
        
        # --- VARIABLE 2: Tunable Time Constant ---
        step_death_prob = base_risk * self.model.time_constant 
        
        if self.random.random() < step_death_prob:
            self.state = "dead"
            self.health = 0.0
        elif self.health < 0.5: self.state = "stressed"
        else: self.state = "healthy"

class Microglia(CellAgent):
    def __init__(self, model, cell):
        super().__init__(model)
        self.cell = cell
        self.state = "M0" 
        self.activation_time = 0 

    def step(self):
        if self.state == "dysfunctional": return 

        params = self.model.genetic_profile
        
        # Movement
        move_success = False
        if self.random.random() < params['chemotaxis_efficiency']:
            best_cell = self.cell
            max_amy = -1.0
            for neighbor in self.cell.neighborhood:
                amy_val = self.model.amyloid[neighbor.coordinate]
                if amy_val > max_amy:
                    max_amy = amy_val
                    best_cell = neighbor
            
            if max_amy > self.model.amyloid[self.cell.coordinate]:
                 self.cell = best_cell
                 move_success = True
        
        if not move_success and self.random.random() < 0.5:
            new_cell = self.cell.neighborhood.select_random_cell()
            if new_cell: self.cell = new_cell

        # Phagocytosis
        x, y = self.cell.coordinate
        local_amyloid = self.model.amyloid[x, y]
        
        clearance_capacity = 0.1 * params['amyloid_clearance'] * self.model.lecanemab_efficacy_multiplier
        if local_amyloid > 0:
            amount_eaten = min(local_amyloid, clearance_capacity)
            self.model.amyloid[x, y] -= amount_eaten
            if self.model.lecanemab_dose > 0:
                self.model.cytokine[x, y] += (amount_eaten * 0.05)

        # Activation
        base_threshold = 0.2 
        frustration_threshold = base_threshold * params['inflammatory_threshold'] * self.model.cromolyn_efficacy_multiplier
        
        if local_amyloid > frustration_threshold:
            self.state = "M1"
            self.activation_time += 1
        else:
            if self.random.random() < 0.2: 
                self.state = "M0"
                self.activation_time = 0
            
        if self.state == "M1":
            self.model.cytokine[x, y] += 0.08
            
            # --- UPDATED: Uses Model Parameter ---
            if self.activation_time > self.model.microglia_burnout_time:
                self.state = "dysfunctional"
            elif self.activation_time > 10 and self.random.random() < (0.1 * params['proliferation_rate']):
                if self.model.num_microglia < 300: 
                    target = self.cell.neighborhood.select_random_cell()
                    if target:
                        new_mg = Microglia(self.model, target)
                        new_mg.state = "M1"
# --- 3. THE MODEL ---

class ADModel(mesa.Model):
    def __init__(
        self, width=20, height=20, n_neurons=200, n_microglia=40,
        genotype="APOE3 (Control)", 
        
        # Tunable Physics
        amyloid_toxicity=0.0,
        cytokine_toxicity=0.0,
        neuron_death_threshold=0.25,
        
        # NEW TUNABLE VARIABLES
        neuron_repair_rate=0.008, 
        time_constant=0.0005,      
        plaque_feedback_rate=0.1,  

        max_steps=1000, seed=None
    ):
        super().__init__(seed=seed)
        self.grid = OrthogonalMooreGrid((width, height), torus=False, random=self.random)
        self.current_step = 0
        self.max_steps = max_steps
        self.total_neurons = n_neurons
        
        # Store parameters
        self.neuron_death_threshold = neuron_death_threshold
        self.neuron_repair_rate = neuron_repair_rate
        self.plaque_feedback_rate = plaque_feedback_rate
        self.time_constant = time_constant
        self.microglia_burnout_time = 62
        
        # Time Mapping
        self.years_per_step = 20.0 / 1000.0
        self.current_year = 0.0
        
        self.genetic_profile = GENETIC_PROFILES.get(genotype, GENETIC_PROFILES["APOE3 (Control)"]).copy()
        
        # Safety Caps for Inputs
        self.amyloid_toxicity = max(0.0, amyloid_toxicity)
        self.cytokine_toxicity = max(0.0, cytokine_toxicity)
        self.lecanemab_efficacy_multiplier = 1.0
        self.cromolyn_efficacy_multiplier = 1.0

        self.amyloid = np.zeros((width, height))
        self.cytokine = np.zeros((width, height))

        # Build Graph
        n_DG, n_CA3, n_CA1 = int(n_neurons*0.4), int(n_neurons*0.3), int(n_neurons*0.3)
        self.synapse_graph, DG_ids, CA3_ids, CA1_ids = build_hippocampal_graph(n_DG, n_CA3, n_CA1, self.random)
        
        self.neuron_map = {}
        node_layer_map = ({i: HippocampalLayer.DG for i in DG_ids} | {i: HippocampalLayer.CA3 for i in CA3_ids} | {i: HippocampalLayer.CA1 for i in CA1_ids})
        for node_id, layer in node_layer_map.items():
            cell = self.grid.all_cells.select_random_cell()
            self.neuron_map[node_id] = Neuron(self, cell, node_id, layer)
        
        for _ in range(n_microglia):
            cell = self.grid.all_cells.select_random_cell()
            Microglia(self, cell)
        
        self.datacollector = mesa.DataCollector({
            "Simulated Atrophy %": lambda m: m.get_atrophy_percent(),
        })
        self.running = True

    @property
    def num_microglia(self):
        return sum(1 for a in self.agents if isinstance(a, Microglia))

    def get_atrophy_percent(self):
        dead_count = sum(1 for n in self.neuron_map.values() if n.state == "dead")
        if self.total_neurons == 0: return 0
        return (dead_count / self.total_neurons) * 100

    def get_target_atrophy(self):
        t = self.current_year
        params = self.genetic_profile.get('target_params', [0, 0, 0])
        L, k, t0 = params
        if k == 0 or L == 0: return 0.0
        try:
            val = L / (1 + np.exp(-k * (t - t0)))
            return val
        except OverflowError:
            return 0.0

    def step(self):
        self.current_step += 1
        self.current_year += self.years_per_step
        
        # Amyloid Dynamics
        prod_rate = 0.1 * self.genetic_profile['amyloid_production']
        noise = np.random.normal(0.01, 0.005, (self.grid.width, self.grid.height))
        self.amyloid += noise * prod_rate
        
        # CRITICAL FIX 1: Clip negatives to 0 BEFORE power calc
        self.amyloid = np.maximum(self.amyloid, 0)
        
        # Feedback (Uses Tunable Variable)
        self.amyloid += (self.amyloid ** 1.5) * self.plaque_feedback_rate
        
        # CRITICAL FIX 2: Prevent Infinity/Overflow
        self.amyloid = np.minimum(self.amyloid, 100.0)
        
        # Diffusion (Simplified)
        self.cytokine *= 0.90 
        
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

# --- 4. VISUALIZATION ---

def agent_portrayal(agent):
    if isinstance(agent, Neuron):
        if agent.state == "healthy": return AgentPortrayalStyle(color="tab:green", size=40)
        if agent.state == "blocked": return AgentPortrayalStyle(color="tab:blue", size=40)
        if agent.state == "stressed": return AgentPortrayalStyle(color="tab:orange", size=40)
        return AgentPortrayalStyle(color="black", size=20)
    
    if isinstance(agent, Microglia):
        if agent.state == "M1": return AgentPortrayalStyle(color="tab:red", size=60)
        if agent.state == "dysfunctional": return AgentPortrayalStyle(color="gray", size=60) 
        return AgentPortrayalStyle(color="cornflowerblue", size=60) 

    return AgentPortrayalStyle(color="lightgray", size=10)

def post_process(ax):
    fig = ax.get_figure()
    fig.set_layout_engine(None) 
    ax.set_aspect("equal")

model_params = {
    "genotype": { "type": "Select", "value": "APOE3 (Control)", "values": list(GENETIC_PROFILES.keys()) },
    "amyloid_toxicity": { "type": "SliderFloat", "value": 5.0, "label": "Amyloid Toxicity", "min": 0.0, "max": 20.0, "step": 0.5 },
    "cytokine_toxicity": { "type": "SliderFloat", "value": 5.0, "label": "Inflammation Toxicity", "min": 0.0, "max": 20.0, "step": 0.5 },
    "lecanemab_dose": { "type": "SliderFloat", "value": 0.0, "label": "Lecanemab (mg/kg)", "min": 0.0, "max": 10.0, "step": 1.0 },
    "cromolyn_dose": { "type": "SliderFloat", "value": 0.0, "label": "Cromolyn (mg/kg)", "min": 0.0, "max": 100.0, "step": 5.0 },
    "n_neurons": { "type": "SliderInt", "value": 200, "label": "Neurons", "min": 100, "max": 500, "step": 10 },
}

AD_model_instance = ADModel(width=20, height=20, n_neurons=200, n_microglia=40)

renderer = SpaceRenderer(model=AD_model_instance, backend="matplotlib").render(
    agent_portrayal=agent_portrayal, post_process=post_process
)

# CHANGED: Graphing Atrophy % vs Target Curve
chart_atrophy = make_plot_component(["Simulated Atrophy %", "Target Atrophy %"])
chart_env = make_plot_component("Total Amyloid")

page = SolaraViz(
    AD_model_instance,
    renderer, 
    components=[chart_atrophy, chart_env],
    model_params=model_params,
    name="Mechanistic AD Model: Clinical Validation",
)

page