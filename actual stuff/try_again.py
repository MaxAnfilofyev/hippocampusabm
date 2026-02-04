import numpy as np
import mesa
import random
import networkx as nx
from enum import Enum
import pandas as pd

# VISUALIZATION IMPORTS
# (These are only needed if running this file directly, not when importing for calibration)
try:
    from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
    from mesa.visualization.components import AgentPortrayalStyle
except ImportError:
    pass # Handle cases where visualization libs aren't installed in the calibration env

from mesa.discrete_space import OrthogonalMooreGrid, FixedAgent, CellAgent

# --- 1. CONFIGURATION & GENETICS ---

class HippocampalLayer(Enum):
    DG = "DG"
    CA3 = "CA3"
    CA1 = "CA1"

# OPTIMIZED GENETICS: APOE4 drives clearance failure & seeding.
GENETIC_PROFILES = {
    "APOE3 (Control)": {
        "amyloid_production": 1.0,      
        "amyloid_clearance": 1.0,       
        "chemotaxis_efficiency": 1.0,   
        "inflammatory_threshold": 1.0,  
        "proliferation_rate": 1.0,      
        "aggregation_rate": 1.0,         
        "target_params": [100.0, 0.2587, 27.72] 
    },
    "APOE 3/4": {
        "amyloid_production": 1.0,       
        "amyloid_clearance": 0.6,        
        "chemotaxis_efficiency": 1.0,
        "inflammatory_threshold": 0.7,   
        "proliferation_rate": 1.0,
        "aggregation_rate": 1.5,         
        "target_params": [25.6069, 0.1778, 20.8447]
    },
    "APOE 4/4": {
        "amyloid_production": 1.0,
        "amyloid_clearance": 0.4,
        "chemotaxis_efficiency": 0.7,
        "inflammatory_threshold": 0.5,
        "proliferation_rate": 0.5,
        "aggregation_rate": 3.0,
        "target_params": [100.0000, 0.1539, 26.7703]
        }
    }

def build_spatial_hippocampal_graph(model, rng):
    """
    Wires neurons based on spatial proximity on the grid.
    Connects DG -> CA3 -> CA1 with a probability that decays with distance.
    """
    G = nx.DiGraph()
    
    # 1. Group neurons by layer for easy access
    layer_map = {
        HippocampalLayer.DG: [],
        HippocampalLayer.CA3: [],
        HippocampalLayer.CA1: []
    }
    
    # Ensure all nodes are added to the graph first
    for neuron in model.neuron_map.values():
        layer_map[neuron.layer].append(neuron)
        G.add_node(neuron.node_id) 

    # Helper function to connect layers with spatial decay
    def connect_layers(source_layer, target_layer, connection_prob, long_range_prob):
        sources = layer_map[source_layer]
        targets = layer_map[target_layer]
        
        if not targets: return

        for src in sources:
            src_pos = np.array(src.cell.coordinate)
            
            # Optimization: Only look at a random subset of potential targets
            potential_targets = rng.sample(targets, k=min(len(targets), 25))
            
            connections_made = 0
            for tgt in potential_targets:
                if connections_made >= 5: break # Max outgoing synapses cap
                
                tgt_pos = np.array(tgt.cell.coordinate)
                dist = np.linalg.norm(src_pos - tgt_pos)
                
                # Probability: Sigmoid decay based on distance
                spatial_p = np.exp(-0.15 * dist) 
                
                # Roll dice: Spatial OR Long-range random chance
                if rng.random() < spatial_p or rng.random() < long_range_prob:
                    weight = rng.uniform(0.5, 1.0)
                    G.add_edge(src.node_id, tgt.node_id, weight=weight)
                    connections_made += 1

    # 2. DG -> CA3 (Mossy Fibers)
    connect_layers(HippocampalLayer.DG, HippocampalLayer.CA3, connection_prob=0.3, long_range_prob=0.05)

    # 3. CA3 -> CA3 (Recurrent Collaterals - Local Loops)
    ca3_neurons = layer_map[HippocampalLayer.CA3]
    for src in ca3_neurons:
        src_pos = np.array(src.cell.coordinate)
        others = [n for n in rng.sample(ca3_neurons, min(len(ca3_neurons), 15)) if n != src]
        for tgt in others:
            dist = np.linalg.norm(src_pos - np.array(tgt.cell.coordinate))
            if rng.random() < (np.exp(-0.2 * dist)):
                 G.add_edge(src.node_id, tgt.node_id, weight=rng.uniform(0.4, 0.9))

    # 4. CA3 -> CA1 (Schaffer Collaterals)
    connect_layers(HippocampalLayer.CA3, HippocampalLayer.CA1, connection_prob=0.4, long_range_prob=0.1)

    return G

# --- 2. MECHANISTIC AGENTS ---

class Neuron(FixedAgent):
    def __init__(self, model, cell, node_id, layer):
        super().__init__(model)
        self.cell = cell
        self.node_id = node_id
        self.layer = layer
        
        self.health = 1.0
        self.state = "healthy"
        
        # --- CONDITIONAL FIX: Restricted Resilience for APOE4 Only ---
        # If genotype has "4" (APOE 4/4 or 3/4), use low variance (0.05).
        # Otherwise (APOE3), use high variance (0.2).
        if "4" in self.model.genotype_name:
            sigma = 0.05
        else:
            sigma = 0.2
            
        self.resilience = self.random.normalvariate(1.0, sigma) 
        
        # COGNITIVE RESERVE
        self.reserve = 1.0 * self.resilience
        
        # INTERNAL TAU (The stress memory)
        self.internal_tau = 0.0

        self.vulnerability = {
            HippocampalLayer.DG: 0.8,
            HippocampalLayer.CA3: 1.0,
            HippocampalLayer.CA1: 1.5, 
        }[layer]

    def step(self):
        if self.state == "dead": return

        x, y = self.cell.coordinate
        local_amyloid = self.model.amyloid[x, y]
        local_cytokine = self.model.cytokine[x, y]
        
        # shielding
        has_shield = False
        has_toxic_neighbor = False
        for agent in self.cell.agents:
            if isinstance(agent, Microglia):
                if agent.state not in ["M3", "dysfunctional"]:
                    has_shield = True
                if agent.state in ["M1", "M3"]:
                    has_toxic_neighbor = True
                break
        
        shield_strength = 0.5 if (has_shield and local_amyloid < 1.0) else 0.0
        
        # damage inputs
        dam_amyloid = (local_amyloid ** 1.2) * (self.model.amyloid_toxicity * 0.001) * (1.0 - shield_strength)
        
        dam_cytokine = 0.0
        if local_cytokine > 0.1:
            dam_cytokine = (local_cytokine ** 2) * (self.model.cytokine_toxicity * 0.02)
            
        dam_proximity = 0.05 if has_toxic_neighbor else 0.0
        
        # network stress from synaptic dysfunction
        network_stress = 0.0
        transmission_factor = max(0.0, 1.0 - (local_amyloid * 1.5))
        
        if self.model.synapse_graph.has_node(self.node_id):
            preds = list(self.model.synapse_graph.predecessors(self.node_id))
            if preds:
                total_signal = 0.0
                for p_id in preds:
                    neighbor = self.model.neuron_map[p_id]
                    if neighbor.state != "dead":
                        total_signal += (1.0 * transmission_factor)
                
                avg_signal = total_signal / len(preds)
                # hockey stick math
                raw_stress = ((1.0 - avg_signal) ** 2) * 2.0 
                if self.model.genotype_name == "APOE 4/4":
                    network_stress = min(raw_stress, 1.0)
                else: 
                    network_stress = raw_stress
                

        total_damage = (dam_amyloid + dam_cytokine + dam_proximity + network_stress) * self.vulnerability
        
        # dynamic repair
        base_repair = self.model.neuron_repair_rate
        
        # LOGISTIC DECAY
        # 1.0 efficiency until Reserve drops below 0.2, then slides down.
        # Changed slope from 10.0 -> 5.0 to prevent instant death.
        metabolic_efficiency = 1.0 / (1.0 + np.exp(5.0 * (0.2 - self.reserve)))
        
        # tau accumulation
        # deduced multiplier from 0.05 -> 0.002 to slow the death
        self.internal_tau += network_stress * 0.002
        tau_toxicity = min(0.9, self.internal_tau)
        
        actual_repair = base_repair * metabolic_efficiency * (1.0 - tau_toxicity)
        
        # apply damage and repair
        if self.reserve > 0:
            self.reserve -= total_damage
            self.reserve += actual_repair
            self.reserve = min(max(self.reserve, 0.0), 2.0)
        else:
            self.health -= total_damage
            self.health = min(max(self.health, 0.0), 1.0)
            
        # how they die
        if self.health <= self.model.neuron_death_threshold:
            self.state = "dead"
            self.health = 0.0
        elif self.health < 0.5:
             self.state = "stressed"
        elif transmission_factor < 0.5:
             self.state = "blocked"
        else:
            self.state = "healthy"

class Microglia(CellAgent):
    def __init__(self, model, cell):
        super().__init__(model)
        self.cell = cell
        self.state = "M0" 
        self.activation_time = 0 
        self.cumulative_stress = 0 

    def step(self):
        # Senescent Behavior (M3)
        if self.state == "M3":
            x, y = self.cell.coordinate
            # OPTIMIZATION FIX: Drug now reduces output from senescent cells
            release = 0.15 / self.model.cromolyn_efficacy_multiplier
            self.model.cytokine[x, y] += release
            return 

        params = self.model.genetic_profile
        x, y = self.cell.coordinate
        local_amyloid = self.model.amyloid[x, y]

        # Movement
        move_success = False
        chemotaxis_strength = params['chemotaxis_efficiency']
        if self.random.random() < chemotaxis_strength:
            best_cell = self.cell
            max_amy = -1.0
            for neighbor in self.cell.neighborhood:
                amy_val = self.model.amyloid[neighbor.coordinate]
                if amy_val > max_amy:
                    max_amy = amy_val
                    best_cell = neighbor
            if max_amy > local_amyloid:
                 self.cell = best_cell
                 move_success = True
        
        if not move_success and self.random.random() < 0.5:
            new_cell = self.cell.neighborhood.select_random_cell()
            if new_cell: self.cell = new_cell

        # Phagocytosis
        clearance_capacity = 0.1 * params['amyloid_clearance'] * self.model.lecanemab_efficacy_multiplier
        if self.state == "M1": clearance_capacity *= 1.5

        if local_amyloid > 0:
            amount_eaten = min(local_amyloid, clearance_capacity)
            self.model.amyloid[x, y] -= amount_eaten
            if self.model.lecanemab_dose > 0:
                self.model.cytokine[x, y] += (amount_eaten * 0.05)

        # Activation
        base_threshold = 0.2 
        activation_thresh = base_threshold * params['inflammatory_threshold'] * self.model.cromolyn_efficacy_multiplier
        
        if self.state == "M0" and local_amyloid > activation_thresh:
            self.state = "M1"
        elif self.state == "M1" and local_amyloid < (activation_thresh * 0.5):
            if self.random.random() < 0.3: 
                self.state = "M0"
                self.activation_time = 0

        # M1 Burnout
        if self.state == "M1":
            # OPTIMIZATION FIX: Drug now reduces output from active M1 cells
            release = 0.05 / self.model.cromolyn_efficacy_multiplier
            self.model.cytokine[x, y] += release 
            
            self.activation_time += 1
            self.cumulative_stress += 1
            if self.cumulative_stress > self.model.microglia_burnout_time:
                self.state = "M3"
            elif self.activation_time > 10 and self.random.random() < (0.1 * params['proliferation_rate']):
                if self.model.num_microglia < 300: 
                    target = self.cell.neighborhood.select_random_cell()
                    if target:
                        new_mg = Microglia(self.model, target)
                        new_mg.state = "M1"

# actual model class

class ADModel(mesa.Model):
    def __init__(
        self, width=40, height=40, n_neurons=1000, n_microglia=200, # SCALED UP
        genotype="APOE3 (Control)", 
        genetic_overrides=None,  
        
        amyloid_toxicity=40,    
        cytokine_toxicity=10,
        lecanemab_dose=0.0, 
        lecanemab_potency=0.01140, # <--- ADDED THIS
        cromolyn_dose=0.0,
        cromolyn_potency=0.0050,  # <--- ADDED THIS
        
        neuron_death_threshold=0.004174, 
        neuron_repair_rate=0.004174, 
        plaque_feedback_rate=0.15,
        microglia_burnout_time=400, 

        max_steps=1200, seed=374281
    ):
        super().__init__(seed=seed)
        
        #RANDOMNESS SET
        self.np_rng = np.random.default_rng(seed)

        # SETUP GRID
        self.grid = OrthogonalMooreGrid((width, height), torus=False, random=self.random)
        self.current_step = 0
        self.max_steps = max_steps
        self.total_neurons = n_neurons
        
        # 2. SETUP GENETICS
        base_profile = GENETIC_PROFILES.get(genotype, GENETIC_PROFILES["APOE3 (Control)"]).copy()
        if genetic_overrides:
            base_profile.update(genetic_overrides)
        self.genetic_profile = base_profile
        
        # 3. SETUP MODEL PARAMETERS
        self.neuron_death_threshold = neuron_death_threshold
        self.neuron_repair_rate = neuron_repair_rate
        self.microglia_burnout_time = microglia_burnout_time

        ##STORE GENOTYPE NAME
        self.genotype_name = genotype
        
        agg_rate = self.genetic_profile.get('aggregation_rate', 1.0)
        self.plaque_feedback_rate = plaque_feedback_rate * agg_rate
        
        self.years_per_step = 20.0 / 1000.0
        self.current_year = 0.0
        
        self.amyloid_toxicity = max(0.0, amyloid_toxicity)
        self.cytokine_toxicity = max(0.0, cytokine_toxicity)
        
        # --- UPDATED DRUG MATH FOR OPTIMIZATION ---
        self.lecanemab_dose = lecanemab_dose
        # Old: 1.0 + (lecanemab_dose / 5.0)
        # New: 1.0 + (lecanemab_dose * lecanemab_potency)
        self.lecanemab_efficacy_multiplier = 1.0 + (lecanemab_dose * lecanemab_potency) 
        
        self.cromolyn_dose = cromolyn_dose
        # Old: 1.0 + (cromolyn_dose / 50.0)
        # New: 1.0 + (cromolyn_dose * cromolyn_potency)
        self.cromolyn_efficacy_multiplier = 1.0 + (cromolyn_dose * cromolyn_potency)

        self.amyloid = np.zeros((width, height))
        self.cytokine = np.zeros((width, height))

        # 4. INITIALIZE AGENTS (REORDERED for Spatial Graph)
        
        # A. Place Neurons FIRST
        self.neuron_map = {}
        n_DG, n_CA3, n_CA1 = int(n_neurons*0.4), int(n_neurons*0.3), int(n_neurons*0.3)
        # Adjustment for rounding errors
        current_total = n_DG + n_CA3 + n_CA1
        if current_total < n_neurons: n_CA1 += (n_neurons - current_total)

        definitions = (
            [(i, HippocampalLayer.DG) for i in range(0, n_DG)] +
            [(i, HippocampalLayer.CA3) for i in range(n_DG, n_DG + n_CA3)] +
            [(i, HippocampalLayer.CA1) for i in range(n_DG + n_CA3, n_neurons)]
        )

        for node_id, layer in definitions:
            cell = self.grid.all_cells.select_random_cell()
            n = Neuron(self, cell, node_id, layer)
            self.neuron_map[node_id] = n

        # B. Build Spatial Graph (Uses positions established above)
        self.synapse_graph = build_spatial_hippocampal_graph(self, self.random)
        
        # C. Place Microglia
        for _ in range(n_microglia):
            cell = self.grid.all_cells.select_random_cell()
            Microglia(self, cell)
        
        # --- DATA COLLECTOR UPDATE ---
        # Added "Total Cytokine" to satisfy calibration script requirements
        self.datacollector = mesa.DataCollector({
            "Simulated Atrophy %": lambda m: m.get_atrophy_percent(),
            "Target Atrophy %": lambda m: m.get_target_atrophy(),
            "Total Amyloid": lambda m: np.sum(m.amyloid),
            "Total Cytokine": lambda m: np.sum(m.cytokine) 
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
            return L / (1 + np.exp(-k * (t - t0)))
        except OverflowError:
            return 0.0

    def step(self):
        self.current_step += 1
        self.current_year += self.years_per_step
        
        # 1. AMYLOID PRODUCTION & FEEDBACK
        prod_rate = 0.1 * self.genetic_profile['amyloid_production']
        # Moderate noise to allow silent phase
        noise = self.np_rng.normal(0.02, 0.005, (self.grid.width, self.grid.height))
        self.amyloid += noise * prod_rate
        self.amyloid = np.maximum(self.amyloid, 0) 
        
        # Feedback (Seeding Effect)
        # --- FIX FOR APOE4 CRASH: Clamp Amyloid Growth ---
        growth = (self.amyloid ** 2.0) * self.plaque_feedback_rate
        
        # Only apply the safety clamp if it is APOE 4/4
        if self.genotype_name == "APOE 4/4":
            growth = np.minimum(growth, 5.0)
            
        self.amyloid += growth
        self.amyloid = np.minimum(self.amyloid, 100.0)
        
        # 2. CYTOKINE DIFFUSION
        diffused_cytokine = np.zeros_like(self.cytokine)
        for cell in self.grid.all_cells:
            x, y = cell.coordinate
            val = self.cytokine[x, y]
            neighbors = cell.neighborhood
            if len(neighbors) > 0:
                share = val * 0.25 
                per_neighbor = share / len(neighbors)
                for n in neighbors:
                    diffused_cytokine[n.coordinate] += per_neighbor
                diffused_cytokine[x, y] += val * 0.75
            else:
                diffused_cytokine[x, y] += val
        
        self.cytokine = diffused_cytokine
        self.cytokine *= 0.90 

        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

# --- 4. VISUALIZATION (PROTECTED) ---

def agent_portrayal(agent):
    if isinstance(agent, Neuron):
        if agent.state == "healthy": return AgentPortrayalStyle(color="tab:green", size=40)
        if agent.state == "blocked": return AgentPortrayalStyle(color="tab:blue", size=40)
        if agent.state == "stressed": return AgentPortrayalStyle(color="tab:orange", size=40)
        return AgentPortrayalStyle(color="black", size=20)
    
    if isinstance(agent, Microglia):
        if agent.state == "M1": return AgentPortrayalStyle(color="tab:red", size=60)
        if agent.state == "M3": return AgentPortrayalStyle(color="purple", size=60) 
        return AgentPortrayalStyle(color="cornflowerblue", size=60) 

    return AgentPortrayalStyle(color="lightgray", size=10)

def post_process(ax):
    fig = ax.get_figure()
    fig.set_layout_engine(None) 
    ax.set_aspect("equal")

# Only run Solara/Viz setup if this script is executed directly
if __name__ == "__main__":
    
    model_params = {
        "genotype": { "type": "Select", "value": "APOE3 (Control)", "values": list(GENETIC_PROFILES.keys()) },
        "lecanemab_dose": { "type": "SliderFloat", "value": 0.0, "label": "Lecanemab (mg/kg)", "min": 0.0, "max": 10.0, "step": 1.0 },
        "cromolyn_dose": { "type": "SliderFloat", "value": 0.0, "label": "Cromolyn (mg/kg)", "min": 0.0, "max": 100.0, "step": 5.0 },
        "n_neurons": { "type": "SliderInt", "value": 1000, "label": "Neurons", "min": 200, "max": 2000, "step": 100 },
    }

    FIXED_SEED = 374281
    AD_model_instance = ADModel(width=40, height=40, n_neurons=1000, n_microglia=200, seed=FIXED_SEED)

    renderer = SpaceRenderer(model=AD_model_instance, backend="matplotlib").render(
        agent_portrayal=agent_portrayal, post_process=post_process
    )

    chart_atrophy = make_plot_component(["Simulated Atrophy %", "Target Atrophy %"])
    chart_env = make_plot_component("Total Amyloid")

    page = SolaraViz(
        AD_model_instance,
        renderer, 
        components=[chart_atrophy, chart_env],
        model_params=model_params,
        name="Mechanistic AD Model: Spatial Scale",
    )
    
    # In Solara, you usually leave the page object at the module level, 
    # but strictly speaking, Solara looks for the 'page' variable.
    # If importing this file as a module, 'page' won't be exposed at top level, 
    # preventing accidental server launch.
    page