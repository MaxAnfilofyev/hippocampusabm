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

# --- 1. CONFIGURATION & DATA ---

class HippocampalLayer(Enum):
    DG = "DG"
    CA3 = "CA3"
    CA1 = "CA1"

GENOTYPE_MAP = {
    "APOE3": "3/3",
    "APOE4": "4/4",
    "TREM2": "2/3",
    "APOE4+TREM2": "2/4",
}

# --- UPDATED WITH YOUR EXACT COEFFICIENTS ---
# Equation: % Loss = at^2 + bt
DATA_CALIBRATION = {
    '3/3': {
        # a = -0.031, b = 0.741
        'atrophy_coeffs': [-0.031, 0.741],
        'years': [0, 20], 'amyloid_rate': [0.02, 0.05], 'gfap_state': [0.6, 0.95] # Placeholders
    },
    '3/4': {
        # a = -0.042, b = 1.015
        'atrophy_coeffs': [-0.042, 1.015],
        'years': [0, 20], 'amyloid_rate': [0.03, 0.35], 'gfap_state': [0.7, 1.0]
    },
    '4/4': {
        # a = -0.109, b = 2.134
        'atrophy_coeffs': [-0.109, 2.134],
        'years': [0, 20], 'amyloid_rate': [0.05, 1.5], 'gfap_state': [0.8, 1.0]
    },
    # Fallbacks for other genotypes using similar averages
    '2/3': { 'atrophy_coeffs': [-0.031, 0.741], 'years': [0, 20], 'amyloid_rate': [0.01, 0.04], 'gfap_state': [0.5, 0.85] },
    '2/4': { 'atrophy_coeffs': [-0.042, 1.015], 'years': [0, 20], 'amyloid_rate': [0.02, 0.18], 'gfap_state': [0.6, 0.95] }
}

# --- 2. HELPER FUNCTIONS ---

def interpolate_rate(years, values, t):
    years = np.array(years)
    values = np.array(values)
    return np.interp(np.clip(t, years.min(), years.max()), years, values)

def build_hippocampal_graph(n_DG, n_CA3, n_CA1, rng):
    G = nx.DiGraph()
    DG_ids = range(0, n_DG)
    CA3_ids = range(n_DG, n_DG + n_CA3)
    CA1_ids = range(n_DG + n_CA3, n_DG + n_CA3 + n_CA1)

    for d in DG_ids:
        k = min(3, len(CA3_ids))
        if k > 0:
            for c in rng.sample(list(CA3_ids), k=k):
                G.add_edge(d, c, weight=rng.uniform(0.4, 0.8))
    for i in CA3_ids:
        for j in CA3_ids:
            if i != j and rng.random() < 0.15:
                G.add_edge(i, j, weight=rng.uniform(0.6, 1.2))
    for c in CA3_ids:
        k = min(4, len(CA1_ids))
        if k > 0:
            for a in rng.sample(list(CA1_ids), k=k):
                G.add_edge(c, a, weight=rng.uniform(0.7, 1.3))
    return G, DG_ids, CA3_ids, CA1_ids

# --- 3. AGENTS ---

class Neuron(FixedAgent):
    def __init__(self, model, cell, node_id, layer):
        super().__init__(model)
        self.cell = cell
        self.node_id = node_id
        self.layer = layer
        self.state = "healthy"

    def step(self):
        if self.state == "dead": return

        # "Stochastic Death" determined by the curve
        # We roll a dice against the calculated probability for this specific timestep
        if self.random.random() < self.model.current_step_death_prob:
            self.state = "dead"
            # Optional: Visual "stress" state before death could be added here
            # But direct death ensures precise volume matching

class Microglia(CellAgent):
    def __init__(self, model, cell):
        super().__init__(model)
        self.cell = cell
        self.state = "M0"

    def step(self):
        # Microglia movement and state logic (Visual candy for now)
        if self.random.random() < 0.2:
            new_cell = self.cell.neighborhood.select_random_cell()
            if new_cell: self.cell = new_cell
        
        # Simple state flip for visual variety
        if self.random.random() < 0.05:
            self.state = "M1" if self.state == "M0" else "M0"

# --- 4. THE MODEL ---

class ADModel(mesa.Model):
    def __init__(self, width=20, height=20, n_neurons=200, n_microglia=80, genotype="APOE3", lecanemab=0.0, cromolyn=0.0, max_steps=500, seed=None):
        super().__init__(seed=seed)
        self.grid = OrthogonalMooreGrid((width, height), torus=False, random=self.random)
        self.simulation_years = 10 # Matches the 10-year curve in your graph
        self.years_per_step = self.simulation_years / max_steps
        self.current_year = 0
        self.current_step_death_prob = 0.0

        calib_key = GENOTYPE_MAP.get(genotype, "3/3")
        self.calib = DATA_CALIBRATION.get(calib_key, DATA_CALIBRATION['3/3'])
        
        # Graph & Agents
        n_DG, n_CA3, n_CA1 = int(n_neurons*0.4), int(n_neurons*0.3), int(n_neurons*0.3)
        self.synapse_graph, DG_ids, CA3_ids, CA1_ids = build_hippocampal_graph(n_DG, n_CA3, n_CA1, self.random)
        
        self.neuron_map = {}
        node_layer_map = ({i: HippocampalLayer.DG for i in DG_ids} | {i: HippocampalLayer.CA3 for i in CA3_ids} | {i: HippocampalLayer.CA1 for i in CA1_ids})

        for node_id, layer in node_layer_map.items():
            cell = self.grid.all_cells.select_random_cell()
            n = Neuron(self, cell, node_id, layer)
            self.neuron_map[node_id] = n
        
        for _ in range(n_microglia):
            cell = self.grid.all_cells.select_random_cell()
            Microglia(self, cell)
        
        self.datacollector = mesa.DataCollector({
            "Alive Neurons": lambda m: sum(1 for n in m.neuron_map.values() if n.state != "dead"),
            "Atrophy %": lambda m: (1 - (sum(1 for n in m.neuron_map.values() if n.state != "dead") / len(m.neuron_map))) * 100
        })
        self.running = True

    def get_expected_volume_fraction(self, t):
        """Calculates expected remaining volume (0.0-1.0) at year t based on curve"""
        a, b = self.calib['atrophy_coeffs']
        # Formula: % Loss = at^2 + bt
        pct_loss = (a * (t**2)) + (b * t)
        # Clamp loss between 0% and 100%
        pct_loss = max(0, min(100, pct_loss))
        return 1.0 - (pct_loss / 100.0)

    def step(self):
        # 1. Calculate Expected Volume Now and Next Step
        t_current = self.current_year
        t_next = self.current_year + self.years_per_step
        
        vol_current = self.get_expected_volume_fraction(t_current)
        vol_next = self.get_expected_volume_fraction(t_next)
        
        # 2. Calculate Required Death Probability
        # If volume drops from 0.99 to 0.98, prob = (0.99 - 0.98) / 0.99
        if vol_current > 0:
            self.current_step_death_prob = (vol_current - vol_next) / vol_current
        else:
            self.current_step_death_prob = 0.0
            
        # Ensure prob is valid (0-1)
        self.current_step_death_prob = max(0.0, min(1.0, self.current_step_death_prob))

        # 3. Advance Time and Agents
        self.current_year += self.years_per_step
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

# --- 5. VISUALIZATION ---

def agent_portrayal(agent):
    if isinstance(agent, Neuron):
        return AgentPortrayalStyle(color="tab:green", size=40) if agent.state == "healthy" else AgentPortrayalStyle(color="black", size=20)
    if isinstance(agent, Microglia):
        return AgentPortrayalStyle(color="cornflowerblue", size=60) if agent.state == "M0" else AgentPortrayalStyle(color="tab:red", size=60)
    return AgentPortrayalStyle(color="lightgray", size=10)

# Fix for Matplotlib Layout Crash in Solara
def post_process(ax):
    fig = ax.get_figure()
    fig.set_layout_engine(None) 
    ax.set_aspect("equal")

model_params = {
    "genotype": {
        "type": "Select", "value": "APOE3", "label": "Genotype",
        "options": ["APOE3", "APOE4", "TREM2", "APOE4+TREM2"]
    },
    "n_neurons": { "type": "SliderInt", "value": 200, "label": "Neurons", "min": 100, "max": 500, "step": 10 },
    "width": 20, "height": 20,
}

# Initialize
AD_model_instance = ADModel(width=20, height=20, n_neurons=200, genotype="APOE3")

# Renderer (Separate from Components)
renderer = SpaceRenderer(model=AD_model_instance, backend="matplotlib").render(
    agent_portrayal=agent_portrayal, post_process=post_process
)

# Charts
chart_atrophy = make_plot_component("Atrophy %")
chart_alive = make_plot_component("Alive Neurons")

page = SolaraViz(
    AD_model_instance,
    renderer, # Correctly placed as 2nd arg
    components=[chart_atrophy, chart_alive],
    model_params=model_params,
    name="Hippocampal Atrophy Simulation",
)

page