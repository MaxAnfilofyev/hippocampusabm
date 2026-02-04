import pandas as pd
import numpy as np
import mesa
from mesa.discrete_space import OrthogonalMooreGrid, CellAgent
from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle

# --- Data Loading (Kept from your code) ---
# We wrap this in a try-except to ensure the viz runs even if the CSV is missing during testing
adni_ahv_norm = None
try:
    df = pd.read_csv("ADNI_Integrated_Master.csv")
    df = df.dropna(subset=["GENOTYPE", "Years", "aHV"])

    # Basic smoothing helper
    from statsmodels.nonparametric.smoothers_lowess import lowess
    def smooth_signal(x, frac=0.3):
        t = np.arange(len(x))
        return lowess(x, t, frac=frac, return_sorted=False)

    def collapse_by_year(years, values):
        years = np.asarray(years)
        values = np.asarray(values)
        uniq = np.unique(years)
        collapsed = [values[years == y].mean() for y in uniq]
        return uniq, np.array(collapsed)

    genotype = df["GENOTYPE"].unique()[0]
    sub = df[df["GENOTYPE"] == genotype].sort_values("Years")

    adni_years, adni_ahv = collapse_by_year(sub["Years"].values, sub["aHV"].values)
    adni_ahv = smooth_signal(adni_ahv)
    adni_ahv_norm = adni_ahv / adni_ahv[0]
except FileNotFoundError:
    print("⚠️ ADNI_Integrated_Master.csv not found. Running visualization without comparison data.")

# --- Agent Definition ---
class NeuronAgent(CellAgent):
    def __init__(self, model, cell):
        super().__init__(model)
        self.cell = cell
        self.health = 1.0
        self.alive = True

    def degenerate(self):
        if not self.alive:
            return

        rate = (
            self.model.base_degeneration
            + self.model.amyloid_sensitivity * self.model.amyloid_level
            + self.model.inflammation_sensitivity * self.model.inflammation_level
            + self.model.cytokine_sensitivity * self.model.cytokine_level
        )

        self.health -= rate
        if self.health <= 0:
            self.health = 0
            self.alive = False

    def step(self):
        """Required for the scheduler."""
        self.degenerate()

# --- Model Definition ---
def compute_volume(model):
    """Calculate the percentage of living neurons."""
    if model.num_agents == 0: return 0
    alive = sum(1 for a in model.agents if a.alive)
    return alive / model.num_agents

class HippocampusModel(mesa.Model):
    def __init__(
        self,
        n_neurons=400, # Lower default for visualization performance
        width=20,      # Grid dimensions for visualization
        height=20,
        amyloid_start=0.3,
        amyloid_rate=0.04,
        cytokine_level=0.15,
        inflammation_level=0.2,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.num_agents = n_neurons
        self.current_year = 0
        
        # Model Parameters
        self.amyloid_level = amyloid_start
        self.amyloid_rate = amyloid_rate
        self.cytokine_level = cytokine_level
        self.inflammation_level = inflammation_level

        # Biologic Constants
        self.base_degeneration = 0.002
        self.amyloid_sensitivity = 0.01
        self.cytokine_sensitivity = 0.008
        self.inflammation_sensitivity = 0.01

        # Visualization: Create a Grid
        self.grid = OrthogonalMooreGrid((width, height), random=self.random)

        # Create Agents
        # We define a helper to create agents and place them on the grid
        NeuronAgent.create_agents(
            self,
            self.num_agents,
            self.random.choices(self.grid.all_cells.cells, k=self.num_agents),
        )

        # Data Collection
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Volume": compute_volume,
                "Amyloid Load": lambda m: m.amyloid_level
            }
        )
        self.datacollector.collect(self)

    def step(self):
        self.current_year += 1
        self.amyloid_level += self.amyloid_rate
        
        # Shuffle step (agents degenerate)
        self.agents.shuffle_do("step")
        
        self.datacollector.collect(self)

# --- Visualization Components ---

# 1. Define how agents look
def agent_portrayal(agent):
    # Default style: Healthy (Blue)
    portrayal = AgentPortrayalStyle(
        size=80, 
        color="tab:blue", 
    )
    
    # Dynamic update based on health
    if agent.alive:
        # Fade color based on health? Or just keep blue.
        # Let's turn them Orange as they get sick
        if agent.health < 0.5:
             portrayal.update(("color", "tab:orange"))
    else:
        # Dead (Red/Grey and small)
        portrayal.update(("color", "lightgray"), ("size", 30))
        
    return portrayal

# 2. Define Dashboard Parameters (User Inputs)
model_params = {
    "n_neurons": {
        "type": "SliderInt",
        "value": 400,
        "label": "Number of Neurons",
        "min": 100,
        "max": 1000,
        "step": 50,
    },
    "amyloid_rate": {
        "type": "SliderFloat",
        "value": 0.04,
        "label": "Amyloid Accumulation Rate",
        "min": 0.0,
        "max": 0.1,
        "step": 0.01,
    },
    "inflammation_level": {
        "type": "SliderFloat",
        "value": 0.2,
        "label": "Inflammation Level",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },
    "width": 20,
    "height": 20,
}

# 3. Instantiate the Model and Renderer
model_instance = HippocampusModel()

# Renderer handles the Grid View
renderer = SpaceRenderer(model=model_instance, backend="matplotlib").render(
    agent_portrayal=agent_portrayal
)

# Plot Components handles the Charts
VolumePlot = make_plot_component("Volume")
AmyloidPlot = make_plot_component("Amyloid Load")

# 4. Launch the Solara App
page = SolaraViz(
    model_instance,
    renderer,
    components=[VolumePlot, AmyloidPlot],
    model_params=model_params,
    name="Hippocampus Degeneration Model"
)

# If running in VS Code / Standard Python script, this line does nothing 
# but allows the file to be passed to `solara run`
page