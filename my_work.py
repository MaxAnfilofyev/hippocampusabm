import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import mesa
from mesa.visualization.components import AgentPortrayalStyle
from mesa.discrete_space import OrthogonalMooreGrid, CellAgent, FixedAgent

# # VIS
# from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
# from mesa.visualization.components import AgentPortrayalStyle
# from mesa.discrete_space import OrthogonalMooreGrid, FixedAgent, CellAgent


# Load ADNI data
df = pd.read_csv("ADNI_Integrated_Master.csv")
df = df.dropna(subset=["GENOTYPE", "Years", "aHV"])

def collapse_by_year(years, values):
    years = np.asarray(years)
    values = np.asarray(values)
    uniq = np.unique(years)
    collapsed = [values[years == y].mean() for y in uniq]
    return uniq, np.array(collapsed)


def smooth_signal(x, frac=0.3):
    t = np.arange(len(x))
    return lowess(x, t, frac=frac, return_sorted=False)

# ---- SELECT ONE GENOTYPE FOR CALIBRATION ----
genotype = df["GENOTYPE"].unique()[0]
sub = df[df["GENOTYPE"] == genotype].sort_values("Years")

adni_years, adni_ahv = collapse_by_year(
    sub["Years"].values,
    sub["aHV"].values
)

adni_ahv = smooth_signal(adni_ahv)

# Normalize ADNI aHV to [0, 1]
adni_ahv_norm = adni_ahv / adni_ahv[0]

class Neuron(FixedAgent):
    def __init__(self, model, cell):
        super().__init__(model)
        self.cell = cell
        self.health = 1.0
        self.state = "healthy"  # healthy, damaged, dead
        self.damage = 0.0

    def step(self):
        if self.state == "dead": return

        x, y = self.cell.coordinate
        amy = self.model.amyloid[x, y]
        inflam = self.model.cytokine[x, y]

        # Damage based on amyloid + cytokine directly
        damage_factor = amy * 0.3 + inflam * 0.2
        self.damage += damage_factor * self.model.years_per_step

        if self.damage >= 1.0:
            self.state = "dead"
            self.health = 0.0
            self.model.neuron_deaths += 1
        elif self.damage >= 0.3:
            self.state = "damaged"

class HippocampusModel(mesa.Model):

    def __init__(
        self,
        n_neurons=50_000,
        amyloid_start=0.3,
        amyloid_rate=0.04,
        cytokine_level=0.15,
        inflammation_level=0.2,
        seed=None,
    ):
        super().__init__(seed=seed)

        self.current_year = 0

        self.amyloid_level = amyloid_start
        self.amyloid_rate = amyloid_rate
        self.cytokine_level = cytokine_level
        self.inflammation_level = inflammation_level

        self.base_degeneration = 0.002
        self.amyloid_sensitivity = 0.01
        self.cytokine_sensitivity = 0.008
        self.inflammation_sensitivity = 0.01

        Neuron.create_agents(self, n=n_neurons)
        self.history = []

    def step(self):
        self.current_year += 1
        self.amyloid_level += self.amyloid_rate

        self.agents.shuffle_do("degenerate")

        alive = sum(a.alive for a in self.agents)
        self.history.append({
            "Year": self.current_year,
            "Volume": alive / len(self.agents)
        })


model = HippocampusModel(
    n_neurons=50_000,
    amyloid_start=0.3,
    amyloid_rate=0.04,
    cytokine_level=0.15,
    inflammation_level=0.2,
    seed=42,
)

for _ in range(int(np.ceil(adni_years.max()))):
    model.step()

sim_years = np.array([h["Year"] for h in model.history])
sim_volume = np.array([h["Volume"] for h in model.history])

sim_interp = np.interp(adni_years, sim_years, sim_volume)

sim_interp = np.interp(
    adni_years,
    sim_years,
    sim_volume
)

def loss(simulated, observed):
    return np.mean((simulated - observed) ** 2)

mse = loss(sim_interp, adni_ahv_norm)
print("MSE:", mse)

