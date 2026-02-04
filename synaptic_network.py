import numpy as np
import mesa
import random
import networkx as nx

# VIS
from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle
from mesa.discrete_space import OrthogonalMooreGrid, FixedAgent, CellAgent

# --- Visualization ---

renderer = SpaceRenderer(
    model=None, backend="matplotlib"
)


from enum import Enum

class HippocampalLayer(Enum):
    DG = "DG"
    CA3 = "CA3"
    CA1 = "CA1"


DATA_CALIBRATION = {'3/3': {'years': [1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.5, 3.5, 3.5, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 5.0, 5.0, 5.0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.5, 7.5, 7.5, 8.0, 10.5, 11.0, 11.0, 11.0, 11.0, 11.0, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.5, 12.5, 12.5, 12.5, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.5, 13.5, 14.0, 18.0, 18.0, 18.0, 18.5, 18.5, 19.0], 'amyloid_rate': [0.02564759696969265, 0.028023791566470068, 0.01704430764554802, 0.0, 0.0, 0.0, 0.051531096215421804, 0.07369855969244231, 0.043130846605318386, 0.004573451617572317, 0.0, 0.0, 0.0, 0.0331110862705317, 0.019921940662883095, 0.054470356357593966, 0.09384986731187141, 0.07857257251349971, 0.03389600069169621, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'gfap_state': [0.6773367402816173, 0.6914759154388956, 0.6914759154388956, 0.6914759154388956, 0.6914759154388956, 0.6914759154388956, 0.6914759154388956, 0.6914759154388956, 0.6914759154388956, 0.6914759154388956, 0.6914759154388956, 0.6914759154388956, 0.6914759154388956, 0.6914759154388956, 0.7129904232931358, 0.7129904232931358, 0.7129904232931358, 0.7129904232931358, 0.7129904232931358, 0.7129904232931358, 0.7129904232931358, 0.7129904232931358, 0.7129904232931358, 0.73788567413181, 0.73788567413181, 0.73788567413181, 0.73788567413181, 0.73788567413181, 0.73788567413181, 0.73788567413181, 0.73788567413181, 0.7612890905659155, 0.7612890905659155, 0.7612890905659155, 0.7612890905659155, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8829845253039252, 0.8829845253039252, 0.8829845253039252, 0.8829845253039252, 0.8829845253039252, 0.8829845253039252, 0.8829845253039252, 0.9620693802106273, 0.9620693802106273, 0.9620693802106273, 0.9983714393304093, 0.9983714393304093, 0.9983714393304093, 0.9983714393304093, 0.9983714393304093, 0.9983714393304093, 0.9983714393304093, 0.9983714393304093, 0.9983714393304093, 0.9983714393304093, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.08901853776681286, 0.08901853776681286, 0.08901853776681286, -0.059580639002946086, 0.6466508784399325, 0.7065814547301852, 0.7065814547301852, 0.7065814547301852, 0.7065814547301852, 0.7065814547301852, 0.765982650932287, 0.765982650932287, 0.765982650932287, 0.765982650932287, 0.765982650932287, 0.765982650932287, 0.765982650932287, 0.8235557087802514, 0.8235557087802514, 0.8235557087802514, 0.8235557087802514, 0.8235557087802514, 0.8235557087802514, 0.875655685121528, 0.875655685121528, 0.875655685121528, 0.875655685121528, 0.8987740457429261, 0.8987740457429261, 0.8987740457429261, 0.8987740457429261, 0.8987740457429261, 0.8987740457429261, 0.8987740457429261, 0.9038094410101969, 0.9038094410101969, 0.9094106586210285, 0.9833003028950507, 0.9833003028950507, 0.9833003028950507, 0.9919247400046192, 0.9919247400046192, 1.0], 'atrophy_per_year': np.float64(6.703235153798502e-05)}, '3/4': {'years': [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 3.0, 3.0, 3.0, 3.0, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.5, 4.5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 7.0, 7.0, 7.0, 7.0, 7.0, 7.5, 8.0, 9.0, 10.5, 11.0, 11.0, 11.0, 11.0, 11.5, 11.5, 12.0, 12.0, 12.5, 13.0, 13.0, 13.0, 13.0, 13.0, 13.5, 14.5, 17.5, 17.5], 'amyloid_rate': [0.0, 0.0, 0.0, 0.03649505959765513, 0.23834889151731226, 0.31728683681649683, 0.20477937932153936, 0.0, 0.0, 0.04255886103682329, 0.13566837684595384, 0.010245666379558144, 0.0, 0.0, 0.0, 0.24024406129630735, 0.21938111102899874, 0.0744457824623143, 0.021915518186807786, 0.0016546495588495701, 0.0, 0.0, 0.0, 0.0], 'gfap_state': [0.7013339549375971, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7374776601887171, 0.7374776601887171, 0.7374776601887171, 0.7374776601887171, 0.7374776601887171, 0.7374776601887171, 0.7374776601887171, 0.7627344170172603, 0.7627344170172603, 0.7627344170172603, 0.7627344170172603, 0.8052787553890155, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.8717745762355131, 0.8717745762355131, 0.8764252633633285, 0.8764252633633285, 0.8764252633633285, 0.8764252633633285, 0.8764252633633285, 0.8764252633633285, 0.8764252633633285, 0.8764252633633285, 0.8764252633633285, 0.8166588444162589, 0.8166588444162589, 0.8166588444162589, 0.8166588444162589, 0.8166588444162589, 0.8166588444162589, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.5686283035711803, 0.5686283035711803, 0.5686283035711803, 0.5686283035711803, 0.5686283035711803, 0.5686283035711803, 0.5686283035711803, 0.5686283035711803, 0.5007119201331709, 0.5007119201331709, 0.5007119201331709, 0.5007119201331709, 0.5007119201331709, 0.5097840659712757, 0.6067591723208281, 0.8086528158167928, 0.9022770972190789, 0.9273745004984169, 0.9273745004984169, 0.9273745004984169, 0.9273745004984169, 0.9604748548288756, 0.9604748548288756, 0.9789063590010643, 0.9789063590010643, 0.9872210600515884, 0.9922288945807765, 0.9922288945807765, 0.9922288945807765, 0.9922288945807765, 0.9922288945807765, 0.996106983965646, 1.0, 0.9810804060176965, 0.9810804060176965], 'atrophy_per_year': np.float64(0.00012508680440807352)}, '2/3': {'years': [2.0, 2.0, 2.5, 3.0, 3.0, 3.0, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.5, 5.5, 6.0, 6.0, 6.0, 6.5, 6.5, 6.5, 6.5, 6.5, 8.5, 11.0, 11.5, 11.5, 11.5, 12.0, 14.0, 14.5], 'amyloid_rate': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.008867892023262006, 0.05311237495160634, 0.04574760396710102, 0.35671665953261356, 0.6952170017989628, 0.6776666666666661, 0.12299999999999998, 0.28], 'gfap_state': [0.9061493150058789, 0.9061493150058789, 0.8166146887339554, 0.7397337666568018, 0.7397337666568018, 0.7397337666568018, 0.7004681079091445, 0.7679742413491103, 0.7679742413491103, 0.7679742413491103, 0.7679742413491103, 0.7679742413491103, 0.7679742413491103, 0.7679742413491103, 0.7679742413491103, 1.0, 0.9636295671157158, 0.742002293671864, 0.742002293671864, 0.742002293671864, 0.5116407645736814, 0.5116407645736814, 0.5116407645736814, 0.5116407645736814, 0.5116407645736814, 0.5246824467757437, 0.44688824878079636, 0.39069609661781823, 0.39069609661781823, 0.39069609661781823, 0.3300454289626558, 0.05220204329190618, -0.0236971143685538], 'atrophy_per_year': np.float64(0.0001921842555129491)}, '4/4': {'years': [2.0, 2.0, 2.0, 2.0, 4.0, 5.0, 5.0, 5.5, 6.0, 6.0, 6.0, 6.0, 6.5, 8.5, 12.5, 13.5], 'amyloid_rate': [0.0, 0.115, 1.7, 0.0025, 0.0, 0.49999999999999994, 0.0, 0.0, 0.0], 'gfap_state': [0.4702713246745659, 0.4702713246745659, 0.4702713246745659, 0.4702713246745659, 0.2494039599460533, 0.561569485190347, 0.561569485190347, 0.605318133376454, 0.6269743240139404, 0.6269743240139404, 0.6269743240139404, 0.6269743240139404, 0.8635946646543847, 0.7670605499743773, 0.9607546756495884, 1.0], 'atrophy_per_year': np.float64(0.0004132711444226352)}, '2/4': {'years': [3.0, 5.0, 6.0], 'amyloid_rate': [0.23, 0.0, 0.0], 'gfap_state': [0.1756015037593985, 1.0, 0.28827067669172934], 'atrophy_per_year': np.float64(0.00026394433321837776)}}

GENOTYPE_MAP = {
    "APOE3": "3/3",
    "APOE4": "4/4",
    "TREM2": "2/3",
    "APOE4+TREM2": "2/4",
}

def interpolate_rate(years, values, t):
    years = np.array(years)
    values = np.array(values)
    if len(values) == 1:
        return values[0]
    return np.interp(np.clip(t, years.min(), years.max()), years, values)

def lecanemab_clearance_multiplier(dose):
    return 1.0 + 0.6 * (dose / 10)

def lecanemab_inflammation_increase(dose):
    return 0.03 * (dose / 10)

def cromolyn_cytokine_reduction(dose):
    return 0.4 * (dose / 40)

def cromolyn_reduce_M1_transition(dose):
    return 0.5 * (dose / 40)


class Neuron(FixedAgent):
    def __init__(self, model, cell, node_id, layer):
        super().__init__(model)
        self.cell = cell
        self.node_id = node_id
        self.layer = layer

        self.health = 1.0
        self.synaptic_load = 0.0
        self.state = "healthy"

        # Layer vulnerability (biologically grounded)
        self.vulnerability = {
            HippocampalLayer.DG: 0.6,
            HippocampalLayer.CA3: 1.0,
            HippocampalLayer.CA1: 1.4,
        }[layer]

    def step(self):
        if self.state == "dead":
            return

        x, y = self.cell.coordinate
        inflam = self.model.cytokine[x, y]

        incoming = 0.0
        for src in self.model.synapse_graph.predecessors(self.node_id):
            w = self.model.synapse_graph[src][self.node_id]["weight"]
            neighbor = self.model.neuron_map[src]
            if neighbor.state != "dead":
                incoming += w * (1 - neighbor.health)

        # Synaptic stress accumulation
        self.synaptic_load += (
            0.01 * incoming * self.vulnerability
            + 0.002 * inflam * self.vulnerability
        )

        # State transitions
        if self.state == "healthy" and self.synaptic_load > 0.4:
            self.state = "stressed"

        if self.state == "stressed":
            self.health -= 0.004 * self.synaptic_load

        if self.health <= 0.05:
            self.state = "dead"
            self.health = 0.0
            self.model.neuron_deaths[self.layer] += 1



class Microglia(CellAgent):
    def __init__(self, model, cell):
        super().__init__(model)
        self.cell = cell
        self.state = "M0"

    def step(self):
        if self.random.random() < 0.2:
            new_cell = self.cell.neighborhood.select_random_cell()
            if new_cell:
                self.cell = new_cell

        x, y = self.cell.coordinate
        amy = self.model.amyloid[x, y]

        p_M1 = 0.01 * amy * (1 - self.model.cromolyn_M1_reduction)
        p_M2 = 0.01

        r = self.random.random()
        if r < p_M1:
            self.state = "M1"
        elif r < p_M1 + p_M2:
            self.state = "M2"

        if self.state == "M1":
            self.model.cytokine[x, y] += 0.002

        if self.state == "M2":
            self.model.amyloid[x, y] *= 0.98
            self.model.cytokine[x, y] *= 0.995


class ADModel(mesa.Model):
    def __init__(
        self,
        width=20,
        height=20,
        n_DG=80,
        n_CA3=60,
        n_CA1=60,
        genotype="APOE3",
        max_steps=500,
        seed=None,
        data_calibration=None,
    ):
        super().__init__(seed=seed)

        self.grid = OrthogonalMooreGrid((width, height), torus=False, random=self.random)

        # --- Time ---
        self.simulation_years = 10
        self.years_per_step = self.simulation_years / max_steps
        self.current_year = 0

        # --- ADNI calibration ---
        self.calib = data_calibration.get(genotype)

        # --- Environment ---
        self.amyloid = np.zeros((width, height))
        self.cytokine = np.full(
            (width, height),
            interpolate_rate(self.calib["years"], self.calib["gfap_state"], 0),
        )

        # --- Synaptic graph ---
        self.synapse_graph, DG_ids, CA3_ids, CA1_ids = build_hippocampal_graph(
            n_DG, n_CA3, n_CA1, self.random
        )

        self.neuron_map = {}
        self.neuron_deaths = {
            HippocampalLayer.DG: 0,
            HippocampalLayer.CA3: 0,
            HippocampalLayer.CA1: 0,
        }

        # --- Create neurons ---
        node_layer_map = (
            {i: HippocampalLayer.DG for i in DG_ids}
            | {i: HippocampalLayer.CA3 for i in CA3_ids}
            | {i: HippocampalLayer.CA1 for i in CA1_ids}
        )

        for node_id, layer in node_layer_map.items():
            cell = self.grid[self.random.randrange(width), self.random.randrange(height)]
            n = Neuron(self, cell, node_id, layer)
            self.neuron_map[node_id] = n

        self.agents += list(self.neuron_map.values())

        self.running = True

        def step(self):
            self.current_year += self.years_per_step

            amy_rate = interpolate_rate(
                self.calib["years"], self.calib["amyloid_rate"], self.current_year
            )
            gfap_target = interpolate_rate(
                self.calib["years"], self.calib["gfap_state"], self.current_year
            )

            # Global amyloid accumulation
            self.amyloid += amy_rate * self.years_per_step

            # Slow inflammatory drift (NO explosion)
            self.cytokine += 0.01 * (gfap_target - self.cytokine)

            for agent in self.agents:
                agent.step()


def agent_portrayal(agent):
    if isinstance(agent, Neuron):
        color = {"healthy": "green", "stressed": "orange", "dead": "black"}[agent.state]
        return AgentPortrayalStyle(color=color, size=40)

    if isinstance(agent, Microglia):
        return AgentPortrayalStyle(color="cornflowerblue", size=60)

    return AgentPortrayalStyle(color="gray", size=10)

def build_hippocampal_graph(n_DG, n_CA3, n_CA1, rng):
    G = nx.DiGraph()

    # Node IDs
    DG_ids = range(0, n_DG)
    CA3_ids = range(n_DG, n_DG + n_CA3)
    CA1_ids = range(n_DG + n_CA3, n_DG + n_CA3 + n_CA1)

    # DG → CA3 (sparse)
    for d in DG_ids:
        for c in rng.sample(list(CA3_ids), k=3):
            G.add_edge(d, c, weight=rng.uniform(0.4, 0.8))

    # CA3 recurrent (dense)
    for i in CA3_ids:
        for j in CA3_ids:
            if i != j and rng.random() < 0.15:
                G.add_edge(i, j, weight=rng.uniform(0.6, 1.2))

    # CA3 → CA1 (strong feedforward)
    for c in CA3_ids:
        for a in rng.sample(list(CA1_ids), k=4):
            G.add_edge(c, a, weight=rng.uniform(0.7, 1.3))

    return G, DG_ids, CA3_ids, CA1_ids


# 1. Define the model parameters for the dashboard
model_params = {
    "n_neurons": {
        "type": "SliderInt", "value": 200, "label": "Number of Neurons:",
        "min": 50, "max": 500, "step": 10,
    },
    "n_microglia": {
        "type": "SliderInt", "value": 80, "label": "Number of Microglia:",
        "min": 20, "max": 200, "step": 5,
    },
    "genotype": {
        "type": "Select", "value": "APOE3", "label": "Genotype:",
        "options": ["APOE3", "APOE4", "TREM2", "APOE4+TREM2"],
    },
    "lecanemab": {
        "type": "SliderFloat", "value": 0.0, "label": "Lecanemab Dose:",
        "min": 0.0, "max": 20.0, "step": 1.0,
    },
    "cromolyn": {
        "type": "SliderFloat", "value": 0.0, "label": "Cromolyn Dose:",
        "min": 0.0, "max": 40.0, "step": 2.0,
    },
    # Fix width and height for this visualization
    "width": 20,
    "height": 20,
}

# 2. Create chart components
# These 'page=1' components will appear on a separate tab
charts = [
    make_plot_component("Alive Neurons", page=1),
    make_plot_component("Dead Neurons", page=1),
    make_plot_component("Mean Amyloid", page=1),
    make_plot_component("Mean Cytokine", page=1),
]

# 3. Create a dummy model instance ONLY for the renderer to initialize
# This is required so the renderer knows grid dimensions
dummy_model = ADModel(
    width=20,
    height=20,
    n_DG=80,
    n_CA3=60,
    n_CA1=60,
    genotype="APOE3",
)


# 4. Create the SpaceRenderer
renderer = SpaceRenderer(
    model=dummy_model,
    backend="matplotlib"
).render(agent_portrayal=agent_portrayal)

# 5. Create the SolaraViz page
# IMPORTANT: Pass the CLASS 'ADModel', not the instance 'dummy_model'
Page = SolaraViz(
    model=ADModel,
    components=[renderer],
    model_params={
        "n_DG": 80,
        "n_CA3": 60,
        "n_CA1": 60,
        "genotype": {
            "type": "Select",
            "value": "APOE3",
            "options": ["APOE3", "APOE4"]
        }
    },
    name="Multilayer Hippocampus AD Model",
)

