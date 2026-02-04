import numpy as np
import pandas as pd
import random
import math
import mesa

#VIS IMPORTS
from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle
from mesa.discrete_space import OrthogonalMooreGrid, CellAgent, FixedAgent

DATA_CALIBRATION = {'3/3': {'years': [1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.5, 3.5, 3.5, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 5.0, 5.0, 5.0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.5, 7.5, 7.5, 8.0, 10.5, 11.0, 11.0, 11.0, 11.0, 11.0, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.5, 12.5, 12.5, 12.5, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.5, 13.5, 14.0, 18.0, 18.0, 18.0, 18.5, 18.5, 19.0], 'amyloid_rate': [0.02564759696969265, 0.028023791566470068, 0.01704430764554802, 0.0, 0.0, 0.0, 0.051531096215421804, 0.07369855969244231, 0.043130846605318386, 0.004573451617572317, 0.0, 0.0, 0.0, 0.0331110862705317, 0.019921940662883095, 0.054470356357593966, 0.09384986731187141, 0.07857257251349971, 0.03389600069169621, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'gfap_state': [0.6773367402816173, 0.6914759154388956, 0.6914759154388956, 0.6914759154388956, 0.6914759154388956, 0.6914759154388956, 0.6914759154388956, 0.6914759154388956, 0.6914759154388956, 0.6914759154388956, 0.6914759154388956, 0.6914759154388956, 0.6914759154388956, 0.6914759154388956, 0.7129904232931358, 0.7129904232931358, 0.7129904232931358, 0.7129904232931358, 0.7129904232931358, 0.7129904232931358, 0.7129904232931358, 0.7129904232931358, 0.7129904232931358, 0.73788567413181, 0.73788567413181, 0.73788567413181, 0.73788567413181, 0.73788567413181, 0.73788567413181, 0.73788567413181, 0.73788567413181, 0.7612890905659155, 0.7612890905659155, 0.7612890905659155, 0.7612890905659155, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8023342908901909, 0.8829845253039252, 0.8829845253039252, 0.8829845253039252, 0.8829845253039252, 0.8829845253039252, 0.8829845253039252, 0.8829845253039252, 0.9620693802106273, 0.9620693802106273, 0.9620693802106273, 0.9983714393304093, 0.9983714393304093, 0.9983714393304093, 0.9983714393304093, 0.9983714393304093, 0.9983714393304093, 0.9983714393304093, 0.9983714393304093, 0.9983714393304093, 0.9983714393304093, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.8992459168267923, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.6572987243485026, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.3842668015307131, 0.08901853776681286, 0.08901853776681286, 0.08901853776681286, -0.059580639002946086, 0.6466508784399325, 0.7065814547301852, 0.7065814547301852, 0.7065814547301852, 0.7065814547301852, 0.7065814547301852, 0.765982650932287, 0.765982650932287, 0.765982650932287, 0.765982650932287, 0.765982650932287, 0.765982650932287, 0.765982650932287, 0.8235557087802514, 0.8235557087802514, 0.8235557087802514, 0.8235557087802514, 0.8235557087802514, 0.8235557087802514, 0.875655685121528, 0.875655685121528, 0.875655685121528, 0.875655685121528, 0.8987740457429261, 0.8987740457429261, 0.8987740457429261, 0.8987740457429261, 0.8987740457429261, 0.8987740457429261, 0.8987740457429261, 0.9038094410101969, 0.9038094410101969, 0.9094106586210285, 0.9833003028950507, 0.9833003028950507, 0.9833003028950507, 0.9919247400046192, 0.9919247400046192, 1.0], 'atrophy_per_year': np.float64(6.703235153798502e-05)}, '3/4': {'years': [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 3.0, 3.0, 3.0, 3.0, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.5, 4.5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 7.0, 7.0, 7.0, 7.0, 7.0, 7.5, 8.0, 9.0, 10.5, 11.0, 11.0, 11.0, 11.0, 11.5, 11.5, 12.0, 12.0, 12.5, 13.0, 13.0, 13.0, 13.0, 13.0, 13.5, 14.5, 17.5, 17.5], 'amyloid_rate': [0.0, 0.0, 0.0, 0.03649505959765513, 0.23834889151731226, 0.31728683681649683, 0.20477937932153936, 0.0, 0.0, 0.04255886103682329, 0.13566837684595384, 0.010245666379558144, 0.0, 0.0, 0.0, 0.24024406129630735, 0.21938111102899874, 0.0744457824623143, 0.021915518186807786, 0.0016546495588495701, 0.0, 0.0, 0.0, 0.0], 'gfap_state': [0.7013339549375971, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7179993261084626, 0.7374776601887171, 0.7374776601887171, 0.7374776601887171, 0.7374776601887171, 0.7374776601887171, 0.7374776601887171, 0.7374776601887171, 0.7627344170172603, 0.7627344170172603, 0.7627344170172603, 0.7627344170172603, 0.8052787553890155, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.83891629733864, 0.8717745762355131, 0.8717745762355131, 0.8764252633633285, 0.8764252633633285, 0.8764252633633285, 0.8764252633633285, 0.8764252633633285, 0.8764252633633285, 0.8764252633633285, 0.8764252633633285, 0.8764252633633285, 0.8166588444162589, 0.8166588444162589, 0.8166588444162589, 0.8166588444162589, 0.8166588444162589, 0.8166588444162589, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.6937820074242522, 0.5686283035711803, 0.5686283035711803, 0.5686283035711803, 0.5686283035711803, 0.5686283035711803, 0.5686283035711803, 0.5686283035711803, 0.5686283035711803, 0.5007119201331709, 0.5007119201331709, 0.5007119201331709, 0.5007119201331709, 0.5007119201331709, 0.5097840659712757, 0.6067591723208281, 0.8086528158167928, 0.9022770972190789, 0.9273745004984169, 0.9273745004984169, 0.9273745004984169, 0.9273745004984169, 0.9604748548288756, 0.9604748548288756, 0.9789063590010643, 0.9789063590010643, 0.9872210600515884, 0.9922288945807765, 0.9922288945807765, 0.9922288945807765, 0.9922288945807765, 0.9922288945807765, 0.996106983965646, 1.0, 0.9810804060176965, 0.9810804060176965], 'atrophy_per_year': np.float64(0.00012508680440807352)}, '2/3': {'years': [2.0, 2.0, 2.5, 3.0, 3.0, 3.0, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.5, 5.5, 6.0, 6.0, 6.0, 6.5, 6.5, 6.5, 6.5, 6.5, 8.5, 11.0, 11.5, 11.5, 11.5, 12.0, 14.0, 14.5], 'amyloid_rate': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.008867892023262006, 0.05311237495160634, 0.04574760396710102, 0.35671665953261356, 0.6952170017989628, 0.6776666666666661, 0.12299999999999998, 0.28], 'gfap_state': [0.9061493150058789, 0.9061493150058789, 0.8166146887339554, 0.7397337666568018, 0.7397337666568018, 0.7397337666568018, 0.7004681079091445, 0.7679742413491103, 0.7679742413491103, 0.7679742413491103, 0.7679742413491103, 0.7679742413491103, 0.7679742413491103, 0.7679742413491103, 0.7679742413491103, 1.0, 0.9636295671157158, 0.742002293671864, 0.742002293671864, 0.742002293671864, 0.5116407645736814, 0.5116407645736814, 0.5116407645736814, 0.5116407645736814, 0.5116407645736814, 0.5246824467757437, 0.44688824878079636, 0.39069609661781823, 0.39069609661781823, 0.39069609661781823, 0.3300454289626558, 0.05220204329190618, -0.0236971143685538], 'atrophy_per_year': np.float64(0.0001921842555129491)}, '4/4': {'years': [2.0, 2.0, 2.0, 2.0, 4.0, 5.0, 5.0, 5.5, 6.0, 6.0, 6.0, 6.0, 6.5, 8.5, 12.5, 13.5], 'amyloid_rate': [0.0, 0.115, 1.7, 0.0025, 0.0, 0.49999999999999994, 0.0, 0.0, 0.0], 'gfap_state': [0.4702713246745659, 0.4702713246745659, 0.4702713246745659, 0.4702713246745659, 0.2494039599460533, 0.561569485190347, 0.561569485190347, 0.605318133376454, 0.6269743240139404, 0.6269743240139404, 0.6269743240139404, 0.6269743240139404, 0.8635946646543847, 0.7670605499743773, 0.9607546756495884, 1.0], 'atrophy_per_year': np.float64(0.0004132711444226352)}, '2/4': {'years': [3.0, 5.0, 6.0], 'amyloid_rate': [0.23, 0.0, 0.0], 'gfap_state': [0.1756015037593985, 1.0, 0.28827067669172934], 'atrophy_per_year': np.float64(0.00026394433321837776)}}
# calibration data is sorted
for g in DATA_CALIBRATION:
    years = np.array(DATA_CALIBRATION[g]["years"])
    # Sort years first
    sort_idx = np.argsort(years)
    years = years[sort_idx]

    # Interpolate amyloid_rate and gfap_state to match years
    amy = np.array(DATA_CALIBRATION[g]["amyloid_rate"])
    gfap = np.array(DATA_CALIBRATION[g]["gfap_state"])

    # If length mismatch, interpolate to match years
    if len(amy) != len(years):
        x_old = np.linspace(years.min(), years.max(), len(amy))
        amy = np.interp(years, x_old, amy)

    if len(gfap) != len(years):
        x_old = np.linspace(years.min(), years.max(), len(gfap))
        gfap = np.interp(years, x_old, gfap)

    DATA_CALIBRATION[g]["years"] = list(years)
    DATA_CALIBRATION[g]["amyloid_rate"] = list(amy)
    DATA_CALIBRATION[g]["gfap_state"] = list(gfap)

# ---------------------------------------------------------

GENOTYPE_MAP = {
    "APOE3": "3/3",
    "APOE4": "4/4",
    "TREM2": "2/3",
    "APOE4+TREM2": "2/4",
}



def interpolate_rate(years, values, t):
    """
    Linearly interpolate a value at time t.
    If t is outside range, clamp to nearest endpoint.
    """
    years = np.array(years)
    values = np.array(values)

    if len(values) == 1:
        return values[0]

    if t <= years.min():
        return values[0]
    if t >= years.max():
        return values[-1]

    return np.interp(t, years, values)

# ---------------------------------------------------------
# Drug effect helper functions
# ---------------------------------------------------------

def lecanemab_clearance_multiplier(dose):
    return 1.0 + 0.8 * (dose / 10.0)

def lecanemab_inflammation_increase(dose):
    return 0.075 * (dose / 10.0)

def cromolyn_cytokine_reduction(dose):
    return 0.5 * (dose / 40.0)

def cromolyn_reduce_M1_transition(dose):
    return 0.5 * (dose / 40.0)

# ---------------------------------------------------------
# DATACOLLECTOR
# ---------------------------------------------------------

def get_alive_neurons(model):
    "counts alive neurons"
    return sum(1 for n in model.neurons if n.health > 0)

def get_dead_neurons(model):
    "Helper function to count dead neurons."""
    return model.neuron_deaths

def get_mean_amyloid(model):
    """Helper function to get mean amyloid."""
    return float(model.amyloid.mean())

def get_mean_cytokine(model):
    """Helper function to get mean cytokine."""
    return float(model.cytokine.mean())

#Each neuron agent

class Neuron(FixedAgent):
    def __init__(self, model, cell):
        super().__init__(model)
        self.cell = cell
        self.health = 1.0
        self.damage = 0.0
        self.state = "healthy"  # healthy, damaged, dead

    def step(self):
        # 1. EARLY EXIT: If dead, do nothing
        if self.state == "dead":
            return

        # 2. GET ENVIRONMENT DATA FIRST
        # We must define 'inflam' before we use it in the logic below
        x, y = self.cell.coordinate
        inflam = self.model.cytokine[x, y]
        
        # 3. CHECK HEALTHY DYNAMICS (Seeding OR Propagation)
        if self.state == "healthy":
            
            # A. Seeding (Inflammation-driven)
            # Only checking if inflammation is high enough
            seeded = False
            if inflam > 0.8:
                if self.model.random.random() < 0.01 * inflam:
                    self.damage += 0.2
                    if self.damage >= 0.3:
                        self.state = "damaged"
                        seeded = True

            # B. Propagation (Neighbor-driven)
            # Only check neighbors if we didn't already get damaged by seeding this step
            if not seeded:
                neighbors = self.cell.neighborhood.cells
                damaged_neighbors = sum(
                    1 for c in neighbors
                    for a in c.agents
                    if isinstance(a, Neuron) and a.state == "damaged"
                )

                if damaged_neighbors >= 2:
                    self.damage += 0.05 * damaged_neighbors
                    if self.damage >= 0.3:
                        self.state = "damaged"

        # 4. CHECK DAMAGED DYNAMICS (Degeneration)
        elif self.state == "damaged":
            self.damage += 0.01  # Constant degeneration once damaged
            
            # Death Threshold
            if self.damage >= 1.0:
                self.state = "dead"
                self.health = 0.0
                self.model.neuron_deaths += 1


# ---------------------------------------------------------
# Microglia Agent (No changes needed)
# ---------------------------------------------------------

class Microglia(CellAgent):
    """ Mobile immune cell with M0/M1/M2 states """
    def __init__(self, model, cell):
        super().__init__(model)
        self.cell = cell
        self.state = "M0"

    def step(self):
        # NEW MOVEMENT
        if self.random.random() < 0.2:
            new_cell = self.cell.neighborhood.select_random_cell()
            if new_cell:
                self.cell = new_cell

        x, y = self.cell.coordinate
        amy = self.model.amyloid[x, y]

        # State transitions -----------------------------------------
        p_M1 = 0.02 * (1 + 3 * min(1.0, amy))
        p_M1 *= (1 - self.model.cromolyn_M1_reduction)
        p_M1 *= (1 + self.model.lecanemab_inflam_increase)

        p_M2 = 0.01 * (1 + 1.5 * min(1.0, amy))

        r = self.random.random()
        if r < p_M1:
            self.state = "M1"
        elif r < (p_M1 + p_M2):
            self.state = "M2"

        # Amyloid phagocytosis --------------------------------------
        if self.state in ["M0", "M2"]:
            eff = 0.02 * self.model.phag_eff
            eff *= self.model.lecanemab_clearance_multiplier
            removed = self.model.amyloid[x, y] * eff
            self.model.amyloid[x, y] = max(0, self.model.amyloid[x, y] - removed)

            if self.state == "M2":
                self.model.cytokine[x, y] *= 0.995

        # Cytokine production ---------------------------------------
        if self.state == "M1":
            inc = 0.005 * (1 - self.model.cromolyn_cytokine_reduction)
            inc *= (1 + self.model.lecanemab_inflam_increase)
            self.model.cytokine[x, y] += inc

# ---------------------------------------------------------
# MODEL CLASS — NOW WITH DATACOLLECTOR
# ---------------------------------------------------------

# ---------------------------
# ADModel (data-driven version)
# ---------------------------
class ADModel(mesa.Model):
    def __init__(
        self,
        width=20,
        height=20,
        n_neurons=200,
        n_microglia=80,
        genotype="3/3",
        lecanemab=0.0,
        cromolyn=0.0,
        max_steps=500,
        seed=None,
        data_calibration=DATA_CALIBRATION,
    ):
        super().__init__(seed=seed)

        # --- Time ---
        self.max_steps = max_steps
        self.simulation_years = 10.0
        self.years_per_step = self.simulation_years / self.max_steps
        self.current_year = 0.0

        # --- Calibration ---
        calib_key = GENOTYPE_MAP.get(genotype, genotype)
        self.calib = data_calibration.get(calib_key, None)

        # Data-driven toxicity scaling
        self.toxicity_sensitivity = (
            5.0 + (self.calib["atrophy_per_year"] * 1e5) if self.calib else 5.0
        )

        # --- Grid ---
        self.grid = OrthogonalMooreGrid((width, height), torus=False, random=self.random)

        # --- Environment ---
        self.amyloid = np.zeros((width, height))
        base_gfap = (
            interpolate_rate(self.calib["years"], self.calib["gfap_state"], 0.0)
            if self.calib else 0.0
        )
        self.cytokine = np.full((width, height), base_gfap)

        # --- Drugs ---
        self.lecanemab_clearance_multiplier = lecanemab_clearance_multiplier(lecanemab)
        self.lecanemab_inflam_increase = lecanemab_inflammation_increase(lecanemab)
        self.cromolyn_cytokine_reduction = cromolyn_cytokine_reduction(cromolyn)
        self.cromolyn_M1_reduction = cromolyn_reduce_M1_transition(cromolyn)

        # --- Phagocytosis efficiency ---
        self.phag_eff = 0.60 if "4" in genotype else 1.0

        # --- Agents ---
        self.neuron_deaths = 0
        neuron_cells = [
            self.grid[self.random.randrange(width), self.random.randrange(height)]
            for _ in range(n_neurons)
        ]
        self.neurons = Neuron.create_agents(self, n_neurons, cell=neuron_cells)

        micro_cells = [
            self.grid[self.random.randrange(width), self.random.randrange(height)]
            for _ in range(n_microglia)
        ]
        self.microglia = Microglia.create_agents(self, n_microglia, cell=micro_cells)

        # --- DataCollector ---
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Alive Neurons": get_alive_neurons,
                "Dead Neurons": get_dead_neurons,
                "Mean Amyloid": get_mean_amyloid,
                "Mean Cytokine": get_mean_cytokine,
            }
        )

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        # --- Update simulation time ---
        self.current_year += self.years_per_step

        # --- Interpolate rates from data ---
        self.amy_rate = (
            interpolate_rate(self.calib["years"], self.calib["amyloid_rate"], self.current_year)
            if self.calib else 0.0
        )
        self.gfap_target = (
            interpolate_rate(self.calib["years"], self.calib["gfap_state"], self.current_year)
            if self.calib else None
        )

        # --- Apply amyloid growth ---
        stress = self.cytokine / (self.cytokine.max() + 1e-6)
        self.amyloid += self.amy_rate * self.years_per_step * (0.5 + stress)

        # --- Adjust cytokine toward GFAP target ---
        if self.gfap_target is not None:
            self.cytokine += (self.gfap_target - self.cytokine) * 0.01

        # --- Step neurons ---
        for neuron in self.neurons:
            neuron.step()

        # --- Step microglia ---
        for micro in self.microglia:
            micro.step()

        # --- Collect data ---
        self.datacollector.collect(self)

        # --- Stop condition ---
        if self.current_year >= self.simulation_years:
            self.running = False



# ---------------------------------------------------------
# --- NEW VISUALIZATION SETUP ---
# ---------------------------------------------------------

def agent_portrayal(agent):
    """
    Defines how to draw each agent.
    """
    if isinstance(agent, Neuron):
        if agent.health > 0:
            return AgentPortrayalStyle(color="green", size=40)
        else:
            return AgentPortrayalStyle(color="black", size=20)
            
    elif isinstance(agent, Microglia):
        state_colors = {
            "M0": "gray",
            "M1": "orange",
            "M2": "cornflowerblue"
        }
        return AgentPortrayalStyle(color=state_colors.get(agent.state, "purple"), size=60)
    
    # Default for any other agent type
    return AgentPortrayalStyle(color="purple", size=10)

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
    width=model_params["width"],
    height=model_params["height"],
    n_neurons=model_params["n_neurons"]["value"],
    n_microglia=model_params["n_microglia"]["value"]
)

# 4. Create the SpaceRenderer
renderer = SpaceRenderer(model=dummy_model, backend="matplotlib").render(
    agent_portrayal=agent_portrayal
)

# 5. Create the SolaraViz page
# IMPORTANT: Pass the CLASS 'ADModel', not the instance 'dummy_model'
Page = SolaraViz(
    model=ADModel,  
    components=[renderer] + charts, # Combine renderer and charts
    model_params=model_params,
    name="Alzheimer's Disease Model",
)

# This file no longer uses the `if __name__ == "__main__":` block
# To run this visualization:
# 1. Save the code above as `viz_model.py`
# 2. Open your terminal or command prompt
# 3. Navigate to the directory where you saved the file
# 4. Run the command: solara run viz_model.py