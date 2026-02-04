import numpy as np
import pandas as pd
import random
import math
import mesa
from mesa.discrete_space import OrthogonalMooreGrid, CellAgent, FixedAgent

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
# Neuron Agent
# ---------------------------------------------------------

class Neuron(FixedAgent):
    """ Stationary neuron with health and exposure to amyloid/inflammation """
    def __init__(self, model, cell):
        super().__init__(model)  # <-- Pass ONLY the model to super()
        self.cell = cell          # <-- Assign the cell manually
        self.health = 1.0

    def step(self):
        if self.health <= 0:
            return

        x, y = self.cell.coordinate  # <-- THE FIX
        amy = self.model.amyloid[x, y]
        inflam = self.model.cytokine[x, y]

        exposure = amy + inflam
        death_prob = 1 / (1 + math.exp(-(exposure - 0.2) * 5))

        if self.random.random() < death_prob:
            self.health = 0.0
            self.model.neuron_deaths += 1


# ---------------------------------------------------------
# Microglia Agent
# ---------------------------------------------------------

class Microglia(CellAgent):
    """ Mobile immune cell with M0/M1/M2 states """
    def __init__(self, model, cell):
        super().__init__(model)  # <-- Pass ONLY the model to super()
        self.cell = cell          # <-- Assign the cell manually
        self.state = "M0"

    def step(self):
        # NEW MOVEMENT
        if self.random.random() < 0.2:
            # 1. Get neighbor *cells* from the agent's current cell
            #    (self.cell.neighborhood already knows its grid)
            # 2. select_random_cell() returns a cell object
            new_cell = self.cell.neighborhood.select_random_cell()
            
            # 3. To "move", just assign the new cell to self.cell
            if new_cell:
                self.cell = new_cell

        x, y = self.cell.coordinate
        amy = self.model.amyloid[x, y]
        # inflam = self.model.cytokine[x, y]

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
# MODEL CLASS — MESA 3.x COMPATIBLE
# ---------------------------------------------------------

class ADModel(mesa.Model):
    def __init__(
        self,
        width=20, height=20,
        n_neurons=200, n_microglia=80,
        genotype="APOE3",
        lecanemab=0.0,
        cromolyn=0.0,
        max_steps=500,
        seed=None
    ):
        super().__init__(seed=seed)

        # Grid ------------------------------------------------------
        # Note: 'OrthogonalGrid' and dimensions are passed as a tuple
        self.grid = OrthogonalMooreGrid((width, height), torus=False, random=self.random)        # Environment layers ----------------------------------------
        self.amyloid = np.zeros((width, height))
        self.cytokine = np.zeros((width, height))

        # Drug parameters -------------------------------------------
        self.lecanemab_clearance_multiplier = lecanemab_clearance_multiplier(lecanemab)
        self.lecanemab_inflam_increase = lecanemab_inflammation_increase(lecanemab)
        self.cromolyn_cytokine_reduction = cromolyn_cytokine_reduction(cromolyn)
        self.cromolyn_M1_reduction = cromolyn_reduce_M1_transition(cromolyn)

        # Genetic effects -------------------------------------------
        if genotype == "APOE4":
            self.phag_eff = 1 - 0.40
        elif genotype == "TREM2":
            self.phag_eff = 1 - 0.65
        elif genotype == "APOE4+TREM2":
            self.phag_eff = 1 - 0.40 - 0.65
        else:
            self.phag_eff = 1.0

        self.max_steps = max_steps
        self.step_count = 0
        self.neuron_deaths = 0

        # Agent creation (Mesa 3.x) ---------------------------------
        # 1. Get a list of *cell objects* for neurons
        #    You get a cell object from a grid using grid[x, y]
        neuron_cells = [
            self.grid[self.random.randrange(width), self.random.randrange(height)]
            for _ in range(n_neurons)
        ]
        # 2. Pass this list to create_agents using the 'cell' kwarg
        self.neurons = Neuron.create_agents(self, n_neurons, cell=neuron_cells)
        # NO MORE 'for' LOOP OR 'place_agent' NEEDED

        # 3. Do the same for microglia
        microglia_cells = [
        self.grid[self.random.randrange(width), self.random.randrange(height)]
            for _ in range(n_microglia)
        ]
        self.microglia = Microglia.create_agents(self, n_microglia, cell=microglia_cells)
        # NO MORE 'for' LOOP OR 'place_agent' NEEDED


    # --------------------------------------------------------------
    # ONE MODEL STEP
    # --------------------------------------------------------------
    def step(self):
        self.step_count += 1

        # Amyloid baseline accumulation
        self.amyloid += 0.01

        # Agent steps
        self.agents.shuffle_do("step")

        # Check for stopping condition
        if self.step_count >= self.max_steps:
            self.running = False  # This is the new way to stop the model

    # --------------------------------------------------------------
    # EXPORT RESULTS
    # --------------------------------------------------------------
    def get_results(self):
        alive = sum(1 for n in self.neurons if n.health > 0)
        return {
            "alive_neurons": alive,
            "dead_neurons": self.neuron_deaths,
            "final_mean_amyloid": float(self.amyloid.mean()),
            "final_mean_cytokine": float(self.cytokine.mean()),
        }


# ---------------------------------------------------------
# Running the model as a script
# ---------------------------------------------------------

if __name__ == "__main__":
    model = ADModel(
        genotype="APOE4",
        lecanemab=5.0,
        cromolyn=10.0,
        max_steps=300
    )

    while model.running:
        model.step()

    df = pd.DataFrame([model.get_results()])
    df.to_csv("results.csv", index=False)
    print("Simulation complete → results.csv generated.")
