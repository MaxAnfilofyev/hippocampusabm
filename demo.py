import mesa
import numpy as np
# from mesa.time import RandomActivation  <- This module no longer exists

# ---------- AGENTS ----------

class Neuron(mesa.Agent):
    """Neuron agent with health and amyloid load"""

    # Note: unique_id is no longer passed in __init__
    # The 'create_agents' classmethod handles it.
    def __init__(self, model):
        super().__init__(model)
        self.health = 100
        self.amyloid = self.random.uniform(10, 20) # Use self.random
        self.alive = True

    def step(self):
        if not self.alive:
            return

        # Amyloid-induced damage
        self.health -= 0.05 * self.amyloid

        # Inflammation damage from microglia
        # We can access the model's agent lists directly
        for mg in self.model.microglia:
            self.health -= 0.01 * mg.cytokines

        # Death check
        if self.health <= 0:
            self.alive = False


class Microglia(mesa.Agent):
    """Microglia with inflammatory state & clearance ability"""

    # Note: unique_id is no longer passed in __init__
    def __init__(self, model, genotype):
        super().__init__(model)

        self.genotype = genotype
        self.cytokines = 1.0

        self.clearance_factor = model.genotype_modifiers[genotype]["clearance"]
        self.inflammation_factor = model.genotype_modifiers[genotype]["inflammation"]

    def step(self):
        # Amyloid sensed in environment
        # Access the list of neurons from the model
        alive_neurons = [n for n in self.model.neurons if n.alive]
        if alive_neurons:
            local_amyloid = np.mean([n.amyloid for n in alive_neurons])
        else:
            local_amyloid = 0

        # Increase inflammation
        self.cytokines += 0.001 * local_amyloid * self.inflammation_factor

        # Cromolyn reduces cytokines
        self.cytokines *= (1 - self.model.cromolyn_dose)

        # Amyloid clearance based on Lecanemab
        clearance_rate = (0.01 * self.clearance_factor *
                          (1 + self.model.lecanemab_dose))

        for n in self.model.neurons:
            n.amyloid = max(0, n.amyloid - clearance_rate)


# ---------- MODEL ----------

class BrainModel(mesa.Model):
    def __init__(self,
                 n_neurons=50,
                 n_microglia=10,
                 genotype="APOE4",
                 lecanemab_dose=0.0,
                 cromolyn_dose=0.0,
                 seed=None):

        super().__init__(seed=seed)

        self.lecanemab_dose = lecanemab_dose
        self.cromolyn_dose = cromolyn_dose

        # Genotype parameters
        self.genotype_modifiers = {
            "APOE3":       {"clearance": 1.0,  "inflammation": 1.0},
            "APOE4":       {"clearance": 0.6,  "inflammation": 1.3},
            "TREM2":       {"clearance": 0.4,  "inflammation": 1.1},
            "APOE4_TREM2": {"clearance": 0.25, "inflammation": 1.5}
        }

        # --- NEW AGENT CREATION ---
        # No more self.schedule!
        # Use the .create_agents() classmethod, which automatically
        # adds them to the model's 'self.agents' AgentSet.
        
        # We store them in lists just for easy access in the step methods
        self.neurons = Neuron.create_agents(self, n_neurons)
        
        # We can pass extra kwargs to __init__ through create_agents
        self.microglia = Microglia.create_agents(
            self, n_microglia, genotype=genotype
        )


    def step(self):
        # --- NEW SCHEDULING ---
        # Instead of self.schedule.step(), you call a method
        # on the model's 'self.agents' AgentSet.
        # 'shuffle_do' is the new 'RandomActivation'
        self.agents.shuffle_do("step")


# ---------- RUN SIMULATION ----------

model = BrainModel(
    n_neurons=50,
    genotype="APOE4",
    lecanemab_dose=0.5,
    cromolyn_dose=0.5
)

for i in range(200):
    model.step()

survival = sum(n.alive for n in model.neurons)
print(f"Simulation complete. Neuron survival at step 200: {survival}")