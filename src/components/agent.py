from src.components.architecture import CognitiveArchitecture
from src.components.world import World
from src.config.schema import AgentConfig


class AgentRunner:
    def __init__(self, config: AgentConfig) -> None:
        self.c = config

        self.reset()

    def reset(self) -> None:
        self.coins = 0
        self.step_count = 0
        self.energy = self.c.initial_energy

    def run(
        self,
        architecture: CognitiveArchitecture,
        world: World,
    ) -> int:
        architecture.reset()

        while self.energy > 0 and self.step_count < self.c.max_agent_steps:
            self.step_count += 1
            sensory = world.extract_sensory_vector(self.energy, self.coins)

            action, active_layers = architecture.step(sensory)
            self.energy -= (
                self.c.move_cost + active_layers * self.c.thinking_cost_per_layer
            )

            if action is None:
                break

            out = world.step(action)
            if out[0]:
                self.energy = min(
                    self.c.max_energy, self.energy + self.c.food_energy_gain
                )
            if out[1]:
                self.coins += 1

        return self.coins
