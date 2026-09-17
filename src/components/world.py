import numpy as np

from src.config.schema import WorldConfig


class World:
    MOVE_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(self, seed: int, config: WorldConfig) -> None:
        self.rng = np.random.default_rng(seed)
        self.c = config
        vr = self.c.vision_radius
        self.window_size = vr * 2 + 1
        self.neighbor_mask = np.ones((self.window_size, self.window_size), dtype=bool)
        self.neighbor_mask[vr, vr] = False
        self.n_neighbors = int(np.sum(self.neighbor_mask))
        self.sensory_dim = self.n_neighbors * 3 + 2

        self.reset()

    def reset(self) -> None:
        self.generate_world()

        vr = self.c.vision_radius
        playable_grid = self.grid[
            vr : vr + self.c.grid_size, vr : vr + self.c.grid_size
        ]
        empty_indices = np.argwhere(playable_grid == self.c.id_elements[0])
        if len(empty_indices) > 0:
            choice = empty_indices[self.rng.integers(0, len(empty_indices))]
            self.position = [int(choice[0] + vr), int(choice[1] + vr)]
        else:
            self.position = [vr, vr]

    def generate_world(self) -> None:
        weights = [
            self.c.empty_prob,
            self.c.food_prob,
            self.c.coin_prob,
            self.c.barrier_prob,
        ]
        elements = np.array(self.c.id_elements)

        grid = self.rng.choice(
            elements, size=(self.c.grid_size, self.c.grid_size), p=weights
        )
        self.grid = np.pad(
            grid, pad_width=self.c.vision_radius, mode="constant", constant_values=3
        )

    def extract_sensory_vector(
        self,
        energy: float,
        coins: int,
    ) -> np.ndarray:
        vr = self.c.vision_radius
        py, px = self.position[0], self.position[1]
        window = self.grid[py - vr : py + vr + 1, px - vr : px + vr + 1]
        neighbors = window[self.neighbor_mask]

        n = self.n_neighbors
        sensory = np.empty(self.sensory_dim, dtype=np.float64)
        sensory[:n] = neighbors == 1
        sensory[n : 2 * n] = neighbors == 2
        sensory[2 * n : 3 * n] = neighbors == 3
        sensory[3 * n] = energy / 100.0
        sensory[3 * n + 1] = float(coins) / 50.0
        return sensory

    def step(self, action: int) -> tuple[bool, bool]:
        if action < 0 or action > 3:
            raise ValueError(f"Invalid action: {action}")

        dy, dx = self.MOVE_DELTAS[action]
        future_y = self.position[0] + dy
        future_x = self.position[1] + dx

        target = self.grid[future_y, future_x]
        # if it is a barrier
        if target == self.c.id_elements[3]:
            return False, False

        is_a_food = target == self.c.id_elements[1]
        is_a_coin = target == self.c.id_elements[2]

        self.grid[future_y, future_x] = self.c.id_elements[0]
        self.position = [future_y, future_x]
        return is_a_food, is_a_coin
