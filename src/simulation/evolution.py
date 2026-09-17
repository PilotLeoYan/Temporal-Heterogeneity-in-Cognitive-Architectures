from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING

import numpy as np

from src.components.agent import AgentRunner
from src.components.architecture import CognitiveArchitecture
from src.components.world import World
from src.config.schema import AppConfig
from src.simulation.reproducibility import set_seed

if TYPE_CHECKING:
    from src.simulation.logger import RunLogger


class EvolutionarySimulation:
    def __init__(
        self,
        config: AppConfig,
        buffer_type: str,
        short_term_mem: bool,
        n_activations: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.c = config
        self.buffer_type = buffer_type
        self.short_term_mem = short_term_mem
        self.seed = seed if seed is not None else self.c.evolution.seed
        self.n_activations = (
            n_activations
            if n_activations is not None
            else self.c.architecture.n_activations
        )

        set_seed(self.seed)

        arch_config = self.c.architecture
        if n_activations is not None:
            arch_config = copy.copy(self.c.architecture)
            arch_config.n_activations = n_activations

        self.architecture = [
            CognitiveArchitecture(
                self.c.p_dims,
                self.c.m_dims,
                self.buffer_type,
                arch_config,
                self.short_term_mem,
            )
            for _ in range(self.c.evolution.pop_size)
        ]

    def one_evolution(
        self,
        verbose: bool = True,
        logger: RunLogger | None = None,
    ) -> list[float]:
        runner = AgentRunner(self.c.agent)
        world = World(self.seed, self.c.world)

        history_top_averages: list[float] = []

        for gen in range(self.c.evolution.n_generations):
            gen_start_time = time.time()
            coins_log: list[tuple[int, int]] = []

            for j, architecture in enumerate(self.architecture):
                runner.reset()
                architecture.reset()
                world.reset()

                coins = runner.run(architecture, world)
                coins_log.append((j, coins))

            coins_log.sort(key=lambda item: item[1], reverse=True)

            n_elite = max(
                1, int(round(self.c.evolution.pop_size * self.c.evolution.elite_ratio))
            )
            elite_indices = [idx for idx, _ in coins_log[:n_elite]]
            rest_indices = [idx for idx, _ in coins_log[n_elite:]]

            k_top = min(len(coins_log), self.c.evolution.top_average_agents)
            top_agents_coins = [coins for _, coins in coins_log[:k_top]]
            top_average = float(np.mean(top_agents_coins))
            history_top_averages.append(top_average)

            max_coins = float(coins_log[0][1])
            min_coins = float(coins_log[-1][1])
            all_coins = [c for _, c in coins_log]
            std_coins = float(np.std(all_coins))

            if logger is not None:
                logger.log_generation(
                    generation=gen + 1,
                    top_mean=top_average,
                    max_fitness=max_coins,
                    min_fitness=min_coins,
                    std_fitness=std_coins,
                    elapsed_sec=time.time() - gen_start_time,
                )

            if verbose:
                print(
                    f"Gen {gen + 1:03d}/{self.c.evolution.n_generations:03d}, "
                    f"Top {k_top} Mean Fitness: {top_average:.2f} coins, "
                    f"Max: {int(max_coins)}"
                )

            if gen == self.c.evolution.n_generations - 1:
                break

            n_elite_offspring = int(
                round(
                    self.c.evolution.pop_size * self.c.evolution.elite_offspring_ratio
                )
            )
            n_rest_offspring = self.c.evolution.pop_size - n_elite_offspring

            selected_elite = np.random.choice(
                elite_indices, size=n_elite_offspring, replace=True
            )
            selected_rest = np.random.choice(
                rest_indices if len(rest_indices) > 0 else elite_indices,
                size=n_rest_offspring,
                replace=True,
            )
            selected_parents = list(selected_elite) + list(selected_rest)

            next_gen_weights: list[dict[str, np.ndarray]] = []
            for parent_idx in selected_parents:
                parent_w = self.architecture[parent_idx].get_weights()
                child_w = {}
                for layer_name, w_matrix in parent_w.items():
                    noise = np.random.uniform(
                        -self.c.evolution.mutation_scale,
                        self.c.evolution.mutation_scale,
                        size=w_matrix.shape,
                    )
                    child_w[layer_name] = w_matrix + noise
                next_gen_weights.append(child_w)

            for arch, new_w in zip(self.architecture, next_gen_weights):
                arch.set_weights(new_w)
                arch.reset()

        return history_top_averages
