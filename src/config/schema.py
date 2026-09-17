from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentConfig(BaseModel):
    max_agent_steps: int = 10_000
    initial_energy: float = 100.0
    move_cost: float = 0.5
    thinking_cost_per_layer: float = 0.0833
    max_energy: float = 100.0
    food_energy_gain: float = 25.0

    @model_validator(mode="after")
    def validate_agent(self) -> AgentConfig:
        if self.max_agent_steps <= 0:
            raise ValueError(
                f"max_agent_steps must be positive, got {self.max_agent_steps}"
            )
        if self.initial_energy <= 0 or self.max_energy <= 0:
            raise ValueError("initial_energy and max_energy must be positive")
        if self.initial_energy > self.max_energy:
            raise ValueError(
                f"initial_energy ({self.initial_energy}) cannot exceed "
                f"max_energy ({self.max_energy})"
            )
        if self.move_cost < 0:
            raise ValueError(f"move_cost cannot be negative, got {self.move_cost}")
        if self.thinking_cost_per_layer < 0:
            raise ValueError(
                "thinking_cost_per_layer cannot be negative, "
                f"got {self.thinking_cost_per_layer}"
            )
        if self.food_energy_gain < 0:
            raise ValueError(
                f"food_energy_gain cannot be negative, got {self.food_energy_gain}"
            )
        return self


class CognitiveArchitectureConfig(BaseModel):
    weight_init: float = 1e-4
    learning_rate: float = 1e-3
    n_activations: int = 3

    @model_validator(mode="after")
    def validate_arch(self) -> CognitiveArchitectureConfig:
        if self.weight_init <= 0:
            raise ValueError(f"weight_init must be positive, got {self.weight_init}")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.n_activations < 1:
            raise ValueError(
                f"n_activations must be at least 1, got {self.n_activations}"
            )
        return self


class PerceptualDims(BaseModel):
    p1_dims: tuple[int, int, int, int, int] = (97, 39, 74, 4, 19)
    p2_dims: tuple[int, int, int, int, int] = (56, 19, 39, 8, 9)
    p3_dims: tuple[int, int, int, int, int] = (28, 9, 19, 9, 0)
    dims: list[tuple[int, int, int, int, int]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dims(self) -> PerceptualDims:
        if not self.dims:
            self.dims = [self.p1_dims, self.p2_dims, self.p3_dims]
        for idx, dim in enumerate(self.dims):
            total, _out, bottom, lateral, top = dim
            if total != bottom + lateral + top:
                raise ValueError(
                    f"P-layer {idx + 1} total input mismatch: total ({total}) != "
                    f"bottom ({bottom}) + lateral ({lateral}) + top ({top})"
                )
        return self


class MotorDims(BaseModel):
    m1_dims: tuple[int, int, int, int, int] = (47, 4, 0, 39, 8)
    m2_dims: tuple[int, int, int, int, int] = (32, 8, 4, 19, 9)
    m3_dims: tuple[int, int, int, int, int] = (17, 9, 8, 9, 0)
    dims: list[tuple[int, int, int, int, int]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dims(self) -> MotorDims:
        if not self.dims:
            self.dims = [self.m1_dims, self.m2_dims, self.m3_dims]
        for idx, dim in enumerate(self.dims):
            total, _out, bottom, lateral, top = dim
            if total != bottom + lateral + top:
                raise ValueError(
                    f"M-layer {idx + 1} total input mismatch: total ({total}) != "
                    f"bottom ({bottom}) + lateral ({lateral}) + top ({top})"
                )
        return self


class WorldConfig(BaseModel):
    empty_prob: float = 0.26
    food_prob: float = 0.23
    coin_prob: float = 0.25
    barrier_prob: float = 0.26
    id_elements: tuple[int, int, int, int] = (0, 1, 2, 3)
    grid_size: int = 50
    vision_radius: int = 2

    @model_validator(mode="after")
    def validate_world(self) -> WorldConfig:
        total = self.empty_prob + self.food_prob + self.coin_prob + self.barrier_prob
        if not abs(total - 1.0) < 1e-5:
            raise ValueError(f"World item probabilities must sum to 1.0, got: {total}")
        if self.grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got: {self.grid_size}")
        if self.vision_radius <= 0:
            raise ValueError(
                f"vision_radius must be positive, got: {self.vision_radius}"
            )
        if len(self.id_elements) != 4:
            raise ValueError(
                "id_elements must contain exactly 4 elements, "
                f"got: {len(self.id_elements)}"
            )
        return self


class EvolutionConfig(BaseModel):
    pop_size: int = 100
    elite_ratio: float = 0.10
    elite_offspring_ratio: float = 0.70
    rest_offspring_ratio: float = 0.30
    mutation_scale: float = 1e-4
    n_generations: int = 100
    seed: int = 42
    top_average_agents: int = 10

    @model_validator(mode="after")
    def validate_evolution(self) -> EvolutionConfig:
        if self.pop_size < 1:
            raise ValueError(f"pop_size must be at least 1, got {self.pop_size}")
        if not 0.0 < self.elite_ratio <= 1.0:
            raise ValueError(
                f"elite_ratio must be between 0 and 1, got {self.elite_ratio}"
            )
        offspring_total = self.elite_offspring_ratio + self.rest_offspring_ratio
        if not abs(offspring_total - 1.0) < 1e-5:
            raise ValueError(
                "elite_offspring_ratio + rest_offspring_ratio must sum to 1.0, "
                f"got {offspring_total}"
            )
        if self.mutation_scale < 0:
            raise ValueError(
                f"mutation_scale cannot be negative, got {self.mutation_scale}"
            )
        if self.n_generations < 1:
            raise ValueError(
                f"n_generations must be at least 1, got {self.n_generations}"
            )
        if self.top_average_agents < 1:
            raise ValueError(
                f"top_average_agents must be at least 1, got {self.top_average_agents}"
            )
        return self


class SimulationConfig(BaseModel):
    generations: int = 100
    num_runs: int = 1
    results_dir: str = "results"
    buffer_types: list[str] = Field(
        default_factory=lambda: ["average", "decay", "peek_end"]
    )
    memory_regimes: list[str] = Field(
        default_factory=lambda: ["short_term", "long_term"]
    )
    activations: dict[str, int] = Field(
        default_factory=lambda: {"heterogeneous": 3, "homogeneous": 1}
    )
    top_average_agents: int = 10
    max_workers: int | None = 4

    @model_validator(mode="after")
    def validate_simulation(self) -> SimulationConfig:
        if self.generations < 1:
            raise ValueError(f"generations must be at least 1, got {self.generations}")
        if self.num_runs < 1:
            raise ValueError(f"num_runs must be at least 1, got {self.num_runs}")
        if not self.results_dir or not self.results_dir.strip():
            raise ValueError("results_dir cannot be empty")
        if self.top_average_agents < 1:
            raise ValueError(
                f"top_average_agents must be at least 1, got {self.top_average_agents}"
            )
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {self.max_workers}")
        valid_buffers = {"average", "decay", "peek_end"}
        for bt in self.buffer_types:
            if bt not in valid_buffers:
                raise ValueError(
                    f"Unknown buffer type '{bt}', expected one of {valid_buffers}"
                )
        if not self.activations:
            raise ValueError("activations cannot be empty")
        for mode, act in self.activations.items():
            if act < 1:
                raise ValueError(
                    f"Activation count for mode '{mode}' must be at least 1, got {act}"
                )
        return self


class AppConfig(BaseSettings):
    """Root application configuration."""

    agent: AgentConfig = Field(default_factory=AgentConfig)
    architecture: CognitiveArchitectureConfig = Field(
        default_factory=CognitiveArchitectureConfig,
        alias="cognitive_architecture",
    )
    p_dims: PerceptualDims = Field(default_factory=PerceptualDims)
    m_dims: MotorDims = Field(default_factory=MotorDims)
    world: WorldConfig = Field(default_factory=WorldConfig)
    evolution: EvolutionConfig = Field(
        default_factory=EvolutionConfig,
        alias="evolutionparameters",
    )
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)

    model_config = SettingsConfigDict(
        env_prefix="UTH_",
        env_nested_delimiter="__",
        extra="ignore",
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def validate_cross_layer_dimensions(self) -> AppConfig:
        # Check sensory dimensions matching world vision radius
        # 5x5 window minus center = 24 cells * 3 features + 2 scalars = 74
        vr = self.world.vision_radius
        observable_cells = (2 * vr + 1) ** 2 - 1
        expected_sensory_dim = observable_cells * 3 + 2
        p1_bottom = self.p_dims.dims[0][2]
        if p1_bottom != expected_sensory_dim:
            raise ValueError(
                f"P1 bottom input dimension ({p1_bottom}) does not match expected "
                f"sensory vector dimension ({expected_sensory_dim}) "
                f"for vision_radius={vr}"
            )

        # Perceptual feedforward consistency:
        # P1 output -> P2 bottom, P2 output -> P3 bottom
        p1_out = self.p_dims.dims[0][1]
        p2_bottom = self.p_dims.dims[1][2]
        if p1_out != p2_bottom:
            raise ValueError(
                f"P1 output ({p1_out}) does not match P2 bottom input ({p2_bottom})"
            )

        p2_out = self.p_dims.dims[1][1]
        p3_bottom = self.p_dims.dims[2][2]
        if p2_out != p3_bottom:
            raise ValueError(
                f"P2 output ({p2_out}) does not match P3 bottom input ({p3_bottom})"
            )

        # Motor action output consistency: M1 output represents actions (4)
        m1_out = self.m_dims.dims[0][1]
        if m1_out != 4:
            raise ValueError(
                f"M1 output must be 4 (representing 4 actions), got {m1_out}"
            )

        return self

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> AppConfig:
        """Loads configuration from a YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        return cls(**data)
