from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

import numpy as np


@dataclass
class BufferOutput:
    fired: bool
    data: np.ndarray | None


class Buffer(ABC):
    def __init__(
        self,
        n_input_size: int,
        n_activations: int,
        short_term_memory: bool,
    ) -> None:
        """
        Docstring here
        """
        self.n_input_size = n_input_size
        self.n_activations = n_activations
        self.short_term_memory = short_term_memory

        self.reset()

    @abstractmethod
    def _update_state(
        self,
        input_: np.ndarray,
    ) -> np.ndarray:
        pass

    def _reset_state(self) -> None:
        self.v: np.ndarray = np.zeros(shape=(self.n_input_size,))

    def reset(self) -> None:
        self.n_count = 0
        self._reset_state()

    def step(self, input_: np.ndarray) -> BufferOutput:
        self.n_count += 1
        self.v = self._update_state(input_)

        if self.n_count < self.n_activations:
            # not fired
            return BufferOutput(False, None)

        # got fired
        self.n_count = 0
        temp_v = self.v.copy()

        if self.short_term_memory:
            self._reset_state()

        return BufferOutput(True, temp_v)


class AverageBuffer(Buffer):
    def __init__(
        self,
        n_input_size: int,
        n_activations: int,
        short_term_memory: bool,
    ) -> None:
        super().__init__(n_input_size, n_activations, short_term_memory)
        self.n_history = 0

    def _update_state(
        self,
        input_: np.ndarray,
    ) -> np.ndarray:
        self.n_history += 1
        return cast(
            np.ndarray,
            ((self.n_history - 1) * self.v + input_) / self.n_history,
        )

    def _reset_state(self) -> None:
        super()._reset_state()
        self.n_history = 0


class DecayBuffer(Buffer):
    def _update_state(
        self,
        input_: np.ndarray,
    ) -> np.ndarray:
        return cast(np.ndarray, (self.v + input_) / 2.0)


class PeekEndBuffer(Buffer):
    def _update_state(
        self,
        input_: np.ndarray,
    ) -> np.ndarray:
        return cast(np.ndarray, (self.v + np.maximum(self.v, input_)) / 2.0)
