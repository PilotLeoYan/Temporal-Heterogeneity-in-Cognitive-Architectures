from typing import cast

import numpy as np


class NN:
    def __init__(
        self,
        total_input: int,
        total_output: int,
        weight_init: float,
    ) -> None:
        """
        Docstring here
        """
        self.size = (total_output, total_input)
        self.w: np.ndarray = np.random.uniform(
            low=-weight_init, high=weight_init, size=self.size
        )

    def get_weights(self) -> np.ndarray:
        return self.w.copy()

    def set_weights(self, w: np.ndarray) -> None:
        self.w = np.clip(w.copy(), -1.0, 1.0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        raw_y = self.w @ x
        return cast(np.ndarray, np.clip(raw_y, -1.0, 1.0))

    def update(self, x: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
        """
        ΔW[i, j] = η * y[i] * ( x[j] - Σ_{k=1}^i y[k] * W[k, j] )
        W = W + ΔW
        """
        x_clean = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        y_clean = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        proyection = np.tril(np.outer(y_clean, y_clean)) @ self.w
        dw = learning_rate * (np.outer(y_clean, x_clean) - proyection)

        # numeric stability (in-place)
        np.nan_to_num(dw, nan=0.0, posinf=0.01, neginf=-0.01, copy=False)
        np.clip(dw, -0.01, 0.01, out=dw)

        self.w += dw
        np.clip(self.w, -1.0, 1.0, out=self.w)


class ConcatLayer:
    def __init__(self, total_input: int) -> None:
        self.total_input = total_input

    def forward(
        self,
        bottom: np.ndarray | None,
        lateral: np.ndarray | None,
        top: np.ndarray | None,
    ) -> np.ndarray:
        inputs_ = [inp for inp in (bottom, lateral, top) if inp is not None]
        if len(inputs_) == 1:
            return inputs_[0]
        return np.concatenate(inputs_)


class CognitiveLayer:
    def __init__(
        self,
        bottom_size: int,
        lateral_size: int,
        top_size: int,
        concat_layer: ConcatLayer,
        nn: NN,
    ) -> None:
        self.bottom_size = bottom_size
        self.lateral_size = lateral_size
        self.top_size = top_size
        self.concat_layer = concat_layer
        self.nn = nn

        self.reset()

    def reset(self) -> None:
        self.is_active = False
        self.in_bottom = np.zeros(shape=(self.bottom_size,))
        self.in_lateral = np.zeros(shape=(self.lateral_size,))
        self.in_top = np.zeros(shape=(self.top_size,))

    def receive_bottom(self, data: np.ndarray) -> None:
        self.in_bottom = data

    def receive_lateral(self, data: np.ndarray) -> None:
        self.in_lateral = data

    def receive_top(self, data: np.ndarray) -> None:
        self.in_top = data

    def step(self, learning_rate: float) -> np.ndarray | None:
        if not self.is_active:
            return None

        x = self.concat_layer.forward(
            self.in_bottom if self.bottom_size != 0 else None,
            self.in_lateral if self.lateral_size != 0 else None,
            self.in_top if self.top_size != 0 else None,
        )

        y = self.nn.forward(x)
        self.nn.update(x, y, learning_rate)

        self.is_active = False

        return y
