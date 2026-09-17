import numpy as np

from src.components.buffer import AverageBuffer, Buffer, DecayBuffer, PeekEndBuffer
from src.components.layer import NN, CognitiveLayer, ConcatLayer
from src.config.schema import CognitiveArchitectureConfig, MotorDims, PerceptualDims


class CognitiveArchitecture:
    def __init__(
        self,
        p_dims: PerceptualDims,
        m_dims: MotorDims,
        buffer_type: str,
        config: CognitiveArchitectureConfig,
        short_term_memory: bool,
    ) -> None:
        buffer_cls: type[Buffer]
        match buffer_type:
            case "average":
                buffer_cls = AverageBuffer
            case "decay":
                buffer_cls = DecayBuffer
            case "peek_end":
                buffer_cls = PeekEndBuffer
            case _:
                raise ValueError

        self.c = config
        self.p_dims = p_dims
        self.p_layers = [
            CognitiveLayer(
                self.p_dims.dims[i][2],
                self.p_dims.dims[i][3],
                self.p_dims.dims[i][4],
                ConcatLayer(self.p_dims.dims[i][0]),
                NN(self.p_dims.dims[i][0], self.p_dims.dims[i][1], self.c.weight_init),
            )
            for i in range(len(self.p_dims.dims))
        ]
        self.p_buffers = [
            buffer_cls(self.p_dims.dims[i][1], self.c.n_activations, short_term_memory)
            for i in range(len(self.p_dims.dims) - 1)
        ]

        self.m_dims = m_dims
        self.m_layers = [
            CognitiveLayer(
                self.m_dims.dims[i][2],
                self.m_dims.dims[i][3],
                self.m_dims.dims[i][4],
                ConcatLayer(self.m_dims.dims[i][0]),
                NN(self.m_dims.dims[i][0], self.m_dims.dims[i][1], self.c.weight_init),
            )
            for i in range(len(self.m_dims.dims))
        ]
        self.m_buffers = [
            buffer_cls(self.m_dims.dims[i][1], self.c.n_activations, short_term_memory)
            for i in range(len(self.m_dims.dims) - 1)
        ]
        self.num_p_layers = len(self.p_layers)
        self.num_m_layers = len(self.m_layers)
        self.p_indices = list(range(self.num_p_layers))
        self.m_indices_rev = list(range(self.num_m_layers - 1, -1, -1))

    def reset(self) -> None:
        for p_layer, m_layer in zip(self.p_layers, self.m_layers):
            p_layer.reset()
            m_layer.reset()

        for p_buffer, m_buffer in zip(self.p_buffers, self.m_buffers):
            p_buffer.reset()
            m_buffer.reset()

    def get_weights(self) -> dict[str, np.ndarray]:
        p_weights = {
            f"p_w{i}": p_layer.nn.get_weights()
            for i, p_layer in enumerate(self.p_layers)
        }
        m_weights = {
            f"m_w{i}": m_layers.nn.get_weights()
            for i, m_layers in enumerate(self.m_layers)
        }
        return p_weights | m_weights

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        for k, v in weights.items():
            if k[:-1] == "p_w":
                self.p_layers[int(k[-1])].nn.set_weights(v)
            if k[:-1] == "m_w":
                self.m_layers[int(k[-1])].nn.set_weights(v)

    def step(self, sensory_vector: np.ndarray) -> tuple[int | None, int]:
        active_layers_count = 0

        self.p_layers[0].receive_bottom(sensory_vector)
        self.p_layers[0].is_active = True

        for i in self.p_indices:
            p_layer = self.p_layers[i]
            yp = p_layer.step(self.c.learning_rate)

            if yp is None:
                continue

            active_layers_count += 1
            self.m_layers[i].receive_lateral(yp)
            self.m_layers[i].is_active = True

            if i > 0:
                self.p_layers[i - 1].receive_top(yp)

            if i == self.num_p_layers - 1:
                continue

            y_buffer = self.p_buffers[i].step(yp)
            if y_buffer.fired and y_buffer.data is not None:
                self.p_layers[i + 1].receive_bottom(y_buffer.data)
                self.p_layers[i + 1].is_active = True

        action_vector = None
        for i in self.m_indices_rev:
            m_layer = self.m_layers[i]
            ym = m_layer.step(self.c.learning_rate)

            if ym is None:
                continue

            if i == 0:
                action_vector = ym

            active_layers_count += 1
            self.p_layers[i].receive_lateral(ym)

            if i > 0:
                self.m_layers[i - 1].receive_top(ym)
                self.m_layers[i - 1].is_active = True

            if i == self.num_m_layers - 1:
                continue

            y_buffer = self.m_buffers[i].step(ym)
            if y_buffer.fired and y_buffer.data is not None:
                self.m_layers[i + 1].receive_bottom(y_buffer.data)
                self.m_layers[i + 1].is_active = True

        if action_vector is not None:
            action = int(np.argmax(action_vector))
            return action, active_layers_count
        return None, active_layers_count
