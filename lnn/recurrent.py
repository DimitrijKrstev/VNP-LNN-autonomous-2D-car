# -*- coding: utf-8 -*-
from math import sqrt
from statistics import mean
from typing import List

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

from .cell import LiquidCell
from .timenorm import TimeNorm
from .conv2d import Conv2d


class LiquidRecurrent(nn.Module):
    def __init__(
            self,
            neuron_number: int,
            input_size: int,
            unfolding_steps: int,
            output_size: int,
    ) -> None:
        super().__init__()

        self.__cell = LiquidCell(neuron_number, input_size, unfolding_steps)
        self.__neuron_number = neuron_number
        self.__to_output = nn.Linear(neuron_number, output_size)

    def __get_first_x(self, batch_size: int) -> th.Tensor:
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"
        return F.mish(
            th.randn(batch_size, self.__neuron_number, device=device)
        )

    def _process_input(self, i: th.Tensor) -> th.Tensor:
        return i

    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        out = self.__to_output(out)
        return out

    def _sequence_processing(self, outputs: List[th.Tensor]) -> th.Tensor:
        return th.stack(outputs, -1)

    def forward(self, i: th.Tensor, delta_t: th.Tensor) -> th.Tensor:
        b, _, _ = i.size()

        x_t = self.__get_first_x(b)
        i = self._process_input(i)

        results = []

        for t in range(i.size(2)):
            x_t = self.__cell(x_t, i[:, :, t], delta_t[:, t])
            results.append(self._output_processing(x_t))

        return self._sequence_processing(results)

    def count_parameters(self) -> int:
        return sum(
            int(np.prod(p.size()))
            for p in self.parameters()
            if p.requires_grad
        )

    def grad_norm(self) -> float:
        return mean(
            float(p.grad.norm().item())
            for p in self.parameters()
            if p.grad is not None
        )


class LiquidRecurrentReg(LiquidRecurrent):
    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        return th.sigmoid(super()._output_processing(out))


class LiquidRecurrentDQN(LiquidRecurrent):
    def __init__(
        self,
        neuron_number: int,       # Number of neurons in the recurrent liquid layer
        input_size: int,          # Input size for the Conv2D layer (e.g., 1 for grayscale)
        unfolding_steps: int,     # Number of time steps to unfold the liquid dynamics
        output_size: int,         # Number of discrete actions (e.g., 5 for CarRacing-v2)
        hidden_size: int = 256    # Hidden size for LSTM/Liquid Layer
    ) -> None:
        nb_layer = 6              # Number of layers in the convolutional encoder
        factor = sqrt(2)          # Scaling factor for the channel dimensions
        encoder_dim = 16          # Initial number of channels

        # Define the channels for each convolutional layer
        channels = [
            (
                int(encoder_dim * factor**i),
                int(encoder_dim * factor ** (i + 1)),
            )
            for i in range(nb_layer)
        ]

        # Initialize the parent LiquidRecurrent class with the final channel size as input
        super().__init__(neuron_number, channels[-1][1], unfolding_steps, output_size)

        # Define a convolutional encoder with multiple layers, TimeNorm, and Conv2D
        self.__conv = nn.Sequential(
            Conv2d(input_size, channels[0][0], kernel_size=3, stride=2),  # First Conv2D layer
            nn.Mish(),
            TimeNorm(channels[0][0]),                                     # Apply TimeNorm
            *[
                nn.Sequential(
                    nn.AvgPool2d(2, 2),                                   # Pooling layer
                    nn.Conv2d(c_i, c_o, kernel_size=3, stride=2),         # Conv2D for spatial dependencies
                    nn.Mish(),                                            # Non-linearity
                    TimeNorm(c_o),                                        # Apply TimeNorm
                )
                for c_i, c_o in channels                                  # Apply for all convolutional layers
            ]
        )

    # Process the input image using the convolutional layers
    def _process_input(self, i: th.Tensor) -> th.Tensor:
        out: th.Tensor = self.__conv(i)
        return out

    # Output processing for Q-values (for each discrete action)
    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        return super()._output_processing(out)

    # Sequence processing to return the final result (use the last step)
    def _sequence_processing(self, outputs: List[th.Tensor]) -> th.Tensor:
        return outputs[-1]  # Return the last output (final Q-values for DQN)


class LiquidRecurrentBrainActivity(LiquidRecurrent):
    def __init__(
            self,
            neuron_number: int,
            input_size: int,
            unfolding_steps: int,
            output_size: int,
    ) -> None:
        nb_layer = 6
        factor = sqrt(2)
        encoder_dim = 16

        channels = [
            (
                int(encoder_dim * factor ** i),
                int(encoder_dim * factor ** (i + 1)),
            )
            for i in range(nb_layer)
        ]

        super().__init__(
            neuron_number,
            channels[-1][1],
            unfolding_steps,
            output_size,
        )

        self.__conv = nn.Sequential(
            Conv2d(input_size, channels[0][0], dilation=1),
            nn.Mish(),
            TimeNorm(channels[0][0]),
            *[
                nn.Sequential(
                    nn.AvgPool1d(2, 2),
                    CausalConv1d(c_i, c_o, dilation=1),
                    nn.Mish(),
                    TimeNorm(c_o),
                )
                for c_i, c_o in channels
            ]
        )

    def _process_input(self, i: th.Tensor) -> th.Tensor:
        out: th.Tensor = self.__conv(i)
        return out

    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        return F.softmax(super()._output_processing(out), dim=-1)

    def _sequence_processing(self, outputs: List[th.Tensor]) -> th.Tensor:
        return outputs[-1]


class LiquidRecurrentLast(LiquidRecurrent):
    def _sequence_processing(self, outputs: List[th.Tensor]) -> th.Tensor:
        return outputs[-1]
