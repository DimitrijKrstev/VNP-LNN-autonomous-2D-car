# -*- coding: utf-8 -*-
import torch as th
from torch import nn


class Conv2d(nn.Conv2d):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
        )

        def __init_weights(module: nn.Module) -> None:
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=1e-3)
                if module.bias is not None:
                    nn.init.normal_(module.bias, std=1e-3)

        self.apply(__init_weights)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(x)
