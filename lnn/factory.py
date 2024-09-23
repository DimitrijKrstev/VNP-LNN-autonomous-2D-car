# -*- coding: utf-8 -*-
from typing import Callable, Dict, Final, Type

from functions import *
from recurrent import *

_MODEL_DICT: Final[Dict[str, Type[LiquidRecurrent]]] = {
    "regression": LiquidRecurrentReg,
    "classification": LiquidRecurrent,
    "last_classification": LiquidRecurrentLast,
    "brain_activity": LiquidRecurrentBrainActivity,
}

_LOSS_DICT: Final[
    Dict[str, Callable[[th.Tensor, th.Tensor, ReductionType], th.Tensor]]
] = {
    "regression": mse_loss,
    "classification": cross_entropy_time_series,
    "last_classification": cross_entropy,
    "brain_activity": kl_div,
}

TaskType = Literal[
    "regression", "classification", "last_classification", "brain_activity"
]


def get_model_constructor(task_type: TaskType) -> Type[LiquidRecurrent]:
    return _MODEL_DICT[task_type]


def get_loss_function(
    task_type: TaskType,
) -> Callable[[th.Tensor, th.Tensor, ReductionType], th.Tensor]:
    return _LOSS_DICT[task_type]