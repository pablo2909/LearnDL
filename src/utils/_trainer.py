from abc import ABC, abstractmethod
from typing import NoReturn, Union

import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class BaseTrainer(ABC):
    def __init__(self, hp):
        self.hp = hp

    @property
    @abstractmethod
    def train_dl(self) -> DataLoader:
        pass

    @property
    def valid_dl(self) -> Union[NoReturn, DataLoader]:
        pass

    @property
    def test_dl(self) -> Union[NoReturn, DataLoader]:
        pass

    @property
    @abstractmethod
    def opt(self) -> Optimizer:
        pass

    @property
    @abstractmethod
    def loss_func(self) -> nn.Module:
        pass

    @abstractmethod
    def __call__(self):
        pass
