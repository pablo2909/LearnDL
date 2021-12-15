from typing import Dict

import torch
import torch.nn as nn
import torchvision.transforms as tf
from attr import attrib, attrs
from attr.setters import validate
from attr.validators import instance_of, optional
from src.data import make_mnist
from src.utils import BaseTrainer, DataBunch, Learner, Runner
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.transforms import transforms

from _callbacks import Novalidation, PrintInfo, TrainEvalCallback, TrainModel


@attrs(kw_only=True, on_setattr=validate)
class TrainerArguments:
    epochs = attrib(default=10, validator=instance_of(int))
    which_ds = attrib(default="mnist", validator=instance_of(str))
    train_valid_split = attrib(default=0.8, validator=instance_of(float))
    batch_size = attrib(default=8, validator=instance_of(int))
    opt = attrib(default="Adam", validator=instance_of(str))
    lr = attrib(default=0.0001, validator=instance_of(float))


class Trainer(BaseTrainer):
    def __init__(self, hp: TrainerArguments) -> None:
        super().__init__(hp)
        self._create_train_dataset()

    def _create_transformation(self) -> tf.transforms.Compose:
        transformation = tf.Compose([tf.ToTensor()])
        return transformation

    def _create_train_dataset(self):
        if self.hp.which_ds == "mnist":
            ds = make_mnist(train=True, transform=self._create_transformation())
            train_ds_length = int(len(ds) * self.hp.train_valid_split)
            ds = Subset(ds, torch.arange(1000))
        self.train_ds, self.valid_ds = random_split(
            ds, [train_ds_length, len(ds) - train_ds_length,],
        )

    @property
    def train_dl(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.hp.batch_size)

    @property
    def valid_dl(self) -> DataLoader:
        return DataLoader(self.valid_ds, batch_size=self.hp.batch_size)

    # @property
    def opt(self, params) -> Optimizer:
        if self.hp.opt == "Adam":
            return Adam(params=params, lr=self.hp.lr)

    @property
    def loss_func(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def __call__(self, model: nn.Module):
        learner = Learner(
            model=model,
            opt=self.opt(model.parameters()),
            data=DataBunch(self.train_dl, None),
            loss_func=self.loss_func,
        )
        callbacks = [TrainEvalCallback(), TrainModel(), PrintInfo(), Novalidation()]
        run = Runner(callbacks)
        run.fit(self.hp.epochs, learner)
