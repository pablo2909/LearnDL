from src.model import MostSimpleNetwork

from _trainer import Trainer, TrainerArguments

train_args = TrainerArguments()
model = MostSimpleNetwork()
T = Trainer(train_args)
T(model)
