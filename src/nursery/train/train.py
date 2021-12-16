from src.model import MostSimpleNetwork

from _callbacks import Novalidation, PrintInfo, TrainEvalCallback, TrainModel, WandB
from _trainer import Trainer, TrainerArguments


def main():

    train_args = TrainerArguments(batch_size=32)
    model = MostSimpleNetwork()
    T = Trainer(train_args)
    callbacks = [
        TrainEvalCallback(),
        TrainModel(),
        PrintInfo(),
        Novalidation(),
        WandB(train_args),
    ]
    T(model, callbacks)


if __name__ == "__main__":
    main()
