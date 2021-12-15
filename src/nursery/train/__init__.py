from ._trainer import Trainer, TrainerArguments

__all__ = ["Trainer", "TrainerArguments"]

for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)
