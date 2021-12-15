from .ResNet._model import MostSimpleNetwork

__all__ = [
    "MostSimpleNetwork",
]


for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)
