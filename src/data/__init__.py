from .ds_mnist import make_mnist

__all__ = ["make_mnist"]

for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)
