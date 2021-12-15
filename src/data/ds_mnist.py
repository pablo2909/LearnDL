from pathlib import Path
from typing import Callable, Optional

from torchvision import datasets, transforms

from .helpers import create_dir


def make_mnist(train: bool = True, transform: Optional[Callable] = None) -> datasets:
    BASE_DIR = Path(__file__).parents[2]
    root_dir = create_dir(BASE_DIR / "data")
    print(root_dir)
    print(train)
    ds = datasets.MNIST(
        root=root_dir, train=train, download=True, transform=transforms.ToTensor()
    )
    return ds

