import torch
import torch.nn as nn


class VisionNetwork(nn.Module):
    """Base class for the computer vision network we will use.

    This class inherits from nn.Module. Its purpose is to implement the common functions
    used in all the computer vision network we will use.
    """

    def __init__(self) -> None:
        super(VisionNetwork, self).__init__()
        pass

    def _flatten(self, image: torch.Tensor) -> torch.Tensor:
        """This function flatten an input tensor.

        Takes as input a tensor of shape (B, C1,...,Cn) and returns a tensor of
        shape (B, C1 * ... * Cn).

        Args:
            image: a torch.tensor of shape (B, C1,...,Cn)
        Returns:
            tensor of shape (B, C1 * ... * Cn)
        """

        pass


class MostSimpleNetwork(VisionNetwork):
    """First simple module for classification.
    """

    def __init__(self) -> None:
        super(MostSimpleNetwork, self).__init__()
        self.net = nn.ModuleList(
            [
                nn.Linear(32 * 32, 64 * 64),
                nn.ReLU(),
                nn.Linear(64 * 64, 64 * 64),
                nn.ReLU(),
                nn.Linear(64 * 64, 64 * 64),
                nn.ReLU(),
                nn.Linear(64 * 64, 32 * 32),
                nn.ReLU(),
                nn.Linear(32 * 32, 16 * 16),
                nn.ReLU(),
                nn.Linear(16 * 16, 8 * 8),
                nn.ReLU(),
                nn.Linear(8 * 8, 10),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method of the network.

        Args:
            x: Input tensor of shape (batch_size, height, width)

        Returns:
            A tensor of shape (batch_size, 10).
        """
        x = self._flatten(x)
