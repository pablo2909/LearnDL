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

    @staticmethod
    def _flatten(image: torch.Tensor, close_channel: bool = False) -> torch.Tensor:
        """This function flatten an input tensor.

        Takes as input a tensor of shape (B, C1, C2, C3) and returns a tensor of
        shape (B, C1 * C2 * C3). Typically C1 is the number of channels
        of the image.
        You can flatten the image in two ways.


        Args:
            image: a torch.tensor of shape (B, C1, C2, C3)
        Returns:
            tensor of shape (B, C1 * C2 * C3)
        """
        B, C1 = image.shape[0], image.shape[1]
        if close_channel:
            reshaped_image = image.reshape(B, C1, -1).permute(0, 2, 1).reshape(B, -1)
        else:
            reshaped_image = image.reshape(B, -1)
        return reshaped_image


class MostSimpleNetwork(VisionNetwork):
    """First simple module for classification.
    """

    def __init__(self) -> None:
        super(MostSimpleNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 64 * 64),
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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method of the network.

        Args:
            x: Input tensor of shape (B, height, width)

        Returns:
            A tensor of shape (B, 10).
        """
        x = self._flatten(x)
        x = self.net(x)
        return x
