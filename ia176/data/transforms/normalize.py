from torch import Tensor
import torch
from torch_geometric.transforms import BaseTransform
from torchvision.transforms import functional as F


class Normalize(BaseTransform):  # type: ignore[misc]
    """Z-Score normalization for images using precomputed channel-wise mean and std."""

    def __init__(self, mean: list[float], std: list[float]) -> None:
        """Initialize the normalization transform.

        Args:
            mean (list[float]): Channel-wise mean [R, G, B]
            std (list[float]): Channel-wise std [R, G, B]
        """
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)


    def __call__(self, img: Tensor) -> Tensor:
        return (img - self.mean[:, None, None]) / self.std[:, None, None]
    
    forward = __call__

