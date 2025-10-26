"""Script to compute dataset normalization statistics (mean and standard deviation) from train data.

Usage: From the project root folder run: `uv run python -m preprocessing.normalization_stats`
"""

import hydra
import torch
from tqdm import tqdm

from torch import Tensor
from omegaconf import DictConfig
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from ia176.data.data_module import DataModule


def compute_dataset_stats(loader: DataLoader[tuple[Tensor, Tensor]]) -> None:
    """Compute mean and std for the training dataset."""
    n_pixels = 0
    s1 = torch.zeros(3)
    s2 = torch.zeros(3)

    for images, _ in tqdm(loader, desc="Computing dataset stats"):
        # images shape: [B, C, H, W]
        batch_pixels = images.size(0) * images.size(2) * images.size(3)
        s1 += images.sum(dim=[0, 2, 3])
        s2 += (images ** 2).sum(dim=[0, 2, 3])
        n_pixels += batch_pixels

    mean = s1 / n_pixels
    std = ((s2 / n_pixels) - mean ** 2).sqrt()

    print(f"Dataset stats:\n  mean: {mean}\n  std:  {std}")
    

@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
def main(config: DictConfig) -> None:
    data_module = instantiate(config.data, _recursive_=False, _target_=DataModule)
    data_module.setup(stage="fit")
    loader = data_module.train_dataloader()
    compute_dataset_stats(loader)


if __name__ == "__main__":
    main()
