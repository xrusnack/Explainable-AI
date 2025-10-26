"""Script for training a decision tree on concept bottleneck outputs.

Usage: From the project root folder run: `uv run python -m ia176.decision_tree`.
"""

import hydra
import torch
import pickle
from pathlib import Path

from torch import Tensor
from hydra.utils import instantiate
from omegaconf import DictConfig
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from ia176.data.data_module import DataModule


def train_tree(
    concepts_path: Path, 
    loader: DataLoader[tuple[Tensor, Tensor]], 
    tree_cfg: DictConfig, 
    save_path: Path
    ) -> None:
    X_train = torch.load(concepts_path)  # (n, 39)
    print("Loaded concept predictions with shape:", X_train.shape)

    y_list = [targets for _, targets in loader]
    y_train = torch.cat(y_list, dim=0).cpu().numpy()
    print("Loaded labels with shape:", y_train.shape)

    tree = instantiate(tree_cfg)
    tree.fit(X_train.cpu().numpy(), y_train)
    print("Decision Tree trained. Depth:", tree.get_depth(), "Number of leaves:", tree.get_n_leaves())

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(tree, f)
    print(f"Trained Decision Tree saved at: {save_path}")


@hydra.main(config_path="../configs", config_name="experiment/train_tree", version_base=None)
def main(config: DictConfig) -> None:
    data_module = instantiate(config.data, _recursive_=False, _target_=DataModule)
    data_module.setup(stage="fit")
    loader = data_module.train_dataloader()
    concepts_path = Path(config.concepts_path)
    if not concepts_path.exists():
        raise FileNotFoundError(f"Concept predictions not found: {concepts_path}")

    train_tree(
        concepts_path=concepts_path,
        loader=loader,
        tree_cfg=config.decision_tree,
        save_path=config.save_path
    )


if __name__ == "__main__":
    main()
