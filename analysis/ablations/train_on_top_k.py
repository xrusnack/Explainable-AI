"""Script for training a decision tree on the top k concept bottleneck outputs.

Usage: From the project root folder run: `uv run python -m analysis.ablations.train_on_top_k`.
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


def train_tree_on_top_k(
    concepts_path: Path, 
    loader: DataLoader[tuple[Tensor, Tensor]], 
    tree_cfg: DictConfig, 
    save_path: Path,
    mask_indices: Tensor
    ) -> None:
    X_train = torch.load(concepts_path)  # (n, 39)
    print("Loaded concept predictions with shape:", X_train.shape)
    X_train_masked = X_train[:, mask_indices]
    print("Filtered to top-k concepts with shape:", X_train_masked.shape)

    y_list = [targets for _, targets in loader]
    y_train = torch.cat(y_list, dim=0).cpu().numpy()
    print("Loaded labels with shape:", y_train.shape)

    tree = instantiate(tree_cfg)
    tree.fit(X_train_masked.cpu().numpy(), y_train)
    print("Decision Tree trained. Depth:", tree.get_depth(), "Number of leaves:", tree.get_n_leaves())

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(tree, f)
    print(f"Trained Decision Tree saved at: {save_path}")


@hydra.main(config_path="../../configs", config_name="experiment/train_tree_on_top_k", version_base=None)
def main(config: DictConfig) -> None:
    data_module = instantiate(config.data, _recursive_=False, _target_=DataModule)
    data_module.setup(stage="fit")
    loader = data_module.train_dataloader()
    concepts_path = Path(config.concepts_path)
    if not concepts_path.exists():
        raise FileNotFoundError(f"Concept predictions not found: {concepts_path}")
    mask_indices = torch.zeros(config.num_concepts, dtype=torch.bool)
    mask_indices[list(config.top_k_indices)] = True

    train_tree_on_top_k(
        concepts_path=concepts_path,
        loader=loader,
        tree_cfg=config.decision_tree,
        save_path=config.save_path,
        mask_indices=mask_indices
    )


if __name__ == "__main__":
    main()
