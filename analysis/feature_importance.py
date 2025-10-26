"""Loads a pickled decision tree and prints feature importances for CelebA concepts.

Assumes the tree was trained on the same 39 concepts, in the same order 
(specified in `configs/experiment/feature_importance.yaml`).

Usage: From the project root folder run: `uv run python -m analysis.feature_importance`.
"""

import pickle
import hydra
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import export_text

from omegaconf import DictConfig


def feature_importance(tree_path: Path, output_plot_path: Path, concepts: list[str]) -> None:
    with open(tree_path, "rb") as f:
        tree = pickle.load(f)

    importance = tree.feature_importances_
    print(export_text(tree, feature_names=concepts))
    
    plt.figure(figsize=(40, 12))
    output_plot_path.mkdir(parents=True, exist_ok=True)
    plot_tree(
        tree,
        feature_names=concepts,
        class_names=["Not Attractive", "Attractive"],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.savefig(output_plot_path / Path("decision_tree.png"), dpi=300, bbox_inches="tight")
    plt.close() 

    for name, imp in sorted(zip(concepts, importance), key=lambda x: x[1], reverse=True):
        print(f"{name}: {imp:.3f}")


@hydra.main(config_path="../configs", config_name="experiment/feature_importance", version_base=None)
def main(cfg: DictConfig) -> None:
    decision_tree_path = Path(cfg.tree_path)
    output_plot_path = Path(cfg.output_plot_path)
    concepts = list(cfg.concepts)
    feature_importance(decision_tree_path, output_plot_path, concepts=concepts)


if __name__ == "__main__":
    main()