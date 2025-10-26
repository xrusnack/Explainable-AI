"""Compute saliency maps for selected concepts on selected test images.

Loads a trained ConceptModel from checkpoint, fetches two samples from the test set,
computes and saves saliency maps for two concept indices. 

Usage: From the project root folder run: `uv run python -m analysis.saliency_maps`.
"""

from pathlib import Path
from typing import Sequence

import hydra
import torch
import torchvision
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from omegaconf import DictConfig

from ia176.concept_model import ConceptModel
from ia176.data.data_module import DataModule


def normalize_to_uint8(x: torch.Tensor) -> torch.Tensor:
    x = x - x.min()
    denom = x.max().clamp(min=1e-8)
    x = x / denom
    return (x * 255.0).to(torch.uint8)


def compute_saliency(model: ConceptModel, image: torch.Tensor, concept_idx: int) -> torch.Tensor:
    model.eval()
    image = image.unsqueeze(0)
    image.requires_grad_(True)
    logits = model(image)
    logit = logits[0, concept_idx]

    model.zero_grad(set_to_none=True)
    if image.grad is not None:
        image.grad.zero_()
    logit.backward()

    assert image.grad is not None
    grad = image.grad.detach().abs().squeeze(0)
    sal = grad.max(dim=0).values
    return sal


def _to_display(orig: torch.Tensor) -> torch.Tensor:
    """Return an image tensor in [0,1] for visualization.

    If input is already in [0,1], return as-is; otherwise apply per-image min-max scaling.
    """
    img = orig.detach().cpu()
    if img.min() < 0.0 or img.max() > 1.0:
        mn, mx = img.min(), img.max()
        img = (img - mn) / (mx - mn + 1e-8)
    return img


def save_overlay(orig: torch.Tensor, sal: torch.Tensor, out_path: Path) -> None:
    sal_uint8 = normalize_to_uint8(sal)
    sal_img = torchvision.transforms.ToPILImage()(sal_uint8)

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        disp = _to_display(orig).permute(1, 2, 0)
        
        ax[0].imshow(disp)
        ax[0].axis("off")
        ax[0].set_title("Image")
        ax[1].imshow(disp)
        ax[1].imshow(sal.detach().cpu(), cmap="jet", alpha=0.5)
        ax[1].axis("off")
        ax[1].set_title("Saliency")
        fig.tight_layout()
        fig.savefig(out_path.as_posix(), dpi=150)
        plt.close(fig)
    except Exception:
        sal_img.save(out_path)


@hydra.main(config_path="../configs", config_name="experiment/saliency", version_base=None)
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_module: DataModule = instantiate(cfg.data, _recursive_=False, _target_=DataModule)
    data_module.setup(stage="test")

    test_ds = data_module.test

    model: ConceptModel = ConceptModel.load_from_checkpoint(cfg.checkpoint)
    model.to(device)
    model.eval()

    sample_indices: Sequence[int] = cfg.samples
    concept_indices: Sequence[int] = cfg.concepts

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_idx in sample_indices:
        img, _ = test_ds[img_idx]
        img = img.to(device)
        for c in concept_indices:
            sal = compute_saliency(model, img, concept_idx=int(c))
            save_overlay(img.detach().cpu(), 
                         sal.detach().cpu(), 
                         out_dir / f"saliency_img{img_idx}_concept{c}.png"
            )
    print(f"Saved saliency maps to {out_dir}")


if __name__ == "__main__":
    main()
