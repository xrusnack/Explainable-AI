"""PyTorch Dataset for CelebA images and concept/attribute labels."""

import torch
import pandas as pd
import numpy as np
from torchvision.transforms import ToTensor
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from ia176.ia176_typing import Transforms


class CelebADataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self, 
        image_ids: list[str], 
        images_dir: Path, 
        metadata_path: Path, 
        transforms: Transforms | None = None,
        concept_mode: bool = True
    ) -> None:
        """Initialize the dataset.

        Args:
            image_ids: List of image IDs to include in the dataset.
            images_dir: Path to the directory containing the images.
            metadata_path: Path to the CSV file containing image metadata (image_id and attributes).
            transforms: Optional (list of) transforms to apply to the data (e.g. normalization, ...).
            concept_mode: Whether to use all attributes or a single target attribute (e.g. "Attractive").
        """
        self.image_ids = image_ids
        self.images_dir = images_dir
        self.images_paths = self._load_paths(metadata_path)
        self.concept_mode = concept_mode
        self.labels = self._load_labels(metadata_path)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_ids)
    
    def _load_paths(self, metadata_path: Path) -> dict[str, Path]:
        """Create a mapping between image IDs and their corresponding image paths."""
        df = pd.read_csv(metadata_path)
        image_ids_set = set(self.image_ids)
        images_paths: dict[str, Path] = {}

        for _, row in df.iterrows():
            image_id = Path(row["image_id"]).stem
            if image_id not in image_ids_set:
                continue
            image_path = self.images_dir / Path(row["image_id"])
            images_paths[image_id] = image_path
            
        return images_paths

    def _load_labels(self, metadata_path: Path) -> dict[str, Tensor]:
        """Create a mapping between image IDs and their corresponding attribute labels."""
        df = pd.read_csv(metadata_path)
        image_ids_set = set(self.image_ids)
        labels: dict[str, Tensor] = {}

        for _, row in df.iterrows():
            image_id = Path(row["image_id"]).stem
            if image_id not in image_ids_set:
                continue
            if self.concept_mode:
                attributes = row.drop(["image_id", "Attractive"]).values.astype(np.float32)
            else:
                attributes = np.array([row["Attractive"]], dtype=np.float32)
            binary_attributes = (attributes == 1).astype(np.float32)  # convert {-1, 1} → {0, 1}
            label_tensor = torch.tensor(binary_attributes, dtype=torch.float32)
            labels[image_id] = label_tensor
        return labels

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        image_id = self.image_ids[idx]
        image = Image.open(self.images_paths[image_id]).convert("RGB")
        labels = self.labels[image_id]

        if self.transforms:
            image = self.transforms(image)  # type: ignore

        if not isinstance(image, Tensor):
            image = ToTensor()(image)

        return image, labels
