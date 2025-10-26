import pandas as pd
from pathlib import Path
from copy import deepcopy
from collections.abc import Iterable

from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader

from ia176.data.datasets.concept_dataset import CelebADataset


class DataModule(LightningDataModule):
    def __init__(
        self, batch_size: int, num_workers: int = 0, predict_on_train: bool = False, **datasets: DictConfig
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.predict_on_train = predict_on_train
        self.datasets = datasets

    def setup(self, stage: str) -> None:
        def prepare(conf: DictConfig, partition_value: int, transforms_cfg: DictConfig) -> CelebADataset:
            conf_copy = deepcopy(conf)
            df = pd.read_csv(conf_copy.partition_csv_path)
            subset = df[df["partition"] == partition_value]
            image_ids = [Path(img).stem for img in subset["image_id"]]
            conf_copy.image_ids = image_ids
            conf_copy.pop("partition_csv_path", None)
            dataset: CelebADataset = instantiate(conf_copy)
            dataset.transforms = (
                instantiate(transforms_cfg, _recursive_=True) if transforms_cfg is not None else None
            )
            return dataset

        if stage == "fit":
            self.train = prepare(self.datasets["dataset"], 0, self.datasets["train"]["transforms"])
            self.val = prepare(self.datasets["dataset"], 1, self.datasets["val"]["transforms"])
        elif stage == "validate":
            self.val = prepare(self.datasets["dataset"], 1, self.datasets["val"]["transforms"])
        elif stage == "test":
            self.test = prepare(self.datasets["dataset"], 2, self.datasets["test"]["transforms"])
        elif stage == "predict":
            if self.predict_on_train:
                self.predict = prepare(self.datasets["dataset"], 0, self.datasets["train"]["transforms"])
            else:
                self.predict = prepare(self.datasets["dataset"], 2, self.datasets["test"]["transforms"])

    def train_dataloader(self) -> Iterable[Tensor]:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[Tensor]:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> Iterable[Tensor]:
        return DataLoader(
            self.test, 
            batch_size=self.batch_size, 
            drop_last=False,
            num_workers=self.num_workers
        )
        
    def predict_dataloader(self) -> Iterable[Tensor]:
        return DataLoader(
            self.predict, batch_size=1, num_workers=0
        )
