from random import randint

import hydra
import torch
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from lightning import Trainer

from ia176.data import DataModule
from ia176.concept_model import ConceptModel


OmegaConf.register_new_resolver(
    "random_seed", lambda: randint(0, 2**31), use_cache=True
)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(config: DictConfig) -> None:
    seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    logger = instantiate(config.logger) if "logger" in config else None
    data = instantiate(config.data, _recursive_=False, _target_=DataModule)
    model = instantiate(config.model, _target_=ConceptModel)
    trainer = instantiate(config.trainer, _target_=Trainer, logger=logger)
    getattr(trainer, config.mode)(model, datamodule=data, ckpt_path=config.checkpoint)


if __name__ == "__main__":
    main()
