"""Lightning callback to save model predictions."""

import mlflow
import torch
from typing import cast
from pathlib import Path

from torch import Tensor
from lightning import Callback, LightningModule, Trainer
from torchmetrics import MetricCollection


class PredictionCallback(Callback):
    def __init__(self, save_path: str):
        super().__init__()
        self.save_path = save_path
        self.all_predictions: list[Tensor] = []
    
    def on_predict_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        metrics_module = cast("MetricCollection", pl_module.predict_metrics)
        metrics = metrics_module.compute()
        for key, value in metrics.items():
            metric_name = key.split("/")[-1]
            mlflow.log_metric(f"prediction/{metric_name}", float(value))
        metrics_module.reset()

        all_preds_tensor = torch.cat(self.all_predictions, dim=0)
        save_dir = Path(self.save_path)
        save_dir.mkdir(exist_ok=True, parents=True)
        save_file = save_dir / Path("predictions.pt")
        torch.save(all_preds_tensor, save_file)
        print(f"Saved concept predictions to {save_file}")

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor,
        batch: Tensor,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, targets = batch
        probs = torch.sigmoid(outputs)
        self.all_predictions.append(probs)
        metrics_module = cast("MetricCollection", pl_module.predict_metrics)
        metrics_module.update(probs, targets)

