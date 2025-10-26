from lightning import LightningModule
from torch import Tensor, nn
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics import MetricCollection, Metric
from torchmetrics.classification import (
    MultilabelAccuracy, 
    MultilabelF1Score, 
    MultilabelPrecision, 
    MultilabelRecall
)


class ConceptModel(LightningModule):
    def __init__(
        self, 
        lr_backbone: float,
        lr_class_head: float,
        bottleneck_num_classes: int, 
        backbone: nn.Module, 
        classifier_head: nn.Module,
    ) -> None:
        """LightningModule for a concept model.

            Args:
                lr_backbone (float): learning rate for backbone
                lr_class_head (float): learning rate for classifier head
                bottleneck_num_classes (int): number of classes for the classifier head
                backbone (nn.Module): backbone model (e.g. resnet, vgg...)
                classifier_head (nn.Module): classifier head model (e.g. linear layer, mlp...)
        """
        super().__init__()
        self.lr_backbone = lr_backbone
        self.lr_class_head = lr_class_head
        self.num_classes = bottleneck_num_classes
        self.backbone = backbone
        self.classifier_head = classifier_head
        self.criterion = nn.BCEWithLogitsLoss()

        metrics: dict[str, Metric | MetricCollection] = {
            "Accuracy": MultilabelAccuracy(num_labels=self.num_classes),
            "Precision": MultilabelPrecision(num_labels=self.num_classes),
            "Recall": MultilabelRecall(num_labels=self.num_classes),
            "F1": MultilabelF1Score(num_labels=self.num_classes)
        }
        self.val_metrics = MetricCollection(metrics, prefix="validation/")
        self.test_metrics = MetricCollection(metrics, prefix="test/")
        self.predict_metrics = MetricCollection(metrics, prefix="prediction/")
        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        return self.classifier_head(features)

    def training_step(self, batch: Tensor) -> Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log("train/loss", loss, on_step=True, prog_bar=True, batch_size=targets.size(0))
        return loss

    def validation_step(self, batch: Tensor) -> None:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log("validation/loss", loss, on_epoch=True, prog_bar=True, batch_size=targets.size(0))
        self.val_metrics.update(torch.sigmoid(outputs), targets)
        
    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch: Tensor) -> None:
        inputs, targets = batch
        outputs = self(inputs)
        self.test_metrics.update(torch.sigmoid(outputs), targets)

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        self.test_metrics.reset()

    def predict_step(self, batch: Tensor) -> Tensor:
        inputs, _ = batch
        logits = self(inputs)
        return logits

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW([
            {"params": self.backbone.parameters(), "lr": self.lr_backbone},
            {"params": self.classifier_head.parameters(), 
             "lr": self.lr_class_head, "weight_decay": 1e-4}
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "monitor": "validation/loss"
            }
        }
