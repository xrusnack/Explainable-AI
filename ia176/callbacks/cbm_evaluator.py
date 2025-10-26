"""Lightning callbacks to evaluate a decision tree on CBM concept predictions."""

import pickle
import mlflow
import torch
from typing import cast
from pathlib import Path
from typing import Optional
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from torch import Tensor
from lightning import Callback, LightningModule, Trainer


class EvaluationCallback(Callback):
    def __init__(self, tree_path: str):
        super().__init__()
        self.tree_path = Path(tree_path)
        self.all_predictions: list[Tensor] = []
        self.all_targets: list[Tensor] = []
        self.tree: Optional[DecisionTreeClassifier] = None

    def on_predict_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        if self.tree is None:
            with open(self.tree_path, "rb") as f:
                self.tree = cast(DecisionTreeClassifier, pickle.load(f))
        
        X = torch.cat(self.all_predictions, dim=0).cpu().numpy()
        y = torch.cat(self.all_targets, dim=0).cpu().numpy()

        y_pred = self.tree.predict(X)
        y_prob = self.tree.predict_proba(X)[:, 1]

        acc = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        auc = roc_auc_score(y, y_prob)

        print(f"Decision Tree Accuracy: {acc:.4f}")
        print(f"Decision Tree ROC AUC: {auc:.4f}")
        print("Classification Report:\n", report)

        mlflow.log_metric("decision_tree/accuracy", acc)
        mlflow.log_metric("decision_tree/auc", auc)

        self.all_predictions.clear()
        self.all_targets.clear()

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
        self.all_targets.append(targets)  # targets are the "Attractive" labels
        

class EvaluationTopKCallback(EvaluationCallback):
    def __init__(self, tree_path: str, top_k_indices: list[int], num_concepts: int):
        super().__init__(tree_path)
        self.top_k_indices = top_k_indices
        self.num_concepts = num_concepts
        self.mask = self._get_mask()

    def _get_mask(self) -> Tensor:
        mask = torch.zeros(self.num_concepts, dtype=torch.bool)
        mask[self.top_k_indices] = True
        return mask

    def on_predict_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        if self.tree is None:
            with open(self.tree_path, "rb") as f:
                self.tree = cast(DecisionTreeClassifier, pickle.load(f))
        
        X = torch.cat(self.all_predictions, dim=0).cpu().numpy()
        X_masked = X[:, self.mask.cpu().numpy()]
        y = torch.cat(self.all_targets, dim=0).cpu().numpy()

        y_pred = self.tree.predict(X_masked)
        y_prob = self.tree.predict_proba(X_masked)[:, 1]

        acc = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        auc = roc_auc_score(y, y_prob)

        print(f"Decision Tree Accuracy: {acc:.4f}")
        print(f"Decision Tree ROC AUC: {auc:.4f}")
        print("Classification Report:\n", report)

        mlflow.log_metric("decision_tree/accuracy", acc)
        mlflow.log_metric("decision_tree/auc", auc)

        self.all_predictions.clear()
        self.all_targets.clear()
