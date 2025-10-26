import torch
from torch import nn

from ia176.ia176_typing import ClassifierHeadParams


class ConceptPredictorHead(nn.Module):
    def __init__(self, parameters: ClassifierHeadParams) -> None:
        super().__init__()
        self.fc = nn.Linear(parameters["input_dim"], parameters["num_concepts"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
