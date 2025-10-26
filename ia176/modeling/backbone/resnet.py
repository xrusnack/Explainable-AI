from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18

from ia176.ia176_typing import ResnetParams


class ResNetBackbone(nn.Module):
    def __init__(self, parameters: ResnetParams) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if parameters["pretrained"] else None
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Identity()
        self.out_features = 512

        if not parameters["requires_grad"]:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
