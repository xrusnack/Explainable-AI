from typing import TypedDict
from torch_geometric.transforms import BaseTransform

class ResnetParams(TypedDict):
    requires_grad: bool
    pretrained: bool
    
class ClassifierHeadParams(TypedDict):
    input_dim: int
    num_concepts: int

type Transforms = list[BaseTransform] | BaseTransform