from functools import partial
from typing import Any, Optional

from torchvision.models._api import WeightsEnum, Weights
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import handle_legacy_interface, _ovewrite_named_param
from torchvision.utils import _log_api_usage_once

import torch
import torch.nn as nn

class AlexNetCustom(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# @register_model()
# @handle_legacy_interface(weights=("pretrained", AlexNet_Weights.IMAGENET1K_V1))
# def alexnet(*, weights: Optional[AlexNet_Weights] = None, progress: bool = True, **kwargs: Any) -> AlexNet:

#     weights = AlexNet_Weights.verify(weights)

#     if weights is not None:
#         _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

#     model = AlexNet(**kwargs)

#     if weights is not None:
#         model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

#     return model

if __name__ == '__main__':
    model = AlexNetCustom(num_classes=11, dropout=0.5)
    state_dict = torch.load('/kaggle/input/resnet/pytorch/default/1/resnet50-0676ba61.pth', map_location='cpu')
    for key in state_dict:
        print(key)
    model.load_state_dict(state_dict=state_dict, strict=False)