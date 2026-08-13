"""Model definitions: ResNet-18 with a CIFAR stem, and a small CNN."""
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _resnet18_cifar(num_classes: int) -> nn.Module:
    # The default 7x7 stride-2 stem and maxpool are too aggressive for 32x32 images.
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    architecture = (architecture or "resnet18").lower()
    if architecture == "resnet18":
        return _resnet18_cifar(num_classes)
    if architecture in {"simple_cnn", "cnn"}:
        return SimpleCNN(num_classes)
    raise ValueError(f"Unknown architecture '{architecture}'. Use resnet18 or simple_cnn.")
