import torch
import torch.nn as nn


class SmallVoiceNet(nn.Module):
    """
    A tiny ConvNet for voice classification.
    Designed to be lightweight and fast for low-latency inference.
    """

    def __init__(self, n_mels: int = 64, n_classes: int = 2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        logits = self.head(x)
        return logits
