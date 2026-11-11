"""ParkNet: small CNN trained from scratch that classifies a parking spot as free/occupied.

  RGB input of input_size x input_size (+ the spot mask as a 4th channel with in_channels=4)
  4 blocks: [conv3x3 -> BN -> ReLU] x2 -> maxpool 2x2, with 32/64/128/128 filters (x width)
  global average pooling -> dropout -> linear to num_classes logits

~0.6 M parameters with width=1.0.
"""
from __future__ import annotations

from torch import nn

BASE_CHANNELS = (32, 64, 128, 128)


def conv_bn_relu(c_in: int, c_out: int) -> list[nn.Module]:
    # bias=False: the following BatchNorm absorbs it
    return [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]


class ParkNet(nn.Module):
    def __init__(self, num_classes: int = 2, width: float = 1.0, dropout: float = 0.3, in_channels: int = 3):
        super().__init__()
        channels = [max(8, int(round(c * width))) for c in BASE_CHANNELS]
        layers, c_in = [], in_channels
        for c_out in channels:
            layers += conv_bn_relu(c_in, c_out) + conv_bn_relu(c_out, c_out) + [nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c_in, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.dropout(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
