"""
Small CNN for 8-class defect classification — under 2 MB, train from scratch.
Designed to avoid overfitting: moderate capacity + strong regularization in training.
"""
import torch
import torch.nn as nn

# Target: <= 2 MB = 2 * 1024 * 1024 bytes => <= 524288 float32 params
# This design stays ~1.5–1.8 MB (~400–460K params)


class SmallDefectNet(nn.Module):
    """
    Lightweight CNN: 4 conv blocks + GAP + 2 FC. Fits under 2 MB.
    Input: (B, 3, 224, 224) or (B, 3, 112, 112).
    """

    def __init__(self, num_classes=8, in_channels=3):
        super(SmallDefectNet, self).__init__()
        # 224 -> 112 -> 56 -> 28 -> 14 (4 poolings)
        self.features = nn.Sequential(
            _conv_block(in_channels, 32, pool=True),   # 224->112
            _conv_block(32, 64, pool=True),           # 112->56
            _conv_block(64, 128, pool=True),         # 56->28
            _conv_block(128, 256, pool=True),        # 28->14
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


def _conv_block(in_c, out_c, pool=True):
    layers = [
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    """Approximate model size in MB (float32)."""
    n = count_parameters(model)
    return n * 4 / (1024 * 1024)


if __name__ == '__main__':
    m = SmallDefectNet(num_classes=8)
    print('Params:', count_parameters(m))
    print('Size (MB):', round(model_size_mb(m), 2))
    x = torch.randn(1, 3, 224, 224)
    y = m(x)
    print('Out shape:', y.shape)
