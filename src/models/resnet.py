"""
ResNet50 with transfer learning for 8-class defect classification
"""
import torch.nn as nn
from torchvision import models


class ResNet50Classifier(nn.Module):
    """ResNet50-based classifier; same setup as Phase 1 binary model."""

    def __init__(self, num_classes=8, pretrained=True, freeze_backbone=False):
        super(ResNet50Classifier, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def unfreeze_partial_backbone(self, num_layers=2):
        for p in self.backbone.parameters():
            p.requires_grad = False
        if num_layers >= 1:
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True
        if num_layers >= 2:
            for p in self.backbone.layer3.parameters():
                p.requires_grad = True
        if num_layers >= 3:
            for p in self.backbone.layer2.parameters():
                p.requires_grad = True
        for p in self.backbone.fc.parameters():
            p.requires_grad = True
