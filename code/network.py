"""Network: MobileNetV3-Small + testa di deconvoluzione (17 heatmap).
"""
import torch
import torch.nn as nn
from torchvision import models


class DeconvHead(nn.Module):
    """3 layer di ConvTranspose2d (stride 2 ciascuno -> upsample 8x) + 1x1 finale."""

    def __init__(self, in_channels, num_keypoints, num_deconv_layers=3, deconv_channels=256):
        super().__init__()
        layers = []
        ch_in = in_channels
        for _ in range(num_deconv_layers):
            layers += [
                nn.ConvTranspose2d(ch_in, deconv_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(deconv_channels),
                nn.ReLU(inplace=True),
            ]
            ch_in = deconv_channels
        self.deconv_layers = nn.Sequential(*layers)
        self.final_layer = nn.Conv2d(deconv_channels, num_keypoints, kernel_size=1)

    def forward(self, x):
        x = self.deconv_layers(x)
        return self.final_layer(x)


class PoseMobileNet(nn.Module):
    def __init__(self, num_keypoints, input_size, pretrained=True):
        super().__init__()
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = backbone.features
        # canali in uscita calcolati a runtime (non hardcoded)
        with torch.no_grad():
            dummy = torch.randn(1, 3, *input_size)
            backbone_out_channels = self.backbone(dummy).shape[1]
        self.head = DeconvHead(backbone_out_channels, num_keypoints)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)
