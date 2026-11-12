import torch
import torch.nn as nn


class SliceAwareAttention(nn.Module):
    """
    Slice-Aware Attention Module
    """

    def __init__(self, channels):

        super().__init__()

        reduction = max(channels // 8, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Linear(channels, reduction),

            nn.ReLU(inplace=True),

            nn.Linear(reduction, channels),

            nn.Sigmoid()

        )

    def forward(self, x):

        b, c, _, _ = x.size()

        attention = self.avg_pool(x).view(b, c)

        attention = self.fc(attention)

        attention = attention.view(b, c, 1, 1)

        return x * attention


class SANet(nn.Module):
    """
    Slice-Aware Network Block
    """

    def __init__(self, channels=512):

        super().__init__()

        self.slice_attention = SliceAwareAttention(channels)

        self.conv = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(channels),

            nn.ReLU(inplace=True),

        )

    def forward(self, x):

        residual = x

        x = self.slice_attention(x)

        x = self.conv(x)

        x = x + residual

        return x