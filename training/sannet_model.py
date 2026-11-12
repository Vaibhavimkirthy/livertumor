import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from training.sanet import SANet


# =====================================================
# Double Convolution Block
# =====================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):

        return self.block(x)


# =====================================================
# Decoder Block
# =====================================================

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):

        super().__init__()

        self.conv = DoubleConv(

            in_channels + skip_channels,

            out_channels

        )

    def forward(self, x, skip):

        x = F.interpolate(

            x,

            size=skip.shape[2:],

            mode="bilinear",

            align_corners=False

        )

        x = torch.cat(

            [x, skip],

            dim=1

        )

        x = self.conv(x)

        return x


    # =====================================================
# SANNet Segmentation Model
# =====================================================

class SANNetSegmentation(nn.Module):

    def __init__(self):

        super().__init__()

        # ==========================================
        # ResNet34 Encoder
        # ==========================================

        encoder = models.resnet34(weights=None)

        self.initial = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu
        )

        self.maxpool = encoder.maxpool

        self.encoder1 = encoder.layer1
        self.encoder2 = encoder.layer2
        self.encoder3 = encoder.layer3
        self.encoder4 = encoder.layer4

        # ==========================================
        # SANet Attention
        # ==========================================

        self.sanet = SANet(
            channels=512
        )

        # ==========================================
        # Decoder
        # ==========================================

        self.decoder4 = DecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256
        )

        self.decoder3 = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128
        )

        self.decoder2 = DecoderBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64
        )

        self.decoder1 = DecoderBlock(
            in_channels=64,
            skip_channels=64,
            out_channels=32
        )

        # ==========================================
        # Final Upsampling
        # ==========================================

        self.final = nn.Sequential(

            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                16,
                1,
                kernel_size=1
            )

        )

    # ==========================================
    # Forward
    # ==========================================

    def forward(self, x):

        # ----------------------------
        # Encoder
        # ----------------------------

        x0 = self.initial(x)          # 64 × 128 × 128

        x1 = self.maxpool(x0)

        x1 = self.encoder1(x1)        # 64 × 64 × 64

        x2 = self.encoder2(x1)        # 128 × 32 × 32

        x3 = self.encoder3(x2)        # 256 × 16 × 16

        x4 = self.encoder4(x3)        # 512 × 8 × 8


        # ----------------------------
        # SANet
        # ----------------------------

        x4 = self.sanet(x4)


        # ----------------------------
        # Decoder
        # ----------------------------

        d4 = self.decoder4(
            x4,
            x3
        )

        d3 = self.decoder3(
            d4,
            x2
        )

        d2 = self.decoder2(
            d3,
            x1
        )

        d1 = self.decoder1(
            d2,
            x0
        )


        # ----------------------------
        # Final Segmentation
        # ----------------------------

        out = self.final(d1)


        # ----------------------------
        # Guarantee output is 256×256
        # ----------------------------

        out = F.interpolate(
            out,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        return out   