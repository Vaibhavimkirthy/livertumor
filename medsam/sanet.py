import torch
import torch.nn as nn
import torchvision.models as models

from training.sanet import SANet


class SANNetSegmentation(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet34(weights=None)

        self.encoder = nn.Sequential(*list(backbone.children())[:-2])

        self.sanet = SANet(channels=512)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.sanet(x)
        x = self.decoder(x)
        return x