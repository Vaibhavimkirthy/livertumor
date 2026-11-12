import torch

from sanet import SANet

model = SANet(channels=256)

x = torch.randn(2, 256, 64, 64)

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)

print("SANet working successfully!")