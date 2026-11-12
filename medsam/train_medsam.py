import torch
import torch.nn as nn

from medsam_model import medsam_model
from sanet import SANet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Creating MedSAM + SANet Pipeline...")

# MedSAM
model = medsam_model.to(device)

# SANet
sanet = SANet(channels=256).to(device)

# Optimizer
optimizer = torch.optim.Adam(
    list(model.parameters()) +
    list(sanet.parameters()),
    lr=1e-4
)

criterion = nn.BCEWithLogitsLoss()

print("Pipeline created successfully!")

print(model)
print(sanet)