import os
from torch.utils.data import DataLoader

from dataset import LiverDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset = LiverDataset(
    image_dir=os.path.join(BASE_DIR, "datasets", "train", "images"),
    mask_dir=os.path.join(BASE_DIR, "datasets", "train", "masks")
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

images, masks, boxes = next(iter(loader))

print("Images :", images.shape)
print("Masks  :", masks.shape)
print("Boxes  :", boxes.shape)
print(boxes)