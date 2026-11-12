import os
import torch
import cv2
import numpy as np

from medsam_model import medsam_model


# ==========================
# Configuration
# ==========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "../weights/medsam_liver.pth"

IMAGE_PATH = "../datasets/test/images/sample.png"

OUTPUT_MASK = "../output/medsam_mask.png"
OUTPUT_OVERLAY = "../output/medsam_overlay.png"


# ==========================
# Load Model
# ==========================

model = medsam_model()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model.to(DEVICE)

model.eval()

print("Model loaded successfully")


# ==========================
# Load Image
# ==========================

image = cv2.imread(
    IMAGE_PATH,
    cv2.IMREAD_GRAYSCALE
)

original = image.copy()


image = cv2.resize(
    image,
    (256,256)
)


image = image / 255.0


image = torch.tensor(
    image,
    dtype=torch.float32
)


image = image.unsqueeze(0)
image = image.unsqueeze(0)


image = image.to(DEVICE)



# ==========================
# Prediction
# ==========================

with torch.no_grad():

    output = model(image)


mask = torch.sigmoid(output)

mask = mask.cpu().numpy()

mask = mask[0][0]


mask = (mask > 0.5).astype(np.uint8)



# ==========================
# Save Mask
# ==========================

cv2.imwrite(
    OUTPUT_MASK,
    mask*255
)


# ==========================
# Overlay
# ==========================

original = cv2.resize(
    original,
    (256,256)
)

overlay = cv2.cvtColor(
    original,
    cv2.COLOR_GRAY2BGR
)


overlay[mask==1] = [
    0,
    0,
    255
]


cv2.imwrite(
    OUTPUT_OVERLAY,
    overlay
)


pixels = np.sum(mask)


if pixels > 0:
    print("Tumor Detected")
else:
    print("No Tumor Detected")


print("Tumor Pixels:",pixels)

print("Mask saved:",OUTPUT_MASK)

print("Overlay saved:",OUTPUT_OVERLAY)