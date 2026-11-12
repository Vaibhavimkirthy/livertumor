import matplotlib.pyplot as plt
from dataset import LiverDataset

# ============================
# Dataset Paths
# ============================

TRAIN_IMAGE_DIR = "../datasets/train/images"
TRAIN_MASK_DIR = "../datasets/train/masks"

# ============================
# Load Dataset
# ============================

dataset = LiverDataset(
    image_dir=TRAIN_IMAGE_DIR,
    mask_dir=TRAIN_MASK_DIR,
    image_size=256
)

print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print(f"Total Images : {len(dataset)}")

# ============================
# Load First Image
# ============================

image, mask = dataset[0]

print("\nFirst Sample")

print("Image Shape :", image.shape)
print("Mask Shape  :", mask.shape)

print("Image Type  :", image.dtype)
print("Mask Type   :", mask.dtype)

print("Image Min :", image.min().item())
print("Image Max :", image.max().item())

print("Mask Min :", mask.min().item())
print("Mask Max :", mask.max().item())

# ============================
# Convert Tensor to NumPy
# ============================

image = image.permute(1, 2, 0).numpy()

mask = mask.squeeze().numpy()

# ============================
# Display Image and Mask
# ============================

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(image)
plt.title("CT Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask, cmap="gray")
plt.title("Ground Truth Mask")
plt.axis("off")

plt.show()

print("\nDataset Loader Working Successfully!")