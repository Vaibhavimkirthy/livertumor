import os
import shutil

BASE = r"E:\liver_project\datasets"

train_images = os.path.join(BASE, "train", "images")
train_masks = os.path.join(BASE, "train", "masks")

images = sorted([f for f in os.listdir(train_images) if f.endswith(".png")])
masks = sorted([f for f in os.listdir(train_masks) if f.endswith(".png")])

# Use exactly the same number of pairs
length = min(len(images), len(masks))

images = images[:length]
masks = masks[:length]

print(f"Total Pairs: {length}")

pairs = list(zip(images, masks))

chunk = len(pairs) // 5
remainder = len(pairs) % 5

start = 0

for h in range(1, 6):

    img_dir = os.path.join(BASE, f"hospital{h}", "images")
    mask_dir = os.path.join(BASE, f"hospital{h}", "masks")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    extra = 1 if h <= remainder else 0
    end = start + chunk + extra

    hospital_pairs = pairs[start:end]

    print(f"Hospital {h}: {len(hospital_pairs)} pairs")

    for img_name, mask_name in hospital_pairs:

        shutil.copy2(
            os.path.join(train_images, img_name),
            os.path.join(img_dir, img_name)
        )

        shutil.copy2(
            os.path.join(train_masks, mask_name),
            os.path.join(mask_dir, mask_name)
        )

    start = end

print("\nHospital dataset created successfully.")