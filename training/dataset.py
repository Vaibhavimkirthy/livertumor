import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class LiverDataset(Dataset):

    def __init__(self, image_dir, mask_dir, image_size=256):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        self.images = sorted(
            [
                f for f in os.listdir(image_dir)
                if f.endswith(".png")
            ]
        )

        self.masks = sorted(
            [
                f for f in os.listdir(mask_dir)
                if f.endswith(".png")
            ]
        )

        # Make sure image and mask counts match
        if len(self.images) != len(self.masks):
            raise RuntimeError(
                f"Image/mask count mismatch: "
                f"{len(self.images)} images, "
                f"{len(self.masks)} masks"
            )

        # Make sure filenames match
        if set(self.images) != set(self.masks):

            missing_masks = set(self.images) - set(self.masks)
            missing_images = set(self.masks) - set(self.images)

            raise RuntimeError(
                f"Image/mask filenames do not match.\n"
                f"Missing masks: {list(missing_masks)[:10]}\n"
                f"Missing images: {list(missing_images)[:10]}"
            )

        self.length = len(self.images)

        print("Dataset Loaded")
        print("Images:", len(self.images))
        print("Masks :", len(self.masks))


    def __len__(self):

        return self.length


    def __getitem__(self, idx):

        image_path = os.path.join(
            self.image_dir,
            self.images[idx]
        )

        mask_path = os.path.join(
            self.mask_dir,
            self.masks[idx]
        )

        # --------------------------------------------------
        # Read image
        # --------------------------------------------------

        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )

        # --------------------------------------------------
        # Read mask WITHOUT converting to grayscale
        # --------------------------------------------------

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_UNCHANGED
        )

        if image is None:
            raise FileNotFoundError(image_path)

        if mask is None:
            raise FileNotFoundError(mask_path)

        # --------------------------------------------------
        # Resize image
        # --------------------------------------------------

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR
        )

        # --------------------------------------------------
        # Resize mask
        # Use nearest-neighbor to preserve mask values
        # --------------------------------------------------

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        # --------------------------------------------------
        # Convert mask to binary 0/1
        # --------------------------------------------------

        if len(mask.shape) == 3:

            mask = np.any(
                mask > 0,
                axis=2
            )

        else:

            mask = mask > 0

        mask = mask.astype(np.float32)

        # --------------------------------------------------
        # Normalize image
        # --------------------------------------------------

        image = image.astype(
            np.float32
        ) / 255.0

        # --------------------------------------------------
        # HWC -> CHW
        # --------------------------------------------------

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # --------------------------------------------------
        # Convert to tensors
        # --------------------------------------------------

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        )

        # Add channel dimension
        mask = mask.unsqueeze(0)

        return image, mask