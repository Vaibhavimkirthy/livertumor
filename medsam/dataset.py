import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class LiverDataset(Dataset):

    def __init__(self, image_dir, mask_dir, image_size=1024):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".png")
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_name = self.images[idx]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        mask_path = os.path.join(
            self.mask_dir,
            image_name
        )

        # ---------------------------------
        # Read Image
        # ---------------------------------
        image = cv2.imread(image_path)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (self.image_size, self.image_size)
        )

        image = image.astype(np.float32) / 255.0

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        # ---------------------------------
        # Read Mask
        # ---------------------------------
        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        mask = (mask > 0).astype(np.uint8)

        # ---------------------------------
        # Generate Bounding Box
        # ---------------------------------
        ys, xs = np.where(mask == 1)

        if len(xs) == 0 or len(ys) == 0:

            bbox = np.array(
                [0, 0, self.image_size, self.image_size],
                dtype=np.float32
            )

        else:

            x_min = xs.min()
            y_min = ys.min()
            x_max = xs.max()
            y_max = ys.max()

            bbox = np.array(
                [x_min, y_min, x_max, y_max],
                dtype=np.float32
            )

        # ---------------------------------
        # Convert to Torch
        # ---------------------------------
        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        bbox = torch.tensor(
            bbox,
            dtype=torch.float32
        )

        return image, mask, bbox