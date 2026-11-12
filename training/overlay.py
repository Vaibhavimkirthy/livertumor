import cv2
import numpy as np
import os


class OverlayGenerator:

    def __init__(self):
        pass

    def generate_overlay(self, image_path, mask, output_path):

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Image not found.")

        # Convert boolean mask to uint8
        mask = (mask * 255).astype(np.uint8)

        # Resize mask if necessary
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # Create red overlay
        color_mask = np.zeros_like(image)
        color_mask[:, :, 2] = mask

        alpha = 0.4

        overlay = cv2.addWeighted(
            image,
            1.0,
            color_mask,
            alpha,
            0
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cv2.imwrite(output_path, overlay)

        return output_path