import cv2
import numpy as np
import torch

from segment_anything import SamPredictor

from models.medsam import MedSAM

from config import DEVICE


class Predictor:

    def __init__(self):

        self.model = MedSAM().get_model()

        self.predictor = SamPredictor(self.model)

    def predict(self, image_path):

        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.predictor.set_image(image)

        h, w = image.shape[:2]

        box = np.array([[0, 0, w, h]])

        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=False
        )

        return masks[0], scores[0]