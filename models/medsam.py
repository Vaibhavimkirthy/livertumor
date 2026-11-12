import torch
from segment_anything import sam_model_registry
from config import DEVICE, MODEL_PATH


class MedSAM:

    def __init__(self):

        self.model = sam_model_registry["vit_b"](
            checkpoint=MODEL_PATH
        )

        self.model.to(DEVICE)

        self.model.eval()

    def get_model(self):

        return self.model