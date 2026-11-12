import os
import sys
import torch

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, BASE_DIR)


from segment_anything.build_sam import sam_model_registry


CHECKPOINT = os.path.join(
    BASE_DIR,
    "weights",
    "medsam_vit_b.pth"
)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


print("Loading MedSAM...")


medsam_model = sam_model_registry["vit_b"](
    checkpoint=CHECKPOINT
)


medsam_model.to(DEVICE)

medsam_model.eval()


print("MedSAM Loaded Successfully!")