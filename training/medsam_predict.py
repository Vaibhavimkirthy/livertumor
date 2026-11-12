import os
import sys
import torch
import cv2
import numpy as np


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(BASE_DIR)


from medsam_model import medsam_model


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


IMAGE_PATH = "../datasets/test/images/volume-115_92.png"

OUTPUT_PATH = "../output/medsam_prediction.png"


model = medsam_model.to(DEVICE)

checkpoint = torch.load(
    "../weights/medsam_liver.pth",
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint
)

model.eval()


print("MedSAM weight loaded")



# Load image

image = cv2.imread(
    IMAGE_PATH
)


image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


h,w,_ = image_rgb.shape



# Example bounding box
# whole image box
box = torch.tensor(
    [[0,0,w,h]],
    dtype=torch.float32
).to(DEVICE)



# Resize image for SAM

image_resized = cv2.resize(
    image_rgb,
    (1024,1024)
)


image_tensor = torch.tensor(
    image_resized,
    dtype=torch.float32
).permute(2,0,1)

image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(DEVICE)



with torch.no_grad():

    image_embedding = model.image_encoder(
        image_tensor
    )


    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=None,
        boxes=box,
        masks=None
    )


    low_res_masks,_ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False
    )



mask = torch.sigmoid(
    low_res_masks
)


mask = mask.cpu().numpy()[0][0]


mask = cv2.resize(
    mask,
    (w,h)
)


mask = (mask>0.5).astype(np.uint8)



cv2.imwrite(
    OUTPUT_PATH,
    mask*255
)


print("Prediction completed")
print("Tumor pixels:",np.sum(mask))
print("Saved:",OUTPUT_PATH)