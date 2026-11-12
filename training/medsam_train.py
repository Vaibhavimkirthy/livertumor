import os
import sys

# Allow imports from training folder
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.append(BASE_DIR)


import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm


from medsam_dataset import MedSAMDataset
from medsam_model import medsam_model



# ==========================
# Configuration
# ==========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


IMAGE_SIZE = 1024

BATCH_SIZE = 1

EPOCHS = 2

LEARNING_RATE = 1e-5



# Correct paths from E:\liver_project
TRAIN_IMAGES = "../datasets/train/images"

TRAIN_MASKS = "../datasets/train/masks"

SAVE_PATH = "../weights/medsam_liver.pth"



# ==========================
# Training
# ==========================

def train():


    print("=" * 50)
    print("MedSAM Training")
    print("Device:", DEVICE)
    print("=" * 50)



    # Dataset

    dataset = MedSAMDataset(

        image_dir=TRAIN_IMAGES,

        mask_dir=TRAIN_MASKS,

        image_size=IMAGE_SIZE

    )



    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0

    )



    model = medsam_model.to(DEVICE)



    # Freeze SAM image encoder

    for param in model.image_encoder.parameters():

        param.requires_grad = False



    # Train only mask decoder

    for param in model.mask_decoder.parameters():

        param.requires_grad = True



    optimizer = torch.optim.Adam(

        model.mask_decoder.parameters(),

        lr=LEARNING_RATE

    )



    criterion = nn.BCEWithLogitsLoss()



    best_loss = float("inf")



    # ==========================
    # Epoch Loop
    # ==========================


    for epoch in range(EPOCHS):


        model.train()


        total_loss = 0



        progress = tqdm(

            loader,

            desc=f"Epoch {epoch+1}/{EPOCHS}"

        )



        for images, boxes, masks in progress:



            images = images.to(DEVICE)

            boxes = boxes.to(DEVICE)

            masks = masks.to(DEVICE)



            optimizer.zero_grad()



            # SAM Image Encoder

            image_embeddings = model.image_encoder(

                images

            )



            # Prompt Encoder

            sparse_embeddings, dense_embeddings = model.prompt_encoder(

                points=None,

                boxes=boxes,

                masks=None

            )



            # Mask Decoder

            low_res_masks, _ = model.mask_decoder(

                image_embeddings=image_embeddings,

                image_pe=model.prompt_encoder.get_dense_pe(),

                sparse_prompt_embeddings=sparse_embeddings,

                dense_prompt_embeddings=dense_embeddings,

                multimask_output=False

            )



            # Resize output

            predicted_masks = torch.nn.functional.interpolate(

                low_res_masks,

                size=masks.shape[-2:],

                mode="bilinear",

                align_corners=False

            )



            loss = criterion(

                predicted_masks,

                masks

            )



            loss.backward()


            optimizer.step()



            total_loss += loss.item()



            progress.set_postfix(

                loss=loss.item()

            )



        avg_loss = total_loss / len(loader)



        print()

        print(
            "Epoch:",
            epoch + 1,
            "Loss:",
            avg_loss
        )



        # Save best model

        if avg_loss < best_loss:


            best_loss = avg_loss


            os.makedirs(

                "weights",

                exist_ok=True

            )


            torch.save(

                model.state_dict(),

                SAVE_PATH

            )


            print("MedSAM model saved!")



    print()

    print("=" * 50)

    print("MedSAM Training Completed")

    print("Saved:", SAVE_PATH)

    print("=" * 50)




if __name__ == "__main__":

    train()