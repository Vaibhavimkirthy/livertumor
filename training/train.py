import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

from dataset import LiverDataset


# ===========================
# Hyperparameters
# ===========================

IMAGE_SIZE = 256
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 2


# ===========================
# Paths
# ===========================

TRAIN_IMAGE_DIR = "datasets/train/images"
TRAIN_MASK_DIR = "datasets/train/masks"

VAL_IMAGE_DIR = "datasets/val/images"
VAL_MASK_DIR = "datasets/val/masks"

WEIGHTS_DIR = "weights"
MODEL_SAVE_PATH = "weights/liver_model.pth"


# ===========================
# Training Function
# ===========================

def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    running_loss = 0.0

    progress = tqdm(
        loader,
        desc="Training",
        total=len(loader)
    )

    for images, masks in progress:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return running_loss / len(loader)



# ===========================
# Validation Function
# ===========================

def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0

    print("\n" + "=" * 50)
    print("Starting Validation...")
    print("=" * 50)


    progress = tqdm(
        loader,
        desc="Validation",
        total=len(loader)
    )


    with torch.no_grad():

        for batch_idx, (images, masks) in enumerate(progress):

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(outputs, masks)

            running_loss += loss.item()


            progress.set_postfix(
                batch=batch_idx + 1,
                loss=f"{loss.item():.4f}"
            )


    avg_loss = running_loss / len(loader)


    print("\nValidation Finished!")
    print(
        f"Average Validation Loss : {avg_loss:.4f}"
    )


    return avg_loss



# ===========================
# Main Training
# ===========================

def main():


    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )


    print("=" * 50)
    print("Using Device :", device)
    print("=" * 50)



    # ===========================
    # Dataset
    # ===========================


    train_dataset = LiverDataset(
        image_dir=TRAIN_IMAGE_DIR,
        mask_dir=TRAIN_MASK_DIR,
        image_size=IMAGE_SIZE
    )


    val_dataset = LiverDataset(
        image_dir=VAL_IMAGE_DIR,
        mask_dir=VAL_MASK_DIR,
        image_size=IMAGE_SIZE
    )


    print(
        "Training Images   :",
        len(train_dataset)
    )

    print(
        "Validation Images :",
        len(val_dataset)
    )



    # ===========================
    # Data Loader
    # ===========================


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )



    # ===========================
    # Model
    # ===========================


    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )


    model = model.to(device)



    # ===========================
    # Loss Function
    # ===========================


    criterion = nn.BCEWithLogitsLoss()



    # ===========================
    # Optimizer
    # ===========================


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )



    # ===========================
    # Save Folder
    # ===========================


    os.makedirs(
        WEIGHTS_DIR,
        exist_ok=True
    )


    best_val_loss = float("inf")



    # ===========================
    # Training Loop
    # ===========================


    for epoch in range(EPOCHS):


        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )


        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )


        print(
            "\nTraining Finished Successfully!"
        )


        val_loss = validate(
            model,
            val_loader,
            criterion,
            device
        )


        print(
            f"Training Loss   : {train_loss:.4f}"
        )


        print(
            f"Validation Loss : {val_loss:.4f}"
        )



        if val_loss < best_val_loss:


            best_val_loss = val_loss


            torch.save(
                model.state_dict(),
                MODEL_SAVE_PATH
            )


            print(
                "Best model saved!"
            )



    print(
        "\nTraining Completed Successfully!"
    )


    print(
        "Model saved at:",
        MODEL_SAVE_PATH
    )



if __name__ == "__main__":
    main()