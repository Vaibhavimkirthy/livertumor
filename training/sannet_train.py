import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LiverDataset
from sannet_model import SANNetSegmentation


# ==========================================================
# CONFIGURATION
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4


# ==========================================================
# DATASET PATHS
# ==========================================================

TRAIN_IMAGE_DIR = r"C:\Users\NIHAR\Desktop\Final\liver_project\datasets\train\images"
TRAIN_MASK_DIR = r"C:\Users\NIHAR\Desktop\Final\liver_project\datasets\train\masks"

VAL_IMAGE_DIR = r"C:\Users\NIHAR\Desktop\Final\liver_project\datasets\val\images"
VAL_MASK_DIR = r"C:\Users\NIHAR\Desktop\Final\liver_project\datasets\val\masks"


# ==========================================================
# SAVE DIRECTORY
# ==========================================================

SAVE_DIR = r"C:\Users\NIHAR\Desktop\Final\liver_project\weights"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


# ==========================================================
# TRAIN DATASET
# ==========================================================

print()
print("=" * 60)
print("Loading Training Dataset")
print("=" * 60)

train_dataset = LiverDataset(
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    image_size=IMAGE_SIZE
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print("Training Images:", len(train_dataset))


# ==========================================================
# VALIDATION DATASET
# ==========================================================

print()
print("=" * 60)
print("Loading Validation Dataset")
print("=" * 60)

val_dataset = LiverDataset(
    VAL_IMAGE_DIR,
    VAL_MASK_DIR,
    image_size=IMAGE_SIZE
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Validation Images:", len(val_dataset))


# ==========================================================
# MODEL
# ==========================================================

print()
print("=" * 60)
print("Creating SANet Model")
print("=" * 60)

model = SANNetSegmentation().to(DEVICE)

print("Model Ready")


# ==========================================================
# LOSS FUNCTIONS
# ==========================================================

bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(outputs, masks):

    probabilities = torch.sigmoid(outputs)

    smooth = 1e-6

    intersection = (
        probabilities * masks
    ).sum(dim=(1, 2, 3))

    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + masks.sum(dim=(1, 2, 3))
    )

    dice = (
        (2.0 * intersection + smooth)
        / (denominator + smooth)
    )

    return 1.0 - dice.mean()


def combined_loss(outputs, masks):

    bce = bce_loss(
        outputs,
        masks
    )

    dice = dice_loss(
        outputs,
        masks
    )

    return bce + dice


# ==========================================================
# METRICS
# ==========================================================

def calculate_metrics(outputs, masks):

    probabilities = torch.sigmoid(outputs)

    predictions = (
        probabilities > 0.5
    ).float()

    smooth = 1e-6

    intersection = (
        predictions * masks
    ).sum(dim=(1, 2, 3))

    prediction_area = predictions.sum(
        dim=(1, 2, 3)
    )

    mask_area = masks.sum(
        dim=(1, 2, 3)
    )

    # ------------------------------------------------------
    # Dice
    # ------------------------------------------------------

    dice = (
        (2.0 * intersection + smooth)
        /
        (
            prediction_area
            + mask_area
            + smooth
        )
    )

    # ------------------------------------------------------
    # IoU
    # ------------------------------------------------------

    union = (
        prediction_area
        + mask_area
        - intersection
    )

    iou = (
        (intersection + smooth)
        /
        (union + smooth)
    )

    return (
        dice.mean().item(),
        iou.mean().item()
    )


# ==========================================================
# OPTIMIZER
# ==========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)


# ==========================================================
# BEST MODEL TRACKING
# ==========================================================

best_val_dice = 0.0


# ==========================================================
# TRAINING
# ==========================================================

print()
print("=" * 60)
print("Starting Training")
print("=" * 60)

for epoch in range(EPOCHS):

    # ======================================================
    # TRAIN
    # ======================================================

    model.train()

    total_train_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images, masks in progress:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # ----------------------------------------------
        # Clear gradients
        # ----------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------
        # Forward pass
        # ----------------------------------------------

        outputs = model(images)

        # ----------------------------------------------
        # Loss
        # ----------------------------------------------

        loss = combined_loss(
            outputs,
            masks
        )

        # ----------------------------------------------
        # Backpropagation
        # ----------------------------------------------

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    avg_train_loss = (
        total_train_loss
        /
        len(train_loader)
    )


    # ======================================================
    # VALIDATION
    # ======================================================

    model.eval()

    total_val_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    validation_batches = 0

    with torch.no_grad():

        for images, masks in tqdm(
            val_loader,
            desc="Validation"
        ):

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = combined_loss(
                outputs,
                masks
            )

            dice, iou = calculate_metrics(
                outputs,
                masks
            )

            total_val_loss += loss.item()
            total_dice += dice
            total_iou += iou

            validation_batches += 1


    avg_val_loss = (
        total_val_loss
        /
        validation_batches
    )

    avg_val_dice = (
        total_dice
        /
        validation_batches
    )

    avg_val_iou = (
        total_iou
        /
        validation_batches
    )


    # ======================================================
    # PRINT RESULTS
    # ======================================================

    print()
    print("=" * 60)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss : {avg_train_loss:.6f}"
    )

    print(
        f"Val Loss   : {avg_val_loss:.6f}"
    )

    print(
        f"Val Dice   : {avg_val_dice:.4f} "
        f"({avg_val_dice * 100:.2f}%)"
    )

    print(
        f"Val IoU    : {avg_val_iou:.4f} "
        f"({avg_val_iou * 100:.2f}%)"
    )

    print("=" * 60)


    # ======================================================
    # SAVE EVERY EPOCH
    # ======================================================

    epoch_path = os.path.join(
        SAVE_DIR,
        f"sannet_epoch_{epoch + 1}.pth"
    )

    torch.save(
        model.state_dict(),
        epoch_path
    )

    print(
        f"Saved: {epoch_path}"
    )


    # ======================================================
    # SAVE BEST MODEL
    # ======================================================

    if avg_val_dice > best_val_dice:

        best_val_dice = avg_val_dice

        best_path = os.path.join(
            SAVE_DIR,
            "best_sannet_model.pth"
        )

        torch.save(
            model.state_dict(),
            best_path
        )

        print()
        print(
            "NEW BEST MODEL!"
        )

        print(
            f"Best Validation Dice: "
            f"{best_val_dice * 100:.2f}%"
        )

        print(
            f"Saved: {best_path}"
        )


# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

final_path = os.path.join(
    SAVE_DIR,
    "federated_sannet_model.pth"
)

torch.save(
    model.state_dict(),
    final_path
)


# ==========================================================
# TRAINING COMPLETE
# ==========================================================

print()
print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(
    f"Best Validation Dice: "
    f"{best_val_dice * 100:.2f}%"
)

print(
    "Final Model:",
    final_path
)

print(
    "Best Model:",
    os.path.join(
        SAVE_DIR,
        "best_sannet_model.pth"
    )
)