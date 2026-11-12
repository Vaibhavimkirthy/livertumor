import os
import cv2
import numpy as np
import torch

from training.sannet_model import SANNetSegmentation


# ==========================================================
# BASE DIRECTORY
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Device:", device)
print("=" * 60)


# ==========================================================
# MODEL
# ==========================================================

model = SANNetSegmentation()


# ==========================================================
# BEST MODEL
# ==========================================================

weights_path = os.path.join(
    BASE_DIR,
    "weights",
    "best_sannet_model.pth"
)


if not os.path.exists(weights_path):

    raise FileNotFoundError(
        f"Best SANet model not found:\n{weights_path}\n\n"
        "Train the model first so that "
        "best_sannet_model.pth exists."
    )


checkpoint = torch.load(
    weights_path,
    map_location=device
)


# ==========================================================
# LOAD STATE DICT
# ==========================================================

if isinstance(checkpoint, dict):

    model.load_state_dict(
        checkpoint
    )

else:

    raise RuntimeError(
        "The saved SANet checkpoint is not a valid "
        "PyTorch state_dict."
    )


model.to(device)
model.eval()


print("Best SANet Model Loaded Successfully!")
print("Weights:", weights_path)
print("=" * 60)


# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_image(
    image_path,
    save_path=None,
    gt_mask_path=None
):

    # ======================================================
    # CHECK IMAGE
    # ======================================================

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            image_path
        )


    img = cv2.imread(
        image_path,
        cv2.IMREAD_COLOR
    )


    if img is None:

        raise ValueError(
            "Unable to read image."
        )


    original_h, original_w = img.shape[:2]


    # ======================================================
    # IMAGE PREPROCESSING
    # ======================================================

    image = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )


    image = cv2.resize(
        image,
        (256, 256),
        interpolation=cv2.INTER_LINEAR
    )


    image = image.astype(
        np.float32
    ) / 255.0


    image = np.transpose(
        image,
        (2, 0, 1)
    )


    image = np.expand_dims(
        image,
        axis=0
    )


    image = torch.tensor(
        image,
        dtype=torch.float32
    ).to(device)


    # ======================================================
    # MODEL PREDICTION
    # ======================================================

    with torch.no_grad():

        output = model(
            image
        )

        probability = torch.sigmoid(
            output
        )

        probability = (
            probability
            .squeeze()
            .cpu()
            .numpy()
        )


    # ======================================================
    # PROBABILITY INFORMATION
    # ======================================================

    print("=" * 60)

    print(
        "Probability Min :",
        float(probability.min())
    )

    print(
        "Probability Max :",
        float(probability.max())
    )

    print(
        "Probability Mean:",
        float(probability.mean())
    )

    print("=" * 60)


    # ======================================================
    # BINARY PREDICTION
    # ======================================================

    # SAME threshold used during validation
    threshold = 0.5


    pred = (
        probability >= threshold
    ).astype(
        np.uint8
    )


    # ======================================================
    # TUMOR PIXELS
    # ======================================================

    tumor_pixels = int(
        np.sum(pred)
    )


    tumor_ratio = (
        tumor_pixels /
        pred.size
    )


    # ======================================================
    # DIAGNOSIS
    # ======================================================

    if tumor_ratio > 0:

        diagnosis = "Tumor Detected"

    else:

        diagnosis = "No Tumor Detected"

    # =====================================================
    # Tumor Severity
    # =====================================================

    if tumor_pixels == 0:

        severity = "No Tumor"

    elif tumor_ratio < 0.005:

        severity = "Small"

    elif tumor_ratio < 0.02:

        severity = "Moderate"

    else:

        severity = "Large"


    # ======================================================
    # DISPLAY SCORE
    # ======================================================

    # This is NOT used as model accuracy.
    # It simply reports the mean probability
    # over predicted tumor pixels.

    if tumor_pixels > 0:

        confidence_value = float(
            probability[pred == 1].mean()
        )

    else:

        confidence_value = float(
            probability.mean()
        )


    confidence = (
        f"{confidence_value * 100:.2f}%"
    )


    # ======================================================
    # CREATE OUTPUT MASK
    # ======================================================

    mask = (
        pred * 255
    ).astype(
        np.uint8
    )


    mask = cv2.resize(
        mask,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    )


    # ======================================================
    # SAVE PREDICTED MASK
    # ======================================================

    if save_path is None:

        save_folder = os.path.join(
            BASE_DIR,
            "static",
            "results"
        )

        os.makedirs(
            save_folder,
            exist_ok=True
        )

        save_path = os.path.join(
            save_folder,
            "predicted_mask.png"
        )


    save_folder = os.path.dirname(
        save_path
    )

    if save_folder:

        os.makedirs(
            save_folder,
            exist_ok=True
        )


    success = cv2.imwrite(
        save_path,
        mask
    )


    if not success:

        raise RuntimeError(
            f"Could not save predicted mask:\n{save_path}"
        )


    print("=" * 60)
    print("Mask saved successfully!")
    print("Saved at:", save_path)
    print("Tumor pixels:", tumor_pixels)
    print("=" * 60)


    # ======================================================
    # METRICS
    # ======================================================

    dice = "N/A"
    iou = "N/A"
    precision = "N/A"
    recall = "N/A"


    if (
        gt_mask_path is not None
        and os.path.exists(gt_mask_path)
    ):

        gt = cv2.imread(
            gt_mask_path,
            cv2.IMREAD_UNCHANGED
        )


        if gt is None:

            print(
                "WARNING: Could not read ground-truth mask."
            )

        else:

            # ----------------------------------------------
            # Convert RGB/RGBA mask to binary
            # ----------------------------------------------

            if len(gt.shape) == 3:

                gt = np.any(
                    gt > 0,
                    axis=2
                )

            else:

                gt = gt > 0


            gt = gt.astype(
                np.uint8
            )


            # ----------------------------------------------
            # Resize to prediction size
            # ----------------------------------------------

            gt = cv2.resize(
                gt,
                (256, 256),
                interpolation=cv2.INTER_NEAREST
            )


            # ----------------------------------------------
            # Intersection / Union
            # ----------------------------------------------

            intersection = np.logical_and(
                pred == 1,
                gt == 1
            ).sum()


            union = np.logical_or(
                pred == 1,
                gt == 1
            ).sum()


            tp = intersection


            fp = np.logical_and(
                pred == 1,
                gt == 0
            ).sum()


            fn = np.logical_and(
                pred == 0,
                gt == 1
            ).sum()


            # ----------------------------------------------
            # Dice
            # ----------------------------------------------

            dice_value = (
                2.0 * intersection
                /
                (
                    pred.sum()
                    + gt.sum()
                    + 1e-8
                )
            )


            # ----------------------------------------------
            # IoU
            # ----------------------------------------------

            iou_value = (
                intersection
                /
                (
                    union
                    + 1e-8
                )
            )


            # ----------------------------------------------
            # Precision
            # ----------------------------------------------

            precision_value = (
                tp
                /
                (
                    tp
                    + fp
                    + 1e-8
                )
            )


            # ----------------------------------------------
            # Recall
            # ----------------------------------------------

            recall_value = (
                tp
                /
                (
                    tp
                    + fn
                    + 1e-8
                )
            )


            dice = (
                f"{dice_value * 100:.2f}%"
            )

            iou = (
                f"{iou_value * 100:.2f}%"
            )

            precision = (
                f"{precision_value * 100:.2f}%"
            )

            recall = (
                f"{recall_value * 100:.2f}%"
            )


    # ======================================================
    # FINAL INFORMATION
    # ======================================================

    print("=" * 60)
    print("Prediction Complete")
    print("=" * 60)

    print(
        "Diagnosis    :",
        diagnosis
    )

    print(
        "Confidence   :",
        confidence
    )

    print(
        "Tumor Pixels :",
        tumor_pixels
    )

    print(
        "Dice Score   :",
        dice
    )

    print(
        "IoU          :",
        iou
    )

    print(
        "Precision    :",
        precision
    )

    print(
        "Recall       :",
        recall
    )

    print("=" * 60)


    return (
        save_path,
        diagnosis,
        confidence,
        severity,
        dice,
        iou,
        precision,
        recall
    )