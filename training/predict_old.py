import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

# ==========================================================
# Base Directory
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==========================================================
# Device
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# Load Model
# ==========================================================
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
)

weights = os.path.join(BASE_DIR, "weights", "liver_model.pth")

if not os.path.exists(weights):
    raise FileNotFoundError(f"Model weights not found:\n{weights}")

model.load_state_dict(torch.load(weights, map_location=device))
model.to(device)
model.eval()

print("Model Loaded Successfully!")

# ==========================================================
# Prediction Function
# ==========================================================
def predict_image(image_path, save_path=None, gt_mask_path=None):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found:\n{image_path}")

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Unable to read image:\n{image_path}")

    h, w = img.shape[:2]

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = image.astype(np.float32) / 255.0

    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    image = torch.tensor(image).float().to(device)

    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output)
        output = output.squeeze().cpu().numpy()

    # ======================================================
    # Confidence
    # ======================================================

    confidence = f"{float(output.max()) * 100:.2f}%"

    # ======================================================
    # Binary Mask
    # ======================================================

    pred = (output > 0.5).astype(np.uint8)

    # ======================================================
    # Diagnosis
    # ======================================================

    white_pixels = np.sum(pred)
    tumor_ratio = white_pixels / pred.size

    if tumor_ratio > 0.01:
        diagnosis = "Tumor Detected"
    else:
        diagnosis = "No Tumor Detected"

    # ======================================================
    # Resize Back
    # ======================================================

    mask = (pred * 255).astype(np.uint8)

    mask = cv2.resize(
        mask,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    # ======================================================
    # Save Prediction
    # ======================================================

    if save_path is None:

        output_dir = os.path.join(BASE_DIR, "output")
        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(
            output_dir,
            "predicted_mask.png"
        )

    else:

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cv2.imwrite(save_path, mask)

    # ======================================================
    # Evaluation Metrics
    # ======================================================

    dice = "N/A"
    iou = "N/A"
    precision = "N/A"
    recall = "N/A"

    if gt_mask_path is not None and os.path.exists(gt_mask_path):

        gt = cv2.imread(gt_mask_path, 0)

        gt = cv2.resize(
            gt,
            (256, 256),
            interpolation=cv2.INTER_NEAREST
        )

        gt = (gt > 127).astype(np.uint8)

        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()

        tp = intersection
        fp = np.logical_and(pred == 1, gt == 0).sum()
        fn = np.logical_and(pred == 0, gt == 1).sum()

        dice_value = (2 * intersection) / (
            pred.sum() + gt.sum() + 1e-8
        )

        iou_value = intersection / (union + 1e-8)

        precision_value = tp / (tp + fp + 1e-8)

        recall_value = tp / (tp + fn + 1e-8)

        dice = f"{dice_value:.4f}"
        iou = f"{iou_value:.4f}"
        precision = f"{precision_value:.4f}"
        recall = f"{recall_value:.4f}"

    print("\n==============================")
    print("Prediction Complete")
    print("==============================")
    print("Diagnosis :", diagnosis)
    print("Confidence:", confidence)
    print("Dice      :", dice)
    print("IoU       :", iou)
    print("Precision :", precision)
    print("Recall    :", recall)

    return (
        save_path,
        diagnosis,
        confidence,
        dice,
        iou,
        precision,
        recall
    )

# ==========================================================
# Standalone Test
# ==========================================================

if __name__ == "__main__":

    test_image = os.path.join(
        BASE_DIR,
        "datasets",
        "test",
        "images",
        "volume-116_430.png"
    )

    test_mask = os.path.join(
        BASE_DIR,
        "datasets",
        "test",
        "masks",
        "volume-116_430.png"
    )

    predict_image(
        test_image,
        gt_mask_path=test_mask
    )