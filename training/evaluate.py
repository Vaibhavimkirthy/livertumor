import cv2
import numpy as np

from training.utils import (
    dice_score,
    iou_score,
    precision_score,
    recall_score
)


class Evaluator:

    def __init__(self):
        pass

    def evaluate(self, pred_mask, gt_mask):

        pred = cv2.imread(pred_mask, 0)
        gt = cv2.imread(gt_mask, 0)

        pred = (pred > 127).astype(np.uint8)
        gt = (gt > 127).astype(np.uint8)

        dice = dice_score(pred, gt)
        iou = iou_score(pred, gt)
        precision = precision_score(pred, gt)
        recall = recall_score(pred, gt)

        return {
            "dice": round(float(dice), 4),
            "iou": round(float(iou), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4)
        }