import numpy as np


def dice_score(pred, target):

    pred = pred.astype(np.float32)
    target = target.astype(np.float32)

    intersection = np.sum(pred * target)

    return (2 * intersection + 1e-7) / (
        np.sum(pred) + np.sum(target) + 1e-7
    )


def iou_score(pred, target):

    pred = pred.astype(np.float32)
    target = target.astype(np.float32)

    intersection = np.sum(pred * target)

    union = np.sum(pred) + np.sum(target) - intersection

    return (intersection + 1e-7) / (union + 1e-7)


def precision_score(pred, target):

    tp = np.sum(pred * target)

    fp = np.sum(pred * (1 - target))

    return (tp + 1e-7) / (tp + fp + 1e-7)


def recall_score(pred, target):

    tp = np.sum(pred * target)

    fn = np.sum((1 - pred) * target)

    return (tp + 1e-7) / (tp + fn + 1e-7)