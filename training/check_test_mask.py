import cv2
import numpy as np

mask=cv2.imread(
"../datasets/test/masks/volume-115_92.png",
0
)

print("Tumor pixels:",np.sum(mask>0))