import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset


class MedSAMDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        image_size=1024
    ):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size


        self.samples = []


        images = sorted(
            os.listdir(image_dir)
        )


        masks = sorted(
            os.listdir(mask_dir)
        )


        for img in images:

            if not img.endswith(".png"):
                continue


            # volume-130_95.png
            slice_id = img.split("_")[-1].replace(".png","")


            matched_mask = None


            for mask in masks:

                if mask.endswith(
                    f"_{slice_id}.png"
                ):

                    matched_mask = mask
                    break


            if matched_mask:

                self.samples.append(
                    (
                        img,
                        matched_mask
                    )
                )


        print("MedSAM Dataset")
        print("Matched samples:",len(self.samples))


    def __len__(self):

        return len(self.samples)



    def __getitem__(self,idx):


        img_name, mask_name = self.samples[idx]


        image = cv2.imread(
            os.path.join(
                self.image_dir,
                img_name
            )
        )


        mask = cv2.imread(
            os.path.join(
                self.mask_dir,
                mask_name
            ),
            0
        )


        image = cv2.resize(
            image,
            (self.image_size,self.image_size)
        )


        mask = cv2.resize(
            mask,
            (self.image_size,self.image_size)
        )


        image = image / 255.0

        mask = mask / 255.0



        ys,xs=np.where(mask>0)


        if len(xs)>0:

            box=[
                xs.min(),
                ys.min(),
                xs.max(),
                ys.max()
            ]

        else:

            box=[
                0,
                0,
                self.image_size,
                self.image_size
            ]


        image=np.transpose(
            image,
            (2,0,1)
        )


        image=torch.tensor(
            image,
            dtype=torch.float32
        )


        mask=torch.tensor(
            mask,
            dtype=torch.float32
        )


        mask=mask.unsqueeze(0)


        box=torch.tensor(
            box,
            dtype=torch.float32
        )


        return image,box,mask