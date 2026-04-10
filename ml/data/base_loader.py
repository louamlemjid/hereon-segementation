import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, img_size=(128, 128)):
        self.root_dir = root_dir
        self.img_size = img_size

        self.image_ids = sorted(
            os.listdir(os.path.join(root_dir, "images"))
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]

        img_path = os.path.join(self.root_dir, "images", img_name)
        mask_path = os.path.join(self.root_dir, "masks", img_name)
        
        # -------------------
        # IMAGE
        # -------------------
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).permute(2, 0, 1)

        # -------------------
        # MASK
        # -------------------
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Mask not found: {mask_path}")

        mask = cv2.resize(mask, self.img_size)
        mask = (mask > 127).astype(np.float32)

        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask