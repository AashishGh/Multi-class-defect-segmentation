import cv2
import os
from torch.utils.data import Dataset
import numpy as np
import torch


class DentalSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # TODO: map your actual pixel values → {0,1,2,3}
        # Example if you encoded them as 0, 85, 170, 255:
        # mask = mask // 85
        # Or if they are already 0,1,2,3 you can keep as is.

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # For CrossEntropyLoss: mask should be Long, shape [H, W]
        mask = torch.as_tensor(mask, dtype=torch.long)

        return image, mask
