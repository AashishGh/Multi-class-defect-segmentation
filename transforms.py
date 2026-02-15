import cv2
from albumentations.pytorch import ToTensorV2
import albumentations as A

# Augmentations
def get_train_transforms():
    return A.Compose([
        A.Resize(512, 256),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.08, rotate_limit=7,
                           border_mode=cv2.BORDER_REFLECT, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3),
        A.RandomBrightnessContrast(0.08, 0.08, p=0.3),
        A.GaussNoise(var_limit=(5.0,15.0), p=0.15),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.25,0.25,0.25)),  # backboneless-friendly
        ToTensorV2()
    ])
# (Remove VerticalFlip and RandomRotate90 for panoramics; avoid RandomResizedCrop)

# Validation transforms: resize + normalize only (no aug)
def get_val_transforms():
    return A.Compose([
        A.Resize(512, 256),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.25,0.25,0.25)),
        ToTensorV2()
    ])