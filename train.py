import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import albumentations as A
import matplotlib.pyplot as plt
import argparse
import csv
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score, fbeta_score
from models.models_mapping import models_dict  # Update with your model path

from dataset_loader import DentalSegmentationDataset
from loss import FocalTverskyLossFG
from transforms import get_train_transforms, get_val_transforms
from evaluate import evaluate

# Dataset class
# class DentalSegmentationDataset(Dataset):
#     def __init__(self, image_dir, mask_dir, file_list, transform=None):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.file_list = file_list
#         self.transform = transform

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
#         img_name = self.file_list[idx]
#         img_path = os.path.join(self.image_dir, img_name)
#         mask_path = os.path.join(self.mask_dir, img_name)

#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         mask = (mask == 255).astype(np.uint8)  # Binary mask: 0 or 1

#         if self.transform:
#             augmented = self.transform(image=image, mask=mask)
#             image = augmented['image']
#             mask = torch.as_tensor(augmented['mask'], dtype=torch.float32)  # For BCEWithLogits

#         return image, mask.unsqueeze(0)  # Shape: [1, H, W]


# class FocalTverskyLossFG(nn.Module):
#     def __init__(self, alpha=0.7, beta=0.3, gamma=1.0, smooth=1e-6):
#         super().__init__()
#         self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth
#     def forward(self, inputs, targets):
#         p = torch.sigmoid(inputs)          # [B,1,H,W]
#         y = targets                        # [B,1,H,W] in {0,1}
#         TP = (p * y).sum()
#         FP = ((1 - y) * p).sum()
#         FN = (y * (1 - p)).sum()
#         t = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
#         return (1 - t) ** self.gamma


# # Augmentations
# def get_train_transforms():
#     return A.Compose([
#         A.Resize(512, 256),
#         A.HorizontalFlip(p=0.5),
#         A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.08, rotate_limit=7,
#                            border_mode=cv2.BORDER_REFLECT, p=0.5),
#         A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3),
#         A.RandomBrightnessContrast(0.08, 0.08, p=0.3),
#         A.GaussNoise(var_limit=(5.0,15.0), p=0.15),
#         A.Normalize(mean=(0.5,0.5,0.5), std=(0.25,0.25,0.25)),  # backboneless-friendly
#         ToTensorV2()
#     ])
# # (Remove VerticalFlip and RandomRotate90 for panoramics; avoid RandomResizedCrop)

# # Validation transforms: resize + normalize only (no aug)
# def get_val_transforms():
#     return A.Compose([
#         A.Resize(512, 256),
#         A.Normalize(mean=(0.5,0.5,0.5), std=(0.25,0.25,0.25)),
#         ToTensorV2()
#     ])


# # Metrics
# def compute_metrics(preds, targets):
#     p = preds.flatten()
#     t = targets.flatten()
#     return {
#         'iou': jaccard_score(t, p, zero_division=0),
#         'precision': precision_score(t, p, zero_division=0),
#         'recall': recall_score(t, p, zero_division=0),
#         'f1': f1_score(t, p, zero_division=0),
#         'f2': fbeta_score(t, p, beta=2, zero_division=0)
#     }


# def evaluate(model, loader, device):
#     model.eval()
#     all_preds, all_targets = [], []
#     with torch.no_grad():
#         for images, masks in loader:
#             images = images.to(device)
#             outputs = model(images)
#             preds = (torch.sigmoid(outputs) > 0.5).long().cpu().numpy()
#             all_preds.append(preds)
#             all_targets.append(masks.long().numpy())
#     return compute_metrics(np.concatenate(all_preds), np.concatenate(all_targets))


# Training pipeline
def run_training(dataset_dir, model_name, epochs=200, batch_size=4, lr=1e-4, resume_checkpoint=None, patience=20):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    os.makedirs(f"results/{model_name}/checkpoints", exist_ok=True)
    os.makedirs(f"results/{model_name}/training_results", exist_ok=True)

    # --- Train paths ---
    img_dir = os.path.join(dataset_dir, 'train_dataset', 'images')
    mask_dir = os.path.join(dataset_dir, 'train_dataset', 'masks')
    file_list = sorted([f for f in os.listdir(img_dir) if os.path.exists(os.path.join(mask_dir, f))])

    # --- Val paths ---
    val_img_dir = os.path.join(dataset_dir, 'val_dataset', 'images')
    val_mask_dir = os.path.join(dataset_dir, 'val_dataset', 'masks')
    if not (os.path.isdir(val_img_dir) and os.path.isdir(val_mask_dir)):
        raise FileNotFoundError("val_dataset/{images,masks} not found under dataset_dir")

    val_file_list = sorted([f for f in os.listdir(val_img_dir) if os.path.exists(os.path.join(val_mask_dir, f))])
    if len(val_file_list) == 0:
        raise RuntimeError("No validation files found in val_dataset.")

    # --- Loaders ---
    loader = DataLoader(
        DentalSegmentationDataset(img_dir, mask_dir, file_list, transform=get_train_transforms()),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        DentalSegmentationDataset(val_img_dir, val_mask_dir, val_file_list, transform=get_val_transforms()),
        batch_size=batch_size, shuffle=False
    )

    model = models_dict[model_name]
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # <<< use lr arg
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    NUM_CLASSES = 4  # background + caries + PARL + IT

    # w[0] = background, w[1] = caries, w[2] = PARL, w[3] = IT
    class_weights = torch.tensor([1.0, 10.0, 10.0, 10.0], dtype=torch.float32).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def loss_fn(out, mask):
        # out: [B, 4, H, W], mask: [B, H, W] (Long)
        return ce_loss(out, mask)

    # <<< this is now "best validation mean IoU over caries+parl+it"
    best_val_miou = -1.0
    epochs_no_improve = 0
    start_epoch = 1

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1

    # CSV logging (interpret IoU, etc. as MEAN over 3 defects)
    with open(f"results/{model_name}/training_results/train_iou.csv", 'a', newline='') as f:
        csv.writer(f).writerow(['Epoch', 'IoU', 'Precision', 'Recall', 'F1', 'F2'])
    with open(f"results/{model_name}/training_results/val_metrics.csv", 'a', newline='') as f:
        csv.writer(f).writerow(['Epoch', 'Val_IoU', 'Val_Precision', 'Val_Recall', 'Val_F1', 'Val_F2'])
    with open(f"results/{model_name}/training_results/train_loss.csv", 'a', newline='') as f:
        csv.writer(f).writerow(['Epoch', 'Loss'])

    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        model.train()
        total_loss = 0.0

        for imgs, masks in tqdm(loader):
            imgs, masks = imgs.to(device), masks.to(device).long()
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        # --- Evaluate on TRAIN (returns dict with caries/parl/it/mean) ---
        train_metrics = evaluate(model, loader, device)

        # --- Evaluate on VAL ---
        val_metrics = evaluate(model, val_loader, device)

        scheduler.step()

        # --- Logging (use MEAN metrics) ---
        train_mean = train_metrics['mean']
        val_mean = val_metrics['mean']

        with open(f"results/{model_name}/training_results/train_loss.csv", 'a', newline='') as f:
            csv.writer(f).writerow([epoch, avg_loss])

        with open(f"results/{model_name}/training_results/train_iou.csv", 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch,
                train_mean['iou'],
                train_mean['precision'],
                train_mean['recall'],
                train_mean['f1'],
                train_mean['f2']
            ])

        with open(f"results/{model_name}/training_results/val_metrics.csv", 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch,
                val_mean['iou'],
                val_mean['precision'],
                val_mean['recall'],
                val_mean['f1'],
                val_mean['f2']
            ])

        print(f" Train Loss: {avg_loss:.4f} | "
              f"Train mIoU(defects): {train_mean['iou']:.4f} | "
              f"Val mIoU(defects): {val_mean['iou']:.4f} | "
              f"Val mF1(defects): {val_mean['f1']:.4f}")

        # --- Early stopping based on VAL mean IoU over caries+parl+it ---
        current_val_miou = val_mean['iou']

        # Save checkpoint (keep only current + best, as before)
        ckpt_path = f"results/{model_name}/checkpoints/epoch{epoch:02d}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, ckpt_path)
        for fname in os.listdir(f"results/{model_name}/checkpoints"):
            if fname.startswith('epoch') and fname.endswith('.pth') and fname != os.path.basename(ckpt_path):
                os.remove(os.path.join(f"results/{model_name}/checkpoints", fname))

        # Save best model based on VAL mean IoU (no baf"results/{model_name}/ckground)
        if current_val_miou > best_val_miou:
            best_val_miou = current_val_miou
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"results/{model_name}/checkpoints/best_model.pth")
            print(f" ✅ Best VAL model updated at epoch {epoch} with Val mIoU(defects) {best_val_miou:.4f}")
        else:
            epochs_no_improve += 1
            print(f" No Val mIoU(defects) improvement for {epochs_no_improve} epoch(s) "
                  f"(best={best_val_miou:.4f}, patience={patience})")

        # --- Early stopping ---
        if epochs_no_improve >= patience:
            print(f"🛑 Early stopping triggered at epoch {epoch} (patience={patience}).")
            break

    # Plot Train IoU curve (unchanged; it's mean over defects now)
    df = pd.read_csv(f"results/{model_name}/training_results/train_iou.csv")
    plt.figure()
    for col in ['IoU', 'Precision', 'Recall', 'F1', 'F2']:
        plt.plot(df['Epoch'], df[col], label=col)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Training Metrics (mean over defects)")
    plt.legend()
    plt.savefig(f"results/{model_name}/training_results/train_iou_curve.png")

    # Plot Val IoU curve
    try:
        dfv = pd.read_csv(f"results/{model_name}/training_results/val_metrics.csv")
        plt.figure()
        for col in ['Val_IoU', 'Val_Precision', 'Val_Recall', 'Val_F1', 'Val_F2']:
            plt.plot(dfv['Epoch'], dfv[col], label=col)
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Validation Metrics (mean over defects)")
        plt.legend()
        plt.savefig(f"results/{model_name}/training_results/val_iou_curve.png")
    except Exception as e:
        print("Validation plot skipped:", e)



# CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='path to DC1000 dataset')
    parser.add_argument('--model_name', type=str, required=True, help='State the model to train')    
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, help='checkpoint file name without path')
    parser.add_argument('--patience', type=int, default=50, help='early-stopping patience (epochs)')  # NEW
    args = parser.parse_args()

    resume_path = f"results/{args.model_name}/checkpoints/{args.resume}.pth" if args.resume else None
    run_training(args.dataset_dir,args.model_name, args.epochs, args.batch_size, args.lr, resume_path, patience=args.patience)


