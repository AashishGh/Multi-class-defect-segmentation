# get_per_image_IOUs.py
import os
import argparse
import csv
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.models_mapping import models_dict
from dataset_loader import DentalSegmentationDataset
from transforms import get_val_transforms


DEFECT_CLASSES = {
    "caries": 1,
    "parl": 2,
    "it": 3,
}


def compute_iou_per_class(pred: torch.Tensor, target: torch.Tensor, cls: int) -> float:
    """
    pred, target: [H, W] torch.LongTensor
    IoU = |A∩B| / |A∪B|, with 0 when union==0 (matches zero_division=0 behavior).
    """
    pred_c = (pred == cls)
    tgt_c = (target == cls)

    intersection = torch.logical_and(pred_c, tgt_c).sum().item()
    union = torch.logical_or(pred_c, tgt_c).sum().item()

    if union == 0:
        return 0.0
    return float(intersection / union)


@torch.no_grad()
def run_per_image_iou(
    dataset_dir: str,
    model_name: str,
    split: str,
    ckpt_path: str,
    out_csv: str,
    device: str,
    num_workers: int = 2,
) -> None:
    # ---- paths ----
    img_dir = os.path.join(dataset_dir, split, "images")
    mask_dir = os.path.join(dataset_dir, split, "masks")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Images dir not found: {img_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Masks dir not found: {mask_dir}")

    file_list = sorted([f for f in os.listdir(img_dir) if os.path.exists(os.path.join(mask_dir, f))])
    if len(file_list) == 0:
        raise RuntimeError(f"No paired image/mask files found in {img_dir} and {mask_dir}")

    # ---- dataset/loader (batch_size=1 for per-image metrics) ----
    ds = DentalSegmentationDataset(
        image_dir=img_dir,
        mask_dir=mask_dir,
        file_list=file_list,
        transform=get_val_transforms(),  # same resize/normalize as eval/val
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    # ---- model ----
    if model_name not in models_dict:
        raise KeyError(
            f"model_name='{model_name}' not in models_dict. "
            f"Available: {list(models_dict.keys())}"
        )

    model = models_dict[model_name]
    model = model.to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    # best_model.pth is saved as model.state_dict(), not a dict with keys
    if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        model.load_state_dict(state, strict=True)
    else:
        raise RuntimeError("Checkpoint format not recognized. Expected a state_dict.")

    model.eval()

    # ---- output dir ----
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # ---- compute & write CSV ----
    rows: List[Dict[str, object]] = []

    for (img, mask), fname in tqdm(zip(loader, file_list), total=len(file_list), desc=f"{model_name} | {split}"):
        img = img.to(device)
        mask = mask.to(device).long()  # expected [B,H,W] or [B,1,H,W]? handle below

        # normalize mask shape to [H,W]
        if mask.ndim == 4 and mask.shape[1] == 1:
            mask_hw = mask[0, 0]
        elif mask.ndim == 3:
            mask_hw = mask[0]
        else:
            raise RuntimeError(f"Unexpected mask shape: {tuple(mask.shape)}")

        logits = model(img)  # expected [B,4,H,W]
        if logits.ndim != 4 or logits.shape[1] < 4:
            raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)} (need [B,4,H,W])")

        pred = torch.argmax(logits, dim=1)  # [B,H,W]
        pred_hw = pred[0].long()

        iou_caries = compute_iou_per_class(pred_hw, mask_hw, DEFECT_CLASSES["caries"])
        iou_parl = compute_iou_per_class(pred_hw, mask_hw, DEFECT_CLASSES["parl"])
        iou_it = compute_iou_per_class(pred_hw, mask_hw, DEFECT_CLASSES["it"])
        carries_flag= int(iou_caries >= 0.0001)
        parl_flag= int(iou_parl >= 0.0001)
        it_flag= int(iou_it >= 0.0001)

        mean_iou=0
        if carries_flag + parl_flag + it_flag >0:
            mean_iou = float((iou_caries + iou_parl + iou_it) / (carries_flag + parl_flag + it_flag))

        rows.append({
            "filename": fname,
            "iou_caries": iou_caries,
            "iou_parl": iou_parl,
            "iou_it": iou_it,
            "mean_iou": mean_iou,
        })

    # write CSV
    fieldnames = ["filename", "iou_caries", "iou_parl", "iou_it", "mean_iou"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # quick summary
    means = np.array([r["mean_iou"] for r in rows], dtype=np.float64)
    print(f"\nSaved: {out_csv}")
    print(f"{model_name} | {split} | per-image mean IoU(avg over defects): {means.mean():.6f} ± {means.std():.6f}")
    print(f"Num images: {len(rows)}")



def sort_csv_by_mean_iou_desc(csv_path: str, mean_col: str = "mean_iou") -> str:
    """
    Sort a per-image IoU CSV by `mean_col` in descending order and overwrite the same file.

    Returns:
        csv_path (str): the path of the sorted CSV (same as input).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if mean_col not in df.columns:
        raise KeyError(f"Column '{mean_col}' not found in CSV. Available columns: {list(df.columns)}")

    # Coerce to numeric in case it's accidentally read as string
    df[mean_col] = pd.to_numeric(df[mean_col], errors="coerce")

    # Sort desc (NaNs, if any, go to bottom)
    df = df.sort_values(by=mean_col, ascending=False, na_position="last").reset_index(drop=True)

    df.to_csv(csv_path, index=False)
    return csv_path



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path containing train_dataset/val_dataset/test_dataset")
    parser.add_argument("--model_name", type=str, required=True, help="One of keys in models_dict (e.g., unet, vmunetv2, ...)")
    parser.add_argument("--split", type=str, default="test_dataset", choices=["train_dataset", "val_dataset", "test_dataset"])
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to checkpoint .pth (default: results/{model_name}/checkpoints/best_model.pth)",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Output CSV path (default: results/{model_name}/qualitative/per_image_iou_{split}.csv)",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = args.ckpt or f"results/{args.model_name}/checkpoints/best_model.pth"
    out_csv = args.out_csv or f"results/{args.model_name}/qualitative/per_image_iou_{args.split}.csv"

    run_per_image_iou(
        dataset_dir=args.dataset_dir,
        model_name=args.model_name,
        split=args.split,
        ckpt_path=ckpt_path,
        out_csv=out_csv,
        device=device,
        num_workers=args.num_workers,
    )
    sort_csv_by_mean_iou_desc(out_csv)
    print(f"Sorted CSV (desc by mean_iou): {out_csv}")



if __name__ == "__main__":
    main()
