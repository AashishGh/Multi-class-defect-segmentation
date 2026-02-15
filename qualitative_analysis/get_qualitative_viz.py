# qualitative_predict_all_test_original_dim.py

import os
import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.models_mapping import models_dict
from dataset_loader import DentalSegmentationDataset
from transforms import get_val_transforms


# OpenCV uses BGR
COLOR_MAP_BGR = {
    0: (0, 0, 0),          # background
    1: (200, 140, 0),      # caries: dark sky blue  (RGB≈0,140,200)
    2: (133, 0, 199),     # PARL: dark pink        (RGB≈199,21,133)
    3: (0, 0, 255),        # IT: red (keep as is)
}


def colorize_mask(mask_hw: np.ndarray) -> np.ndarray:
    """mask_hw: [H,W] with values {0,1,2,3} -> [H,W,3] BGR"""
    out = np.zeros((mask_hw.shape[0], mask_hw.shape[1], 3), dtype=np.uint8)
    for cls, bgr in COLOR_MAP_BGR.items():
        out[mask_hw == cls] = bgr
    return out


def overlay_on_image(image_bgr: np.ndarray, color_mask_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay only where mask is not background."""
    out = image_bgr.copy()
    fg = np.any(color_mask_bgr != 0, axis=2)
    if fg.any():
        out[fg] = cv2.addWeighted(image_bgr[fg], 1 - alpha, color_mask_bgr[fg], alpha, 0)
    return out


def load_model_like_testpy(model_name: str, device: str, checkpoint: str = None):
    """
    Same checkpoint logic as your test.py:
      default: results/<model_name>/checkpoints/best_model.pth
      supports either raw state_dict OR dict['model_state_dict']
    """
    if checkpoint is None:
        checkpoint = os.path.join("results", model_name, "checkpoints", "best_model.pth")

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = models_dict[model_name].to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict_mask_transformed(model, images: torch.Tensor, device: str) -> np.ndarray:
    """
    images: [1,C,H,W] transformed
    returns pred_mask at transformed size: [H,W] uint8 in {0,1,2,3}
    """
    logits = model(images.to(device))         # [1,4,H,W]
    pred = torch.argmax(logits, dim=1)[0]
    return pred.detach().cpu().numpy().astype(np.uint8)


def run_qualitative_predict_all_original_dim(
    dataset_dir: str,
    model_name: str,
    checkpoint: str = None,
    out_root: str = "results/qualitative",
    alpha: float = 0.45,
    save_color: bool = False,
    save_pred_mask: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # --- EXACT paths like test.py ---
    test_img_dir = os.path.join(dataset_dir, "test_dataset", "images")
    test_mask_dir = os.path.join(dataset_dir, "test_dataset", "masks")

    if not (os.path.isdir(test_img_dir) and os.path.isdir(test_mask_dir)):
        raise FileNotFoundError("test_dataset/{images,masks} not found under dataset_dir")

    test_file_list = sorted(
        [f for f in os.listdir(test_img_dir) if os.path.exists(os.path.join(test_mask_dir, f))]
    )
    if len(test_file_list) == 0:
        raise RuntimeError("No test files found in test_dataset.")

    # --- DataLoader (same style as test.py) ---
    loader = DataLoader(
        DentalSegmentationDataset(test_img_dir, test_mask_dir, test_file_list, transform=get_val_transforms()),
        batch_size=1,
        shuffle=False
    )

    # --- Model ---
    model = load_model_like_testpy(model_name, device=device, checkpoint=checkpoint)

    # --- Output dir ---
    out_dir = os.path.join(out_root, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # --- Iterate, but overlay on RAW ORIGINAL image ---
    for i, (images, _) in tqdm(enumerate(loader), total=len(loader), desc=f"Overlay(pred) on original: {model_name}"):
        image_id = test_file_list[i]
        img_path = os.path.join(test_img_dir, image_id)

        # Load raw/original image (BGR) -> THIS sets the final output size
        raw_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if raw_bgr is None:
            print(f"[WARN] Could not read image: {img_path}, skipping.")
            continue
        orig_h, orig_w = raw_bgr.shape[:2]

        # Predict at transformed size
        pred_t = predict_mask_transformed(model, images, device=device)   # [Ht, Wt]

        # Resize prediction back to original image size (NEAREST to preserve labels)
        pred_orig = cv2.resize(pred_t, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)  # [H,W]

        # Colorize and overlay on original image
        pred_color = colorize_mask(pred_orig)
        pred_overlay = overlay_on_image(raw_bgr, pred_color, alpha=alpha)

        base = os.path.splitext(image_id)[0]
        cv2.imwrite(os.path.join(out_dir, f"{base}_overlay.png"), pred_overlay)

        if save_color:
            cv2.imwrite(os.path.join(out_dir, f"{base}_color.png"), pred_color)

        if save_pred_mask:
            # Save label mask as PNG (0..3). Use uint8.
            cv2.imwrite(os.path.join(out_dir, f"{base}_predmask.png"), pred_orig.astype(np.uint8))

    print(f"\n✅ Done. Saved ORIGINAL-dimension overlays to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="dataset root containing test_dataset/")
    parser.add_argument("--model_name", type=str, required=True, help="model key in models_dict")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path to checkpoint (default: results/<model_name>/checkpoints/best_model.pth)")
    parser.add_argument("--out_root", type=str, default="results/qualitative", help="base output directory")
    parser.add_argument("--alpha", type=float, default=0.45, help="overlay alpha")
    parser.add_argument("--save_color", action="store_true", help="also save pure color mask (original size)")
    parser.add_argument("--save_pred_mask", action="store_true", help="also save raw predicted label mask (0..3)")
    args = parser.parse_args()

    run_qualitative_predict_all_original_dim(
        dataset_dir=args.dataset_dir,
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        out_root=args.out_root,
        alpha=args.alpha,
        save_color=args.save_color,
        save_pred_mask=args.save_pred_mask,
    )
