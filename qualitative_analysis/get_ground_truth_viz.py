# qualitative_ground_truth_all_test_original_dim.py

import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm


# OpenCV BGR colors (updated)
COLOR_MAP_BGR = {
    0: (0, 0, 0),          # background
    1: (200, 140, 0),      # caries: dark sky blue  (RGB≈0,140,200)
    2: (133, 0, 199),     # PARL: dark pink        (RGB≈199,21,133)
    3: (0, 0, 255),        # IT: red
}


def colorize_mask(mask_hw: np.ndarray) -> np.ndarray:
    out = np.zeros((mask_hw.shape[0], mask_hw.shape[1], 3), dtype=np.uint8)
    for cls, bgr in COLOR_MAP_BGR.items():
        out[mask_hw == cls] = bgr
    return out


def overlay_on_image(image_bgr: np.ndarray, color_mask_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = image_bgr.copy()
    fg = np.any(color_mask_bgr != 0, axis=2)
    if fg.any():
        out[fg] = cv2.addWeighted(image_bgr[fg], 1 - alpha, color_mask_bgr[fg], alpha, 0)
    return out


def run_save_gt_overlays_all_original_dim(
    dataset_dir: str,
    out_dir: str = "results/qualitative/ground_truth",
    alpha: float = 0.45,
    save_color: bool = False,
    save_gt_mask: bool = False,
):
    test_img_dir = os.path.join(dataset_dir, "test_dataset", "images")
    test_mask_dir = os.path.join(dataset_dir, "test_dataset", "masks")

    if not (os.path.isdir(test_img_dir) and os.path.isdir(test_mask_dir)):
        raise FileNotFoundError("test_dataset/{images,masks} not found under dataset_dir")

    file_list = sorted([f for f in os.listdir(test_img_dir) if os.path.exists(os.path.join(test_mask_dir, f))])
    if len(file_list) == 0:
        raise RuntimeError("No test files found in test_dataset.")

    os.makedirs(out_dir, exist_ok=True)

    for fname in tqdm(file_list, desc="Saving GT overlays (original size)"):
        img_path = os.path.join(test_img_dir, fname)
        mask_path = os.path.join(test_mask_dir, fname)

        raw_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if raw_bgr is None:
            continue
        orig_h, orig_w = raw_bgr.shape[:2]

        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        gt = gt.astype(np.uint8)

        # Resize GT mask to original image size (preserve class ids)
        gt_orig = cv2.resize(gt, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        gt_color = colorize_mask(gt_orig)
        gt_overlay = overlay_on_image(raw_bgr, gt_color, alpha=alpha)

        base = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(out_dir, f"{base}_gt_overlay.png"), gt_overlay)

        if save_color:
            cv2.imwrite(os.path.join(out_dir, f"{base}_gt_color.png"), gt_color)

        if save_gt_mask:
            cv2.imwrite(os.path.join(out_dir, f"{base}_gt_mask.png"), gt_orig.astype(np.uint8))

    print(f"\n✅ Done. Saved GT overlays to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="dataset root containing test_dataset/")
    parser.add_argument("--out_dir", type=str, default="results/qualitative/ground_truth")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--save_color", action="store_true", help="also save GT color masks (original size)")
    parser.add_argument("--save_gt_mask", action="store_true", help="also save raw GT label masks (original size)")
    args = parser.parse_args()

    run_save_gt_overlays_all_original_dim(
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        alpha=args.alpha,
        save_color=args.save_color,
        save_gt_mask=args.save_gt_mask,
    )
