# test.py

import os
import argparse
import csv
import numpy as np
from sklearn.metrics import roc_curve, auc


import torch
from torch.utils.data import DataLoader

from models.models_mapping import models_dict
from dataset_loader import DentalSegmentationDataset
from transforms import get_val_transforms
from test_evaluate import evaluate   # <-- use the separated evaluator

def collect_roc_data(model, loader, device, class_ids=(1,2,3)):
    """
    Collect per-pixel ROC data for each defect class.
    Returns dict: { 'Caries'|'PARL'|'IT': {'y_true': np.array, 'y_score': np.array} }
    """
    model.eval()
    id2name = {1: "Caries", 2: "PARL", 3: "IT"}
    buf = {id2name[c]: {"y_true": [], "y_score": []} for c in class_ids}

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device).long()           # [B,H,W]
            logits = model(images)                    # [B,4,H,W]
            probs  = torch.softmax(logits, dim=1)

            for c in class_ids:
                # scores: P(class=c) per pixel
                p = probs[:, c].contiguous().view(-1).detach().cpu().numpy()
                # labels: 1 if GT==c else 0
                y = (masks == c).contiguous().view(-1).detach().cpu().numpy().astype(np.uint8)
                buf[id2name[c]]["y_score"].append(p)
                buf[id2name[c]]["y_true"].append(y)

    for k in buf:
        buf[k]["y_score"] = np.concatenate(buf[k]["y_score"], axis=0)
        buf[k]["y_true"]  = np.concatenate(buf[k]["y_true"],  axis=0)
    return buf


def run_test(dataset_dir, model_name, checkpoint=None, batch_size=4):
    """
    Evaluate a trained model on:
        dataset_dir/test_dataset/images
        dataset_dir/test_dataset/masks

    Default checkpoint:
        <model_name>/checkpoints/best_model.pth
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # --- Test paths ---
    test_img_dir = os.path.join(dataset_dir, 'test_dataset', 'images')
    test_mask_dir = os.path.join(dataset_dir, 'test_dataset', 'masks')

    if not (os.path.isdir(test_img_dir) and os.path.isdir(test_mask_dir)):
        raise FileNotFoundError("test_dataset/{images,masks} not found under dataset_dir")

    test_file_list = sorted(
        [f for f in os.listdir(test_img_dir) if os.path.exists(os.path.join(test_mask_dir, f))]
    )
    if len(test_file_list) == 0:
        raise RuntimeError("No test files found in test_dataset.")

    # --- Loader ---
    test_loader = DataLoader(
        DentalSegmentationDataset(test_img_dir, test_mask_dir, test_file_list, transform=get_val_transforms()),
        batch_size=batch_size,
        shuffle=False
    )

    # --- Model ---
    if checkpoint is None:
        checkpoint = os.path.join("results",model_name, "checkpoints", "best_model.pth")
    print(f"Loading checkpoint from: {checkpoint}")

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = models_dict[model_name]
    model.to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)

    # --- Evaluate on TEST ---
    print("Evaluating on test set...")
    test_metrics, fps = evaluate(model, test_loader, device)

    caries_m = test_metrics["caries"]
    parl_m = test_metrics["parl"]
    it_m = test_metrics["it"]
    mean_m = test_metrics["mean"]

    print("\n=== Test Metrics (per defect) ===")
    print(f"Caries -> IoU: {caries_m['iou']:.4f}, "
          f"P: {caries_m['precision']:.4f}, R: {caries_m['recall']:.4f}, "
          f"F1: {caries_m['f1']:.4f}, F2: {caries_m['f2']:.4f}, "
          f"hd: {caries_m['hd']:.2f}")
    print(f"PARL   -> IoU: {parl_m['iou']:.4f}, "
          f"P: {parl_m['precision']:.4f}, R: {parl_m['recall']:.4f}, "
          f"F1: {parl_m['f1']:.4f}, F2: {parl_m['f2']:.4f}, "
          f"hd: {parl_m['hd']:.2f}")
    print(f"IT     -> IoU: {it_m['iou']:.4f}, "
          f"P: {it_m['precision']:.4f}, R: {it_m['recall']:.4f}, "
          f"F1: {it_m['f1']:.4f}, F2: {it_m['f2']:.4f}, "
          f"hd: {it_m['hd']:.2f}")

    print("\n=== Test Metrics (mean over defects: caries, PARL, IT) ===")
    print(f"mIoU: {mean_m['iou']:.4f}, "
          f"mPrecision: {mean_m['precision']:.4f}, "
          f"mRecall: {mean_m['recall']:.4f}, "
          f"mF1: {mean_m['f1']:.4f}, "
          f"mF2: {mean_m['f2']:.4f}, "
          f"mhd: {mean_m['hd']:.2f}")

    print(f"\n=== Inference Speed ===")
    print(f"FPS (images/sec): {fps:.2f}")

    # --- Save to CSV ---
    results_dir = os.path.join("results", model_name, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "test_metrics.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Caries", "PARL", "IT", "Mean"])
        for key in ["iou", "precision", "recall", "f1", "f2", "hd"]:
            writer.writerow([
                key.upper(),
                caries_m[key],
                parl_m[key],
                it_m[key],
                mean_m[key],
            ])
        writer.writerow(["FPS", "", "", "", fps])

    print(f"\n✅ Test metrics saved to: {csv_path}")

        # --- ROC artifacts (no plots, just data) ---
    roc_dir = os.path.join(results_dir, "roc")
    os.makedirs(roc_dir, exist_ok=True)

    roc_data = collect_roc_data(model, test_loader, device, class_ids=(1,2,3))

    auc_rows = []
    for name in ["Caries", "PARL", "IT"]:
        y_true  = roc_data[name]["y_true"]
        y_score = roc_data[name]["y_score"]

        fpr, tpr, thr = roc_curve(y_true, y_score)
        auc_val = auc(fpr, tpr)
        auc_rows.append((name, auc_val))

        # Save curve points for later multi-model plotting
        out_csv = os.path.join(roc_dir, f"roc_points_{name.lower()}.csv")
        with open(out_csv, "w", newline="") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["fpr", "tpr", "threshold"])
            for a, b, c in zip(fpr, tpr, thr):
                w.writerow([a, b, c])

    # Save per-class and mean AUC
    auc_csv = os.path.join(roc_dir, "roc_auc.csv")
    with open(auc_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Defect", "AUC"])
        for name, val in auc_rows:
            w.writerow([name, val])
        w.writerow(["Mean", float(np.mean([v for _, v in auc_rows]))])

    print(f"✅ ROC data saved under: {roc_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="path to dataset root")
    parser.add_argument("--model_name", type=str, required=True, help="model key in models_dict")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path to checkpoint (default: <results/model_name>/checkpoints/best_model.pth)")
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()

    run_test(
        dataset_dir=args.dataset_dir,
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
    )
