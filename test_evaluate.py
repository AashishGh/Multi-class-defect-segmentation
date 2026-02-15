# test_evaluate.py

import time
import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff, cdist

# 0 = background (ignored)
# 1 = caries, 2 = PARL, 3 = IT
DEFECT_CLASSES = {
    1: "caries",
    2: "parl",
    3: "it",
}


# def hausdorff_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     """
#     Symmetric 95th-percentile Hausdorff distance (hd) between two 2D binary masks (in pixels).

#     hd = max( 95th percentile of distances from GT->Pred,
#                 95th percentile of distances from Pred->GT )

#     Uses scipy.spatial.distance.cdist for pairwise distances.
#     """
#     y_true = y_true.astype(bool)
#     y_pred = y_pred.astype(bool)

#     coords_true = np.argwhere(y_true)
#     coords_pred = np.argwhere(y_pred)

#     # Both empty -> perfect match
#     if coords_true.size == 0 and coords_pred.size == 0:
#         return 0.0

#     # One empty, one not -> max distance = image diagonal
#     if coords_true.size == 0 or coords_pred.size == 0:
#         h, w = y_true.shape
#         return float(np.sqrt(h ** 2 + w ** 2))

#     # Pairwise Euclidean distances
#     dists = cdist(coords_true, coords_pred)  # [N_true, N_pred]

#     # For each GT point: min distance to any pred point
#     d_true = dists.min(axis=1)
#     # For each pred point: min distance to any GT point
#     d_pred = dists.min(axis=0)

#     hd_true = np.percentile(d_true, 95)
#     hd_pred = np.percentile(d_pred, 95)

#     return float(max(hd_true, hd_pred))
def hausdorff_distance(targets, preds):
    haussdorf_dist = directed_hausdorff(preds, targets)[0]
    return haussdorf_dist


def _compute_from_confusion(tp, fp, fn, eps=1e-8):
    """IoU, precision, recall, F1, F2 from aggregate TP/FP/FN."""
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    f1 = 2 * precision * recall / (precision + recall + eps)
    beta2 = 4.0  # beta=2
    f2 = (1 + beta2) * precision * recall / (beta2 * precision + recall + eps)

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
    }


def evaluate(model, loader, device):
    """
    Multi-class evaluation for TEST:
      - model outputs: [B, 4, H, W]
      - masks: [B, H, W] with values in {0,1,2,3}

    Returns:
      metrics: {
        "caries": {iou, precision, recall, f1, f2, hd},
        "parl":   {...},
        "it":     {...},
        "mean":   {iou, precision, recall, f1, f2, hd}
      }
      fps: images/sec (inference only)
    """
    model.eval()

    # confusion + hd accumulators
    confusions = {
        name: {"TP": 0.0, "FP": 0.0, "FN": 0.0}
        for name in DEFECT_CLASSES.values()
    }
    hd_sums = {name: 0.0 for name in DEFECT_CLASSES.values()}
    hd_counts = {name: 0 for name in DEFECT_CLASSES.values()}

    total_images = 0
    total_inf_time = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device).long()          # [B, H, W]
            bsz = images.size(0)
            total_images += bsz

            # inference timing
            start_t = time.time()
            outputs = model(images)                  # [B, 4, H, W]
            if device == "cuda":
                torch.cuda.synchronize()
            end_t = time.time()
            total_inf_time += (end_t - start_t)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)       # [B, H, W]

            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            # confusion counts per defect
            for cls_id, name in DEFECT_CLASSES.items():
                y_true_flat = (masks_flat == cls_id)
                y_pred_flat = (preds_flat == cls_id)

                tp = (y_true_flat & y_pred_flat).sum().item()
                fp = (~y_true_flat & y_pred_flat).sum().item()
                fn = (y_true_flat & ~y_pred_flat).sum().item()

                confusions[name]["TP"] += tp
                confusions[name]["FP"] += fp
                confusions[name]["FN"] += fn

            # hd per image & class
            preds_np = preds.cpu().numpy()          # [B, H, W]
            masks_np = masks.cpu().numpy()          # [B, H, W]
            for b in range(bsz):
                for cls_id, name in DEFECT_CLASSES.items():
                    y_true = (masks_np[b] == cls_id).astype(np.uint8)
                    y_pred = (preds_np[b] == cls_id).astype(np.uint8)

                    hd = hausdorff_distance(y_true, y_pred)
                    hd_sums[name] += hd
                    hd_counts[name] += 1

    # metrics dict
    metrics = {}
    for name in DEFECT_CLASSES.values():
        c = confusions[name]
        m = _compute_from_confusion(c["TP"], c["FP"], c["FN"])
        m["hd"] = float(hd_sums[name] / hd_counts[name]) if hd_counts[name] > 0 else 0.0
        metrics[name] = m

    # mean over 3 defects
    mean_metrics = {}
    for key in ["iou", "precision", "recall", "f1", "f2", "hd"]:
        vals = [metrics[name][key] for name in DEFECT_CLASSES.values()]
        mean_metrics[key] = float(np.mean(vals))
    metrics["mean"] = mean_metrics

    fps = total_images / max(total_inf_time, 1e-8)

    return metrics, fps
