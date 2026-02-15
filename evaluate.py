import torch
import numpy as np

# Classes:
# 0 = background (ignored for reporting)
# 1 = caries
# 2 = PARL
# 3 = IT

DEFECT_CLASSES = {
    1: "caries",
    2: "parl",
    3: "it",
}


def _compute_from_confusion(tp, fp, fn, eps=1e-8):
    """Compute IoU, precision, recall, F1, F2 from aggregate TP/FP/FN."""
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


def compute_metrics_from_confusions(confusions):
    """
    confusions: dict[class_name] = dict(TP=..., FP=..., FN=...)
    Returns:
      {
        "caries": {...},
        "parl":   {...},
        "it":     {...},
        "mean":   {...}
      }
    """
    metrics = {}

    # Per-defect metrics
    for name, counts in confusions.items():
        tp = counts["TP"]
        fp = counts["FP"]
        fn = counts["FN"]
        metrics[name] = _compute_from_confusion(tp, fp, fn)

    # Mean over 3 defects
    mean_metrics = {}
    for key in ["iou", "precision", "recall", "f1", "f2"]:
        vals = [metrics[name][key] for name in DEFECT_CLASSES.values()]
        mean_metrics[key] = float(np.mean(vals))
    metrics["mean"] = mean_metrics

    return metrics


def evaluate(model, loader, device):
    """
    Memory-safe multi-class evaluation:
    - Model outputs: [B, 4, H, W]
    - Masks: [B, H, W] with values in {0,1,2,3}
    - Returns dict with per-defect + mean metrics.
    """
    model.eval()

    # Global confusion counts for each defect class
    confusions = {
        name: {"TP": 0.0, "FP": 0.0, "FN": 0.0}
        for name in DEFECT_CLASSES.values()
    }

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device).long()    # [B, H, W]

            outputs = model(images)            # [B, 4, H, W]
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)  # [B, H, W]

            # Flatten for easier boolean ops
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            for cls_id, name in DEFECT_CLASSES.items():
                y_true = (masks_flat == cls_id)
                y_pred = (preds_flat == cls_id)

                tp = (y_true & y_pred).sum().item()
                fp = (~y_true & y_pred).sum().item()
                fn = (y_true & ~y_pred).sum().item()

                confusions[name]["TP"] += tp
                confusions[name]["FP"] += fp
                confusions[name]["FN"] += fn

    return compute_metrics_from_confusions(confusions)
