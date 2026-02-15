# Multi-class defect detection

Multi-class dental lesion segmentation project (4 classes):
- `0`: background
- `1`: caries
- `2`: PARL
- `3`: IT

This repository includes:
- training (`train.py`)
- validation metrics (`evaluate.py`)
- test evaluation with HD/FPS/ROC export (`test.py`, `test_evaluate.py`)
- model implementations (`models/`)
- qualitative analysis utilities (`qualitative_analysis/`)

## Project Structure

```text
Multi-class-defect-detection/
├── train.py
├── test.py
├── evaluate.py
├── test_evaluate.py
├── dataset_loader.py
├── transforms.py
├── loss.py
├── models/
├── qualitative_analysis/
├── selective_scan/
├── pretrained_pth/
├── requirements.txt
└── readme.md
```

## Dataset Layout

Pass `--dataset_dir` as a root folder that contains:

```text
<dataset_dir>/
├── train_dataset/
│   ├── images/
│   └── masks/
├── val_dataset/
│   ├── images/
│   └── masks/
└── test_dataset/
    ├── images/
    └── masks/
```

Important:
- Image and mask filenames must match.
- Masks are expected as class-index maps (`0,1,2,3`) for `CrossEntropyLoss`.
- If masks are encoded differently (for example `0, 85, 170, 255`), update mapping in `dataset_loader.py`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you use models that depend on custom CUDA ops, also build/install `selective_scan` as needed.

## Available Model Names

Use one of these values for `--model_name`:
- `unet`
- `doubleunet`
- `transnetr`
- `resunetplusplus`
- `transrupnet`
- `pvtformer`
- `rmamambas`

Source: `models/models_mapping.py`.

## Training

```bash
python train.py \
  --dataset_dir /path/to/dataset_root \
  --model_name unet \
  --epochs 500 \
  --batch_size 4 \
  --lr 1e-4 \
  --patience 50
```

Resume from checkpoint name (without `.pth`):

```bash
python train.py \
  --dataset_dir /path/to/dataset_root \
  --model_name unet \
  --resume epoch12
```

Training outputs are saved under:

```text
results/<model_name>/
├── checkpoints/
│   ├── best_model.pth
│   └── epochXX.pth
└── training_results/
    ├── train_loss.csv
    ├── train_iou.csv
    ├── val_metrics.csv
    ├── train_iou_curve.png
    └── val_iou_curve.png
```

## Testing

Evaluate on `test_dataset`:

```bash
python test.py \
  --dataset_dir /path/to/dataset_root \
  --model_name unet
```

Optional explicit checkpoint:

```bash
python test.py \
  --dataset_dir /path/to/dataset_root \
  --model_name unet \
  --checkpoint results/unet/checkpoints/best_model.pth
```

Test outputs:

```text
results/<model_name>/test_results/
├── test_metrics.csv
└── roc/
    ├── roc_auc.csv
    ├── roc_points_caries.csv
    ├── roc_points_parl.csv
    └── roc_points_it.csv
```

## Qualitative Visualization

Example:

```bash
python qualitative_analysis/get_qualitative_viz.py \
  --dataset_dir /path/to/dataset_root \
  --model_name unet \
  --out_root results/qualitative \
  --save_color \
  --save_pred_mask
```

## Notes

- `readme.md` is lowercase in this repository by design.
- Set `CUDA_VISIBLE_DEVICES` if needed for multi-GPU environments.
- `results/`, Python caches, and build artifacts should not be committed.
  
# Multi-class-defect-segmentation
