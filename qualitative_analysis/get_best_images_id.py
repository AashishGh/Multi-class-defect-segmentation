import os
import pandas as pd
from typing import Dict, List

def get_first_n_image_ids_from_model_csvs(
    model_csv_map: Dict[str, str],
    n: int,
    filename_col: str = "filename"
) -> List[Dict[str, List[str]]]:
    """
    Read each model's already-sorted CSV and return the first N image_ids (filenames).

    Args:
        model_csv_map: dict like {"unet": "path/to.csv", "vmunetv2": "path/to.csv", ...}
        n: number of top rows (image_ids) to take from each CSV
        filename_col: column name holding image id / filename (default: "filename")

    Returns:
        A list containing one dictionary:
          [{"unet": [img1, img2, ...], "vmunetv2": [...], ...}]
        (If a CSV has fewer than N rows, returns as many as available.)
    """
    out: Dict[str, List[str]] = {}

    for model_name, csv_path in model_csv_map.items():
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found for model '{model_name}': {csv_path}")

        df = pd.read_csv(csv_path)

        if filename_col not in df.columns:
            raise KeyError(
                f"Column '{filename_col}' not found in '{csv_path}'. "
                f"Available columns: {list(df.columns)}"
            )

        # Keep order as-is (already sorted), take first N
        out[model_name] = df[filename_col].head(n).astype(str).tolist()

    return out

def round_robin(model_csv_map):
    bucket=[]
    flag=True
    for image_id in model_csv_map['doubleunet']:
        for model,images in model_csv_map.items():
            if model =='doubleunet' or image_id in images:
                continue
            flag=False
            break
        if flag:
            bucket.append(image_id)
    return bucket


            



TOP4_MODELS = ["doubleunet", "transrupnet", "transnetr", "vmunetv2"]
MODEL_CSV_MAP = {m: f"results/{m}/qualitative/per_image_iou_test_dataset.csv" for m in TOP4_MODELS}

top_lists = get_first_n_image_ids_from_model_csvs(MODEL_CSV_MAP, n=44)
print("Top list")
print(top_lists)  # -> [{"vmunetv2":[...], "pvtformer":[...], ...}]
print("Bucket:")
print(round_robin(top_lists))
